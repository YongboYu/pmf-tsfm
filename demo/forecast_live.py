"""Assemble a live-path forecast for a custom-uploaded log.

The sibling of :func:`forecast.forecast_bundled`, for the *live* path (ADR-0004):
the forecast origin is the **log end**, so it forecasts the genuine, unseen next
window and compares it against the **last-known-window** DFG. There is no future
truth for an upload, so this path reports **drift** (DF relations the forecast
adds/removes vs the recent past) and **never** an accuracy metric (ER/MAE/RMSE).

The public :func:`forecast_live` is agent-clean (typed args + docstring, no Gradio
objects) — its signature is the future MCP/REST tool schema. The actual
preprocessing + model inference sits behind :func:`_live_windows`, which is wired
to ZeroGPU in a follow-up slice (#115); the assembly, guard, and drift logic here
are GPU-free and tested in isolation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import upload_guard
from dfg_diff import dfg_diff
from precompute_demo import frequencies_to_dfg_json
from render import dfg_json_to_svg, diff_svg


def _live_windows(
    log: str | Path, model: str, horizon: int
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Preprocess + forecast one upload, the GPU seam wired to ZeroGPU in #115.

    Returns ``(forecast_window, last_known_window, feature_names)`` where both windows
    are ``(horizon, n_features)`` DF-relation frequency arrays in the **same** feature
    space (so the two DFGs key consistently): the forecast for the next ``horizon`` days
    from the log end, and the actual last ``horizon`` days of the log. #115 implements
    this (preprocess log → DF-relation series; model.predict → forecast window; series
    tail → last-known window); the tests monkeypatch it.
    """
    raise NotImplementedError(
        "live preprocessing + inference is wired on ZeroGPU in #115; tests monkeypatch this seam"
    )


def drift_report(
    forecast_dfg: dict[str, Any], comparison_dfg: dict[str, Any]
) -> dict[str, list[dict[str, Any]]]:
    """Partition DF relations as **drift** of the forecast vs the last-known window.

    Reuses :func:`dfg_diff.dfg_diff` but reframes it relative to the recent past
    (not an accuracy comparison): ``dfg_diff(forecast, comparison)`` calls a relation
    present only in the forecast ``removed`` and one present only in the comparison
    ``added`` — the *opposite* of the drift reading — so the two are swapped here.

    Args:
        forecast_dfg:   The forecast DFG (``{"nodes": [...], "arcs": [...]}``).
        comparison_dfg: The last-known-window DFG, same shape.

    Returns:
        ``{"added": [...], "removed": [...], "stable": [...]}`` where ``added`` are
        relations the forecast introduces vs the recent past, ``removed`` are ones it
        drops, and ``stable`` are common to both. Each entry is
        ``{"from": label, "to": label, "forecast_freq": int, "recent_freq": int}`` —
        the ``recent_freq`` (not ``actual_freq``) naming keeps this off the accuracy
        framing: there is no future truth to score against (ADR-0004).
    """
    diff = dfg_diff(forecast_dfg, comparison_dfg)

    def entry(e: dict[str, Any]) -> dict[str, Any]:
        return {
            "from": e["from"],
            "to": e["to"],
            "forecast_freq": e["forecast_freq"],
            "recent_freq": e["actual_freq"],
        }

    return {
        "added": [entry(e) for e in diff["removed"]],  # in forecast, not in recent past
        "removed": [entry(e) for e in diff["added"]],  # in recent past, not in forecast
        "stable": [entry(e) for e in diff["matched"]],
    }


def forecast_live(
    log: str | Path,
    model: str = "chronos2",
    horizon: int = 7,
) -> dict[str, Any]:
    """Forecast a custom-uploaded log's true future and report its drift.

    The live path (ADR-0004): the forecast origin is the log end, so the forecast is
    the genuine unseen next ``horizon`` days, compared against the **last-known-window**
    DFG. There is no future truth for an upload, so the bundle carries **drift** (DF
    relations the forecast adds/removes vs the recent past) and **never** an accuracy
    metric (no ER/MAE/RMSE). This is the function the MCP/REST tool exposes.

    Args:
        log:     Path to the uploaded XES log.
        model:   TSFM id (default ``"chronos2"``; ``moirai2``/``timesfm2.5`` are gated to
                 small logs by the upload guard).
        horizon: Forecast horizon in days. Fixed at 7 in v1.

    Returns:
        ``{"forecast_dfg": <dfg json>, "comparison_dfg": <dfg json>,
           "drift": {"added": [...], "removed": [...], "stable": [...]},
           "forecast_svg": str, "comparison_svg": str,
           "diff_absolute_svg": str, "diff_relative_svg": str}`` — the JSON DFGs (the
        regenerable source of truth) alongside pre-rendered figures. The ``comparison_*``
        is the last-known-window DFG (not actual future), and the bundle reports drift,
        never accuracy.

    Raises:
        UploadRejected: the log is too large or the model is gated for its size.

    Note:
        The reused ``diff_svg`` legend still reads "forecast | actual" / "over/under-forecast
        %"; relabelling the live diff view ("last-known window" / "% change from recent
        past") is a GUI concern handled when the live path is wired (#115). The bundle
        *data* already uses the correct drift vocabulary and carries no accuracy metric.
    """
    model = upload_guard.check_upload(log, model)
    forecast_window, last_known_window, feature_names = _live_windows(log, model, horizon)

    forecast_dfg = frequencies_to_dfg_json(forecast_window, feature_names)
    comparison_dfg = frequencies_to_dfg_json(last_known_window, feature_names)
    return {
        "forecast_dfg": forecast_dfg,
        "comparison_dfg": comparison_dfg,
        "drift": drift_report(forecast_dfg, comparison_dfg),
        "forecast_svg": dfg_json_to_svg(forecast_dfg),
        "comparison_svg": dfg_json_to_svg(comparison_dfg),
        "diff_absolute_svg": diff_svg(forecast_dfg, comparison_dfg, mode="absolute"),
        "diff_relative_svg": diff_svg(forecast_dfg, comparison_dfg, mode="relative"),
    }
