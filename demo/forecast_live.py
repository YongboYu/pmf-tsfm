"""Assemble a live-path forecast for a custom-uploaded log.

The sibling of :func:`forecast.forecast_bundled`, for the *live* path (ADR-0004):
the forecast origin is the **log end**, so it forecasts the genuine, unseen next
window and compares it against the **last-known-window** DFG. There is no future
truth for an upload, so this path reports **drift** (DF relations the forecast
adds/removes vs the recent past) and **never** an accuracy metric (ER/MAE/RMSE).

:func:`forecast_live` takes typed args and no Gradio objects. The preprocessing + model
inference sits behind :func:`_live_windows` (wired to ZeroGPU in ``app.py``); the assembly,
guard, and drift logic here are GPU-free and tested in isolation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import upload_guard
from dfg_build import frequencies_to_dfg_json
from dfg_diff import dfg_diff
from log_to_series import log_to_df_series
from render import dfg_json_to_svg, diff_svg

# The committed sample (a dense six-week slice of the bundled sepsis log) is the live tab's
# one-click "Try an example"; re-exported by app.py.
EXAMPLE_LOG = Path(__file__).resolve().parent / "examples" / "sepsis_sample.xes"

# Chronos-2 is the only model wired on the hosted live path this slice (#115);
# moirai2 / timesfm2.5 stay a documented follow-up. The id matches
# configs/model/chronos/chronos2.yaml. Loading an ``s3://`` checkpoint needs boto3.
_CHRONOS2_MODEL_ID = "s3://autogluon/chronos-2/"
_chronos2_pipeline: object | None = None  # cache, loaded once per (forked) worker process


def _load_chronos2() -> object:
    """Load (and cache) the Chronos-2 pipeline. Imports torch/chronos lazily so the
    live-core module stays importable — and testable — without the model libs.

    **Called only from inside the ``@spaces.GPU`` worker** (via :func:`_chronos2_forecast`),
    never at module import: ZeroGPU forks a worker per GPU call, and initialising CUDA in
    the parent process (e.g. an eager ``device_map="cuda"`` load at import) breaks that fork
    with "process PID not found (pid=0)". So the model is loaded on the real GPU that exists
    inside the worker; ``torch.cuda.is_available()`` is ``True`` there. The weights cache to
    disk on first download, so subsequent workers load fast.
    """
    global _chronos2_pipeline
    if _chronos2_pipeline is None:
        import torch
        from chronos import BaseChronosPipeline

        device = "cuda" if torch.cuda.is_available() else "cpu"
        _chronos2_pipeline = BaseChronosPipeline.from_pretrained(
            _CHRONOS2_MODEL_ID, device_map=device
        )
    return _chronos2_pipeline


def _chronos2_forecast(context: np.ndarray, horizon: int) -> np.ndarray:
    """Forecast the next ``horizon`` days of every DF relation with Chronos-2.

    A minimal port of ``pmf_tsfm.models.chronos.ChronosAdapter._predict_chronos2_batched``
    (batch every feature into one ``predict_df`` call via a per-feature ``item_id``, take
    the median) — kept here so the live path needs only chronos-forecasting + torch, not
    the full ``pmf_tsfm`` dep tree. **This is the GPU work** (wrapped by ``@spaces.GPU`` in
    the GUI) and is what tests monkeypatch.

    Args:
        context: ``(n_days, n_features)`` history — the whole log's daily DF series.
        horizon: number of future days to forecast.

    Returns:
        ``(horizon, n_features)`` forecast frequencies (median).
    """
    import pandas as pd

    pipeline = _load_chronos2()
    n_features = context.shape[1]
    rows = [
        {"item_id": feat_idx, "timestamp": t, "target": float(value)}
        for feat_idx in range(n_features)
        for t, value in enumerate(context[:, feat_idx])
    ]
    pred_df = pipeline.predict_df(  # type: ignore[attr-defined]
        pd.DataFrame(rows), prediction_length=horizon, quantile_levels=[0.1, 0.5, 0.9]
    )
    forecast = np.zeros((horizon, n_features))
    for feat_idx in range(n_features):
        feat = pred_df[pred_df["item_id"] == feat_idx]
        forecast[:, feat_idx] = feat["predictions"].to_numpy()[:horizon]
    return forecast


def _live_windows(
    log: str | Path, model: str, horizon: int
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Preprocess + forecast one upload — the GPU seam (ADR-0004, ZeroGPU in #115).

    Returns ``(forecast_window, last_known_window, feature_names)`` where both windows
    are ``(horizon, n_features)`` DF-relation frequency arrays in the **same** feature
    space (so the two DFGs key consistently): the model forecast for the next ``horizon``
    days *from the log end* (the genuine unseen future), and the actual last ``horizon``
    days of the log (the recent past the drift is measured against).

    The whole daily series is the forecast context; its tail is the last-known window.
    Tests monkeypatch ``log_to_df_series`` / ``_chronos2_forecast`` to keep real model
    inference out of CI (as slice 3a kept this whole seam mocked).

    Raises:
        UploadRejected: the model is not wired on the hosted live path, or the log spans
            too few days to forecast a ``horizon``-day window from history.
    """
    if model != "chronos2":
        raise upload_guard.UploadRejected(
            f"{model} is not yet available on the hosted live demo; use chronos2."
        )

    series = log_to_df_series(log)
    feature_names = list(series.columns)
    values = series.to_numpy(dtype=float)
    if len(values) < horizon + 1:
        raise upload_guard.UploadRejected(
            f"the log spans only {len(values)} day(s); at least {horizon + 1} are needed to "
            f"forecast a {horizon}-day window from prior history."
        )

    last_known_window = values[-horizon:]
    forecast_window = _chronos2_forecast(values, horizon)
    return forecast_window, last_known_window, feature_names


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
        The diff SVGs are rendered with ``framing="drift"`` so their legend reads
        "forecast | last-known window" / "% change from recent past" (never "actual" or
        accuracy) — the relabel the bundle carries, so the same SVGs are honest whether
        shown in the GUI or returned by the future MCP tool.
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
        "diff_absolute_svg": diff_svg(
            forecast_dfg, comparison_dfg, mode="absolute", framing="drift"
        ),
        "diff_relative_svg": diff_svg(
            forecast_dfg, comparison_dfg, mode="relative", framing="drift"
        ),
    }
