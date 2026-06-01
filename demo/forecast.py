"""Assemble a bundled forecast from precomputed assets.

This is the agent-clean orchestrator for the bundled (holdout-backtest) path:
typed arguments, a structured return, and **no Gradio objects** — its signature
is the future MCP/REST schema. It reads only the committed assets produced by
``scripts/precompute_demo.py``; no model is run here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ASSETS_ROOT = Path(__file__).parent / "assets"


def forecast_bundled(
    dataset: str,
    model: str = "chronos2",
    horizon: int = 7,
) -> dict[str, Any]:
    """Return the bundled forecast triple for one dataset × model.

    The bundled path is a holdout backtest (ADR-0004): the held-out last
    ``horizon`` days are forecast from the rest of the log and compared against
    what actually happened, so the metrics are genuine accuracy.

    Args:
        dataset: Bundled dataset id (e.g. ``"bpi2017"``).
        model:   TSFM id (e.g. ``"chronos2"``).
        horizon: Forecast horizon in days. Fixed at 7 in v1.

    Returns:
        ``{"forecast_dfg": <dfg json>, "actual_dfg": <dfg json>,
           "metrics": {"er": float, "mae": float, "rmse": float},
           "forecast_svg": str, "actual_svg": str, "diff_svg": str}`` — the JSON
        DFGs (the regenerable source of truth) alongside the pre-rendered figures
        so the served path needs no ``dot`` binary at runtime.
    """
    base = ASSETS_ROOT / dataset / model
    forecast_dfg = json.loads((base / "forecast_dfg.json").read_text())
    actual_dfg = json.loads((base / "actual_dfg.json").read_text())
    metrics = json.loads((base / "metrics.json").read_text())
    return {
        "forecast_dfg": forecast_dfg,
        "actual_dfg": actual_dfg,
        "metrics": metrics,
        "forecast_svg": (base / "forecast.svg").read_text(),
        "actual_svg": (base / "actual.svg").read_text(),
        "diff_svg": (base / "diff.svg").read_text(),
    }
