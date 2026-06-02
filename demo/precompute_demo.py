"""Precompute bundled demo assets from existing zero-shot outputs.

The bundled path is a holdout backtest (ADR-0004): for each bundled dataset ×
demo model we reuse the **final forecast window** already in ``predictions.npy``
(the held-out last week — forecast origin ``log_end − horizon``) and the matching
``targets.npy`` (what actually happened). No model is re-run.

For each pair this writes
``demo/assets/<dataset>/<model>/{forecast_dfg.json, actual_dfg.json, metrics.json,
forecast.svg, actual.svg, diff.svg}``, reusing ``pmf_tsfm.er.dfg.dfg_to_json`` for
DFG serialisation, the precomputed ER sweep for ER,
``pmf_tsfm.utils.metrics.compute_metrics`` for MAE/RMSE, and ``render`` for the SVGs.

The JSON files stay the regenerable source of truth; the SVGs are pre-rendered so
HF Spaces can serve the bundled path with no ``dot`` binary at runtime (the same
``render`` code later serves user-submitted logs).

    uv run python demo/precompute_demo.py
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from render import dfg_json_to_svg, diff_svg

from pmf_tsfm.er.automaton import compute_er
from pmf_tsfm.er.dfg import (
    build_truth_dfg,
    dfg_to_json,
    extract_sublog,
    extract_traces,
    load_event_log,
    prepare_log,
)
from pmf_tsfm.utils.metrics import compute_metrics

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_ROOT = REPO_ROOT / "outputs"
ASSETS_ROOT = REPO_ROOT / "demo" / "assets"
# XES event logs (lowercase-named), the same source the ER sweep parses.
LOG_DIR = REPO_ROOT / "data" / "processed_logs"

# demo model id -> on-disk model directory under outputs/zero_shot/<dataset>/
MODEL_DIRS: dict[str, str] = {
    "chronos2": "chronos_2",
    "moirai2": "moirai_2_0_small",
    "timesfm2.5": "timesfm_2_5_200m",
}

# demo dataset id (lowercase) -> Capitalized on-disk dir under outputs/zero_shot/.
# The ER path derives its Capitalized prefix from metadata, but the zero_shot
# source dir must be looked up explicitly so the bundled path is correct on a
# case-sensitive filesystem (e.g. HF Spaces), not just macOS's case-insensitive one.
DATASET_DIRS: dict[str, str] = {
    "bpi2017": "BPI2017",
    "bpi2019_1": "BPI2019_1",
    "sepsis": "Sepsis",
    "hospital_billing": "Hospital_Billing",
}

# The full bundled matrix: every dataset by every demo model (4 * 3 = 12 pairs).
BUNDLED: list[tuple[str, str]] = [
    (dataset, model) for dataset in DATASET_DIRS for model in MODEL_DIRS
]


def frequencies_to_dfg_json(window: np.ndarray, feature_names: list[str]) -> dict[str, Any]:
    """Build a clean DFG JSON from one window's DF-relation frequencies.

    Sums the horizon, rounds, drops zero-frequency relations, and keeps the raw
    ▶/■ markers so ``dfg_to_json`` maps them to the canonical start/end nodes —
    no duplicate Start/End nodes and no artificial freq-1 arcs.

    Args:
        window:        shape (horizon, n_features) — one forecast/actual window.
        feature_names: ``"A -> B"`` column names aligned with the last axis.

    Returns:
        DFG in ``{"nodes": [...], "arcs": [...]}`` format.
    """
    freqs = np.clip(np.round(window.sum(axis=0)), 0, None).astype(int)
    dfg: dict[tuple[str, str], int] = {}
    for name, freq in zip(feature_names, freqs, strict=True):
        if freq > 0:
            src, tgt = name.split("->", 1)
            dfg[(src.strip(), tgt.strip())] = int(freq)
    return dfg_to_json(dfg)


def truth_er_from_sublog(sublog: pd.DataFrame) -> float:
    """Truth-DFG ER of one window: ER of ``build_truth_dfg`` against its own traces.

    This is the canonical baseline (fitting ratio ≈ 1.0) the forecast ER is judged
    against — the truth DFG, with its artificial ▶/■ closure, near-perfectly explains
    the week that actually happened. Reuses the ER sweep's own utilities so the value
    matches the paper's per-window truth ER (which the sweep computes but never saves).

    Returns ``nan`` when the window has no traces or no DF relations (``compute_er``
    already degrades gracefully there).
    """
    truth_dfg = dfg_to_json(build_truth_dfg(sublog))
    er, _, _ = compute_er(truth_dfg, extract_traces(sublog))
    return float(er)


@cache
def _prepared_log(log_path: Path) -> pd.DataFrame:
    """Load + prepare one XES log, cached so the 12-pair sweep parses only 4 files."""
    return prepare_log(load_event_log(log_path))


def truth_er_for_window(log_path: Path, window: str) -> float:
    """Truth-DFG ER baseline for the final forecast window of one log.

    ``window`` is the ``"YYYY-MM-DD_YYYY-MM-DD"`` string the ER sweep stores. The
    inclusive end-of-day boundary mirrors ``evaluate_er_all`` so the extracted
    sublog (and thus the ER) matches the held-out week the demo displays.
    """
    ws, we = (pd.Timestamp(d, tz="UTC") for d in window.split("_"))
    we_eod = we + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    sublog = extract_sublog(_prepared_log(log_path), ws, we_eod)
    return truth_er_from_sublog(sublog)


def precompute_one(
    dataset: str,
    model: str,
    *,
    outputs_root: Path,
    assets_root: Path,
    log_dir: Path = LOG_DIR,
) -> Path:
    """Precompute the bundled assets for one dataset × demo model.

    Args:
        dataset:      bundled dataset id (e.g. ``"bpi2017"``).
        model:        demo model id (e.g. ``"chronos2"``).
        outputs_root: root of the existing ``outputs/`` artifacts.
        assets_root:  root under which ``<dataset>/<model>/`` assets are written.
        log_dir:      dir of XES logs (``<dataset>.xes``) for the truth-ER baseline.

    Returns:
        The asset directory that was written.
    """
    model_dir = MODEL_DIRS[model]
    src_dir = outputs_root / "zero_shot" / DATASET_DIRS[dataset] / model_dir

    metadata = json.loads(next(src_dir.glob("*_metadata.json")).read_text())
    feature_names = metadata["feature_names"]
    prefix = metadata["data_metadata"]["dataset_name"]  # e.g. "BPI2017"

    predictions = np.load(src_dir / f"{prefix}_{model_dir}_predictions.npy")
    targets = np.load(src_dir / f"{prefix}_{model_dir}_targets.npy")

    # Final window = the held-out last week (forecast origin = log_end - horizon).
    forecast_dfg = frequencies_to_dfg_json(predictions[-1], feature_names)
    actual_dfg = frequencies_to_dfg_json(targets[-1], feature_names)

    # MAE / RMSE of just the final window (paper formula, reused).
    final = compute_metrics(predictions[-1:], targets[-1:], feature_names)["summary"]

    # ER of the final window from the precomputed ER sweep.
    er_path = (
        outputs_root / "er" / "zero_shot" / prefix / model_dir / f"{prefix}_{model_dir}_er.json"
    )
    er_windows = json.loads(er_path.read_text())["windows"]
    # Truth-DFG ER baseline on the same held-out final window, so the demo can show
    # how much better the truth DFG explains that week than the forecast does.
    truth_er = truth_er_for_window(log_dir / f"{dataset}.xes", er_windows[-1]["window"])
    metrics = {
        "er": float(er_windows[-1]["pred_er"]),
        "truth_er": truth_er,
        "mae": float(final["MAE_mean"]),
        "rmse": float(final["RMSE_mean"]),
    }

    out_dir = assets_root / dataset / model
    out_dir.mkdir(parents=True, exist_ok=True)
    # Trailing newline keeps the committed assets stable under end-of-file-fixer.
    (out_dir / "forecast_dfg.json").write_text(
        json.dumps(forecast_dfg, ensure_ascii=False, indent=2) + "\n"
    )
    (out_dir / "actual_dfg.json").write_text(
        json.dumps(actual_dfg, ensure_ascii=False, indent=2) + "\n"
    )
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    # Pre-render the figures so HF Spaces serves them statically (no `dot` at
    # runtime). The JSON above stays the regenerable source of truth.
    (out_dir / "forecast.svg").write_text(dfg_json_to_svg(forecast_dfg))
    (out_dir / "actual.svg").write_text(dfg_json_to_svg(actual_dfg))
    (out_dir / "diff.svg").write_text(diff_svg(forecast_dfg, actual_dfg))
    return out_dir


def main() -> None:
    for dataset, model in BUNDLED:
        out_dir = precompute_one(dataset, model, outputs_root=OUTPUTS_ROOT, assets_root=ASSETS_ROOT)
        print(f"wrote {out_dir.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
