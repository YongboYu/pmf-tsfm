"""
Entropic Relevance evaluation script.

Computes ER of forecasted DFGs against the ground-truth event log for each
rolling test window, following the paper's evaluation framework.

Pipeline
--------
1. Load XES event log from data/processed_logs/{dataset}.xes
2. Derive test-period window dates from the DF time-series parquet + val_end
3. Build training DFG once (first 80 % of event log by time)
4. For each test window [ws, we]:
   a. Extract sublog + traces (vectorised pandas)
   b. Build truth DFG from sublog (vectorised pandas)
   c. Build prediction DFG from predictions.npy (vectorised numpy)
   d. Compute ER for truth, prediction, and training DFGs
5. Report mean ± std ER and average fitting ratio across windows

Output (JSON) saved to:
    outputs/er/{task}/{dataset}/{model}/
        {dataset}_{model}_er.json

Usage
-----
    # Chronos-2 zero-shot on BPI2017
    python -m pmf_tsfm.er.evaluate_er model=chronos/chronos2 data=bpi2017

    # Evaluate all zero-shot best models in one multirun
    python -m pmf_tsfm.er.evaluate_er --multirun \\
        model=chronos/chronos2,moirai/2_0_small,timesfm/2_5_200m \\
        data=bpi2017,bpi2019_1,sepsis,hospital_billing
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from pmf_tsfm.er.automaton import compute_er
from pmf_tsfm.er.dfg import (
    build_prediction_dfg,
    build_truth_dfg,
    dfg_to_json,
    extract_sublog,
    extract_traces,
    load_event_log,
    prepare_log,
)
from pmf_tsfm.utils.wandb_logger import init_run

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_predictions(
    output_dir: Path, dataset_name: str, model_name: str
) -> tuple[np.ndarray, list[str]]:
    """Load predictions array and feature names from saved inference output."""
    prefix = f"{dataset_name}_{model_name}"
    pred_path = output_dir / f"{prefix}_predictions.npy"
    meta_path = output_dir / f"{prefix}_metadata.json"

    if not pred_path.exists():
        raise FileNotFoundError(f"Predictions not found: {pred_path}")

    predictions = np.load(pred_path)  # (n_windows, prediction_length, n_features)

    feature_names: list[str] = []
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        feature_names = meta.get("feature_names", [])

    return predictions, feature_names


def _get_test_dates(
    ts_parquet: Path, val_end: int, prediction_length: int
) -> tuple[list[pd.Timestamp], list[pd.Timestamp]]:
    """
    Derive per-window [start, end] dates from the DF time-series parquet.

    n_windows = len(df) - val_end - prediction_length + 1
    Window i covers df.index[val_end + i] … df.index[val_end + i + prediction_length - 1]
    """
    df = pd.read_parquet(ts_parquet, columns=[])  # load index only
    idx = df.index

    n_windows = len(idx) - val_end - prediction_length + 1
    starts = [idx[val_end + i] for i in range(n_windows)]
    ends = [idx[val_end + i + prediction_length - 1] for i in range(n_windows)]
    return starts, ends


def _training_cutoff(log_df: pd.DataFrame, train_ratio: float = 0.8) -> pd.Timestamp:
    """Return the training cutoff timestamp (first train_ratio of the log by time)."""
    ts = log_df["time:timestamp"]
    t_min, t_max = ts.min(), ts.max()
    total_days = (t_max - t_min).days + 1
    cutoff = t_min + pd.Timedelta(days=int(total_days * train_ratio))
    return cutoff


def _build_training_dfg(log_df: pd.DataFrame, train_ratio: float = 0.8) -> dict:
    """Compute training DFG once from the first train_ratio of the log."""
    cutoff = _training_cutoff(log_df, train_ratio)
    training_df = log_df[log_df["time:timestamp"] < cutoff]
    return dfg_to_json(build_truth_dfg(training_df))


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------


def run_er_evaluation(cfg: DictConfig) -> dict:
    """
    Full ER evaluation pipeline.

    Returns a dict with per-window results and summary statistics.
    """
    t_start = time.perf_counter()

    dataset_name: str = cfg.data.name
    model_name: str = cfg.model.name
    task: str = cfg.get("task", "zero_shot")
    pred_len: int = cfg.prediction_length
    val_end: int = cfg.data.val_end

    run = init_run(
        cfg,
        job_type="er",
        name=f"er/{task}/{dataset_name}/{model_name}",
        tags=[model_name, dataset_name, task, "er"],
        group=f"er/{task}/{dataset_name}",
    )

    print("=" * 70)
    print("ENTROPIC RELEVANCE EVALUATION")
    print(f"  Dataset : {dataset_name}")
    print(f"  Model   : {model_name}")
    print(f"  Task    : {task}")
    print("=" * 70)

    # ---- locate prediction files ----
    inference_dir = Path(cfg.paths.output_dir) / task / dataset_name / model_name
    predictions, feature_names = _load_predictions(inference_dir, dataset_name, model_name)
    n_windows, _pred_length, _n_features = predictions.shape
    print(f"\n[1/5] Predictions: {predictions.shape}  ({n_windows} windows)")

    # ---- derive test-window date ranges ----
    ts_parquet = Path(cfg.data.path)
    starts, ends = _get_test_dates(ts_parquet, val_end, pred_len)
    assert len(starts) == n_windows, (
        f"Window count mismatch: parquet gives {len(starts)}, predictions have {n_windows}"
    )
    print(f"[2/5] Test period: {starts[0].date()} → {ends[-1].date()}")

    # ---- load and prepare event log ----
    log_path = Path(cfg.paths.log_dir) / f"{dataset_name}.xes"
    print(f"[3/5] Loading XES log: {log_path}")
    raw_log = load_event_log(log_path)
    prepared = prepare_log(raw_log)  # adds _next_ts column once

    # ---- training DFG (computed once, reused for all windows) ----
    print("[4/5] Building training DFG (first 80 % by time) ...")
    training_dfg_json = _build_training_dfg(raw_log)

    # ---- per-window ER computation ----
    print(f"[5/5] Computing ER for {n_windows} windows ...")
    window_results: list[dict] = []

    for i, (ws, we) in enumerate(zip(starts, ends, strict=True)):
        # Ensure timezone-aware comparison with the XES log (which uses UTC)
        if ws.tzinfo is None:
            ws = ws.tz_localize("UTC")
            we = we.tz_localize("UTC")
        # Extend window_end to cover the entire last day (events occur at any time of day)
        we_eod = we + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)

        # Extract sublog and traces
        sublog = extract_sublog(prepared, ws, we_eod)
        traces = extract_traces(sublog)

        if not traces:
            window_results.append(
                {
                    "window": f"{ws.date()}_{we.date()}",
                    "truth_er": float("nan"),
                    "truth_fitting_ratio": 0.0,
                    "pred_er": float("nan"),
                    "pred_fitting_ratio": 0.0,
                    "training_er": float("nan"),
                    "training_fitting_ratio": 0.0,
                    "n_traces": 0,
                }
            )
            continue

        # Truth DFG
        truth_dfg_json = dfg_to_json(build_truth_dfg(sublog))
        truth_er, truth_fit, n_traces = compute_er(truth_dfg_json, traces)

        # Prediction DFG
        pred_dfg_json = build_prediction_dfg(predictions[i], feature_names)
        pred_er, pred_fit, _ = compute_er(pred_dfg_json, traces)

        # Training DFG (reused)
        train_er, train_fit, _ = compute_er(training_dfg_json, traces)

        window_results.append(
            {
                "window": f"{ws.date()}_{we.date()}",
                "truth_er": truth_er,
                "truth_fitting_ratio": truth_fit,
                "pred_er": pred_er,
                "pred_fitting_ratio": pred_fit,
                "training_er": train_er,
                "training_fitting_ratio": train_fit,
                "n_traces": n_traces,
            }
        )

        if (i + 1) % 10 == 0 or (i + 1) == n_windows:
            print(
                f"  {i + 1:3d}/{n_windows}  truth={truth_er:.3f}  "
                f"pred={pred_er:.3f}  train={train_er:.3f}  "
                f"fit={pred_fit:.1%}"
            )

    # ---- summary statistics ----
    valid = [r for r in window_results if not math.isnan(r["pred_er"])]

    def _stats(key: str) -> tuple[float, float]:
        vals = [r[key] for r in valid if not math.isnan(r[key])]
        return (float(np.mean(vals)), float(np.std(vals))) if vals else (float("nan"), float("nan"))

    truth_mean, truth_std = _stats("truth_er")
    pred_mean, pred_std = _stats("pred_er")
    train_mean, train_std = _stats("training_er")
    pred_fit_mean, _ = _stats("pred_fitting_ratio")
    truth_fit_mean, _ = _stats("truth_fitting_ratio")

    summary = {
        "model": model_name,
        "dataset": dataset_name,
        "task": task,
        "n_windows": n_windows,
        "n_valid_windows": len(valid),
        "truth_er_mean": truth_mean,
        "truth_er_std": truth_std,
        "truth_fitting_ratio_mean": truth_fit_mean,
        "pred_er_mean": pred_mean,
        "pred_er_std": pred_std,
        "pred_fitting_ratio_mean": pred_fit_mean,
        "training_er_mean": train_mean,
        "training_er_std": train_std,
    }

    print("\n" + "=" * 70)
    print(f"RESULTS — {model_name} on {dataset_name}")
    print(f"  Truth    ER : {truth_mean:.4f} +/- {truth_std:.4f}  (fit {truth_fit_mean:.1%})")
    print(f"  Pred     ER : {pred_mean:.4f} +/- {pred_std:.4f}  (fit {pred_fit_mean:.1%})")
    print(f"  Training ER : {train_mean:.4f} +/- {train_std:.4f}")
    print("=" * 70)

    result = {"summary": summary, "windows": window_results}

    elapsed = time.perf_counter() - t_start
    print(f"\n  Elapsed: {elapsed:.1f}s")

    if cfg.get("save", True):
        out_dir = Path(cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{dataset_name}_{model_name}_er.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"  ER results saved to {out_path}")

    run.log_summary(
        {
            "er/truth_er_mean": truth_mean,
            "er/truth_er_std": truth_std,
            "er/truth_fitting_ratio": truth_fit_mean,
            "er/pred_er_mean": pred_mean,
            "er/pred_er_std": pred_std,
            "er/pred_fitting_ratio": pred_fit_mean,
            "er/training_er_mean": train_mean,
            "er/training_er_std": train_std,
            "er/elapsed_s": elapsed,
            "er/n_windows": n_windows,
            "er/n_valid_windows": len(valid),
        }
    )
    run.finish()

    return result


@hydra.main(version_base="1.3", config_path="../../../configs", config_name="er")
def main(cfg: DictConfig) -> None:
    if cfg.get("print_config", False):
        print(OmegaConf.to_yaml(cfg))
    run_er_evaluation(cfg)


if __name__ == "__main__":
    main()
