"""
Multi-model Entropic Relevance evaluation — one XES parse per dataset.

Instead of loading and parsing the XES log separately for each model
(as evaluate_er.py does), this script amortises the expensive I/O:

  1. Load XES log once per dataset
  2. Build training DFG once
  3. For each test window:
     a. Extract sublog + traces ONCE
     b. Compute truth ER
     c. For each model: build prediction DFG → compute ER
  4. Save per-model JSON + a combined summary JSON

For N models × M windows this reduces XES parsing from O(N) to O(1)
and sublog extraction from O(N × M) to O(M).  On large logs (BPI2017
with ~1 million events) this is the dominant saving.

Models are auto-discovered by scanning
    outputs/{task}/{data.name}/*/
for ``*_predictions.npy`` files.  Pass ``model_names=[...]`` to restrict
to a specific subset (useful when only the paper models are of interest).

Usage
-----
    # Evaluate all discovered models for BPI2017 (zero-shot)
    python -m pmf_tsfm.er.evaluate_er_all data=bpi2017

    # All datasets in one go (one XES parse per dataset)
    python -m pmf_tsfm.er.evaluate_er_all --multirun \\
        data=bpi2017,bpi2019_1,sepsis,hospital_billing

    # Restrict to the three paper models
    python -m pmf_tsfm.er.evaluate_er_all data=bpi2017 \\
        'model_names=[chronos2,moirai_2_0_small,timesfm_2_5_200m]'

    # With W&B logging
    python -m pmf_tsfm.er.evaluate_er_all data=bpi2017 logger=wandb
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

from pmf_tsfm.data.assets import resolve_dataset_asset_path
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
from pmf_tsfm.er.evaluate_er import (
    _build_training_dfg,
    _get_test_dates,
    _load_predictions,
)
from pmf_tsfm.utils.wandb_logger import init_run

# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------


def _discover_models(
    outputs_dir: Path,
    dataset_name: str,
    task: str,
    model_names: list[str] | None = None,
) -> list[tuple[str, Path]]:
    """
    Return list of (model_name, inference_dir) pairs for the given dataset.

    When model_names is non-empty, only those models are returned; otherwise
    all subdirectories of outputs/{task}/{dataset_name}/ that contain a
    *_predictions.npy file are returned.
    """
    base = outputs_dir / task / dataset_name
    if not base.exists():
        return []

    found: list[tuple[str, Path]] = []
    for sub in sorted(base.iterdir()):
        if not sub.is_dir():
            continue
        name = sub.name
        if model_names and name not in model_names:
            continue
        pred_files = list(sub.glob("*_predictions.npy"))
        if pred_files:
            found.append((name, sub))
    return found


# ---------------------------------------------------------------------------
# Per-window stats helper
# ---------------------------------------------------------------------------


def _stats(values: list[float]) -> tuple[float, float]:
    vals = [v for v in values if not math.isnan(v)]
    return (float(np.mean(vals)), float(np.std(vals))) if vals else (float("nan"), float("nan"))


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------


def run_er_all(cfg: DictConfig) -> dict:
    """
    Multi-model ER evaluation pipeline.

    Returns a dict mapping model_name → per-model result dict.
    """
    t_start = time.perf_counter()

    dataset_name: str = cfg.data.name
    task: str = cfg.get("task", "zero_shot")
    pred_len: int = cfg.prediction_length
    val_end: int = cfg.data.val_end
    train_ratio: float = cfg.get("train_ratio", 0.8)
    model_names: list[str] = list(cfg.get("model_names") or [])

    run = init_run(
        cfg,
        job_type="er_all",
        name=f"er_all/{task}/{dataset_name}",
        tags=[dataset_name, task, "er", "multi-model"],
        group=f"er/{task}/{dataset_name}",
    )

    print("=" * 70)
    print("MULTI-MODEL ENTROPIC RELEVANCE EVALUATION")
    print(f"  Dataset : {dataset_name}")
    print(f"  Task    : {task}")
    print("=" * 70)

    # ---- discover model prediction directories ----
    outputs_dir = Path(cfg.paths.output_dir)
    model_dirs = _discover_models(outputs_dir, dataset_name, task, model_names or None)

    if not model_dirs:
        print(f"\nNo prediction files found under {outputs_dir / task / dataset_name}/")
        run.finish()
        return {}

    print(f"\nModels to evaluate ({len(model_dirs)}):")
    for name, d in model_dirs:
        print(f"  {name:40s}  {d}")

    # ---- load predictions for all models upfront ----
    model_preds: list[tuple[str, np.ndarray, list[str]]] = []
    for model_name, inference_dir in model_dirs:
        try:
            preds, feature_names = _load_predictions(inference_dir, dataset_name, model_name)
            model_preds.append((model_name, preds, feature_names))
        except FileNotFoundError as e:
            print(f"  Warning: {e} — skipping {model_name}")

    if not model_preds:
        print("No valid prediction files found.")
        run.finish()
        return {}

    # ---- derive test-window date ranges (use first model as reference) ----
    ts_parquet = resolve_dataset_asset_path(
        Path(cfg.data.path),
        dataset_name=dataset_name,
        asset_label="time-series parquet",
    )
    n_windows_ref = model_preds[0][1].shape[0]
    starts, ends = _get_test_dates(ts_parquet, val_end, pred_len)
    assert len(starts) == n_windows_ref, (
        f"Window mismatch: parquet gives {len(starts)}, predictions have {n_windows_ref}"
    )
    print(f"\n[1/4] Test period: {starts[0].date()} → {ends[-1].date()} ({n_windows_ref} windows)")

    # ---- load and prepare event log ONCE ----
    log_path = resolve_dataset_asset_path(
        Path(cfg.paths.log_dir) / f"{dataset_name}.xes",
        dataset_name=dataset_name,
        asset_label="processed event log",
    )
    print(f"[2/4] Loading XES log: {log_path}")
    t_xes = time.perf_counter()
    raw_log = load_event_log(log_path)
    prepared = prepare_log(raw_log)
    print(f"      Loaded in {time.perf_counter() - t_xes:.1f}s  ({len(raw_log):,} events)")

    # ---- training DFG ONCE ----
    print("[3/4] Building training DFG (first 80 % by time) ...")
    training_dfg_json = _build_training_dfg(raw_log, train_ratio)

    # ---- per-window ER for all models ----
    print(f"[4/4] Computing ER for {n_windows_ref} windows × {len(model_preds)} models ...")

    # Accumulate per-window results per model
    model_window_results: dict[str, list[dict]] = {name: [] for name, _, _ in model_preds}
    truth_ers: list[float] = []
    truth_fits: list[float] = []
    train_ers: list[float] = []

    for i, (ws, we) in enumerate(zip(starts, ends, strict=True)):
        if ws.tzinfo is None:
            ws = ws.tz_localize("UTC")
            we = we.tz_localize("UTC")
        we_eod = we + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)

        # Sublog + traces extracted ONCE for this window
        sublog = extract_sublog(prepared, ws, we_eod)
        traces = extract_traces(sublog)

        if not traces:
            for model_name, _, _ in model_preds:
                model_window_results[model_name].append(
                    {
                        "window": f"{ws.date()}_{we.date()}",
                        "pred_er": float("nan"),
                        "pred_fitting_ratio": 0.0,
                        "n_traces": 0,
                    }
                )
            truth_ers.append(float("nan"))
            truth_fits.append(0.0)
            train_ers.append(float("nan"))
            continue

        # Truth DFG (one per window, shared across models)
        truth_dfg_json = dfg_to_json(build_truth_dfg(sublog))
        truth_er, truth_fit, n_traces = compute_er(truth_dfg_json, traces)
        truth_ers.append(truth_er)
        truth_fits.append(truth_fit)

        # Training DFG (pre-built once, reused)
        train_er, _, _ = compute_er(training_dfg_json, traces)
        train_ers.append(train_er)

        # Per-model prediction DFG
        window_step_metrics: dict[str, float] = {
            "er/window/truth_er": truth_er,
            "er/window/training_er": train_er,
            "er/window/n_traces": float(n_traces),
        }
        for model_name, preds_arr, feature_names in model_preds:
            pred_dfg_json = build_prediction_dfg(preds_arr[i], feature_names)
            pred_er, pred_fit, _ = compute_er(pred_dfg_json, traces)
            model_window_results[model_name].append(
                {
                    "window": f"{ws.date()}_{we.date()}",
                    "pred_er": pred_er,
                    "pred_fitting_ratio": pred_fit,
                    "n_traces": n_traces,
                }
            )
            window_step_metrics[f"er/window/{model_name}/pred_er"] = pred_er
            window_step_metrics[f"er/window/{model_name}/fitting_ratio"] = pred_fit

        # Log per-window ER time series (x-axis = window index)
        run.log(window_step_metrics, step=i)

        if (i + 1) % 10 == 0 or (i + 1) == n_windows_ref:
            print(f"  {i + 1:3d}/{n_windows_ref}  truth={truth_er:.3f}  train={train_er:.3f}")

    # ---- aggregate results ----
    truth_mean, truth_std = _stats(truth_ers)
    train_mean, train_std = _stats(train_ers)
    truth_fit_mean, _ = _stats(truth_fits)

    print("\n" + "=" * 70)
    print(f"RESULTS — {dataset_name}  (Truth: {truth_mean:.4f} ± {truth_std:.4f})")
    print(f"{'Model':<42} {'ER mean':>9} {'ER std':>8} {'Fit%':>7}")
    print("-" * 70)

    all_results: dict = {}
    table_rows: list[list] = []
    for model_name, _, _ in model_preds:
        window_list = model_window_results[model_name]
        pred_ers = [r["pred_er"] for r in window_list]
        pred_fits = [r["pred_fitting_ratio"] for r in window_list]
        pred_mean, pred_std = _stats(pred_ers)
        pred_fit_mean, _ = _stats(pred_fits)
        n_valid = sum(1 for v in pred_ers if not math.isnan(v))

        print(f"  {model_name:<40} {pred_mean:9.4f} {pred_std:8.4f}  {pred_fit_mean:.1%}")

        summary = {
            "model": model_name,
            "dataset": dataset_name,
            "task": task,
            "n_windows": n_windows_ref,
            "n_valid_windows": n_valid,
            "truth_er_mean": truth_mean,
            "truth_er_std": truth_std,
            "truth_fitting_ratio_mean": truth_fit_mean,
            "pred_er_mean": pred_mean,
            "pred_er_std": pred_std,
            "pred_fitting_ratio_mean": pred_fit_mean,
            "training_er_mean": train_mean,
            "training_er_std": train_std,
        }
        all_results[model_name] = {"summary": summary, "windows": window_list}
        table_rows.append(
            [
                dataset_name,
                model_name,
                round(truth_mean, 4),
                round(pred_mean, 4),
                round(pred_std, 4),
                f"{pred_fit_mean:.1%}",
                round(train_mean, 4),
            ]
        )

    elapsed = time.perf_counter() - t_start
    print("=" * 70)
    print(f"Total elapsed: {elapsed:.1f}s")

    # ---- save outputs ----
    if cfg.get("save", True):
        output_base = Path(cfg.output_base)

        # Per-model JSON (same structure as evaluate_er.py for compatibility)
        for model_name, result in all_results.items():
            out_dir = output_base / model_name
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{dataset_name}_{model_name}_er.json"
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2, default=str)

        # Combined summary JSON
        combined_path = output_base / f"{dataset_name}_er_all_summary.json"
        combined = {
            "dataset": dataset_name,
            "task": task,
            "n_windows": n_windows_ref,
            "elapsed_s": elapsed,
            "truth_er_mean": truth_mean,
            "truth_er_std": truth_std,
            "training_er_mean": train_mean,
            "training_er_std": train_std,
            "models": {name: r["summary"] for name, r in all_results.items()},
        }
        with open(combined_path, "w") as f:
            json.dump(combined, f, indent=2, default=str)
        print(f"\n  Combined summary: {combined_path}")

    # ---- W&B logging ----
    run.log_table(
        key="er/model_comparison",
        columns=["dataset", "model", "truth_er", "pred_er_mean", "pred_er_std", "fit%", "train_er"],
        rows=table_rows,
    )
    for model_name, result in all_results.items():
        s = result["summary"]
        run.log(
            {
                f"er/{model_name}/pred_er_mean": s["pred_er_mean"],
                f"er/{model_name}/pred_er_std": s["pred_er_std"],
                f"er/{model_name}/pred_fitting_ratio": s["pred_fitting_ratio_mean"],
            }
        )
    run.log_summary(
        {
            "er/truth_er_mean": truth_mean,
            "er/truth_er_std": truth_std,
            "er/training_er_mean": train_mean,
            "er/elapsed_s": elapsed,
            "er/n_models": len(model_preds),
            "er/n_windows": n_windows_ref,
        }
    )
    run.finish()

    return all_results


@hydra.main(version_base="1.3", config_path="../../../configs", config_name="er_all")
def main(cfg: DictConfig) -> None:
    if cfg.get("print_config", False):
        print(OmegaConf.to_yaml(cfg))
    run_er_all(cfg)


if __name__ == "__main__":
    main()
