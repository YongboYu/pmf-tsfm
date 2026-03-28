"""
Evaluation script for time series forecasting.

Computes MAE and RMSE from saved predictions, following the paper's
Eq. (1)-(2):
  MAE  = mean_d( mean_{s,h} |error| )  +/-  std_d
  RMSE = mean_d( mean_s( sqrt(mean_h(error^2)) ) )  +/-  std_d

Predictions are stored at:
    outputs/{task}/{dataset_name}/{model_name}/
        {dataset_name}_{model_name}_predictions.npy
        {dataset_name}_{model_name}_targets.npy
        {dataset_name}_{model_name}_metadata.json

This script discovers them recursively under the given results_dir.

Usage:
    # Evaluate all results under the zero_shot task directory
    python -m pmf_tsfm.evaluate task=zero_shot

    # Evaluate a specific subdirectory (single model + dataset)
    python -m pmf_tsfm.evaluate \\
        results_dir=outputs/zero_shot/BPI2017/chronos_bolt_small

    # Evaluate lora or full_tune results
    python -m pmf_tsfm.evaluate task=lora_tune
"""

import json
import time
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig

from pmf_tsfm.utils.metrics import (
    compute_metrics,
    print_aggregate_summary,
    print_metrics,
    save_metrics,
)
from pmf_tsfm.utils.wandb_logger import init_run


def find_result_files(results_dir: Path) -> list[tuple[Path, str, str]]:
    """
    Recursively find all prediction files under results_dir.

    Predictions are saved as {dataset}_{model}_predictions.npy and may live
    in subdirectories (outputs/{task}/{dataset}/{model}/).

    Returns:
        List of (file_dir, dataset_name, model_name) triples
    """
    results = []
    for pred_file in sorted(results_dir.rglob("*_predictions.npy")):
        # Filename: {dataset_name}_{model_name}_predictions.npy
        stem = pred_file.stem.replace("_predictions", "")
        # Split on first underscore: dataset names may contain underscores
        # Prefer splitting using the parent directory names when available:
        #   .../BPI2017/chronos_bolt_small/BPI2017_chronos_bolt_small_predictions.npy
        parent_parts = pred_file.parent.parts
        if len(parent_parts) >= 2:
            model_name = parent_parts[-1]  # e.g. chronos_bolt_small
            dataset_name = parent_parts[-2]  # e.g. BPI2017
        else:
            # Fallback: split filename on first underscore
            parts = stem.split("_", 1)
            dataset_name, model_name = (parts[0], parts[1]) if len(parts) == 2 else (stem, "")
        results.append((pred_file.parent, dataset_name, model_name))
    return results


def load_predictions(
    results_dir: Path,
    dataset_name: str,
    model_name: str,
) -> tuple[np.ndarray, np.ndarray, dict | None]:
    """
    Load predictions, targets, and metadata from disk.

    Args:
        results_dir: Directory containing the .npy files
        dataset_name: Name of dataset
        model_name: Name of model

    Returns:
        Tuple of (predictions, targets, metadata)
    """
    prefix = f"{dataset_name}_{model_name}"

    predictions = np.load(results_dir / f"{prefix}_predictions.npy")
    targets = np.load(results_dir / f"{prefix}_targets.npy")

    metadata_path = results_dir / f"{prefix}_metadata.json"
    metadata = None
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)

    return predictions, targets, metadata


def evaluate_single(
    results_dir: Path,
    dataset_name: str,
    model_name: str,
    save: bool = True,
) -> dict:
    """
    Evaluate one model-dataset pair and optionally save metrics JSON.

    Args:
        results_dir: Directory that contains the prediction .npy files
        dataset_name: Name of dataset (e.g. BPI2017)
        model_name:   Name of model   (e.g. chronos_bolt_small)
        save:         Write metrics JSON next to the prediction files

    Returns:
        Metrics dictionary
    """
    print(f"\nEvaluating: {model_name} on {dataset_name}")

    predictions, targets, metadata = load_predictions(results_dir, dataset_name, model_name)

    feature_names: list[str] | None = None
    if metadata and "feature_names" in metadata:
        feature_names = metadata["feature_names"]

    metrics = compute_metrics(predictions, targets, feature_names)
    metrics["model_name"] = model_name
    metrics["dataset_name"] = dataset_name
    if metadata:
        metrics["prediction_length"] = metadata.get("prediction_shape", [0, 0, 0])[1]

    print_metrics(metrics, f"{model_name} on {dataset_name}")

    if save:
        prefix = f"{dataset_name}_{model_name}"
        save_metrics(metrics, results_dir / f"{prefix}_metrics.json")

    return metrics


def evaluate_all(results_dir: Path, save: bool = True) -> dict[str, dict]:
    """
    Evaluate all prediction files discovered recursively under results_dir.

    Args:
        results_dir: Root directory to search (e.g. outputs/zero_shot/)
        save:        Write per-result metrics JSON files

    Returns:
        Dict mapping "{dataset}_{model}" to metrics dict
    """
    result_triples = find_result_files(results_dir)

    if not result_triples:
        print(f"No prediction files found under {results_dir}")
        return {}

    print(f"Found {len(result_triples)} result file(s) under {results_dir}")

    all_metrics: dict[str, dict] = {}
    for file_dir, dataset_name, model_name in result_triples:
        key = f"{dataset_name}_{model_name}"
        try:
            metrics = evaluate_single(file_dir, dataset_name, model_name, save)
            all_metrics[key] = metrics
        except Exception as e:
            print(f"Error evaluating {key}: {e}")

    if len(all_metrics) > 1:
        print_aggregate_summary(all_metrics)

    return all_metrics


@hydra.main(version_base="1.3", config_path="../../configs", config_name="eval")
def main(cfg: DictConfig):
    """Main entry point with Hydra configuration."""
    t_start = time.perf_counter()
    task: str = cfg.get("task", "zero_shot")

    run = init_run(cfg, job_type="evaluate", name=f"eval/{task}", tags=[task])

    results_dir = Path(cfg.results_dir)
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        run.finish()
        return None

    all_metrics = evaluate_all(results_dir, save=cfg.get("save", True))

    elapsed = time.perf_counter() - t_start

    # Log each model-dataset pair as a W&B summary entry and a comparison table
    if all_metrics:
        rows = []
        for key, m in all_metrics.items():
            s = m.get("summary", {})
            mae_mean = s.get("MAE_mean", float("nan"))
            mae_std = s.get("MAE_std", float("nan"))
            rmse_mean = s.get("RMSE_mean", float("nan"))
            rmse_std = s.get("RMSE_std", float("nan"))
            run.log(
                {
                    f"eval/{key}/mae_mean": mae_mean,
                    f"eval/{key}/rmse_mean": rmse_mean,
                }
            )
            rows.append(
                [
                    m.get("dataset_name", ""),
                    m.get("model_name", ""),
                    round(mae_mean, 4),
                    round(mae_std, 4),
                    round(rmse_mean, 4),
                    round(rmse_std, 4),
                ]
            )
        run.log_table(
            key="eval/results_table",
            columns=["dataset", "model", "mae_mean", "mae_std", "rmse_mean", "rmse_std"],
            rows=rows,
        )

    # Per-feature MAE/RMSE table across all model-dataset pairs
    if all_metrics:
        feat_rows = []
        for _key, m in all_metrics.items():
            dataset = m.get("dataset_name", "")
            model = m.get("model_name", "")
            for feat_name, feat_vals in (m.get("per_feature") or {}).items():
                feat_rows.append(
                    [
                        dataset,
                        model,
                        feat_name,
                        round(feat_vals.get("MAE", float("nan")), 4),
                        round(feat_vals.get("RMSE", float("nan")), 4),
                    ]
                )
        if feat_rows:
            run.log_table(
                key="eval/per_feature_table",
                columns=["dataset", "model", "feature", "mae", "rmse"],
                rows=feat_rows,
            )

    run.log_summary({"eval/elapsed_s": elapsed, "eval/n_results": len(all_metrics or {})})
    run.finish()

    return all_metrics


if __name__ == "__main__":
    main()
