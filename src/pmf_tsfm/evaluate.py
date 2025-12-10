"""
Evaluation script for time series forecasting.

Computes metrics from saved predictions:
- MAE: Mean Absolute Error (overall)
- Per-sequence RMSE: RMSE per sequence, then mean ± std

Future: Entropic Relevance for process model assessment.

Usage:
    # Evaluate single result
    python -m pmf_tsfm.evaluate results_dir=results/zero_shot

    # Evaluate specific model-dataset pair
    python -m pmf_tsfm.evaluate \\
        results_dir=results/zero_shot \\
        model_name=chronos_bolt_small \\
        dataset_name=BPI2017

    # Evaluate all results in directory
    python -m pmf_tsfm.evaluate results_dir=results/zero_shot --all
"""

import json
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


def find_result_files(results_dir: Path) -> list[tuple[str, str]]:
    """
    Find all prediction files in results directory.

    Returns:
        List of (dataset_name, model_name) tuples
    """
    results = []
    for pred_file in results_dir.glob("*_predictions.npy"):
        # Parse filename: {dataset}_{model}_predictions.npy
        name = pred_file.stem.replace("_predictions", "")
        parts = name.split("_", 1)
        if len(parts) == 2:
            dataset_name, model_name = parts
            results.append((dataset_name, model_name))
    return results


def load_predictions(
    results_dir: Path,
    dataset_name: str,
    model_name: str,
) -> tuple[np.ndarray, np.ndarray, dict | None]:
    """
    Load predictions, targets, and metadata from disk.

    Args:
        results_dir: Directory containing results
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
    Evaluate a single model-dataset result.

    Args:
        results_dir: Directory containing results
        dataset_name: Name of dataset
        model_name: Name of model
        save: Whether to save metrics to disk

    Returns:
        Metrics dictionary
    """
    print(f"\nEvaluating: {model_name} on {dataset_name}")

    predictions, targets, metadata = load_predictions(results_dir, dataset_name, model_name)

    # Get feature names from metadata if available
    feature_names = None
    if metadata and "feature_names" in metadata:
        feature_names = metadata["feature_names"]

    metrics = compute_metrics(predictions, targets, feature_names)

    # Add metadata
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
    Evaluate all results in directory.

    Args:
        results_dir: Directory containing results
        save: Whether to save metrics to disk

    Returns:
        Dictionary mapping "{dataset}_{model}" to metrics
    """
    result_pairs = find_result_files(results_dir)

    if not result_pairs:
        print(f"No prediction files found in {results_dir}")
        return {}

    print(f"Found {len(result_pairs)} result files")

    all_metrics = {}
    for dataset_name, model_name in result_pairs:
        key = f"{dataset_name}_{model_name}"
        try:
            metrics = evaluate_single(results_dir, dataset_name, model_name, save)
            all_metrics[key] = metrics
        except Exception as e:
            print(f"Error evaluating {key}: {e}")
            continue

    if len(all_metrics) > 1:
        print_aggregate_summary(all_metrics)

    return all_metrics


@hydra.main(version_base="1.3", config_path="../../configs", config_name="eval")
def main(cfg: DictConfig):
    """Main entry point with Hydra configuration."""
    results_dir = Path(cfg.results_dir)

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return

    if cfg.get("all", False):
        # Evaluate all results
        return evaluate_all(results_dir, save=cfg.get("save", True))
    else:
        # Evaluate specific model-dataset pair
        if not cfg.get("model_name") or not cfg.get("dataset_name"):
            # If no specific pair, evaluate all
            return evaluate_all(results_dir, save=cfg.get("save", True))

        return evaluate_single(
            results_dir,
            cfg.dataset_name,
            cfg.model_name,
            save=cfg.get("save", True),
        )


if __name__ == "__main__":
    main()
