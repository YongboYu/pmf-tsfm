#!/usr/bin/env python
"""
Evaluation script for time series forecasting.

Computes metrics from saved predictions:
- Statistical: MAE (overall), Per-sequence RMSE (mean ± std)
- Process-aware: Entropic Relevance (future)

Input structure: outputs/{task}/{dataset}/{model}/
    - predictions.npy
    - targets.npy
    - metadata.json

Output: outputs/{task}/{dataset}/{model}/metrics/
    - statistical.json

Usage:
    # Evaluate all results for a task
    python evaluate.py task=zero_shot

    # Evaluate specific dataset
    python evaluate.py task=zero_shot data=bpi2017

    # Evaluate specific model on dataset
    python evaluate.py task=zero_shot data=bpi2017 model=chronos_bolt_small
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent / "src"))

import hydra
import numpy as np
from omegaconf import DictConfig

from pmf_tsfm.utils.metrics import (
    compute_metrics,
    print_aggregate_summary,
    print_metrics,
)


def find_result_dirs(
    base_dir: Path, dataset: Optional[str] = None, model: Optional[str] = None
) -> List[Tuple[str, str, Path]]:
    """
    Find all result directories in the new structure.

    Structure: base_dir/{dataset}/{model}/

    Returns:
        List of (dataset_name, model_name, result_dir) tuples
    """
    results = []

    if not base_dir.exists():
        return results

    # If specific dataset requested
    if dataset:
        dataset_dirs = [base_dir / dataset] if (base_dir / dataset).exists() else []
    else:
        dataset_dirs = [d for d in base_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]

    for dataset_dir in dataset_dirs:
        dataset_name = dataset_dir.name

        # If specific model requested
        if model:
            model_dirs = [dataset_dir / model] if (dataset_dir / model).exists() else []
        else:
            model_dirs = [d for d in dataset_dir.iterdir() if d.is_dir()]

        for model_dir in model_dirs:
            # Check if this is a valid result directory
            if (model_dir / "predictions.npy").exists():
                results.append((dataset_name, model_dir.name, model_dir))

    return results


def load_predictions(result_dir: Path) -> Tuple[np.ndarray, np.ndarray, Optional[Dict]]:
    """Load predictions, targets, and metadata from a result directory."""
    predictions = np.load(result_dir / "predictions.npy")
    targets = np.load(result_dir / "targets.npy")

    metadata = None
    metadata_path = result_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)

    return predictions, targets, metadata


def save_metrics(metrics: Dict, output_dir: Path) -> None:
    """Save statistical metrics to the metrics subdirectory."""
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    with open(metrics_dir / "statistical.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"  Metrics saved to {metrics_dir}/statistical.json")


def evaluate_single(
    result_dir: Path,
    dataset_name: str,
    model_name: str,
    save: bool = True,
) -> Dict:
    """Evaluate a single model-dataset result."""
    print(f"\nEvaluating: {model_name} on {dataset_name}")

    predictions, targets, metadata = load_predictions(result_dir)

    # Get feature names from metadata if available
    feature_names = None
    if metadata and "feature_names" in metadata:
        feature_names = metadata["feature_names"]

    metrics = compute_metrics(predictions, targets, feature_names)

    # Add identifiers
    metrics["model_name"] = model_name
    metrics["dataset_name"] = dataset_name
    if metadata:
        metrics["prediction_length"] = metadata.get(
            "prediction_length", metadata.get("shapes", {}).get("predictions", [0, 0, 0])[1]
        )

    print_metrics(metrics, f"{model_name} on {dataset_name}")

    if save:
        save_metrics(metrics, result_dir)

    return metrics


def evaluate_all(
    base_dir: Path,
    dataset: Optional[str] = None,
    model: Optional[str] = None,
    save: bool = True,
) -> Dict[str, Dict]:
    """Evaluate all results in directory structure."""
    result_dirs = find_result_dirs(base_dir, dataset, model)

    if not result_dirs:
        print(f"No prediction files found in {base_dir}")
        if dataset:
            print(f"  Dataset filter: {dataset}")
        if model:
            print(f"  Model filter: {model}")
        return {}

    print(f"Found {len(result_dirs)} result directories")

    all_metrics = {}
    for dataset_name, model_name, result_dir in result_dirs:
        key = f"{dataset_name}/{model_name}"
        try:
            metrics = evaluate_single(result_dir, dataset_name, model_name, save)
            all_metrics[key] = metrics
        except Exception as e:
            print(f"Error evaluating {key}: {e}")
            continue

    if len(all_metrics) > 1:
        print_aggregate_summary(all_metrics)

    return all_metrics


@hydra.main(version_base="1.3", config_path="configs", config_name="eval")
def main(cfg: DictConfig):
    """Main entry point with Hydra configuration."""
    base_dir = Path(cfg.results_dir)

    if not base_dir.exists():
        print(f"Results directory not found: {base_dir}")
        return {}

    # Get optional filters (these are simple strings, not nested configs)
    dataset_filter = cfg.get("dataset")
    model_filter = cfg.get("model")

    return evaluate_all(
        base_dir=base_dir,
        dataset=dataset_filter,
        model=model_filter,
        save=cfg.get("save", True),
    )


if __name__ == "__main__":
    main()
