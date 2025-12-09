"""
Metrics computation for time series forecasting evaluation.

Core metrics (std computed across features/variables/dimensions):
- MAE: mean ± std across features
- RMSE: mean ± std across features

For predictions of shape (num_sequences, prediction_length, num_features):
- Each feature's metric is computed over all sequences and timesteps
- Then mean ± std is computed across the num_features dimensions

Future extension point:
- Entropic Relevance for process model assessment
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def compute_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    feature_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compute comprehensive forecasting metrics.

    Std is computed across features/variables/dimensions (not sequences).

    For predictions of shape (num_sequences, prediction_length, num_features):
    - Each feature's MAE/RMSE is computed over all sequences × timesteps
    - Then mean ± std is computed across the num_features

    Args:
        predictions: Shape (num_sequences, prediction_length, num_features)
        targets: Same shape as predictions
        feature_names: Optional feature names

    Returns:
        Dictionary with:
        - summary: MAE (mean±std), RMSE (mean±std) across features
        - per_feature: MAE and RMSE for each feature
    """
    num_sequences, pred_len, num_features = predictions.shape

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(num_features)]

    # Per-feature metrics
    # For each feature, flatten across sequences and timesteps
    feature_maes = []
    feature_rmses = []
    per_feature = {}

    for feat_idx, feat_name in enumerate(feature_names):
        # Shape: (num_sequences, prediction_length) -> flatten to (num_sequences * prediction_length,)
        pred_feat = predictions[:, :, feat_idx].flatten()
        tgt_feat = targets[:, :, feat_idx].flatten()

        mae = mean_absolute_error(tgt_feat, pred_feat)
        rmse = np.sqrt(mean_squared_error(tgt_feat, pred_feat))

        feature_maes.append(mae)
        feature_rmses.append(rmse)

        per_feature[feat_name] = {
            "MAE": float(mae),
            "RMSE": float(rmse),
        }

    # Mean ± std across features
    return {
        "summary": {
            # MAE: mean ± std across features
            "MAE_mean": float(np.mean(feature_maes)),
            "MAE_std": float(np.std(feature_maes)),
            # RMSE: mean ± std across features
            "RMSE_mean": float(np.mean(feature_rmses)),
            "RMSE_std": float(np.std(feature_rmses)),
            # Metadata
            "num_sequences": num_sequences,
            "num_features": num_features,
            "prediction_length": pred_len,
        },
        "per_feature": per_feature,
    }


def print_metrics(metrics: Dict[str, Any], title: str = "") -> None:
    """Print metrics summary."""
    s = metrics["summary"]

    print(f"\n{'=' * 60}")
    print(f"EVALUATION RESULTS{' - ' + title if title else ''}")
    print(f"{'=' * 60}")
    print(
        f"\n📊 Metrics ({s['num_sequences']} sequences × {s['prediction_length']} steps × {s['num_features']} features):"
    )
    print(
        f"   MAE:  {s['MAE_mean']:.4f} ± {s['MAE_std']:.4f} (across {s['num_features']} features)"
    )
    print(
        f"   RMSE: {s['RMSE_mean']:.4f} ± {s['RMSE_std']:.4f} (across {s['num_features']} features)"
    )
    print(f"{'=' * 60}\n")


def save_metrics(
    metrics: Dict[str, Any],
    output_path: Union[str, Path],
) -> None:
    """Save metrics to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"  Metrics saved to {output_path}")


def print_aggregate_summary(all_metrics: Dict[str, Dict[str, Any]]) -> None:
    """
    Print aggregate summary across multiple datasets/models.

    Args:
        all_metrics: Dict mapping names to metrics dicts
    """
    print(f"\n{'=' * 75}")
    print("AGGREGATE SUMMARY")
    print(f"{'=' * 75}\n")

    mae_values = []
    rmse_values = []

    print(f"{'Name':<35} {'MAE (mean±std)':<20} {'RMSE (mean±std)':<20}")
    print(f"{'-' * 75}")

    for name, metrics in all_metrics.items():
        s = metrics["summary"]
        mae_mean = s["MAE_mean"]
        mae_std = s["MAE_std"]
        rmse_mean = s["RMSE_mean"]
        rmse_std = s["RMSE_std"]

        mae_values.append(mae_mean)
        rmse_values.append(rmse_mean)

        print(f"{name:<35} {mae_mean:.4f} ± {mae_std:.4f}       {rmse_mean:.4f} ± {rmse_std:.4f}")

    print(f"{'-' * 75}")
    print(f"{'AVERAGE':<35} {np.mean(mae_values):.4f}                {np.mean(rmse_values):.4f}")
    print(f"{'=' * 75}\n")
