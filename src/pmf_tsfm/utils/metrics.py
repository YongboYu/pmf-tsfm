"""
Metrics computation for time series forecasting evaluation.

Follows the paper's Eq. (1)-(2) exactly; std is across DF series (features).

Predictions shape: (num_sequences, prediction_length, num_features)
  e.g. for BPI2017: (58, 7, 21)

MAE (Eq. 1):
    MAE_d  = mean over (seq s, horizon h) of |y_{s,h,d} - yhat_{s,h,d}|
    Report : mean_d(MAE_d) +/- std_d(MAE_d)

RMSE (Eq. 2 + user clarification):
    Per-sequence RMSE for each (series d, sequence s):
        RMSE_{d,s} = sqrt(mean_h( (y_{s,h,d} - yhat_{s,h,d})^2 ))
    Per-series RMSE averaged over sequences:
        RMSE_d = mean_s( RMSE_{d,s} )
    Report: mean_d(RMSE_d) +/- std_d(RMSE_d)
"""

import json
from pathlib import Path
from typing import Any

import numpy as np


def compute_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    feature_names: list[str] | None = None,
) -> dict[str, Any]:
    """
    Compute MAE and RMSE following the paper's Eq. (1)-(2).

    Args:
        predictions: Shape (num_sequences, prediction_length, num_features)
        targets:     Same shape as predictions
        feature_names: Optional list of DF-series names

    Returns:
        Dictionary with:
        - summary: MAE_mean, MAE_std, RMSE_mean, RMSE_std (all across features)
        - per_feature: per-series MAE and RMSE
    """
    num_sequences, pred_len, num_features = predictions.shape

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(num_features)]

    errors = predictions - targets  # (num_seq, pred_len, num_features)
    abs_errors = np.abs(errors)

    # --- MAE per feature: mean over all (seq, timestep) pairs ---
    # shape (num_features,)
    feature_maes: np.ndarray = np.mean(abs_errors, axis=(0, 1))

    # --- RMSE per feature: mean-of-per-sequence-RMSEs (paper Eq. 2) ---
    # Per-sequence RMSE over horizon: (num_seq, num_features)
    seq_feature_rmse = np.sqrt(np.mean(errors**2, axis=1))
    # Average over sequences to get per-feature RMSE: (num_features,)
    feature_rmses: np.ndarray = np.mean(seq_feature_rmse, axis=0)

    per_feature: dict[str, Any] = {
        name: {
            "MAE": float(feature_maes[i]),
            "RMSE": float(feature_rmses[i]),
        }
        for i, name in enumerate(feature_names)
    }

    return {
        "summary": {
            "MAE_mean": float(np.mean(feature_maes)),
            "MAE_std": float(np.std(feature_maes)),
            "RMSE_mean": float(np.mean(feature_rmses)),
            "RMSE_std": float(np.std(feature_rmses)),
            "num_sequences": num_sequences,
            "num_features": num_features,
            "prediction_length": pred_len,
        },
        "per_feature": per_feature,
    }


def print_metrics(metrics: dict[str, Any], title: str = "") -> None:
    """Print metrics summary in paper format: mean +/- std across DF series."""
    s = metrics["summary"]

    print(f"\n{'=' * 60}")
    print(f"EVALUATION RESULTS{' - ' + title if title else ''}")
    print(f"{'=' * 60}")
    print(
        f"\n  {s['num_sequences']} sequences x {s['prediction_length']} horizon"
        f" x {s['num_features']} DF series"
    )
    print(f"  MAE:  {s['MAE_mean']:.4f} +/- {s['MAE_std']:.4f}")
    print(f"  RMSE: {s['RMSE_mean']:.4f} +/- {s['RMSE_std']:.4f}")
    print(f"{'=' * 60}\n")


def save_metrics(
    metrics: dict[str, Any],
    output_path: str | Path,
) -> None:
    """Save metrics to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"  Metrics saved to {output_path}")


def print_aggregate_summary(all_metrics: dict[str, dict[str, Any]]) -> None:
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

    print(f"{'Name':<35} {'MAE (mean+/-std)':<22} {'RMSE (mean+/-std)':<22}")
    print(f"{'-' * 79}")

    for name, metrics in all_metrics.items():
        s = metrics["summary"]
        mae_mean = s["MAE_mean"]
        mae_std = s["MAE_std"]
        rmse_mean = s["RMSE_mean"]
        rmse_std = s["RMSE_std"]

        mae_values.append(mae_mean)
        rmse_values.append(rmse_mean)

        print(
            f"{name:<35} {mae_mean:.4f} +/- {mae_std:.4f}      {rmse_mean:.4f} +/- {rmse_std:.4f}"
        )

    print(f"{'-' * 79}")
    print(f"{'AVERAGE':<35} {np.mean(mae_values):.4f}                  {np.mean(rmse_values):.4f}")
    print(f"{'=' * 79}\n")
