"""
Chronos adapter for zero-shot time series forecasting.

Supports:
- Chronos Bolt: tiny, mini, small, base (uses predict_quantiles API)
- Chronos 2.0: uses predict_df API with DataFrames (s3://autogluon/chronos-2/)

Handles multivariate time series by forecasting each feature separately.
"""

import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

from pmf_tsfm.models.base import BaseAdapter

# Suppress deprecation warnings
warnings.filterwarnings("ignore", message=".*torch_dtype.*deprecated.*")
warnings.filterwarnings("ignore", message=".*past_key_values.*deprecated.*")
warnings.filterwarnings("ignore", message=".*dtype.*deprecated.*")


class ChronosAdapter(BaseAdapter):
    """
    Adapter for Amazon Chronos models.

    Forecasts multivariate time series by treating each feature as
    a separate univariate series.

    Supports:
    - Chronos Bolt (tiny, mini, small, base) - uses predict_quantiles API
    - Chronos 2.0 - uses predict_df API with DataFrames (batched by item_id)
    """

    QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    @classmethod
    def from_config(
        cls, model_cfg: DictConfig, device: str = "mps", prediction_length: int = 7
    ) -> "ChronosAdapter":
        """Create adapter from Hydra model config."""
        return cls(
            model_name=model_cfg.name,
            model_id=model_cfg.id,
            variant=model_cfg.variant,
            device=device,
            torch_dtype=model_cfg.get("torch_dtype", "float32"),
            prediction_length=prediction_length,
            quantile_levels=list(model_cfg.get("quantile_levels", cls.QUANTILE_LEVELS)),
        )

    def __init__(
        self,
        model_name: str,
        model_id: str,
        variant: str = "bolt_small",
        device: str = "mps",
        torch_dtype: str = "float32",
        prediction_length: int = 7,
        quantile_levels: Optional[List[float]] = None,
        **kwargs,
    ):
        """
        Initialize the Chronos adapter.

        Args:
            model_name: Display name
            model_id: HuggingFace model ID or S3 path
                      (e.g., 'amazon/chronos-bolt-small' or 's3://autogluon/chronos-2/')
            variant: Model variant for API detection (bolt_*, chronos2)
            device: Device to run on
            torch_dtype: Data type
            prediction_length: Forecasting horizon
            quantile_levels: Quantile levels for probabilistic forecasts
        """
        super().__init__(
            model_name=model_name,
            model_id=model_id,
            model_family="chronos",
            variant=variant,
            device=device,
            torch_dtype=torch_dtype,
            prediction_length=prediction_length,
            **kwargs,
        )

        self.quantile_levels = quantile_levels or self.QUANTILE_LEVELS
        self._is_chronos2 = self._detect_chronos2()

    def _detect_chronos2(self) -> bool:
        """Detect if model is Chronos 2.0 (uses different API)."""
        variant_lower = self.variant.lower()
        model_id_lower = self.model_id.lower()

        return (
            "chronos2" in variant_lower or "chronos_2" in variant_lower or "s3://" in model_id_lower
        )

    def load_model(self) -> None:
        """Load Chronos model."""
        model_type = "Chronos 2.0" if self._is_chronos2 else "Chronos Bolt"
        print(f"Loading {model_type} model: {self.model_id}")
        start_time = time.time()

        try:
            from chronos import BaseChronosPipeline

            # Chronos 2.0 doesn't use torch_dtype
            if self._is_chronos2:
                self.pipeline = BaseChronosPipeline.from_pretrained(
                    self.model_id,
                    device_map=self.device,
                )
            else:
                self.pipeline = BaseChronosPipeline.from_pretrained(
                    self.model_id,
                    device_map=self.device,
                    torch_dtype=self.torch_dtype,
                )

            self._is_loaded = True
            load_time = time.time() - start_time
            print(f"  Loaded in {load_time:.2f}s on {self.device}")

        except ImportError:
            raise ImportError("chronos package not found. Install: pip install chronos-forecasting")
        except Exception as e:
            raise RuntimeError(f"Failed to load {self.model_id}: {e}")

    def predict(
        self,
        prepared_data: Dict[str, Any],
        prediction_length: Optional[int] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions for multivariate time series.

        Args:
            prepared_data: Dict with 'inputs', 'targets', 'feature_names'
            prediction_length: Override prediction length

        Returns:
            predictions: (num_sequences, prediction_length, num_features)
            quantiles: (num_sequences, prediction_length, num_features, num_quantiles)
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        pred_len = prediction_length or self.prediction_length

        if self._is_chronos2:
            return self._predict_chronos2_batched(prepared_data, pred_len)
        else:
            return self._predict_bolt(prepared_data, pred_len)

    def _predict_bolt(
        self,
        prepared_data: Dict[str, Any],
        prediction_length: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Chronos Bolt prediction using predict_quantiles API."""
        inputs = prepared_data["inputs"]
        feature_names = prepared_data["feature_names"]

        num_sequences = len(inputs)
        num_features = len(feature_names)
        num_quantiles = len(self.quantile_levels)

        predictions = np.zeros((num_sequences, prediction_length, num_features))
        quantiles_out = np.zeros((num_sequences, prediction_length, num_features, num_quantiles))

        print(f"Forecasting {num_sequences} sequences × {num_features} features...")
        start_time = time.time()

        for seq_idx in range(num_sequences):
            if seq_idx % 10 == 0:
                elapsed = time.time() - start_time
                print(f"  Progress: {seq_idx}/{num_sequences} ({elapsed:.1f}s)")

            input_seq = inputs[seq_idx]  # (input_length, num_features)

            for feat_idx in range(num_features):
                univariate = input_seq[:, feat_idx]
                context = torch.tensor(univariate, dtype=torch.float32).unsqueeze(0)

                try:
                    quantiles, mean = self.pipeline.predict_quantiles(
                        inputs=context,
                        prediction_length=prediction_length,
                        quantile_levels=self.quantile_levels,
                    )

                    predictions[seq_idx, :, feat_idx] = mean.cpu().numpy().flatten()
                    quantiles_out[seq_idx, :, feat_idx, :] = quantiles.cpu().numpy()

                except Exception:
                    # Fallback: use last value
                    last_val = float(univariate[-1]) if len(univariate) > 0 else 0.0
                    predictions[seq_idx, :, feat_idx] = last_val
                    quantiles_out[seq_idx, :, feat_idx, :] = last_val

        total_time = time.time() - start_time
        print(f"Completed in {total_time:.1f}s | Output: {predictions.shape}")

        return predictions, quantiles_out

    def _predict_chronos2_batched(
        self,
        prepared_data: Dict[str, Any],
        prediction_length: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Chronos 2.0 prediction using predict_df API with batched features.

        Batches all features for each sequence into a single predict_df call
        using different item_id values, significantly improving efficiency.
        """
        inputs = prepared_data["inputs"]
        feature_names = prepared_data["feature_names"]

        num_sequences = len(inputs)
        num_features = len(feature_names)
        quantile_levels_c2 = [0.1, 0.5, 0.9]

        predictions = np.zeros((num_sequences, prediction_length, num_features))
        quantiles_out = np.zeros((num_sequences, prediction_length, num_features, 3))

        print(
            f"Forecasting {num_sequences} sequences × {num_features} features (Chronos 2.0 batched)..."
        )
        start_time = time.time()

        for seq_idx in range(num_sequences):
            if seq_idx % 10 == 0:
                elapsed = time.time() - start_time
                print(f"  Progress: {seq_idx}/{num_sequences} ({elapsed:.1f}s)")

            input_seq = inputs[seq_idx]  # (input_length, num_features)

            # Build batched DataFrame with all features
            # Each feature has a unique item_id
            df_rows = []
            for feat_idx, feat_name in enumerate(feature_names):
                univariate = input_seq[:, feat_idx]
                for t, value in enumerate(univariate):
                    df_rows.append(
                        {
                            "item_id": feat_name,
                            "timestamp": t,
                            "target": float(value),
                        }
                    )

            batch_df = pd.DataFrame(df_rows)

            try:
                # Single batched prediction call for all features
                pred_df = self.pipeline.predict_df(
                    batch_df,
                    prediction_length=prediction_length,
                    quantile_levels=quantile_levels_c2,
                )

                # Extract predictions for each feature
                for feat_idx, feat_name in enumerate(feature_names):
                    feat_preds = pred_df[pred_df["item_id"] == feat_name]

                    if len(feat_preds) >= prediction_length:
                        median = feat_preds["predictions"].values[:prediction_length]
                        predictions[seq_idx, :, feat_idx] = median

                        if "0.1" in feat_preds.columns:
                            quantiles_out[seq_idx, :, feat_idx, 0] = feat_preds["0.1"].values[
                                :prediction_length
                            ]
                            quantiles_out[seq_idx, :, feat_idx, 1] = median
                            quantiles_out[seq_idx, :, feat_idx, 2] = feat_preds["0.9"].values[
                                :prediction_length
                            ]
                        else:
                            quantiles_out[seq_idx, :, feat_idx, :] = np.expand_dims(median, -1)
                    else:
                        # Fallback
                        last_val = float(input_seq[-1, feat_idx])
                        predictions[seq_idx, :, feat_idx] = last_val
                        quantiles_out[seq_idx, :, feat_idx, :] = last_val

            except Exception as e:
                # Fallback for entire sequence
                print(f"  Warning: Batch prediction failed for seq {seq_idx}: {e}")
                for feat_idx in range(num_features):
                    last_val = float(input_seq[-1, feat_idx])
                    predictions[seq_idx, :, feat_idx] = last_val
                    quantiles_out[seq_idx, :, feat_idx, :] = last_val

        total_time = time.time() - start_time
        print(f"Completed in {total_time:.1f}s | Output: {predictions.shape}")

        return predictions, quantiles_out
