"""
Supports:
- Chronos Bolt (tiny, mini, small, base) - uses predict_quantiles API
- Chronos 2.0 - uses predict_df API (different interface)
"""

from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
import warnings
import time
from omegaconf import DictConfig, OmegaConf
from .base_adapter import BaseAdapter

# Suppress warnings
warnings.filterwarnings("ignore", message=".*torch_dtype.*deprecated.*")
warnings.filterwarnings("ignore", message=".*past_key_values.*deprecated.*")


class ChronosAdapter(BaseAdapter):
    @classmethod
    def from_config(cls, model_cfg: DictConfig, device: str = "mps", prediction_length: int = 7) -> "ChronosAdapter":
        return cls(
            model_name=model_cfg.name,
            model_id=model_cfg.id,
            variant=model_cfg.variant,
            device=device,
            torch_dtype=model_cfg.get("torch_dtype", "float32"),
            prediction_length=prediction_length,
            quantile_levels=list(model_cfg.get("quantile_levels", [0.1, 0.5, 0.9])),
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
        **kwargs
    ):
        """
        Initialize the Chronos adapter.

        Args:
            model_name: Display name for the model
            model_id: HuggingFace model ID (e.g., "amazon/chronos-bolt-small")
            variant: Model variant for detection (bolt_small, chronos2, etc.)
            device: Device to run on ('cpu', 'cuda', 'mps')
            torch_dtype: Data type ('float32', 'float16', 'bfloat16')
            prediction_length: Forecasting horizon
            quantile_levels: Quantile levels for probabilistic forecasts
        """
        super().__init__(
            model_name=model_name,
            model_id=model_id,
            model_type="chronos",
            variant=variant,
            device_map=device,
            torch_dtype=torch_dtype,
            **kwargs
        )

        self.device = device
        self.prediction_length = prediction_length
        self.quantile_levels = quantile_levels or [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        self._is_chronos2 = self._detect_chronos2()

    def _detect_chronos2(self) -> bool:
        variant_lower = self.variant.lower()
        model_id_lower = self.model_id.lower()

        return (
            "chronos2" in variant_lower or
            "chronos-2" in variant_lower or
            "s3://" in model_id_lower
        )

    def load_model(self):
        print(f"Loading Chronos model: {self.model_id}")
        start_time = time.time()

        try:
            from chronos import BaseChronosPipeline

            self.pipeline = BaseChronosPipeline.from_pretrained(
                self.model_id,
                device_map=self.device,
                torch_dtype=self.torch_dtype,
            )

            load_time = time.time() - start_time
            print(f"  Loaded in {load_time:.2f}s on {self.device}")

        except ImportError:
            raise ImportError(
                "chronos package not found. Install: pip install chronos-forecasting"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load {self.model_id}: {e}")

    def predict(
        self,
        prepared_data: Dict[str, Any],
        prediction_length: Optional[int] = None,
        **kwargs
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
        if self.pipeline is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        pred_len = prediction_length or self.prediction_length

        if self._is_chronos2:
            return self._predict_chronos2(prepared_data, pred_len)
        else:
            return self._predict_bolt(prepared_data, pred_len)

    def _predict_bolt(
        self,
        prepared_data: Dict[str, Any],
        prediction_length: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Chronos Bolt prediction using predict_quantiles API."""
        inputs = prepared_data['inputs']
        feature_names = prepared_data['feature_names']

        num_sequences = len(inputs)
        num_features = len(feature_names)
        num_quantiles = len(self.quantile_levels)

        # Initialize outputs
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

                except Exception as e:
                    # Fallback: use last value
                    last_val = float(univariate[-1]) if len(univariate) > 0 else 0.0
                    predictions[seq_idx, :, feat_idx] = last_val
                    quantiles_out[seq_idx, :, feat_idx, :] = last_val

        total_time = time.time() - start_time
        print(f"Completed in {total_time:.1f}s")
        print(f"Output shape: {predictions.shape}")

        return predictions, quantiles_out

    def _predict_chronos2(
        self,
        prepared_data: Dict[str, Any],
        prediction_length: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Chronos 2.0 prediction using predict_df API with DataFrames."""
        inputs = prepared_data['inputs']
        feature_names = prepared_data['feature_names']

        num_sequences = len(inputs)
        num_features = len(feature_names)
        quantile_levels_c2 = [0.1, 0.5, 0.9]

        predictions = np.zeros((num_sequences, prediction_length, num_features))
        quantiles_out = np.zeros((num_sequences, prediction_length, num_features, 3))

        print(f"Forecasting {num_sequences} sequences × {num_features} features (Chronos 2.0)...")
        start_time = time.time()

        for seq_idx in range(num_sequences):
            if seq_idx % 10 == 0:
                elapsed = time.time() - start_time
                print(f"  Progress: {seq_idx}/{num_sequences} ({elapsed:.1f}s)")

            input_seq = inputs[seq_idx]

            for feat_idx in range(num_features):
                univariate = input_seq[:, feat_idx]
                feat_name = feature_names[feat_idx]

                # Create DataFrame for Chronos 2.0
                df = pd.DataFrame({
                    'item_id': feat_name,
                    'timestamp': range(len(univariate)),
                    'target': univariate.astype(float),
                })

                try:
                    pred_df = self.pipeline.predict_df(
                        df,
                        prediction_length=prediction_length,
                        quantile_levels=quantile_levels_c2,
                    )

                    median = pred_df['predictions'].values[:prediction_length]
                    predictions[seq_idx, :, feat_idx] = median

                    if '0.1' in pred_df.columns:
                        quantiles_out[seq_idx, :, feat_idx, 0] = pred_df['0.1'].values[:prediction_length]
                        quantiles_out[seq_idx, :, feat_idx, 1] = median
                        quantiles_out[seq_idx, :, feat_idx, 2] = pred_df['0.9'].values[:prediction_length]
                    else:
                        quantiles_out[seq_idx, :, feat_idx, :] = np.expand_dims(median, -1)

                except Exception as e:
                    last_val = float(univariate[-1]) if len(univariate) > 0 else 0.0
                    predictions[seq_idx, :, feat_idx] = last_val
                    quantiles_out[seq_idx, :, feat_idx, :] = last_val

        total_time = time.time() - start_time
        print(f"Completed in {total_time:.1f}s")
        print(f"Output shape: {predictions.shape}")

        return predictions, quantiles_out

    # Required abstract methods (not used for zero-shot)
    def prepare_for_training(self, mode: str = "lora"):
        raise NotImplementedError("Training not implemented")

    def save_adapter(self, path: str):
        raise NotImplementedError("Saving not implemented")

    def load_adapter(self, path: str):
        raise NotImplementedError("Loading not implemented")
