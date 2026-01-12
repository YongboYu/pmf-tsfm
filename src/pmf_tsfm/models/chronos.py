"""
Chronos adapter for zero-shot time series forecasting.

Supports:
- Chronos Bolt: tiny, mini, small, base (uses predict_quantiles API)
- Chronos 2.0: uses predict_df API with DataFrames (s3://autogluon/chronos-2/)

Handles multivariate time series by forecasting each feature separately.
"""

import time
import warnings
from typing import Any

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

from pmf_tsfm.models.base import BaseAdapter
from pmf_tsfm.models.mixins import LoRAMixin

# Suppress deprecation warnings
warnings.filterwarnings("ignore", message=".*torch_dtype.*deprecated.*")
warnings.filterwarnings("ignore", message=".*past_key_values.*deprecated.*")
warnings.filterwarnings("ignore", message=".*dtype.*deprecated.*")


class ChronosAdapter(BaseAdapter, LoRAMixin):
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
        quantile_levels: list[float] | None = None,
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
        self.pipeline: Any = None  # Set in load_model()

        # LoRA fine-tuning state
        self._lora_applied = False
        self._lora_adapter_path: str | None = None
        self._peft_model = None  # Stores PEFT-wrapped model for training

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
        prepared_data: dict[str, Any],
        prediction_length: int | None = None,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
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
        prepared_data: dict[str, Any],
        prediction_length: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Chronos Bolt prediction using predict_quantiles API."""
        if self.pipeline is None:
            raise RuntimeError("Model pipeline not loaded. Call load_model() first.")
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
                    assert self.pipeline is not None
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
        prepared_data: dict[str, Any],
        prediction_length: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Chronos 2.0 prediction using predict_df API with batched features.

        Batches all features for each sequence into a single predict_df call
        using different item_id values, significantly improving efficiency.
        """
        inputs = prepared_data["inputs"]
        feature_names = prepared_data["feature_names"]
        if self.pipeline is None:
            raise RuntimeError("Model pipeline not loaded. Call load_model() first.")

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
                assert self.pipeline is not None
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

    # ========================================================================
    # LoRA Fine-Tuning Methods (via LoRAMixin)
    # ========================================================================

    def _create_base_model_for_lora(self, context_length: int):
        """Create Chronos model for LoRA wrapping.

        Only supported for Chronos Bolt models (T5-based).
        Chronos 2.0 uses a different architecture and is not supported.

        Returns the inner T5 model from the ChronosBoltPipeline.
        """
        if self._is_chronos2:
            raise NotImplementedError(
                "LoRA fine-tuning is not supported for Chronos 2.0. "
                "Use Chronos Bolt (tiny/mini/small/base) instead."
            )

        if self.pipeline is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # ChronosBoltPipeline has 'inner_model' attribute for the T5 model
        # Store the pipeline reference for later use
        self._context_length = context_length

        # Get the inner model (T5ForConditionalGeneration)
        if hasattr(self.pipeline, "inner_model"):
            return self.pipeline.inner_model
        elif hasattr(self.pipeline, "model"):
            return self.pipeline.model
        else:
            raise AttributeError(
                "Cannot find inner model in Chronos pipeline. "
                "Expected 'inner_model' or 'model' attribute."
            )

    def _get_default_lora_targets(self) -> list[str]:
        """Return default LoRA target modules for Chronos (T5-based).

        Chronos Bolt uses T5 architecture with attention projections:
        - q, k, v, o for attention
        """
        # T5 attention layer names (matching reference implementation)
        return ["q", "v"]  # Minimal set from reference, can expand to ["q", "k", "v", "o"]

    def apply_lora(self, lora_config: dict, context_length: int = 48) -> None:
        """Apply LoRA adaptation to Chronos Bolt model.

        Overrides the mixin method to handle Chronos-specific setup.

        Args:
            lora_config: Dict with keys: r, alpha, dropout, target_modules, bias
            context_length: Fixed context length for training
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if self._is_chronos2:
            raise NotImplementedError(
                "LoRA fine-tuning is not supported for Chronos 2.0. "
                "Use Chronos Bolt (tiny/mini/small/base) instead."
            )

        from peft import LoraConfig, get_peft_model

        # Get the inner model
        inner_model = self._create_base_model_for_lora(context_length)

        # Get target modules (use config or defaults)
        target_modules = lora_config.get("target_modules")
        if target_modules is None:
            target_modules = self._get_default_lora_targets()

        peft_config = LoraConfig(
            r=lora_config.get("r", 2),
            lora_alpha=lora_config.get("alpha", 4),
            lora_dropout=lora_config.get("dropout", 0.1),
            target_modules=list(target_modules),
            bias=lora_config.get("bias", "none"),
        )

        print("Applying LoRA to Chronos Bolt...")
        print(f"  Rank: {peft_config.r}")
        print(f"  Alpha: {peft_config.lora_alpha}")
        print(f"  Target modules: {peft_config.target_modules}")

        self._peft_model = get_peft_model(inner_model, peft_config)
        self._peft_model.print_trainable_parameters()
        self._lora_applied = True

        # Store context length for training
        self._context_length = context_length

    def load_lora_adapter(self, adapter_path: str, context_length: int = 48) -> None:
        """Load a pre-trained LoRA adapter for Chronos Bolt inference.

        Args:
            adapter_path: Path to the saved LoRA adapter
            context_length: Context length for the model (should match training)
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if self._is_chronos2:
            raise NotImplementedError("LoRA fine-tuning is not supported for Chronos 2.0.")

        from peft import PeftModel

        # Get the inner model
        inner_model = self._create_base_model_for_lora(context_length)

        print(f"Loading LoRA adapter from: {adapter_path}")
        self._peft_model = PeftModel.from_pretrained(inner_model, adapter_path)

        # Merge LoRA weights for efficient inference
        print("  Merging LoRA weights...")
        self._peft_model = self._peft_model.merge_and_unload()

        # Update the pipeline's inner model with merged weights
        if hasattr(self.pipeline, "inner_model"):
            self.pipeline.inner_model = self._peft_model
        elif hasattr(self.pipeline, "model"):
            self.pipeline.model = self._peft_model

        self._lora_applied = True
        self._lora_adapter_path = adapter_path
        print("  LoRA adapter loaded and merged successfully")

    def forward_train(
        self,
        batch: dict,
    ) -> tuple:
        """Forward pass for Chronos Bolt training.

        Args:
            batch: Dict with 'context' (past values) and 'target' (future values)

        Returns:
            Tuple of (predictions, loss)
        """
        if self._peft_model is None:
            raise RuntimeError("No PEFT model available. Call apply_lora() first.")

        # Expected format from DataLoader: context (B, context_len), target (B, pred_len)
        context = batch["context"].to(self.device).float()
        target = batch["target"].to(self.device).float()

        # Chronos Bolt model takes context/target float tensors and returns a loss.
        try:
            outputs = self._peft_model(
                context=context,
                target=target,
            )

            loss = outputs.loss
            quantiles = outputs.quantile_preds if hasattr(outputs, "quantile_preds") else None

            return quantiles, loss

        except Exception as e:
            raise RuntimeError(
                f"Chronos forward pass failed: {e}. "
                "Chronos Bolt training expects context/target float tensors. "
                "See temp/lora_tune/chronos_lora.py for a working reference."
            )

    def to(self, device: str) -> "ChronosAdapter":
        """Move model to device."""
        self.device = device
        if self._peft_model is not None:
            self._peft_model = self._peft_model.to(device)
        return self
