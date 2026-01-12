"""
Moirai adapter for zero-shot time series forecasting.

Supports:
- Moirai 1.1: small, base, large
- Moirai 2.0: small (and others as released)
- Moirai MoE: base (Mixture of Experts)

Handles multivariate time series by forecasting each feature separately.
"""

import time
import warnings
from typing import Any

import numpy as np
from omegaconf import DictConfig

from pmf_tsfm.models.base import BaseAdapter
from pmf_tsfm.models.mixins import LoRAMixin

# Suppress warnings
warnings.filterwarnings("ignore")


class MoiraiAdapter(BaseAdapter, LoRAMixin):
    """
    Unified adapter for Salesforce Moirai models.

    Automatically detects model version (1.1, 2.0, MoE) from variant
    and uses the appropriate module.
    """

    @classmethod
    def from_config(
        cls,
        model_cfg: DictConfig,
        device: str = "mps",
        prediction_length: int = 7,
    ) -> "MoiraiAdapter":
        """Create adapter from Hydra model config."""
        return cls(
            model_name=model_cfg.name,
            model_id=model_cfg.id,
            variant=model_cfg.variant,
            device=device,
            torch_dtype=model_cfg.get("torch_dtype", "float32"),
            prediction_length=prediction_length,
            num_samples=model_cfg.get("num_samples", 100),
            patch_size=model_cfg.get("patch_size", 32),
        )

    def __init__(
        self,
        model_name: str,
        model_id: str,
        variant: str = "1_1_small",
        device: str = "mps",
        torch_dtype: str = "float32",
        prediction_length: int = 7,
        num_samples: int = 100,
        patch_size: int = 32,
        **kwargs,
    ):
        """
        Initialize the Moirai adapter.

        Args:
            model_name: Display name
            model_id: HuggingFace model ID (e.g., 'Salesforce/moirai-1.1-R-small')
            variant: Model variant (e.g., '1_1_small', '2_0_small', 'moe_base')
            device: Device to run on
            torch_dtype: Data type
            prediction_length: Forecasting horizon
            num_samples: Number of samples for probabilistic forecasts
            patch_size: Patch size (fixed at 16 for MoE)
        """
        super().__init__(
            model_name=model_name,
            model_id=model_id,
            model_family="moirai",
            variant=variant,
            device=device,
            torch_dtype=torch_dtype,
            prediction_length=prediction_length,
            **kwargs,
        )

        self.num_samples = num_samples
        self.patch_size = patch_size

        # Detect model version
        self.model_version = self._detect_version()
        if self.model_version == "moe":
            self.patch_size = 16  # Fixed for MoE

        self.base_module = None

        # LoRA fine-tuning state
        self._lora_applied = False
        self._lora_adapter_path: str | None = None
        self._peft_model = None  # Stores PEFT-wrapped model for training

    def _detect_version(self) -> str:
        """Detect Moirai version from variant."""
        variant_lower = self.variant.lower()

        if "moe" in variant_lower:
            return "moe"
        elif "2_0" in variant_lower or "2.0" in variant_lower:
            return "2.0"
        else:
            return "1.1"

    def load_model(self) -> None:
        """Load Moirai model based on version."""
        print(f"Loading Moirai {self.model_version} model: {self.model_id}")
        start_time = time.time()

        try:
            if self.model_version == "moe":
                from uni2ts.model.moirai_moe import MoiraiMoEModule

                self.base_module = MoiraiMoEModule.from_pretrained(self.model_id)
            elif self.model_version == "2.0":
                from uni2ts.model.moirai2 import Moirai2Module

                self.base_module = Moirai2Module.from_pretrained(self.model_id)
            else:  # 1.1
                from uni2ts.model.moirai import MoiraiModule

                self.base_module = MoiraiModule.from_pretrained(self.model_id)

            self._is_loaded = True
            load_time = time.time() - start_time
            print(f"  Loaded in {load_time:.2f}s on {self.device}")

        except ImportError as e:
            raise ImportError(f"uni2ts package not found. Install: pip install uni2ts\nError: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load {self.model_id}: {e}")

    def _create_sequence_model(self, context_length: int):
        """Create a Moirai model with specific context length."""
        if self.model_version == "moe":
            from uni2ts.model.moirai_moe import MoiraiMoEForecast

            return MoiraiMoEForecast(
                module=self.base_module,
                prediction_length=self.prediction_length,
                context_length=context_length,
                patch_size=self.patch_size,
                num_samples=self.num_samples,
                target_dim=1,
                feat_dynamic_real_dim=0,
                past_feat_dynamic_real_dim=0,
            )
        elif self.model_version == "2.0":
            from uni2ts.model.moirai2 import Moirai2Forecast

            return Moirai2Forecast(
                module=self.base_module,
                prediction_length=self.prediction_length,
                context_length=context_length,
                target_dim=1,
                feat_dynamic_real_dim=0,
                past_feat_dynamic_real_dim=0,
            )
        else:  # 1.1
            from uni2ts.model.moirai import MoiraiForecast

            return MoiraiForecast(
                module=self.base_module,
                prediction_length=self.prediction_length,
                context_length=context_length,
                target_dim=1,
                feat_dynamic_real_dim=0,
                past_feat_dynamic_real_dim=0,
            )

    def _forecast_single_variable(
        self,
        univariate_series: np.ndarray,
        context_length: int | None = None,
    ) -> dict[str, Any]:
        """Forecast a single univariate time series."""
        from gluonts.dataset.common import ListDataset

        try:
            if context_length is None:
                context_length = len(univariate_series)

            model = self._create_sequence_model(context_length)
            predictor = model.create_predictor(batch_size=16)

            data = ListDataset(
                [{"target": univariate_series.tolist(), "start": "2020-01-01"}],
                freq="D",
            )

            forecasts = list(predictor.predict(data))
            forecast = forecasts[0]

            # Extract quantiles
            if hasattr(forecast, "quantile"):
                low = forecast.quantile(0.1)
                median = forecast.quantile(0.5)
                high = forecast.quantile(0.9)
            elif hasattr(forecast, "mean"):
                median = forecast.mean
                low = median * 0.9
                high = median * 1.1
            else:
                raise RuntimeError(f"Unexpected forecast type: {type(forecast)}")

            # Ensure correct length
            return {
                "median": np.asarray(median)[: self.prediction_length],
                "low": np.asarray(low)[: self.prediction_length],
                "high": np.asarray(high)[: self.prediction_length],
                "success": True,
            }

        except Exception:
            # Fallback to last value
            last_val = float(univariate_series[-1]) if len(univariate_series) > 0 else 0.0
            median = np.full(self.prediction_length, last_val)
            return {
                "median": median,
                "low": median * 0.95,
                "high": median * 1.05,
                "success": False,
            }

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
            quantiles: (num_sequences, prediction_length, num_features, 3)
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        pred_len = prediction_length or self.prediction_length

        inputs = prepared_data["inputs"]
        feature_names = prepared_data["feature_names"]

        num_sequences = len(inputs)
        num_features = len(feature_names)

        # Initialize outputs (3 quantiles: low, median, high)
        predictions = np.zeros((num_sequences, pred_len, num_features))
        quantiles_out = np.zeros((num_sequences, pred_len, num_features, 3))

        print(f"Forecasting {num_sequences} sequences × {num_features} features...")

        failed = 0
        total = 0
        start_time = time.time()

        for seq_idx in range(num_sequences):
            if seq_idx % 10 == 0:
                elapsed = time.time() - start_time
                print(f"  Progress: {seq_idx}/{num_sequences} ({elapsed:.1f}s)")

            input_seq = inputs[seq_idx]
            context_len = input_seq.shape[0]

            for feat_idx in range(num_features):
                univariate = input_seq[:, feat_idx]

                result = self._forecast_single_variable(univariate, context_len)

                predictions[seq_idx, :, feat_idx] = result["median"]
                quantiles_out[seq_idx, :, feat_idx, 0] = result["low"]
                quantiles_out[seq_idx, :, feat_idx, 1] = result["median"]
                quantiles_out[seq_idx, :, feat_idx, 2] = result["high"]

                total += 1
                if not result["success"]:
                    failed += 1

        total_time = time.time() - start_time
        success_rate = ((total - failed) / total * 100) if total > 0 else 0

        print(f"Completed in {total_time:.1f}s | Success: {success_rate:.1f}%")
        print(f"Output: {predictions.shape}")

        return predictions, quantiles_out

    # ========================================================================
    # LoRA Fine-Tuning Methods (via LoRAMixin)
    # ========================================================================

    def _create_base_model_for_lora(self, context_length: int):
        """Create Moirai forecast model for LoRA wrapping.

        Only supported for Moirai 1.1 models currently.
        """
        if self.model_version != "1.1":
            raise NotImplementedError(
                f"LoRA fine-tuning only supported for Moirai 1.1, got {self.model_version}"
            )

        from uni2ts.model.moirai import MoiraiForecast

        return MoiraiForecast(
            module=self.base_module,
            prediction_length=self.prediction_length,
            context_length=context_length,
            patch_size=self.patch_size,
            num_samples=self.num_samples,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )

    def _get_default_lora_targets(self) -> list[str]:
        """Return default LoRA target modules for Moirai."""
        return ["q_proj", "k_proj", "v_proj", "out_proj"]

    def forward_train(
        self,
        batch: dict,
    ) -> tuple:
        """Forward pass for training returning predictions and loss.

        Args:
            batch: Dict with past_target, past_observed_target, past_is_pad, future_target

        Returns:
            Tuple of (predictions, loss)
        """
        import torch

        if self._peft_model is None:
            raise RuntimeError("No PEFT model available. Call apply_lora() first.")

        past_target = batch["past_target"].to(self.device).float()
        past_observed_target = batch["past_observed_target"].to(self.device).bool()
        past_is_pad = batch["past_is_pad"].to(self.device).bool()
        future_target = batch["future_target"].to(self.device).float()

        # Get distribution from model
        outputs = self._peft_model._get_distr(
            patch_size=self.patch_size,
            past_target=past_target,
            past_observed_target=past_observed_target,
            past_is_pad=past_is_pad,
        )

        # Format predictions
        preds_mean = outputs.mean
        if preds_mean.ndim == 3:
            preds_mean = preds_mean.unsqueeze(0)
        predictions = self._peft_model._format_preds(self.patch_size, preds_mean, 1)
        if predictions.ndim == 3:
            predictions = predictions.mean(dim=1)
        if predictions.ndim == 2:
            predictions = predictions.unsqueeze(-1)

        # Compute MSE loss
        loss = torch.nn.MSELoss()(predictions, future_target)

        return predictions, loss

    def to(self, device: str) -> "MoiraiAdapter":
        """Move model to device."""
        self.device = device
        if self._peft_model is not None:
            self._peft_model = self._peft_model.to(device)
        return self
