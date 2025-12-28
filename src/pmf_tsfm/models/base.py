"""Base adapter class for time series foundation models."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch


class BaseAdapter(ABC):
    """
    Base class for all model adapters.

    Provides a unified interface for loading models and generating predictions.
    Supports multivariate time series by treating each feature as a univariate series.
    """

    def __init__(
        self,
        model_name: str,
        model_id: str,
        model_family: str,
        variant: str,
        device: str = "mps",
        torch_dtype: str = "float32",
        prediction_length: int = 7,
        **kwargs,
    ):
        """
        Initialize the base adapter.

        Args:
            model_name: Display name for the model
            model_id: HuggingFace model ID or S3 path
            model_family: Model family (chronos, moirai)
            variant: Model variant (e.g., bolt_small, 1_1_large)
            device: Device to run on ('cpu', 'cuda', 'mps')
            torch_dtype: Data type ('float32', 'float16', 'bfloat16')
            prediction_length: Forecasting horizon
        """
        self.model_name = model_name
        self.model_id = model_id
        self.model_family = model_family
        self.variant = variant
        self.device = device
        self.torch_dtype = self._parse_dtype(torch_dtype)
        self.prediction_length = prediction_length
        self.kwargs = kwargs

        # Model state
        self.model: Any | None = None
        self.pipeline: Any | None = None
        self._is_loaded = False

    def _parse_dtype(self, dtype_str: str) -> torch.dtype:
        """Parse dtype string to torch dtype."""
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        return dtype_map.get(dtype_str, torch.float32)

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded

    @abstractmethod
    def load_model(self) -> None:
        """Load the model. Must be called before predict()."""
        pass

    @abstractmethod
    def predict(
        self,
        prepared_data: dict[str, Any],
        prediction_length: int | None = None,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions for multivariate time series.

        Args:
            prepared_data: Data dict from ZeroShotDataModule with keys:
                - inputs: List of (seq_len, num_features) arrays
                - targets: (num_sequences, prediction_length, num_features) array
                - feature_names: List of feature names
            prediction_length: Override prediction length
            **kwargs: Additional model-specific parameters

        Returns:
            predictions: (num_sequences, prediction_length, num_features)
            quantiles: (num_sequences, prediction_length, num_features, num_quantiles)
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.model_name}, variant={self.variant})"
