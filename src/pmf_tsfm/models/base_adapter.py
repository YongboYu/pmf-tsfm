from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import torch


class BaseAdapter(ABC):
    
    def __init__(
        self,
        model_name: str,
        model_id: str,
        model_type: str,
        variant: str,
        device_map: str = "auto",
        torch_dtype: str = "float32",
        **kwargs
    ):
        self.model_name = model_name
        self.model_id = model_id
        self.model_type = model_type
        self.variant = variant
        self.device_map = device_map
        self.torch_dtype = self._parse_dtype(torch_dtype)
        self.kwargs = kwargs
        
        self.model = None
        self.pipeline = None
        
    def _parse_dtype(self, dtype_str: str):
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        return dtype_map.get(dtype_str, torch.float32)
    
    @abstractmethod
    def load_model(self):
        pass
    
    @abstractmethod
    def predict(
        self,
        prepared_data: Dict[str, Any],
        prediction_length: int,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions
        
        Args:
            prepared_data: Data dictionary from data module
            prediction_length: Number of steps to predict
            **kwargs: Additional model-specific parameters
            
        Returns:
            Tuple of (predictions, quantiles)
        """
        pass
    
    @abstractmethod
    def prepare_for_training(self, mode: str = "lora"):
        pass
