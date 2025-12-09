"""Model adapters for time series foundation models."""

from pmf_tsfm.models.base import BaseAdapter
from pmf_tsfm.models.chronos import ChronosAdapter
from pmf_tsfm.models.moirai import MoiraiAdapter


def get_model_adapter(model_cfg, device: str = "mps", prediction_length: int = 7) -> BaseAdapter:
    """
    Factory function to create model adapter from config.

    Args:
        model_cfg: Hydra model config with 'family' field
        device: Device to run on
        prediction_length: Forecasting horizon

    Returns:
        Appropriate model adapter instance
    """
    family = model_cfg.family.lower()

    if family == "chronos":
        return ChronosAdapter.from_config(model_cfg, device, prediction_length)
    elif family == "moirai":
        return MoiraiAdapter.from_config(model_cfg, device, prediction_length)
    else:
        raise ValueError(f"Unknown model family: {family}. Supported: chronos, moirai")


__all__ = [
    "BaseAdapter",
    "ChronosAdapter",
    "MoiraiAdapter",
    "get_model_adapter",
]
