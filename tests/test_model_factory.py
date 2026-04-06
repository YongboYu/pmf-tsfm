"""Tests for the model adapter factory."""

import pytest
from omegaconf import OmegaConf

import pmf_tsfm.models as models_module


@pytest.mark.parametrize(
    ("family", "adapter_attr"),
    [
        ("chronos", "ChronosAdapter"),
        ("moirai", "MoiraiAdapter"),
        ("timesfm", "TimesFMAdapter"),
    ],
)
def test_get_model_adapter_dispatches_to_family_factory(
    family: str, adapter_attr: str, monkeypatch
) -> None:
    sentinel = object()
    captured: dict[str, object] = {}

    def fake_from_config(model_cfg, device, prediction_length):
        captured["family"] = model_cfg.family
        captured["device"] = device
        captured["prediction_length"] = prediction_length
        return sentinel

    monkeypatch.setattr(
        getattr(models_module, adapter_attr),
        "from_config",
        staticmethod(fake_from_config),
    )

    cfg = OmegaConf.create({"family": family})
    adapter = models_module.get_model_adapter(cfg, device="cpu", prediction_length=11)

    assert adapter is sentinel
    assert captured == {
        "family": family,
        "device": "cpu",
        "prediction_length": 11,
    }


def test_get_model_adapter_rejects_unknown_family() -> None:
    cfg = OmegaConf.create({"family": "unknown"})

    with pytest.raises(ValueError, match="Unknown model family"):
        models_module.get_model_adapter(cfg)
