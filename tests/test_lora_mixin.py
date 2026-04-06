"""Tests for the LoRA fine-tuning mixin."""

import json
import sys
import types
from pathlib import Path

import pytest
import torch.nn as nn

from pmf_tsfm.models.mixins.lora_mixin import LoRAMixin


class DummyPeftModel(nn.Module):
    """Small PEFT-like wrapper used to test the mixin offline."""

    def __init__(self, base_model: nn.Module, config=None) -> None:
        super().__init__()
        self.base_model = base_model
        self.config = config
        self.last_device: str | None = None
        self.printed = False
        self.loaded_from: str | None = None

    def print_trainable_parameters(self) -> None:
        self.printed = True

    def save_pretrained(self, output_dir: str) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "adapter_config.json").write_text(json.dumps({"mock": True}))

    def to(self, device: str):
        self.last_device = device
        return self


class DummyLoRAAdapter(LoRAMixin):
    """Minimal concrete adapter for testing LoRAMixin behavior."""

    def __init__(self) -> None:
        self.device = "cpu"
        self.prediction_length = 2
        self.pipeline = None
        self.base_module = None
        self._model = None
        self._is_loaded = False
        self._lora_applied = False
        self._lora_adapter_path = None
        self._peft_model = None
        self._full_tune_enabled = False
        self._full_tune_model = None
        self.context_lengths: list[int] = []

    def load_model(self) -> None:
        self._is_loaded = True

    def _create_base_model_for_lora(self, context_length: int) -> nn.Module:
        self.context_lengths.append(context_length)
        return nn.Linear(3, 3)

    def _get_default_lora_targets(self) -> list[str]:
        return ["q_proj", "v_proj"]


def _install_fake_peft(monkeypatch) -> None:
    class FakeLoraConfig:
        def __init__(
            self,
            *,
            r,
            lora_alpha,
            lora_dropout,
            target_modules,
            bias,
            task_type,
        ) -> None:
            self.r = r
            self.lora_alpha = lora_alpha
            self.lora_dropout = lora_dropout
            self.target_modules = target_modules
            self.bias = bias
            self.task_type = task_type

    def fake_get_peft_model(base_model, peft_config):
        return DummyPeftModel(base_model, peft_config)

    class FakePeftModel:
        @staticmethod
        def from_pretrained(base_model, adapter_path: str):
            model = DummyPeftModel(base_model)
            model.loaded_from = adapter_path
            return model

    fake_peft = types.ModuleType("peft")
    fake_peft.LoraConfig = FakeLoraConfig
    fake_peft.PeftModel = FakePeftModel
    fake_peft.get_peft_model = fake_get_peft_model
    monkeypatch.setitem(sys.modules, "peft", fake_peft)


class TestLoRAMixin:
    def test_apply_lora_requires_loaded_model(self) -> None:
        adapter = DummyLoRAAdapter()

        with pytest.raises(RuntimeError, match="Model not loaded"):
            adapter.apply_lora({})

    def test_apply_lora_uses_default_target_modules(self, monkeypatch) -> None:
        adapter = DummyLoRAAdapter()
        adapter.load_model()
        _install_fake_peft(monkeypatch)

        adapter.apply_lora({"r": 4, "alpha": 8}, context_length=12)

        assert adapter._lora_applied
        assert adapter.context_lengths == [12]
        assert adapter._peft_model is not None
        assert adapter._peft_model.config.target_modules == ["q_proj", "v_proj"]
        assert adapter._peft_model.printed

    def test_load_lora_adapter_sets_flags_and_adapter_path(self, tmp_path, monkeypatch) -> None:
        adapter = DummyLoRAAdapter()
        adapter.load_model()
        _install_fake_peft(monkeypatch)

        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        adapter.load_lora_adapter(str(adapter_dir), context_length=16)

        assert adapter._lora_applied
        assert adapter._lora_adapter_path == str(adapter_dir)
        assert adapter.context_lengths == [16]
        assert adapter._peft_model is not None
        assert adapter._peft_model.loaded_from == str(adapter_dir)

    def test_save_lora_adapter_requires_applied_model_then_writes_files(
        self, tmp_path, monkeypatch
    ) -> None:
        adapter = DummyLoRAAdapter()

        with pytest.raises(RuntimeError, match="No LoRA adapter to save"):
            adapter.save_lora_adapter(str(tmp_path / "adapter"))

        adapter.load_model()
        _install_fake_peft(monkeypatch)
        adapter.apply_lora({}, context_length=10)

        out_dir = tmp_path / "adapter"
        adapter.save_lora_adapter(str(out_dir))

        assert (out_dir / "adapter_config.json").exists()

    def test_get_trainable_parameters_model_precedence_and_lora_to(self, monkeypatch) -> None:
        adapter = DummyLoRAAdapter()
        adapter.load_model()
        _install_fake_peft(monkeypatch)
        adapter.apply_lora({}, context_length=8)

        trainable, total = adapter.get_trainable_parameters()
        assert trainable > 0
        assert total >= trainable

        pipeline = object()
        base_module = object()
        fallback = object()
        full_tune_model = object()

        adapter._peft_model = DummyPeftModel(nn.Linear(2, 2))
        assert adapter.model is adapter._peft_model

        adapter.lora_to("cpu")
        assert adapter._peft_model.last_device == "cpu"

        adapter._full_tune_enabled = True
        adapter._full_tune_model = full_tune_model
        assert adapter.model is full_tune_model

        adapter._full_tune_enabled = False
        adapter._full_tune_model = None
        adapter._peft_model = None
        adapter.pipeline = pipeline
        assert adapter.model is pipeline

        adapter.pipeline = None
        adapter.base_module = base_module
        assert adapter.model is base_module

        adapter.base_module = None
        adapter.model = fallback
        assert adapter.model is fallback
