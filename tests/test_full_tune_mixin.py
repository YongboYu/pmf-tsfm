"""Tests for the full fine-tuning mixin."""

import pytest
import torch
import torch.nn as nn

from pmf_tsfm.models.mixins.full_tune_mixin import FullTuneMixin


class TinyTuneModule(nn.Module):
    """Tiny trainable module with a traceable .to() call."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2)
        self.last_device: str | None = None
        for param in self.parameters():
            param.requires_grad = False

    def to(self, device: str):
        self.last_device = device
        return self


class DummyFullTuneAdapter(FullTuneMixin):
    """Concrete adapter for unit-testing FullTuneMixin."""

    def __init__(self) -> None:
        self.device = "cpu"
        self.pipeline = None
        self.base_module = None
        self._peft_model = None
        self._full_tune_model = None
        self._full_tune_enabled = False
        self._checkpoint_path = None
        self._is_loaded = False
        self.context_lengths: list[int] = []

    def load_model(self) -> None:
        self._is_loaded = True

    def _get_trainable_model(self):
        return self._full_tune_model

    def _create_base_model_for_training(self, context_length: int):
        self.context_lengths.append(context_length)
        return TinyTuneModule()


class TestFullTuneMixin:
    def test_prepare_for_full_tuning_requires_loaded_model(self) -> None:
        adapter = DummyFullTuneAdapter()

        with pytest.raises(RuntimeError, match="Model not loaded"):
            adapter.prepare_for_full_tuning()

    def test_prepare_for_full_tuning_unfreezes_parameters_and_counts_them(self) -> None:
        adapter = DummyFullTuneAdapter()
        adapter.load_model()

        adapter.prepare_for_full_tuning(context_length=12)

        assert adapter._full_tune_enabled
        assert adapter.context_lengths == [12]
        assert all(param.requires_grad for param in adapter._full_tune_model.parameters())
        trainable, total = adapter.get_full_tune_parameters()
        assert trainable > 0
        assert trainable == total

    @pytest.mark.parametrize("use_file_path", [False, True])
    def test_save_and_load_full_checkpoint_accepts_directory_and_file_paths(
        self, tmp_path, use_file_path: bool
    ) -> None:
        source = DummyFullTuneAdapter()
        source.load_model()
        source.prepare_for_full_tuning(context_length=6)
        checkpoint_dir = tmp_path / "ckpt"
        source.save_full_checkpoint(str(checkpoint_dir))

        target = DummyFullTuneAdapter()
        target.load_model()
        checkpoint_path = checkpoint_dir / "model.pt" if use_file_path else checkpoint_dir

        target.load_full_checkpoint(str(checkpoint_path), context_length=6)

        assert target._full_tune_enabled
        assert target._checkpoint_path == str(checkpoint_dir / "model.pt")
        assert target.context_lengths == [6]
        assert torch.allclose(
            target._full_tune_model.linear.weight,
            source._full_tune_model.linear.weight,
        )

    def test_load_full_checkpoint_raises_for_missing_file(self, tmp_path) -> None:
        adapter = DummyFullTuneAdapter()
        adapter.load_model()

        with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
            adapter.load_full_checkpoint(str(tmp_path / "missing"), context_length=4)

    def test_model_precedence_and_full_tune_to(self) -> None:
        adapter = DummyFullTuneAdapter()
        peft_model = object()
        pipeline = object()
        base_module = object()

        adapter._peft_model = peft_model
        assert adapter.model is peft_model

        adapter._peft_model = None
        adapter.pipeline = pipeline
        assert adapter.model is pipeline

        delattr(adapter, "pipeline")
        adapter.base_module = base_module
        assert adapter.model is base_module

        adapter.load_model()
        adapter.prepare_for_full_tuning(context_length=8)
        full_model = adapter._full_tune_model
        adapter.full_tune_to("cpu")
        assert full_model.last_device == "cpu"
        assert adapter.model is full_model
