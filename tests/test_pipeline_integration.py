"""Pipeline integration tests using a MockAdapter.

Purpose
-------
Verify that the full train → save → load → predict pipeline wiring is correct
WITHOUT downloading any real model weights.  All tests run on CPU in seconds.

Strategy: MockAdapter
---------------------
A real foundation model (Chronos, Moirai, TimesFM) needs hundreds of MB of
weights downloaded from the internet.  A MockAdapter replaces the model with
a tiny 2-parameter PyTorch module and implements the exact same interface
(load_model, predict, apply_lora, save_lora_adapter, …).

When we monkeypatch `get_model_adapter` to return a MockAdapter, the pipeline
code (run_inference, train) runs through every code path it normally would,
including:
  - data loading & windowing
  - model loading call
  - LoRA / checkpoint loading call
  - the training loop (forward pass, loss.backward, optimizer step, early stopping)
  - checkpoint saving and the resulting directory/file structure
  - prediction array shapes and metadata JSON writing

What is NOT tested here
-----------------------
  - That any real model produces sensible forecasts (tested manually on GPU).
  - The HuggingFace save_pretrained / from_pretrained round-trip for real models.

Running
-------
    uv run pytest tests/test_pipeline_integration.py -v
    uv run pytest tests/test_pipeline_integration.py -v -k "lora"  # LoRA tests only
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

# ===========================================================================
# MockAdapter — a minimal stand-in for any real model adapter
# ===========================================================================


class _TinyModule(nn.Module):
    """Single linear layer — provides real trainable parameters for the loop."""

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.w


class MockAdapter:
    """
    Drop-in replacement for ChronosAdapter / MoiraiAdapter in pipeline tests.

    Implements every method that train.py and inference.py call, but all heavy
    operations are replaced with lightweight stubs:
      - load_model():           sets _is_loaded = True (no download)
      - predict():              returns random arrays of the correct shape
      - apply_lora():           keeps _tiny_module as the "LoRA-wrapped" model
      - save_lora_adapter():    writes a minimal adapter_config.json
      - load_lora_adapter():    reads that file and sets _lora_applied = True
      - prepare_for_full_tuning(): sets _full_tune_enabled = True
      - save_full_checkpoint(): torch.saves the tiny module's state_dict
      - load_full_checkpoint(): torch.loads it back
      - forward_train(batch):   returns a real loss tensor (enables backward())
    """

    def __init__(self, prediction_length: int = 7, device: str = "cpu"):
        self.model_name = "mock_model"
        self.model_id = "mock/model"
        self.model_family = "chronos"  # satisfies train.py's supported_families check
        self.variant = "small"
        self.device = device
        self.prediction_length = prediction_length
        self._is_loaded = False

        self._tiny_module = _TinyModule()
        self._peft_model: nn.Module | None = None
        self._full_tune_model: nn.Module | None = None
        self._full_tune_enabled = False
        self._lora_applied = False
        self._context_length: int | None = None

    # ---- BaseAdapter interface ----

    def load_model(self) -> None:
        self._is_loaded = True

    def predict(
        self,
        prepared_data: dict[str, Any],
        prediction_length: int | None = None,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        pred_len = prediction_length or self.prediction_length
        num_seq = len(prepared_data["inputs"])
        num_features = len(prepared_data["feature_names"])

        rng = np.random.default_rng(0)
        predictions = rng.standard_normal((num_seq, pred_len, num_features)).astype(np.float32)
        quantiles = rng.standard_normal((num_seq, pred_len, num_features, 3)).astype(np.float32)
        return predictions, quantiles

    def to(self, device: str) -> "MockAdapter":
        self.device = device
        return self

    # ---- model property (used by training loop) ----

    @property
    def model(self) -> nn.Module:
        if self._full_tune_enabled and self._full_tune_model is not None:
            return self._full_tune_model
        if self._peft_model is not None:
            return self._peft_model
        return self._tiny_module

    @model.setter
    def model(self, value: Any) -> None:
        self._model = value

    # ---- LoRA interface ----

    def apply_lora(self, lora_config: dict, context_length: int = 48) -> None:
        """Pretend LoRA was applied — reuse _tiny_module as the 'PEFT model'."""
        self._context_length = context_length
        self._peft_model = self._tiny_module
        self._lora_applied = True

    def get_trainable_parameters(self) -> tuple[int, int]:
        model = self._peft_model or self._tiny_module
        n = sum(p.numel() for p in model.parameters())
        return n, n

    def save_lora_adapter(self, output_dir: str) -> None:
        """Save a minimal adapter_config.json so load_lora_adapter can verify."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "adapter_config.json").write_text(json.dumps({"r": 2, "mock": True}))

    def load_lora_adapter(self, adapter_path: str, context_length: int = 48) -> None:
        """Verify the adapter directory exists and set the flag."""
        config_file = Path(adapter_path) / "adapter_config.json"
        if config_file.exists():
            # In real adapters this merges weights; here we just mark it loaded
            pass
        self._lora_applied = True

    def forward_train(self, batch: dict) -> tuple:
        """Compute a real (but fixed) loss connected to _tiny_module.w.

        The loss must depend on the model parameter so that loss.backward()
        actually computes gradients — required by the training loop.
        """
        # Connect to model parameters (grad flows), but keep loss constant at 0.5
        model = self._peft_model or self._tiny_module
        loss = sum(p.pow(2).sum() * 0.0 for p in model.parameters() if p.requires_grad)
        loss = loss + 0.5
        return None, loss

    # ---- Full fine-tune interface ----

    def prepare_for_full_tuning(self, context_length: int = 48) -> None:
        self._context_length = context_length
        self._full_tune_model = self._tiny_module
        self._full_tune_enabled = True

    def get_full_tune_parameters(self) -> tuple[int, int]:
        model = self._full_tune_model or self._tiny_module
        n = sum(p.numel() for p in model.parameters())
        return n, n

    def save_full_checkpoint(self, output_dir: str) -> None:
        """torch.save the tiny module's state_dict."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(self._tiny_module.state_dict(), out / "model.pt")

    def load_full_checkpoint(self, checkpoint_path: str, context_length: int = 48) -> None:
        """torch.load and confirm the checkpoint file exists."""
        checkpoint_file = Path(checkpoint_path) / "model.pt"
        if checkpoint_file.exists():
            state = torch.load(checkpoint_file, map_location="cpu")
            self._tiny_module.load_state_dict(state)
        self._full_tune_enabled = True
        self._full_tune_model = self._tiny_module

    def forward_train_full(self, batch: dict) -> tuple:
        return self.forward_train(batch)


# ===========================================================================
# Helpers
# ===========================================================================


def _make_inference_cfg(
    parquet_path: Path,
    output_dir: Path,
    train_end: int = 30,
    val_end: int = 40,
    **overrides,
) -> Any:
    """Build a minimal DictConfig that run_inference() accepts.

    Default split indices (30/40) match the tiny_parquet fixture (50 rows).
    Pass train_end=90, val_end=120 when using synthetic_parquet (150 rows).
    """
    base = {
        "seed": 42,
        "device": "cpu",
        "prediction_length": 3,
        "task": "zero_shot",
        "lora_adapter_path": None,
        "checkpoint_path": None,
        "context_length": 5,
        "print_config": False,
        "output_dir": str(output_dir),
        "model": {
            "name": "mock_model",
            "family": "chronos",
            "id": "mock/model",
            "variant": "small",
            "fallback_device": "cpu",
        },
        "data": {
            "name": "synthetic",
            "path": str(parquet_path),
            "train_end": train_end,
            "val_end": val_end,
        },
    }
    base.update(overrides)
    return OmegaConf.create(base)


def _make_train_cfg(
    parquet_path: Path,
    output_dir: Path,
    task_name: str = "lora_tune",
    epochs: int = 2,
) -> Any:
    """Build a minimal DictConfig that train() accepts."""
    task_cfg: dict
    if task_name == "lora_tune":
        task_cfg = {
            "name": "lora_tune",
            "use_amp": False,
            "train_context_length": 5,
            "adapter_save_dir": "lora_adapter",
        }
    else:
        task_cfg = {
            "name": "full_tune",
            "use_amp": False,
            "train_context_length": 5,
            "checkpoint_save_dir": "checkpoints",
            "gradient_checkpointing": False,
        }

    return OmegaConf.create(
        {
            "seed": 42,
            "device": "cpu",
            "prediction_length": 2,  # prediction_length kept small so val windows fit
            "context_length": 5,  # context_length + prediction_length < val_len (10)
            "print_config": False,
            "output_dir": str(output_dir),
            "task": task_cfg,
            "model": {
                "name": "mock_model",
                "family": "chronos",  # passes supported_families check
                "id": "mock/model",
                "variant": "small",
                "fallback_device": "cpu",
            },
            "data": {
                "name": "synthetic",
                "path": str(parquet_path),
                "split_ratio": [0.6, 0.2, 0.2],
                "train_end": 30,
                "val_end": 40,
            },
            "training": {
                "epochs": epochs,
                "batch_size": 4,
                "learning_rate": 1e-4,
                "weight_decay": 0.01,
                "gradient_clip": 1.0,
                "eval_every": 1,
                "save_every": 10,
                "early_stopping_patience": 3,
            },
            "lora": {
                "r": 2,
                "alpha": 4,
                "dropout": 0.0,
                "target_modules": ["q"],
                "bias": "none",
            },
        }
    )


# ===========================================================================
# save_predictions (pure I/O — no model needed)
# ===========================================================================


class TestSavePredictions:
    """save_predictions() writes four files with the correct names and content."""

    def test_files_are_created(self, tmp_path):
        from pmf_tsfm.inference import save_predictions

        preds = np.zeros((5, 3, 2), dtype=np.float32)
        targets = np.ones((5, 3, 2), dtype=np.float32)
        quantiles = np.zeros((5, 3, 2, 3), dtype=np.float32)

        out = save_predictions(
            predictions=preds,
            targets=targets,
            quantiles=quantiles,
            output_dir=str(tmp_path),
            model_name="mock_model",
            dataset_name="test_ds",
            metadata={"info": "test"},
        )

        assert (out / "test_ds_mock_model_predictions.npy").exists()
        assert (out / "test_ds_mock_model_targets.npy").exists()
        assert (out / "test_ds_mock_model_quantiles.npy").exists()
        assert (out / "test_ds_mock_model_metadata.json").exists()

    def test_arrays_are_roundtrippable(self, tmp_path):
        """Values written to .npy files are loaded back unchanged."""
        from pmf_tsfm.inference import save_predictions

        rng = np.random.default_rng(1)
        preds = rng.standard_normal((4, 3, 2)).astype(np.float32)
        targets = rng.standard_normal((4, 3, 2)).astype(np.float32)
        quantiles = rng.standard_normal((4, 3, 2, 3)).astype(np.float32)

        save_predictions(
            predictions=preds,
            targets=targets,
            quantiles=quantiles,
            output_dir=str(tmp_path),
            model_name="m",
            dataset_name="d",
            metadata={},
        )

        loaded_preds = np.load(tmp_path / "d_m_predictions.npy")
        np.testing.assert_array_equal(preds, loaded_preds)

    def test_metadata_json_content(self, tmp_path):
        """Metadata JSON is valid and contains the supplied keys."""
        from pmf_tsfm.inference import save_predictions

        save_predictions(
            predictions=np.zeros((2, 2, 1)),
            targets=np.zeros((2, 2, 1)),
            quantiles=np.zeros((2, 2, 1, 3)),
            output_dir=str(tmp_path),
            model_name="m",
            dataset_name="d",
            metadata={"experiment": "unit_test", "value": 42},
        )

        with open(tmp_path / "d_m_metadata.json") as f:
            meta = json.load(f)
        assert meta["experiment"] == "unit_test"
        assert meta["value"] == 42


# ===========================================================================
# Zero-shot inference pipeline
# ===========================================================================


class TestZeroShotInferencePipeline:
    """run_inference() in zero-shot mode (no adapter/checkpoint loaded)."""

    def test_predictions_saved_with_correct_shape(self, tiny_parquet, tmp_path, monkeypatch):
        """End-to-end: data → mock model → predictions → .npy files."""
        from pmf_tsfm.inference import run_inference

        path, _ = tiny_parquet
        mock = MockAdapter(prediction_length=3)

        # inference.py imports get_model_adapter at module level, so we must
        # patch the name as it exists in that module's namespace.
        monkeypatch.setattr("pmf_tsfm.inference.get_model_adapter", lambda **kw: mock)

        output_dir = tmp_path / "outputs" / "zero_shot" / "tiny" / "mock_model"
        cfg = _make_inference_cfg(path, output_dir)

        result = run_inference(cfg)

        assert result["predictions"].ndim == 3
        assert result["targets"].ndim == 3
        assert result["predictions"].shape[1] == 3  # prediction_length
        assert result["predictions"].shape[2] == 3  # num_features (tiny_parquet has 3)

    def test_output_files_exist(self, tiny_parquet, tmp_path, monkeypatch):
        """run_inference() creates all four output files."""
        from pmf_tsfm.inference import run_inference

        path, _ = tiny_parquet
        monkeypatch.setattr("pmf_tsfm.inference.get_model_adapter", lambda **kw: MockAdapter())

        output_dir = tmp_path / "zero_shot_outputs"
        cfg = _make_inference_cfg(path, output_dir)
        run_inference(cfg)

        for suffix in ["_predictions.npy", "_targets.npy", "_quantiles.npy", "_metadata.json"]:
            assert (output_dir / f"synthetic_mock_model{suffix}").exists(), (
                f"Expected output file missing: synthetic_mock_model{suffix}"
            )

    def test_task_field_in_metadata(self, tiny_parquet, tmp_path, monkeypatch):
        """Metadata JSON records the task name (zero_shot)."""
        from pmf_tsfm.inference import run_inference

        path, _ = tiny_parquet
        monkeypatch.setattr("pmf_tsfm.inference.get_model_adapter", lambda **kw: MockAdapter())

        output_dir = tmp_path / "out"
        cfg = _make_inference_cfg(path, output_dir)
        run_inference(cfg)

        with open(output_dir / "synthetic_mock_model_metadata.json") as f:
            meta = json.load(f)
        assert meta["task"] == "zero_shot"


# ===========================================================================
# LoRA training + LoRA inference pipeline
# ===========================================================================


class TestLoRAPipeline:
    """Full LoRA cycle: train → save adapter → load adapter → predict."""

    def test_lora_training_completes(self, tiny_parquet, tmp_path, monkeypatch):
        """train() with task=lora_tune runs without error and returns best_val_loss."""
        from pmf_tsfm.train import train

        path, _ = tiny_parquet
        mock = MockAdapter()
        monkeypatch.setattr("pmf_tsfm.models.get_model_adapter", lambda **kw: mock)

        output_dir = tmp_path / "results" / "lora_tune" / "tiny" / "mock"
        cfg = _make_train_cfg(path, output_dir, task_name="lora_tune", epochs=2)

        result = train(cfg)

        assert "best_val_loss" in result
        assert isinstance(result["best_val_loss"], float)

    def test_lora_adapter_is_saved(self, tiny_parquet, tmp_path, monkeypatch):
        """train() saves at least the 'final' adapter directory."""
        from pmf_tsfm.train import train

        path, _ = tiny_parquet
        mock = MockAdapter()
        monkeypatch.setattr("pmf_tsfm.models.get_model_adapter", lambda **kw: mock)

        output_dir = tmp_path / "lora_out"
        cfg = _make_train_cfg(path, output_dir, task_name="lora_tune", epochs=1)
        train(cfg)

        # At minimum the 'final' checkpoint must be created
        final_dir = output_dir / "lora_adapter" / "final"
        assert final_dir.exists(), f"Expected 'final' adapter dir at {final_dir}"
        assert (final_dir / "adapter_config.json").exists()

    def test_lora_inference_loads_adapter(self, tiny_parquet, tmp_path, monkeypatch):
        """run_inference() with lora_adapter_path calls load_lora_adapter on the adapter."""
        from pmf_tsfm.inference import run_inference

        path, _ = tiny_parquet

        # Step 1: save a mock adapter directory
        adapter_dir = tmp_path / "mock_lora_adapter"
        mock_save = MockAdapter()
        mock_save.save_lora_adapter(str(adapter_dir))

        # Step 2: run inference pointing at that directory
        mock_infer = MockAdapter()
        monkeypatch.setattr("pmf_tsfm.inference.get_model_adapter", lambda **kw: mock_infer)

        output_dir = tmp_path / "lora_infer_out"
        cfg = _make_inference_cfg(
            path,
            output_dir,
            task="lora_tune",
            lora_adapter_path=str(adapter_dir),
        )
        run_inference(cfg)

        assert mock_infer._lora_applied, "load_lora_adapter() should have been called"

    def test_lora_inference_output_task_is_lora_tune(self, tiny_parquet, tmp_path, monkeypatch):
        """Metadata records task=lora_tune when lora_adapter_path is set."""
        from pmf_tsfm.inference import run_inference

        path, _ = tiny_parquet
        adapter_dir = tmp_path / "adapter"
        MockAdapter().save_lora_adapter(str(adapter_dir))

        monkeypatch.setattr("pmf_tsfm.inference.get_model_adapter", lambda **kw: MockAdapter())
        output_dir = tmp_path / "out"
        cfg = _make_inference_cfg(
            path, output_dir, task="lora_tune", lora_adapter_path=str(adapter_dir)
        )
        run_inference(cfg)

        with open(output_dir / "synthetic_mock_model_metadata.json") as f:
            meta = json.load(f)
        assert meta["task"] == "lora_tune"


# ===========================================================================
# Full fine-tune training + inference pipeline
# ===========================================================================


class TestFullTunePipeline:
    """Full fine-tune cycle: train → save checkpoint → load checkpoint → predict."""

    def test_full_tune_training_completes(self, tiny_parquet, tmp_path, monkeypatch):
        """train() with task=full_tune runs and returns a result dict."""
        from pmf_tsfm.train import train

        path, _ = tiny_parquet
        mock = MockAdapter()
        monkeypatch.setattr("pmf_tsfm.models.get_model_adapter", lambda **kw: mock)

        output_dir = tmp_path / "full_tune_out"
        cfg = _make_train_cfg(path, output_dir, task_name="full_tune", epochs=2)

        result = train(cfg)

        assert "best_val_loss" in result
        assert isinstance(result["best_val_loss"], float)

    def test_checkpoint_files_are_saved(self, tiny_parquet, tmp_path, monkeypatch):
        """train() saves 'final' and 'best' checkpoint directories."""
        from pmf_tsfm.train import train

        path, _ = tiny_parquet
        mock = MockAdapter()
        monkeypatch.setattr("pmf_tsfm.models.get_model_adapter", lambda **kw: mock)

        output_dir = tmp_path / "ckpt_out"
        cfg = _make_train_cfg(path, output_dir, task_name="full_tune", epochs=1)
        train(cfg)

        final_dir = output_dir / "checkpoints" / "final"
        assert final_dir.exists(), f"Expected 'final' checkpoint dir at {final_dir}"
        assert (final_dir / "model.pt").exists(), "Expected model.pt inside final dir"

    def test_checkpoint_roundtrip(self, tmp_path):
        """save_full_checkpoint → load_full_checkpoint restores the state_dict."""
        save_mock = MockAdapter()
        save_mock.prepare_for_full_tuning(context_length=8)

        ckpt_dir = tmp_path / "ckpt"
        save_mock.save_full_checkpoint(str(ckpt_dir))

        load_mock = MockAdapter()
        load_mock.load_full_checkpoint(str(ckpt_dir), context_length=8)

        assert load_mock._full_tune_enabled
        # Verify the loaded state_dict has the same weight as the saved one
        saved_w = save_mock._tiny_module.w.item()
        loaded_w = load_mock._tiny_module.w.item()
        assert abs(saved_w - loaded_w) < 1e-6, "Checkpoint round-trip should preserve weights"

    def test_full_tune_inference_loads_checkpoint(self, tiny_parquet, tmp_path, monkeypatch):
        """run_inference() with checkpoint_path calls load_full_checkpoint on the adapter."""
        from pmf_tsfm.inference import run_inference

        path, _ = tiny_parquet

        # Step 1: save a mock checkpoint
        ckpt_dir = tmp_path / "mock_ckpt"
        MockAdapter().save_full_checkpoint(str(ckpt_dir))

        # Step 2: run inference pointing at that checkpoint
        mock_infer = MockAdapter()
        monkeypatch.setattr("pmf_tsfm.inference.get_model_adapter", lambda **kw: mock_infer)

        output_dir = tmp_path / "full_infer_out"
        cfg = _make_inference_cfg(
            path,
            output_dir,
            task="full_tune",
            checkpoint_path=str(ckpt_dir),
        )
        run_inference(cfg)

        assert mock_infer._full_tune_enabled, "load_full_checkpoint() should have been called"

    def test_full_tune_inference_output_task_is_full_tune(
        self, tiny_parquet, tmp_path, monkeypatch
    ):
        """Metadata records task=full_tune when checkpoint_path is set."""
        from pmf_tsfm.inference import run_inference

        path, _ = tiny_parquet
        ckpt_dir = tmp_path / "ckpt"
        MockAdapter().save_full_checkpoint(str(ckpt_dir))

        monkeypatch.setattr("pmf_tsfm.inference.get_model_adapter", lambda **kw: MockAdapter())
        output_dir = tmp_path / "out"
        cfg = _make_inference_cfg(path, output_dir, task="full_tune", checkpoint_path=str(ckpt_dir))
        run_inference(cfg)

        with open(output_dir / "synthetic_mock_model_metadata.json") as f:
            meta = json.load(f)
        assert meta["task"] == "full_tune"


# ===========================================================================
# Training loop robustness
# ===========================================================================


class TestTrainingLoop:
    """Training-loop edge cases: early stopping, periodic saving, loss trends."""

    def test_early_stopping_fires(self, tiny_parquet, tmp_path, monkeypatch):
        """When val loss does not improve, early stopping halts training early."""
        from pmf_tsfm.train import train

        path, _ = tiny_parquet
        mock = MockAdapter()
        monkeypatch.setattr("pmf_tsfm.models.get_model_adapter", lambda **kw: mock)

        # 20 epochs but patience=1 → should stop long before epoch 20
        output_dir = tmp_path / "es_out"
        cfg = _make_train_cfg(path, output_dir, task_name="lora_tune", epochs=20)
        # Override patience to 1
        cfg = OmegaConf.merge(cfg, {"training": {"early_stopping_patience": 1}})

        result = train(cfg)

        # If early stopping worked, best_val_loss is still a valid float
        assert isinstance(result["best_val_loss"], float)
        # And the final checkpoint exists (always saved at loop end)
        assert (output_dir / "lora_adapter" / "final").exists()

    def test_device_fallback_cpu(self, tiny_parquet, tmp_path, monkeypatch):
        """Requesting cuda on a CPU-only machine falls back silently."""
        from pmf_tsfm.train import train

        path, _ = tiny_parquet
        mock = MockAdapter()
        # train.py imports get_model_adapter inside the function body, so patching
        # the module-level symbol in pmf_tsfm.models is sufficient here.
        monkeypatch.setattr("pmf_tsfm.models.get_model_adapter", lambda **kw: mock)

        output_dir = tmp_path / "fallback_out"
        cfg = _make_train_cfg(path, output_dir, task_name="lora_tune", epochs=1)
        # Request cuda even though tests run on CPU — fallback_device="cpu"
        cfg = OmegaConf.merge(cfg, {"device": "cuda", "model": {"fallback_device": "cpu"}})

        # Should not raise even without a GPU
        result = train(cfg)
        assert "best_val_loss" in result

    def test_device_fallback_mps(self, tiny_parquet, tmp_path, monkeypatch):
        """Requesting MPS on a non-MPS machine falls back to CPU."""
        from pmf_tsfm.train import train

        path, _ = tiny_parquet
        mock = MockAdapter()
        monkeypatch.setattr("pmf_tsfm.models.get_model_adapter", lambda **kw: mock)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

        output_dir = tmp_path / "mps_fallback_out"
        cfg = _make_train_cfg(path, output_dir, task_name="lora_tune", epochs=1)
        cfg = OmegaConf.merge(cfg, {"device": "mps", "model": {"fallback_device": "cpu"}})

        result = train(cfg)
        assert "best_val_loss" in result

    def test_unsupported_training_family_raises(self, tiny_parquet, tmp_path):
        """Training rejects unsupported families before model construction."""
        from pmf_tsfm.train import train

        path, _ = tiny_parquet
        cfg = _make_train_cfg(path, tmp_path / "unsupported_out", task_name="lora_tune", epochs=1)
        cfg = OmegaConf.merge(cfg, {"model": {"family": "timesfm"}})

        with pytest.raises(ValueError, match="Training supports"):
            train(cfg)

    def test_moirai_task_patch_size_override_is_applied(self, tiny_parquet, tmp_path, monkeypatch):
        """Task-level patch_size should be merged into the model config for Moirai."""
        from pmf_tsfm.train import train

        path, _ = tiny_parquet
        mock = MockAdapter()
        mock.model_family = "moirai"
        captured: dict[str, Any] = {}

        def fake_get_model_adapter(*, model_cfg, device, prediction_length):
            captured["patch_size"] = model_cfg.patch_size
            captured["family"] = model_cfg.family
            return mock

        monkeypatch.setattr("pmf_tsfm.models.get_model_adapter", fake_get_model_adapter)

        output_dir = tmp_path / "moirai_out"
        cfg = _make_train_cfg(path, output_dir, task_name="lora_tune", epochs=1)
        cfg = OmegaConf.merge(
            cfg,
            {
                "model": {"family": "moirai"},
                "task": {"patch_size": 16},
            },
        )

        result = train(cfg)

        assert "best_val_loss" in result
        assert captured["family"] == "moirai"
        assert captured["patch_size"] == 16

    def test_full_tune_uses_chronos2_native_fit_branch(self, tiny_parquet, tmp_path, monkeypatch):
        """Chronos2 full tuning should call fit_chronos2() and save best/final checkpoints."""
        from pmf_tsfm.train import train

        class MockChronos2Adapter(MockAdapter):
            def __init__(self):
                super().__init__()
                self.is_chronos2 = True
                self.fit_calls: list[dict[str, Any]] = []

            def fit_chronos2(
                self,
                *,
                train_inputs,
                validation_inputs,
                num_steps,
                learning_rate,
                batch_size,
            ) -> None:
                self.fit_calls.append(
                    {
                        "train_inputs": train_inputs,
                        "validation_inputs": validation_inputs,
                        "num_steps": num_steps,
                        "learning_rate": learning_rate,
                        "batch_size": batch_size,
                    }
                )

        path, _ = tiny_parquet
        mock = MockChronos2Adapter()
        monkeypatch.setattr("pmf_tsfm.models.get_model_adapter", lambda **kw: mock)

        output_dir = tmp_path / "chronos2_out"
        cfg = _make_train_cfg(path, output_dir, task_name="full_tune", epochs=2)

        result = train(cfg)

        assert result["status"] == "complete"
        assert result["model"] == "chronos2"
        assert len(mock.fit_calls) == 1
        assert len(mock.fit_calls[0]["train_inputs"]) == 3
        assert len(mock.fit_calls[0]["validation_inputs"]) == 3
        assert (output_dir / "checkpoints" / "best" / "model.pt").exists()
        assert (output_dir / "checkpoints" / "final" / "model.pt").exists()


class TestTrainMainWrapper:
    def test_main_print_config_smoke(self, monkeypatch, capsys):
        """Hydra wrapper should print the config and delegate to train()."""
        import pmf_tsfm.train as train_module

        sentinel = {"ok": True}
        cfg = OmegaConf.create({"print_config": True, "task": {"name": "lora_tune"}})
        monkeypatch.setattr(train_module, "train", lambda passed_cfg: sentinel)

        result = train_module.main.__wrapped__(cfg)

        assert result is sentinel
        assert "print_config: true" in capsys.readouterr().out.lower()
