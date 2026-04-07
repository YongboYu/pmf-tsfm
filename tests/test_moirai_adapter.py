"""Offline unit tests for the Moirai adapter."""

import sys
import types
from typing import ClassVar

import numpy as np
import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from pmf_tsfm.models.moirai import MoiraiAdapter


def _make_adapter(*, variant: str = "1_1_small", patch_size: int = 32) -> MoiraiAdapter:
    return MoiraiAdapter(
        model_name="moirai",
        model_id="Salesforce/moirai",
        variant=variant,
        device="cpu",
        prediction_length=2,
        num_samples=4,
        patch_size=patch_size,
    )


def _install_fake_uni2ts(monkeypatch) -> None:
    fake_uni2ts = types.ModuleType("uni2ts")
    fake_model_pkg = types.ModuleType("uni2ts.model")

    def _module_class(name: str):
        class FakeModule:
            loaded_from: ClassVar[list[str]] = []

            @classmethod
            def from_pretrained(cls, model_id: str):
                cls.loaded_from.append(model_id)
                return types.SimpleNamespace(kind=name, model_id=model_id)

        return FakeModule

    def _forecast_class(name: str):
        class FakeForecast:
            def __init__(self, **kwargs) -> None:
                self.kind = name
                self.kwargs = kwargs

            def create_predictor(self, batch_size: int):
                return types.SimpleNamespace(batch_size=batch_size)

        return FakeForecast

    fake_moirai = types.ModuleType("uni2ts.model.moirai")
    fake_moirai.MoiraiModule = _module_class("moirai-1.1")
    fake_moirai.MoiraiForecast = _forecast_class("forecast-1.1")

    fake_moirai2 = types.ModuleType("uni2ts.model.moirai2")
    fake_moirai2.Moirai2Module = _module_class("moirai-2.0")
    fake_moirai2.Moirai2Forecast = _forecast_class("forecast-2.0")

    fake_moe = types.ModuleType("uni2ts.model.moirai_moe")
    fake_moe.MoiraiMoEModule = _module_class("moirai-moe")
    fake_moe.MoiraiMoEForecast = _forecast_class("forecast-moe")

    monkeypatch.setitem(sys.modules, "uni2ts", fake_uni2ts)
    monkeypatch.setitem(sys.modules, "uni2ts.model", fake_model_pkg)
    monkeypatch.setitem(sys.modules, "uni2ts.model.moirai", fake_moirai)
    monkeypatch.setitem(sys.modules, "uni2ts.model.moirai2", fake_moirai2)
    monkeypatch.setitem(sys.modules, "uni2ts.model.moirai_moe", fake_moe)


def _install_fake_gluonts(monkeypatch) -> None:
    fake_gluonts = types.ModuleType("gluonts")
    fake_dataset = types.ModuleType("gluonts.dataset")
    fake_common = types.ModuleType("gluonts.dataset.common")
    fake_common.ListDataset = lambda data, freq: list(data)
    monkeypatch.setitem(sys.modules, "gluonts", fake_gluonts)
    monkeypatch.setitem(sys.modules, "gluonts.dataset", fake_dataset)
    monkeypatch.setitem(sys.modules, "gluonts.dataset.common", fake_common)


def _install_fake_peft(monkeypatch, merged_model) -> None:
    class FakePeftModel:
        @staticmethod
        def from_pretrained(base_model, adapter_path: str):
            return types.SimpleNamespace(
                merge_and_unload=lambda: merged_model,
            )

    fake_peft = types.ModuleType("peft")
    fake_peft.PeftModel = FakePeftModel
    monkeypatch.setitem(sys.modules, "peft", fake_peft)


class DummyMoiraiTrainModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.last_device: str | None = None

    def _get_distr(self, *, patch_size, past_target, past_observed_target, past_is_pad):
        return types.SimpleNamespace(mean=torch.ones(2, 2, 1))

    def _format_preds(self, patch_size, preds_mean, target_dim):
        return torch.ones(1, 2, 2)

    def to(self, device: str):
        self.last_device = device
        return self


class DummyStateDictForecast:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.module = "state-dict-base"
        self.loaded_state = None

    def load_state_dict(self, state_dict) -> None:
        self.loaded_state = state_dict

    def state_dict(self):
        return {"weight": torch.tensor([1.0])}


class TestMoiraiAdapter:
    def test_from_config_and_version_detection(self) -> None:
        cfg = OmegaConf.create(
            {
                "name": "Moirai",
                "id": "Salesforce/moirai-2.0",
                "variant": "2_0_small",
                "num_samples": 5,
                "patch_size": 24,
            }
        )
        adapter = MoiraiAdapter.from_config(cfg, device="cpu", prediction_length=3)

        assert adapter.model_version == "2.0"
        assert adapter.num_samples == 5
        assert adapter.patch_size == 24

        moe = _make_adapter(variant="moe_base", patch_size=32)
        assert moe.model_version == "moe"
        assert moe.patch_size == 16

    @pytest.mark.parametrize(
        ("variant", "expected_version", "expected_kind"),
        [
            ("1_1_small", "1.1", "forecast-1.1"),
            ("2_0_small", "2.0", "forecast-2.0"),
            ("moe_base", "moe", "forecast-moe"),
        ],
    )
    def test_load_model_and_create_sequence_model(
        self, monkeypatch, variant: str, expected_version: str, expected_kind: str
    ) -> None:
        _install_fake_uni2ts(monkeypatch)
        adapter = _make_adapter(variant=variant, patch_size=32)

        adapter.load_model()
        sequence_model = adapter._create_sequence_model(12)

        assert adapter.model_version == expected_version
        assert adapter.is_loaded
        assert sequence_model.kind == expected_kind
        assert sequence_model.kwargs["context_length"] == 12
        assert sequence_model.kwargs["target_dim"] == 1
        if expected_version == "1.1":
            assert "patch_size" not in sequence_model.kwargs
            assert "num_samples" not in sequence_model.kwargs
        if expected_version == "moe":
            assert sequence_model.kwargs["patch_size"] == 16
            assert sequence_model.kwargs["num_samples"] == 4

    def test_create_training_models_require_moirai_1_1(self, monkeypatch) -> None:
        _install_fake_uni2ts(monkeypatch)
        adapter = _make_adapter(variant="2_0_small")

        with pytest.raises(NotImplementedError, match="LoRA fine-tuning only supported"):
            adapter._create_base_model_for_lora(10)

        with pytest.raises(NotImplementedError, match="Full fine-tuning only supported"):
            adapter._create_base_model_for_training(10)

    def test_forecast_sequence_handles_quantile_mean_and_feature_fallback(
        self, monkeypatch
    ) -> None:
        _install_fake_gluonts(monkeypatch)
        adapter = _make_adapter()

        class QuantileForecast:
            def quantile(self, level: float):
                return {
                    0.1: np.asarray([0.5, 0.6]),
                    0.5: np.asarray([1.0, 1.1]),
                    0.9: np.asarray([1.5, 1.6]),
                }[level]

        class MeanForecast:
            mean = np.asarray([3.0, 4.0])

        class FakePredictor:
            def predict(self, dataset):
                return [QuantileForecast(), MeanForecast(), object()]

        adapter._create_sequence_model = lambda context_len: types.SimpleNamespace(
            create_predictor=lambda batch_size: FakePredictor()
        )

        input_seq = np.asarray(
            [
                [1.0, 2.0, 7.0],
                [1.0, 2.0, 8.0],
                [1.0, 2.0, 9.0],
            ],
            dtype=np.float32,
        )
        predictions = np.zeros((2, 3), dtype=np.float32)
        quantiles = np.zeros((2, 3, 3), dtype=np.float32)

        failed = adapter._forecast_sequence(input_seq, 3, 3, 2, predictions, quantiles)

        assert failed == 1
        np.testing.assert_allclose(predictions[:, 0], [1.0, 1.1])
        np.testing.assert_allclose(quantiles[:, 0, 0], [0.5, 0.6])
        np.testing.assert_allclose(predictions[:, 1], [3.0, 4.0])
        np.testing.assert_allclose(quantiles[:, 1, 0], [2.7, 3.6])
        np.testing.assert_allclose(predictions[:, 2], [9.0, 9.0])
        np.testing.assert_allclose(quantiles[:, 2, 0], [8.55, 8.55])

    def test_forecast_sequence_batch_failure_falls_back_all_features(self, monkeypatch) -> None:
        _install_fake_gluonts(monkeypatch)
        adapter = _make_adapter()
        adapter._create_sequence_model = lambda context_len: (_ for _ in ()).throw(
            RuntimeError("oom")
        )

        input_seq = np.asarray([[1.0, 5.0], [2.0, 6.0]], dtype=np.float32)
        predictions = np.zeros((2, 2), dtype=np.float32)
        quantiles = np.zeros((2, 2, 3), dtype=np.float32)

        failed = adapter._forecast_sequence(input_seq, 2, 2, 2, predictions, quantiles)

        assert failed == 2
        np.testing.assert_allclose(predictions[:, 0], [2.0, 2.0])
        np.testing.assert_allclose(predictions[:, 1], [6.0, 6.0])

    def test_predict_runs_sequence_loop_and_clears_cuda_cache(self, monkeypatch) -> None:
        adapter = _make_adapter()
        adapter._is_loaded = True
        empty_cache_calls: list[str] = []

        def fake_forecast_sequence(input_seq, context_len, num_features, pred_len, preds, quants):
            preds[:] = np.arange(pred_len)[:, None]
            quants[:] = 1.0
            return 1

        monkeypatch.setattr(adapter, "_forecast_sequence", fake_forecast_sequence)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "empty_cache", lambda: empty_cache_calls.append("called"))

        prepared = {
            "inputs": [
                np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
                np.asarray([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32),
            ],
            "feature_names": ["a", "b"],
        }

        predictions, quantiles = adapter.predict(prepared)

        assert predictions.shape == (2, 2, 2)
        assert quantiles.shape == (2, 2, 2, 3)
        assert len(empty_cache_calls) == 2

    def test_load_lora_adapter_merges_into_base_module(self, monkeypatch, tmp_path) -> None:
        merged_model = types.SimpleNamespace(module="merged-base")
        _install_fake_peft(monkeypatch, merged_model)

        adapter = _make_adapter()
        adapter._is_loaded = True
        adapter.base_module = "original-base"
        monkeypatch.setattr(
            adapter, "_create_base_model_for_lora", lambda context_length: "base-model"
        )

        adapter.load_lora_adapter(str(tmp_path / "adapter"), context_length=9)

        assert adapter._lora_applied
        assert adapter._lora_adapter_path == str(tmp_path / "adapter")
        assert adapter.base_module == "merged-base"

    def test_forward_train_forward_train_full_and_to(self) -> None:
        adapter = _make_adapter()
        batch = {
            "past_target": torch.ones(1, 2, 1),
            "past_observed_target": torch.ones(1, 2, 1),
            "past_is_pad": torch.zeros(1, 2, 1),
            "future_target": torch.ones(1, 2, 1),
        }

        adapter._peft_model = DummyMoiraiTrainModel()
        predictions, loss = adapter.forward_train(batch)
        assert predictions.shape == (1, 2, 1)
        assert float(loss) == pytest.approx(0.0)

        adapter._full_tune_enabled = True
        adapter._full_tune_model = DummyMoiraiTrainModel()
        predictions, loss = adapter.forward_train_full(batch)
        assert predictions.shape == (1, 2, 1)
        assert float(loss) == pytest.approx(0.0)

        adapter.to("cpu")
        assert adapter._peft_model.last_device == "cpu"
        assert adapter._full_tune_model.last_device == "cpu"

    def test_save_and_load_full_checkpoint_paths(self, monkeypatch, tmp_path) -> None:
        _install_fake_uni2ts(monkeypatch)

        # HF save path
        saved_paths: list[str] = []
        adapter = _make_adapter()
        adapter.base_module = types.SimpleNamespace(
            save_pretrained=lambda output_dir: saved_paths.append(output_dir)
        )
        adapter._full_tune_enabled = True
        adapter._full_tune_model = DummyStateDictForecast()

        adapter.save_full_checkpoint(str(tmp_path / "hf_ckpt"))
        assert saved_paths == [str(tmp_path / "hf_ckpt")]

        # HF load path
        hf_dir = tmp_path / "hf_load"
        hf_dir.mkdir()
        (hf_dir / "config.json").write_text("{}")
        adapter = _make_adapter()
        adapter._is_loaded = True
        adapter.load_full_checkpoint(str(hf_dir))
        assert adapter.base_module.kind == "moirai-1.1"
        assert adapter._full_tune_enabled

        # State-dict save path
        state_dir = tmp_path / "state_ckpt"
        state_adapter = _make_adapter()
        state_adapter.base_module = object()
        state_adapter._full_tune_enabled = True
        state_adapter._full_tune_model = DummyStateDictForecast()
        state_adapter.save_full_checkpoint(str(state_dir))
        assert (state_dir / "model.pt").exists()

        # State-dict load path
        fake_moirai = sys.modules["uni2ts.model.moirai"]
        fake_moirai.MoiraiForecast = DummyStateDictForecast

        fresh = _make_adapter()
        fresh._is_loaded = True
        fresh.base_module = "pretrained-base"
        fresh.load_full_checkpoint(str(state_dir), context_length=7)

        assert fresh._full_tune_enabled
        assert fresh.base_module == "state-dict-base"
