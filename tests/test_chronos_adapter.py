"""Offline unit tests for the Chronos adapter."""

import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from pmf_tsfm.models.chronos import ChronosAdapter


class DummyPeftModel(nn.Module):
    """Tiny PEFT-like wrapper used to test Chronos LoRA paths."""

    def __init__(self, base_model: nn.Module, *, merged_model=None, config=None) -> None:
        super().__init__()
        self.base_model = base_model
        self.config = config
        self.merged_model = merged_model or base_model
        self.printed = False

    def print_trainable_parameters(self) -> None:
        self.printed = True

    def merge_and_unload(self):
        return self.merged_model

    def to(self, device: str):
        return self


def _make_adapter(*, variant: str = "bolt_small", model_id: str = "amazon/chronos-bolt-small"):
    return ChronosAdapter(
        model_name="chronos",
        model_id=model_id,
        variant=variant,
        device="cpu",
        prediction_length=2,
    )


def _install_fake_chronos(monkeypatch, from_pretrained):
    fake_chronos = types.ModuleType("chronos")
    fake_chronos.BaseChronosPipeline = type(
        "FakeBaseChronosPipeline",
        (),
        {"from_pretrained": staticmethod(from_pretrained)},
    )
    monkeypatch.setitem(sys.modules, "chronos", fake_chronos)


def _install_fake_peft(monkeypatch, *, get_peft_model=None, peft_from_pretrained=None) -> None:
    class FakeLoraConfig:
        def __init__(self, *, r, lora_alpha, lora_dropout, target_modules, bias) -> None:
            self.r = r
            self.lora_alpha = lora_alpha
            self.lora_dropout = lora_dropout
            self.target_modules = list(target_modules)
            self.bias = bias

    class FakePeftModel:
        @staticmethod
        def from_pretrained(base_model, adapter_path: str):
            if peft_from_pretrained is not None:
                return peft_from_pretrained(base_model, adapter_path)
            return DummyPeftModel(base_model)

    fake_peft = types.ModuleType("peft")
    fake_peft.LoraConfig = FakeLoraConfig
    fake_peft.PeftModel = FakePeftModel
    fake_peft.get_peft_model = get_peft_model
    monkeypatch.setitem(sys.modules, "peft", fake_peft)


class DummyChronosOutputs:
    def __init__(self, *, loss: float, quantile_preds=None) -> None:
        self.loss = torch.tensor(loss)
        self.quantile_preds = quantile_preds


class DummyChronosTrainModel:
    def __init__(self, *, should_fail: bool = False) -> None:
        self.should_fail = should_fail

    def __call__(self, *, context, target):
        if self.should_fail:
            raise ValueError("bad batch")
        return DummyChronosOutputs(loss=1.5, quantile_preds=context + target)

    def to(self, device: str):
        return self


class DummySaveModel:
    def __init__(self) -> None:
        self.saved_to: str | None = None

    def save_pretrained(self, output_dir: str) -> None:
        self.saved_to = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / "config.json").write_text("{}")


class TestChronosAdapter:
    def test_from_config_and_detect_chronos2(self) -> None:
        bolt_cfg = OmegaConf.create(
            {
                "name": "Chronos",
                "id": "amazon/chronos-bolt-small",
                "variant": "bolt_small",
                "torch_dtype": "float16",
                "quantile_levels": [0.2, 0.5, 0.8],
            }
        )
        bolt = ChronosAdapter.from_config(bolt_cfg, device="cpu", prediction_length=3)
        assert bolt.quantile_levels == [0.2, 0.5, 0.8]
        assert not bolt.is_chronos2

        chronos2 = _make_adapter(variant="chronos_2", model_id="s3://autogluon/chronos-2")
        assert chronos2.is_chronos2

    @pytest.mark.parametrize(
        ("variant", "model_id", "expected_dtype"),
        [
            ("bolt_small", "amazon/chronos-bolt-small", torch.float32),
            ("chronos_2", "s3://autogluon/chronos-2", None),
        ],
    )
    def test_load_model_uses_expected_kwargs(
        self, monkeypatch, variant: str, model_id: str, expected_dtype
    ) -> None:
        calls: list[dict] = []
        pipeline = types.SimpleNamespace(model=DummySaveModel())

        def fake_from_pretrained(model_name: str, **kwargs):
            calls.append({"model_name": model_name, **kwargs})
            return pipeline

        _install_fake_chronos(monkeypatch, fake_from_pretrained)

        adapter = _make_adapter(variant=variant, model_id=model_id)
        adapter.load_model()

        assert adapter.is_loaded
        assert adapter.pipeline is pipeline
        assert calls[0]["model_name"] == model_id
        assert calls[0]["device_map"] == "cpu"
        if expected_dtype is None:
            assert "torch_dtype" not in calls[0]
        else:
            assert calls[0]["torch_dtype"] == expected_dtype

    def test_load_model_wraps_import_errors(self, monkeypatch) -> None:
        def fake_from_pretrained(model_name: str, **kwargs):
            raise ImportError("missing chronos")

        _install_fake_chronos(monkeypatch, fake_from_pretrained)
        adapter = _make_adapter()

        with pytest.raises(ImportError, match="chronos package not found"):
            adapter.load_model()

    def test_predict_bolt_success_and_last_value_fallback(self) -> None:
        class FakeBoltPipeline:
            def predict_quantiles(self, *, inputs, prediction_length, quantile_levels):
                if float(inputs[0, -1]) == 4.0:
                    raise RuntimeError("force fallback")
                quantiles = torch.full((prediction_length, len(quantile_levels)), 2.5)
                mean = torch.full((1, prediction_length), 1.5)
                return quantiles, mean

        adapter = _make_adapter()
        adapter.pipeline = FakeBoltPipeline()
        adapter._is_loaded = True
        prepared = {
            "inputs": [np.asarray([[1.0, 3.0], [2.0, 4.0]], dtype=np.float32)],
            "feature_names": ["feat_a", "feat_b"],
        }

        predictions, quantiles = adapter.predict(prepared)

        assert predictions.shape == (1, 2, 2)
        np.testing.assert_allclose(predictions[0, :, 0], [1.5, 1.5])
        np.testing.assert_allclose(predictions[0, :, 1], [4.0, 4.0])
        np.testing.assert_allclose(quantiles[0, :, 0, :], 2.5)
        np.testing.assert_allclose(quantiles[0, :, 1, :], 4.0)

    def test_predict_chronos2_batched_uses_per_feature_and_sequence_fallbacks(self) -> None:
        class FakeChronos2Pipeline:
            def __init__(self) -> None:
                self.calls = 0

            def predict_df(self, df: pd.DataFrame, *, prediction_length, quantile_levels):
                assert list(quantile_levels) == [0.1, 0.5, 0.9]
                self.calls += 1
                if self.calls == 2:
                    raise RuntimeError("bad batch")
                return pd.DataFrame(
                    [
                        {
                            "item_id": "feat_a",
                            "predictions": 10.0,
                            "0.1": 9.0,
                            "0.9": 11.0,
                        },
                        {
                            "item_id": "feat_a",
                            "predictions": 12.0,
                            "0.1": 11.0,
                            "0.9": 13.0,
                        },
                        {
                            "item_id": "feat_b",
                            "predictions": 99.0,
                            "0.1": 98.0,
                            "0.9": 100.0,
                        },
                    ]
                )

        adapter = _make_adapter(variant="chronos_2", model_id="s3://autogluon/chronos-2")
        adapter.pipeline = FakeChronos2Pipeline()
        adapter._is_loaded = True
        prepared = {
            "inputs": [
                np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
                np.asarray([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32),
            ],
            "feature_names": ["feat_a", "feat_b"],
        }

        predictions, quantiles = adapter.predict(prepared)

        np.testing.assert_allclose(predictions[0, :, 0], [10.0, 12.0])
        np.testing.assert_allclose(quantiles[0, :, 0, 0], [9.0, 11.0])
        np.testing.assert_allclose(quantiles[0, :, 0, 2], [11.0, 13.0])
        np.testing.assert_allclose(predictions[0, :, 1], [4.0, 4.0])
        np.testing.assert_allclose(predictions[1, :, :], [[7.0, 8.0], [7.0, 8.0]])

    def test_apply_lora_falls_back_to_default_targets(self, monkeypatch) -> None:
        calls: list[list[str]] = []

        def fake_get_peft_model(base_model, peft_config):
            calls.append(list(peft_config.target_modules))
            if len(calls) == 1:
                raise ValueError("Target modules not found")
            return DummyPeftModel(base_model, config=peft_config)

        _install_fake_peft(monkeypatch, get_peft_model=fake_get_peft_model)

        adapter = _make_adapter()
        adapter._is_loaded = True
        adapter.pipeline = types.SimpleNamespace(inner_model=nn.Linear(2, 2))

        adapter.apply_lora({"target_modules": ["bad_target"]}, context_length=11)

        assert adapter._lora_applied
        assert adapter._context_length == 11
        assert calls == [["bad_target"], ["q", "v"]]
        assert adapter._peft_model.printed

    def test_load_lora_adapter_merges_weights_into_pipeline(self, monkeypatch, tmp_path) -> None:
        merged_model = nn.Linear(2, 2)

        def fake_from_pretrained(base_model, adapter_path: str):
            return DummyPeftModel(base_model, merged_model=merged_model)

        _install_fake_peft(monkeypatch, peft_from_pretrained=fake_from_pretrained)

        adapter = _make_adapter()
        adapter._is_loaded = True
        adapter.pipeline = types.SimpleNamespace(inner_model=nn.Linear(2, 2))

        adapter.load_lora_adapter(str(tmp_path / "adapter"), context_length=7)

        assert adapter._lora_applied
        assert adapter._lora_adapter_path == str(tmp_path / "adapter")
        assert adapter.pipeline.inner_model is merged_model
        assert adapter._peft_model is merged_model

    def test_forward_train_and_forward_train_full_wrap_errors(self) -> None:
        batch = {
            "context": torch.ones(2, 3),
            "target": torch.ones(2, 3),
        }
        adapter = _make_adapter()
        adapter._peft_model = DummyChronosTrainModel()
        quantiles, loss = adapter.forward_train(batch)
        assert quantiles.shape == (2, 3)
        assert float(loss) == pytest.approx(1.5)

        adapter._peft_model = DummyChronosTrainModel(should_fail=True)
        with pytest.raises(RuntimeError, match="Chronos forward pass failed"):
            adapter.forward_train(batch)

        adapter._full_tune_enabled = True
        adapter._full_tune_model = DummyChronosTrainModel()
        quantiles, loss = adapter.forward_train_full(batch)
        assert quantiles.shape == (2, 3)
        assert float(loss) == pytest.approx(1.5)

        adapter._full_tune_model = DummyChronosTrainModel(should_fail=True)
        with pytest.raises(RuntimeError, match="Chronos Bolt forward pass failed"):
            adapter.forward_train_full(batch)

    def test_save_and_load_full_checkpoint_for_bolt(self, monkeypatch, tmp_path) -> None:
        saved_model = DummySaveModel()
        adapter = _make_adapter()
        adapter.pipeline = types.SimpleNamespace(model=saved_model)

        adapter.save_full_checkpoint(str(tmp_path / "checkpoint"))
        assert saved_model.saved_to == str(tmp_path / "checkpoint")

        calls: list[dict] = []
        loaded_pipeline = types.SimpleNamespace(model=DummySaveModel())

        def fake_from_pretrained(model_name: str, **kwargs):
            calls.append({"model_name": model_name, **kwargs})
            return loaded_pipeline

        _install_fake_chronos(monkeypatch, fake_from_pretrained)

        fresh = _make_adapter()
        fresh.load_full_checkpoint(str(tmp_path / "checkpoint"))

        assert fresh.pipeline is loaded_pipeline
        assert fresh._full_tune_enabled
        assert fresh.is_loaded
        assert calls[0]["model_name"] == str(tmp_path / "checkpoint")
        assert calls[0]["torch_dtype"] == torch.float32

    def test_load_full_checkpoint_for_chronos2_and_fit_native(self, monkeypatch, tmp_path) -> None:
        fake_chronos = types.ModuleType("chronos")
        fake_chronos.BaseChronosPipeline = type(
            "FakeBaseChronosPipeline",
            (),
            {"from_pretrained": staticmethod(lambda *args, **kwargs: None)},
        )
        fake_chronos2 = types.ModuleType("chronos.chronos2")

        class FakeArchitecture:
            @classmethod
            def from_pretrained(cls, checkpoint_path: str, device_map: str):
                return {"checkpoint_path": checkpoint_path, "device_map": device_map}

        fake_chronos2.FakeArchitecture = FakeArchitecture
        fake_chronos.chronos2 = fake_chronos2

        class FakeChronos2Pipeline:
            def __init__(self, model) -> None:
                self.model = model
                self.fit_calls: list[dict] = []

            def fit(self, **kwargs):
                self.fit_calls.append(kwargs)
                return types.SimpleNamespace(fitted=True, kwargs=kwargs)

        fake_pipeline_mod = types.ModuleType("chronos.chronos2.pipeline")
        fake_pipeline_mod.Chronos2Pipeline = FakeChronos2Pipeline

        fake_transformers = types.ModuleType("transformers")
        fake_transformers.AutoConfig = type(
            "FakeAutoConfig",
            (),
            {
                "from_pretrained": staticmethod(
                    lambda checkpoint_path: types.SimpleNamespace(
                        architectures=["FakeArchitecture"]
                    )
                )
            },
        )

        monkeypatch.setitem(sys.modules, "chronos", fake_chronos)
        monkeypatch.setitem(sys.modules, "chronos.chronos2", fake_chronos2)
        monkeypatch.setitem(sys.modules, "chronos.chronos2.pipeline", fake_pipeline_mod)
        monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

        adapter = _make_adapter(variant="chronos_2", model_id="s3://autogluon/chronos-2")
        adapter.load_full_checkpoint(str(tmp_path / "chronos2_ckpt"))

        assert adapter._full_tune_enabled
        assert adapter.is_loaded
        assert adapter.pipeline.model["checkpoint_path"] == str(tmp_path / "chronos2_ckpt")

        adapter._is_loaded = True
        train_inputs = [{"target": np.asarray([1.0, 2.0])}]
        validation_inputs = [{"target": np.asarray([2.0, 3.0])}]
        fitted = adapter.fit_chronos2(
            train_inputs=train_inputs,
            validation_inputs=validation_inputs,
            num_steps=4,
            learning_rate=1e-4,
            batch_size=2,
        )

        assert fitted.fitted
        assert adapter.pipeline is fitted

    def test_fit_chronos2_validates_variant_and_loaded_state(self) -> None:
        bolt = _make_adapter()
        with pytest.raises(ValueError, match=r"only for Chronos 2\.0"):
            bolt.fit_chronos2(train_inputs=[])

        chronos2 = _make_adapter(variant="chronos_2", model_id="s3://autogluon/chronos-2")
        with pytest.raises(RuntimeError, match="Model not loaded"):
            chronos2.fit_chronos2(train_inputs=[])
