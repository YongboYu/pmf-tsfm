"""Offline unit tests for the TimesFM adapter."""

import importlib
import sys
import types
from typing import ClassVar

import numpy as np
import pytest
from omegaconf import OmegaConf

from pmf_tsfm.models.timesfm import TimesFMAdapter


def _make_adapter(
    *,
    variant: str = "2_5_200m",
    device: str = "cpu",
    forecast_kwargs: dict | None = None,
    legacy_src_path: str | None = None,
    legacy_hparams_overrides: dict | None = None,
) -> TimesFMAdapter:
    return TimesFMAdapter(
        model_name="timesfm",
        model_id="google/timesfm",
        variant=variant,
        device=device,
        prediction_length=2,
        freq="D",
        quantile_levels=[0.1, 0.5, 0.9],
        forecast_kwargs=forecast_kwargs,
        legacy_src_path=legacy_src_path,
        legacy_hparams_overrides=legacy_hparams_overrides,
    )


class FakeForecastConfig:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


class FakeV25Model:
    _from_pretrained_calls: ClassVar[list[dict]] = []

    def __init__(self) -> None:
        self.compiled_with = None
        self.forecast_calls: list[dict] = []

    @classmethod
    def _from_pretrained(cls, *, model_id=None, **kwargs):
        cls._from_pretrained_calls.append(kwargs)
        instance = cls()
        instance.model_id = model_id
        return instance

    @classmethod
    def from_pretrained(cls, model_id: str):
        return cls._from_pretrained(
            model_id=model_id,
            proxies={"https": "proxy"},
            resume_download=True,
            revision="main",
        )

    def compile(self, forecast_config) -> None:
        self.compiled_with = forecast_config

    def forecast(self, *, horizon: int, inputs: list[np.ndarray]):
        self.forecast_calls.append({"horizon": horizon, "inputs": inputs})
        num_features = len(inputs)
        point = np.asarray(
            [[float(idx + step + 1) for step in range(horizon)] for idx in range(num_features)],
            dtype=np.float32,
        )
        quantiles = np.stack([point - 1.0, point, point + 1.0, point + 2.0], axis=-1)
        return point, quantiles


def _make_fake_timesfm_module(*, include_v25_class: bool = True):
    fake_timesfm = types.ModuleType("timesfm")
    fake_timesfm.ForecastConfig = FakeForecastConfig
    fake_timesfm.__version__ = "2.1.0"
    if include_v25_class:
        fake_timesfm.TimesFM_2p5_200M_torch = FakeV25Model
    return fake_timesfm


def _make_fake_legacy_module(*, freq_map=None):
    class FakeTimesFmHparams:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    class FakeTimesFmCheckpoint:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    class FakeTimesFm:
        def __init__(self, *, hparams, checkpoint) -> None:
            self.hparams = hparams
            self.checkpoint = checkpoint
            self.forecast_calls: list[dict] = []

        def forecast(self, *, inputs, freq, forecast_context_len, normalize):
            self.forecast_calls.append(
                {
                    "inputs": inputs,
                    "freq": freq,
                    "forecast_context_len": forecast_context_len,
                    "normalize": normalize,
                }
            )
            num_features = len(inputs)
            horizon = 2
            point = np.asarray(
                [
                    [float(idx + step + 10) for step in range(horizon)]
                    for idx in range(num_features)
                ],
                dtype=np.float32,
            )
            quantiles = np.stack([point - 1.0, point, point + 1.0, point + 2.0], axis=-1)
            return point, quantiles

    fake_legacy = types.ModuleType("timesfm")
    fake_legacy.TimesFm = FakeTimesFm
    fake_legacy.TimesFmHparams = FakeTimesFmHparams
    fake_legacy.TimesFmCheckpoint = FakeTimesFmCheckpoint
    fake_legacy.freq_map = freq_map if freq_map is not None else {"D": 0}
    fake_legacy.__version__ = "1.3.0"
    return fake_legacy


class TestTimesFMAdapter:
    def test_from_config_merges_forecast_overrides_and_backend(self) -> None:
        cfg = OmegaConf.create(
            {
                "name": "TimesFM",
                "id": "google/timesfm-2.5",
                "variant": "2_5_200m",
                "forecast_config": {
                    "max_context": 256,
                    "normalize_inputs": False,
                },
            }
        )

        adapter = TimesFMAdapter.from_config(cfg, device="cpu", prediction_length=4)

        assert adapter._backend == "v2_5"
        assert adapter.max_context == 256
        assert adapter.forecast_kwargs["normalize_inputs"] is False

    def test_load_model_delegates_by_backend(self, monkeypatch) -> None:
        calls: list[str] = []
        v25 = _make_adapter(variant="2_5_200m")
        legacy = _make_adapter(variant="1_0_small")

        monkeypatch.setattr(v25, "_load_v2p5_model", lambda: calls.append("v25"))
        monkeypatch.setattr(legacy, "_load_legacy_model", lambda: calls.append("legacy"))

        v25.load_model()
        legacy.load_model()

        assert calls == ["v25", "legacy"]
        assert v25.is_loaded and legacy.is_loaded

    def test_load_v25_model_compiles_and_strips_transport_kwargs(self, monkeypatch) -> None:
        fake_timesfm = _make_fake_timesfm_module()
        original_descriptor = FakeV25Model.__dict__["_from_pretrained"]

        monkeypatch.setattr(importlib, "import_module", lambda name: fake_timesfm)
        monkeypatch.setattr(importlib.metadata, "version", lambda name: "2.1.0")

        adapter = _make_adapter(
            forecast_kwargs={
                "max_context": 8,
                "max_horizon": 4,
                "normalize_inputs": True,
            }
        )
        adapter._load_v2p5_model()

        assert isinstance(adapter.model, FakeV25Model)
        assert adapter.model.compiled_with.kwargs["max_context"] == 8
        assert FakeV25Model._from_pretrained_calls[-1] == {"revision": "main"}
        assert FakeV25Model.__dict__["_from_pretrained"] is original_descriptor
        assert adapter._quantile_dim == 3

    def test_load_v25_model_validates_version_and_variant(self, monkeypatch) -> None:
        monkeypatch.setattr(importlib, "import_module", lambda name: _make_fake_timesfm_module())
        monkeypatch.setattr(importlib.metadata, "version", lambda name: "1.9.0")

        old = _make_adapter()
        with pytest.raises(ImportError, match=r"requires timesfm>=2\.0\.0"):
            old._load_v2p5_model()

        monkeypatch.setattr(importlib.metadata, "version", lambda name: "2.1.0")
        unknown = _make_adapter(variant="2_5_unknown")
        with pytest.raises(ValueError, match=r"Unknown TimesFM 2\.5 variant"):
            unknown._load_v2p5_model()

        monkeypatch.setattr(
            importlib,
            "import_module",
            lambda name: _make_fake_timesfm_module(include_v25_class=False),
        )
        missing_class = _make_adapter()
        with pytest.raises(RuntimeError, match="not exported by timesfm"):
            missing_class._load_v2p5_model()

    def test_import_legacy_module_uses_source_tree_and_cleanup(self, monkeypatch, tmp_path) -> None:
        legacy_repo = tmp_path / "timesfm-v1"
        (legacy_repo / "src" / "timesfm").mkdir(parents=True)
        fake_legacy = _make_fake_legacy_module()

        monkeypatch.setattr(importlib, "import_module", lambda name: fake_legacy)

        adapter = _make_adapter(variant="1_0_small", legacy_src_path=str(legacy_repo))
        module = adapter._import_legacy_module()

        assert module is fake_legacy
        assert adapter._legacy_path_added
        assert adapter._active_legacy_path == str(legacy_repo / "src")
        assert sys.path[0] == str(legacy_repo / "src")

        adapter._remove_legacy_path_if_needed()
        assert not adapter._legacy_path_added
        assert adapter._active_legacy_path is None
        assert str(legacy_repo / "src") not in sys.path

    def test_import_legacy_module_rejects_modern_package(self, monkeypatch) -> None:
        fake_modern = _make_fake_legacy_module()
        fake_modern.__version__ = "2.1.0"

        monkeypatch.setattr(importlib, "import_module", lambda name: fake_modern)
        monkeypatch.setattr(importlib.metadata, "version", lambda name: "2.1.0")

        adapter = _make_adapter(variant="1_0_small")
        with pytest.raises(ImportError, match=r"Legacy TimesFM 1\.x/2\.0 models require"):
            adapter._import_legacy_module()

    def test_load_legacy_model_builds_hparams_and_checkpoint(self, monkeypatch) -> None:
        fake_legacy = _make_fake_legacy_module()
        adapter = _make_adapter(
            variant="1_0_small",
            device="cuda",
            legacy_hparams_overrides={"per_core_batch_size": 3},
        )
        monkeypatch.setattr(adapter, "_import_legacy_module", lambda: fake_legacy)

        adapter._load_legacy_model()

        assert adapter._quantile_dim == 3
        assert adapter.model.hparams.kwargs["backend"] == "gpu"
        assert adapter.model.hparams.kwargs["per_core_batch_size"] == 3
        assert adapter.model.checkpoint.kwargs["huggingface_repo_id"] == "google/timesfm"

    def test_predict_empty_inputs_and_validation_errors(self) -> None:
        adapter = _make_adapter()
        adapter._is_loaded = True

        with pytest.raises(ValueError, match="No input sequences or feature names"):
            adapter.predict({"inputs": [], "feature_names": []})

        predictions, quantiles = adapter.predict({"inputs": [], "feature_names": ["a", "b"]})
        assert predictions.shape == (0, 2, 2)
        assert quantiles.shape == (0, 2, 2, 3)

        adapter.max_horizon = 1
        with pytest.raises(ValueError, match="exceeds max_horizon"):
            adapter.predict({"inputs": [np.ones((3, 2))], "feature_names": ["a", "b"]})

    def test_predict_v25_transposes_outputs_and_trims_context(self) -> None:
        adapter = _make_adapter(
            forecast_kwargs={"max_context": 2, "max_horizon": 4, "normalize_inputs": True}
        )
        adapter._is_loaded = True
        adapter.model = FakeV25Model()
        adapter._quantile_dim = 3

        prepared = {
            "inputs": [np.asarray([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]], dtype=np.float32)],
            "feature_names": ["a", "b"],
        }

        predictions, quantiles = adapter.predict(prepared)

        assert predictions.shape == (1, 2, 2)
        assert quantiles.shape == (1, 2, 2, 3)
        np.testing.assert_allclose(predictions[0, :, 0], [1.0, 2.0])
        np.testing.assert_allclose(predictions[0, :, 1], [2.0, 3.0])
        np.testing.assert_allclose(quantiles[0, :, 0, 0], [1.0, 2.0])
        trimmed_inputs = adapter.model.forecast_calls[0]["inputs"]
        assert all(series.shape[0] == 2 for series in trimmed_inputs)

    def test_forecast_batch_legacy_and_helper_branches(self) -> None:
        fake_legacy = _make_fake_legacy_module(freq_map=lambda freq: 7)
        adapter = _make_adapter(
            variant="1_0_small",
            forecast_kwargs={"max_context": 3, "max_horizon": 4, "normalize_inputs": False},
        )
        adapter._backend = "legacy"
        adapter.model = fake_legacy.TimesFm(
            hparams=fake_legacy.TimesFmHparams(),
            checkpoint=fake_legacy.TimesFmCheckpoint(),
        )
        adapter._legacy_module = fake_legacy

        point, quantiles = adapter._forecast_batch(
            [
                np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
                np.asarray([10.0, 20.0, 30.0, 40.0], dtype=np.float32),
            ],
            prediction_length=2,
        )

        np.testing.assert_allclose(point[:, 0], [10.0, 11.0])
        assert quantiles.shape == (2, 2, 3)
        assert adapter.model.forecast_calls[0]["freq"] == [7, 7]
        assert adapter.model.forecast_calls[0]["forecast_context_len"] == 3

        split = adapter._split_features(np.asarray([[1.0, 2.0], [3.0, 4.0]]), 2)
        assert len(split) == 2
        with pytest.raises(ValueError, match="Expected input sequence"):
            adapter._split_features(np.asarray([1.0, 2.0]), 2)

        adapter._legacy_module.freq_map = {"D": 5}
        assert adapter._legacy_freq_code() == 5
        adapter._legacy_module.freq_map = {"W": 9}
        with pytest.raises(KeyError, match="missing frequency 'D'"):
            adapter._legacy_freq_code()
        adapter._legacy_module.freq_map = 123
        with pytest.raises(TypeError, match="not callable or a dict"):
            adapter._legacy_freq_code()

        assert adapter._resolve_legacy_src_path() is None
        assert adapter._parse_major_minor("timesfm-2.5.1") == (2, 5)
        assert adapter._parse_major_minor("unknown") is None
        assert adapter._is_version_at_least("2.5.1", 2, 0)

        sys.modules["timesfm.helper"] = types.ModuleType("timesfm.helper")
        sys.modules["timesfm"] = types.ModuleType("timesfm")
        adapter._clear_timesfm_modules()
        assert "timesfm" not in sys.modules
        assert "timesfm.helper" not in sys.modules

    def test_resolve_legacy_src_path_from_file_and_env(self, monkeypatch, tmp_path) -> None:
        repo = tmp_path / "repo"
        script = repo / "src" / "config.txt"
        (repo / "src" / "timesfm").mkdir(parents=True)
        script.write_text("")

        adapter = _make_adapter(variant="1_0_small", legacy_src_path=str(script))
        assert adapter._resolve_legacy_src_path() == repo / "src"

        adapter = _make_adapter(variant="1_0_small")
        monkeypatch.setenv("TIMESFM_V1_PATH", str(repo))
        assert adapter._resolve_legacy_src_path() == repo / "src"
