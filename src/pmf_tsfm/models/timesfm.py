"""TimesFM adapter for zero-shot forecasting."""

from __future__ import annotations

import importlib
import importlib.metadata
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig

from pmf_tsfm.models.base import BaseAdapter

logger = logging.getLogger(__name__)


class TimesFMAdapter(BaseAdapter):
    """
    Adapter that wraps Google's TimesFM models.

    Supports:
        - TimesFM 2.5 PyTorch checkpoints via the public `timesfm` package.
        - Legacy TimesFM 1.x / 2.0 checkpoints via the v1 codebase.

    Both APIs expose a univariate forecasting interface. Like the Chronos and
    Moirai adapters we forecast every feature independently and then stack the
    results back into a multivariate tensor.
    """

    DEFAULT_QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    DEFAULT_FORECAST_KWARGS = {
        "max_context": 1024,
        "max_horizon": 64,
        "normalize_inputs": True,
        "per_core_batch_size": 1,
        "use_continuous_quantile_head": True,
        "force_flip_invariance": True,
        "infer_is_positive": True,
        "fix_quantile_crossing": True,
    }

    V2P5_CLASS_MAP = {
        "2_5_200m": "TimesFM_2p5_200M_torch",
    }

    @classmethod
    def from_config(
        cls,
        model_cfg: DictConfig,
        device: str = "cuda",
        prediction_length: int = 7,
    ) -> TimesFMAdapter:
        """Create adapter from Hydra model config."""
        forecast_kwargs = dict(cls.DEFAULT_FORECAST_KWARGS)
        forecast_overrides = model_cfg.get("forecast_config") or {}
        forecast_kwargs.update(forecast_overrides)

        return cls(
            model_name=model_cfg.name,
            model_id=model_cfg.id,
            variant=model_cfg.variant,
            device=device,
            torch_dtype=model_cfg.get("torch_dtype", "float32"),
            prediction_length=prediction_length,
            freq=model_cfg.get("freq", "D"),
            quantile_levels=list(model_cfg.get("quantile_levels", cls.DEFAULT_QUANTILES)),
            legacy_src_path=model_cfg.get("legacy_src_path"),
            legacy_hparams_overrides=dict(model_cfg.get("legacy_hparams") or {}),
            forecast_kwargs=forecast_kwargs,
        )

    def __init__(
        self,
        model_name: str,
        model_id: str,
        variant: str,
        device: str = "cuda",
        torch_dtype: str = "float32",
        prediction_length: int = 7,
        freq: str = "D",
        quantile_levels: list[float] | None = None,
        legacy_src_path: str | None = None,
        legacy_hparams_overrides: dict[str, Any] | None = None,
        forecast_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        super().__init__(
            model_name=model_name,
            model_id=model_id,
            model_family="timesfm",
            variant=variant,
            device=device,
            torch_dtype=torch_dtype,
            prediction_length=prediction_length,
            **kwargs,
        )
        self.freq = freq
        self.quantile_levels = quantile_levels or self.DEFAULT_QUANTILES
        self.forecast_kwargs = dict(forecast_kwargs or self.DEFAULT_FORECAST_KWARGS)
        self.max_context = self.forecast_kwargs.get("max_context", 1024)
        self.max_horizon = self.forecast_kwargs.get("max_horizon", prediction_length)
        self.forecast_kwargs["max_context"] = self.max_context
        self.forecast_kwargs["max_horizon"] = self.max_horizon
        self.legacy_src_path = legacy_src_path
        self.legacy_hparams_overrides = dict(legacy_hparams_overrides or {})

        self._backend = self._determine_backend()
        self._timesfm_package_version: str | None = None
        self._quantile_dim: int | None = None
        self._legacy_path_added = False
        self._active_legacy_path: str | None = None

        self._legacy_module: Any | None = None

    def _determine_backend(self) -> str:
        variant_lower = self.variant.lower()
        if "2_5" in variant_lower or "2.5" in variant_lower:
            return "v2_5"
        return "legacy"

    def load_model(self) -> None:
        """Load the requested TimesFM model."""
        if self._backend == "v2_5":
            self._load_v2p5_model()
        else:
            self._load_legacy_model()
        self._is_loaded = True

    # ------------------------------------------------------------------ #
    # Backends
    # ------------------------------------------------------------------ #

    def _load_v2p5_model(self) -> None:
        """Load TimesFM 2.5 PyTorch implementation."""
        self._clear_timesfm_modules()
        self._remove_legacy_path_if_needed()

        try:
            timesfm = importlib.import_module("timesfm")
            try:
                self._timesfm_package_version = importlib.metadata.version("timesfm")
            except importlib.metadata.PackageNotFoundError:
                self._timesfm_package_version = getattr(timesfm, "__version__", "unknown")
            logger.info("Loaded timesfm module v%s for TimesFM 2.5", self._timesfm_package_version)
        except ModuleNotFoundError as exc:
            raise ImportError("timesfm>=2.0.0 is required for TimesFM 2.5 models.") from exc
        except Exception as exc:  # pragma: no cover - defensive logging
            raise RuntimeError(f"Failed to import timesfm: {exc}") from exc

        if self._timesfm_package_version:
            parsed = self._parse_major_minor(self._timesfm_package_version)
            if parsed is not None and parsed < (2, 0):
                raise ImportError(
                    "TimesFM 2.5 requires timesfm>=2.0.0, but detected timesfm=="
                    f"{self._timesfm_package_version}. "
                    "Upgrade the installed package or use a dedicated environment for TimesFM 2.5."
                )

        class_name = self.V2P5_CLASS_MAP.get(self.variant.lower())
        if not class_name:
            raise ValueError(
                f"Unknown TimesFM 2.5 variant '{self.variant}'. "
                f"Supported variants: {', '.join(self.V2P5_CLASS_MAP.keys())}"
            )

        try:
            model_cls = getattr(timesfm, class_name)
        except AttributeError as exc:
            raise RuntimeError(
                f"{class_name} is not exported by timesfm=={self._timesfm_package_version}. "
                "Upgrade the package to a newer release."
            ) from exc

        logger.info("Loading TimesFM 2.5 checkpoint '%s' (%s)", self.model_id, class_name)
        self.model = model_cls.from_pretrained(self.model_id)

        forecast_config = timesfm.ForecastConfig(**self.forecast_kwargs)
        assert self.model is not None
        self.model.compile(forecast_config)

        # timesfm v2.5 returns an extra head (typically the median) in addition to quantiles.
        # We normalize outputs to match `quantile_levels`.
        self._quantile_dim = len(self.quantile_levels)

    def _load_legacy_model(self) -> None:
        """Load TimesFM 1.x / 2.0 model via legacy code path."""
        legacy_module = self._import_legacy_module()
        TimesFm = legacy_module.TimesFm
        TimesFmHparams = legacy_module.TimesFmHparams
        TimesFmCheckpoint = legacy_module.TimesFmCheckpoint

        backend = "gpu" if self.device.startswith("cuda") else "cpu"
        hparams_kwargs: dict[str, Any] = {
            "context_len": self.max_context,
            "horizon_len": max(self.max_horizon, self.prediction_length),
            "backend": backend,
            "per_core_batch_size": self.forecast_kwargs.get("per_core_batch_size", 1),
            "quantiles": self.quantile_levels,
        }
        hparams_kwargs.update(self.legacy_hparams_overrides)
        hparams = TimesFmHparams(**hparams_kwargs)
        checkpoint = TimesFmCheckpoint(version="torch", huggingface_repo_id=self.model_id)

        logger.info("Loading TimesFM legacy checkpoint '%s' (%s)", self.model_id, self.variant)
        self.model = TimesFm(hparams=hparams, checkpoint=checkpoint)
        self._quantile_dim = len(self.quantile_levels)

    def _import_legacy_module(self):
        """Import the legacy v1 TimesFM package."""
        if self._legacy_module is not None:
            return self._legacy_module

        try:
            legacy_src_path = self._resolve_legacy_src_path()
            if legacy_src_path is not None:
                self._clear_timesfm_modules()
                legacy_src_str = str(legacy_src_path)
                if legacy_src_str not in sys.path:
                    sys.path.insert(0, legacy_src_str)
                    self._legacy_path_added = True
                    self._active_legacy_path = legacy_src_str

                legacy_module = importlib.import_module("timesfm")
                self._timesfm_package_version = getattr(legacy_module, "__version__", "v1")
                logger.info("Loaded timesfm legacy module from %s", legacy_src_path)
            else:
                legacy_module = importlib.import_module("timesfm")
                try:
                    self._timesfm_package_version = importlib.metadata.version("timesfm")
                except Exception:
                    self._timesfm_package_version = getattr(legacy_module, "__version__", "unknown")

                if self._timesfm_package_version and self._is_version_at_least(
                    self._timesfm_package_version, 2, 0
                ):
                    raise ImportError(
                        "Legacy TimesFM 1.x/2.0 models require either timesfm==1.3.0 "
                        "or the v1 source tree. Detected timesfm=="
                        f"{self._timesfm_package_version}. Set `legacy_src_path`/TIMESFM_V1_PATH "
                        "to the cloned timesfm/v1/src directory, or use a dedicated environment "
                        "with timesfm==1.3.0."
                    )

                missing = [
                    name
                    for name in ("TimesFm", "TimesFmHparams", "TimesFmCheckpoint")
                    if not hasattr(legacy_module, name)
                ]
                if missing:
                    raise ImportError(
                        "Installed timesfm does not expose the legacy API "
                        f"(missing: {', '.join(missing)}). "
                        "Install timesfm==1.3.0 or set `legacy_src_path`/TIMESFM_V1_PATH."
                    )

            missing = [
                name
                for name in ("TimesFm", "TimesFmHparams", "TimesFmCheckpoint")
                if not hasattr(legacy_module, name)
            ]
            if missing:
                raise ImportError(
                    "Imported legacy TimesFM module does not expose the expected API "
                    f"(missing: {', '.join(missing)}). "
                    "Ensure `legacy_src_path`/TIMESFM_V1_PATH points to the v1/src directory "
                    "or install timesfm==1.3.0."
                )
        except ImportError:
            raise
        except Exception as exc:  # pragma: no cover - defensive logging
            raise ImportError("Failed to import legacy TimesFM.") from exc

        self._legacy_module = legacy_module
        return legacy_module

    # ------------------------------------------------------------------ #
    # Prediction
    # ------------------------------------------------------------------ #

    def predict(
        self,
        prepared_data: dict[str, Any],
        prediction_length: int | None = None,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() before predict().")

        pred_len = prediction_length or self.prediction_length
        if pred_len > self.max_horizon:
            raise ValueError(
                f"Requested prediction_length={pred_len} exceeds max_horizon={self.max_horizon}. "
                "Increase `forecast_config.max_horizon` in the model config."
            )

        inputs = prepared_data["inputs"]
        feature_names = prepared_data.get("feature_names") or []
        num_sequences = len(inputs)

        if num_sequences == 0:
            num_features = len(feature_names)
            if num_features == 0:
                raise ValueError(
                    "No input sequences or feature names available for TimesFM inference."
                )
            quantile_dim = self._quantile_dim or len(self.quantile_levels)
            return (
                np.empty((0, pred_len, num_features)),
                np.empty((0, pred_len, num_features, quantile_dim)),
            )

        num_features = len(feature_names) if feature_names else inputs[0].shape[1]

        predictions = np.zeros((num_sequences, pred_len, num_features), dtype=np.float32)
        quantile_dim = self._quantile_dim or len(self.quantile_levels)
        quantiles_out = np.zeros(
            (num_sequences, pred_len, num_features, quantile_dim), dtype=np.float32
        )

        for seq_idx, input_seq in enumerate(inputs):
            feature_series = self._split_features(input_seq, num_features)
            point_forecast, quantile_forecast = self._forecast_batch(feature_series, pred_len)

            predictions[seq_idx] = point_forecast[:, :pred_len].T
            quantiles_out[seq_idx] = np.transpose(quantile_forecast[:, :pred_len, :], (1, 0, 2))

        return predictions, quantiles_out

    def _forecast_batch(
        self,
        feature_series: list[np.ndarray],
        prediction_length: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        trimmed_series = [
            np.asarray(series, dtype=np.float32)[-self.max_context :] for series in feature_series
        ]

        if self._backend == "v2_5":
            assert self.model is not None
            point_forecast, quantile_forecast = self.model.forecast(
                horizon=prediction_length,
                inputs=trimmed_series,
            )
            point_arr = np.asarray(point_forecast)
            quantile_arr = np.asarray(quantile_forecast)
            if quantile_arr.ndim == 3 and quantile_arr.shape[-1] == len(self.quantile_levels) + 1:
                quantile_arr = quantile_arr[:, :, 1:]
            return point_arr, quantile_arr

        # Legacy API
        assert self.model is not None
        freq_code = self._legacy_freq_code()
        freq = [freq_code] * len(trimmed_series)
        point_forecast, quantile_forecast = self.model.forecast(
            inputs=trimmed_series,
            freq=freq,
            forecast_context_len=self.max_context,
            normalize=self.forecast_kwargs.get("normalize_inputs", False),
        )
        # Drop the leading point forecast (TimesFM legacy returns mean + quantiles)
        return np.asarray(point_forecast), np.asarray(quantile_forecast)[:, :, 1:]

    @staticmethod
    def _split_features(input_seq: np.ndarray, num_features: int) -> list[np.ndarray]:
        if input_seq.ndim != 2 or input_seq.shape[1] != num_features:
            raise ValueError(
                f"Expected input sequence with shape (*, {num_features}), got {input_seq.shape}"
            )
        return [input_seq[:, idx] for idx in range(num_features)]

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _legacy_freq_code(self) -> int:
        if self._legacy_module is None:
            self._legacy_module = self._import_legacy_module()
        freq_map = getattr(self._legacy_module, "freq_map", None)
        if freq_map is None:
            raise AttributeError("Legacy TimesFM module is missing freq_map.")
        if callable(freq_map):
            return int(freq_map(self.freq))  # pylint: disable=not-callable
        if isinstance(freq_map, dict):
            try:
                return int(freq_map[self.freq])
            except KeyError as exc:
                raise KeyError(
                    f"Legacy TimesFM freq_map is missing frequency '{self.freq}'."
                ) from exc
        raise TypeError("Legacy TimesFM freq_map is not callable or a dict.")

    def _resolve_legacy_src_path(self) -> Path | None:
        """Locate the timesfm/v1/src folder."""
        candidates: list[Path] = []

        if self.legacy_src_path:
            candidates.append(Path(self.legacy_src_path).expanduser())

        env_path = os.environ.get("TIMESFM_V1_PATH")
        if env_path:
            candidates.append(Path(env_path).expanduser())

        for candidate in candidates:
            if not candidate:
                continue
            src_path = candidate
            if src_path.is_file():
                src_path = src_path.parent
            if (src_path / "timesfm").is_dir():
                return src_path
            if (src_path / "src" / "timesfm").is_dir():
                return src_path / "src"

        return None

    @staticmethod
    def _parse_major_minor(version_str: str) -> tuple[int, int] | None:
        match = re.search(r"(\d+)\.(\d+)", version_str)
        if not match:
            return None
        return int(match.group(1)), int(match.group(2))

    @classmethod
    def _is_version_at_least(cls, version_str: str, major: int, minor: int) -> bool:
        parsed = cls._parse_major_minor(version_str)
        if parsed is None:
            return False
        return parsed >= (major, minor)

    @staticmethod
    def _clear_timesfm_modules() -> None:
        """Remove cached timesfm modules so we can import different versions."""
        to_delete = [
            name for name in sys.modules if name == "timesfm" or name.startswith("timesfm.")
        ]
        for name in to_delete:
            del sys.modules[name]

    def _remove_legacy_path_if_needed(self) -> None:
        if not self._legacy_path_added or not self._active_legacy_path:
            return
        if self._active_legacy_path in sys.path:
            sys.path.remove(self._active_legacy_path)
        self._legacy_path_added = False
        self._active_legacy_path = None
