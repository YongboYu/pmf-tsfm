"""
Tests for utility modules:
  - pmf_tsfm.utils.metrics  (print_metrics, save_metrics, print_aggregate_summary)
  - pmf_tsfm.utils.wandb_logger (WandbRun, init_run)
  - pmf_tsfm.models.base (BaseAdapter._parse_dtype, is_loaded, __repr__)

No real ML models or GPU required.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from omegaconf import OmegaConf

from pmf_tsfm.models.base import BaseAdapter
from pmf_tsfm.utils.metrics import (
    print_aggregate_summary,
    print_metrics,
    save_metrics,
)
from pmf_tsfm.utils.wandb_logger import WandbRun, init_run

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_metrics(mae_mean=0.5, mae_std=0.1, rmse_mean=0.8, rmse_std=0.15, n=10) -> dict:
    """Build a minimal metrics dict matching compute_metrics output."""
    return {
        "summary": {
            "MAE_mean": mae_mean,
            "MAE_std": mae_std,
            "RMSE_mean": rmse_mean,
            "RMSE_std": rmse_std,
            "num_sequences": n,
            "num_features": 3,
            "prediction_length": 7,
        },
        "per_feature": {
            "feat_0": {"MAE": mae_mean, "RMSE": rmse_mean},
            "feat_1": {"MAE": mae_mean + 0.1, "RMSE": rmse_mean + 0.1},
        },
    }


def _make_predictions(n_seq=8, pred_len=7, n_features=3, seed=42):
    rng = np.random.default_rng(seed)
    predictions = rng.random((n_seq, pred_len, n_features)).astype(np.float32)
    targets = rng.random((n_seq, pred_len, n_features)).astype(np.float32)
    return predictions, targets


# ---------------------------------------------------------------------------
# Concrete stub for BaseAdapter (abstract)
# ---------------------------------------------------------------------------


class _StubAdapter(BaseAdapter):
    def load_model(self):
        self._is_loaded = True

    def predict(self, prepared_data, prediction_length=None, **kwargs):
        return np.zeros((1, 7, 3)), np.zeros((1, 7, 3, 1))


# ---------------------------------------------------------------------------
# metrics: print_metrics
# ---------------------------------------------------------------------------


class TestPrintMetrics:
    def test_prints_mae_value(self, capsys):
        metrics = _make_metrics(mae_mean=1.2345)
        print_metrics(metrics)
        captured = capsys.readouterr()
        assert "1.2345" in captured.out

    def test_prints_rmse_value(self, capsys):
        metrics = _make_metrics(rmse_mean=2.3456)
        print_metrics(metrics)
        captured = capsys.readouterr()
        assert "2.3456" in captured.out

    def test_prints_title_when_provided(self, capsys):
        metrics = _make_metrics()
        print_metrics(metrics, title="my_model on BPI2017")
        captured = capsys.readouterr()
        assert "my_model on BPI2017" in captured.out

    def test_prints_plus_minus_separator(self, capsys):
        metrics = _make_metrics()
        print_metrics(metrics)
        captured = capsys.readouterr()
        assert "+/-" in captured.out

    def test_prints_num_sequences(self, capsys):
        metrics = _make_metrics(n=42)
        print_metrics(metrics)
        captured = capsys.readouterr()
        assert "42" in captured.out


# ---------------------------------------------------------------------------
# metrics: save_metrics
# ---------------------------------------------------------------------------


class TestSaveMetrics:
    def test_creates_json_file(self, tmp_path):
        metrics = _make_metrics()
        out = tmp_path / "metrics.json"
        save_metrics(metrics, out)
        assert out.exists()

    def test_json_is_valid(self, tmp_path):
        metrics = _make_metrics()
        out = tmp_path / "metrics.json"
        save_metrics(metrics, out)
        with open(out) as f:
            loaded = json.load(f)
        assert "summary" in loaded

    def test_mae_mean_preserved(self, tmp_path):
        metrics = _make_metrics(mae_mean=3.14159)
        out = tmp_path / "metrics.json"
        save_metrics(metrics, out)
        with open(out) as f:
            loaded = json.load(f)
        assert abs(loaded["summary"]["MAE_mean"] - 3.14159) < 1e-4

    def test_creates_parent_dirs(self, tmp_path):
        metrics = _make_metrics()
        out = tmp_path / "nested" / "deep" / "metrics.json"
        save_metrics(metrics, out)
        assert out.exists()

    def test_accepts_string_path(self, tmp_path):
        metrics = _make_metrics()
        out = str(tmp_path / "metrics.json")
        save_metrics(metrics, out)
        assert Path(out).exists()


# ---------------------------------------------------------------------------
# metrics: print_aggregate_summary
# ---------------------------------------------------------------------------


class TestPrintAggregateSummary:
    def test_prints_all_model_names(self, capsys):
        all_metrics = {
            "BPI2017_chronos_bolt_small": _make_metrics(mae_mean=0.5),
            "BPI2017_moirai_1_1_small": _make_metrics(mae_mean=0.7),
        }
        print_aggregate_summary(all_metrics)
        out = capsys.readouterr().out
        assert "BPI2017_chronos_bolt_small" in out
        assert "BPI2017_moirai_1_1_small" in out

    def test_prints_average_row(self, capsys):
        all_metrics = {
            "model_a": _make_metrics(mae_mean=0.4),
            "model_b": _make_metrics(mae_mean=0.6),
        }
        print_aggregate_summary(all_metrics)
        out = capsys.readouterr().out
        assert "AVERAGE" in out

    def test_average_mae_is_correct(self, capsys):
        all_metrics = {
            "model_a": _make_metrics(mae_mean=0.4),
            "model_b": _make_metrics(mae_mean=0.6),
        }
        print_aggregate_summary(all_metrics)
        out = capsys.readouterr().out
        # Average of 0.4 and 0.6 = 0.5
        assert "0.5000" in out

    def test_prints_header(self, capsys):
        print_aggregate_summary({"m": _make_metrics()})
        out = capsys.readouterr().out
        assert "AGGREGATE SUMMARY" in out


# ---------------------------------------------------------------------------
# WandbRun
# ---------------------------------------------------------------------------


class TestWandbRun:
    def test_enabled_false_when_run_is_none(self):
        wr = WandbRun(None)
        assert wr.enabled is False

    def test_enabled_true_when_run_set(self):
        wr = WandbRun(MagicMock())
        assert wr.enabled is True

    def test_log_calls_underlying_run(self):
        mock_run = MagicMock()
        wr = WandbRun(mock_run)
        wr.log({"loss": 0.5})
        mock_run.log.assert_called_once_with({"loss": 0.5})

    def test_log_with_step_passes_step(self):
        mock_run = MagicMock()
        wr = WandbRun(mock_run)
        wr.log({"loss": 0.5}, step=10)
        mock_run.log.assert_called_once_with({"loss": 0.5}, step=10)

    def test_log_noop_when_disabled(self):
        wr = WandbRun(None)
        wr.log({"loss": 0.5})  # should not raise

    def test_log_summary_sets_values(self):
        mock_run = MagicMock()
        wr = WandbRun(mock_run)
        wr.log_summary({"mae": 1.0, "rmse": 2.0})
        mock_run.summary.__setitem__.assert_any_call("mae", 1.0)
        mock_run.summary.__setitem__.assert_any_call("rmse", 2.0)

    def test_log_summary_noop_when_disabled(self):
        wr = WandbRun(None)
        wr.log_summary({"mae": 1.0})  # should not raise

    def test_finish_calls_run_finish(self):
        mock_run = MagicMock()
        wr = WandbRun(mock_run)
        wr.finish()
        mock_run.finish.assert_called_once()

    def test_finish_noop_when_disabled(self):
        wr = WandbRun(None)
        wr.finish()  # should not raise

    def test_log_table_calls_wandb_table(self):
        mock_run = MagicMock()
        wr = WandbRun(mock_run)
        with patch("wandb.Table") as mock_table:
            mock_table.return_value = MagicMock()
            wr.log_table("my_table", ["col_a", "col_b"], [[1, 2], [3, 4]])
            mock_table.assert_called_once_with(columns=["col_a", "col_b"], data=[[1, 2], [3, 4]])

    def test_log_table_noop_when_disabled(self):
        wr = WandbRun(None)
        wr.log_table("t", ["a"], [[1]])  # should not raise


# ---------------------------------------------------------------------------
# init_run
# ---------------------------------------------------------------------------


class TestInitRun:
    def _disabled_cfg(self):
        return OmegaConf.create({"logger": {"enabled": False}})

    def _enabled_cfg(self, mode="online"):
        return OmegaConf.create(
            {
                "logger": {
                    "enabled": True,
                    "project": "test-project",
                    "entity": None,
                    "mode": mode,
                    "tags": [],
                    "group": None,
                    "notes": None,
                }
            }
        )

    def test_returns_noop_run_when_disabled(self):
        cfg = self._disabled_cfg()
        run = init_run(cfg, job_type="test")
        assert isinstance(run, WandbRun)
        assert not run.enabled

    def test_returns_noop_run_when_no_logger_key(self):
        cfg = OmegaConf.create({})
        run = init_run(cfg, job_type="test")
        assert not run.enabled

    def test_returns_noop_run_when_wandb_not_installed(self):
        cfg = self._enabled_cfg()
        with patch.dict("sys.modules", {"wandb": None}):
            run = init_run(cfg, job_type="test")
        assert not run.enabled

    def test_calls_wandb_init_when_enabled(self):
        cfg = self._enabled_cfg()
        mock_wandb_run = MagicMock()
        with patch("wandb.init", return_value=mock_wandb_run) as mock_init:
            run = init_run(cfg, job_type="inference", name="test/BPI2017/model")
        assert run.enabled
        mock_init.assert_called_once()

    def test_passes_job_type_to_wandb(self):
        cfg = self._enabled_cfg()
        with patch("wandb.init", return_value=MagicMock()) as mock_init:
            init_run(cfg, job_type="evaluate")
        call_kwargs = mock_init.call_args.kwargs
        assert call_kwargs["job_type"] == "evaluate"

    def test_passes_name_to_wandb(self):
        cfg = self._enabled_cfg()
        with patch("wandb.init", return_value=MagicMock()) as mock_init:
            init_run(cfg, job_type="inference", name="eval/zero_shot")
        call_kwargs = mock_init.call_args.kwargs
        assert call_kwargs["name"] == "eval/zero_shot"

    def test_extra_tags_appended(self):
        cfg = self._enabled_cfg()
        with patch("wandb.init", return_value=MagicMock()) as mock_init:
            init_run(cfg, job_type="train", tags=["chronos", "BPI2017"])
        call_kwargs = mock_init.call_args.kwargs
        assert "chronos" in call_kwargs["tags"]
        assert "BPI2017" in call_kwargs["tags"]

    def test_group_override_takes_precedence(self):
        cfg = self._enabled_cfg()
        with patch("wandb.init", return_value=MagicMock()) as mock_init:
            init_run(cfg, job_type="train", group="my_group")
        call_kwargs = mock_init.call_args.kwargs
        assert call_kwargs["group"] == "my_group"


# ---------------------------------------------------------------------------
# BaseAdapter
# ---------------------------------------------------------------------------


class TestBaseAdapter:
    def _make_adapter(self, dtype="float32"):
        return _StubAdapter(
            model_name="test_model",
            model_id="test/id",
            model_family="test",
            variant="small",
            device="cpu",
            torch_dtype=dtype,
            prediction_length=7,
        )

    def test_parse_dtype_float32(self):
        adapter = self._make_adapter("float32")
        assert adapter.torch_dtype == torch.float32

    def test_parse_dtype_float16(self):
        adapter = self._make_adapter("float16")
        assert adapter.torch_dtype == torch.float16

    def test_parse_dtype_bfloat16(self):
        adapter = self._make_adapter("bfloat16")
        assert adapter.torch_dtype == torch.bfloat16

    def test_parse_dtype_unknown_defaults_to_float32(self):
        adapter = self._make_adapter("unknown_dtype")
        assert adapter.torch_dtype == torch.float32

    def test_is_loaded_false_before_load(self):
        adapter = self._make_adapter()
        assert adapter.is_loaded is False

    def test_is_loaded_true_after_load(self):
        adapter = self._make_adapter()
        adapter.load_model()
        assert adapter.is_loaded is True

    def test_repr_contains_model_name(self):
        adapter = self._make_adapter()
        assert "test_model" in repr(adapter)

    def test_repr_contains_variant(self):
        adapter = self._make_adapter()
        assert "small" in repr(adapter)

    def test_repr_contains_class_name(self):
        adapter = self._make_adapter()
        assert "_StubAdapter" in repr(adapter)

    def test_prediction_length_stored(self):
        adapter = self._make_adapter()
        assert adapter.prediction_length == 7

    def test_device_stored(self):
        adapter = self._make_adapter()
        assert adapter.device == "cpu"
