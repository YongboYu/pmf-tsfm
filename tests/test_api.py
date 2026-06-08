"""Tests for the agent-clean ``pmf_tsfm.api`` seam (issue #132).

The API *composes* the existing Hydra configs and *calls* the existing cores
(``run_inference`` / ``evaluate_single`` / ``run_er_evaluation``). These tests verify the
**wiring** without downloading model weights — following the repo convention
(``test_pipeline_integration.py``): monkeypatch ``get_model_adapter`` with a tiny mock so
``run_inference`` exercises every code path on CPU in seconds. The genuine Chronos-2 run is
a slow, file-gated tracer bullet at the bottom (skips in CI, which lacks the local log).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pmf_tsfm import api

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SEPSIS_XES = _REPO_ROOT / "data" / "processed_logs" / "sepsis.xes"

CASE, ACT, TS = "case:concept:name", "concept:name", "time:timestamp"


# ===========================================================================
# Tiny mock adapter (no weights, no network) — matches the BaseAdapter calls
# run_inference makes: get_model_adapter(...) → load_model() → predict(...).
# ===========================================================================


class _MockAdapter:
    def __init__(self, prediction_length: int = 7, device: str = "cpu", **_):
        self.prediction_length = prediction_length
        self.device = device

    def load_model(self) -> None:
        pass

    def predict(self, prepared_data, prediction_length=None, **_):
        pred_len = prediction_length or self.prediction_length
        n_seq = len(prepared_data["inputs"])
        n_feat = len(prepared_data["feature_names"])
        rng = np.random.default_rng(0)
        preds = np.abs(rng.standard_normal((n_seq, pred_len, n_feat))).astype(np.float32)
        quants = np.abs(rng.standard_normal((n_seq, pred_len, n_feat, 3))).astype(np.float32)
        return preds, quants


@pytest.fixture
def use_mock_adapter(monkeypatch):
    """Make run_inference use the mock adapter (it imports get_model_adapter at module top)."""
    monkeypatch.setattr(
        "pmf_tsfm.inference.get_model_adapter",
        lambda **kw: _MockAdapter(**kw),
    )


@pytest.fixture
def synthetic_parquet(tmp_path):
    """60-day daily DF-relation series with a retained DatetimeIndex."""
    n = 60
    cols = ["▶ -> A", "A -> B", "B -> C", "C -> ■", "A -> C"]
    rng = np.random.default_rng(1)
    idx = pd.date_range("2022-01-01", periods=n, freq="D")
    df = pd.DataFrame(rng.integers(0, 9, size=(n, len(cols))), index=idx, columns=cols)
    df.index.name = "date"
    path = tmp_path / "synthetic.parquet"
    df.to_parquet(path)
    return path


def _write_tiny_xes(path: Path, *, days: int = 15) -> None:
    """A dense daily A→B log spanning ``days`` calendar days → a ``days``-row series."""
    import pm4py

    rows = []
    for d in range(days):
        day = f"2022-05-{d + 1:02d}"
        rows += [(f"c{d}", "A", f"{day} 09:00"), (f"c{d}", "B", f"{day} 10:00")]
    df = pd.DataFrame(rows, columns=[CASE, ACT, TS])
    df[TS] = pd.to_datetime(df[TS], utc=True)
    pm4py.write_xes(df, str(path), case_id_key=CASE)


# ===========================================================================
# list_models + import-light
# ===========================================================================


def test_list_models_includes_known_groups():
    models = api.list_models()
    assert "chronos/chronos2" in models
    assert any(m.startswith("moirai/") for m in models)
    assert any(m.startswith("timesfm/") for m in models)
    assert models == sorted(models)


def test_importing_api_pulls_no_model_library_or_gradio():
    """`import pmf_tsfm.api` must not import a model lib / Gradio / init CUDA."""
    code = (
        "import sys; import pmf_tsfm.api; "
        "bad = [m for m in ('chronos', 'gradio', 'uni2ts', 'timesfm') if m in sys.modules]; "
        "print('LEAKED:' + ','.join(bad) if bad else 'CLEAN')"
    )
    out = subprocess.run(  # noqa: S603 — fixed argv, no shell, hardcoded snippet
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert "CLEAN" in out.stdout, out.stdout + out.stderr


# ===========================================================================
# forecast_backtest / forecast_only on a prepared parquet (mock adapter)
# ===========================================================================


def test_forecast_backtest_parquet_returns_finite_metrics(
    synthetic_parquet, use_mock_adapter, tmp_path
):
    res = api.forecast_backtest(
        synthetic_parquet, model="chronos/chronos2", horizon=7, workdir=tmp_path / "wd"
    )

    # n_windows = total - val_end - horizon + 1 = 60 - 48 - 7 + 1 = 6.
    assert res["n_windows"] == 6
    assert res["model"] == "chronos_2" and res["horizon"] == 7

    m = res["metrics"]
    for key in ("mae", "mae_std", "rmse", "rmse_std"):
        assert np.isfinite(m[key])
    # Parquet-only input → no XES → ER cannot be computed.
    assert m["er"] is None

    assert Path(res["predictions_path"]).exists()
    assert Path(res["quantiles_path"]).exists()
    assert res["feature_names"] == ["▶ -> A", "A -> B", "B -> C", "C -> ■", "A -> C"]


def test_parquet_only_er_branch_is_skipped(
    synthetic_parquet, use_mock_adapter, monkeypatch, tmp_path
):
    """compute_er=True on parquet input must NOT call the ER core (no log available)."""
    called = {"er": False}

    def _boom(_cfg):
        called["er"] = True
        raise AssertionError("ER should not run for parquet-only input")

    monkeypatch.setattr("pmf_tsfm.er.evaluate_er.run_er_evaluation", _boom)
    res = api.forecast_backtest(
        synthetic_parquet, model="chronos/chronos2", workdir=tmp_path / "wd", compute_er=True
    )
    assert res["metrics"]["er"] is None
    assert called["er"] is False


def test_forecast_only_skips_metrics(synthetic_parquet, use_mock_adapter, tmp_path):
    res = api.forecast_only(synthetic_parquet, model="chronos/chronos2", workdir=tmp_path / "wd")
    assert res["metrics"] is None
    assert res["n_windows"] == 6
    assert Path(res["predictions_path"]).exists()


def test_train_end_val_end_override_split(synthetic_parquet, use_mock_adapter, tmp_path):
    res = api.forecast_only(
        synthetic_parquet, horizon=5, train_end=30, val_end=40, workdir=tmp_path / "wd"
    )
    # n_windows = 60 - 40 - 5 + 1 = 16.
    assert res["n_windows"] == 16


# ===========================================================================
# ER branch wiring for XES input — guards the cfg.task node→string gotcha
# ===========================================================================


def test_xes_input_runs_er_with_string_task(use_mock_adapter, monkeypatch, tmp_path):
    """`.xes` input triggers ER; the ER cfg must carry `task` as the *string* "zero_shot".

    run_er_evaluation joins cfg.task into a path expecting a string, but the inference
    config makes cfg.task a node — this regression-guards the coercion the API does.
    """
    xes = tmp_path / "mylog.xes"
    _write_tiny_xes(xes, days=15)

    captured = {}

    def _spy(er_cfg):
        captured["task"] = er_cfg.task
        captured["output_dir"] = er_cfg.output_dir
        captured["data_path"] = er_cfg.data.path
        return {"summary": {"pred_er_mean": 1.23}}

    monkeypatch.setattr("pmf_tsfm.er.evaluate_er.run_er_evaluation", _spy)

    res = api.forecast_backtest(
        xes, model="chronos/chronos2", horizon=2, workdir=tmp_path / "wd", compute_er=True
    )

    assert res["metrics"]["er"] == 1.23
    assert captured["task"] == "zero_shot"
    assert isinstance(captured["task"], str)
    # ER output dir is concrete (resolved) and points under the scratch workdir.
    assert "/er/zero_shot/mylog/chronos_2" in captured["output_dir"]
    assert captured["data_path"].endswith("series.parquet")


# ===========================================================================
# Genuine Chronos-2 tracer bullet (slow; needs the local sepsis log → skips in CI)
# ===========================================================================


@pytest.mark.slow
@pytest.mark.skipif(
    not _SEPSIS_XES.exists(),
    reason="needs local data/processed_logs/sepsis.xes (not in CI); downloads Chronos-2",
)
def test_chronos2_end_to_end_on_sepsis(tmp_path):
    res = api.forecast_backtest(
        _SEPSIS_XES, model="chronos/chronos2", horizon=7, device="cpu", workdir=tmp_path / "wd"
    )
    m = res["metrics"]
    assert np.isfinite(m["mae"]) and np.isfinite(m["rmse"])
    assert isinstance(m["er"], float)  # XES input → ER computed
    assert res["n_windows"] >= 1
