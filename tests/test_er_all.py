"""Tests for pmf_tsfm.er.evaluate_er_all."""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

import pmf_tsfm.er.evaluate_er_all as er_all_module
from pmf_tsfm.er.evaluate_er_all import _discover_models, _stats, run_er_all


class DummyRun:
    """Minimal W&B-like run object for orchestration tests."""

    def __init__(self) -> None:
        self.logged: list[tuple[int | None, dict]] = []
        self.tables: list[tuple[str, list[str], list[list]]] = []
        self.summary: dict = {}
        self.finished = False

    def log(self, metrics: dict, step: int | None = None) -> None:
        self.logged.append((step, metrics))

    def log_table(self, key: str, columns: list[str], rows: list[list]) -> None:
        self.tables.append((key, columns, rows))

    def log_summary(self, summary: dict) -> None:
        self.summary.update(summary)

    def finish(self) -> None:
        self.finished = True


def _touch_prediction_file(base: Path, dataset: str, model: str) -> None:
    model_dir = base / "zero_shot" / dataset / model
    model_dir.mkdir(parents=True, exist_ok=True)
    np.save(model_dir / f"{dataset}_{model}_predictions.npy", np.zeros((1, 1, 1), dtype=np.float32))


def _make_cfg(tmp_path: Path, *, save: bool = True) -> OmegaConf:
    return OmegaConf.create(
        {
            "data": {
                "name": "BPI2017",
                "path": str(tmp_path / "timeseries.parquet"),
                "val_end": 13,
            },
            "task": "zero_shot",
            "prediction_length": 7,
            "train_ratio": 0.8,
            "model_names": [],
            "save": save,
            "output_base": str(tmp_path / "er_outputs"),
            "paths": {
                "output_dir": str(tmp_path / "outputs"),
                "log_dir": str(tmp_path),
            },
        }
    )


class TestDiscoverModels:
    def test_returns_empty_when_task_directory_is_missing(self, tmp_path: Path) -> None:
        assert _discover_models(tmp_path, "BPI2017", "zero_shot") == []

    def test_filters_requested_models_and_skips_invalid_directories(self, tmp_path: Path) -> None:
        base = tmp_path / "outputs"
        _touch_prediction_file(base, "BPI2017", "model_a")
        _touch_prediction_file(base, "BPI2017", "model_c")
        (base / "zero_shot" / "BPI2017" / "model_b").mkdir(parents=True, exist_ok=True)
        (base / "zero_shot" / "BPI2017" / "README.txt").write_text("not a model dir")

        discovered = _discover_models(base, "BPI2017", "zero_shot", model_names=["model_c"])

        assert discovered == [("model_c", base / "zero_shot" / "BPI2017" / "model_c")]


class TestStats:
    def test_filters_nan_values_before_aggregation(self) -> None:
        mean, std = _stats([1.0, float("nan"), 3.0])
        assert mean == 2.0
        assert std == 1.0

    def test_returns_nan_pair_for_all_nan_input(self) -> None:
        mean, std = _stats([float("nan"), float("nan")])
        assert math.isnan(mean)
        assert math.isnan(std)


class TestRunERAll:
    def test_returns_empty_and_finishes_run_when_no_prediction_files_exist(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        cfg = _make_cfg(tmp_path, save=False)
        run = DummyRun()
        monkeypatch.setattr(er_all_module, "init_run", lambda *args, **kwargs: run)

        result = run_er_all(cfg)

        assert result == {}
        assert run.finished

    def test_mocked_multi_model_run_saves_outputs_and_logs_tables(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        outputs_dir = tmp_path / "outputs"
        _touch_prediction_file(outputs_dir, "BPI2017", "model_a")
        _touch_prediction_file(outputs_dir, "BPI2017", "model_b")

        cfg = _make_cfg(tmp_path, save=True)
        run = DummyRun()
        monkeypatch.setattr(er_all_module, "init_run", lambda *args, **kwargs: run)
        monkeypatch.setattr(
            er_all_module,
            "_load_predictions",
            lambda inference_dir, dataset_name, model_name: (
                np.full((2, 7, 1), 1.0 if model_name == "model_a" else 2.0, dtype=np.float32),
                [model_name],
            ),
        )

        starts = pd.DatetimeIndex(
            [
                pd.Timestamp("2020-01-01", tz="UTC"),
                pd.Timestamp("2020-01-08", tz="UTC"),
            ]
        )
        ends = pd.DatetimeIndex(
            [
                pd.Timestamp("2020-01-07", tz="UTC"),
                pd.Timestamp("2020-01-14", tz="UTC"),
            ]
        )
        monkeypatch.setattr(
            er_all_module, "_get_test_dates", lambda *args, **kwargs: (starts, ends)
        )
        monkeypatch.setattr(er_all_module, "load_event_log", lambda path: list(range(12)))
        monkeypatch.setattr(er_all_module, "prepare_log", lambda raw_log: {"prepared": raw_log})
        monkeypatch.setattr(
            er_all_module, "_build_training_dfg", lambda raw_log, ratio: {"kind": "train"}
        )
        monkeypatch.setattr(er_all_module, "dfg_to_json", lambda dfg: dfg)

        window_state = {"idx": -1}

        def fake_extract_sublog(prepared, ws, we):
            window_state["idx"] += 1
            return {"window_idx": window_state["idx"]}

        def fake_extract_traces(sublog):
            return [] if sublog["window_idx"] == 1 else [["A", "B"], ["A", "B"]]

        monkeypatch.setattr(er_all_module, "extract_sublog", fake_extract_sublog)
        monkeypatch.setattr(er_all_module, "extract_traces", fake_extract_traces)
        monkeypatch.setattr(
            er_all_module,
            "build_truth_dfg",
            lambda sublog: {"kind": "truth", "window_idx": sublog["window_idx"]},
        )
        monkeypatch.setattr(
            er_all_module,
            "build_prediction_dfg",
            lambda preds_window, feature_names: {"kind": "pred", "model": feature_names[0]},
        )

        def fake_compute_er(dfg_json, traces):
            kind = dfg_json["kind"]
            if kind == "truth":
                return 1.2, 1.0, len(traces)
            if kind == "train":
                return 0.9, 1.0, len(traces)
            if dfg_json["model"] == "model_a":
                return 0.7, 0.5, len(traces)
            return 0.6, 0.4, len(traces)

        monkeypatch.setattr(er_all_module, "compute_er", fake_compute_er)

        result = run_er_all(cfg)

        assert set(result) == {"model_a", "model_b"}
        assert result["model_a"]["summary"]["n_windows"] == 2
        assert result["model_a"]["summary"]["n_valid_windows"] == 1
        assert math.isnan(result["model_a"]["windows"][1]["pred_er"])
        assert result["model_a"]["windows"][1]["n_traces"] == 0

        model_a_json = tmp_path / "er_outputs" / "model_a" / "BPI2017_model_a_er.json"
        model_b_json = tmp_path / "er_outputs" / "model_b" / "BPI2017_model_b_er.json"
        combined_json = tmp_path / "er_outputs" / "BPI2017_er_all_summary.json"
        assert model_a_json.exists()
        assert model_b_json.exists()
        assert combined_json.exists()

        combined = json.loads(combined_json.read_text())
        assert combined["dataset"] == "BPI2017"
        assert set(combined["models"]) == {"model_a", "model_b"}
        assert any(key == "er/model_comparison" for key, _, _ in run.tables)
        assert run.summary["er/n_models"] == 2
        assert run.summary["er/n_windows"] == 2
        assert run.finished

    def test_main_wrapper_prints_config_and_calls_run_er_all(self, monkeypatch, capsys) -> None:
        cfg = OmegaConf.create({"print_config": True, "data": {"name": "BPI2017"}})
        seen: dict[str, object] = {}

        def fake_run_er_all(passed_cfg):
            seen["cfg"] = passed_cfg
            return {"ok": True}

        monkeypatch.setattr(er_all_module, "run_er_all", fake_run_er_all)

        er_all_module.main.__wrapped__(cfg)

        assert seen["cfg"] is cfg
        assert "print_config: true" in capsys.readouterr().out.lower()
