"""
Tests for pmf_tsfm.export.hf_results — the results-dataset normalizer.

Folds the nested per-(task, dataset, model) ``outputs/`` tree (metrics JSON,
metadata JSON, ER JSON) into one long-format table with a ``level`` discriminator
(summary / per_feature / per_window), ready to publish as an HF Dataset.

Provenance note: the real exporter is pointed at ``data/hpc_sync/outputs/`` (the
canonical VSC/HPC runs), never the top-level local ``outputs/`` tree. These tests
build a tiny synthetic tree of the same shape.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from pmf_tsfm.export.hf_results import (
    COLUMNS,
    build_dataframe,
    collect_rows,
)

# ---------------------------------------------------------------------------
# Synthetic outputs tree
# ---------------------------------------------------------------------------


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def _metadata(dataset: str, model: str, task: str, *, family: str, dtype: str) -> dict:
    # config.task is a plain string for zero-shot but the nested task-config group
    # (a dict) for LoRA/full-tune runs — mirror that so the exporter is exercised on both.
    config_task: object = task
    if task != "zero_shot":
        config_task = {"name": task, "context_mode": "fixed", "use_amp": True}
    return {
        "mode": f"{task.upper()} INFERENCE",
        "task": task,
        "config": {
            "seed": 42,
            "device": "cuda",
            "context_length": 48,
            "prediction_length": 7,
            "task": config_task,
            "model": {
                "name": model,
                "family": family,
                "id": f"amazon/{model.replace('_', '-')}",
                "variant": model,
                "torch_dtype": dtype,
            },
            "data": {"name": dataset, "split_ratio": [0.6, 0.2, 0.2]},
        },
        "data_metadata": {
            "dataset_name": dataset,
            "num_features": 2,
            "prediction_length": 7,
            "test_num_sequences": 3,
        },
    }


def _metrics() -> dict:
    return {
        "summary": {
            "MAE_mean": 7.7,
            "MAE_std": 8.6,
            "RMSE_mean": 10.2,
            "RMSE_std": 11.9,
            "num_sequences": 3,
            "num_features": 2,
            "prediction_length": 7,
        },
        "per_feature": {
            "A -> B": {"MAE": 1.0, "RMSE": 2.0},
            "B -> ■": {"MAE": 3.0, "RMSE": 4.0},
        },
        "model_name": "chronos_bolt_small",
        "dataset_name": "BPI2017",
        "prediction_length": 7,
    }


def _er() -> dict:
    return {
        "summary": {
            "model": "chronos_bolt_small",
            "dataset": "BPI2017",
            "task": "zero_shot",
            "n_windows": 2,
            "n_valid_windows": 2,
            "truth_er_mean": 1.02,
            "truth_er_std": 0.07,
            "truth_fitting_ratio_mean": 1.0,
            "pred_er_mean": 1.11,
            "pred_er_std": 0.12,
            "pred_fitting_ratio_mean": 0.99,
            "training_er_mean": 1.17,
            "training_er_std": 0.10,
        },
        "windows": [
            {"window": "w1", "pred_er": 0.98, "pred_fitting_ratio": 0.999, "n_traces": 100},
            {"window": "w2", "pred_er": 1.04, "pred_fitting_ratio": 0.997, "n_traces": 120},
            # invalid window: ER undefined (source encodes pred_er as NaN), fitting_ratio 0
            {"window": "w3", "pred_er": float("nan"), "pred_fitting_ratio": 0.0, "n_traces": 1},
        ],
    }


def _er_all_summary() -> dict:
    return {
        "dataset": "BPI2017",
        "task": "zero_shot",
        "n_windows": 2,
        "elapsed_s": 30.1,
        "truth_er_mean": 1.02,
        "truth_er_std": 0.07,
        "training_er_mean": 1.17,
        "training_er_std": 0.10,
        "models": {},
    }


@pytest.fixture
def outputs_tree(tmp_path) -> Path:
    """A minimal ``outputs/`` tree: one zero-shot run (+ER) and one LoRA run (no ER)."""
    root = tmp_path / "outputs"
    ds, model = "BPI2017", "chronos_bolt_small"

    # zero-shot run (has metrics + metadata + ER)
    zs = root / "zero_shot" / ds / model
    _write_json(zs / f"{ds}_{model}_metrics.json", _metrics())
    _write_json(
        zs / f"{ds}_{model}_metadata.json",
        _metadata(ds, model, "zero_shot", family="chronos", dtype="float32"),
    )
    er_dir = root / "er" / "zero_shot" / ds / model
    _write_json(er_dir / f"{ds}_{model}_er.json", _er())
    _write_json(
        root / "er" / "zero_shot" / ds / f"{ds}_er_all_summary.json",
        _er_all_summary(),
    )

    # LoRA run (metrics + metadata only, no ER)
    lt = root / "lora_tune" / ds / model
    _write_json(lt / f"{ds}_{model}_metrics.json", _metrics())
    _write_json(
        lt / f"{ds}_{model}_metadata.json",
        _metadata(ds, model, "lora_tune", family="chronos", dtype="float32"),
    )
    return root


# ---------------------------------------------------------------------------
# collect_rows
# ---------------------------------------------------------------------------


def test_every_row_has_the_exact_schema(outputs_tree):
    rows = collect_rows(outputs_tree)
    assert rows, "expected at least some rows"
    for row in rows:
        assert set(row.keys()) == set(COLUMNS)


def test_dataset_name_is_normalized_lowercase(outputs_tree):
    rows = collect_rows(outputs_tree)
    assert {r["dataset"] for r in rows} == {"bpi2017"}


def _one(rows, **filters):
    hits = [r for r in rows if all(r.get(k) == v for k, v in filters.items())]
    assert len(hits) == 1, f"expected exactly 1 row for {filters}, got {len(hits)}"
    return hits[0]


def test_summary_mae_row_carries_mean_std_and_context(outputs_tree):
    rows = collect_rows(outputs_tree)
    row = _one(rows, task="zero_shot", level="summary", metric="MAE")
    assert row["value"] == pytest.approx(7.7)
    assert row["std"] == pytest.approx(8.6)
    assert row["family"] == "chronos"
    assert row["seed"] == 42
    assert row["device"] == "cuda"
    assert row["precision"] == "float32"
    assert row["base_model_id"] == "amazon/chronos-bolt-small"
    assert row["horizon"] == 7
    assert row["n_windows"] == 3
    assert row["n_features"] == 2
    assert row["project"] == "pmf-tsfm"
    assert row["paper"] == "arXiv:2512.07624"


def test_per_feature_rows_preserve_relation_labels(outputs_tree):
    rows = collect_rows(outputs_tree)
    pf = [r for r in rows if r["level"] == "per_feature" and r["task"] == "zero_shot"]
    # 2 features x 2 metrics (MAE, RMSE)
    assert len(pf) == 4
    assert {r["feature"] for r in pf} == {"A -> B", "B -> ■"}
    bb = _one(rows, task="zero_shot", level="per_feature", feature="A -> B", metric="RMSE")
    assert bb["value"] == pytest.approx(2.0)
    assert bb["std"] is None  # per-feature has no std


def test_er_summary_and_per_window_rows(outputs_tree):
    rows = collect_rows(outputs_tree)
    per = _one(
        rows, task="zero_shot", level="summary", metric="pred_er", model="chronos_bolt_small"
    )
    assert per["value"] == pytest.approx(1.11)
    assert per["std"] == pytest.approx(0.12)

    windows = [r for r in rows if r["level"] == "per_window"]
    # w1,w2 valid -> 2 x 2 metrics; w3 invalid -> only its finite fitting_ratio survives
    assert len(windows) == 5
    w1 = _one(rows, level="per_window", window="w1", metric="pred_er")
    assert w1["value"] == pytest.approx(0.98)
    assert w1["n_traces"] == 100


def test_reference_rows_emitted_once_per_dataset(outputs_tree):
    rows = collect_rows(outputs_tree)
    refs = [r for r in rows if r["model"] == "reference"]
    metrics = sorted(r["metric"] for r in refs)
    assert metrics == ["training_er", "truth_er"]
    truth = _one(rows, model="reference", metric="truth_er")
    assert truth["value"] == pytest.approx(1.02)
    assert truth["std"] == pytest.approx(0.07)
    assert truth["runtime_s"] == pytest.approx(30.1)  # ER elapsed_s lives at dataset level


def test_task_resolves_to_string_even_when_config_task_is_a_dict(outputs_tree):
    # LoRA/full-tune metadata carry config.task as the nested task-config dict; every
    # emitted row's `task` must still be the string id (a dict would break Parquet).
    rows = collect_rows(outputs_tree)
    assert all(isinstance(r["task"], str) for r in rows)
    assert {r["task"] for r in rows} == {"zero_shot", "lora_tune"}


def test_lora_run_has_metrics_but_no_er_rows(outputs_tree):
    rows = collect_rows(outputs_tree)
    lora = [r for r in rows if r["task"] == "lora_tune"]
    assert lora, "expected LoRA rows"
    assert all(r["level"] in {"summary", "per_feature"} for r in lora)
    assert all(r["metric"] in {"MAE", "RMSE"} for r in lora)
    assert not any(r["level"] == "per_window" for r in lora)


def test_commit_is_stamped_when_provided(outputs_tree):
    rows = collect_rows(outputs_tree, commit="deadbee")
    assert all(r["commit"] == "deadbee" for r in rows)


def test_exclude_models_drops_that_model_but_keeps_reference_rows(outputs_tree):
    rows = collect_rows(outputs_tree, exclude_models=("chronos_bolt_small",))
    assert not any(r["model"] == "chronos_bolt_small" for r in rows)
    # dataset-level ER baselines are independent of the model filter
    assert any(r["model"] == "reference" for r in rows)


def test_invalid_er_windows_are_dropped_so_value_is_never_null(outputs_tree):
    # w3 has pred_er = NaN (invalid ER window); its pred_er row must not be emitted,
    # keeping the "value is never null" invariant. Its finite fitting_ratio survives.
    rows = collect_rows(outputs_tree)
    assert all(r["value"] is not None for r in rows)
    er_windows = {
        r["window"] for r in rows if r["level"] == "per_window" and r["metric"] == "pred_er"
    }
    assert er_windows == {"w1", "w2"}
    fit_windows = {
        r["window"]
        for r in rows
        if r["level"] == "per_window" and r["metric"] == "pred_fitting_ratio"
    }
    assert "w3" in fit_windows  # fitting_ratio 0.0 is a real value, kept


# ---------------------------------------------------------------------------
# build_dataframe
# ---------------------------------------------------------------------------


def test_build_dataframe_has_columns_in_fixed_order(outputs_tree):
    df = build_dataframe(collect_rows(outputs_tree))
    assert list(df.columns) == COLUMNS
    assert len(df) > 0


def test_dataframe_roundtrips_through_parquet(outputs_tree, tmp_path):
    df = build_dataframe(collect_rows(outputs_tree))
    out = tmp_path / "results.parquet"
    df.to_parquet(out, index=False)
    back = pd.read_parquet(out)
    assert list(back.columns) == COLUMNS
    assert len(back) == len(df)
