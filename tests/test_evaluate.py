"""
Tests for pmf_tsfm.evaluate — find_result_files, load_predictions,
evaluate_single, and evaluate_all.

All tests use synthetic .npy files in tmp_path; no real models required.
"""

import json
from pathlib import Path

import numpy as np

from pmf_tsfm.evaluate import (
    evaluate_all,
    evaluate_single,
    find_result_files,
    load_predictions,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_prediction_files(
    base_dir: Path,
    dataset: str,
    model: str,
    n_seq: int = 10,
    pred_len: int = 7,
    n_features: int = 3,
    rng_seed: int = 0,
    write_metadata: bool = True,
) -> Path:
    """Write synthetic prediction, target, and metadata files under base_dir/{dataset}/{model}/."""
    rng = np.random.default_rng(rng_seed)
    pred_dir = base_dir / dataset / model
    pred_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"{dataset}_{model}"
    predictions = rng.random((n_seq, pred_len, n_features)).astype(np.float32)
    targets = rng.random((n_seq, pred_len, n_features)).astype(np.float32)

    np.save(pred_dir / f"{prefix}_predictions.npy", predictions)
    np.save(pred_dir / f"{prefix}_targets.npy", targets)

    if write_metadata:
        metadata = {
            "feature_names": [f"feat_{i}" for i in range(n_features)],
            "prediction_shape": [n_seq, pred_len, n_features],
        }
        with open(pred_dir / f"{prefix}_metadata.json", "w") as f:
            json.dump(metadata, f)

    return pred_dir


# ---------------------------------------------------------------------------
# find_result_files
# ---------------------------------------------------------------------------


class TestFindResultFiles:
    def test_finds_single_prediction_file(self, tmp_path):
        _write_prediction_files(tmp_path, "BPI2017", "chronos_bolt_small")
        results = find_result_files(tmp_path)
        assert len(results) == 1

    def test_returns_correct_dataset_and_model(self, tmp_path):
        _write_prediction_files(tmp_path, "BPI2017", "chronos_bolt_small")
        _file_dir, dataset, model = find_result_files(tmp_path)[0]
        assert dataset == "BPI2017"
        assert model == "chronos_bolt_small"

    def test_finds_multiple_results(self, tmp_path):
        _write_prediction_files(tmp_path, "BPI2017", "chronos_bolt_small")
        _write_prediction_files(tmp_path, "BPI2017", "moirai_1_1_small")
        _write_prediction_files(tmp_path, "Sepsis", "chronos_bolt_small")
        results = find_result_files(tmp_path)
        assert len(results) == 3

    def test_returns_empty_for_empty_dir(self, tmp_path):
        assert find_result_files(tmp_path) == []

    def test_file_dir_contains_prediction_file(self, tmp_path):
        _write_prediction_files(tmp_path, "BPI2017", "chronos_bolt_small")
        file_dir, dataset, model = find_result_files(tmp_path)[0]
        assert (file_dir / f"{dataset}_{model}_predictions.npy").exists()

    def test_results_are_sorted(self, tmp_path):
        _write_prediction_files(tmp_path, "Sepsis", "model_b")
        _write_prediction_files(tmp_path, "BPI2017", "model_a")
        results = find_result_files(tmp_path)
        names = [(d, m) for _, d, m in results]
        assert names == sorted(names)


# ---------------------------------------------------------------------------
# load_predictions
# ---------------------------------------------------------------------------


class TestLoadPredictions:
    def test_loads_predictions_array(self, tmp_path):
        pred_dir = _write_prediction_files(tmp_path, "BPI2017", "chronos_bolt_small")
        preds, _targets, _ = load_predictions(pred_dir, "BPI2017", "chronos_bolt_small")
        assert preds.shape == (10, 7, 3)

    def test_loads_targets_array(self, tmp_path):
        pred_dir = _write_prediction_files(tmp_path, "BPI2017", "chronos_bolt_small")
        _preds, targets, _ = load_predictions(pred_dir, "BPI2017", "chronos_bolt_small")
        assert targets.shape == (10, 7, 3)

    def test_loads_metadata_when_present(self, tmp_path):
        pred_dir = _write_prediction_files(tmp_path, "BPI2017", "chronos_bolt_small")
        _, _, metadata = load_predictions(pred_dir, "BPI2017", "chronos_bolt_small")
        assert metadata is not None
        assert "feature_names" in metadata

    def test_metadata_is_none_when_missing(self, tmp_path):
        pred_dir = _write_prediction_files(
            tmp_path, "BPI2017", "chronos_bolt_small", write_metadata=False
        )
        _, _, metadata = load_predictions(pred_dir, "BPI2017", "chronos_bolt_small")
        assert metadata is None

    def test_predictions_and_targets_have_same_shape(self, tmp_path):
        pred_dir = _write_prediction_files(tmp_path, "Sepsis", "moirai_1_1_large", n_seq=5)
        preds, targets, _ = load_predictions(pred_dir, "Sepsis", "moirai_1_1_large")
        assert preds.shape == targets.shape


# ---------------------------------------------------------------------------
# evaluate_single
# ---------------------------------------------------------------------------


class TestEvaluateSingle:
    def test_returns_metrics_dict(self, tmp_path):
        pred_dir = _write_prediction_files(tmp_path, "BPI2017", "chronos_bolt_small")
        metrics = evaluate_single(pred_dir, "BPI2017", "chronos_bolt_small", save=False)
        assert "summary" in metrics
        assert "per_feature" in metrics

    def test_summary_has_mae_and_rmse(self, tmp_path):
        pred_dir = _write_prediction_files(tmp_path, "BPI2017", "chronos_bolt_small")
        metrics = evaluate_single(pred_dir, "BPI2017", "chronos_bolt_small", save=False)
        s = metrics["summary"]
        assert "MAE_mean" in s
        assert "RMSE_mean" in s

    def test_model_and_dataset_names_stored(self, tmp_path):
        pred_dir = _write_prediction_files(tmp_path, "BPI2017", "chronos_bolt_small")
        metrics = evaluate_single(pred_dir, "BPI2017", "chronos_bolt_small", save=False)
        assert metrics["model_name"] == "chronos_bolt_small"
        assert metrics["dataset_name"] == "BPI2017"

    def test_saves_metrics_json_when_save_true(self, tmp_path):
        pred_dir = _write_prediction_files(tmp_path, "BPI2017", "chronos_bolt_small")
        evaluate_single(pred_dir, "BPI2017", "chronos_bolt_small", save=True)
        assert (pred_dir / "BPI2017_chronos_bolt_small_metrics.json").exists()

    def test_does_not_save_when_save_false(self, tmp_path):
        pred_dir = _write_prediction_files(tmp_path, "BPI2017", "chronos_bolt_small")
        evaluate_single(pred_dir, "BPI2017", "chronos_bolt_small", save=False)
        assert not (pred_dir / "BPI2017_chronos_bolt_small_metrics.json").exists()

    def test_feature_names_from_metadata(self, tmp_path):
        pred_dir = _write_prediction_files(tmp_path, "BPI2017", "chronos_bolt_small")
        metrics = evaluate_single(pred_dir, "BPI2017", "chronos_bolt_small", save=False)
        assert "feat_0" in metrics["per_feature"]

    def test_prediction_length_stored_from_metadata(self, tmp_path):
        pred_dir = _write_prediction_files(tmp_path, "BPI2017", "chronos_bolt_small", pred_len=7)
        metrics = evaluate_single(pred_dir, "BPI2017", "chronos_bolt_small", save=False)
        assert metrics["prediction_length"] == 7


# ---------------------------------------------------------------------------
# evaluate_all
# ---------------------------------------------------------------------------


class TestEvaluateAll:
    def test_returns_empty_dict_for_empty_dir(self, tmp_path):
        result = evaluate_all(tmp_path, save=False)
        assert result == {}

    def test_finds_and_evaluates_single_result(self, tmp_path):
        _write_prediction_files(tmp_path, "BPI2017", "chronos_bolt_small")
        result = evaluate_all(tmp_path, save=False)
        assert len(result) == 1

    def test_keys_are_dataset_model_joined(self, tmp_path):
        _write_prediction_files(tmp_path, "BPI2017", "chronos_bolt_small")
        result = evaluate_all(tmp_path, save=False)
        assert "BPI2017_chronos_bolt_small" in result

    def test_evaluates_multiple_results(self, tmp_path):
        _write_prediction_files(tmp_path, "BPI2017", "chronos_bolt_small")
        _write_prediction_files(tmp_path, "Sepsis", "moirai_1_1_small")
        result = evaluate_all(tmp_path, save=False)
        assert len(result) == 2

    def test_skips_corrupt_file_and_continues(self, tmp_path):
        pred_dir = _write_prediction_files(tmp_path, "BPI2017", "chronos_bolt_small")
        _write_prediction_files(tmp_path, "Sepsis", "moirai_1_1_small")
        # Corrupt one predictions file
        (pred_dir / "BPI2017_chronos_bolt_small_predictions.npy").write_bytes(b"not_numpy")
        result = evaluate_all(tmp_path, save=False)
        # Should still return one valid result
        assert "Sepsis_moirai_1_1_small" in result

    def test_each_result_has_summary(self, tmp_path):
        _write_prediction_files(tmp_path, "BPI2017", "chronos_bolt_small")
        _write_prediction_files(tmp_path, "Sepsis", "moirai_1_1_small")
        result = evaluate_all(tmp_path, save=False)
        for metrics in result.values():
            assert "summary" in metrics
