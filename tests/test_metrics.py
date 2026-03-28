"""
Tests for metrics.py — verifies MAE and RMSE match the paper's Eq. (1)-(2).

Paper formulas:
  MAE_d  = mean_{s,h} |y - yhat|           (same as flattening seq × horizon)
  RMSE_d = mean_s( sqrt(mean_h(err^2)) )   NOT sqrt(mean_{s,h}(err^2))

Std is computed across DF series (features), not across sequences.
"""

import numpy as np
import pytest

from pmf_tsfm.utils.metrics import compute_metrics

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _manual_mae_per_feature(preds: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """MAE_d = mean over (seq, horizon) of |error|.  Shape: (num_features,)"""
    return np.mean(np.abs(preds - targets), axis=(0, 1))


def _manual_rmse_per_feature(preds: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """
    Paper Eq. (2): RMSE_d = mean_s( sqrt(mean_h(err^2)) ).
    Shape: (num_features,)
    """
    errors = preds - targets  # (S, H, D)
    seq_rmse = np.sqrt(np.mean(errors**2, axis=1))  # (S, D)  — RMSE over horizon
    return np.mean(seq_rmse, axis=0)  # (D,)    — mean over sequences


def _global_rmse_per_feature(preds: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """sqrt(mean_{s,h}(err^2)) — the naive 'flattened' RMSE, NOT the paper formula."""
    errors = preds - targets
    return np.sqrt(np.mean(errors**2, axis=(0, 1)))


# ---------------------------------------------------------------------------
# Core formula tests
# ---------------------------------------------------------------------------


class TestComputeMetricsFormulas:
    """Verify that compute_metrics implements the paper's Eq. (1)-(2) exactly."""

    def setup_method(self):
        """Shared synthetic data: (58, 7, 21) — mirrors BPI2017 shape."""
        rng = np.random.default_rng(42)
        self.preds = rng.standard_normal((58, 7, 21)).astype(np.float32)
        self.targets = rng.standard_normal((58, 7, 21)).astype(np.float32)
        self.result = compute_metrics(self.preds, self.targets)

    # --- MAE ---

    def test_mae_per_feature_matches_manual(self):
        expected = _manual_mae_per_feature(self.preds, self.targets)
        for i in range(21):
            key = f"feature_{i}"
            assert abs(self.result["per_feature"][key]["MAE"] - float(expected[i])) < 1e-5

    def test_mae_mean_is_mean_of_per_feature_maes(self):
        expected_mean = float(np.mean(_manual_mae_per_feature(self.preds, self.targets)))
        assert abs(self.result["summary"]["MAE_mean"] - expected_mean) < 1e-5

    def test_mae_std_is_std_of_per_feature_maes(self):
        expected_std = float(np.std(_manual_mae_per_feature(self.preds, self.targets)))
        assert abs(self.result["summary"]["MAE_std"] - expected_std) < 1e-5

    # --- RMSE (paper Eq. 2: mean of per-sequence RMSEs) ---

    def test_rmse_per_feature_matches_paper_formula(self):
        expected = _manual_rmse_per_feature(self.preds, self.targets)
        for i in range(21):
            key = f"feature_{i}"
            assert abs(self.result["per_feature"][key]["RMSE"] - float(expected[i])) < 1e-5

    def test_rmse_mean_is_mean_of_per_feature_rmses(self):
        expected_mean = float(np.mean(_manual_rmse_per_feature(self.preds, self.targets)))
        assert abs(self.result["summary"]["RMSE_mean"] - expected_mean) < 1e-5

    def test_rmse_std_is_std_of_per_feature_rmses(self):
        expected_std = float(np.std(_manual_rmse_per_feature(self.preds, self.targets)))
        assert abs(self.result["summary"]["RMSE_std"] - expected_std) < 1e-5

    def test_rmse_differs_from_naive_global_rmse(self):
        """
        Confirm the paper RMSE (mean of per-seq RMSEs) differs from the naive
        global sqrt(mean_all(err^2)).  They are NOT interchangeable.
        """
        paper_rmse = _manual_rmse_per_feature(self.preds, self.targets)
        naive_rmse = _global_rmse_per_feature(self.preds, self.targets)
        # They should differ for random data with horizon > 1
        assert not np.allclose(paper_rmse, naive_rmse), (
            "Paper RMSE and naive RMSE should differ — check formula implementation"
        )

    # --- Summary metadata ---

    def test_summary_shape_metadata(self):
        s = self.result["summary"]
        assert s["num_sequences"] == 58
        assert s["prediction_length"] == 7
        assert s["num_features"] == 21


class TestComputeMetricsPerfectPrediction:
    """When predictions == targets, MAE = RMSE = 0."""

    def test_zero_error(self):
        data = np.ones((10, 7, 5), dtype=np.float32)
        result = compute_metrics(data, data)
        assert result["summary"]["MAE_mean"] == pytest.approx(0.0)
        assert result["summary"]["RMSE_mean"] == pytest.approx(0.0)
        assert result["summary"]["MAE_std"] == pytest.approx(0.0)
        assert result["summary"]["RMSE_std"] == pytest.approx(0.0)

    def test_per_feature_zero_error(self):
        data = np.ones((10, 7, 3), dtype=np.float32)
        result = compute_metrics(data, data)
        for key in result["per_feature"]:
            assert result["per_feature"][key]["MAE"] == pytest.approx(0.0)
            assert result["per_feature"][key]["RMSE"] == pytest.approx(0.0)


class TestComputeMetricsConstantOffset:
    """Constant offset of c: MAE_d = c, RMSE_d = c for all d."""

    def test_constant_offset(self):
        c = 3.0
        targets = np.zeros((20, 7, 5), dtype=np.float32)
        preds = targets + c
        result = compute_metrics(preds, targets)
        assert result["summary"]["MAE_mean"] == pytest.approx(c)
        assert result["summary"]["RMSE_mean"] == pytest.approx(c)
        # All features have the same error → std = 0
        assert result["summary"]["MAE_std"] == pytest.approx(0.0)
        assert result["summary"]["RMSE_std"] == pytest.approx(0.0)


class TestComputeMetricsSingleFeature:
    """Edge case: single DF series (num_features=1)."""

    def test_single_feature_mae(self):
        preds = np.array([[[2.0], [3.0]], [[1.0], [4.0]]])  # (2, 2, 1)
        targets = np.array([[[1.0], [1.0]], [[1.0], [1.0]]])  # (2, 2, 1)
        result = compute_metrics(preds, targets)
        # errors: [[1,2],[0,3]] → abs: [[1,2],[0,3]] → mean = 1.5
        assert result["summary"]["MAE_mean"] == pytest.approx(1.5)

    def test_single_feature_rmse_paper_formula(self):
        preds = np.array([[[2.0], [3.0]], [[1.0], [4.0]]])  # (2, 2, 1)
        targets = np.array([[[1.0], [1.0]], [[1.0], [1.0]]])  # (2, 2, 1)
        result = compute_metrics(preds, targets)
        # seq 0: errors=[1,2] → RMSE=sqrt(mean([1,4]))=sqrt(2.5)
        # seq 1: errors=[0,3] → RMSE=sqrt(mean([0,9]))=sqrt(4.5)
        # RMSE_d = mean(sqrt(2.5), sqrt(4.5))
        expected = (np.sqrt(2.5) + np.sqrt(4.5)) / 2
        assert result["summary"]["RMSE_mean"] == pytest.approx(expected, rel=1e-5)

    def test_single_feature_std_is_zero(self):
        """Std across features is 0 when there is only one feature."""
        rng = np.random.default_rng(0)
        preds = rng.standard_normal((10, 7, 1)).astype(np.float32)
        targets = rng.standard_normal((10, 7, 1)).astype(np.float32)
        result = compute_metrics(preds, targets)
        assert result["summary"]["MAE_std"] == pytest.approx(0.0)
        assert result["summary"]["RMSE_std"] == pytest.approx(0.0)


class TestComputeMetricsFeatureNames:
    """Feature names propagate to per_feature dict correctly."""

    def test_custom_feature_names(self):
        rng = np.random.default_rng(1)
        preds = rng.standard_normal((5, 7, 3)).astype(np.float32)
        targets = rng.standard_normal((5, 7, 3)).astype(np.float32)
        names = ["A -> B", "B -> C", "C -> D"]
        result = compute_metrics(preds, targets, feature_names=names)
        assert set(result["per_feature"].keys()) == set(names)
        for key in names:
            assert "MAE" in result["per_feature"][key]
            assert "RMSE" in result["per_feature"][key]

    def test_default_feature_names(self):
        rng = np.random.default_rng(2)
        preds = rng.standard_normal((5, 7, 4)).astype(np.float32)
        targets = rng.standard_normal((5, 7, 4)).astype(np.float32)
        result = compute_metrics(preds, targets)
        assert set(result["per_feature"].keys()) == {
            "feature_0",
            "feature_1",
            "feature_2",
            "feature_3",
        }
