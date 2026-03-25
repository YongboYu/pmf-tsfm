"""Pytest configuration and shared fixtures.

Fixtures defined here are automatically available to every test file
without any import — pytest discovers conftest.py automatically.

Fixtures used across multiple test files live here.
Model-specific helpers (e.g., MockAdapter) live in the test file that needs them.
"""

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Synthetic time-series data
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_time_series():
    """100-step, 5-feature random time series (NumPy array).

    Used by lightweight unit tests that don't need a parquet file.
    """
    np.random.seed(42)
    return np.random.randn(100, 5)


@pytest.fixture
def synthetic_parquet(tmp_path):
    """Small synthetic parquet file that mimics a real PMF dataset.

    Returns (path, dataframe) so tests can verify both file and contents.

    Shape: 150 timesteps × 6 features.
    Splits (matching the 60/20/20 paper ratio):
        train [0, 90)   → 90 rows
        val   [90, 120) → 30 rows
        test  [120, 150)→ 30 rows
    """
    n_timesteps = 150
    n_features = 6
    np.random.seed(42)

    # Simulate slightly realistic data: trend + noise per feature
    t = np.arange(n_timesteps)
    data = {
        f"feat_{i}": np.maximum(0, np.sin(t / 10 + i) * 5 + np.random.randn(n_timesteps) * 2)
        for i in range(n_features)
    }
    df = pd.DataFrame(data)

    path = tmp_path / "synthetic.parquet"
    df.to_parquet(path, index=False)
    return path, df


@pytest.fixture
def tiny_parquet(tmp_path):
    """Minimal 50-step parquet for fast training-loop tests.

    Splits:  train [0, 30)  val [30, 40)  test [40, 50)
    """
    n_timesteps = 50
    n_features = 3
    np.random.seed(0)
    df = pd.DataFrame(
        np.random.randn(n_timesteps, n_features).clip(0),
        columns=[f"df_{i}" for i in range(n_features)],
    )
    path = tmp_path / "tiny.parquet"
    df.to_parquet(path, index=False)
    return path, df


@pytest.fixture
def sample_config():
    """Legacy fixture kept for backward compatibility with existing tests."""
    return {
        "model_name": "test_model",
        "variant": "small",
        "context_length": 64,
        "prediction_length": 10,
        "device": "cpu",
    }
