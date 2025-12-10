"""Pytest configuration and fixtures."""

import numpy as np
import pytest


@pytest.fixture
def sample_time_series():
    """Create a sample time series for testing."""
    np.random.seed(42)
    return np.random.randn(100, 5)  # 100 time steps, 5 features


@pytest.fixture
def sample_config():
    """Create a sample model configuration."""
    return {
        "model_name": "test_model",
        "variant": "small",
        "context_length": 64,
        "prediction_length": 10,
        "device": "cpu",
    }

