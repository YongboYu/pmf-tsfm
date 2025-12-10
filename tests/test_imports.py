"""Basic import tests to verify package structure."""

import pytest


def test_import_pmf_tsfm():
    """Test that main package can be imported."""
    import pmf_tsfm

    assert pmf_tsfm is not None


def test_import_models():
    """Test that models subpackage can be imported."""
    from pmf_tsfm import models

    assert models is not None


def test_import_data():
    """Test that data subpackage can be imported."""
    from pmf_tsfm import data

    assert data is not None


def test_import_utils():
    """Test that utils subpackage can be imported."""
    from pmf_tsfm import utils

    assert utils is not None


class TestBaseAdapter:
    """Tests for BaseAdapter class."""

    def test_base_adapter_is_abstract(self):
        """Test that BaseAdapter cannot be instantiated directly."""
        from pmf_tsfm.models.base import BaseAdapter

        with pytest.raises(TypeError):
            BaseAdapter({})  # type: ignore

