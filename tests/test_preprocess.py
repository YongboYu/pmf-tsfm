"""Tests for data preprocessing helpers and entrypoint branches."""

from pathlib import Path

import pandas as pd
import pytest
from omegaconf import OmegaConf

import pmf_tsfm.data.preprocess as preprocess_module
from pmf_tsfm.data.assets import resolve_dataset_asset_path
from pmf_tsfm.data.preprocess import _clean, compute_split_indices


class TestClean:
    def test_drops_only_exact_time_like_auxiliary_columns(self) -> None:
        df = pd.DataFrame(
            {
                "feat_0": [1.0, 2.0],
                "timestamp": [10, 20],
                "DATE": [30, 40],
                "Time": [50, 60],
                "event_time": [70, 80],
            }
        )

        cleaned = _clean(df)

        assert list(cleaned.columns) == ["feat_0", "event_time"]


class TestComputeSplitIndices:
    def test_prefers_explicit_indices_when_provided(self) -> None:
        assert compute_split_indices(100, 61, 83, [0.6, 0.2, 0.2]) == (61, 83)

    def test_floors_ratio_based_fallback_indices(self) -> None:
        assert compute_split_indices(11, None, None, [0.6, 0.2, 0.2]) == (6, 8)


class TestResolveDatasetAssetPath:
    def test_prefers_existing_canonical_path(self, tmp_path: Path) -> None:
        canonical = tmp_path / "Sepsis.parquet"
        canonical.write_text("placeholder")

        resolved = resolve_dataset_asset_path(
            canonical,
            dataset_name="Sepsis",
            asset_label="raw parquet",
        )

        assert resolved == canonical

    def test_falls_back_to_lowercase_filename(self, tmp_path: Path) -> None:
        lowercase = tmp_path / "sepsis.parquet"
        lowercase.write_text("placeholder")

        resolved = resolve_dataset_asset_path(
            tmp_path / "Sepsis.parquet",
            dataset_name="Sepsis",
            asset_label="raw parquet",
        )

        assert resolved == lowercase


class TestPreprocessMain:
    def test_raises_for_missing_raw_parquet(self, tmp_path: Path) -> None:
        cfg = OmegaConf.create(
            {
                "print_config": False,
                "processed_dir": str(tmp_path / "processed"),
                "data": {
                    "name": "MissingDataset",
                    "path": str(tmp_path / "missing.parquet"),
                    "split_ratio": [0.6, 0.2, 0.2],
                },
            }
        )

        with pytest.raises(FileNotFoundError, match="Raw data file not found"):
            preprocess_module.main.__wrapped__(cfg)

    def test_print_config_main_returns_metadata_and_writes_outputs(
        self, synthetic_parquet, tmp_path: Path, capsys
    ) -> None:
        raw_path, _ = synthetic_parquet
        cfg = OmegaConf.create(
            {
                "print_config": True,
                "processed_dir": str(tmp_path / "processed"),
                "force_overwrite": False,
                "data": {
                    "name": "Synthetic",
                    "path": str(raw_path),
                    "split_ratio": [0.6, 0.2, 0.2],
                },
            }
        )

        metadata = preprocess_module.main.__wrapped__(cfg)

        split_dir = tmp_path / "processed" / "Synthetic"
        assert metadata["dataset_name"] == "Synthetic"
        assert (split_dir / "full.parquet").exists()
        assert (split_dir / "train.parquet").exists()
        assert (split_dir / "val.parquet").exists()
        assert (split_dir / "test.parquet").exists()
        assert (split_dir / "metadata.json").exists()
        assert "print_config: true" in capsys.readouterr().out.lower()

    def test_resolves_lowercase_zenodo_named_raw_parquet(
        self, synthetic_parquet, tmp_path: Path
    ) -> None:
        raw_path, _ = synthetic_parquet
        lowercase_path = tmp_path / "bpi2017.parquet"
        lowercase_path.write_bytes(raw_path.read_bytes())

        cfg = OmegaConf.create(
            {
                "print_config": False,
                "processed_dir": str(tmp_path / "processed"),
                "force_overwrite": False,
                "data": {
                    "name": "BPI2017",
                    "path": str(tmp_path / "BPI2017.parquet"),
                    "split_ratio": [0.6, 0.2, 0.2],
                },
            }
        )

        metadata = preprocess_module.main.__wrapped__(cfg)

        split_dir = tmp_path / "processed" / "BPI2017"
        assert metadata["dataset_name"] == "BPI2017"
        assert (split_dir / "full.parquet").exists()
