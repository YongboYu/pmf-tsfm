"""Tests for ZeroShotDataModule, TrainingDataModule, and the preprocess script.

Purpose
-------
Verify the data-preparation logic independently of any model.
All tests use small synthetic data (created in-memory) and run purely on
CPU with no internet access required.

What is tested
--------------
preprocess_dataset():
  - Creates train / val / test / full parquet files with correct row counts.
  - Saves metadata.json with correct split indices and checksums.
  - Skips re-processing when files already exist (force_overwrite=False).
  - Overwrites existing files when force_overwrite=True.

ZeroShotDataModule:
  - Loads a parquet file and reports correct shapes / metadata.
  - Splits the time series at the exact absolute indices (train_end, val_end).
  - Produces the right number of expanding-window sequences for each split.
  - Expanding window: each sequence i has length (val_end + i), not a fixed width.
  - Targets have the correct shape: (num_sequences, pred_len, num_features).
  - Loads from pre-split full.parquet + metadata.json when processed_dir is given.

TrainingDataModule:
  - Absolute split (train_end/val_end) takes priority over ratio split.
  - Ratio split works as a fallback when indices are not supplied.
  - Sliding-window Moirai dataset produces correct batch keys and shapes.
  - Sliding-window Chronos dataset produces correct batch keys and shapes.
  - DataLoaders are iterable and deliver tensors of the expected shape.
  - Loads train.parquet / val.parquet directly when processed_dir is given.

Running
-------
    uv run pytest tests/test_data_modules.py -v
"""

import pytest
import torch

# ===========================================================================
# ZeroShotDataModule
# ===========================================================================


class TestZeroShotDataModule:
    """Tests for the expanding-window zero-shot data preparation."""

    def test_setup_loads_data(self, synthetic_parquet):
        """setup() reads the parquet and reports total length + feature count."""
        from pmf_tsfm.data.zero_shot import ZeroShotDataModule

        path, _ = synthetic_parquet
        mod = ZeroShotDataModule(
            dataset_name="synthetic",
            data_path=str(path),
            prediction_length=7,
            train_end=90,
            val_end=120,
        )
        mod.setup()

        assert mod.data is not None
        assert len(mod.data) == 150
        assert len(mod.feature_names) == 6
        assert mod.metadata["total_length"] == 150

    def test_absolute_split_metadata(self, synthetic_parquet):
        """Metadata reflects the exact absolute split boundaries."""
        from pmf_tsfm.data.zero_shot import ZeroShotDataModule

        path, _ = synthetic_parquet
        mod = ZeroShotDataModule(
            dataset_name="synthetic",
            data_path=str(path),
            prediction_length=7,
            train_end=90,
            val_end=120,
        )
        mod.setup()

        splits = mod.metadata["split_lengths"]
        assert splits["train"] == 90
        assert splits["val"] == 30  # 120 - 90
        assert splits["test"] == 30  # 150 - 120

    def test_test_split_sequence_count(self, synthetic_parquet):
        """Test split: sequences start at val_end and end at total_length - pred_len."""
        from pmf_tsfm.data.zero_shot import ZeroShotDataModule

        path, _ = synthetic_parquet
        pred_len = 7
        val_end = 120
        total = 150

        mod = ZeroShotDataModule(
            dataset_name="synthetic",
            data_path=str(path),
            prediction_length=pred_len,
            train_end=90,
            val_end=val_end,
        )
        prepared = mod.prepare_data_for_model(split="test")

        # target_start = val_end = 120, target_end = total - pred_len = 143
        # sequences: range(120, 143+1) → 24 sequences
        expected_num_seq = (total - pred_len) - val_end + 1
        assert len(prepared["inputs"]) == expected_num_seq
        assert prepared["targets"].shape == (expected_num_seq, pred_len, 6)

    def test_expanding_window_grows(self, synthetic_parquet):
        """Each successive input sequence is one step longer (expanding window)."""
        from pmf_tsfm.data.zero_shot import ZeroShotDataModule

        path, _ = synthetic_parquet
        mod = ZeroShotDataModule(
            dataset_name="synthetic",
            data_path=str(path),
            prediction_length=7,
            train_end=90,
            val_end=120,
        )
        prepared = mod.prepare_data_for_model(split="test")

        inputs = prepared["inputs"]
        for i in range(len(inputs) - 1):
            assert inputs[i + 1].shape[0] == inputs[i].shape[0] + 1, (
                "Each consecutive input must be exactly one timestep longer"
            )

    def test_first_test_input_starts_from_zero(self, synthetic_parquet):
        """The first test input covers all data from timestep 0 to val_end (exclusive)."""
        from pmf_tsfm.data.zero_shot import ZeroShotDataModule

        path, _ = synthetic_parquet
        val_end = 120
        mod = ZeroShotDataModule(
            dataset_name="synthetic",
            data_path=str(path),
            prediction_length=7,
            train_end=90,
            val_end=val_end,
        )
        prepared = mod.prepare_data_for_model(split="test")

        first_input = prepared["inputs"][0]
        assert first_input.shape[0] == val_end  # rows = val_end
        assert first_input.shape[1] == 6  # columns = num_features

    def test_feature_names_returned(self, synthetic_parquet):
        """prepare_data_for_model includes feature names from the parquet columns."""
        from pmf_tsfm.data.zero_shot import ZeroShotDataModule

        path, df = synthetic_parquet
        mod = ZeroShotDataModule(
            dataset_name="synthetic",
            data_path=str(path),
            prediction_length=7,
            train_end=90,
            val_end=120,
        )
        prepared = mod.prepare_data_for_model(split="test")

        assert prepared["feature_names"] == list(df.columns)

    def test_from_config(self, synthetic_parquet):
        """from_config() factory reads train_end / val_end from a DictConfig."""
        from omegaconf import OmegaConf

        from pmf_tsfm.data.zero_shot import ZeroShotDataModule

        path, _ = synthetic_parquet
        cfg = OmegaConf.create(
            {
                "name": "synthetic",
                "path": str(path),
                "train_end": 90,
                "val_end": 120,
            }
        )
        mod = ZeroShotDataModule.from_config(cfg, prediction_length=7)
        mod.setup()

        assert mod.train_end == 90
        assert mod.val_end == 120

    def test_resolves_lowercase_zenodo_named_parquet(self, synthetic_parquet, tmp_path):
        """Canonical config paths still load Zenodo-style lowercase parquet assets."""
        from pmf_tsfm.data.zero_shot import ZeroShotDataModule

        raw_path, _ = synthetic_parquet
        lowercase_path = tmp_path / "bpi2017.parquet"
        lowercase_path.write_bytes(raw_path.read_bytes())

        mod = ZeroShotDataModule(
            dataset_name="BPI2017",
            data_path=str(tmp_path / "BPI2017.parquet"),
            prediction_length=7,
            train_end=90,
            val_end=120,
        )
        mod.setup()

        assert mod.data_path == lowercase_path
        assert mod.data is not None


# ===========================================================================
# TrainingDataModule
# ===========================================================================


class TestTrainingDataModule:
    """Tests for the sliding-window training data preparation."""

    def test_absolute_split_preferred(self, tiny_parquet):
        """When train_end/val_end are supplied they override ratio-based split."""
        from pmf_tsfm.data.training import TrainingDataModule

        path, _ = tiny_parquet
        mod = TrainingDataModule(
            dataset_name="tiny",
            data_path=str(path),
            context_length=5,
            prediction_length=3,
            train_end=30,
            val_end=40,
            model_family="chronos",
        )
        mod.setup()

        assert len(mod.train_data) == 30
        assert len(mod.val_data) == 10
        assert len(mod.test_data) == 10

    def test_ratio_split_fallback(self, tiny_parquet):
        """Without absolute indices, split is computed from ratio."""
        from pmf_tsfm.data.training import TrainingDataModule

        path, _ = tiny_parquet
        mod = TrainingDataModule(
            dataset_name="tiny",
            data_path=str(path),
            context_length=5,
            prediction_length=3,
            # No train_end / val_end
            train_val_test_ratio=[0.6, 0.2, 0.2],
            model_family="chronos",
        )
        mod.setup()

        # 50 samples x 0.6 = 30
        assert len(mod.train_data) == 30
        assert len(mod.val_data) == 10
        assert len(mod.test_data) == 10

    def test_resolves_lowercase_zenodo_named_parquet(self, tiny_parquet, tmp_path):
        """Runtime training split accepts lowercase parquet names from Zenodo bundles."""
        from pmf_tsfm.data.training import TrainingDataModule

        raw_path, _ = tiny_parquet
        lowercase_path = tmp_path / "hospital_billing.parquet"
        lowercase_path.write_bytes(raw_path.read_bytes())

        mod = TrainingDataModule(
            dataset_name="Hospital_Billing",
            data_path=str(tmp_path / "Hospital_Billing.parquet"),
            context_length=5,
            prediction_length=3,
            train_end=30,
            val_end=40,
            model_family="chronos",
        )
        mod.setup()

        assert mod.data_path == lowercase_path
        assert len(mod.train_data) == 30

    def test_chronos_dataset_shapes(self, tiny_parquet):
        """ChronosTrainingDataset returns tensors with the expected shapes."""
        from pmf_tsfm.data.training import TrainingDataModule

        path, _ = tiny_parquet
        context_length = 8
        prediction_length = 3

        mod = TrainingDataModule(
            dataset_name="tiny",
            data_path=str(path),
            context_length=context_length,
            prediction_length=prediction_length,
            train_end=30,
            val_end=40,
            model_family="chronos",
        )
        mod.setup()

        loader = mod.get_train_dataloader(batch_size=4, num_workers=0)
        batch = next(iter(loader))

        # Chronos format: context (B, ctx_len), target (B, pred_len)
        assert batch["context"].shape == (4, context_length)
        assert batch["target"].shape == (4, prediction_length)
        assert batch["context"].dtype == torch.float32
        assert batch["target"].dtype == torch.float32

    def test_moirai_dataset_shapes(self, tiny_parquet):
        """MoiraiTrainingDataset returns tensors with the expected keys and shapes."""
        from pmf_tsfm.data.training import TrainingDataModule

        path, _ = tiny_parquet
        context_length = 8
        prediction_length = 3

        mod = TrainingDataModule(
            dataset_name="tiny",
            data_path=str(path),
            context_length=context_length,
            prediction_length=prediction_length,
            train_end=30,
            val_end=40,
            model_family="moirai",
        )
        mod.setup()

        loader = mod.get_train_dataloader(batch_size=4, num_workers=0)
        batch = next(iter(loader))

        # Moirai format: (B, ctx_len, 1) for targets, (B, ctx_len) for masks
        assert "past_target" in batch
        assert "past_observed_target" in batch
        assert "past_is_pad" in batch
        assert "future_target" in batch

        assert batch["past_target"].shape == (4, context_length, 1)
        assert batch["future_target"].shape == (4, prediction_length, 1)
        assert batch["past_is_pad"].shape == (4, context_length)

    def test_val_dataloader_no_shuffle(self, tiny_parquet):
        """Validation DataLoader must be deterministic (no shuffle)."""
        from pmf_tsfm.data.training import TrainingDataModule

        path, _ = tiny_parquet
        mod = TrainingDataModule(
            dataset_name="tiny",
            data_path=str(path),
            context_length=5,
            prediction_length=3,
            train_end=30,
            val_end=40,
            model_family="chronos",
        )
        mod.setup()

        loader = mod.get_val_dataloader(batch_size=4, num_workers=0)
        batch1 = next(iter(loader))
        batch2 = next(iter(loader))

        # First batch should be identical across two iterations (no shuffle)
        assert torch.allclose(batch1["context"], batch2["context"])

    def test_sliding_window_sample_count(self, tiny_parquet):
        """Number of training samples = (train_len - ctx - pred + 1) × num_features."""
        from pmf_tsfm.data.training import (
            ChronosTrainingDataset,
            TrainingDataModule,
        )

        path, _ = tiny_parquet
        context_length = 5
        prediction_length = 3
        train_len = 30
        n_features = 3

        mod = TrainingDataModule(
            dataset_name="tiny",
            data_path=str(path),
            context_length=context_length,
            prediction_length=prediction_length,
            train_end=train_len,
            val_end=40,
            model_family="chronos",
        )
        mod.setup()

        ds = ChronosTrainingDataset(
            data=mod.train_data,
            context_length=context_length,
            prediction_length=prediction_length,
        )
        expected_sliding = max(0, train_len - context_length - prediction_length + 1)
        assert len(ds) == expected_sliding * n_features

    def test_get_metadata(self, tiny_parquet):
        """get_metadata() returns correct split information after setup()."""
        from pmf_tsfm.data.training import TrainingDataModule

        path, _ = tiny_parquet
        mod = TrainingDataModule(
            dataset_name="tiny",
            data_path=str(path),
            context_length=5,
            prediction_length=3,
            train_end=30,
            val_end=40,
            model_family="moirai",
        )
        mod.setup()
        meta = mod.get_metadata()

        assert meta["train_samples"] == 30
        assert meta["val_samples"] == 10
        assert meta["test_samples"] == 10
        assert meta["train_end"] == 30
        assert meta["val_end"] == 40
        assert meta["model_family"] == "moirai"


# ===========================================================================
# Preprocessing script
# ===========================================================================


class TestPreprocessDataset:
    """Tests for preprocess_dataset() — the split-and-save function."""

    def test_creates_four_parquet_files(self, synthetic_parquet, tmp_path):
        """preprocess_dataset() must save full/train/val/test parquets."""
        from pmf_tsfm.data.preprocess import preprocess_dataset

        path, _ = synthetic_parquet
        preprocess_dataset(
            raw_path=path,
            dataset_name="synthetic",
            out_dir=tmp_path,
            train_end=90,
            val_end=120,
            split_ratio=[0.6, 0.2, 0.2],
        )

        split_dir = tmp_path / "synthetic"
        for fname in ["full.parquet", "train.parquet", "val.parquet", "test.parquet"]:
            assert (split_dir / fname).exists(), f"Expected {fname} in {split_dir}"

    def test_split_row_counts(self, synthetic_parquet, tmp_path):
        """Each split file contains the correct number of rows."""
        import pandas as pd

        from pmf_tsfm.data.preprocess import preprocess_dataset

        path, _ = synthetic_parquet  # 150 rows, 6 features
        preprocess_dataset(
            raw_path=path,
            dataset_name="synthetic",
            out_dir=tmp_path,
            train_end=90,
            val_end=120,
            split_ratio=[0.6, 0.2, 0.2],
        )

        split_dir = tmp_path / "synthetic"
        assert len(pd.read_parquet(split_dir / "full.parquet")) == 150
        assert len(pd.read_parquet(split_dir / "train.parquet")) == 90
        assert len(pd.read_parquet(split_dir / "val.parquet")) == 30
        assert len(pd.read_parquet(split_dir / "test.parquet")) == 30

    def test_metadata_json_content(self, synthetic_parquet, tmp_path):
        """metadata.json records split indices, feature names, and checksums."""
        import json

        from pmf_tsfm.data.preprocess import preprocess_dataset

        path, df = synthetic_parquet
        preprocess_dataset(
            raw_path=path,
            dataset_name="synthetic",
            out_dir=tmp_path,
            train_end=90,
            val_end=120,
            split_ratio=[0.6, 0.2, 0.2],
        )

        with open(tmp_path / "synthetic" / "metadata.json") as f:
            meta = json.load(f)

        assert meta["split"]["train_end"] == 90
        assert meta["split"]["val_end"] == 120
        assert meta["split"]["train_size"] == 90
        assert meta["split"]["val_size"] == 30
        assert meta["split"]["test_size"] == 30
        assert meta["feature_names"] == list(df.columns)
        assert "checksums" in meta
        assert "full.parquet" in meta["checksums"]

    def test_skips_existing_without_force(self, synthetic_parquet, tmp_path):
        """Re-running without force_overwrite skips processing."""
        from pmf_tsfm.data.preprocess import preprocess_dataset

        path, _ = synthetic_parquet
        kwargs = {
            "raw_path": path,
            "dataset_name": "synthetic",
            "out_dir": tmp_path,
            "train_end": 90,
            "val_end": 120,
            "split_ratio": [0.6, 0.2, 0.2],
        }
        preprocess_dataset(**kwargs)  # First run

        # Corrupt the train.parquet to detect a re-run
        train_path = tmp_path / "synthetic" / "train.parquet"
        original_size = train_path.stat().st_size
        train_path.write_bytes(b"corrupted")

        preprocess_dataset(**kwargs, force_overwrite=False)  # Should skip

        # File still corrupted (skipped)
        assert train_path.stat().st_size != original_size

    def test_force_overwrite_regenerates(self, synthetic_parquet, tmp_path):
        """force_overwrite=True regenerates files even when they exist."""
        import pandas as pd

        from pmf_tsfm.data.preprocess import preprocess_dataset

        path, _ = synthetic_parquet
        kwargs = {
            "raw_path": path,
            "dataset_name": "synthetic",
            "out_dir": tmp_path,
            "train_end": 90,
            "val_end": 120,
            "split_ratio": [0.6, 0.2, 0.2],
        }
        preprocess_dataset(**kwargs)

        # Corrupt train.parquet
        train_path = tmp_path / "synthetic" / "train.parquet"
        train_path.write_bytes(b"corrupted")

        preprocess_dataset(**kwargs, force_overwrite=True)

        # File regenerated — readable again
        df = pd.read_parquet(train_path)
        assert len(df) == 90


# ===========================================================================
# Pre-split loading path
# ===========================================================================


@pytest.fixture
def preprocessed_dir(synthetic_parquet, tmp_path):
    """Run preprocess_dataset and return the processed root directory."""
    from pmf_tsfm.data.preprocess import preprocess_dataset

    path, _ = synthetic_parquet
    preprocess_dataset(
        raw_path=path,
        dataset_name="synthetic",
        out_dir=tmp_path,
        train_end=90,
        val_end=120,
        split_ratio=[0.6, 0.2, 0.2],
    )
    return tmp_path


class TestZeroShotFromProcessedDir:
    """ZeroShotDataModule loads from pre-split files when processed_dir is set."""

    def test_loads_full_parquet(self, synthetic_parquet, preprocessed_dir):
        """setup() reads full.parquet from the processed dir."""
        from pmf_tsfm.data.zero_shot import ZeroShotDataModule

        path, _ = synthetic_parquet
        mod = ZeroShotDataModule(
            dataset_name="synthetic",
            data_path=str(path),
            prediction_length=7,
            train_end=90,  # will be overridden by metadata.json
            val_end=120,
            processed_dir=str(preprocessed_dir),
        )
        mod.setup()

        assert mod.data is not None
        assert len(mod.data) == 150
        assert mod.train_end == 90
        assert mod.val_end == 120

    def test_split_indices_from_metadata(self, synthetic_parquet, preprocessed_dir):
        """Split indices are read from metadata.json, overriding constructor args."""
        from pmf_tsfm.data.zero_shot import ZeroShotDataModule

        path, _ = synthetic_parquet
        mod = ZeroShotDataModule(
            dataset_name="synthetic",
            data_path=str(path),
            prediction_length=7,
            train_end=999,  # wrong — should be overridden by metadata.json
            val_end=999,
            processed_dir=str(preprocessed_dir),
        )
        mod.setup()

        assert mod.train_end == 90
        assert mod.val_end == 120

    def test_fallback_to_raw_when_no_processed(self, synthetic_parquet, tmp_path):
        """Falls back to raw parquet when processed dir does not contain the dataset."""
        from pmf_tsfm.data.zero_shot import ZeroShotDataModule

        path, _ = synthetic_parquet
        mod = ZeroShotDataModule(
            dataset_name="synthetic",
            data_path=str(path),
            prediction_length=7,
            train_end=90,
            val_end=120,
            processed_dir=str(tmp_path / "nonexistent"),
        )
        mod.setup()  # must not raise; falls back to data_path

        assert mod.data is not None
        assert len(mod.data) == 150


class TestTrainingFromProcessedDir:
    """TrainingDataModule loads train.parquet/val.parquet from processed dir."""

    def test_loads_presplit_files(self, synthetic_parquet, preprocessed_dir):
        """setup() reads train/val parquets directly when processed_dir is set."""
        from pmf_tsfm.data.training import TrainingDataModule

        path, _ = synthetic_parquet
        mod = TrainingDataModule(
            dataset_name="synthetic",
            data_path=str(path),
            context_length=10,
            prediction_length=7,
            model_family="chronos",
            processed_dir=str(preprocessed_dir),
        )
        mod.setup()

        # Sizes must match what preprocess_dataset saved
        assert len(mod.train_data) == 90
        assert len(mod.val_data) == 30
        assert len(mod.test_data) == 30

    def test_split_indices_from_metadata_in_training(self, synthetic_parquet, preprocessed_dir):
        """get_metadata() returns correct indices from the pre-split metadata.json."""
        from pmf_tsfm.data.training import TrainingDataModule

        path, _ = synthetic_parquet
        mod = TrainingDataModule(
            dataset_name="synthetic",
            data_path=str(path),
            context_length=10,
            prediction_length=7,
            model_family="moirai",
            processed_dir=str(preprocessed_dir),
        )
        mod.setup()
        meta = mod.get_metadata()

        assert meta["train_end"] == 90
        assert meta["val_end"] == 120

    def test_consistency_zero_shot_and_training(self, synthetic_parquet, preprocessed_dir):
        """ZeroShotDataModule and TrainingDataModule use the same split indices."""
        from pmf_tsfm.data.training import TrainingDataModule
        from pmf_tsfm.data.zero_shot import ZeroShotDataModule

        path, _ = synthetic_parquet

        zs = ZeroShotDataModule(
            dataset_name="synthetic",
            data_path=str(path),
            prediction_length=7,
            train_end=0,  # will be overridden
            val_end=0,
            processed_dir=str(preprocessed_dir),
        )
        zs.setup()

        tr = TrainingDataModule(
            dataset_name="synthetic",
            data_path=str(path),
            context_length=10,
            prediction_length=7,
            model_family="chronos",
            processed_dir=str(preprocessed_dir),
        )
        tr.setup()

        assert zs.train_end == tr._train_end_resolved
        assert zs.val_end == tr._val_end_resolved
