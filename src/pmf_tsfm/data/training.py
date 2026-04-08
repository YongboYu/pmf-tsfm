"""Training data module for LoRA and full fine-tuning.

Data format (paper Section 4.1)
--------------------------------
Fine-tuning training uses a SLIDING window of fixed context_length (48 days).
Every consecutive (context_length + prediction_length) sub-sequence of the
training split becomes one training sample.  Each univariate feature is
treated as a separate sample, so:

    num_samples = (train_len - context_length - prediction_length + 1)
                  × num_features

The sliding window is implemented directly in the PyTorch Dataset classes
below and created on-the-fly from the parquet file (or pre-split parquet
if a processed_dir is provided).

Pre-split loading
-----------------
If `processed_dir` is provided, the module loads:
    {processed_dir}/{dataset_name}/train.parquet   — fine-tuning training
    {processed_dir}/{dataset_name}/val.parquet     — fine-tuning validation
    {processed_dir}/{dataset_name}/test.parquet    — held-out (not used here)

This avoids re-splitting at runtime and enables exact split reproducibility.
Run `python -m pmf_tsfm.data.preprocess data=<dataset>` once to create them.

DataLoader efficiency
---------------------
- `num_workers`: 0 on macOS/MPS (multiprocessing conflicts with MPS context),
  ≥2 on Linux/CUDA.  Pass via task config or override on CLI.
- `pin_memory`: enabled only when CUDA is available (no benefit and warning on MPS).
- `persistent_workers`: enabled on Linux when num_workers > 0 to amortize
  worker startup cost across epochs.
"""

import json
from pathlib import Path
from typing import Any, Literal, TypeAlias

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from pmf_tsfm.data.assets import resolve_dataset_asset_path

# ---------------------------------------------------------------------------
# PyTorch Datasets
# ---------------------------------------------------------------------------


class MoiraiTrainingDataset(Dataset):
    """Sliding-window Dataset for Moirai fine-tuning.

    Each item is a univariate sub-sequence of `context_length + prediction_length`
    rows, formatted for Moirai's past_target / future_target interface.

    The multivariate data is decomposed into independent univariate streams:
    a (T × F) matrix with num_samples sliding positions yields
    num_samples × F training items.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        context_length: int,
        prediction_length: int,
    ):
        self.data = data.values.astype(np.float32)
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.seq_length = context_length + prediction_length
        self.num_features = int(data.shape[1])
        self.num_samples: int = max(0, len(self.data) - self.seq_length + 1)

    def __len__(self) -> int:
        return self.num_samples * self.num_features

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        seq_idx = idx // self.num_features
        feature_idx = idx % self.num_features

        start = seq_idx
        context_end = start + self.context_length
        target_end = context_end + self.prediction_length

        past_target = self.data[start:context_end, feature_idx]
        future_target = self.data[context_end:target_end, feature_idx]

        return {
            "past_target": torch.from_numpy(past_target).unsqueeze(-1),  # (ctx, 1)
            "past_observed_target": torch.ones(self.context_length, 1),  # (ctx, 1)
            "past_is_pad": torch.zeros(self.context_length, dtype=torch.bool),  # (ctx,)
            "future_target": torch.from_numpy(future_target).unsqueeze(-1),  # (pred, 1)
        }


class ChronosTrainingDataset(Dataset):
    """Sliding-window Dataset for Chronos fine-tuning.

    Same sliding-window logic as MoiraiTrainingDataset but outputs the
    flat context / target format expected by Chronos models.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        context_length: int,
        prediction_length: int,
    ):
        self.data = data.values.astype(np.float32)
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.seq_length = context_length + prediction_length
        self.num_features = int(data.shape[1])
        self.num_samples: int = max(0, len(self.data) - self.seq_length + 1)

    def __len__(self) -> int:
        return self.num_samples * self.num_features

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        seq_idx = idx // self.num_features
        feature_idx = idx % self.num_features

        start = seq_idx
        context_end = start + self.context_length
        target_end = context_end + self.prediction_length

        context = self.data[start:context_end, feature_idx]
        target = self.data[context_end:target_end, feature_idx]

        return {
            "context": torch.from_numpy(context),  # (context_length,)
            "target": torch.from_numpy(target),  # (prediction_length,)
        }


TrainingDatasetClass: TypeAlias = type[MoiraiTrainingDataset] | type[ChronosTrainingDataset]


# ---------------------------------------------------------------------------
# Collate functions
# ---------------------------------------------------------------------------


def moirai_collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Stack Moirai samples into batched tensors."""
    return {
        "past_target": torch.stack([s["past_target"] for s in batch]),
        "past_observed_target": torch.stack([s["past_observed_target"] for s in batch]),
        "past_is_pad": torch.stack([s["past_is_pad"] for s in batch]),
        "future_target": torch.stack([s["future_target"] for s in batch]),
    }


def chronos_collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Stack Chronos samples into batched tensors."""
    return {
        "context": torch.stack([s["context"] for s in batch]),
        "target": torch.stack([s["target"] for s in batch]),
    }


# ---------------------------------------------------------------------------
# Data module
# ---------------------------------------------------------------------------


class TrainingDataModule:
    """Data module for fine-tuning with train/val/test splits.

    Supports two loading modes (priority order):

    1. Pre-split files (recommended):
       Provide `processed_dir` pointing to the output of
       `python -m pmf_tsfm.data.preprocess`.  The module loads
       `{processed_dir}/{dataset_name}/train.parquet` etc. directly —
       no runtime splitting, guaranteed reproducibility.

    2. Runtime split from raw parquet:
       Falls back to loading `data_path` and splitting at runtime using
       absolute indices (train_end / val_end) or a ratio.

    Split priority within mode 2:
      a. Absolute indices (train_end, val_end) — consistent with ZeroShotDataModule.
      b. Ratio-based split (train_val_test_ratio) — when indices absent.

    Paper setting: 60% train / 20% val / 20% test (Section 4.1).
    """

    def __init__(
        self,
        dataset_name: str,
        data_path: str,
        context_length: int,
        prediction_length: int,
        train_end: int | None = None,
        val_end: int | None = None,
        train_val_test_ratio: list[float] | None = None,
        model_family: Literal["moirai", "chronos"] = "moirai",
        processed_dir: str | None = None,
    ):
        """
        Args:
            dataset_name:          Dataset identifier (must match preprocessed subdir name).
            data_path:             Path to raw parquet (used only in runtime-split mode).
            context_length:        Sliding-window context length for fine-tuning.
            prediction_length:     Forecasting horizon.
            train_end:             Absolute end index of training split (exclusive).
            val_end:               Absolute end index of validation split (exclusive).
            train_val_test_ratio:  Fallback ratio split [train, val, test].
            model_family:          "moirai" or "chronos" — determines batch format.
            processed_dir:         Root dir of pre-split files.  If the split files
                                   exist here they take priority over runtime splitting.
        """
        self.dataset_name = dataset_name
        self.data_path = Path(data_path)
        self.context_length = context_length
        self.prediction_length = prediction_length
        self._train_end = train_end
        self._val_end = val_end
        self.train_val_test_ratio = train_val_test_ratio or [0.6, 0.2, 0.2]
        self.model_family = model_family
        self._processed_dir = Path(processed_dir) / dataset_name if processed_dir else None

        self.data: pd.DataFrame | None = None
        self.train_data: pd.DataFrame | None = None
        self.val_data: pd.DataFrame | None = None
        self.test_data: pd.DataFrame | None = None
        self.feature_names: list[str] = []

        # Resolved after setup()
        self._train_end_resolved: int = 0
        self._val_end_resolved: int = 0

    def _try_load_from_processed(self) -> bool:
        """Try to load pre-split parquet files.  Returns True if successful."""
        if self._processed_dir is None:
            return False

        train_file = self._processed_dir / "train.parquet"
        val_file = self._processed_dir / "val.parquet"
        test_file = self._processed_dir / "test.parquet"
        meta_file = self._processed_dir / "metadata.json"

        if not (train_file.exists() and val_file.exists()):
            return False

        self.train_data = pd.read_parquet(train_file)
        self.val_data = pd.read_parquet(val_file)
        self.test_data = pd.read_parquet(test_file) if test_file.exists() else None
        self.feature_names = list(self.train_data.columns)

        # Load split indices from metadata
        if meta_file.exists():
            with open(meta_file) as f:
                meta = json.load(f)
            self._train_end_resolved = meta["split"]["train_end"]
            self._val_end_resolved = meta["split"]["val_end"]
        else:
            self._train_end_resolved = len(self.train_data)
            self._val_end_resolved = self._train_end_resolved + len(self.val_data)

        n_train = len(self.train_data)
        n_val = len(self.val_data)
        n_test = len(self.test_data) if self.test_data is not None else 0

        print(f"Data loaded (pre-split): {self.dataset_name}")
        print(f"  Source : {self._processed_dir}")
        print(f"  Features: {len(self.feature_names)}")
        print(f"  Train: {n_train}, Val: {n_val}, Test: {n_test}")
        print(f"  Model family: {self.model_family}")
        return True

    def setup(self) -> None:
        """Load and (if necessary) split the data."""
        if self._try_load_from_processed():
            return

        # Fall back to runtime splitting
        self.data_path = resolve_dataset_asset_path(
            self.data_path,
            dataset_name=self.dataset_name,
            asset_label="raw parquet",
        )
        self.data = pd.read_parquet(self.data_path)
        drop_cols = [c for c in ["timestamp", "date"] if c in self.data.columns]
        if drop_cols:
            self.data = self.data.drop(columns=drop_cols)
        self.feature_names = list(self.data.columns)

        n_samples = len(self.data)

        if self._train_end is not None and self._val_end is not None:
            train_end = int(self._train_end)
            val_end = int(self._val_end)
            split_mode = f"absolute indices (train_end={train_end}, val_end={val_end})"
        else:
            ratio = self.train_val_test_ratio
            train_end = int(n_samples * ratio[0])
            val_end = int(n_samples * (ratio[0] + ratio[1]))
            split_mode = f"ratio {ratio}"

        self._train_end_resolved = train_end
        self._val_end_resolved = val_end

        self.train_data = self.data.iloc[:train_end]
        self.val_data = self.data.iloc[train_end:val_end]
        self.test_data = self.data.iloc[val_end:]

        print(f"Data loaded: {self.dataset_name}")
        print(f"  Total timesteps: {n_samples}  |  Split: {split_mode}")
        print(f"  Features: {len(self.feature_names)}")
        print(
            f"  Train: {len(self.train_data)}, Val: {len(self.val_data)}, "
            f"Test: {len(self.test_data)}"
        )
        print(f"  Model family: {self.model_family}")

    def _get_dataset_class(self) -> TrainingDatasetClass:
        if self.model_family == "chronos":
            return ChronosTrainingDataset
        return MoiraiTrainingDataset

    def _get_collate_fn(self):
        if self.model_family == "chronos":
            return chronos_collate_fn
        return moirai_collate_fn

    def _pin_memory(self) -> bool:
        """Enable pin_memory only when CUDA is available (avoids MPS warning)."""
        return torch.cuda.is_available()

    def get_train_dataloader(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
    ) -> DataLoader:
        """Create training DataLoader.

        Args:
            batch_size:   Samples per batch.
            shuffle:      Shuffle training samples (default True).
            num_workers:  CPU workers for data loading.
                          Recommended: 0 on macOS/MPS, ≥2 on Linux/CUDA.
        """
        if self.train_data is None:
            raise RuntimeError("Call setup() first")

        dataset_cls = self._get_dataset_class()
        dataset = dataset_cls(
            data=self.train_data,
            context_length=self.context_length,
            prediction_length=self.prediction_length,
        )

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=self._pin_memory(),
            persistent_workers=(num_workers > 0),
            collate_fn=self._get_collate_fn(),
        )

    def get_val_dataloader(
        self,
        batch_size: int = 32,
        num_workers: int = 0,
    ) -> DataLoader:
        """Create validation DataLoader (no shuffle, deterministic)."""
        if self.val_data is None:
            raise RuntimeError("Call setup() first")

        dataset_cls = self._get_dataset_class()
        dataset = dataset_cls(
            data=self.val_data,
            context_length=self.context_length,
            prediction_length=self.prediction_length,
        )

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=self._pin_memory(),
            persistent_workers=(num_workers > 0),
            collate_fn=self._get_collate_fn(),
        )

    def get_metadata(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "num_features": len(self.feature_names),
            "feature_names": self.feature_names,
            "context_length": self.context_length,
            "prediction_length": self.prediction_length,
            "train_samples": len(self.train_data) if self.train_data is not None else 0,
            "val_samples": len(self.val_data) if self.val_data is not None else 0,
            "test_samples": len(self.test_data) if self.test_data is not None else 0,
            "model_family": self.model_family,
            "train_end": self._train_end_resolved,
            "val_end": self._val_end_resolved,
        }
