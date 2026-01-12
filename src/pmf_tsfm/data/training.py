"""Training data module for LoRA fine-tuning.

Provides PyTorch Dataset and DataLoader factory for training
time series foundation models like Moirai and Chronos with LoRA.
"""

from pathlib import Path
from typing import Any, Literal, TypeAlias

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class MoiraiTrainingDataset(Dataset):
    """PyTorch Dataset for Moirai LoRA training.

    Converts wide-format time series data (rows=timesteps, columns=features)
    into training samples with past context and future targets.

    Args:
        data: DataFrame with time series (rows=time, columns=features)
        context_length: Length of input context window
        prediction_length: Length of prediction horizon
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

        # Calculate valid starting indices
        self.num_samples: int = max(0, len(self.data) - self.seq_length + 1)

    def __len__(self) -> int:
        return self.num_samples * self.num_features

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # Decompose idx into (sequence_idx, feature_idx)
        seq_idx = idx // self.num_features
        feature_idx = idx % self.num_features

        # Extract context and target for this feature
        start = seq_idx
        context_end = start + self.context_length
        target_end = context_end + self.prediction_length

        # Get univariate series for this feature
        past_target = self.data[start:context_end, feature_idx]
        future_target = self.data[context_end:target_end, feature_idx]

        # Create observation mask (all observed for training data)
        past_observed = np.ones_like(past_target, dtype=bool)

        # Create padding mask (no padding for training data)
        past_is_pad = np.zeros(self.context_length, dtype=bool)

        return {
            "past_target": torch.from_numpy(past_target).unsqueeze(-1),  # (context_length, 1)
            "past_observed_target": torch.from_numpy(past_observed).unsqueeze(
                -1
            ),  # (context_length, 1)
            "past_is_pad": torch.from_numpy(past_is_pad),  # (context_length,)
            "future_target": torch.from_numpy(future_target).unsqueeze(
                -1
            ),  # (prediction_length, 1)
        }


class ChronosTrainingDataset(Dataset):
    """PyTorch Dataset for Chronos Bolt LoRA training.

    Converts wide-format time series data into samples with context/target format
    expected by Chronos Bolt models.

    Args:
        data: DataFrame with time series (rows=time, columns=features)
        context_length: Length of input context window
        prediction_length: Length of prediction horizon
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

        # Calculate valid starting indices
        self.num_samples: int = max(0, len(self.data) - self.seq_length + 1)

    def __len__(self) -> int:
        return self.num_samples * self.num_features

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # Decompose idx into (sequence_idx, feature_idx)
        seq_idx = idx // self.num_features
        feature_idx = idx % self.num_features

        # Extract context and target for this feature
        start = seq_idx
        context_end = start + self.context_length
        target_end = context_end + self.prediction_length

        # Get univariate series for this feature
        context = self.data[start:context_end, feature_idx]
        target = self.data[context_end:target_end, feature_idx]

        return {
            "context": torch.from_numpy(context),  # (context_length,)
            "target": torch.from_numpy(target),  # (prediction_length,)
        }


TrainingDatasetClass: TypeAlias = type[MoiraiTrainingDataset] | type[ChronosTrainingDataset]


def moirai_collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Collate function for Moirai training batches.

    Stacks individual samples into batched tensors with correct dtypes.

    Args:
        batch: List of sample dicts from MoiraiTrainingDataset

    Returns:
        Batched dict with keys:
            - past_target: (batch, context_length, 1)
            - past_observed_target: (batch, context_length, 1)
            - past_is_pad: (batch, context_length)
            - future_target: (batch, prediction_length, 1)
    """
    return {
        "past_target": torch.stack([s["past_target"] for s in batch]),
        "past_observed_target": torch.stack([s["past_observed_target"] for s in batch]),
        "past_is_pad": torch.stack([s["past_is_pad"] for s in batch]),
        "future_target": torch.stack([s["future_target"] for s in batch]),
    }


def chronos_collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Collate function for Chronos training batches.

    Args:
        batch: List of sample dicts from ChronosTrainingDataset

    Returns:
        Batched dict with keys:
            - context: (batch, context_length)
            - target: (batch, prediction_length)
    """
    return {
        "context": torch.stack([s["context"] for s in batch]),
        "target": torch.stack([s["target"] for s in batch]),
    }


class TrainingDataModule:
    """Data module for LoRA training with train/val splits.

    Loads time series data and creates DataLoaders for training.
    Supports both Moirai and Chronos data formats.

    Args:
        dataset_name: Name of the dataset
        data_path: Path to parquet file with time series data
        context_length: Length of input context window
        prediction_length: Length of prediction horizon
        train_val_test_ratio: Split ratios [train, val, test]
        model_family: "moirai" or "chronos" - determines data format
    """

    def __init__(
        self,
        dataset_name: str,
        data_path: str,
        context_length: int,
        prediction_length: int,
        train_val_test_ratio: list[float] | None = None,
        model_family: Literal["moirai", "chronos"] = "moirai",
    ):
        self.dataset_name = dataset_name
        self.data_path = Path(data_path)
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.train_val_test_ratio = train_val_test_ratio or [0.6, 0.2, 0.2]
        self.model_family = model_family

        self.data: pd.DataFrame | None = None
        self.train_data: pd.DataFrame | None = None
        self.val_data: pd.DataFrame | None = None
        self.test_data: pd.DataFrame | None = None
        self.feature_names: list[str] = []

    def setup(self) -> None:
        """Load and split the data."""
        # Load data
        self.data = pd.read_parquet(self.data_path)

        # Drop timestamp columns if present
        drop_cols = [c for c in ["timestamp", "date"] if c in self.data.columns]
        if drop_cols:
            self.data = self.data.drop(columns=drop_cols)

        self.feature_names = list(self.data.columns)

        # Split data temporally
        n_samples = len(self.data)
        train_end = int(n_samples * self.train_val_test_ratio[0])
        val_end = int(n_samples * (self.train_val_test_ratio[0] + self.train_val_test_ratio[1]))

        self.train_data = self.data.iloc[:train_end]
        self.val_data = self.data.iloc[train_end:val_end]
        self.test_data = self.data.iloc[val_end:]

        print(f"Data loaded: {self.dataset_name}")
        print(f"  Total timesteps: {n_samples}")
        print(f"  Features: {len(self.feature_names)}")
        print(
            f"  Train: {len(self.train_data)}, Val: {len(self.val_data)}, Test: {len(self.test_data)}"
        )
        print(f"  Model family: {self.model_family}")

    def _get_dataset_class(self) -> TrainingDatasetClass:
        """Get the appropriate dataset class for the model family."""
        if self.model_family == "chronos":
            return ChronosTrainingDataset
        return MoiraiTrainingDataset

    def _get_collate_fn(self):
        """Get the appropriate collate function for the model family."""
        if self.model_family == "chronos":
            return chronos_collate_fn
        return moirai_collate_fn

    def get_train_dataloader(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = True,
    ) -> DataLoader:
        """Create training DataLoader."""
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
            pin_memory=pin_memory,
            collate_fn=self._get_collate_fn(),
        )

    def get_val_dataloader(
        self,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = True,
    ) -> DataLoader:
        """Create validation DataLoader."""
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
            pin_memory=pin_memory,
            collate_fn=self._get_collate_fn(),
        )

    def get_metadata(self) -> dict[str, Any]:
        """Get dataset metadata."""
        return {
            "dataset_name": self.dataset_name,
            "num_features": len(self.feature_names),
            "feature_names": self.feature_names,
            "context_length": self.context_length,
            "prediction_length": self.prediction_length,
            "train_samples": len(self.train_data) if self.train_data is not None else 0,
            "val_samples": len(self.val_data) if self.val_data is not None else 0,
            "model_family": self.model_family,
        }
