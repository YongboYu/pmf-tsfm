"""
Zero-shot data module for time series forecasting.

Converts multivariate time series into sequence-to-sequence format using expanding window.
- Input: expanding window (all history up to prediction point)
- Output: fixed length horizon

Uses absolute point splits for clear, reproducible data partitioning.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig


class ZeroShotDataModule:
    """
    Data module for zero-shot time series forecasting with expanding window.

    Uses absolute point splits:
    - Train: [0, train_end)
    - Val: [train_end, val_end)
    - Test: [val_end, end)

    For multivariate time series:
    - Input shape: variable length (expanding window)
    - Target shape: (num_sequences, prediction_length, num_features)
    """

    def __init__(
        self,
        dataset_name: str,
        data_path: str,
        prediction_length: int = 7,
        train_end: int = 191,
        val_end: int = 255,
    ):
        """
        Initialize the ZeroShotDataModule.

        Args:
            dataset_name: Name of the dataset
            data_path: Path to the parquet file
            prediction_length: Forecasting horizon (default: 7)
            train_end: End index of training data (exclusive)
            val_end: End index of validation data (exclusive)
        """
        self.dataset_name = dataset_name
        self.data_path = Path(data_path)
        self.prediction_length = prediction_length
        self.train_end = train_end
        self.val_end = val_end

        self.data: pd.DataFrame | None = None
        self.feature_names: list[str] | None = None
        self.metadata: dict[str, Any] = {}

    @classmethod
    def from_config(cls, data_cfg: DictConfig, prediction_length: int = 7) -> "ZeroShotDataModule":
        """Create data module from Hydra config."""
        return cls(
            dataset_name=data_cfg.name,
            data_path=data_cfg.path,
            prediction_length=prediction_length,
            train_end=data_cfg.train_end,
            val_end=data_cfg.val_end,
        )

    def setup(self) -> None:
        """Load and prepare data."""
        print(f"Loading data from {self.data_path}")
        self.data = pd.read_parquet(self.data_path)
        self.feature_names = list(self.data.columns)

        total_length = len(self.data)
        test_length = total_length - self.val_end

        self.metadata = {
            "dataset_name": self.dataset_name,
            "total_length": total_length,
            "num_features": len(self.feature_names),
            "feature_names": self.feature_names,
            "prediction_length": self.prediction_length,
            "splits": {
                "train": f"[0, {self.train_end})",
                "val": f"[{self.train_end}, {self.val_end})",
                "test": f"[{self.val_end}, {total_length})",
            },
            "split_lengths": {
                "train": self.train_end,
                "val": self.val_end - self.train_end,
                "test": test_length,
            },
        }

        print(f"  Shape: {total_length} timesteps × {len(self.feature_names)} features")
        print(
            f"  Splits: train={self.train_end}, val={self.val_end - self.train_end}, test={test_length}"
        )

    def _create_expanding_sequences(self, split: str) -> dict[str, Any]:
        """
        Create sequences with expanding window for specified split.

        For expanding window:
        - Input: all data from beginning up to target_idx
        - Target: next prediction_length points after target_idx

        Args:
            split: 'train', 'val', or 'test'

        Returns:
            Dictionary with 'inputs' (list) and 'targets' (ndarray)
        """
        if self.data is None:
            raise RuntimeError("Data not loaded. Call setup() first.")

        full_data = self.data.values
        total_length = len(full_data)

        # Determine target start and end for this split using absolute indices
        if split == "train":
            target_start = self.prediction_length
            target_end = self.train_end - self.prediction_length
        elif split == "val":
            target_start = self.train_end
            target_end = self.val_end - self.prediction_length
        else:  # test
            target_start = self.val_end
            target_end = total_length - self.prediction_length

        inputs: list[np.ndarray] = []
        targets_list: list[np.ndarray] = []

        for target_idx in range(target_start, target_end + 1):
            # Input: all data from beginning up to target_idx (expanding window)
            input_seq = full_data[:target_idx]
            # Target: next prediction_length points
            target_seq = full_data[target_idx : target_idx + self.prediction_length]

            inputs.append(input_seq)
            targets_list.append(target_seq)

        # Convert targets to numpy array (uniform shape)
        if targets_list:
            targets = np.array(targets_list)
        else:
            targets = np.empty((0, self.prediction_length, full_data.shape[1]))

        return {"inputs": inputs, "targets": targets}

    def prepare_data_for_model(self, split: str = "test") -> dict[str, Any]:
        """
        Prepare data for zero-shot inference.

        Args:
            split: Which split to prepare ('train', 'val', 'test')

        Returns:
            Dictionary containing:
                - inputs: List of input sequences (expanding window)
                - targets: Target sequences array
                - feature_names: List of feature names
                - metadata: Dataset metadata
        """
        if self.data is None:
            self.setup()

        sequences = self._create_expanding_sequences(split)

        # Update metadata
        self.metadata.update(
            {
                f"{split}_num_sequences": len(sequences["inputs"]),
                f"{split}_target_shape": list(sequences["targets"].shape),
            }
        )

        print(f"Prepared {split} data:")
        print(f"  - Sequences: {len(sequences['inputs'])}")
        print(f"  - Features: {len(self.feature_names) if self.feature_names else 0}")
        print(f"  - Target shape: {sequences['targets'].shape}")
        if sequences["inputs"]:
            print(
                f"  - Input lengths: first={len(sequences['inputs'][0])}, last={len(sequences['inputs'][-1])}"
            )

        return {
            "inputs": sequences["inputs"],
            "targets": sequences["targets"],
            "feature_names": self.feature_names,
            "metadata": self.metadata,
        }
