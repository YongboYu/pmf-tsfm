"""
Zero-shot data module for time series forecasting.

Data format (paper Section 4.1)
---------------------------------
Inference uses an EXPANDING window: for each prediction point t in the test
split, the model receives ALL data from t=0 up to (but not including) t as
context, and the model must predict the next `prediction_length` steps.

    Context for test point i:  full_data[0 : val_end + i]
    Target  for test point i:  full_data[val_end + i : val_end + i + pred_len]

This means the first test prediction sees the complete training + validation
history as context.  Each subsequent test prediction has one additional
observed day.

The expanding window is intentional: foundation models like Chronos and Moirai
were pre-trained on variable-length series, so providing more context generally
improves accuracy.

Pre-split loading
-----------------
If `processed_dir` is set, the module loads:
    {processed_dir}/{dataset_name}/full.parquet   — entire cleaned time series
    {processed_dir}/{dataset_name}/metadata.json  — split indices (train_end, val_end)

This guarantees the expanding window uses the exact same boundaries as the
TrainingDataModule used during fine-tuning.  Run:
    python -m pmf_tsfm.data.preprocess data=<dataset>
once before running any experiments.

Efficiency note
---------------
All expanding-window inputs are kept in a Python list because they have
different lengths and cannot be stacked into a uniform tensor.  For the
process-mining datasets in this paper (< 400 timesteps), the full list
fits comfortably in RAM.  For much longer series, a lazy on-demand approach
(reading rows from parquet per call) would be preferable.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from pmf_tsfm.data.assets import resolve_dataset_asset_path


class ZeroShotDataModule:
    """
    Data module for zero-shot and fine-tuned inference using an expanding window.

    Uses absolute split indices:
        Train context  [0, train_end)     — used for fine-tuning
        Val context    [train_end, val_end) — used for fine-tuning validation
        Test sequences [val_end, total)   — evaluated here with full prior context

    For each test point the context expands from [0, val_end) to [0, total - pred_len),
    so the model always sees ALL prior data.

    Supports two loading modes:
    1. Pre-split (preferred): loads `full.parquet` and `metadata.json` from
       `{processed_dir}/{dataset_name}/`.
    2. Runtime: loads the raw parquet from `data_path` and splits by indices.
    """

    def __init__(
        self,
        dataset_name: str,
        data_path: str,
        prediction_length: int = 7,
        train_end: int = 191,
        val_end: int = 255,
        processed_dir: str | None = None,
    ):
        """
        Args:
            dataset_name:     Dataset identifier.
            data_path:        Path to raw parquet (used only if pre-split files absent).
            prediction_length: Forecasting horizon.
            train_end:        End of training split (exclusive row index).
            val_end:          End of validation split (exclusive row index).
            processed_dir:    Root directory of pre-split files.  If
                              `{processed_dir}/{dataset_name}/full.parquet` exists,
                              it is loaded instead of `data_path`.
        """
        self.dataset_name = dataset_name
        self.data_path = Path(data_path)
        self.prediction_length = prediction_length
        self.train_end = train_end
        self.val_end = val_end
        self._processed_dir = Path(processed_dir) / dataset_name if processed_dir else None

        self.data: pd.DataFrame | None = None
        self.feature_names: list[str] | None = None
        self.metadata: dict[str, Any] = {}

    @classmethod
    def from_config(
        cls,
        data_cfg: DictConfig,
        prediction_length: int = 7,
        processed_dir: str | None = None,
    ) -> "ZeroShotDataModule":
        """Create data module from Hydra data config."""
        return cls(
            dataset_name=data_cfg.name,
            data_path=data_cfg.path,
            prediction_length=prediction_length,
            train_end=data_cfg.train_end,
            val_end=data_cfg.val_end,
            processed_dir=processed_dir,
        )

    def _load_raw(self) -> pd.DataFrame:
        """Load from raw parquet and drop auxiliary columns."""
        self.data_path = resolve_dataset_asset_path(
            self.data_path,
            dataset_name=self.dataset_name,
            asset_label="raw parquet",
        )
        df = pd.read_parquet(self.data_path)
        drop_cols = [
            c for c in df.columns if c.lower() in {"timestamp", "date", "datetime", "time"}
        ]
        if drop_cols:
            df = df.drop(columns=drop_cols)
        return df

    def setup(self) -> None:
        """Load data and compute metadata.

        Priority:
          1. {processed_dir}/{dataset_name}/full.parquet  (pre-split, preferred)
          2. Raw parquet at data_path (runtime split)

        When loading from pre-split files, split indices (train_end, val_end)
        are taken from metadata.json if present, otherwise the constructor
        values are used.
        """
        using_preprocessed = False

        if self._processed_dir is not None:
            full_file = self._processed_dir / "full.parquet"
            meta_file = self._processed_dir / "metadata.json"

            if full_file.exists():
                print(f"Loading pre-split data from {self._processed_dir}")
                self.data = pd.read_parquet(full_file)
                using_preprocessed = True

                # Prefer split indices from metadata.json over constructor args
                if meta_file.exists():
                    with open(meta_file) as f:
                        stored = json.load(f)
                    self.train_end = stored["split"]["train_end"]
                    self.val_end = stored["split"]["val_end"]
                    print(
                        f"  Split indices from metadata.json: "
                        f"train_end={self.train_end}, val_end={self.val_end}"
                    )

        if self.data is None:
            print(f"Loading data from {self.data_path}")
            self.data = self._load_raw()

        self.feature_names = list(self.data.columns)
        total_length = len(self.data)
        test_length = total_length - self.val_end

        source = "pre-split" if using_preprocessed else "raw parquet"
        self.metadata = {
            "dataset_name": self.dataset_name,
            "source": source,
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
            f"  Splits: train={self.train_end}, "
            f"val={self.val_end - self.train_end}, test={test_length}"
        )

    def _create_expanding_sequences(self, split: str) -> dict[str, Any]:
        """
        Build expanding-window sequences for the requested split.

        Expanding window:
            Input : full_data[0 : target_idx]      (grows by 1 per step)
            Target: full_data[target_idx : target_idx + prediction_length]

        Split boundaries (row indices into the full time series):
            train: targets in [prediction_length, train_end - prediction_length]
            val:   targets in [train_end,          val_end - prediction_length]
            test:  targets in [val_end,            total - prediction_length]

        For the test split the context always starts at t=0 (including training
        and validation rows), mirroring a realistic deployment where the model
        has access to all historical data.
        """
        if self.data is None:
            raise RuntimeError("Data not loaded. Call setup() first.")

        full_data = self.data.values
        total_length = len(full_data)

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
            input_seq = full_data[:target_idx]  # expanding
            target_seq = full_data[target_idx : target_idx + self.prediction_length]
            inputs.append(input_seq)
            targets_list.append(target_seq)

        targets_arr = (
            np.asarray(targets_list)
            if targets_list
            else np.empty((0, self.prediction_length, full_data.shape[1]))
        )

        return {"inputs": inputs, "targets": targets_arr}

    def prepare_data_for_model(self, split: str = "test") -> dict[str, Any]:
        """
        Prepare expanding-window sequences for model inference.

        Args:
            split: 'train', 'val', or 'test'.

        Returns:
            dict with:
              inputs        — list of (seq_len, num_features) arrays (variable length)
              targets       — (num_sequences, prediction_length, num_features) array
              feature_names — list of feature names
              metadata      — dataset metadata dict
        """
        if self.data is None:
            self.setup()

        if self.feature_names is None:
            raise RuntimeError("Feature names not loaded. Call setup() first.")

        sequences = self._create_expanding_sequences(split)

        self.metadata.update(
            {
                f"{split}_num_sequences": len(sequences["inputs"]),
                f"{split}_target_shape": list(sequences["targets"].shape),
            }
        )

        print(f"Prepared {split} data:")
        print(f"  - Sequences : {len(sequences['inputs'])}")
        print(f"  - Features  : {len(self.feature_names)}")
        print(f"  - Target shape: {sequences['targets'].shape}")
        if sequences["inputs"]:
            print(
                f"  - Context lengths: "
                f"first={len(sequences['inputs'][0])}, "
                f"last={len(sequences['inputs'][-1])}"
            )

        return {
            "inputs": sequences["inputs"],
            "targets": sequences["targets"],
            "feature_names": self.feature_names,
            "metadata": self.metadata,
        }
