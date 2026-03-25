"""
Data preprocessing script.

Reads a raw time-series parquet file and produces a reproducible, shareable
set of split files that can be uploaded directly to HuggingFace as a dataset.

Output layout
-------------
{processed_dir}/{dataset_name}/
    full.parquet      — cleaned full time series (timestamp cols dropped)
    train.parquet     — rows [0, train_end)       60% by default
    val.parquet       — rows [train_end, val_end)  20% by default
    test.parquet      — rows [val_end, total)      20% by default
    metadata.json     — shape, feature names, split indices, checksums

Why pre-split?
--------------
1. Reproducibility: the same exact rows are used in every training and
   inference run.  Splitting at runtime from ratio leaves a rounding
   ambiguity (e.g. 191.4 → 191 or 192?).
2. Shareability: upload the four parquet files + metadata.json to
   HuggingFace Datasets and anyone can reproduce your experiments
   with `pd.read_parquet(hf_hub_download(...))`.
3. Training efficiency: fine-tuning loads only `train.parquet`
   (the smaller file) instead of the full series.
4. Inference: ZeroShotDataModule loads `full.parquet` so the
   expanding-window context is always [0, t) with the correct boundaries.

Data format
-----------
Fine-tuning (LoRA / full-tune):
    SLIDING window of fixed context_length (paper: 48 days).
    The PyTorch Datasets in training.py implement this over train.parquet.

Inference (zero-shot and fine-tuned):
    EXPANDING window starting from t=0.
    The first test prediction uses [0, val_end) as context (all prior data).
    The last test prediction uses [0, total - pred_len) as context.
    This is what ZeroShotDataModule implements over full.parquet.

Usage
-----
    # Single dataset
    python -m pmf_tsfm.data.preprocess data=bpi2017

    # All datasets at once
    python -m pmf_tsfm.data.preprocess --multirun \\
        data=bpi2017,bpi2019_1,sepsis,hospital_billing

    # Force overwrite existing processed files
    python -m pmf_tsfm.data.preprocess data=bpi2017 force_overwrite=true

    # Custom output directory
    python -m pmf_tsfm.data.preprocess data=bpi2017 \\
        processed_dir=/path/to/shared/datasets
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf


def _sha256(path: Path) -> str:
    """Compute SHA-256 hash of a file for data integrity verification."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop non-numeric auxiliary columns (timestamp, date) if present."""
    drop_cols = [c for c in df.columns if c.lower() in {"timestamp", "date", "datetime", "time"}]
    if drop_cols:
        print(f"  Dropping auxiliary columns: {drop_cols}")
        df = df.drop(columns=drop_cols)
    return df


def compute_split_indices(
    total: int,
    train_end: int | None,
    val_end: int | None,
    split_ratio: list[float],
) -> tuple[int, int]:
    """
    Resolve absolute split indices.

    Priority:
      1. Explicit train_end / val_end from config (paper-specified boundaries).
      2. Ratio-based fallback.

    Returns:
        (train_end, val_end) as integer row indices.
    """
    if train_end is not None and val_end is not None:
        return int(train_end), int(val_end)

    r_train, r_val, _ = split_ratio
    t_end = int(np.floor(total * r_train))
    v_end = int(np.floor(total * (r_train + r_val)))
    return t_end, v_end


def preprocess_dataset(
    raw_path: Path,
    dataset_name: str,
    out_dir: Path,
    train_end: int | None,
    val_end: int | None,
    split_ratio: list[float],
    force_overwrite: bool = False,
) -> dict[str, Any]:
    """
    Split and save a single dataset.

    Args:
        raw_path:       Path to original parquet file.
        dataset_name:   Used for output directory name and metadata.
        out_dir:        Root processed directory.  Output goes to out_dir/dataset_name/.
        train_end:      Absolute row index for end of training split (exclusive).
        val_end:        Absolute row index for end of validation split (exclusive).
        split_ratio:    Fallback [train, val, test] ratios when indices absent.
        force_overwrite:Overwrite existing processed files if True.

    Returns:
        Metadata dict (also saved as metadata.json).
    """
    split_dir = out_dir / dataset_name
    metadata_path = split_dir / "metadata.json"

    if metadata_path.exists() and not force_overwrite:
        print(f"  Already processed (skipping): {split_dir}")
        print("  Pass force_overwrite=true to regenerate.")
        with open(metadata_path) as f:
            return json.load(f)

    split_dir.mkdir(parents=True, exist_ok=True)

    # Load and clean
    print(f"  Reading {raw_path}")
    df = pd.read_parquet(raw_path)
    df = _clean(df)

    total = len(df)
    num_features = df.shape[1]
    feature_names = list(df.columns)

    t_end, v_end = compute_split_indices(total, train_end, val_end, split_ratio)

    # Validate
    assert 0 < t_end < v_end < total, (
        f"Invalid split indices: train_end={t_end}, val_end={v_end}, total={total}"
    )

    train_df = df.iloc[:t_end].reset_index(drop=True)
    val_df = df.iloc[t_end:v_end].reset_index(drop=True)
    test_df = df.iloc[v_end:].reset_index(drop=True)
    full_df = df.reset_index(drop=True)

    # Save parquets
    files = {
        "full.parquet": full_df,
        "train.parquet": train_df,
        "val.parquet": val_df,
        "test.parquet": test_df,
    }
    for fname, frame in files.items():
        fpath = split_dir / fname
        frame.to_parquet(fpath, index=False)
        print(f"  Saved {fpath.name}: {len(frame)} rows × {frame.shape[1]} cols")

    # Build metadata
    metadata: dict[str, Any] = {
        "dataset_name": dataset_name,
        "total_timesteps": total,
        "num_features": num_features,
        "feature_names": feature_names,
        "dtypes": {col: str(df[col].dtype) for col in feature_names},
        "split": {
            "train_end": t_end,
            "val_end": v_end,
            "train_size": len(train_df),
            "val_size": len(val_df),
            "test_size": len(test_df),
            "ratio_approx": [
                round(len(train_df) / total, 3),
                round(len(val_df) / total, 3),
                round(len(test_df) / total, 3),
            ],
        },
        "statistics": {
            "global_mean": float(df.values.mean()),
            "global_std": float(df.values.std()),
            "train_mean": float(train_df.values.mean()),
            "train_std": float(train_df.values.std()),
        },
        "checksums": {fname: _sha256(split_dir / fname) for fname in files},
        "created_at": datetime.utcnow().isoformat() + "Z",
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print("  Saved metadata.json")

    print("\n  Split summary:")
    print(f"    Train : [{0}, {t_end})   → {len(train_df):>4} rows")
    print(f"    Val   : [{t_end}, {v_end}) → {len(val_df):>4} rows")
    print(f"    Test  : [{v_end}, {total}) → {len(test_df):>4} rows")

    return metadata


@hydra.main(version_base="1.3", config_path="../../../configs", config_name="preprocess")
def main(cfg: DictConfig) -> dict[str, Any]:
    """Hydra entry point — processes one or more datasets from config."""
    if cfg.get("print_config", False):
        print(OmegaConf.to_yaml(cfg))

    processed_dir = Path(cfg.processed_dir)
    raw_path = Path(cfg.data.path)
    dataset_name = cfg.data.name

    print("=" * 60)
    print(f"PREPROCESSING: {dataset_name}")
    print(f"  Source : {raw_path}")
    print(f"  Output : {processed_dir / dataset_name}")
    print("=" * 60)

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data file not found: {raw_path}")

    split_ratio = list(cfg.data.get("split_ratio", [0.6, 0.2, 0.2]))

    metadata = preprocess_dataset(
        raw_path=raw_path,
        dataset_name=dataset_name,
        out_dir=processed_dir,
        train_end=cfg.data.get("train_end"),
        val_end=cfg.data.get("val_end"),
        split_ratio=split_ratio,
        force_overwrite=cfg.get("force_overwrite", False),
    )

    print("\nDone.")
    return metadata


if __name__ == "__main__":
    main()
