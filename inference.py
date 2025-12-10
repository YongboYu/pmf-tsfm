#!/usr/bin/env python
"""
Zero-shot forecasting inference script.

Generates predictions only. Use evaluate.py for metrics computation.

Output structure: outputs/{task}/{dataset}/{model}/
    - predictions.npy
    - targets.npy
    - quantiles.npy
    - metadata.json

Usage:
    # Default: chronos/bolt_small on BPI2017 (cuda)
    python inference.py

    # Run on macOS with MPS
    python inference.py device=mps

    # Run on CPU (fallback)
    python inference.py device=cpu

    # Specific model and dataset
    python inference.py model=chronos/bolt_base data=sepsis

    # Multi-run experiments
    python inference.py --multirun \\
        model=chronos/bolt_small,chronos/chronos2 \\
        data=bpi2017,bpi2019_1,hospital_billing,sepsis
"""

import json
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent / "src"))

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf


def resolve_device(cfg: DictConfig) -> str:
    """
    Resolve the actual device to use based on config and model compatibility.

    Args:
        cfg: Hydra config with device and model.supported_devices

    Returns:
        Resolved device string
    """
    requested_device = cfg.device
    supported_devices = cfg.model.get("supported_devices", ["cuda", "mps", "cpu"])
    fallback_device = cfg.model.get("fallback_device", "cpu")

    # Check if requested device is supported
    if requested_device not in supported_devices:
        print(f"⚠️  Device '{requested_device}' not supported by {cfg.model.name}")
        print(f"   Supported: {supported_devices}")
        print(f"   Falling back to: {fallback_device}")
        return fallback_device

    # Check device availability
    if requested_device == "cuda" and not torch.cuda.is_available():
        print(f"⚠️  CUDA not available, falling back to {fallback_device}")
        return fallback_device

    if requested_device == "mps":
        if not torch.backends.mps.is_available():
            print(f"⚠️  MPS not available, falling back to {fallback_device}")
            return fallback_device
        if not torch.backends.mps.is_built():
            print(f"⚠️  MPS not built in this PyTorch, falling back to {fallback_device}")
            return fallback_device

    return requested_device


def save_predictions(
    predictions: np.ndarray,
    targets: np.ndarray,
    quantiles: np.ndarray,
    output_dir: str,
    metadata: dict,
) -> Path:
    """
    Save predictions and targets to disk.

    Output structure: {output_dir}/
        - predictions.npy
        - targets.npy
        - quantiles.npy
        - metadata.json
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "predictions.npy", predictions)
    np.save(output_dir / "targets.npy", targets)
    np.save(output_dir / "quantiles.npy", quantiles)

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"Predictions saved to {output_dir}/")
    return output_dir


def run_inference(cfg: DictConfig) -> dict:
    """
    Run zero-shot forecasting pipeline.

    Pipeline:
    1. Resolve device (check compatibility)
    2. Load data with expanding window
    3. Load pre-trained model
    4. Generate predictions
    5. Save predictions to disk
    """
    from pmf_tsfm.data import ZeroShotDataModule
    from pmf_tsfm.models import get_model_adapter

    # Resolve device
    device = resolve_device(cfg)

    # Set seeds
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Print run info
    print("=" * 70)
    print("ZERO-SHOT INFERENCE")
    print(f"  Model: {cfg.model.name}")
    print(f"  Data:  {cfg.data.name}")
    print(f"  Device: {device}")
    print(f"  Prediction Length: {cfg.prediction_length}")
    print(f"  Output: {cfg.output_dir}")
    print("=" * 70)

    # ========== Step 1: Prepare Data ==========
    print("\n[1/3] Preparing data...")
    data_module = ZeroShotDataModule.from_config(
        data_cfg=cfg.data,
        prediction_length=cfg.prediction_length,
    )
    data_module.setup()
    prepared_data = data_module.prepare_data_for_model(split="test")

    # ========== Step 2: Load Model ==========
    print("\n[2/3] Loading model...")
    adapter = get_model_adapter(
        model_cfg=cfg.model,
        device=device,
        prediction_length=cfg.prediction_length,
    )
    adapter.load_model()

    # ========== Step 3: Generate Predictions ==========
    print("\n[3/3] Generating predictions...")
    predictions, quantiles = adapter.predict(
        prepared_data=prepared_data,
        prediction_length=cfg.prediction_length,
    )

    # ========== Save Results ==========
    targets = prepared_data["targets"]

    metadata = {
        "model": {
            "name": cfg.model.name,
            "id": cfg.model.id,
            "family": cfg.model.family,
            "variant": cfg.model.variant,
        },
        "data": {
            "name": cfg.data.name,
            "train_end": cfg.data.train_end,
            "val_end": cfg.data.val_end,
        },
        "task": cfg.task,
        "device": device,
        "seed": cfg.seed,
        "prediction_length": cfg.prediction_length,
        "feature_names": prepared_data["feature_names"],
        "shapes": {
            "predictions": list(predictions.shape),
            "targets": list(targets.shape),
            "quantiles": list(quantiles.shape),
        },
        "data_metadata": prepared_data["metadata"],
    }

    output_path = save_predictions(
        predictions=predictions,
        targets=targets,
        quantiles=quantiles,
        output_dir=cfg.output_dir,
        metadata=metadata,
    )

    print("\n" + "=" * 70)
    print("INFERENCE COMPLETE")
    print(f"  Output: {output_path}")
    print(f"  Run evaluation: python evaluate.py task={cfg.task}")
    print("=" * 70)

    return {
        "predictions": predictions,
        "targets": targets,
        "quantiles": quantiles,
        "output_path": output_path,
    }


@hydra.main(version_base="1.3", config_path="configs", config_name="inference")
def main(cfg: DictConfig):
    """Main entry point with Hydra configuration."""
    if cfg.get("print_config", False):
        print(OmegaConf.to_yaml(cfg))

    return run_inference(cfg)


if __name__ == "__main__":
    main()
