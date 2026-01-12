"""
Zero-shot forecasting inference script.

Generates predictions only. Use evaluate.py for metrics computation.

Usage:
    # Default: chronos/bolt_small on BPI2017
    python -m pmf_tsfm.inference

    # Specific model and dataset
    python -m pmf_tsfm.inference model=chronos/bolt_base data=sepsis

    # Chronos 2.0
    python -m pmf_tsfm.inference model=chronos/chronos2

    # Multi-run with sweep
    python -m pmf_tsfm.inference --multirun model=chronos/bolt_tiny,chronos/bolt_small

    # Run on all 4 datasets
    python -m pmf_tsfm.inference --multirun data=bpi2017,bpi2019_1,hospital_billing,sepsis
"""

import json
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from pmf_tsfm.data import ZeroShotDataModule
from pmf_tsfm.models import get_model_adapter


def save_predictions(
    predictions: np.ndarray,
    targets: np.ndarray,
    quantiles: np.ndarray,
    output_dir: str | Path,
    model_name: str,
    dataset_name: str,
    metadata: dict,
) -> Path:
    """
    Save predictions and targets to disk.

    Creates:
    - {dataset}_{model}_predictions.npy
    - {dataset}_{model}_targets.npy
    - {dataset}_{model}_quantiles.npy
    - {dataset}_{model}_metadata.json

    Returns:
        Path to output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    prefix = f"{dataset_name}_{model_name}"

    np.save(output_path / f"{prefix}_predictions.npy", predictions)
    np.save(output_path / f"{prefix}_targets.npy", targets)
    np.save(output_path / f"{prefix}_quantiles.npy", quantiles)

    with open(output_path / f"{prefix}_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"Predictions saved to {output_path}/{prefix}_*")
    return output_path


def run_inference(cfg: DictConfig) -> dict:
    """
    Run zero-shot forecasting pipeline.

    Pipeline:
    1. Load data with expanding window
    2. Load pre-trained model
    3. Generate predictions
    4. Save predictions to disk

    Args:
        cfg: Hydra config

    Returns:
        Dictionary with predictions, targets, quantiles, output_path
    """
    # Set seeds
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Print run info
    print("=" * 70)
    print("ZERO-SHOT INFERENCE")
    print(f"  Model: {cfg.model.name}")
    print(f"  Data:  {cfg.data.name}")
    print(f"  Device: {cfg.device}")
    print(f"  Prediction Length: {cfg.prediction_length}")
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
        device=cfg.device,
        prediction_length=cfg.prediction_length,
    )
    adapter.load_model()

    # Load LoRA adapter if specified
    lora_adapter_path = cfg.get("lora_adapter_path")
    if lora_adapter_path:
        if cfg.model.family in {"moirai", "chronos"}:
            adapter.load_lora_adapter(lora_adapter_path)
        else:
            print(
                "  Warning: LoRA adapter loading only supported for Moirai/Chronos, "
                "ignoring lora_adapter_path"
            )

    # ========== Step 3: Generate Predictions ==========
    print("\n[3/3] Generating predictions...")
    predictions, quantiles = adapter.predict(
        prepared_data=prepared_data,
        prediction_length=cfg.prediction_length,
    )

    # ========== Save Results ==========
    targets = prepared_data["targets"]

    metadata = {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "data_metadata": prepared_data["metadata"],
        "feature_names": prepared_data["feature_names"],
        "prediction_shape": list(predictions.shape),
        "target_shape": list(targets.shape),
    }

    output_path = save_predictions(
        predictions=predictions,
        targets=targets,
        quantiles=quantiles,
        output_dir=cfg.output_dir,
        model_name=cfg.model.name,
        dataset_name=cfg.data.name,
        metadata=metadata,
    )

    print("\n" + "=" * 70)
    print("INFERENCE COMPLETE")
    print(f"  Output: {output_path}")
    print(f"  Run evaluation: python evaluate.py --results-dir {output_path}")
    print("=" * 70)

    return {
        "predictions": predictions,
        "targets": targets,
        "quantiles": quantiles,
        "output_path": output_path,
    }


@hydra.main(version_base="1.3", config_path="../../configs", config_name="inference")
def main(cfg: DictConfig):
    """Main entry point with Hydra configuration."""
    if cfg.get("print_config", False):
        print(OmegaConf.to_yaml(cfg))

    return run_inference(cfg)


if __name__ == "__main__":
    main()
