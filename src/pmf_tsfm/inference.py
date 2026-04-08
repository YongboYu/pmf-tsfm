"""
Forecasting inference script — zero-shot, LoRA-adapted, and fully fine-tuned.

Generates predictions only. Use evaluate.py for metrics computation.

Usage:
    # Zero-shot (default: chronos/bolt_small on BPI2017)
    python -m pmf_tsfm.inference

    # Specific model and dataset
    python -m pmf_tsfm.inference model=chronos/bolt_base data=sepsis

    # Device override (for macOS testing)
    python -m pmf_tsfm.inference device=mps model=chronos/bolt_small data=bpi2017
    python -m pmf_tsfm.inference device=cpu model=moirai/1_1_small data=bpi2017

    # All datasets (multirun)
    python -m pmf_tsfm.inference --multirun data=bpi2017,bpi2019_1,hospital_billing,sepsis

    # After LoRA fine-tuning
    python -m pmf_tsfm.inference model=chronos/bolt_small data=bpi2017 \\
      task=lora_tune lora_adapter_path=results/lora_tune/BPI2017/chronos_bolt_small/lora_adapter/best

    # After full fine-tuning (Chronos Bolt or Moirai)
    python -m pmf_tsfm.inference model=chronos/bolt_small data=bpi2017 \\
      task=full_tune checkpoint_path=results/full_tune/BPI2017/chronos_bolt_small/checkpoints/best

    # After full fine-tuning (Chronos 2)
    python -m pmf_tsfm.inference model=chronos/chronos2 data=bpi2017 \\
      task=full_tune checkpoint_path=results/full_tune/BPI2017/chronos2/checkpoints/best
"""

import json
import time
from pathlib import Path
from typing import Protocol, cast

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from pmf_tsfm.data import ZeroShotDataModule
from pmf_tsfm.models import get_model_adapter
from pmf_tsfm.utils.precision import resolve_inference_precision
from pmf_tsfm.utils.wandb_logger import init_run


class LoRAInferenceAdapter(Protocol):
    """Protocol for adapters that support LoRA adapter loading."""

    def load_lora_adapter(self, adapter_path: str, context_length: int = 48) -> None: ...


class FullTuneInferenceAdapter(Protocol):
    """Protocol for adapters that support full fine-tune checkpoint loading."""

    def load_full_checkpoint(self, checkpoint_path: str, context_length: int = 48) -> None: ...


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
    Run forecasting pipeline (zero-shot, LoRA-adapted, or fully fine-tuned).

    Pipeline:
    1. Load data with expanding window (all prior data up to each test timestep)
    2. Load pre-trained model; optionally load LoRA adapter or full checkpoint
    3. Generate predictions
    4. Save predictions to disk

    Args:
        cfg: Hydra config

    Returns:
        Dictionary with predictions, targets, quantiles, output_path
    """
    t_start = time.perf_counter()

    # Set seeds
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    dataset_name: str = cfg.data.name
    model_name: str = cfg.model.name
    task: str = cfg.task.name if hasattr(cfg.task, "name") else cfg.task

    # Init W&B run (no-op when logger.enabled=false)
    run = init_run(
        cfg,
        job_type="inference",
        name=f"infer/{task}/{dataset_name}/{model_name}",
        tags=[model_name, dataset_name, task, cfg.model.family],
        group=f"{task}/{dataset_name}",
    )

    # Resolve device with fallback
    device = cfg.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        fallback = cfg.model.get("fallback_device", "cpu")
        print(f"Warning: CUDA not available, falling back to {fallback}")
        device = fallback
    elif device == "mps" and not torch.backends.mps.is_available():
        fallback = cfg.model.get("fallback_device", "cpu")
        print(f"Warning: MPS not available, falling back to {fallback}")
        device = fallback

    precision_policy = resolve_inference_precision(device)

    # Determine run mode
    task = cfg.task.name if hasattr(cfg.task, "name") else cfg.task
    lora_adapter_path = cfg.get("lora_adapter_path")
    checkpoint_path = cfg.get("checkpoint_path")
    adapter_context_length_cfg = cfg.get("context_length")
    if adapter_context_length_cfg is None and hasattr(cfg.task, "get"):
        adapter_context_length_cfg = cfg.task.get("train_context_length")
    adapter_context_length = (
        int(adapter_context_length_cfg) if adapter_context_length_cfg is not None else 48
    )

    if checkpoint_path:
        mode = "FULL FINE-TUNE INFERENCE"
    elif lora_adapter_path:
        mode = "LORA INFERENCE"
    else:
        mode = "ZERO-SHOT INFERENCE"

    print("=" * 70)
    print(mode)
    print(f"  Model:  {cfg.model.name}")
    print(f"  Data:   {cfg.data.name}")
    print(f"  Device: {device}")
    print(f"  Precision: {precision_policy.mode}")
    print(f"  Prediction Length: {cfg.prediction_length}")
    print(f"  Task:   {task}")
    if lora_adapter_path:
        print(f"  LoRA adapter: {lora_adapter_path}")
    if checkpoint_path:
        print(f"  Checkpoint:   {checkpoint_path}")
    print("=" * 70)

    # ========== Step 1: Prepare Data ==========
    print("\n[1/3] Preparing data...")
    # Use pre-split files from paths.processed_dir if available; otherwise
    # fall back to runtime splitting from the raw parquet.
    processed_dir: str | None = cfg.get("paths", {}).get("processed_dir")  # type: ignore[assignment]
    data_module = ZeroShotDataModule.from_config(
        data_cfg=cfg.data,
        prediction_length=cfg.prediction_length,
        processed_dir=processed_dir,
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

    # Load LoRA adapter if specified (Moirai/Chronos only)
    if lora_adapter_path:
        if cfg.model.family in {"moirai", "chronos"}:
            lora_adapter = cast(LoRAInferenceAdapter, adapter)
            lora_adapter.load_lora_adapter(
                lora_adapter_path,
                context_length=adapter_context_length,
            )
        else:
            print(
                f"  Warning: LoRA not supported for {cfg.model.family}, ignoring lora_adapter_path"
            )

    # Load full fine-tune checkpoint if specified
    if checkpoint_path:
        if cfg.model.family in {"moirai", "chronos"}:
            full_adapter = cast(FullTuneInferenceAdapter, adapter)
            full_adapter.load_full_checkpoint(
                checkpoint_path,
                context_length=adapter_context_length,
            )
        else:
            print(
                f"  Warning: Full checkpoint loading not supported for {cfg.model.family}, "
                "ignoring checkpoint_path"
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
        "mode": mode,
        "task": task,
        "lora_adapter_path": lora_adapter_path,
        "checkpoint_path": checkpoint_path,
        "config": OmegaConf.to_container(cfg, resolve=True),
        "data_metadata": prepared_data["metadata"],
        "feature_names": prepared_data["feature_names"],
        "prediction_shape": list(predictions.shape),
        "target_shape": list(targets.shape),
        "precision_mode": precision_policy.mode,
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

    elapsed = time.perf_counter() - t_start

    print("\n" + "=" * 70)
    print(f"{mode} COMPLETE")
    print(f"  Output:  {output_path}")
    print(f"  Elapsed: {elapsed:.1f}s")
    print("=" * 70)

    # All inference modes use expanding windows (all prior data as context).
    # context_length=48 is only the training context for LoRA/full-tune adapters.
    context_summary: dict = {"inference/context_window": "expanding"}
    if lora_adapter_path or checkpoint_path:
        context_summary["inference/adapter_trained_context_length"] = adapter_context_length

    run.log_summary(
        {
            "inference/elapsed_s": elapsed,
            "inference/n_windows": int(predictions.shape[0]),
            "inference/n_features": int(predictions.shape[2]),
            "inference/model": model_name,
            "inference/dataset": dataset_name,
            "inference/task": task,
            "inference/device": device,
            "inference/precision_mode": precision_policy.mode,
            **context_summary,
        }
    )
    run.finish()

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
