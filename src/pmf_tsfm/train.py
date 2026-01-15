"""Fine-tuning training script for time series foundation models.

Supports:
- LoRA fine-tuning (task=lora_tune): Low-rank adaptation for efficient training
- Full fine-tuning (task=full_tune): All parameters trainable

Models supported:
- Moirai 1.1: small, large
- Chronos Bolt: tiny, mini, small, base
- Chronos 2.0: Uses native fit API (task=full_tune only)

Usage:
    # LoRA fine-tuning (default)
    python -m pmf_tsfm.train task=lora_tune model=moirai/1_1_small data=bpi2017
    python -m pmf_tsfm.train task=lora_tune model=chronos/bolt_small data=bpi2017

    # Full fine-tuning
    python -m pmf_tsfm.train task=full_tune model=moirai/1_1_small data=bpi2017
    python -m pmf_tsfm.train task=full_tune model=chronos/bolt_small data=bpi2017
"""

import sys
from pathlib import Path
from typing import Any, Protocol, cast

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).parent / "src"))


class TrainAdapter(Protocol):
    """Protocol describing adapters used by the training loop.

    Both LoRA and full fine-tuning adapters implement this interface.
    """

    model: torch.nn.Module | None

    def load_model(self) -> None: ...
    def to(self, device: str) -> Any: ...
    def forward_train(self, batch: dict) -> tuple: ...


class LoRATrainAdapter(TrainAdapter, Protocol):
    """Protocol for LoRA-capable adapters."""

    def apply_lora(self, lora_config: dict, context_length: int = 48) -> None: ...
    def get_trainable_parameters(self) -> tuple[int, int]: ...
    def save_lora_adapter(self, output_dir: str) -> None: ...


class FullTuneTrainAdapter(TrainAdapter, Protocol):
    """Protocol for full fine-tuning capable adapters."""

    _full_tune_model: torch.nn.Module | None

    def prepare_for_full_tuning(self, context_length: int) -> None: ...
    def get_full_tune_parameters(self) -> tuple[int, int]: ...
    def save_full_checkpoint(self, output_dir: str) -> None: ...
    def forward_train_full(self, batch: dict) -> tuple: ...


def train_epoch(
    model: LoRATrainAdapter | FullTuneTrainAdapter,
    train_loader,
    optimizer,
    device: str,
    forward_fn,
    use_amp: bool = True,
    gradient_clip: float = 1.0,
) -> float:
    """Run one training epoch."""
    trainable_model = model.model
    assert trainable_model is not None
    trainable_model.train()
    total_loss = 0.0
    num_batches = 0

    scaler: torch.cuda.amp.GradScaler | None = None
    if use_amp and device == "cuda":
        scaler = torch.cuda.amp.GradScaler()

    for batch_idx, batch in enumerate(train_loader):
        optimizer.zero_grad()

        if use_amp and device == "cuda":
            assert scaler is not None
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                _, loss = forward_fn(batch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_model.parameters(), gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            _, loss = forward_fn(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_model.parameters(), gradient_clip)
            optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if (batch_idx + 1) % 20 == 0:
            print(f"    Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")

    return total_loss / num_batches


def validate(
    model: LoRATrainAdapter | FullTuneTrainAdapter,
    val_loader,
    device: str,
    forward_fn,
    use_amp: bool = True,
) -> float:
    """Run validation."""
    trainable_model = model.model
    assert trainable_model is not None
    trainable_model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            if use_amp and device == "cuda":
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    _, loss = forward_fn(batch)
            else:
                _, loss = forward_fn(batch)

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


def train_lora(cfg: DictConfig, adapter: LoRATrainAdapter, device: str, data_module) -> dict:
    """Train with LoRA fine-tuning."""
    # Apply LoRA
    lora_config = cast(dict[str, Any], OmegaConf.to_container(cfg.lora, resolve=True))
    adapter.apply_lora(lora_config, context_length=cfg.context_length)
    adapter.to(device)

    assert adapter.model is not None
    trainable, total = adapter.get_trainable_parameters()
    print(f"  Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    # Get data loaders
    train_loader = data_module.get_train_dataloader(
        batch_size=cfg.training.batch_size, shuffle=True
    )
    val_loader = data_module.get_val_dataloader(batch_size=cfg.training.batch_size)

    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")

    # Training loop
    return _training_loop(
        cfg=cfg,
        adapter=adapter,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        forward_fn=adapter.forward_train,
        save_fn=adapter.save_lora_adapter,
        save_dir_key="adapter_save_dir",
        mode_name="LoRA",
    )


def train_full(
    cfg: DictConfig,
    adapter: FullTuneTrainAdapter,
    device: str,
    data_module,
    context_length: int,
) -> dict:
    """Train with full fine-tuning (all parameters)."""
    # Prepare for full fine-tuning
    adapter.prepare_for_full_tuning(context_length=context_length)
    adapter.to(device)

    # Get trainable model
    trainable_model = adapter._full_tune_model
    assert trainable_model is not None

    trainable, total = adapter.get_full_tune_parameters()
    print(f"  Trainable: {trainable:,} / {total:,} (100.00%)")

    if cfg.task.get("gradient_checkpointing", False):
        if hasattr(trainable_model, "gradient_checkpointing_enable"):
            print("  Enabling gradient checkpointing")
            trainable_model.gradient_checkpointing_enable()
        else:
            print("  Warning: gradient checkpointing not supported by this model")

    # Get data loaders
    train_loader = data_module.get_train_dataloader(
        batch_size=cfg.training.batch_size, shuffle=True
    )
    val_loader = data_module.get_val_dataloader(batch_size=cfg.training.batch_size)

    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")

    # Training loop
    return _training_loop(
        cfg=cfg,
        adapter=adapter,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        forward_fn=adapter.forward_train_full,
        save_fn=adapter.save_full_checkpoint,
        save_dir_key="checkpoint_save_dir",
        mode_name="Full",
    )


def _training_loop(
    cfg: DictConfig,
    adapter,
    device: str,
    train_loader,
    val_loader,
    forward_fn,
    save_fn,
    save_dir_key: str,
    mode_name: str,
) -> dict:
    """Common training loop for both LoRA and full fine-tuning."""
    print(f"\n[3/4] Training ({mode_name})...")
    print(f"  Epochs: {cfg.training.epochs}")
    print(f"  Batch size: {cfg.training.batch_size}")
    print(f"  Learning rate: {cfg.training.learning_rate}")
    print(f"  Mixed precision: {cfg.task.use_amp}")

    trainable_model = adapter.model
    assert trainable_model is not None

    optimizer = torch.optim.AdamW(
        trainable_model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )

    best_val_loss = float("inf")
    patience_counter = 0

    # Get save directory from task config
    save_dir = getattr(cfg.task, save_dir_key, "checkpoints")

    for epoch in range(cfg.training.epochs):
        print(f"\nEpoch {epoch + 1}/{cfg.training.epochs}")
        print("-" * 50)

        # Train
        train_loss = train_epoch(
            model=adapter,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            forward_fn=forward_fn,
            use_amp=cfg.task.use_amp,
            gradient_clip=cfg.training.gradient_clip,
        )
        print(f"  Train Loss: {train_loss:.4f}")

        # Validate
        val_loss = validate(
            model=adapter,
            val_loader=val_loader,
            device=device,
            forward_fn=forward_fn,
            use_amp=cfg.task.use_amp,
        )
        print(f"  Val Loss: {val_loss:.4f}")

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            best_path = Path(cfg.output_dir) / save_dir / "best"
            save_fn(str(best_path))
        else:
            patience_counter += 1
            if patience_counter >= cfg.training.early_stopping_patience:
                print(f"\n  Early stopping at epoch {epoch + 1}")
                break

        # Periodic checkpoint
        if (epoch + 1) % cfg.training.save_every == 0:
            checkpoint_path = Path(cfg.output_dir) / save_dir / f"epoch_{epoch + 1}"
            save_fn(str(checkpoint_path))

    # Save final
    print("\n[4/4] Saving final checkpoint...")
    final_path = Path(cfg.output_dir) / save_dir / "final"
    save_fn(str(final_path))

    print("\n" + "=" * 70)
    print(f"{mode_name.upper()} FINE-TUNING COMPLETE")
    print("=" * 70)
    print(f"  Best validation loss: {best_val_loss:.4f}")
    print(f"  Best checkpoint: {Path(cfg.output_dir) / save_dir / 'best'}")
    print(f"  Final checkpoint: {final_path}")

    return {
        "best_val_loss": best_val_loss,
        "final_path": str(final_path),
    }


def train(cfg: DictConfig) -> dict:
    """Main training function supporting both LoRA and full fine-tuning."""
    from pmf_tsfm.data import TrainingDataModule
    from pmf_tsfm.models import get_model_adapter

    # Set seeds
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = cfg.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        fallback_device = cfg.model.get("fallback_device", "cpu")
        print(f"Warning: CUDA not available, falling back to {fallback_device}")
        device = fallback_device
    elif device == "mps" and not torch.backends.mps.is_available():
        fallback_device = cfg.model.get("fallback_device", "cpu")
        print(f"Warning: MPS not available, falling back to {fallback_device}")
        device = fallback_device

    # Determine training mode
    task_name = cfg.task.name
    is_full_tune = task_name == "full_tune"
    mode_str = "FULL FINE-TUNING" if is_full_tune else "LORA FINE-TUNING"
    context_length = cfg.context_length
    if is_full_tune:
        context_length = int(cfg.task.get("train_context_length", context_length))

    print("=" * 70)
    print(mode_str)
    print(f"  Model: {cfg.model.name}")
    print(f"  Data:  {cfg.data.name}")
    print(f"  Device: {device}")
    print("=" * 70)

    # ========== Data ==========
    print("\n[1/4] Preparing data...")
    data_module = TrainingDataModule(
        dataset_name=cfg.data.name,
        data_path=cfg.data.path,
        context_length=context_length,
        prediction_length=cfg.prediction_length,
        train_val_test_ratio=list(cfg.data.split_ratio),
        model_family=cfg.model.family,
    )
    data_module.setup()

    # ========== Model ==========
    print(f"\n[2/4] Loading model for {mode_str.lower()}...")

    # Supported families
    supported_families = ["moirai", "chronos"]
    if cfg.model.family not in supported_families:
        raise ValueError(f"Training supports: {supported_families}, got: {cfg.model.family}")

    adapter = get_model_adapter(
        model_cfg=cfg.model,
        device=device,
        prediction_length=cfg.prediction_length,
    )
    adapter.load_model()

    # ========== Training Mode ==========
    if is_full_tune:
        # Check for Chronos 2.0 which uses native fit API
        if hasattr(adapter, "_is_chronos2") and adapter._is_chronos2:
            print("\n  Chronos 2.0 detected - using native fit() API")
            print("  Note: Native fit requires different data format, using data_module arrays")

            # Prepare data for Chronos 2.0 native fit
            train_data = data_module.train_data
            val_data = data_module.val_data

            # Convert to Chronos 2.0 format
            train_inputs = []
            if train_data is None:
                raise RuntimeError("Training data not loaded. Call data_module.setup() first.")
            train_values = train_data.to_numpy()
            for col_idx in range(train_values.shape[1]):
                train_inputs.append({"target": train_values[:, col_idx]})

            val_inputs = None
            if val_data is not None:
                val_inputs = []
                val_values = val_data.to_numpy()
                for col_idx in range(val_values.shape[1]):
                    val_inputs.append({"target": val_values[:, col_idx]})

            # Use native fit API
            adapter.fit_chronos2(
                train_inputs=train_inputs,
                validation_inputs=val_inputs,
                num_steps=cfg.training.epochs * 10,  # Convert epochs to steps
                learning_rate=cfg.training.learning_rate,
                batch_size=cfg.training.batch_size,
            )

            print("\n" + "=" * 70)
            print("CHRONOS 2.0 FINE-TUNING COMPLETE")
            print("=" * 70)
            save_dir = getattr(cfg.task, "checkpoint_save_dir", "checkpoints")
            final_path = Path(cfg.output_dir) / save_dir / "final"
            adapter.save_full_checkpoint(str(final_path))
            print(f"  Final checkpoint: {final_path}")
            return {"status": "complete", "model": "chronos2", "final_path": str(final_path)}

        # Standard full fine-tuning for Chronos Bolt and Moirai
        return train_full(
            cfg=cfg,
            adapter=cast(FullTuneTrainAdapter, adapter),
            device=device,
            data_module=data_module,
            context_length=context_length,
        )
    else:
        # LoRA fine-tuning
        return train_lora(
            cfg=cfg,
            adapter=cast(LoRATrainAdapter, adapter),
            device=device,
            data_module=data_module,
        )


@hydra.main(version_base="1.3", config_path="../../configs", config_name="train")
def main(cfg: DictConfig):
    if cfg.get("print_config", False):
        print(OmegaConf.to_yaml(cfg))

    return train(cfg)


if __name__ == "__main__":
    main()
