"""LoRA fine-tuning training script for time series foundation models.

Supports:
- Moirai 1.1: small, large
- Chronos Bolt: tiny, mini, small, base

Usage:
    python -m pmf_tsfm.train model=moirai/1_1_small data=bpi2017
    python -m pmf_tsfm.train model=chronos/bolt_small data=bpi2017
    python -m pmf_tsfm.train model=moirai/1_1_large data=bpi2017 training.epochs=5
"""

import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))


def train_epoch(
    model,
    train_loader,
    optimizer,
    device: str,
    use_amp: bool = True,
    gradient_clip: float = 1.0,
) -> float:
    """Run one training epoch."""
    model.model.train()
    total_loss = 0.0
    num_batches = 0

    scaler = torch.cuda.amp.GradScaler() if use_amp and device == "cuda" else None

    for batch_idx, batch in enumerate(train_loader):
        optimizer.zero_grad()

        if use_amp and device == "cuda":
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                _, loss = model.forward_train(batch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.model.parameters(), gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            _, loss = model.forward_train(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.model.parameters(), gradient_clip)
            optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if (batch_idx + 1) % 20 == 0:
            print(f"    Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")

    return total_loss / num_batches


def validate(
    model,
    val_loader,
    device: str,
    use_amp: bool = True,
) -> float:
    """Run validation."""
    model.model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            if use_amp and device == "cuda":
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    _, loss = model.forward_train(batch)
            else:
                _, loss = model.forward_train(batch)

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


def train(cfg: DictConfig) -> dict:
    """Main training function."""
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

    print("=" * 70)
    print("LORA FINE-TUNING")
    print(f"  Model: {cfg.model.name}")
    print(f"  Data:  {cfg.data.name}")
    print(f"  Device: {device}")
    print("=" * 70)

    # ========== Data ==========
    print("\n[1/4] Preparing data...")
    data_module = TrainingDataModule(
        dataset_name=cfg.data.name,
        data_path=cfg.data.path,
        context_length=cfg.context_length,
        prediction_length=cfg.prediction_length,
        train_val_test_ratio=list(cfg.data.split_ratio),
        model_family=cfg.model.family,  # Pass model family for correct batch format
    )
    data_module.setup()

    train_loader = data_module.get_train_dataloader(
        batch_size=cfg.training.batch_size,
        shuffle=True,
    )
    val_loader = data_module.get_val_dataloader(
        batch_size=cfg.training.batch_size,
    )

    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")

    # ========== Model ==========
    print("\n[2/4] Loading model and applying LoRA...")

    # Supported families for LoRA
    supported_families = ["moirai", "chronos"]
    if cfg.model.family not in supported_families:
        raise ValueError(f"LoRA training supports: {supported_families}, got: {cfg.model.family}")

    adapter = get_model_adapter(
        model_cfg=cfg.model,
        device=device,
        prediction_length=cfg.prediction_length,
    )
    adapter.load_model()

    # Apply LoRA
    lora_config = OmegaConf.to_container(cfg.lora, resolve=True)
    adapter.apply_lora(lora_config, context_length=cfg.context_length)
    adapter.to(device)

    trainable, total = adapter.get_trainable_parameters()
    print(f"  Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    # ========== Training ==========
    print("\n[3/4] Training...")
    print(f"  Epochs: {cfg.training.epochs}")
    print(f"  Batch size: {cfg.training.batch_size}")
    print(f"  Learning rate: {cfg.training.learning_rate}")
    print(f"  Mixed precision: {cfg.task.use_amp}")

    optimizer = torch.optim.AdamW(
        adapter.model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(cfg.training.epochs):
        print(f"\nEpoch {epoch + 1}/{cfg.training.epochs}")
        print("-" * 50)

        # Train
        train_loss = train_epoch(
            model=adapter,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            use_amp=cfg.task.use_amp,
            gradient_clip=cfg.training.gradient_clip,
        )
        print(f"  Train Loss: {train_loss:.4f}")

        # Validate
        val_loss = validate(
            model=adapter,
            val_loader=val_loader,
            device=device,
            use_amp=cfg.task.use_amp,
        )
        print(f"  Val Loss: {val_loss:.4f}")

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            best_adapter_path = Path(cfg.output_dir) / cfg.task.adapter_save_dir / "best"
            adapter.save_lora_adapter(str(best_adapter_path))
        else:
            patience_counter += 1
            if patience_counter >= cfg.training.early_stopping_patience:
                print(f"\n  Early stopping at epoch {epoch + 1}")
                break

        # Periodic checkpoint
        if (epoch + 1) % cfg.training.save_every == 0:
            checkpoint_path = Path(cfg.checkpoint.save_dir) / f"epoch_{epoch + 1}"
            adapter.save_lora_adapter(str(checkpoint_path))

    # ========== Save Final ==========
    print("\n[4/4] Saving final adapter...")
    final_adapter_path = Path(cfg.output_dir) / cfg.task.adapter_save_dir / "final"
    adapter.save_lora_adapter(str(final_adapter_path))

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Best validation loss: {best_val_loss:.4f}")
    print(f"  Best adapter saved to: {Path(cfg.output_dir) / cfg.task.adapter_save_dir / 'best'}")
    print(f"  Final adapter saved to: {final_adapter_path}")
    print("\nTo use for inference:")
    print(f"  python -m pmf_tsfm.inference model={cfg.model.family}/{cfg.model.variant} \\")
    print(f"      lora_adapter_path={final_adapter_path}")

    return {
        "best_val_loss": best_val_loss,
        "final_adapter_path": str(final_adapter_path),
    }


@hydra.main(version_base="1.3", config_path="../../configs", config_name="train")
def main(cfg: DictConfig):
    if cfg.get("print_config", False):
        print(OmegaConf.to_yaml(cfg))

    return train(cfg)


if __name__ == "__main__":
    main()
