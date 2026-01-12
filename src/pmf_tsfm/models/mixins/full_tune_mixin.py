"""Full fine-tuning mixin for model adapters.

Provides full fine-tuning capabilities that can be mixed into
adapter classes via multiple inheritance.

Usage:
    class MyAdapter(BaseAdapter, FullTuneMixin):
        def _get_trainable_model(self):
            # Return the model to train
            ...
"""

from abc import abstractmethod
from pathlib import Path
from typing import Any

import torch


class FullTuneMixin:
    """Mixin providing full fine-tuning capabilities.

    Subclasses must implement:
        - _get_trainable_model() -> model to train
        - _create_base_model_for_training(context_length) -> model instance

    Expects base class to provide:
        - device: str
        - _is_loaded: bool
    """

    _full_tune_enabled: bool = False
    _checkpoint_path: str | None = None

    @abstractmethod
    def _get_trainable_model(self) -> Any:
        """Return the model instance to train.

        Returns:
            Model with parameters to optimize
        """
        pass

    @abstractmethod
    def _create_base_model_for_training(self, context_length: int) -> Any:
        """Create the base model for full fine-tuning.

        Args:
            context_length: Fixed context length for training

        Returns:
            Model instance ready for training
        """
        pass

    def prepare_for_full_tuning(self, context_length: int = 48) -> None:
        """Prepare model for full fine-tuning by unfreezing all parameters.

        Args:
            context_length: Fixed context length for training
        """
        if not getattr(self, "_is_loaded", False):
            raise RuntimeError("Model not loaded. Call load_model() first.")

        model = self._create_base_model_for_training(context_length)

        # Unfreeze all parameters
        for param in model.parameters():
            param.requires_grad = True

        self._full_tune_model = model
        self._full_tune_enabled = True

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Full fine-tuning enabled: {trainable:,} / {total:,} parameters trainable")

    def save_full_checkpoint(self, output_dir: str) -> None:
        """Save complete model checkpoint.

        Args:
            output_dir: Directory to save the checkpoint
        """
        if not self._full_tune_enabled:
            raise RuntimeError(
                "Full fine-tuning not enabled. Call prepare_for_full_tuning() first."
            )

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        checkpoint_file = output_path / "model.pt"
        print(f"Saving full checkpoint to: {checkpoint_file}")

        torch.save(self._full_tune_model.state_dict(), checkpoint_file)
        print("  Checkpoint saved successfully")

    def load_full_checkpoint(self, checkpoint_path: str, context_length: int = 48) -> None:
        """Load complete model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint directory or file
            context_length: Context length for the model
        """
        if not getattr(self, "_is_loaded", False):
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Handle both directory and file paths
        checkpoint_file = Path(checkpoint_path)
        if checkpoint_file.is_dir():
            checkpoint_file = checkpoint_file / "model.pt"

        if not checkpoint_file.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")

        # Create model and load weights
        model = self._create_base_model_for_training(context_length)

        print(f"Loading full checkpoint from: {checkpoint_file}")
        state_dict = torch.load(checkpoint_file, map_location=getattr(self, "device", "cpu"))
        model.load_state_dict(state_dict)

        self._full_tune_model = model
        self._full_tune_enabled = True
        self._checkpoint_path = str(checkpoint_file)
        print("  Checkpoint loaded successfully")

    def get_full_tune_parameters(self) -> tuple[int, int]:
        """Get count of trainable and total parameters for full tuning.

        Returns:
            Tuple of (trainable_params, total_params)
        """
        if not self._full_tune_enabled:
            raise RuntimeError("Full fine-tuning not enabled.")

        model = self._full_tune_model
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        return trainable, total

    def full_tune_to(self, device: str) -> None:
        """Move full-tune model to device."""
        if hasattr(self, "_full_tune_model") and self._full_tune_model is not None:
            self._full_tune_model = self._full_tune_model.to(device)
