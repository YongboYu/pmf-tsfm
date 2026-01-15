"""LoRA fine-tuning mixin for model adapters.

Provides LoRA (Low-Rank Adaptation) capabilities that can be mixed into
adapter classes via multiple inheritance.

Usage:
    class MyAdapter(BaseAdapter, LoRAMixin):
        def _create_base_model_for_lora(self, context_length):
            # Return model to wrap with LoRA
            ...
"""

from abc import abstractmethod
from pathlib import Path
from typing import Any


class LoRAMixin:
    """Mixin providing LoRA fine-tuning capabilities.

    Subclasses must implement:
        - _create_base_model_for_lora(context_length) -> model to wrap
        - _get_default_lora_targets() -> list of target module names

    Expects base class to provide:
        - device: str
        - prediction_length: int
        - _is_loaded: bool
    """

    # State attributes (initialized in subclass __init__)
    _lora_applied: bool = False
    _lora_adapter_path: str | None = None
    _peft_model: Any = None
    _full_tune_enabled: bool = False
    _full_tune_model: Any | None = None

    @abstractmethod
    def _create_base_model_for_lora(self, context_length: int) -> Any:
        """Create the base model to wrap with LoRA.

        Args:
            context_length: Fixed context length for training

        Returns:
            Model instance ready for PEFT wrapping
        """
        raise NotImplementedError

    @abstractmethod
    def _get_default_lora_targets(self) -> list[str]:
        """Return default target module names for this model family.

        Returns:
            List of module name patterns to apply LoRA to
        """
        raise NotImplementedError

    def apply_lora(self, lora_config: dict, context_length: int = 48) -> None:
        """Apply LoRA adaptation to the model for fine-tuning.

        Args:
            lora_config: Dict with keys: r, alpha, dropout, target_modules, bias
            context_length: Fixed context length for training
        """
        if not getattr(self, "_is_loaded", False):
            raise RuntimeError("Model not loaded. Call load_model() first.")

        from peft import LoraConfig, get_peft_model

        # Create base model for LoRA wrapping
        base_model = self._create_base_model_for_lora(context_length)

        # Get target modules (use config or defaults)
        target_modules = lora_config.get("target_modules")
        if target_modules is None:
            target_modules = self._get_default_lora_targets()

        peft_config = LoraConfig(
            r=lora_config.get("r", 8),
            lora_alpha=lora_config.get("alpha", 16),
            lora_dropout=lora_config.get("dropout", 0.0),
            target_modules=list(target_modules),
            bias=lora_config.get("bias", "none"),
            task_type=None,
        )

        print("Applying LoRA...")
        print(f"  Rank: {peft_config.r}")
        print(f"  Alpha: {peft_config.lora_alpha}")
        print(f"  Target modules: {peft_config.target_modules}")

        self._peft_model = get_peft_model(base_model, peft_config)
        self._peft_model.print_trainable_parameters()
        self._lora_applied = True

    def load_lora_adapter(self, adapter_path: str, context_length: int = 48) -> None:
        """Load a pre-trained LoRA adapter for inference.

        Args:
            adapter_path: Path to the saved LoRA adapter
            context_length: Context length for the model (should match training)
        """
        if not getattr(self, "_is_loaded", False):
            raise RuntimeError("Model not loaded. Call load_model() first.")

        from peft import PeftModel

        # Create base model
        base_model = self._create_base_model_for_lora(context_length)

        print(f"Loading LoRA adapter from: {adapter_path}")
        self._peft_model = PeftModel.from_pretrained(base_model, adapter_path)
        self._lora_applied = True
        self._lora_adapter_path = adapter_path
        print("  LoRA adapter loaded successfully")

    def save_lora_adapter(self, output_dir: str) -> None:
        """Save the LoRA adapter weights.

        Args:
            output_dir: Directory to save the adapter
        """
        if not self._lora_applied or self._peft_model is None:
            raise RuntimeError("No LoRA adapter to save. Call apply_lora() first.")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"Saving LoRA adapter to: {output_dir}")
        self._peft_model.save_pretrained(output_dir)
        print("  Adapter saved successfully")

    def get_trainable_parameters(self) -> tuple[int, int]:
        """Get count of trainable and total parameters.

        Returns:
            Tuple of (trainable_params, total_params)
        """
        if self._peft_model is None:
            raise RuntimeError("No PEFT model available. Call apply_lora() first.")

        trainable = sum(p.numel() for p in self._peft_model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self._peft_model.parameters())
        return trainable, total

    @property
    def model(self) -> Any:
        """Return the trainable model (PEFT model if LoRA applied)."""
        if (
            getattr(self, "_full_tune_enabled", False)
            and getattr(self, "_full_tune_model", None) is not None
        ):
            return self._full_tune_model
        if self._peft_model is not None:
            return self._peft_model
        base_model = getattr(self, "pipeline", None) or getattr(self, "base_module", None)
        if base_model is not None:
            return base_model
        return getattr(self, "_model", None)

    @model.setter
    def model(self, value: Any) -> None:
        """Allow base adapters to set a model attribute without conflicting with LoRA."""
        self._model = value

    def lora_to(self, device: str) -> None:
        """Move LoRA model to device."""
        if self._peft_model is not None:
            self._peft_model = self._peft_model.to(device)
