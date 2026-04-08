"""Precision helpers for GPU execution."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PrecisionPolicy:
    """Resolved execution precision for a run."""

    mode: str
    use_amp: bool = False
    amp_dtype: torch.dtype | None = None
    tf32_enabled: bool = False


def _normalize_moirai_override(raw_override: str | None) -> str | None:
    """Normalize optional Moirai precision overrides from the environment."""
    if raw_override is None:
        return None

    normalized = raw_override.strip().lower()
    if normalized in {"", "default"}:
        return None
    if normalized in {"bf16", "bf16_amp"}:
        return "bf16_amp"
    if normalized == "tf32":
        return "tf32"

    raise ValueError(
        "Unsupported Moirai training precision override. Use one of: bf16_amp, tf32, default."
    )


def enable_cuda_tf32(device: str) -> bool:
    """Enable TF32 math on CUDA backends when available."""
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return False

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = True

    return True


def resolve_training_precision(
    model_family: str,
    requested_amp: bool,
    device: str,
    moirai_override: str | None = None,
) -> PrecisionPolicy:
    """Resolve the training precision policy for a model family.

    Current default behavior keeps all CUDA fine-tuning on the TF32 path.
    Moirai can still be explicitly overridden to BF16 AMP for comparison runs.
    """
    normalized_override = _normalize_moirai_override(moirai_override)
    tf32_enabled = enable_cuda_tf32(device)
    use_cuda = device.startswith("cuda") and torch.cuda.is_available()

    if model_family == "moirai" and use_cuda and normalized_override == "bf16_amp":
        return PrecisionPolicy(
            mode="bf16_amp",
            use_amp=True,
            amp_dtype=torch.bfloat16,
            tf32_enabled=tf32_enabled,
        )

    if model_family == "moirai" and normalized_override == "tf32":
        if tf32_enabled:
            return PrecisionPolicy(mode="tf32", tf32_enabled=True)
        return PrecisionPolicy(mode="float32")

    if tf32_enabled:
        return PrecisionPolicy(mode="tf32", tf32_enabled=True)

    return PrecisionPolicy(mode="float32")


def resolve_inference_precision(device: str) -> PrecisionPolicy:
    """Resolve inference precision.

    Inference stays on the float32 execution path; on CUDA/H100 we explicitly
    enable TF32 kernels for the default GPU path.
    """
    tf32_enabled = enable_cuda_tf32(device)
    if tf32_enabled:
        return PrecisionPolicy(mode="tf32", tf32_enabled=True)
    return PrecisionPolicy(mode="float32")
