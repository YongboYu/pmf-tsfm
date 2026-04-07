"""Precision helpers for paper-aligned GPU execution."""

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
) -> PrecisionPolicy:
    """Resolve the training precision policy for a model family.

    Paper-era reference behavior was mixed:
    - Moirai fine-tuning used BF16 AMP on CUDA.
    - Other model families ran on the float32/TF32 path on H100.
    """
    tf32_enabled = enable_cuda_tf32(device)
    use_cuda = device.startswith("cuda") and torch.cuda.is_available()

    if use_cuda and requested_amp and model_family == "moirai":
        return PrecisionPolicy(
            mode="bf16_amp",
            use_amp=True,
            amp_dtype=torch.bfloat16,
            tf32_enabled=tf32_enabled,
        )

    if tf32_enabled:
        return PrecisionPolicy(mode="tf32", tf32_enabled=True)

    return PrecisionPolicy(mode="float32")


def resolve_inference_precision(device: str) -> PrecisionPolicy:
    """Resolve inference precision.

    Inference stays on the float32 execution path; on CUDA/H100 we explicitly
    enable TF32 kernels to mirror the paper-era setup more closely.
    """
    tf32_enabled = enable_cuda_tf32(device)
    if tf32_enabled:
        return PrecisionPolicy(mode="tf32", tf32_enabled=True)
    return PrecisionPolicy(mode="float32")
