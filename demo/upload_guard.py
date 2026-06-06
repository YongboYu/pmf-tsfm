"""Guard a custom-upload forecast before it ever reaches the GPU.

The live path runs user-supplied XES logs on a serverless GPU with a hard
wall-time cap (ADR-0001), so an upload is screened *up front*: logs past a byte
cap are rejected, and the heavier model families are gated to small logs. The
byte size is a cheap upfront proxy for runtime; a finer feature-count gate is a
later (#115) concern. Rejections raise :class:`UploadRejected` with a clear,
human-readable message so the GUI/agent can show *why*, not an opaque failure.

This module is deliberately gradio-free and stdlib-only.
"""

from __future__ import annotations

from pathlib import Path

MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # hard cap: logs past this are rejected outright

SMALL_LOG_BYTES = 5 * 1024 * 1024  # gate: the heavy models are offered only below this

DEFAULT_MODEL = "chronos2"  # GPU-cheap default; the only model offered for large logs
KNOWN_MODELS = {"chronos2", "moirai2", "timesfm2.5"}
# Heavier families gated to small logs so a single call stays under the GPU wall-time cap.
GATED_MODELS = {"moirai2", "timesfm2.5"}


class UploadRejected(ValueError):  # noqa: N818 - domain name reads "upload rejected", not *Error
    """An upload was refused by the guard (too large, or a gated model choice)."""


def check_upload(log: str | Path, model: str = DEFAULT_MODEL) -> str:
    """Screen an uploaded log + model choice before forecasting.

    Args:
        log:   Path to the uploaded XES log.
        model: Requested TSFM id (e.g. ``"chronos2"``).

    Returns:
        The validated model id to forecast with.

    Raises:
        UploadRejected: the model is unknown, the log exceeds the hard size cap, or a
            gated model (Moirai/TimesFM) was chosen for a log over the small-log threshold.
    """
    if model not in KNOWN_MODELS:
        raise UploadRejected(f"unknown model {model!r}; choose one of {sorted(KNOWN_MODELS)}.")
    size = Path(log).stat().st_size
    if size > MAX_UPLOAD_BYTES:
        raise UploadRejected(
            f"log is {size / 1e6:.1f} MB; the upload cap is {MAX_UPLOAD_BYTES / 1e6:.1f} MB."
        )
    if model in GATED_MODELS and size > SMALL_LOG_BYTES:
        raise UploadRejected(
            f"{model} is only available for small logs (< {SMALL_LOG_BYTES / 1e6:.1f} MB); "
            f"use {DEFAULT_MODEL} for this {size / 1e6:.1f} MB log."
        )
    return model
