"""Tests for precision policy helpers."""

import torch

from pmf_tsfm.utils.precision import (
    enable_cuda_tf32,
    resolve_inference_precision,
    resolve_training_precision,
)


class TestEnableCudaTf32:
    def test_noop_on_non_cuda_device(self, monkeypatch) -> None:
        calls: list[str] = []
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(
            torch,
            "set_float32_matmul_precision",
            lambda mode: calls.append(mode),
        )

        enabled = enable_cuda_tf32("cpu")

        assert not enabled
        assert calls == []

    def test_enables_tf32_flags_on_cuda(self, monkeypatch) -> None:
        calls: list[str] = []
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(
            torch,
            "set_float32_matmul_precision",
            lambda mode: calls.append(mode),
        )
        monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False, raising=False)
        monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False, raising=False)

        enabled = enable_cuda_tf32("cuda")

        assert enabled
        assert calls == ["high"]
        assert torch.backends.cuda.matmul.allow_tf32 is True
        assert torch.backends.cudnn.allow_tf32 is True


class TestResolveTrainingPrecision:
    def test_moirai_defaults_to_tf32_on_cuda(self, monkeypatch) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

        policy = resolve_training_precision("moirai", requested_amp=True, device="cuda")

        assert policy.mode == "tf32"
        assert policy.use_amp is False
        assert policy.amp_dtype is None
        assert policy.tf32_enabled is True

    def test_moirai_override_can_force_tf32(self, monkeypatch) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

        policy = resolve_training_precision(
            "moirai",
            requested_amp=True,
            device="cuda",
            moirai_override="tf32",
        )

        assert policy.mode == "tf32"
        assert policy.use_amp is False
        assert policy.amp_dtype is None
        assert policy.tf32_enabled is True

    def test_moirai_override_can_force_bf16_amp(self, monkeypatch) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

        policy = resolve_training_precision(
            "moirai",
            requested_amp=False,
            device="cuda",
            moirai_override="bf16_amp",
        )

        assert policy.mode == "bf16_amp"
        assert policy.use_amp is True
        assert policy.amp_dtype == torch.bfloat16
        assert policy.tf32_enabled is True

    def test_invalid_moirai_override_raises(self, monkeypatch) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

        try:
            resolve_training_precision(
                "moirai",
                requested_amp=True,
                device="cuda",
                moirai_override="fp64",
            )
        except ValueError as exc:
            assert "Unsupported Moirai training precision override" in str(exc)
        else:
            raise AssertionError("Expected invalid Moirai precision override to raise ValueError")

    def test_non_moirai_cuda_prefers_tf32_even_when_amp_requested(self, monkeypatch) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

        policy = resolve_training_precision("chronos", requested_amp=True, device="cuda")

        assert policy.mode == "tf32"
        assert policy.use_amp is False
        assert policy.amp_dtype is None
        assert policy.tf32_enabled is True

    def test_cpu_falls_back_to_float32(self, monkeypatch) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        policy = resolve_training_precision("moirai", requested_amp=True, device="cpu")

        assert policy.mode == "float32"
        assert policy.use_amp is False
        assert policy.amp_dtype is None
        assert policy.tf32_enabled is False


class TestResolveInferencePrecision:
    def test_cuda_uses_tf32(self, monkeypatch) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

        policy = resolve_inference_precision("cuda")

        assert policy.mode == "tf32"
        assert policy.use_amp is False
        assert policy.tf32_enabled is True

    def test_cpu_uses_float32(self, monkeypatch) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        policy = resolve_inference_precision("cpu")

        assert policy.mode == "float32"
        assert policy.use_amp is False
        assert policy.tf32_enabled is False
