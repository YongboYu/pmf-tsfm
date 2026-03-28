#!/usr/bin/env bash
# =============================================================================
# env.sh — Environment setup for the Linux host (RTX A6000).
# Source this file at the top of every script: source "$(dirname "$0")/env.sh"
# =============================================================================

# Resolve project root relative to this script's location
export PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Load .env (private keys, machine-specific paths) — never committed to git
if [[ -f "${PROJECT_ROOT}/.env" ]]; then
    set -a
    # shellcheck source=/dev/null
    source "${PROJECT_ROOT}/.env"
    set +a
    echo "[env] Loaded .env"
else
    echo "[env] Warning: no .env found at ${PROJECT_ROOT}/.env  (see .env.example)"
fi

# Python path
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

# GPU — override CUDA_VISIBLE_DEVICES from .env or default to GPU 0
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Suppress HuggingFace tokeniser parallelism warning
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# Use uv-managed venv if present, otherwise rely on PATH
if [[ -f "${PROJECT_ROOT}/.venv/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "${PROJECT_ROOT}/.venv/bin/activate"
    echo "[env] Activated .venv"
fi

echo "[env] PROJECT_ROOT=${PROJECT_ROOT}"
echo "[env] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[env] Python: $(python --version 2>&1)"
