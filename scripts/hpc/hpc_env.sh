#!/usr/bin/env bash
# =============================================================================
# hpc_env.sh — Cluster configuration and environment setup for VSC wICE.
#
# SOURCE this file at the top of every Slurm script:
#   source "$(dirname "$0")/hpc_env.sh"
#
# Edit the USER CONFIG section below for your account/paths.
# =============================================================================

# ── USER CONFIG ───────────────────────────────────────────────────────────────
# Slurm account and partition (KU Leuven genius / wICE)
export SLURM_ACCOUNT="lp_lirisnlp"
export SLURM_PARTITION="gpu"                    # wICE: gpu (H100) | genius: gpu_p100 (P100)
export SLURM_CLUSTER="wice"                     # wice | genius
export SLURM_MAIL_USER="yongbo.yu@student.kuleuven.be"

# W&B project — one project for all devices; runs tagged by host for filtering
export WANDB_PROJECT="pmf-tsfm"
export WANDB_ENTITY=""                          # leave empty to use default
export WANDB_HOST_TAG="${SLURM_CLUSTER}"        # "genius" or "wice" — passed as logger.tags

# TimesFM v1 source (needed for timesfm/1_0_200m and timesfm/2_0_500m)
# Clone once to VSC_DATA and point here:
#   git clone --depth 1 --branch v1.2.6 \
#       https://github.com/google-research/timesfm.git $VSC_DATA/timesfm-v1-repo
export TIMESFM_V1_PATH="${VSC_DATA}/timesfm-v1-repo/src"

# ── PROJECT PATHS ─────────────────────────────────────────────────────────────
# VSC_DATA  = permanent storage (slow); persists between sessions
# VSC_SCRATCH = fast parallel filesystem; files >21 days old may be deleted

export PROJECT_NAME="pmf-tsfm"
export DATA_ROOT="${VSC_DATA}/${PROJECT_NAME}"       # permanent storage root
export SCRATCH_ROOT="${VSC_SCRATCH}/${PROJECT_NAME}" # fast working directory

# Repo location on scratch (cloned/pulled by setup_vsc.sh)
export PROJECT_ROOT="${SCRATCH_ROOT}/repo"

# ── CACHE DIRS — CRITICAL: ALL caches must point to SCRATCH, not DATA ────────
# Putting caches in DATA is slow (GPFS metadata overhead) and wastes quota.
export HF_HOME="${VSC_SCRATCH}/.cache/huggingface"
export HF_HUB_CACHE="${VSC_SCRATCH}/.cache/huggingface/hub"
export HF_DATASETS_CACHE="${VSC_SCRATCH}/.cache/huggingface/datasets"
export TRANSFORMERS_CACHE="${VSC_SCRATCH}/.cache/huggingface/transformers"
export TORCH_HOME="${VSC_SCRATCH}/.cache/torch"
export UV_CACHE_DIR="${VSC_SCRATCH}/.cache/uv"

# W&B local dir (offline runs buffer here before sync)
export WANDB_DIR="${SCRATCH_ROOT}/wandb"

# Suppress HF tokenizer parallelism warnings
export TOKENIZERS_PARALLELISM="false"

# ── RUNTIME PATHS (derived, do not edit) ─────────────────────────────────────
export OUTPUTS_DIR="${SCRATCH_ROOT}/outputs"   # inference + eval outputs
export RESULTS_DIR="${SCRATCH_ROOT}/results"   # training checkpoints / adapters
export DATA_DIR="${SCRATCH_ROOT}/data"         # processed event log data
export LOGS_DIR="${SCRATCH_ROOT}/logs"         # Slurm stdout/stderr

# ── PYTHON / UV ───────────────────────────────────────────────────────────────
# uv binary (installed to DATA so it survives scratch cleanup)
export PATH="${VSC_DATA}/.local/bin:${PATH}"
export UV="${VSC_DATA}/.local/bin/uv"

# ── CUDA MODULES ─────────────────────────────────────────────────────────────
# wICE/gpu: H100 cards require CUDA 12.x (compute 9.0).
# Run `module spider CUDA` on the wICE login node to see available versions.
# PyTorch wheels bundle their own cuDNN — only the CUDA driver module is needed.
_load_modules() {
    module purge 2>/dev/null || true
    module load CUDA/12.6.0 2>/dev/null || \
    module load CUDA/12.4.0 2>/dev/null || \
    module load CUDA/12.3.0 2>/dev/null || \
    module load CUDA/12.1.1 2>/dev/null || \
    module load CUDA/12.0.0 2>/dev/null || \
    module load CUDA 2>/dev/null || true
}

# ── HELPER: enable HF/Transformers offline mode (opt-in) ─────────────────────
# VSC compute nodes have internet access, so this is NOT called by default.
# Call it explicitly if you need fully air-gapped, cache-only execution
# (e.g. debugging, reproducibility audits, or bypassing HF rate limits).
_set_offline_mode() {
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    echo "[env] HF offline mode enabled (using pre-cached models in ${HF_HOME})"
}

# ── HELPER: ensure scratch directories exist ──────────────────────────────────
_ensure_scratch_dirs() {
    mkdir -p \
        "${SCRATCH_ROOT}" \
        "${OUTPUTS_DIR}" \
        "${RESULTS_DIR}" \
        "${DATA_DIR}" \
        "${LOGS_DIR}" \
        "${HF_HOME}" \
        "${TORCH_HOME}" \
        "${WANDB_DIR}"
}

# ── HELPER: sync processed data from DATA → SCRATCH ──────────────────────────
# Call at start of each job to ensure data is available on fast filesystem.
sync_data_to_scratch() {
    echo "[sync] Copying processed data from DATA to SCRATCH..."
    rsync -av --progress \
        "${DATA_ROOT}/data/" \
        "${DATA_DIR}/" \
        2>&1 | tail -5
    echo "[sync] Data ready at ${DATA_DIR}"
}

# ── HELPER: sync outputs/results from SCRATCH → DATA ─────────────────────────
# Call at end of each job to persist results.
sync_results_to_data() {
    echo "[sync] Copying outputs/results from SCRATCH to DATA..."
    rsync -av \
        "${OUTPUTS_DIR}/" \
        "${DATA_ROOT}/outputs/" 2>&1 | tail -5
    rsync -av \
        "${RESULTS_DIR}/" \
        "${DATA_ROOT}/results/" 2>&1 | tail -5
    echo "[sync] Results persisted to ${DATA_ROOT}"
}

# ── HELPER: run a python module via uv in the project virtualenv ──────────────
uv_run() {
    (
        cd "${PROJECT_ROOT}" && \
        "${UV}" run --no-sync python "$@"
    )
}

# ── HELPER: print job environment summary ────────────────────────────────────
print_job_info() {
    echo "================================================================"
    echo "  Job:        ${SLURM_JOB_ID:-local}  Array: ${SLURM_ARRAY_TASK_ID:-N/A}"
    echo "  Node:       $(hostname)"
    echo "  Project:    ${PROJECT_ROOT}"
    echo "  Scratch:    ${SCRATCH_ROOT}"
    echo "  HF cache:   ${HF_HOME}"
    echo "  GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo 'N/A')"
    echo "  Started:    $(date)"
    echo "================================================================"
}
