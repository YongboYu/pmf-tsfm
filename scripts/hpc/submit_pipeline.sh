#!/usr/bin/env bash
# =============================================================================
# submit_pipeline.sh — Master pipeline: submit all stages with Slurm dependencies.
#
# Stage dependency graph:
#
#   [1] zero_shot_infer (array 52)
#          │
#          ├──[2] eval_zero_shot (CPU)
#          └──[3] er_zero_shot   (CPU)
#
#   [4] lora_train (array 16)
#          │
#         [5] lora_infer (array 16, aftercorr on [4])
#                │
#               [6] eval_lora (CPU)
#
#   [7] full_tune_train (array 20)
#          │
#         [8] full_tune_infer (array 20, aftercorr on [7])
#                │
#               [9] eval_full_tune (CPU)
#
# Usage:
#   bash scripts/hpc/submit_pipeline.sh          # submit all stages
#   bash scripts/hpc/submit_pipeline.sh --skip-zero-shot
#   bash scripts/hpc/submit_pipeline.sh --only lora
#   LOGGER=wandb bash scripts/hpc/submit_pipeline.sh
#
# Flags (can combine):
#   --skip-zero-shot     skip stages 1-3
#   --skip-lora          skip stages 4-6
#   --skip-full-tune     skip stages 7-9
#   --only zero_shot     run only zero-shot stages
#   --only lora          run only LoRA stages
#   --only full_tune     run only full fine-tune stages
# =============================================================================
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${DIR}/hpc_env.sh"

export LOGGER="${LOGGER:-disabled}"

# ── Parse flags ───────────────────────────────────────────────────────────────
SKIP_ZERO_SHOT=0
SKIP_LORA=0
SKIP_FULL_TUNE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-zero-shot) SKIP_ZERO_SHOT=1 ;;
        --skip-lora)      SKIP_LORA=1 ;;
        --skip-full-tune) SKIP_FULL_TUNE=1 ;;
        --only)
            shift
            case "$1" in
                zero_shot) SKIP_LORA=1; SKIP_FULL_TUNE=1 ;;
                lora)      SKIP_ZERO_SHOT=1; SKIP_FULL_TUNE=1 ;;
                full_tune) SKIP_ZERO_SHOT=1; SKIP_LORA=1 ;;
                *) echo "Unknown --only value: $1"; exit 1 ;;
            esac
            ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
    shift
done

_ensure_scratch_dirs

echo "================================================================"
echo "  PMF-TSFM Pipeline Submission"
echo "  Logger:        ${LOGGER}"
echo "  Skip zero-shot: ${SKIP_ZERO_SHOT}"
echo "  Skip lora:      ${SKIP_LORA}"
echo "  Skip full-tune: ${SKIP_FULL_TUNE}"
echo "  Submitted at:  $(date)"
echo "================================================================"
echo ""

# Track all submitted job IDs for final summary
declare -A SUBMITTED_JOBS

# ── Stage 1-3: Zero-shot inference + evaluation + ER ─────────────────────────
if [[ "${SKIP_ZERO_SHOT}" -eq 0 ]]; then
    echo "[Stage 1] Submitting zero-shot inference array..."
    ZS_JOBID=$(bash "${DIR}/submit_zero_shot.sh" | tail -1)
    SUBMITTED_JOBS["zero_shot_infer"]="${ZS_JOBID}"
    echo ""

    echo "[Stage 2] Submitting zero-shot evaluation (depends on stage 1)..."
    ZS_EVAL_JOBID=$(bash "${DIR}/submit_evaluate.sh" zero_shot "${ZS_JOBID}" | tail -1)
    SUBMITTED_JOBS["eval_zero_shot"]="${ZS_EVAL_JOBID}"
    echo ""

    echo "[Stage 3] Submitting ER computation (depends on stage 1)..."
    ER_JOBID=$(bash "${DIR}/submit_er.sh" "${ZS_JOBID}" | tail -1)
    SUBMITTED_JOBS["er_zero_shot"]="${ER_JOBID}"
    echo ""
fi

# ── Stage 4-6: LoRA training → inference → evaluation ────────────────────────
if [[ "${SKIP_LORA}" -eq 0 ]]; then
    echo "[Stage 4-5] Submitting LoRA train + inference arrays..."
    read -r LORA_TRAIN_JOBID LORA_INFER_JOBID <<< \
        $(bash "${DIR}/submit_lora.sh" | tail -1)
    SUBMITTED_JOBS["lora_train"]="${LORA_TRAIN_JOBID}"
    SUBMITTED_JOBS["lora_infer"]="${LORA_INFER_JOBID}"
    echo ""

    echo "[Stage 6] Submitting LoRA evaluation (depends on stage 5)..."
    LORA_EVAL_JOBID=$(bash "${DIR}/submit_evaluate.sh" lora_tune "${LORA_INFER_JOBID}" | tail -1)
    SUBMITTED_JOBS["eval_lora"]="${LORA_EVAL_JOBID}"
    echo ""
fi

# ── Stage 7-9: Full fine-tuning → inference → evaluation ─────────────────────
if [[ "${SKIP_FULL_TUNE}" -eq 0 ]]; then
    echo "[Stage 7-8] Submitting full fine-tune train + inference arrays..."
    read -r FT_TRAIN_JOBID FT_INFER_JOBID <<< \
        $(bash "${DIR}/submit_full_tune.sh" | tail -1)
    SUBMITTED_JOBS["full_tune_train"]="${FT_TRAIN_JOBID}"
    SUBMITTED_JOBS["full_tune_infer"]="${FT_INFER_JOBID}"
    echo ""

    echo "[Stage 9] Submitting full-tune evaluation (depends on stage 8)..."
    FT_EVAL_JOBID=$(bash "${DIR}/submit_evaluate.sh" full_tune "${FT_INFER_JOBID}" | tail -1)
    SUBMITTED_JOBS["eval_full_tune"]="${FT_EVAL_JOBID}"
    echo ""
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo "================================================================"
echo "  Submitted Jobs"
echo "================================================================"
for stage in zero_shot_infer eval_zero_shot er_zero_shot \
             lora_train lora_infer eval_lora \
             full_tune_train full_tune_infer eval_full_tune; do
    if [[ -n "${SUBMITTED_JOBS[$stage]+x}" ]]; then
        printf "  %-22s  JOBID: %s\n" "${stage}" "${SUBMITTED_JOBS[$stage]}"
    fi
done
echo ""
echo "  Monitor with:  squeue -u \$USER --cluster=${SLURM_CLUSTER}"
echo "  Cancel all:    scancel --cluster=${SLURM_CLUSTER} -u \$USER"
echo "  View log:      tail -f ${LOGS_DIR}/zero_shot_<JOBID>_<TASKID>.out"
echo ""
echo "  When complete, results are in:"
echo "    ${DATA_ROOT}/outputs/   (metrics JSONs)"
echo "    ${DATA_ROOT}/results/   (checkpoints / adapters)"
echo "================================================================"
