#!/usr/bin/env bash
# =============================================================================
# run_full_pipeline.sh — Complete PMF-TSFM experiment pipeline.
#
# Stages:
#    1. Preprocess       — parquet → train/val/test splits + metadata (4 datasets)
#    2. Zero-shot infer  — 13 variants × 4 datasets = 52 runs
#    3. LoRA train       — 4 models × 4 datasets = 16 runs
#    4. LoRA infer       — 4 models × 4 datasets = 16 runs
#    5. Full-tune train  — 5 models × 4 datasets = 20 runs
#    6. Full-tune infer  — 5 models × 4 datasets = 20 runs
#    7. Eval zero-shot   — MAE/RMSE for zero_shot outputs
#    8. Eval LoRA        — MAE/RMSE for lora_tune outputs
#    9. Eval full-tune   — MAE/RMSE for full_tune outputs
#   10. ER zero-shot     — Entropic Relevance (one XES parse per dataset)
#
# Usage:
#   bash scripts/run_full_pipeline.sh                        # all stages
#   LOGGER=wandb bash scripts/run_full_pipeline.sh           # with W&B
#
#   # Skip individual stages (set to 1 to skip):
#   SKIP_PREPROCESS=1        bash scripts/run_full_pipeline.sh
#   SKIP_ZERO_SHOT_INFER=1   bash scripts/run_full_pipeline.sh
#   SKIP_LORA_TRAIN=1        bash scripts/run_full_pipeline.sh
#   SKIP_LORA_INFER=1        bash scripts/run_full_pipeline.sh
#   SKIP_FULL_TUNE_TRAIN=1   bash scripts/run_full_pipeline.sh
#   SKIP_FULL_TUNE_INFER=1   bash scripts/run_full_pipeline.sh
#   SKIP_EVAL_ZERO_SHOT=1    bash scripts/run_full_pipeline.sh
#   SKIP_EVAL_LORA=1         bash scripts/run_full_pipeline.sh
#   SKIP_EVAL_FULL_TUNE=1    bash scripts/run_full_pipeline.sh
#   SKIP_ER_ZERO_SHOT=1      bash scripts/run_full_pipeline.sh
#
#   # Example: infer + eval only (training already done)
#   SKIP_PREPROCESS=1 SKIP_ZERO_SHOT_INFER=1 \
#   SKIP_LORA_TRAIN=1 SKIP_FULL_TUNE_TRAIN=1 \
#   SKIP_EVAL_ZERO_SHOT=1 SKIP_ER_ZERO_SHOT=1 \
#   bash scripts/run_full_pipeline.sh
#
# All output is tee'd to logs/pipeline_<timestamp>.log.
# =============================================================================
set -uo pipefail
source "$(dirname "$0")/env.sh"

LOGGER="${LOGGER:-disabled}"

SKIP_PREPROCESS="${SKIP_PREPROCESS:-0}"
SKIP_ZERO_SHOT_INFER="${SKIP_ZERO_SHOT_INFER:-0}"
SKIP_LORA_TRAIN="${SKIP_LORA_TRAIN:-0}"
SKIP_LORA_INFER="${SKIP_LORA_INFER:-0}"
SKIP_FULL_TUNE_TRAIN="${SKIP_FULL_TUNE_TRAIN:-0}"
SKIP_FULL_TUNE_INFER="${SKIP_FULL_TUNE_INFER:-0}"
SKIP_EVAL_ZERO_SHOT="${SKIP_EVAL_ZERO_SHOT:-0}"
SKIP_EVAL_LORA="${SKIP_EVAL_LORA:-0}"
SKIP_EVAL_FULL_TUNE="${SKIP_EVAL_FULL_TUNE:-0}"
SKIP_ER_ZERO_SHOT="${SKIP_ER_ZERO_SHOT:-0}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${PROJECT_ROOT}/logs"
LOG_FILE="${LOG_DIR}/pipeline_${TIMESTAMP}.log"
mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

PIPELINE_START=$(date +%s)

echo ""
echo "================================================================"
echo "  PMF-TSFM Full Experiment Pipeline (10 stages)"
echo "  Logger              : ${LOGGER}"
echo "  Started             : $(date)"
echo "  Log file            : ${LOG_FILE}"
echo "  SKIP_PREPROCESS     : ${SKIP_PREPROCESS}"
echo "  SKIP_ZERO_SHOT_INFER: ${SKIP_ZERO_SHOT_INFER}"
echo "  SKIP_LORA_TRAIN     : ${SKIP_LORA_TRAIN}"
echo "  SKIP_LORA_INFER     : ${SKIP_LORA_INFER}"
echo "  SKIP_FULL_TUNE_TRAIN: ${SKIP_FULL_TUNE_TRAIN}"
echo "  SKIP_FULL_TUNE_INFER: ${SKIP_FULL_TUNE_INFER}"
echo "  SKIP_EVAL_ZERO_SHOT : ${SKIP_EVAL_ZERO_SHOT}"
echo "  SKIP_EVAL_LORA      : ${SKIP_EVAL_LORA}"
echo "  SKIP_EVAL_FULL_TUNE : ${SKIP_EVAL_FULL_TUNE}"
echo "  SKIP_ER_ZERO_SHOT   : ${SKIP_ER_ZERO_SHOT}"
echo "================================================================"
echo ""

step_elapsed() {
    echo "    Step elapsed: $(( $(date +%s) - "$1" ))s"
    echo ""
}

# ---------------------------------------------------------------------------
# Stage 1: Preprocess
# ---------------------------------------------------------------------------
if [[ "${SKIP_PREPROCESS}" == "0" ]]; then
    echo "=== Stage 1/10: Preprocess ==="
    S=$(date +%s)
    python -m pmf_tsfm.data.preprocess --multirun \
        data=bpi2017,bpi2019_1,sepsis,hospital_billing
    step_elapsed "$S"
else
    echo "=== Stage 1/10: Preprocess — SKIPPED ==="; echo ""
fi

# ---------------------------------------------------------------------------
# Stage 2: Zero-shot Inference (13 variants × 4 datasets)
# ---------------------------------------------------------------------------
if [[ "${SKIP_ZERO_SHOT_INFER}" == "0" ]]; then
    echo "=== Stage 2/10: Zero-shot Inference (13 × 4 = 52 runs) ==="
    S=$(date +%s)
    bash "$(dirname "$0")/run_inference_all.sh" \
        || echo "  [WARNING] Zero-shot inference exited with errors (continuing)"
    step_elapsed "$S"
else
    echo "=== Stage 2/10: Zero-shot Inference — SKIPPED ==="; echo ""
fi

# ---------------------------------------------------------------------------
# Stage 3: LoRA Train (4 models × 4 datasets)
# ---------------------------------------------------------------------------
if [[ "${SKIP_LORA_TRAIN}" == "0" ]]; then
    echo "=== Stage 3/10: LoRA Train (4 × 4 = 16 runs) ==="
    S=$(date +%s)
    SKIP_INFER=1 bash "$(dirname "$0")/run_lora_all.sh" \
        || echo "  [WARNING] LoRA training exited with errors (continuing)"
    step_elapsed "$S"
else
    echo "=== Stage 3/10: LoRA Train — SKIPPED ==="; echo ""
fi

# ---------------------------------------------------------------------------
# Stage 4: LoRA Inference (4 models × 4 datasets)
# ---------------------------------------------------------------------------
if [[ "${SKIP_LORA_INFER}" == "0" ]]; then
    echo "=== Stage 4/10: LoRA Inference (4 × 4 = 16 runs) ==="
    S=$(date +%s)
    SKIP_TRAIN=1 bash "$(dirname "$0")/run_lora_all.sh" \
        || echo "  [WARNING] LoRA inference exited with errors (continuing)"
    step_elapsed "$S"
else
    echo "=== Stage 4/10: LoRA Inference — SKIPPED ==="; echo ""
fi

# ---------------------------------------------------------------------------
# Stage 5: Full Fine-tune Train (5 models × 4 datasets)
# ---------------------------------------------------------------------------
if [[ "${SKIP_FULL_TUNE_TRAIN}" == "0" ]]; then
    echo "=== Stage 5/10: Full Fine-tune Train (5 × 4 = 20 runs) ==="
    S=$(date +%s)
    SKIP_INFER=1 bash "$(dirname "$0")/run_full_tune_all.sh" \
        || echo "  [WARNING] Full fine-tune training exited with errors (continuing)"
    step_elapsed "$S"
else
    echo "=== Stage 5/10: Full Fine-tune Train — SKIPPED ==="; echo ""
fi

# ---------------------------------------------------------------------------
# Stage 6: Full Fine-tune Inference (5 models × 4 datasets)
# ---------------------------------------------------------------------------
if [[ "${SKIP_FULL_TUNE_INFER}" == "0" ]]; then
    echo "=== Stage 6/10: Full Fine-tune Inference (5 × 4 = 20 runs) ==="
    S=$(date +%s)
    SKIP_TRAIN=1 bash "$(dirname "$0")/run_full_tune_all.sh" \
        || echo "  [WARNING] Full fine-tune inference exited with errors (continuing)"
    step_elapsed "$S"
else
    echo "=== Stage 6/10: Full Fine-tune Inference — SKIPPED ==="; echo ""
fi

# ---------------------------------------------------------------------------
# Stage 7: Eval — Zero-shot MAE/RMSE
# ---------------------------------------------------------------------------
if [[ "${SKIP_EVAL_ZERO_SHOT}" == "0" ]]; then
    echo "=== Stage 7/10: Eval Zero-shot ==="
    S=$(date +%s)
    RESULTS_PATH="${PROJECT_ROOT}/outputs/zero_shot"
    if [[ -d "${RESULTS_PATH}" ]]; then
        python -m pmf_tsfm.evaluate \
            results_dir="${RESULTS_PATH}" \
            logger="${LOGGER}" \
            logger.group="eval_zero_shot" \
            || echo "  [WARNING] Eval zero_shot failed (continuing)"
    else
        echo "  Skipping (no outputs at ${RESULTS_PATH})"
    fi
    step_elapsed "$S"
else
    echo "=== Stage 7/10: Eval Zero-shot — SKIPPED ==="; echo ""
fi

# ---------------------------------------------------------------------------
# Stage 8: Eval — LoRA MAE/RMSE
# ---------------------------------------------------------------------------
if [[ "${SKIP_EVAL_LORA}" == "0" ]]; then
    echo "=== Stage 8/10: Eval LoRA ==="
    S=$(date +%s)
    RESULTS_PATH="${PROJECT_ROOT}/outputs/lora_tune"
    if [[ -d "${RESULTS_PATH}" ]]; then
        python -m pmf_tsfm.evaluate \
            results_dir="${RESULTS_PATH}" \
            logger="${LOGGER}" \
            logger.group="eval_lora_tune" \
            || echo "  [WARNING] Eval lora_tune failed (continuing)"
    else
        echo "  Skipping (no outputs at ${RESULTS_PATH})"
    fi
    step_elapsed "$S"
else
    echo "=== Stage 8/10: Eval LoRA — SKIPPED ==="; echo ""
fi

# ---------------------------------------------------------------------------
# Stage 9: Eval — Full Fine-tune MAE/RMSE
# ---------------------------------------------------------------------------
if [[ "${SKIP_EVAL_FULL_TUNE}" == "0" ]]; then
    echo "=== Stage 9/10: Eval Full Fine-tune ==="
    S=$(date +%s)
    RESULTS_PATH="${PROJECT_ROOT}/outputs/full_tune"
    if [[ -d "${RESULTS_PATH}" ]]; then
        python -m pmf_tsfm.evaluate \
            results_dir="${RESULTS_PATH}" \
            logger="${LOGGER}" \
            logger.group="eval_full_tune" \
            || echo "  [WARNING] Eval full_tune failed (continuing)"
    else
        echo "  Skipping (no outputs at ${RESULTS_PATH})"
    fi
    step_elapsed "$S"
else
    echo "=== Stage 9/10: Eval Full Fine-tune — SKIPPED ==="; echo ""
fi

# ---------------------------------------------------------------------------
# Stage 10: Entropic Relevance — Zero-shot
# ---------------------------------------------------------------------------
if [[ "${SKIP_ER_ZERO_SHOT}" == "0" ]]; then
    echo "=== Stage 10/10: Entropic Relevance Zero-shot ==="
    S=$(date +%s)
    TASK=zero_shot bash "$(dirname "$0")/run_er_all.sh" \
        || echo "  [WARNING] ER zero_shot failed (continuing)"
    step_elapsed "$S"
else
    echo "=== Stage 10/10: ER Zero-shot — SKIPPED ==="; echo ""
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
TOTAL_ELAPSED=$(( $(date +%s) - PIPELINE_START ))
echo "================================================================"
echo "  Pipeline COMPLETE"
echo "  Total time : ${TOTAL_ELAPSED}s"
echo "  Finished   : $(date)"
echo "  Log file   : ${LOG_FILE}"
echo "================================================================"
