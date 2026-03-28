#!/usr/bin/env bash
# =============================================================================
# run_full_pipeline.sh — Complete PMF-TSFM experiment pipeline.
#
# Stages:
#   1. Preprocess      — parquet → train/val/test splits + metadata (4 datasets)
#   2. Zero-shot       — inference: 13 variants × 4 datasets = 52 runs
#   3. LoRA            — train + infer: 4 variants × 4 datasets = 16+16 runs
#   4. Full fine-tune  — train + infer: 5 variants × 4 datasets = 20+20 runs
#   5. Evaluate        — MAE/RMSE for zero_shot / lora_tune / full_tune
#   6. ER              — Entropic Relevance for zero-shot (one XES parse/dataset)
#
# Usage:
#   bash scripts/run_full_pipeline.sh                    # all stages
#   LOGGER=wandb bash scripts/run_full_pipeline.sh       # with W&B
#   SKIP_PREPROCESS=1 bash scripts/run_full_pipeline.sh  # skip step 1
#   SKIP_ZERO_SHOT=1  bash scripts/run_full_pipeline.sh  # skip step 2
#   SKIP_LORA=1       bash scripts/run_full_pipeline.sh  # skip step 3
#   SKIP_FULL_TUNE=1  bash scripts/run_full_pipeline.sh  # skip step 4
#   SKIP_EVAL=1       bash scripts/run_full_pipeline.sh  # skip step 5
#   SKIP_ER=1         bash scripts/run_full_pipeline.sh  # skip step 6
#
# All output is tee'd to logs/pipeline_<timestamp>.log.
# =============================================================================
set -euo pipefail
source "$(dirname "$0")/env.sh"

LOGGER="${LOGGER:-disabled}"
SKIP_PREPROCESS="${SKIP_PREPROCESS:-0}"
SKIP_ZERO_SHOT="${SKIP_ZERO_SHOT:-0}"
SKIP_LORA="${SKIP_LORA:-0}"
SKIP_FULL_TUNE="${SKIP_FULL_TUNE:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
SKIP_ER="${SKIP_ER:-0}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${PROJECT_ROOT}/logs"
LOG_FILE="${LOG_DIR}/pipeline_${TIMESTAMP}.log"
mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

PIPELINE_START=$(date +%s)

echo ""
echo "================================================================"
echo "  PMF-TSFM Full Experiment Pipeline"
echo "  Logger         : ${LOGGER}"
echo "  Started        : $(date)"
echo "  Log file       : ${LOG_FILE}"
echo "  SKIP_PREPROCESS: ${SKIP_PREPROCESS}"
echo "  SKIP_ZERO_SHOT : ${SKIP_ZERO_SHOT}"
echo "  SKIP_LORA      : ${SKIP_LORA}"
echo "  SKIP_FULL_TUNE : ${SKIP_FULL_TUNE}"
echo "  SKIP_EVAL      : ${SKIP_EVAL}"
echo "  SKIP_ER        : ${SKIP_ER}"
echo "================================================================"
echo ""

step_elapsed() {
    echo "    Step elapsed: $(( $(date +%s) - "$1" ))s"
    echo ""
}

# ---------------------------------------------------------------------------
# Step 1: Preprocess
# ---------------------------------------------------------------------------
if [[ "${SKIP_PREPROCESS}" == "0" ]]; then
    echo "=== Step 1/6: Preprocess ==="
    S=$(date +%s)
    python -m pmf_tsfm.data.preprocess --multirun \
        data=bpi2017,bpi2019_1,sepsis,hospital_billing
    step_elapsed "$S"
else
    echo "=== Step 1/6: Preprocess — SKIPPED ==="; echo ""
fi

# ---------------------------------------------------------------------------
# Step 2: Zero-shot Inference (13 variants × 4 datasets)
# ---------------------------------------------------------------------------
if [[ "${SKIP_ZERO_SHOT}" == "0" ]]; then
    echo "=== Step 2/6: Zero-shot Inference (13 × 4 = 52 runs) ==="
    S=$(date +%s)
    bash "$(dirname "$0")/run_inference_all.sh"
    step_elapsed "$S"
else
    echo "=== Step 2/6: Zero-shot Inference — SKIPPED ==="; echo ""
fi

# ---------------------------------------------------------------------------
# Step 3: LoRA Fine-tuning + Inference (4 × 4 = 16 + 16 runs)
# ---------------------------------------------------------------------------
if [[ "${SKIP_LORA}" == "0" ]]; then
    echo "=== Step 3/6: LoRA Fine-tuning + Inference (4 × 4 runs) ==="
    S=$(date +%s)
    bash "$(dirname "$0")/run_lora_all.sh"
    step_elapsed "$S"
else
    echo "=== Step 3/6: LoRA — SKIPPED ==="; echo ""
fi

# ---------------------------------------------------------------------------
# Step 4: Full Fine-tuning + Inference (5 × 4 = 20 + 20 runs)
# ---------------------------------------------------------------------------
if [[ "${SKIP_FULL_TUNE}" == "0" ]]; then
    echo "=== Step 4/6: Full Fine-tuning + Inference (5 × 4 runs) ==="
    S=$(date +%s)
    bash "$(dirname "$0")/run_full_tune_all.sh"
    step_elapsed "$S"
else
    echo "=== Step 4/6: Full Fine-tune — SKIPPED ==="; echo ""
fi

# ---------------------------------------------------------------------------
# Step 5: MAE/RMSE Evaluation (one W&B run per task type)
# ---------------------------------------------------------------------------
if [[ "${SKIP_EVAL}" == "0" ]]; then
    echo "=== Step 5/6: MAE/RMSE Evaluation ==="
    S=$(date +%s)

    # Each task gets its own W&B run (separate results_dir, separate table)
    for TASK in zero_shot lora_tune full_tune; do
        RESULTS_PATH="${PROJECT_ROOT}/outputs/${TASK}"
        if [[ -d "${RESULTS_PATH}" ]]; then
            echo "  Evaluating ${TASK} ..."
            python -m pmf_tsfm.evaluate \
                results_dir="${RESULTS_PATH}" \
                logger="${LOGGER}" \
                logger.group="eval_${TASK}"
        else
            echo "  Skipping ${TASK} (no outputs found at ${RESULTS_PATH})"
        fi
    done
    step_elapsed "$S"
else
    echo "=== Step 5/6: Evaluation — SKIPPED ==="; echo ""
fi

# ---------------------------------------------------------------------------
# Step 6: Entropic Relevance (zero-shot only, one XES parse per dataset)
# ---------------------------------------------------------------------------
if [[ "${SKIP_ER}" == "0" ]]; then
    echo "=== Step 6/6: Entropic Relevance (zero-shot, 4 datasets) ==="
    S=$(date +%s)
    bash "$(dirname "$0")/run_er_all.sh"
    step_elapsed "$S"
else
    echo "=== Step 6/6: ER — SKIPPED ==="; echo ""
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
