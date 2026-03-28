#!/usr/bin/env bash
# =============================================================================
# run_full_pipeline.sh — Full PMF-TSFM pipeline on the Linux host.
#
# Steps:
#   1. Preprocess — parquet → train/val/test splits + metadata
#   2. Inference  — zero-shot predictions for all paper models × datasets
#   3. Evaluate   — MAE / RMSE from saved predictions
#   4. ER         — Entropic Relevance (one XES parse per dataset)
#
# Usage:
#   bash scripts/run_full_pipeline.sh              # full pipeline, W&B disabled
#   LOGGER=wandb bash scripts/run_full_pipeline.sh # full pipeline with W&B
#   SKIP_PREPROCESS=1 bash scripts/run_full_pipeline.sh  # skip step 1
#
# Each step is logged to logs/pipeline_<timestamp>.log as well as stdout.
# =============================================================================
set -euo pipefail
source "$(dirname "$0")/env.sh"

LOGGER="${LOGGER:-disabled}"
SKIP_PREPROCESS="${SKIP_PREPROCESS:-0}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${PROJECT_ROOT}/logs"
LOG_FILE="${LOG_DIR}/pipeline_${TIMESTAMP}.log"
mkdir -p "${LOG_DIR}"

# Tee all output to a log file
exec > >(tee -a "${LOG_FILE}") 2>&1

echo ""
echo "============================================================"
echo "  PMF-TSFM Full Pipeline"
echo "  Logger     : ${LOGGER}"
echo "  Started    : $(date)"
echo "  Log file   : ${LOG_FILE}"
echo "============================================================"
echo ""

PIPELINE_START=$(date +%s)

step_time() {
    local END
    END=$(date +%s)
    echo "    Elapsed: $((END - "$1"))s"
}

# ---------------------------------------------------------------------------
# Step 1: Preprocess
# ---------------------------------------------------------------------------
if [[ "${SKIP_PREPROCESS}" == "0" ]]; then
    echo "=== Step 1/4: Preprocess ==="
    STEP_START=$(date +%s)

    python -m pmf_tsfm.data.preprocess --multirun \
        data=bpi2017,bpi2019_1,sepsis,hospital_billing

    step_time "${STEP_START}"
    echo ""
else
    echo "=== Step 1/4: Preprocess — SKIPPED ==="
    echo ""
fi

# ---------------------------------------------------------------------------
# Step 2: Zero-shot Inference
# ---------------------------------------------------------------------------
echo "=== Step 2/4: Zero-shot Inference ==="
STEP_START=$(date +%s)

bash "$(dirname "$0")/run_inference_all.sh"

step_time "${STEP_START}"
echo ""

# ---------------------------------------------------------------------------
# Step 3: MAE / RMSE Evaluation
# ---------------------------------------------------------------------------
echo "=== Step 3/4: MAE / RMSE Evaluation ==="
STEP_START=$(date +%s)

python -m pmf_tsfm.evaluate \
    task=zero_shot \
    logger="${LOGGER}"

step_time "${STEP_START}"
echo ""

# ---------------------------------------------------------------------------
# Step 4: Entropic Relevance
# ---------------------------------------------------------------------------
echo "=== Step 4/4: Entropic Relevance ==="
STEP_START=$(date +%s)

bash "$(dirname "$0")/run_er_all.sh"

step_time "${STEP_START}"
echo ""

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
PIPELINE_END=$(date +%s)
PIPELINE_ELAPSED=$((PIPELINE_END - PIPELINE_START))

echo "============================================================"
echo "  Pipeline COMPLETE"
echo "  Total time : ${PIPELINE_ELAPSED}s ($(date -ud "@${PIPELINE_ELAPSED}" "+%H:%M:%S" 2>/dev/null || echo "${PIPELINE_ELAPSED}s"))"
echo "  Finished   : $(date)"
echo "  Log file   : ${LOG_FILE}"
echo "============================================================"
