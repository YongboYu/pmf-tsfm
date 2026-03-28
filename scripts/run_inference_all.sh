#!/usr/bin/env bash
# =============================================================================
# run_inference_all.sh — Zero-shot inference for ALL model variants × datasets.
#
# 13 model variants × 4 datasets = 52 runs (single GPU, sequential).
# (timesfm/1_0_200m requires TIMESFM_V1_PATH in .env — see .env.example)
# Wall time is printed per run.
#
# Usage:
#   bash scripts/run_inference_all.sh                   # all models × all datasets
#   bash scripts/run_inference_all.sh bpi2017           # one dataset, all models
#   LOGGER=wandb bash scripts/run_inference_all.sh      # enable W&B logging
#
# Environment:
#   LOGGER    - logger config name (default: disabled)
#   DATASETS  - space-separated dataset config names (override default)
# =============================================================================
set -euo pipefail
source "$(dirname "$0")/env.sh"

LOGGER="${LOGGER:-disabled}"
DATASETS="${*:-bpi2017 bpi2019_1 sepsis hospital_billing}"

# ---------------------------------------------------------------------------
# All 13 zero-shot model variants (Chronos × 5, Moirai × 5, TimesFM × 3)
# ---------------------------------------------------------------------------
MODELS=(
    # Chronos family (5 variants)
    "chronos/bolt_tiny"
    "chronos/bolt_mini"
    "chronos/bolt_small"
    "chronos/bolt_base"
    "chronos/chronos2"
    # Moirai family (5 variants)
    "moirai/1_1_small"
    "moirai/1_1_base"
    "moirai/1_1_large"
    "moirai/2_0_small"
    "moirai/moe_base"
    # TimesFM family (3 variants)
    # timesfm/1_0_200m requires:
    #   - uv sync --extra timesfm_legacy
    #   - TIMESFM_V1_PATH=/path/to/timesfm-repo/src  (in .env)
    #   See .env.example for details.
    "timesfm/1_0_200m"
    "timesfm/2_0_500m"
    "timesfm/2_5_200m"
)

echo ""
echo "============================================================"
echo "  Zero-shot Inference — All Model Variants × All Datasets"
echo "  Models   : ${#MODELS[@]} variants"
echo "  Datasets : ${DATASETS}"
echo "  Logger   : ${LOGGER}"
echo "  Started  : $(date)"
echo "============================================================"
echo ""

TOTAL_START=$(date +%s)
RUNS=0
FAILED=0

for DATASET in $DATASETS; do
    for MODEL in "${MODELS[@]}"; do
        MODEL_LABEL="${MODEL//\//_}"
        echo "--- [$(date +%H:%M:%S)] ${DATASET} / ${MODEL_LABEL} ---"
        RUN_START=$(date +%s)

        python -m pmf_tsfm.inference \
            device=cuda \
            model="${MODEL}" \
            data="${DATASET}" \
            logger="${LOGGER}" \
            && RUN_STATUS=OK || RUN_STATUS=FAILED

        RUN_ELAPSED=$(( $(date +%s) - RUN_START ))
        if [[ "${RUN_STATUS}" == "OK" ]]; then
            echo "    OK  ${RUN_ELAPSED}s"
            RUNS=$(( RUNS + 1 ))
        else
            echo "    FAILED  ${RUN_ELAPSED}s  (continuing...)"
            FAILED=$(( FAILED + 1 ))
        fi
        echo ""
    done
done

TOTAL_ELAPSED=$(( $(date +%s) - TOTAL_START ))
echo "============================================================"
echo "  Inference complete"
echo "  Successful : ${RUNS} / $(( RUNS + FAILED ))"
echo "  Total time : ${TOTAL_ELAPSED}s"
echo "  Finished   : $(date)"
echo "============================================================"
