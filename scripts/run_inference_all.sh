#!/usr/bin/env bash
# =============================================================================
# run_inference_all.sh — Zero-shot inference for all paper models × all datasets.
#
# Each model-dataset pair runs sequentially (single GPU).
# Wall time is printed per run so you can estimate total runtime.
#
# Usage:
#   bash scripts/run_inference_all.sh                   # all models × all datasets
#   bash scripts/run_inference_all.sh bpi2017           # one dataset, all models
#   WANDB_MODE=online bash scripts/run_inference_all.sh # enable W&B logging
#
# Environment:
#   LOGGER      - logger config name (default: disabled; set to wandb for logging)
#   DATASETS    - space-separated list of dataset config names
#   MODELS      - space-separated list of model config paths (e.g. chronos/chronos2)
# =============================================================================
set -euo pipefail
source "$(dirname "$0")/env.sh"

LOGGER="${LOGGER:-disabled}"

# Default: all four paper datasets
DATASETS="${*:-bpi2017 bpi2019_1 sepsis hospital_billing}"

# Default: all three paper zero-shot models
MODELS=(
    "chronos/chronos2"
    "moirai/2_0_small"
    "timesfm/2_5_200m"
)

echo ""
echo "============================================================"
echo "  Zero-shot Inference — All Models × All Datasets"
echo "  Datasets : ${DATASETS}"
echo "  Models   : ${MODELS[*]}"
echo "  Logger   : ${LOGGER}"
echo "  Started  : $(date)"
echo "============================================================"
echo ""

TOTAL_START=$(date +%s)
RUNS=0
FAILED=0

for DATASET in $DATASETS; do
    for MODEL in "${MODELS[@]}"; do
        MODEL_NAME="${MODEL//\//_}"   # chronos/chronos2 → chronos_chronos2 (for display)
        echo "--- [$(date +%H:%M:%S)] ${DATASET} / ${MODEL_NAME} ---"

        RUN_START=$(date +%s)

        python -m pmf_tsfm.inference \
            device=cuda \
            model="${MODEL}" \
            data="${DATASET}" \
            logger="${LOGGER}" \
            && RUN_STATUS=OK || RUN_STATUS=FAILED

        RUN_END=$(date +%s)
        RUN_ELAPSED=$((RUN_END - RUN_START))

        if [[ "${RUN_STATUS}" == "OK" ]]; then
            echo "    OK  ${RUN_ELAPSED}s"
            RUNS=$((RUNS + 1))
        else
            echo "    FAILED  ${RUN_ELAPSED}s  (continuing...)"
            FAILED=$((FAILED + 1))
        fi
        echo ""
    done
done

TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$((TOTAL_END - TOTAL_START))

echo "============================================================"
echo "  Inference complete"
echo "  Successful: ${RUNS} / $((RUNS + FAILED))"
echo "  Total time: ${TOTAL_ELAPSED}s ($(date -ud "@${TOTAL_ELAPSED}" "+%H:%M:%S" 2>/dev/null || echo "${TOTAL_ELAPSED}s"))"
echo "  Finished: $(date)"
echo "============================================================"
