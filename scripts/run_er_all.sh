#!/usr/bin/env bash
# =============================================================================
# run_er_all.sh — Entropic Relevance evaluation for all datasets.
#
# Uses evaluate_er_all.py: loads each XES log ONCE then evaluates
# Truth, Training-baseline, and all discovered model predictions in one pass.
#
# Usage:
#   bash scripts/run_er_all.sh                   # all four datasets
#   bash scripts/run_er_all.sh bpi2017 sepsis    # specific datasets
#   LOGGER=wandb bash scripts/run_er_all.sh      # enable W&B logging
#
# Environment:
#   LOGGER      - logger config name (default: disabled)
#   TASK        - task type (default: zero_shot)
# =============================================================================
set -uo pipefail
source "$(dirname "$0")/env.sh"

LOGGER="${LOGGER:-disabled}"
TASK="${TASK:-zero_shot}"

DATASETS="${*:-bpi2017 bpi2019_1 sepsis hospital_billing}"

echo ""
echo "============================================================"
echo "  Entropic Relevance — All Models × All Datasets"
echo "  Datasets : ${DATASETS}"
echo "  Task     : ${TASK}"
echo "  Logger   : ${LOGGER}"
echo "  Started  : $(date)"
echo "============================================================"
echo ""

TOTAL_START=$(date +%s)
RUNS=0
FAILED=0

for DATASET in $DATASETS; do
    echo "--- [$(date +%H:%M:%S)] ER: ${DATASET} ---"
    RUN_START=$(date +%s)

    python -m pmf_tsfm.er.evaluate_er_all \
        data="${DATASET}" \
        task_cfg="${TASK}" \
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

TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$((TOTAL_END - TOTAL_START))

echo "============================================================"
echo "  ER evaluation complete"
echo "  Successful: ${RUNS} / $((RUNS + FAILED))"
echo "  Total time: ${TOTAL_ELAPSED}s ($(date -ud "@${TOTAL_ELAPSED}" "+%H:%M:%S" 2>/dev/null || echo "${TOTAL_ELAPSED}s"))"
echo "  Finished: $(date)"
echo "============================================================"
