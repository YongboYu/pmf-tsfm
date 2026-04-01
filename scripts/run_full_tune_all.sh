#!/usr/bin/env bash
# =============================================================================
# run_full_tune_all.sh — Full fine-tuning + inference for all applicable variants.
#
# Full-tune-supported models (paper Section 4.1):
#   Chronos-Bolt: small, base
#   Chronos-2   (uses native fit() API)
#   Moirai-1.1:   small, large
#
# Pipeline per model × dataset:
#   1. Train (full fine-tune, batch_size=32, AdamW)
#   2. Inference using the best checkpoint
#
# 5 models × 4 datasets = 20 train + 20 inference runs.
#
# Usage:
#   bash scripts/run_full_tune_all.sh                   # train + infer
#   SKIP_TRAIN=1 bash scripts/run_full_tune_all.sh      # infer only
#   SKIP_INFER=1 bash scripts/run_full_tune_all.sh      # train only
#   LOGGER=wandb bash scripts/run_full_tune_all.sh
#   DATASETS="bpi2017" bash scripts/run_full_tune_all.sh
#
# Note: Moirai-1.1-large may require gradient_checkpointing=true on 48 GB VRAM.
#       Set via: python ... task.gradient_checkpointing=true
# =============================================================================
set -uo pipefail
source "$(dirname "$0")/env.sh"

LOGGER="${LOGGER:-disabled}"
DATASETS="${DATASETS:-bpi2017 bpi2019_1 sepsis hospital_billing}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_INFER="${SKIP_INFER:-0}"

MODELS=(
    "chronos/bolt_small"
    "chronos/bolt_base"
    "chronos/chronos2"
    "moirai/1_1_small"
    "moirai/1_1_large"
)

# Dataset config name → canonical data.name used in output paths
declare -A DATASET_NAMES=(
    ["bpi2017"]="BPI2017"
    ["bpi2019_1"]="BPI2019_1"
    ["sepsis"]="Sepsis"
    ["hospital_billing"]="Hospital_Billing"
)

echo ""
echo "============================================================"
echo "  Full Fine-tuning + Inference — All Variants × All Datasets"
echo "  Models      : ${MODELS[*]}"
echo "  Datasets    : ${DATASETS}"
echo "  Logger      : ${LOGGER}"
echo "  Skip train  : ${SKIP_TRAIN}"
echo "  Skip infer  : ${SKIP_INFER}"
echo "  Started     : $(date)"
echo "============================================================"
echo ""

TOTAL_START=$(date +%s)
TRAIN_OK=0; TRAIN_FAIL=0; TRAIN_SKIP=0
INFER_OK=0; INFER_FAIL=0; INFER_SKIP=0

for DATASET in $DATASETS; do
    DATA_NAME="${DATASET_NAMES[$DATASET]}"

    for MODEL in "${MODELS[@]}"; do
        MODEL_LABEL="${MODEL//\//_}"
        # model.name from config — must match what Hydra writes to the checkpoint path.
        # chronos/chronos2 has name: chronos_2 (not chronos_chronos2), so we need an explicit map.
        declare -A MODEL_NAME_MAP=(
            ["chronos_bolt_small"]="chronos_bolt_small"
            ["chronos_bolt_base"]="chronos_bolt_base"
            ["chronos_chronos2"]="chronos_2"
            ["moirai_1_1_small"]="moirai_1_1_small"
            ["moirai_1_1_large"]="moirai_1_1_large"
        )
        MODEL_NAME_IN_PATH="${MODEL_NAME_MAP[$MODEL_LABEL]:-$MODEL_LABEL}"

        # ---- Train ----
        if [[ "${SKIP_TRAIN}" == "1" ]]; then
            TRAIN_SKIP=$(( TRAIN_SKIP + 1 ))
            TRAIN_STATUS=SKIPPED
        else
            echo "--- [$(date +%H:%M:%S)] TRAIN full | ${DATASET} / ${MODEL_LABEL} ---"
            RUN_START=$(date +%s)

            python -m pmf_tsfm.train \
                task=full_tune \
                device=cuda \
                model="${MODEL}" \
                data="${DATASET}" \
                training.num_workers=4 \
                logger="${LOGGER}" \
                && TRAIN_STATUS=OK || TRAIN_STATUS=FAILED

            RUN_ELAPSED=$(( $(date +%s) - RUN_START ))
            echo "    ${TRAIN_STATUS}  ${RUN_ELAPSED}s"
            [[ "${TRAIN_STATUS}" == "OK" ]] && TRAIN_OK=$(( TRAIN_OK + 1 )) || TRAIN_FAIL=$(( TRAIN_FAIL + 1 ))
        fi

        # ---- Inference with best checkpoint ----
        if [[ "${SKIP_INFER}" == "1" ]]; then
            INFER_SKIP=$(( INFER_SKIP + 1 ))
            echo ""
            continue
        fi

        if [[ "${TRAIN_STATUS}" == "FAILED" ]]; then
            echo "    Skipping inference (training failed)"
            echo ""
            continue
        fi

        # Path: results/full_tune/{DATA_NAME}/{model.name}/checkpoints/best
        CKPT_PATH="${PROJECT_ROOT}/results/full_tune/${DATA_NAME}/${MODEL_NAME_IN_PATH}/checkpoints/best"

        echo "--- [$(date +%H:%M:%S)] INFER full | ${DATASET} / ${MODEL_LABEL} ---"
        RUN_START=$(date +%s)

        python -m pmf_tsfm.inference \
            task=full_tune \
            device=cuda \
            model="${MODEL}" \
            data="${DATASET}" \
            checkpoint_path="${CKPT_PATH}" \
            logger="${LOGGER}" \
            && STATUS=OK || STATUS=FAILED

        RUN_ELAPSED=$(( $(date +%s) - RUN_START ))
        echo "    ${STATUS}  ${RUN_ELAPSED}s"
        [[ "${STATUS}" == "OK" ]] && INFER_OK=$(( INFER_OK + 1 )) || INFER_FAIL=$(( INFER_FAIL + 1 ))
        echo ""
    done
done

TOTAL_ELAPSED=$(( $(date +%s) - TOTAL_START ))
echo "============================================================"
echo "  Full fine-tune pipeline complete"
echo "  Train   : ${TRAIN_OK} OK / ${TRAIN_FAIL} failed / ${TRAIN_SKIP} skipped"
echo "  Infer   : ${INFER_OK} OK / ${INFER_FAIL} failed / ${INFER_SKIP} skipped"
echo "  Total   : ${TOTAL_ELAPSED}s"
echo "  Finished: $(date)"
echo "============================================================"
