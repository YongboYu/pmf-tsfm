#!/usr/bin/env bash
# =============================================================================
# run_lora_all.sh — LoRA fine-tuning + inference for all applicable variants.
#
# LoRA-supported models (paper Section 4.1):
#   Chronos-Bolt: small, base
#   Moirai-1.1:   small, large
#
# Pipeline per model × dataset:
#   1. Train (LoRA, 3 epochs, AdamW lr=1e-4)
#   2. Inference using the best LoRA adapter
#
# 4 models × 4 datasets = 16 train + 16 inference runs.
#
# Usage:
#   bash scripts/run_lora_all.sh                        # train + infer
#   SKIP_TRAIN=1 bash scripts/run_lora_all.sh           # infer only
#   SKIP_INFER=1 bash scripts/run_lora_all.sh           # train only
#   LOGGER=wandb bash scripts/run_lora_all.sh
#   DATASETS="bpi2017" bash scripts/run_lora_all.sh     # single dataset
# =============================================================================
set -uo pipefail
source "$(dirname "$0")/env.sh"

LOGGER="${LOGGER:-disabled}"
DATASETS="${DATASETS:-bpi2017 bpi2019_1 sepsis hospital_billing}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_INFER="${SKIP_INFER:-0}"

# LoRA: model config path → matching LoRA config name
declare -A LORA_CONFIGS=(
    ["chronos/bolt_small"]="chronos"
    ["chronos/bolt_base"]="chronos"
    ["moirai/1_1_small"]="moirai"
    ["moirai/1_1_large"]="moirai"
)

MODELS=("chronos/bolt_small" "chronos/bolt_base" "moirai/1_1_small" "moirai/1_1_large")

# Dataset config name → canonical data.name used in output paths (must match data config)
declare -A DATASET_NAMES=(
    ["bpi2017"]="BPI2017"
    ["bpi2019_1"]="BPI2019_1"
    ["sepsis"]="Sepsis"
    ["hospital_billing"]="Hospital_Billing"
)

echo ""
echo "============================================================"
echo "  LoRA Fine-tuning + Inference — All Variants × All Datasets"
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
        LORA_CFG="${LORA_CONFIGS[$MODEL]}"
        MODEL_LABEL="${MODEL//\//_}"

        # ---- Train ----
        if [[ "${SKIP_TRAIN}" == "1" ]]; then
            TRAIN_SKIP=$(( TRAIN_SKIP + 1 ))
            TRAIN_STATUS=SKIPPED
        else
            echo "--- [$(date +%H:%M:%S)] TRAIN lora | ${DATASET} / ${MODEL_LABEL} ---"
            RUN_START=$(date +%s)

            python -m pmf_tsfm.train \
                task=lora_tune \
                device=cuda \
                model="${MODEL}" \
                data="${DATASET}" \
                lora="${LORA_CFG}" \
                logger="${LOGGER}" \
                && TRAIN_STATUS=OK || TRAIN_STATUS=FAILED

            RUN_ELAPSED=$(( $(date +%s) - RUN_START ))
            echo "    ${TRAIN_STATUS}  ${RUN_ELAPSED}s"
            [[ "${TRAIN_STATUS}" == "OK" ]] && TRAIN_OK=$(( TRAIN_OK + 1 )) || TRAIN_FAIL=$(( TRAIN_FAIL + 1 ))
        fi

        # ---- Inference with best LoRA adapter ----
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

        # Path must match output_dir in train.yaml:
        # results/lora_tune/{data.name}/{model.name}/lora_adapter/best
        ADAPTER_PATH="${PROJECT_ROOT}/results/lora_tune/${DATA_NAME}/${MODEL_LABEL}/lora_adapter/best"

        echo "--- [$(date +%H:%M:%S)] INFER lora | ${DATASET} / ${MODEL_LABEL} ---"
        RUN_START=$(date +%s)

        python -m pmf_tsfm.inference \
            task=lora_tune \
            device=cuda \
            model="${MODEL}" \
            data="${DATASET}" \
            lora_adapter_path="${ADAPTER_PATH}" \
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
echo "  LoRA pipeline complete"
echo "  Train   : ${TRAIN_OK} OK / ${TRAIN_FAIL} failed / ${TRAIN_SKIP} skipped"
echo "  Infer   : ${INFER_OK} OK / ${INFER_FAIL} failed / ${INFER_SKIP} skipped"
echo "  Total   : ${TOTAL_ELAPSED}s"
echo "  Finished: $(date)"
echo "============================================================"
