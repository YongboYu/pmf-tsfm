#!/usr/bin/env bash
# =============================================================================
# submit_full_tune.sh — Submit full fine-tuning + inference as chained arrays.
#
# 5 models × 4 datasets = 20 array tasks per stage.
# Inference starts per-task when training completes (aftercorr).
#
# Models: chronos_bolt_small, chronos_bolt_base, chronos_2,
#         moirai_1_1_small, moirai_1_1_large
#
# Usage:
#   bash scripts/hpc/submit_full_tune.sh              # train + infer
#   bash scripts/hpc/submit_full_tune.sh train_only
#   LOGGER=wandb bash scripts/hpc/submit_full_tune.sh
# =============================================================================
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${DIR}/hpc_env.sh"

LOGGER="${LOGGER:-wandb}"
MODE="${1:-full}"

MODELS=("chronos/bolt_small" "chronos/bolt_base" "chronos/chronos2" "moirai/1_1_small" "moirai/1_1_large")
DATASETS=("bpi2017" "bpi2019_1" "sepsis" "hospital_billing")
N_DATASETS=${#DATASETS[@]}
N_TASKS=$(( ${#MODELS[@]} * N_DATASETS ))  # 20

# Chronos-2 full fine-tune takes longer (native fit() API)
TRAIN_TIME="01:00:00"   # Linux runs finish in <9min; 1h covers H100 + data sync overhead
INFER_TIME="00:30:00"
MEM="80G"   # larger memory for bigger models
CPUS=8

MODELS_STR="\"chronos/bolt_small\" \"chronos/bolt_base\" \"chronos/chronos2\" \"moirai/1_1_small\" \"moirai/1_1_large\""
DATASETS_STR="\"bpi2017\" \"bpi2019_1\" \"sepsis\" \"hospital_billing\""

echo "Submitting full-tune arrays: ${#MODELS[@]} models × ${N_DATASETS} datasets = ${N_TASKS} tasks"

# ─────────────────────────────────────────────────────────────────────────────
# Stage A: Full fine-tuning training
# ─────────────────────────────────────────────────────────────────────────────
TRAIN_JOBID=$(sbatch --parsable << SLURM_SCRIPT
#!/usr/bin/env bash
#SBATCH --job-name=pmf_full_train
#SBATCH --cluster=${SLURM_CLUSTER}
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --partition=${SLURM_PARTITION}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --gpus-per-node=1
#SBATCH --mem=${MEM}
#SBATCH --time=${TRAIN_TIME}
#SBATCH --array=0-$(( N_TASKS - 1 ))%4
#SBATCH --output=${LOGS_DIR}/full_train_%A_%a.out
#SBATCH --error=${LOGS_DIR}/full_train_%A_%a.err
#SBATCH --mail-type=FAIL,ARRAY_TASKS
#SBATCH --mail-user=${SLURM_MAIL_USER}

source "${DIR}/hpc_env.sh"
_load_modules
_ensure_scratch_dirs
print_job_info

MODELS=(${MODELS_STR})
DATASETS=(${DATASETS_STR})
N_DATASETS=${N_DATASETS}

MODEL_IDX=\$(( SLURM_ARRAY_TASK_ID / N_DATASETS ))
DATASET_IDX=\$(( SLURM_ARRAY_TASK_ID % N_DATASETS ))
MODEL="\${MODELS[\$MODEL_IDX]}"
DATASET="\${DATASETS[\$DATASET_IDX]}"

echo "Task \${SLURM_ARRAY_TASK_ID}: Full tune train model=\${MODEL} data=\${DATASET}"

sync_data_to_scratch

cd "\${PROJECT_ROOT}"
TRAIN_ARGS=(
    "device=cuda"
    "model=\${MODEL}"
    "data=\${DATASET}"
    "task=full_tune"
    "logger=${LOGGER}"
    "logger.tags=[${WANDB_HOST_TAG}]"
    "paths.data_dir=\${DATA_DIR}/time_series"
    "paths.output_dir=\${OUTPUTS_DIR}"
    "paths.results_dir=\${RESULTS_DIR}"
    "paths.processed_dir=\${DATA_DIR}/processed"
)
run_hydra_module pmf_tsfm.train "\${TRAIN_ARGS[@]}"

EXIT=\$?
sync_results_to_data
echo "Task \${SLURM_ARRAY_TASK_ID} done. Exit: \${EXIT}"
exit \${EXIT}
SLURM_SCRIPT
)
TRAIN_JOBID=$(normalize_slurm_jobid "${TRAIN_JOBID}")

echo "Submitted full-tune train array JOBID: ${TRAIN_JOBID}"

if [[ "${MODE}" == "train_only" ]]; then
    echo "${TRAIN_JOBID}"
    exit 0
fi

# ─────────────────────────────────────────────────────────────────────────────
# Stage B: Full fine-tune inference  (aftercorr on training)
# ─────────────────────────────────────────────────────────────────────────────
INFER_JOBID=$(sbatch --parsable --dependency=aftercorr:${TRAIN_JOBID} << SLURM_SCRIPT
#!/usr/bin/env bash
#SBATCH --job-name=pmf_full_infer
#SBATCH --cluster=${SLURM_CLUSTER}
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --partition=${SLURM_PARTITION}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --gpus-per-node=1
#SBATCH --mem=${MEM}
#SBATCH --time=${INFER_TIME}
#SBATCH --array=0-$(( N_TASKS - 1 ))%4
#SBATCH --output=${LOGS_DIR}/full_infer_%A_%a.out
#SBATCH --error=${LOGS_DIR}/full_infer_%A_%a.err
#SBATCH --mail-type=FAIL,ARRAY_TASKS
#SBATCH --mail-user=${SLURM_MAIL_USER}

source "${DIR}/hpc_env.sh"
_load_modules
_ensure_scratch_dirs
print_job_info

MODELS=(${MODELS_STR})
DATASETS=(${DATASETS_STR})
N_DATASETS=${N_DATASETS}

MODEL_IDX=\$(( SLURM_ARRAY_TASK_ID / N_DATASETS ))
DATASET_IDX=\$(( SLURM_ARRAY_TASK_ID % N_DATASETS ))
MODEL="\${MODELS[\$MODEL_IDX]}"
DATASET="\${DATASETS[\$DATASET_IDX]}"
MODEL_LABEL="\${MODEL//\//_}"

declare -A DATASET_NAMES=(
    ["bpi2017"]="BPI2017" ["bpi2019_1"]="BPI2019_1"
    ["sepsis"]="Sepsis" ["hospital_billing"]="Hospital_Billing"
)
# Chronos-2 config name vs directory name differ
declare -A MODEL_NAME_MAP=(
    ["chronos_chronos2"]="chronos_2"
)
DATA_NAME="\${DATASET_NAMES[\${DATASET}]}"
RAW_DIR_NAME="\${MODEL_LABEL}"
DIR_NAME="\${MODEL_NAME_MAP[\${RAW_DIR_NAME}]:-\${RAW_DIR_NAME}}"

CKPT_PATH="\${RESULTS_DIR}/full_tune/\${DATA_NAME}/\${DIR_NAME}/checkpoints/best"

echo "Task \${SLURM_ARRAY_TASK_ID}: Full tune infer model=\${MODEL} data=\${DATASET}"
echo "  Checkpoint: \${CKPT_PATH}"

# Sync checkpoint from DATA if scratch was cleaned
rsync -av "\${DATA_ROOT}/results/full_tune/" "\${RESULTS_DIR}/full_tune/" 2>/dev/null || true
sync_data_to_scratch

if [[ ! -d "\${CKPT_PATH}" ]]; then
    echo "ERROR: Checkpoint not found at \${CKPT_PATH}. Training may have failed."
    exit 1
fi

cd "\${PROJECT_ROOT}"
INFER_ARGS=(
    "device=cuda"
    "model=\${MODEL}"
    "data=\${DATASET}"
    "task=full_tune"
    "checkpoint_path=\${CKPT_PATH}"
    "logger=${LOGGER}"
    "logger.tags=[${WANDB_HOST_TAG}]"
    "paths.data_dir=\${DATA_DIR}/time_series"
    "paths.output_dir=\${OUTPUTS_DIR}"
    "paths.processed_dir=\${DATA_DIR}/processed"
)
run_hydra_module pmf_tsfm.inference "\${INFER_ARGS[@]}"

EXIT=\$?
sync_results_to_data
echo "Task \${SLURM_ARRAY_TASK_ID} done. Exit: \${EXIT}"
exit \${EXIT}
SLURM_SCRIPT
)
INFER_JOBID=$(normalize_slurm_jobid "${INFER_JOBID}")

echo "Submitted full-tune infer array JOBID: ${INFER_JOBID} (depends on ${TRAIN_JOBID})"
echo "${TRAIN_JOBID} ${INFER_JOBID}"
