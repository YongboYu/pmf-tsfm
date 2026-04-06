#!/usr/bin/env bash
# =============================================================================
# submit_lora.sh — Submit LoRA training + inference as chained Slurm arrays.
#
# 4 models × 4 datasets = 16 array tasks per stage.
# Inference array starts only after each training task completes (aftercorr).
# If a training task fails, its corresponding inference task is skipped.
#
# Usage:
#   bash scripts/hpc/submit_lora.sh              # submit train + infer
#   bash scripts/hpc/submit_lora.sh train_only   # submit train only
#   LOGGER=wandb bash scripts/hpc/submit_lora.sh
#
# Returns:
#   Prints TRAIN_JOBID INFER_JOBID (used by submit_pipeline.sh).
# =============================================================================
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${DIR}/hpc_env.sh"

LOGGER="${LOGGER:-wandb}"
MODE="${1:-full}"               # "full" or "train_only"

# Fine-tunable models and datasets
MODELS=("chronos/bolt_small" "chronos/bolt_base" "moirai/1_1_small" "moirai/1_1_large")
DATASETS=("bpi2017" "bpi2019_1" "sepsis" "hospital_billing")
N_DATASETS=${#DATASETS[@]}
N_TASKS=$(( ${#MODELS[@]} * N_DATASETS ))  # 16

TRAIN_TIME="01:00:00"   # Linux runs finish in <8min; 1h covers H100 + data sync overhead
INFER_TIME="00:30:00"
MEM="64G"
CPUS=8

echo "Submitting LoRA arrays: ${#MODELS[@]} models × ${N_DATASETS} datasets = ${N_TASKS} tasks"

# ── Inline model/dataset arrays for embedding in SLURM heredoc ───────────────
MODELS_STR="\"chronos/bolt_small\" \"chronos/bolt_base\" \"moirai/1_1_small\" \"moirai/1_1_large\""
DATASETS_STR="\"bpi2017\" \"bpi2019_1\" \"sepsis\" \"hospital_billing\""

# ─────────────────────────────────────────────────────────────────────────────
# Stage A: LoRA training array
# ─────────────────────────────────────────────────────────────────────────────
TRAIN_JOBID=$(sbatch --parsable << SLURM_SCRIPT
#!/usr/bin/env bash
#SBATCH --job-name=pmf_lora_train
#SBATCH --cluster=${SLURM_CLUSTER}
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --partition=${SLURM_PARTITION}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --gpus-per-node=1
#SBATCH --mem=${MEM}
#SBATCH --time=${TRAIN_TIME}
#SBATCH --array=0-$(( N_TASKS - 1 ))%8
#SBATCH --output=${LOGS_DIR}/lora_train_%A_%a.out
#SBATCH --error=${LOGS_DIR}/lora_train_%A_%a.err
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

echo "Task \${SLURM_ARRAY_TASK_ID}: LoRA train model=\${MODEL} data=\${DATASET}"

sync_data_to_scratch

cd "\${PROJECT_ROOT}"
TRAIN_ARGS=(
    "device=cuda"
    "model=\${MODEL}"
    "data=\${DATASET}"
    "task=lora_tune"
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

echo "Submitted LoRA train array JOBID: ${TRAIN_JOBID}"

if [[ "${MODE}" == "train_only" ]]; then
    echo "${TRAIN_JOBID}"
    exit 0
fi

# ─────────────────────────────────────────────────────────────────────────────
# Stage B: LoRA inference array  (aftercorr = task i waits for train task i)
# ─────────────────────────────────────────────────────────────────────────────
INFER_JOBID=$(sbatch --parsable --dependency=aftercorr:${TRAIN_JOBID} << SLURM_SCRIPT
#!/usr/bin/env bash
#SBATCH --job-name=pmf_lora_infer
#SBATCH --cluster=${SLURM_CLUSTER}
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --partition=${SLURM_PARTITION}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --gpus-per-node=1
#SBATCH --mem=${MEM}
#SBATCH --time=${INFER_TIME}
#SBATCH --array=0-$(( N_TASKS - 1 ))%8
#SBATCH --output=${LOGS_DIR}/lora_infer_%A_%a.out
#SBATCH --error=${LOGS_DIR}/lora_infer_%A_%a.err
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

# Derive dataset display name (matches results directory naming)
declare -A DATASET_NAMES=(
    ["bpi2017"]="BPI2017" ["bpi2019_1"]="BPI2019_1"
    ["sepsis"]="Sepsis" ["hospital_billing"]="Hospital_Billing"
)
DATA_NAME="\${DATASET_NAMES[\${DATASET}]}"

ADAPTER_PATH="\${RESULTS_DIR}/lora_tune/\${DATA_NAME}/\${MODEL_LABEL}/lora_adapter/best"

echo "Task \${SLURM_ARRAY_TASK_ID}: LoRA infer model=\${MODEL} data=\${DATASET}"
echo "  Adapter: \${ADAPTER_PATH}"

# Sync refreshed data + results (adapter may be in DATA if scratch was cleaned)
rsync -av --progress "\${DATA_ROOT}/results/lora_tune/" "\${RESULTS_DIR}/lora_tune/" 2>/dev/null || true
sync_data_to_scratch

if [[ ! -d "\${ADAPTER_PATH}" ]]; then
    echo "ERROR: Adapter not found at \${ADAPTER_PATH}. Training may have failed."
    exit 1
fi

cd "\${PROJECT_ROOT}"
INFER_ARGS=(
    "device=cuda"
    "model=\${MODEL}"
    "data=\${DATASET}"
    "task=lora_tune"
    "lora_adapter_path=\${ADAPTER_PATH}"
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

echo "Submitted LoRA infer array JOBID: ${INFER_JOBID} (depends on ${TRAIN_JOBID})"
echo "${TRAIN_JOBID} ${INFER_JOBID}"
