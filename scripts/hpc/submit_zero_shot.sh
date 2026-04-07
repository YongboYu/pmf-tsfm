#!/usr/bin/env bash
# =============================================================================
# submit_zero_shot.sh — Submit zero-shot inference as a Slurm job array.
#
# 13 models × 4 datasets = 52 array tasks (one H100 GPU each, ~2h each).
# Each task independently syncs data, runs inference, syncs results back.
# Jobs are independent — queue instability only delays individual tasks.
#
# Usage:
#   bash scripts/hpc/submit_zero_shot.sh              # all models × datasets
#   bash scripts/hpc/submit_zero_shot.sh 0-3          # only first 4 tasks
#   LOGGER=wandb bash scripts/hpc/submit_zero_shot.sh
#
# Returns:
#   Prints the submitted JOBID (used by submit_pipeline.sh for dependencies).
# =============================================================================
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${DIR}/hpc_env.sh"

LOGGER="${LOGGER:-wandb}"
ARRAY_RANGE="${1:-0-51}"        # override to run a subset, e.g. "0-12"
TIME_LIMIT="00:20:00"           # debug: short walltime for faster scheduling; restore to 01:00:00 for full runs
MEM="40G"                       # debug: reduced memory; restore to 64G for full runs
CPUS=4                          # debug: reduced CPUs; restore to 8 for full runs

# ── Model × dataset mapping ───────────────────────────────────────────────────
# 13 models × 4 datasets; index = MODEL_IDX * 4 + DATASET_IDX
MODELS=(
    "chronos/bolt_tiny"
    "chronos/bolt_mini"
    "chronos/bolt_small"
    "chronos/bolt_base"
    "chronos/chronos2"
    "moirai/1_1_small"
    "moirai/1_1_base"
    "moirai/1_1_large"
    "moirai/moe_base"
    "moirai/2_0_small"
    "timesfm/1_0_200m"
    "timesfm/2_0_500m"
    "timesfm/2_5_200m"
)
DATASETS=("bpi2017" "bpi2019_1" "sepsis" "hospital_billing")

N_DATASETS=${#DATASETS[@]}  # 4
N_TASKS=$(( ${#MODELS[@]} * N_DATASETS ))  # 52

echo "Submitting zero-shot array: ${#MODELS[@]} models × ${N_DATASETS} datasets = ${N_TASKS} tasks"
echo "Array range: ${ARRAY_RANGE}  |  Time limit: ${TIME_LIMIT}"

# ── Embed the actual Slurm job script ────────────────────────────────────────
JOBID=$(sbatch --parsable << SLURM_SCRIPT
#!/usr/bin/env bash
#SBATCH --job-name=pmf_zero_shot
#SBATCH --cluster=${SLURM_CLUSTER}
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --partition=${SLURM_PARTITION}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --gpus-per-node=1
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --array=${ARRAY_RANGE}%16
#SBATCH --output=${LOGS_DIR}/zero_shot_%A_%a.out
#SBATCH --error=${LOGS_DIR}/zero_shot_%A_%a.err
#SBATCH --mail-type=FAIL,ARRAY_TASKS
#SBATCH --mail-user=${SLURM_MAIL_USER}

export HPC_RUN_SUFFIX="${HPC_RUN_SUFFIX}"
export MOIRAI_TRAIN_PRECISION="${MOIRAI_TRAIN_PRECISION}"
export HPC_HYDRA_VALIDATE="${HPC_HYDRA_VALIDATE}"

# ── Source environment ────────────────────────────────────────────────────────
source "${DIR}/hpc_env.sh"
_load_modules
_ensure_scratch_dirs
print_job_info

# ── Decode array index → model + dataset ────────────────────────────────────
MODELS=(
    "chronos/bolt_tiny" "chronos/bolt_mini" "chronos/bolt_small" "chronos/bolt_base"
    "chronos/chronos2"
    "moirai/1_1_small" "moirai/1_1_base" "moirai/1_1_large" "moirai/moe_base" "moirai/2_0_small"
    "timesfm/1_0_200m" "timesfm/2_0_500m" "timesfm/2_5_200m"
)
DATASETS=("bpi2017" "bpi2019_1" "sepsis" "hospital_billing")
N_DATASETS=${#DATASETS[@]}

MODEL_IDX=\$(( SLURM_ARRAY_TASK_ID / N_DATASETS ))
DATASET_IDX=\$(( SLURM_ARRAY_TASK_ID % N_DATASETS ))
MODEL="\${MODELS[\$MODEL_IDX]}"
DATASET="\${DATASETS[\$DATASET_IDX]}"
MODEL_LABEL="\${MODEL//\//_}"

echo ""
echo "Task \${SLURM_ARRAY_TASK_ID}: model=\${MODEL}  data=\${DATASET}"

# ── Sync data from DATA → SCRATCH ─────────────────────────────────────────────
sync_data_to_scratch

# ── Run inference ────────────────────────────────────────────────────────────
cd "\${PROJECT_ROOT}"
echo "Running inference command:"
echo "  model=\${MODEL} data=\${DATASET} logger=${LOGGER}"
echo "  DATA_DIR=\${DATA_DIR}"
INFER_ARGS=(
    "device=cuda"
    "model=\${MODEL}"
    "data=\${DATASET}"
    "logger=${LOGGER}"
    "logger.tags=[${WANDB_HOST_TAG}]"
    "paths.data_dir=\${DATA_DIR}/time_series"
    "paths.output_dir=\${OUTPUTS_DIR}"
    "paths.processed_dir=\${DATA_DIR}/processed"
)
run_hydra_module pmf_tsfm.inference "\${INFER_ARGS[@]}"

INFER_EXIT=\$?

# ── Sync results back to DATA ─────────────────────────────────────────────────
sync_results_to_data

echo ""
echo "Task \${SLURM_ARRAY_TASK_ID} done. Exit: \${INFER_EXIT}"
exit \${INFER_EXIT}
SLURM_SCRIPT
)
JOBID=$(normalize_slurm_jobid "${JOBID}")

echo "Submitted zero-shot array JOBID: ${JOBID}"
echo "${JOBID}"  # last line → captured by submit_pipeline.sh
