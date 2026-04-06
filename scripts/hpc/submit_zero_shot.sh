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
TIME_LIMIT="01:00:00"           # Linux runs finish in <14min; 1h covers H100 + data sync overhead
MEM="64G"
CPUS=8

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
"\${UV}" run --no-sync python -m pmf_tsfm.inference \\
    device=cuda \\
    model="\${MODEL}" \\
    data="\${DATASET}" \\
    logger="${LOGGER}" \\
    'logger.tags=[${WANDB_HOST_TAG}]' \\
    paths.output_dir="\${OUTPUTS_DIR}" \\
    paths.processed_dir="\${DATA_DIR}/processed"

INFER_EXIT=\$?

# ── Sync results back to DATA ─────────────────────────────────────────────────
sync_results_to_data

echo ""
echo "Task \${SLURM_ARRAY_TASK_ID} done. Exit: \${INFER_EXIT}"
exit \${INFER_EXIT}
SLURM_SCRIPT
)

echo "Submitted zero-shot array JOBID: ${JOBID}"
echo "${JOBID}"  # last line → captured by submit_pipeline.sh
