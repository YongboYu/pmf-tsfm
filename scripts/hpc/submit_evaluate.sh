#!/usr/bin/env bash
# =============================================================================
# submit_evaluate.sh — Submit MAE/RMSE evaluation for one or all tasks.
#
# Runs evaluate.py for zero_shot / lora_tune / full_tune.
# Optionally depends on an upstream inference job array (pass its JOBID).
#
# Usage:
#   bash scripts/hpc/submit_evaluate.sh                       # all three tasks
#   bash scripts/hpc/submit_evaluate.sh zero_shot             # one task
#   bash scripts/hpc/submit_evaluate.sh zero_shot JOBID       # with dependency
#   LOGGER=wandb bash scripts/hpc/submit_evaluate.sh
# =============================================================================
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${DIR}/hpc_env.sh"

LOGGER="${LOGGER:-wandb}"
TASK="${1:-all}"          # zero_shot | lora_tune | full_tune | all
DEP_JOBID="${2:-}"        # upstream JOBID to depend on (optional)

TIME_LIMIT="01:00:00"
MEM="32G"
CPUS=4

_submit_eval() {
    local task="$1"
    local dep="$2"

    local dep_flag=""
    if [[ -n "${dep}" ]]; then
        dep_flag="--dependency=afterany:${dep}"
    fi

    EVAL_JOBID=$(sbatch --parsable ${dep_flag} << SLURM_SCRIPT
#!/usr/bin/env bash
#SBATCH --job-name=pmf_eval_${task}
#SBATCH --cluster=${SLURM_CLUSTER}
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --output=${LOGS_DIR}/evaluate_${task}_%j.out
#SBATCH --error=${LOGS_DIR}/evaluate_${task}_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${SLURM_MAIL_USER}

source "${DIR}/hpc_env.sh"
_ensure_scratch_dirs
print_job_info

echo "Evaluating task: ${task}"

# Sync outputs from DATA → SCRATCH (in case inference ran in a previous session)
rsync -av "\${DATA_ROOT}/outputs/${task}/" "\${OUTPUTS_DIR}/${task}/" 2>/dev/null || true

cd "\${PROJECT_ROOT}"
"\${UV}" run --no-sync python -m pmf_tsfm.evaluate \\
    task=${task} \\
    results_dir="\${OUTPUTS_DIR}/${task}" \\
    logger="${LOGGER}" \\
    paths.output_dir="\${OUTPUTS_DIR}"

EXIT=\$?
sync_results_to_data
echo "Evaluation done. Exit: \${EXIT}"
exit \${EXIT}
SLURM_SCRIPT
)
    echo "Submitted eval/${task} JOBID: ${EVAL_JOBID}"
    echo "${EVAL_JOBID}"
}

if [[ "${TASK}" == "all" ]]; then
    _submit_eval "zero_shot" "${DEP_JOBID}"
    _submit_eval "lora_tune" "${DEP_JOBID}"
    _submit_eval "full_tune" "${DEP_JOBID}"
else
    _submit_eval "${TASK}" "${DEP_JOBID}"
fi
