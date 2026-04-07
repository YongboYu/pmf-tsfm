#!/usr/bin/env bash
# =============================================================================
# submit_er.sh — Submit Entropic Relevance computation for zero-shot results.
#
# Runs run_er_all.sh logic as a Slurm job (CPU-only, no GPU needed).
# Optionally depends on upstream zero-shot inference job array.
#
# Usage:
#   bash scripts/hpc/submit_er.sh               # submit immediately
#   bash scripts/hpc/submit_er.sh JOBID          # run after JOBID completes
#   LOGGER=wandb bash scripts/hpc/submit_er.sh
# =============================================================================
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${DIR}/hpc_env.sh"

LOGGER="${LOGGER:-wandb}"
DEP_JOBID="${1:-}"
if [[ -n "${DEP_JOBID}" ]]; then
    DEP_JOBID=$(normalize_slurm_jobid "${DEP_JOBID}")
fi

TIME_LIMIT="00:30:00"   # ER is CPU-bound; observed <15s per run, 30min covers all models
MEM="64G"               # pm4py can be memory-hungry for large datasets
CPUS=16

dep_flag=""
if [[ -n "${DEP_JOBID}" ]]; then
    dep_flag="--dependency=afterany:${DEP_JOBID}"
fi

ER_JOBID=$(sbatch --parsable ${dep_flag} << SLURM_SCRIPT
#!/usr/bin/env bash
#SBATCH --job-name=pmf_er
#SBATCH --cluster=${SLURM_CLUSTER}
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --output=${LOGS_DIR}/er_%j.out
#SBATCH --error=${LOGS_DIR}/er_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${SLURM_MAIL_USER}

source "${DIR}/hpc_env.sh"
_ensure_scratch_dirs
print_job_info

# Sync inference outputs and processed data
rsync -av "\${DATA_OUTPUTS_DIR}/zero_shot/" "\${OUTPUTS_DIR}/zero_shot/" 2>/dev/null || true
sync_data_to_scratch

echo "Running ER computation for all zero-shot models × datasets..."

cd "\${PROJECT_ROOT}"

# Use evaluate_er_all.py if available; otherwise call per-model ER via Hydra
EXIT=0
if [[ -f "src/pmf_tsfm/er/evaluate_er_all.py" ]]; then
    for DATASET in bpi2017 bpi2019_1 sepsis hospital_billing; do
        echo "  ER all: \${DATASET}"
        ER_ALL_ARGS=(
            "data=\${DATASET}"
            "task_cfg=zero_shot"
            "logger=${LOGGER}"
            "paths.data_dir=\${DATA_DIR}/time_series"
            "paths.log_dir=\${DATA_DIR}/processed_logs"
            "paths.output_dir=\${OUTPUTS_DIR}"
        )
        if ! run_hydra_module pmf_tsfm.er.evaluate_er_all "\${ER_ALL_ARGS[@]}"; then
            EXIT=1
            echo "  FAILED: \${DATASET} (continuing)"
        fi
    done
else
    for DATASET in bpi2017 bpi2019_1 sepsis hospital_billing; do
        for MODEL in \\
            chronos/bolt_tiny chronos/bolt_mini chronos/bolt_small \\
            chronos/bolt_base chronos/chronos2 \\
            moirai/1_1_small moirai/1_1_base moirai/1_1_large \\
            moirai/moe_base moirai/2_0_small \\
            timesfm/1_0_200m timesfm/2_0_500m timesfm/2_5_200m; do
            MODEL_LABEL="\${MODEL//\//_}"
            PRED_DIR="\${OUTPUTS_DIR}/zero_shot/\${DATASET}/\${MODEL_LABEL}"  # placeholder path
            echo "  ER: \${DATASET}/\${MODEL_LABEL}"
            ER_ARGS=(
                "model=\${MODEL}"
                "data=\${DATASET}"
                "logger=${LOGGER}"
                "paths.data_dir=\${DATA_DIR}/time_series"
                "paths.log_dir=\${DATA_DIR}/processed_logs"
                "paths.output_dir=\${OUTPUTS_DIR}"
            )
            if ! run_hydra_module pmf_tsfm.er.evaluate_er "\${ER_ARGS[@]}"; then
                EXIT=1
                echo "  FAILED: \${DATASET}/\${MODEL_LABEL} (continuing)"
            fi
        done
    done
fi
sync_results_to_data
echo "ER computation done. Exit: \${EXIT}"
exit \${EXIT}
SLURM_SCRIPT
)
ER_JOBID=$(normalize_slurm_jobid "${ER_JOBID}")

echo "Submitted ER job JOBID: ${ER_JOBID}"
echo "${ER_JOBID}"
