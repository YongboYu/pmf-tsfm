#!/usr/bin/env bash
# =============================================================================
# sync_wandb_offline.sh — Upload offline W&B runs from SCRATCH to the cloud.
#
# Run from a login node after compute jobs finish if you used LOGGER=wandb_offline.
# Useful when compute nodes have no internet, or for bulk sync after a run.
#
# Usage:
#   bash scripts/hpc/sync_wandb_offline.sh
# =============================================================================
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${DIR}/hpc_env.sh"

export PATH="${VSC_DATA}/.local/bin:${PATH}"

if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "ERROR: WANDB_API_KEY not set. Source your .env first:"
    echo "  source ${PROJECT_ROOT}/.env"
    exit 1
fi

if [[ ! -d "${WANDB_DIR}" ]]; then
    echo "No offline W&B runs found at ${WANDB_DIR}"
    exit 0
fi

echo "Syncing offline W&B runs from ${WANDB_DIR}..."
(
    cd "${PROJECT_ROOT}"
    "${UV}" run --no-sync wandb sync "${WANDB_DIR}"/offline-run-*
)
echo "Sync complete."
