#!/usr/bin/env bash
# =============================================================================
# pull_metrics_from_vsc.sh — Sync lightweight HPC result summaries to a laptop.
#
# Pulls only JSON summaries from the permanent VSC outputs directory:
#   - *_metrics.json
#   - *_er.json
#   - *_er_all_summary.json
#   - *_metadata.json
#
# This avoids copying large NumPy prediction arrays and tuned checkpoints.
#
# Usage:
#   bash scripts/hpc/pull_metrics_from_vsc.sh <ssh-host-or-alias>
#   bash scripts/hpc/pull_metrics_from_vsc.sh <ssh-host-or-alias> \
#       /data/leuven/362/vsc36274/pmf-tsfm \
#       /path/to/local/sync_root
#
# Examples:
#   bash scripts/hpc/pull_metrics_from_vsc.sh vsc-login
#   PMF_TSFM_HPC_SYNC_ROOT="$PWD/data/hpc_sync" \
#       bash scripts/hpc/pull_metrics_from_vsc.sh vsc-login
# =============================================================================
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${DIR}/../.." && pwd)"

REMOTE_SPEC="${1:-${VSC_REMOTE:-}}"
REMOTE_ROOT="${2:-${VSC_REMOTE_ROOT:-/data/leuven/362/vsc36274/pmf-tsfm}}"
LOCAL_SYNC_ROOT="${3:-${PMF_TSFM_HPC_SYNC_ROOT:-${REPO_ROOT}/data/hpc_sync}}"

if [[ -z "${REMOTE_SPEC}" ]]; then
    echo "Usage: bash scripts/hpc/pull_metrics_from_vsc.sh <ssh-host-or-alias> [remote_root] [local_sync_root]"
    echo ""
    echo "Example:"
    echo "  bash scripts/hpc/pull_metrics_from_vsc.sh vsc-login"
    exit 1
fi

REMOTE_OUTPUTS="${REMOTE_ROOT%/}/outputs"
LOCAL_OUTPUTS="${LOCAL_SYNC_ROOT%/}/outputs"

mkdir -p "${LOCAL_OUTPUTS}"

echo "================================================================"
echo "  Pulling HPC metrics to local machine"
echo "  Remote: ${REMOTE_SPEC}:${REMOTE_OUTPUTS}"
echo "  Local:  ${LOCAL_OUTPUTS}"
echo "================================================================"

rsync -av --prune-empty-dirs \
    --include='*/' \
    --include='*_metrics.json' \
    --include='*_metadata.json' \
    --include='*_er.json' \
    --include='*_er_all_summary.json' \
    --exclude='*' \
    "${REMOTE_SPEC}:${REMOTE_OUTPUTS}/" \
    "${LOCAL_OUTPUTS}/"

echo ""
echo "Sync complete."
echo "Open notebooks/hpc_results_vs_baselines.ipynb to inspect the pulled metrics."
