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
#       /data/leuven/.../pmf-tsfm \
#       /path/to/local/sync_root \
#       moirai-bf16
#
# Examples:
#   bash scripts/hpc/pull_metrics_from_vsc.sh vsc-login
#   VSC_REMOTE_ROOT=/data/leuven/.../pmf-tsfm \
#       bash scripts/hpc/pull_metrics_from_vsc.sh vsc-login
#   PMF_TSFM_HPC_SYNC_ROOT="$PWD/data/hpc_sync" \
#       bash scripts/hpc/pull_metrics_from_vsc.sh vsc-login
# =============================================================================
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${DIR}/../.." && pwd)"

if [[ -f "${REPO_ROOT}/.env" ]]; then
    set -a
    # shellcheck source=/dev/null
    source "${REPO_ROOT}/.env"
    set +a
fi

REMOTE_SPEC="${1:-${VSC_REMOTE:-}}"
REMOTE_ROOT="${2:-${VSC_REMOTE_ROOT:-}}"
LOCAL_SYNC_ROOT="${3:-${PMF_TSFM_HPC_SYNC_ROOT:-${REPO_ROOT}/data/hpc_sync}}"
RUN_SUFFIX="${4:-${HPC_RUN_SUFFIX:-}}"
RUN_SUFFIX_SAFE="${RUN_SUFFIX//[^[:alnum:]._-]/_}"
RUN_PATH_SUFFIX=""
if [[ -n "${RUN_SUFFIX_SAFE}" ]]; then
    RUN_PATH_SUFFIX="-${RUN_SUFFIX_SAFE}"
fi

if [[ -z "${REMOTE_SPEC}" ]]; then
    echo "Usage: bash scripts/hpc/pull_metrics_from_vsc.sh <ssh-host-or-alias> [remote_root] [local_sync_root] [run_suffix]"
    echo ""
    echo "Example:"
    echo "  bash scripts/hpc/pull_metrics_from_vsc.sh vsc-login"
    exit 1
fi

if [[ -z "${REMOTE_ROOT}" ]]; then
    echo "Remote root not set."
    echo "Pass it as the second argument or set VSC_REMOTE_ROOT in your environment/.env."
    echo ""
    echo "Example:"
    echo "  VSC_REMOTE_ROOT=/data/leuven/.../pmf-tsfm bash scripts/hpc/pull_metrics_from_vsc.sh vsc-login"
    exit 1
fi

REMOTE_OUTPUTS="${REMOTE_ROOT%/}/outputs${RUN_PATH_SUFFIX}"
LOCAL_OUTPUTS="${LOCAL_SYNC_ROOT%/}/outputs${RUN_PATH_SUFFIX}"

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
