#!/usr/bin/env bash
#SBATCH --job-name=full79-lstm-audit
#SBATCH --output=slurm/logs/full79-lstm-audit-%j.out
#SBATCH --error=slurm/logs/full79-lstm-audit-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --partition=long-cpu

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$HOME/communicative_efficiency_repos/generate_baselines_mila}}"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
: "${RUN_ROOT:?RUN_ROOT is required}"
: "${AUDIT_STAGE:?AUDIT_STAGE is required}"
: "${CELL_INDICES:?CELL_INDICES is required}"
PYTHON_CMD="${PYTHON_CMD:-${PYTHON:-python3}}"
read -r -a PYTHON_RUN <<< "$PYTHON_CMD"
export PYTHONPYCACHEPREFIX="$RUN_ROOT/pycache/audit/$AUDIT_STAGE"

"${PYTHON_RUN[@]}" -m generate_baselines_mila audit-full79-lstm \
  --run-root "$RUN_ROOT" \
  --stage "$AUDIT_STAGE" \
  --indices "$CELL_INDICES"
