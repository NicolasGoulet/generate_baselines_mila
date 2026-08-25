#!/usr/bin/env bash
#SBATCH --job-name=full79-lstm-report
#SBATCH --output=slurm/logs/full79-lstm-report-%j.out
#SBATCH --error=slurm/logs/full79-lstm-report-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --partition=long-cpu

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$HOME/communicative_efficiency_repos/generate_baselines_mila}}"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
: "${RUN_ROOT:?RUN_ROOT is required}"
test -f "$RUN_ROOT/COMPLETE_AND_AUDITED"
PYTHON_CMD="${PYTHON_CMD:-${PYTHON:-python3}}"
read -r -a PYTHON_RUN <<< "$PYTHON_CMD"
export PYTHONPYCACHEPREFIX="$RUN_ROOT/pycache/final-report"

"${PYTHON_RUN[@]}" -m generate_baselines_mila finalize-full79-lstm-report \
  --run-root "$RUN_ROOT"

test -f "$RUN_ROOT/FINAL_REPORT_READY"
