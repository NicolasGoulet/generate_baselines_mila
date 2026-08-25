#!/usr/bin/env bash
#SBATCH --job-name=pbm-transformer-audit
#SBATCH --output=slurm/logs/pbm-transformer-audit-%j.out
#SBATCH --error=slurm/logs/pbm-transformer-audit-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --partition=long-cpu

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$HOME/communicative_efficiency_repos/generate_baselines_mila}}"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
: "${RUN_ROOT:?RUN_ROOT is required}"
: "${AUDIT_STAGE:?AUDIT_STAGE is required}"
PYTHON_CMD="${PYTHON_CMD:-${PYTHON:-python3}}"
read -r -a PYTHON_RUN <<< "$PYTHON_CMD"
export PYTHONPYCACHEPREFIX="$RUN_ROOT/pycache/audit/$AUDIT_STAGE"

case "$AUDIT_STAGE" in
  smoke)
    "${PYTHON_RUN[@]}" -m generate_baselines_mila audit-pbm-transformer-smoke \
      --run-root "$RUN_ROOT" --job-id "${SLURM_JOB_ID:-}"
    ;;
  wave1)
    "${PYTHON_RUN[@]}" -m generate_baselines_mila audit-pbm-transformers \
      --run-root "$RUN_ROOT" --stage wave1 --indices 0-7
    ;;
  wave2)
    "${PYTHON_RUN[@]}" -m generate_baselines_mila audit-pbm-transformers \
      --run-root "$RUN_ROOT" --stage wave2 --indices 8-15
    ;;
  final)
    "${PYTHON_RUN[@]}" -m generate_baselines_mila audit-pbm-transformers \
      --run-root "$RUN_ROOT" --stage final --indices 0-15
    ;;
  *)
    echo "Unknown AUDIT_STAGE: $AUDIT_STAGE" >&2
    exit 2
    ;;
esac
