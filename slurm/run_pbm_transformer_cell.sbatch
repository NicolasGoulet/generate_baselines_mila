#!/usr/bin/env bash
#SBATCH --job-name=pbm-transformer
#SBATCH --output=slurm/logs/pbm-transformer-%A_%a.out
#SBATCH --error=slurm/logs/pbm-transformer-%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --partition=long

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$HOME/communicative_efficiency_repos/generate_baselines_mila}}"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
: "${RUN_ROOT:?RUN_ROOT is required}"
: "${SLURM_ARRAY_TASK_ID:?A Slurm array index is required}"
PYTHON_CMD="${PYTHON_CMD:-${PYTHON:-python3}}"
read -r -a PYTHON_RUN <<< "$PYTHON_CMD"
MODE="${1:?Usage: sbatch $0 smoke|production}"

test -f "$RUN_ROOT/PREPARED_AND_AUDITED"
test -f "$RUN_ROOT/tokenizer/TOKENIZER_READY"
"${PYTHON_RUN[@]}" -m generate_baselines_mila validate-pbm-transformer-runtime

case "$MODE" in
  smoke)
    [[ "$SLURM_ARRAY_TASK_ID" =~ ^[01]$ ]]
    MANIFEST="$("${PYTHON_RUN[@]}" -m generate_baselines_mila pbm-transformer-manifest \
      --run-root "$RUN_ROOT" --index "$SLURM_ARRAY_TASK_ID" --smoke)"
    ;;
  production)
    test -f "$RUN_ROOT/SMOKE_PASSED"
    if (( SLURM_ARRAY_TASK_ID >= 8 )); then
      test -f "$RUN_ROOT/WAVE1_READY"
    fi
    MANIFEST="$("${PYTHON_RUN[@]}" -m generate_baselines_mila pbm-transformer-manifest \
      --run-root "$RUN_ROOT" --index "$SLURM_ARRAY_TASK_ID")"
    if "${PYTHON_RUN[@]}" -m generate_baselines_mila audit-pbm-transformer-cell \
      --manifest "$MANIFEST" >/dev/null 2>&1; then
      echo "Validated existing cell; skipping: $MANIFEST"
      exit 0
    fi
    ;;
  *)
    echo "Unknown cell mode: $MODE" >&2
    exit 2
    ;;
esac

export PYTHONPYCACHEPREFIX="$RUN_ROOT/pycache/$MODE/$SLURM_ARRAY_TASK_ID"
"${PYTHON_RUN[@]}" -m generate_baselines_mila run-pbm-transformer-cell \
  --manifest "$MANIFEST"
"${PYTHON_RUN[@]}" -m generate_baselines_mila audit-pbm-transformer-cell \
  --manifest "$MANIFEST"
