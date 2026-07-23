#!/usr/bin/env bash
#SBATCH --job-name=full79-lstm
#SBATCH --output=slurm/logs/full79-lstm-%A_%a.out
#SBATCH --error=slurm/logs/full79-lstm-%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=long

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$HOME/communicative_efficiency_repos/generate_baselines_mila}}"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
: "${RUN_ROOT:?RUN_ROOT is required}"
test -f "$RUN_ROOT/PREPARED_AND_AUDITED"
PYTHON_CMD="${PYTHON_CMD:-${PYTHON:-python3}}"
read -r -a PYTHON_RUN <<< "$PYTHON_CMD"
MODE="${1:?Usage: sbatch $0 smoke|production}"

"${PYTHON_RUN[@]}" -c \
  'import torch; assert torch.cuda.is_available(), "CUDA is not available in the selected Python environment"'

case "$MODE" in
  smoke)
    MANIFEST="$RUN_ROOT/manifests/smoke_manifest.json"
    if [[ -e "$RUN_ROOT/SMOKE_PASSED" ]]; then
      echo "Refusing to reuse a prior smoke; submit with a fresh RUN_ROOT." >&2
      exit 2
    fi
    ;;
  production)
    : "${SLURM_ARRAY_TASK_ID:?production mode requires SLURM_ARRAY_TASK_ID}"
    MANIFEST="$("${PYTHON_RUN[@]}" -m generate_baselines_mila full79-lstm-manifest \
      --run-root "$RUN_ROOT" --index "$SLURM_ARRAY_TASK_ID")"
    ;;
  *)
    echo "Unknown cell mode: $MODE" >&2
    exit 2
    ;;
esac

export PYTHONPYCACHEPREFIX="$RUN_ROOT/pycache/$MODE/${SLURM_ARRAY_TASK_ID:-smoke}"
"${PYTHON_RUN[@]}" -m generate_baselines_mila validate-manifest \
  --manifest "$MANIFEST" --check-inputs

if [[ "$MODE" == "production" ]] && \
  "${PYTHON_RUN[@]}" -m generate_baselines_mila audit-lstm-output --manifest "$MANIFEST"; then
  echo "Validated existing production output; skipping retraining: $MANIFEST"
  exit 0
fi

"${PYTHON_RUN[@]}" -m generate_baselines_mila generate-lstm --manifest "$MANIFEST"
"${PYTHON_RUN[@]}" -m generate_baselines_mila audit-lstm-output --manifest "$MANIFEST"

if [[ "$MODE" == "smoke" ]]; then
  "${PYTHON_RUN[@]}" -m generate_baselines_mila audit-full79-lstm-smoke \
    --run-root "$RUN_ROOT" --job-id "${SLURM_JOB_ID:-}"
fi
