#!/usr/bin/env bash
#SBATCH --job-name=full79-lstm-prep
#SBATCH --output=slurm/logs/full79-lstm-prep-%j.out
#SBATCH --error=slurm/logs/full79-lstm-prep-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --partition=long-cpu

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$HOME/communicative_efficiency_repos/generate_baselines_mila}}"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
: "${RUN_ROOT:?RUN_ROOT is required}"
export PYTHONPYCACHEPREFIX="${TMPDIR:-/tmp}/generate_baselines_mila_prep_pycache_${SLURM_JOB_ID:-manual}"
: "${BUNDLE_ROOT:?BUNDLE_ROOT is required}"
PYTHON_CMD="${PYTHON_CMD:-${PYTHON:-python3}}"
read -r -a PYTHON_RUN <<< "$PYTHON_CMD"

"${PYTHON_RUN[@]}" -m generate_baselines_mila prepare-full79-lstm \
  --bundle-root "$BUNDLE_ROOT" \
  --run-root "$RUN_ROOT" \
  --contexts 3 \
  --max-context-tokens "${MAX_CONTEXT_TOKENS:-60}" \
  --epochs "${EPOCHS:-20}" \
  --batch-size "${BATCH_SIZE:-256}" \
  --embedding-dim "${EMBEDDING_DIM:-256}" \
  --hidden-dim "${HIDDEN_DIM:-512}" \
  --num-layers "${NUM_LAYERS:-2}" \
  --dropout "${DROPOUT:-0.2}" \
  --max-vocab-size "${MAX_VOCAB_SIZE:-30000}" \
  --smoke-train-examples "${SMOKE_TRAIN_EXAMPLES:-1024}" \
  --smoke-target-rows "${SMOKE_TARGET_ROWS:-25}" \
  --seed "${SEED:-123}"

test -f "$RUN_ROOT/PREPARED_AND_AUDITED"
echo "Prepared and audited full-79 LSTM inputs under $RUN_ROOT"
