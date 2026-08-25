#!/usr/bin/env bash
#SBATCH --job-name=pbm-transformer-prep
#SBATCH --output=slurm/logs/pbm-transformer-prep-%j.out
#SBATCH --error=slurm/logs/pbm-transformer-prep-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --partition=long-cpu

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$HOME/communicative_efficiency_repos/generate_baselines_mila}}"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
: "${RUN_ROOT:?RUN_ROOT is required}"
: "${HANDOFF_ROOT:?HANDOFF_ROOT is required}"
PYTHON_CMD="${PYTHON_CMD:-${PYTHON:-python3}}"
read -r -a PYTHON_RUN <<< "$PYTHON_CMD"
export PYTHONPYCACHEPREFIX="${TMPDIR:-/tmp}/pbm_transformer_prep_${SLURM_JOB_ID:-manual}"

"${PYTHON_RUN[@]}" -m generate_baselines_mila prepare-pbm-transformers \
  --handoff-root "$HANDOFF_ROOT" \
  --run-root "$RUN_ROOT" \
  --seed "${SEED:-20260825}" \
  --max-epochs "${MAX_EPOCHS:-10}" \
  --patience "${PATIENCE:-2}" \
  --per-device-batch-size "${PER_DEVICE_BATCH_SIZE:-16}" \
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS:-16}" \
  --learning-rate "${LEARNING_RATE:-0.0005}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-128}" \
  --temperature "${TEMPERATURE:-1.0}" \
  --top-p "${TOP_P:-1.0}" \
  --max-censored-fraction "${MAX_CENSORED_FRACTION:-0.001}" \
  --smoke-train-examples "${SMOKE_TRAIN_EXAMPLES:-1024}" \
  --smoke-target-rows "${SMOKE_TARGET_ROWS:-25}"

"${PYTHON_RUN[@]}" -m generate_baselines_mila train-pbm-transformer-tokenizer \
  --source-file "$HANDOFF_ROOT/examples/train_all_ages.jsonl.gz" \
  --output-dir "$RUN_ROOT/tokenizer" \
  --vocab-size 16000

test -f "$RUN_ROOT/PREPARED_AND_AUDITED"
test -f "$RUN_ROOT/tokenizer/TOKENIZER_READY"
