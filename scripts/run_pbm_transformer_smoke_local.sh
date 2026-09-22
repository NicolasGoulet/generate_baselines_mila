#!/usr/bin/env bash
set -euo pipefail

HANDOFF_ROOT="${1:?Usage: bash scripts/run_pbm_transformer_smoke_local.sh HANDOFF_ROOT RUN_ROOT}"
RUN_ROOT="${2:?Usage: bash scripts/run_pbm_transformer_smoke_local.sh HANDOFF_ROOT RUN_ROOT}"
case "$HANDOFF_ROOT" in /*) ;; *) HANDOFF_ROOT="$PWD/$HANDOFF_ROOT" ;; esac
case "$RUN_ROOT" in /*) ;; *) RUN_ROOT="$PWD/$RUN_ROOT" ;; esac
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_ROOT"
test -f "$HANDOFF_ROOT/manifest.json"
test -f "$HANDOFF_ROOT/BUILD_COMPLETE_AND_AUDITED"

if [[ -e "$RUN_ROOT" ]]; then
  echo "Fresh RUN_ROOT required: $RUN_ROOT" >&2
  exit 2
fi

GPU_GRES="${GPU_GRES:-gpu:1}"
if ! [[ "$GPU_GRES" =~ ^gpu(:[A-Za-z0-9_-]+)?:1$ ]]; then
  echo "GPU_GRES must request exactly one GPU, e.g. gpu:1 or gpu:l40s:1" >&2
  exit 2
fi

PYTHON_CMD="${PYTHON_CMD:-${PYTHON:-python3}}"
if [[ -x "$PYTHON_CMD" ]]; then
  PYTHON_RUN=("$PYTHON_CMD")
else
  read -r -a PYTHON_RUN <<< "$PYTHON_CMD"
fi
if [[ "${#PYTHON_RUN[@]}" -eq 0 ]]; then
  echo "PYTHON_CMD must name a Python interpreter" >&2
  exit 2
fi

export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
# Preparation requires an empty run root. Imports must not create pycache there.
export PYTHONDONTWRITEBYTECODE=1
COMMIT_SHA="${COMMIT_SHA:-$(git rev-parse HEAD)}"

export SEED="${SEED:-20260825}"
export MAX_EPOCHS="${MAX_EPOCHS:-10}"
export PATIENCE="${PATIENCE:-2}"
export PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-16}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"
export LEARNING_RATE="${LEARNING_RATE:-0.0005}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
export TEMPERATURE="${TEMPERATURE:-1.0}"
export TOP_P="${TOP_P:-1.0}"
export MAX_CENSORED_FRACTION="${MAX_CENSORED_FRACTION:-0.001}"
export SMOKE_TRAIN_EXAMPLES="${SMOKE_TRAIN_EXAMPLES:-1024}"
export SMOKE_TARGET_ROWS="${SMOKE_TARGET_ROWS:-25}"

# Fail before preparing data or a tokenizer if the GPU runtime is unavailable.
"${PYTHON_RUN[@]}" -m generate_baselines_mila validate-pbm-transformer-runtime

"${PYTHON_RUN[@]}" -m generate_baselines_mila prepare-pbm-transformers \
  --handoff-root "$HANDOFF_ROOT" \
  --run-root "$RUN_ROOT" \
  --seed "$SEED" \
  --max-epochs "$MAX_EPOCHS" \
  --patience "$PATIENCE" \
  --per-device-batch-size "$PER_DEVICE_BATCH_SIZE" \
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS" \
  --learning-rate "$LEARNING_RATE" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --temperature "$TEMPERATURE" \
  --top-p "$TOP_P" \
  --max-censored-fraction "$MAX_CENSORED_FRACTION" \
  --smoke-train-examples "$SMOKE_TRAIN_EXAMPLES" \
  --smoke-target-rows "$SMOKE_TARGET_ROWS"

"${PYTHON_RUN[@]}" -m generate_baselines_mila train-pbm-transformer-tokenizer \
  --source-file "$HANDOFF_ROOT/examples/train_all_ages.jsonl.gz" \
  --output-dir "$RUN_ROOT/tokenizer" \
  --vocab-size 16000

test -f "$RUN_ROOT/PREPARED_AND_AUDITED"
test -f "$RUN_ROOT/tokenizer/TOKENIZER_READY"

{
  printf 'mode=local\n'
  printf 'handoff_root=%s\n' "$HANDOFF_ROOT"
  printf 'run_root=%s\n' "$RUN_ROOT"
  printf 'python_cmd=%s\n' "$PYTHON_CMD"
  printf 'commit_sha=%s\n' "$COMMIT_SHA"
  printf 'gpu_gres_contract=%s\n' "$GPU_GRES"
} > "$RUN_ROOT/LOCAL_SMOKE_EXECUTION.txt"

for index in 0 1; do
  manifest="$("${PYTHON_RUN[@]}" -m generate_baselines_mila pbm-transformer-manifest \
    --run-root "$RUN_ROOT" --index "$index" --smoke)"
  "${PYTHON_RUN[@]}" -m generate_baselines_mila run-pbm-transformer-cell \
    --manifest "$manifest"
  "${PYTHON_RUN[@]}" -m generate_baselines_mila audit-pbm-transformer-cell \
    --manifest "$manifest"
done

"${PYTHON_RUN[@]}" -m generate_baselines_mila audit-pbm-transformer-smoke \
  --run-root "$RUN_ROOT" --job-id "${SMOKE_JOB_ID:-local}"

test -f "$RUN_ROOT/SMOKE_PASSED"
echo "LOCAL_SMOKE_PASSED=$RUN_ROOT/SMOKE_PASSED"
