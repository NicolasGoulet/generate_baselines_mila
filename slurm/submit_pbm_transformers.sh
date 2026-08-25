#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_ROOT"
mkdir -p slurm/logs reports/submissions

HANDOFF_ROOT="${1:?Usage: bash slurm/submit_pbm_transformers.sh /path/to/full_20260825}"
test -f "$HANDOFF_ROOT/manifest.json"
test -f "$HANDOFF_ROOT/BUILD_COMPLETE_AND_AUDITED"
: "${SCRATCH:?SCRATCH is required}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$SCRATCH/generate_baselines_mila/pbm_transformers_from_scratch/$RUN_ID}"
PYTHON_CMD="${PYTHON_CMD:-${PYTHON:-python3}}"
MAX_CONCURRENT="${MAX_CONCURRENT:-2}"
GPU_GRES="${GPU_GRES:-gpu:1}"
GPU_CONSTRAINT="${GPU_CONSTRAINT:-}"
COMMIT_SHA="${COMMIT_SHA:-$(git rev-parse HEAD)}"
SUBMISSION_METADATA_JSON="$PROJECT_ROOT/reports/submissions/pbm_transformers_${RUN_ID}.json"

if [[ -e "$RUN_ROOT" ]]; then
  echo "Fresh RUN_ROOT required: $RUN_ROOT" >&2
  exit 2
fi
if ! [[ "$MAX_CONCURRENT" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_CONCURRENT must be a positive integer" >&2
  exit 2
fi
if ! [[ "$GPU_GRES" =~ ^gpu(:[A-Za-z0-9_-]+)?:1$ ]]; then
  echo "GPU_GRES must request exactly one GPU, e.g. gpu:1 or gpu:l40s:1" >&2
  exit 2
fi

export PROJECT_ROOT HANDOFF_ROOT RUN_ID RUN_ROOT PYTHON_CMD COMMIT_SHA SUBMISSION_METADATA_JSON
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

GPU_ARGS=(--gres="$GPU_GRES")
[[ -n "$GPU_CONSTRAINT" ]] && GPU_ARGS+=(--constraint="$GPU_CONSTRAINT")

PREP_RAW="$(sbatch --parsable --ntasks=1 --export=ALL slurm/prepare_pbm_transformers.sbatch)"
PREP_JOB="${PREP_RAW%%;*}"
SMOKE_RAW="$(sbatch --parsable --ntasks=1 --dependency="afterok:$PREP_JOB" \
  --array=0-1%2 "${GPU_ARGS[@]}" --export=ALL slurm/run_pbm_transformer_cell.sbatch smoke)"
SMOKE_JOB="${SMOKE_RAW%%;*}"
SMOKE_AUDIT_RAW="$(sbatch --parsable --ntasks=1 --dependency="afterok:$SMOKE_JOB" \
  --export="ALL,AUDIT_STAGE=smoke" slurm/audit_pbm_transformers.sbatch)"
SMOKE_AUDIT_JOB="${SMOKE_AUDIT_RAW%%;*}"
WAVE1_RAW="$(sbatch --parsable --ntasks=1 --dependency="afterok:$SMOKE_AUDIT_JOB" \
  --array="0-7%$MAX_CONCURRENT" "${GPU_ARGS[@]}" --export=ALL \
  slurm/run_pbm_transformer_cell.sbatch production)"
WAVE1_JOB="${WAVE1_RAW%%;*}"
WAVE1_AUDIT_RAW="$(sbatch --parsable --ntasks=1 --dependency="afterok:$WAVE1_JOB" \
  --export="ALL,AUDIT_STAGE=wave1" slurm/audit_pbm_transformers.sbatch)"
WAVE1_AUDIT_JOB="${WAVE1_AUDIT_RAW%%;*}"
WAVE2_RAW="$(sbatch --parsable --ntasks=1 --dependency="afterok:$WAVE1_AUDIT_JOB" \
  --array="8-15%$MAX_CONCURRENT" "${GPU_ARGS[@]}" --export=ALL \
  slurm/run_pbm_transformer_cell.sbatch production)"
WAVE2_JOB="${WAVE2_RAW%%;*}"
WAVE2_AUDIT_RAW="$(sbatch --parsable --ntasks=1 --dependency="afterok:$WAVE2_JOB" \
  --export="ALL,AUDIT_STAGE=wave2" slurm/audit_pbm_transformers.sbatch)"
WAVE2_AUDIT_JOB="${WAVE2_AUDIT_RAW%%;*}"
FINAL_RAW="$(sbatch --parsable --ntasks=1 --dependency="afterok:$WAVE2_AUDIT_JOB" \
  --export="ALL,AUDIT_STAGE=final" slurm/audit_pbm_transformers.sbatch)"
FINAL_JOB="${FINAL_RAW%%;*}"
FINAL_REPORT_RAW="$(sbatch --parsable --ntasks=1 --dependency="afterok:$FINAL_JOB" \
  --export=ALL slurm/report_pbm_transformers.sbatch)"
FINAL_REPORT_JOB="${FINAL_REPORT_RAW%%;*}"

export PREP_JOB SMOKE_JOB SMOKE_AUDIT_JOB WAVE1_JOB WAVE1_AUDIT_JOB
export WAVE2_JOB WAVE2_AUDIT_JOB FINAL_JOB FINAL_REPORT_JOB
python3 - "$SUBMISSION_METADATA_JSON" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "run_id": os.environ["RUN_ID"],
    "run_root": os.environ["RUN_ROOT"],
    "handoff_root": os.environ["HANDOFF_ROOT"],
    "commit_sha": os.environ["COMMIT_SHA"],
    "python_cmd": os.environ["PYTHON_CMD"],
    "job_ids": {
        "preparation_and_tokenizer": os.environ["PREP_JOB"],
        "gpu_smoke_array": os.environ["SMOKE_JOB"],
        "smoke_audit": os.environ["SMOKE_AUDIT_JOB"],
        "wave1_array": os.environ["WAVE1_JOB"],
        "wave1_audit": os.environ["WAVE1_AUDIT_JOB"],
        "wave2_array": os.environ["WAVE2_JOB"],
        "wave2_audit": os.environ["WAVE2_AUDIT_JOB"],
        "final_audit": os.environ["FINAL_JOB"],
    },
    "final_report_job_id": os.environ["FINAL_REPORT_JOB"],
    "configuration": {
        "architectures": ["babyllama_sized_llama_58m", "t5_58m"],
        "age_models_per_architecture": 8,
        "initialization": "from_scratch",
        "teacher_distillation": False,
        "shared_vocab_size": 16000,
        "samples_per_target": 1,
        "generation_variant": "unconstrained_length",
        "max_new_tokens": int(os.environ["MAX_NEW_TOKENS"]),
    },
}
path.parent.mkdir(parents=True, exist_ok=True)
temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, path)
PY

echo "RUN_ID=$RUN_ID"
echo "RUN_ROOT=$RUN_ROOT"
echo "PREP_JOB=$PREP_JOB"
echo "SMOKE_JOB=$SMOKE_JOB"
echo "SMOKE_AUDIT_JOB=$SMOKE_AUDIT_JOB"
echo "WAVE1_JOB=$WAVE1_JOB"
echo "WAVE1_AUDIT_JOB=$WAVE1_AUDIT_JOB"
echo "WAVE2_JOB=$WAVE2_JOB"
echo "WAVE2_AUDIT_JOB=$WAVE2_AUDIT_JOB"
echo "FINAL_AUDIT_JOB=$FINAL_JOB"
echo "FINAL_REPORT_JOB=$FINAL_REPORT_JOB"
echo "SUBMISSION_METADATA_JSON=$SUBMISSION_METADATA_JSON"
echo "SCORER_READY_HANDOFF=$RUN_ROOT/handoff/pbm_transformer_responses_scorer_ready.csv.gz"
echo "Production arrays are blocked on the exact-wrapper two-architecture smoke and staged audits."
