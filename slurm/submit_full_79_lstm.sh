#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_ROOT"
mkdir -p slurm/logs reports/submissions

BUNDLE_ROOT="${1:?Usage: bash slurm/submit_full_79_lstm.sh /path/to/default_naturalistic_merged_006_023}"
test -f "$BUNDLE_ROOT/manifest.csv"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$SCRATCH/generate_baselines_mila/full79_lstm_additive_k3_same_length/$RUN_ID}"
PYTHON_CMD="${PYTHON_CMD:-${PYTHON:-python3}}"
MAX_CONCURRENT="${MAX_CONCURRENT:-3}"
GPU_GRES="${GPU_GRES:-gpu:1}"
GPU_CONSTRAINT="${GPU_CONSTRAINT:-}"
COMMIT_SHA="${COMMIT_SHA:-$(git rev-parse HEAD)}"
SUBMISSION_METADATA_JSON="$PROJECT_ROOT/reports/submissions/full79_lstm_${RUN_ID}.json"

if [[ -e "$RUN_ROOT" ]]; then
  echo "Fresh RUN_ROOT required: $RUN_ROOT" >&2
  exit 2
fi
if ! [[ "$MAX_CONCURRENT" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_CONCURRENT must be a positive integer" >&2
  exit 2
fi

export PROJECT_ROOT BUNDLE_ROOT RUN_ID RUN_ROOT PYTHON_CMD COMMIT_SHA SUBMISSION_METADATA_JSON
export EPOCHS="${EPOCHS:-20}"
export BATCH_SIZE="${BATCH_SIZE:-256}"
export EMBEDDING_DIM="${EMBEDDING_DIM:-256}"
export HIDDEN_DIM="${HIDDEN_DIM:-512}"
export NUM_LAYERS="${NUM_LAYERS:-2}"
export DROPOUT="${DROPOUT:-0.2}"
export MAX_VOCAB_SIZE="${MAX_VOCAB_SIZE:-30000}"
export MAX_CONTEXT_TOKENS="${MAX_CONTEXT_TOKENS:-60}"
export SMOKE_TRAIN_EXAMPLES="${SMOKE_TRAIN_EXAMPLES:-1024}"
export SMOKE_TARGET_ROWS="${SMOKE_TARGET_ROWS:-25}"
export SEED="${SEED:-123}"

GPU_ARGS=(--gres="$GPU_GRES")
[[ -n "$GPU_CONSTRAINT" ]] && GPU_ARGS+=(--constraint="$GPU_CONSTRAINT")

PREP_RAW="$(sbatch --parsable --ntasks=1 --export=ALL slurm/prepare_full_79_lstm.sbatch)"
PREP_JOB="${PREP_RAW%%;*}"
SMOKE_RAW="$(sbatch --parsable \
  --ntasks=1 \
  --dependency="afterok:$PREP_JOB" \
  "${GPU_ARGS[@]}" \
  --export=ALL \
  slurm/run_full_79_lstm_cell.sbatch smoke)"
SMOKE_JOB="${SMOKE_RAW%%;*}"

WAVE1_INDICES="0-3"
WAVE1_RAW="$(sbatch --parsable \
  --ntasks=1 \
  --dependency="afterok:$SMOKE_JOB" \
  --array="$WAVE1_INDICES%$MAX_CONCURRENT" \
  "${GPU_ARGS[@]}" \
  --export=ALL \
  slurm/run_full_79_lstm_cell.sbatch production)"
WAVE1_JOB="${WAVE1_RAW%%;*}"
WAVE1_AUDIT_RAW="$(sbatch --parsable \
  --ntasks=1 \
  --dependency="afterok:$WAVE1_JOB" \
  --export="ALL,AUDIT_STAGE=wave1,CELL_INDICES=$WAVE1_INDICES" \
  slurm/audit_full_79_lstm.sbatch)"
WAVE1_AUDIT_JOB="${WAVE1_AUDIT_RAW%%;*}"

WAVE2_INDICES="4-7"
WAVE2_RAW="$(sbatch --parsable \
  --ntasks=1 \
  --dependency="afterok:$WAVE1_AUDIT_JOB" \
  --array="$WAVE2_INDICES%$MAX_CONCURRENT" \
  "${GPU_ARGS[@]}" \
  --export=ALL \
  slurm/run_full_79_lstm_cell.sbatch production)"
WAVE2_JOB="${WAVE2_RAW%%;*}"
WAVE2_AUDIT_RAW="$(sbatch --parsable \
  --ntasks=1 \
  --dependency="afterok:$WAVE2_JOB" \
  --export="ALL,AUDIT_STAGE=wave2,CELL_INDICES=$WAVE2_INDICES" \
  slurm/audit_full_79_lstm.sbatch)"
WAVE2_AUDIT_JOB="${WAVE2_AUDIT_RAW%%;*}"

FINAL_RAW="$(sbatch --parsable \
  --ntasks=1 \
  --dependency="afterok:$WAVE2_AUDIT_JOB" \
  --export="ALL,AUDIT_STAGE=final,CELL_INDICES=0-7" \
  slurm/audit_full_79_lstm.sbatch)"
FINAL_JOB="${FINAL_RAW%%;*}"
FINAL_REPORT_RAW="$(sbatch --parsable \
  --ntasks=1 \
  --dependency="afterok:$FINAL_JOB" \
  --export=ALL \
  slurm/report_full_79_lstm.sbatch)"
FINAL_REPORT_JOB="${FINAL_REPORT_RAW%%;*}"

export PREP_JOB SMOKE_JOB WAVE1_JOB WAVE1_AUDIT_JOB WAVE2_JOB WAVE2_AUDIT_JOB FINAL_JOB FINAL_REPORT_JOB
python3 - "$SUBMISSION_METADATA_JSON" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
path.parent.mkdir(parents=True, exist_ok=True)
payload = {
    "run_id": os.environ["RUN_ID"],
    "run_root": os.environ["RUN_ROOT"],
    "bundle_root": os.environ["BUNDLE_ROOT"],
    "commit_sha": os.environ["COMMIT_SHA"],
    "python_cmd": os.environ["PYTHON_CMD"],
    "job_ids": {
        "preparation": os.environ["PREP_JOB"],
        "smoke": os.environ["SMOKE_JOB"],
        "wave1": os.environ["WAVE1_JOB"],
        "wave1_audit": os.environ["WAVE1_AUDIT_JOB"],
        "wave2": os.environ["WAVE2_JOB"],
        "wave2_audit": os.environ["WAVE2_AUDIT_JOB"],
        "final_audit": os.environ["FINAL_JOB"],
    },
    "final_report_job_id": os.environ["FINAL_REPORT_JOB"],
    "configuration": {
        "embedding_dim": int(os.environ["EMBEDDING_DIM"]),
        "hidden_dim": int(os.environ["HIDDEN_DIM"]),
        "num_layers": int(os.environ["NUM_LAYERS"]),
        "epochs": int(os.environ["EPOCHS"]),
        "batch_size": int(os.environ["BATCH_SIZE"]),
        "dropout": float(os.environ["DROPOUT"]),
        "seed": int(os.environ["SEED"]),
        "generation_context": "k3",
        "same_length": True,
    },
}
temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, path)
PY

SUBMISSION_REPORT="$PROJECT_ROOT/reports/submissions/full79_lstm_${RUN_ID}.md"
cat > "$SUBMISSION_REPORT" <<EOF
# Full-79 LSTM Submission

- run id: \`$RUN_ID\`
- run root: \`$RUN_ROOT\`
- bundle root: \`$BUNDLE_ROOT\`
- preparation job: \`$PREP_JOB\`
- exact-wrapper GPU smoke job: \`$SMOKE_JOB\`
- wave 1 array job: \`$WAVE1_JOB\` (\`$WAVE1_INDICES\`)
- wave 1 audit job: \`$WAVE1_AUDIT_JOB\`
- wave 2 array job: \`$WAVE2_JOB\` (\`$WAVE2_INDICES\`)
- wave 2 audit job: \`$WAVE2_AUDIT_JOB\`
- final audit job: \`$FINAL_JOB\`
- final state/hash report job: \`$FINAL_REPORT_JOB\`
- maximum concurrent GPU cells: \`$MAX_CONCURRENT\`
- Python command: \`$PYTHON_CMD\`
- architecture: \`seq2seq_lstm\`
- generation context: \`k3\`
- additive age-bin models: \`8\`
- variant: \`same_length\`
- commit SHA: \`$COMMIT_SHA\`
- submission metadata: \`$SUBMISSION_METADATA_JSON\`
- scorer-ready handoff (after final audit): \`$RUN_ROOT/handoff/full79_lstm_scorer_ready.csv.gz\`
EOF

echo "RUN_ID=$RUN_ID"
echo "RUN_ROOT=$RUN_ROOT"
echo "PREP_JOB=$PREP_JOB"
echo "SMOKE_JOB=$SMOKE_JOB"
echo "WAVE1_JOB=$WAVE1_JOB"
echo "WAVE1_AUDIT_JOB=$WAVE1_AUDIT_JOB"
echo "WAVE2_JOB=$WAVE2_JOB"
echo "WAVE2_AUDIT_JOB=$WAVE2_AUDIT_JOB"
echo "FINAL_AUDIT_JOB=$FINAL_JOB"
echo "FINAL_REPORT_JOB=$FINAL_REPORT_JOB"
echo "SUBMISSION_REPORT=$SUBMISSION_REPORT"
echo "SUBMISSION_METADATA_JSON=$SUBMISSION_METADATA_JSON"
echo "SMOKE_REPORT=$RUN_ROOT/reports/smoke/smoke_report.md"
echo "SCORER_READY_HANDOFF=$RUN_ROOT/handoff/full79_lstm_scorer_ready.csv.gz"
echo "Production is blocked on the exact-wrapper smoke and each later wave is blocked on the prior audit."
echo "After the audit job finishes, run this on the local laptop to retrieve compact reports:"
echo "  rsync -avhP 'mila:$RUN_ROOT/reports/' '/home/apaixonada/EvaPortelance/Projet_1/communicative_efficiency/results/mila_modular_runs_2026_07_08/products/full79_lstm_reports/$RUN_ID/'"
