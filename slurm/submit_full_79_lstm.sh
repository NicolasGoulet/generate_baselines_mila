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

if [[ -e "$RUN_ROOT" ]]; then
  echo "Fresh RUN_ROOT required: $RUN_ROOT" >&2
  exit 2
fi
if ! [[ "$MAX_CONCURRENT" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_CONCURRENT must be a positive integer" >&2
  exit 2
fi

export PROJECT_ROOT BUNDLE_ROOT RUN_ROOT PYTHON_CMD
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
- maximum concurrent GPU cells: \`$MAX_CONCURRENT\`
- Python command: \`$PYTHON_CMD\`
- architecture: \`seq2seq_lstm\`
- generation context: \`k3\`
- additive age-bin models: \`8\`
- variant: \`same_length\`
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
echo "SUBMISSION_REPORT=$SUBMISSION_REPORT"
echo "SMOKE_REPORT=$RUN_ROOT/reports/smoke/smoke_report.md"
echo "Production is blocked on the exact-wrapper smoke and each later wave is blocked on the prior audit."
echo "After the audit job finishes, run this on the local laptop to retrieve compact reports:"
echo "  rsync -avhP 'mila:$RUN_ROOT/reports/' '/home/apaixonada/EvaPortelance/Projet_1/communicative_efficiency/results/mila_modular_runs_2026_07_08/products/full79_lstm_reports/$RUN_ID/'"
