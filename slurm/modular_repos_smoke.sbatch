#!/bin/bash
#SBATCH --job-name=modular-smoke
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=modular-smoke-%j.out

set -euo pipefail

THIS_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GENERATE_REPO="$(cd "${THIS_SCRIPT_DIR}/.." && pwd)"
PROJET_ROOT="$(cd "${GENERATE_REPO}/.." && pwd)"

BAYES_REPO="${BAYES_REPO:-${PROJET_ROOT}/bayes_efficiency_mila}"
COMPLEXITY_REPO="${COMPLEXITY_REPO:-${PROJET_ROOT}/child_complexity_predictors}"
PYTHON_CMD="${PYTHON_CMD:-${PYTHON:-python3}}"
read -r -a PYTHON_RUN <<< "${PYTHON_CMD}"
SMOKE_BASE="${SMOKE_BASE:-${SCRATCH:-${GENERATE_REPO}/results}/modular_repo_smoke}"
SMOKE_ROOT="${SMOKE_ROOT:-${SMOKE_BASE}/${SLURM_JOB_ID:-manual}}"

mkdir -p "${SMOKE_ROOT}"

echo "Smoke root: ${SMOKE_ROOT}"
echo "Generate repo: ${GENERATE_REPO}"
echo "Bayes repo: ${BAYES_REPO}"
echo "Complexity repo: ${COMPLEXITY_REPO}"
echo "Python command: ${PYTHON_CMD}"
echo "Smoke output base: ${SMOKE_BASE}"
echo "After rsyncing any needed outputs, remove this scratch directory:"
echo "  rm -rf -- '${SMOKE_ROOT}'"

run_generate_smoke() {
  local repo="${GENERATE_REPO}"
  local root="${SMOKE_ROOT}/generate_baselines_mila"
  mkdir -p "${root}/data" "${root}/results"
  cd "${repo}"
  export PYTHONPATH="${repo}/src:${PYTHONPATH:-}"
  export PYTHONPYCACHEPREFIX="${SMOKE_ROOT}/pycache_generate"

  "${PYTHON_RUN[@]}" -m unittest discover -s tests
  bash -n slurm/*.sbatch

  cat > "${root}/data/train.csv" <<'CSV'
utterance_id,age_bin,context_text,utterance_clean,target_utterance_clean,dataset,child_id,session_id,age_months
t1,006-023,do you want,more milk,,Smoke,c1,s1,18
t2,024-029,where did it,go there,,Smoke,c1,s2,25
CSV
  cat > "${root}/data/target.csv" <<'CSV'
utterance_id,age_bin,context_text,utterance_clean,target_utterance_clean,dataset,child_id,session_id,age_months
u1,006-023,do you want,,want milk,Smoke,c1,s1,18
u2,024-029,where did it,,go there now,Smoke,c1,s2,25
CSV
  cat > "${root}/ngram_manifest.json" <<JSON
{
  "run_id": "mila_smoke_ngram",
  "train_csv": "${root}/data/train.csv",
  "target_csv": "${root}/data/target.csv",
  "output_csv": "${root}/results/ngram_generated.csv.gz",
  "text_column": "utterance_clean",
  "target_text_column": "target_utterance_clean",
  "id_columns": ["utterance_id"],
  "carry_columns": ["dataset", "child_id", "session_id", "age_months", "age_bin", "context_text"],
  "age_bin_column": "age_bin",
  "age_bins": ["006-023", "024-029"],
  "context_column": "context_text",
  "context_tail_words": 2,
  "generators": ["random", "unigram", "bigram", "trigram"],
  "same_length": true,
  "samples_per_target": 1,
  "seed": 11
}
JSON
  "${PYTHON_RUN[@]}" -m generate_baselines_mila validate-manifest --manifest "${root}/ngram_manifest.json" --check-inputs
  "${PYTHON_RUN[@]}" -m generate_baselines_mila generate-ngram --manifest "${root}/ngram_manifest.json"

  if "${PYTHON_RUN[@]}" -c "import torch" >/dev/null 2>&1; then
    cat > "${root}/lstm_manifest.json" <<JSON
{
  "run_id": "mila_smoke_lstm",
  "train_csv": "${root}/data/train.csv",
  "target_csv": "${root}/data/target.csv",
  "output_csv": "${root}/results/lstm_generated.csv.gz",
  "text_column": "utterance_clean",
  "target_text_column": "target_utterance_clean",
  "id_columns": ["utterance_id"],
  "carry_columns": ["dataset", "child_id", "session_id", "age_months", "age_bin", "context_text"],
  "age_bin_column": "age_bin",
  "age_bins": ["006-023", "024-029"],
  "context_column": "context_text",
  "context_tail_words": 2,
  "same_length": true,
  "samples_per_target": 1,
  "seed": 11,
  "source_model": "lstm_smoke",
  "model_dir": "${root}/results/lstm_models",
  "device": "cpu",
  "epochs": 1,
  "batch_size": 2,
  "embedding_dim": 8,
  "hidden_dim": 16,
  "max_vocab_size": 100,
  "temperature": 1.0,
  "top_k": 0
}
JSON
    "${PYTHON_RUN[@]}" -m generate_baselines_mila generate-lstm --manifest "${root}/lstm_manifest.json"
  else
    echo "torch not importable; skipped LSTM smoke generation" > "${root}/results/lstm_smoke_skipped.txt"
  fi
}

run_bayes_smoke() {
  local repo="${BAYES_REPO}"
  local root="${SMOKE_ROOT}/bayes_efficiency_mila"
  mkdir -p "${root}/data" "${root}/results"
  cd "${repo}"
  export PYTHONPATH="${repo}/src:${PYTHONPATH:-}"
  export PYTHONPYCACHEPREFIX="${SMOKE_ROOT}/pycache_bayes"

  "${PYTHON_RUN[@]}" -m unittest discover -s tests
  bash -n slurm/*.sbatch

  cat > "${root}/data/train.csv" <<'CSV'
context_id,utterance_id,source_model,context_text,utterance_clean,target_utterance_clean,dataset,child_id,session_id,age_months,age_bin
c1,t1,real,do you want,more milk,,Smoke,c1,s1,18,006-023
c2,t2,real,where did it,go there,,Smoke,c1,s2,25,024-029
CSV
  cat > "${root}/data/candidates.csv" <<'CSV'
context_id,utterance_id,source_model,context_text,target_utterance_clean,dataset,child_id,session_id,age_months,age_bin,word_count
c1,u1,real,do you want,more milk,Smoke,c1,s1,18,006-023,2
c1,u2,trigram,do you want,want milk,Smoke,c1,s1,18,006-023,2
CSV
  cat > "${root}/ngram_bayes_manifest.json" <<JSON
{
  "run_id": "mila_smoke_ngram_bayes",
  "train_csv": "${root}/data/train.csv",
  "candidate_csv": "${root}/data/candidates.csv",
  "output_csv": "${root}/results/ngram_bayes_scores.csv.gz",
  "utterance_column": "utterance_clean",
  "candidate_utterance_column": "target_utterance_clean",
  "context_column": "context_text",
  "join_keys": ["context_id", "utterance_id", "source_model"],
  "carry_columns": ["dataset", "child_id", "session_id", "age_months", "age_bin", "context_text", "target_utterance_clean", "word_count"],
  "order": 2,
  "alpha": 0.1,
  "condition_tail_words": 8
}
JSON
  "${PYTHON_RUN[@]}" -m bayes_efficiency_mila validate-ngram-manifest --manifest "${root}/ngram_bayes_manifest.json" --check-inputs
  "${PYTHON_RUN[@]}" -m bayes_efficiency_mila score-ngram-bayes --manifest "${root}/ngram_bayes_manifest.json"
}

run_complexity_smoke() {
  local repo="${COMPLEXITY_REPO}"
  local root="${SMOKE_ROOT}/child_complexity_predictors"
  mkdir -p "${root}/data" "${root}/results"
  cd "${repo}"
  export PYTHONPATH="${repo}/src:${PYTHONPATH:-}"
  export PYTHONPYCACHEPREFIX="${SMOKE_ROOT}/pycache_complexity"

  "${PYTHON_RUN[@]}" -m unittest discover -s tests
  bash -n slurm/*.sbatch

  cat > "${root}/data/child.csv" <<'CSV'
utterance_id,dataset,child_id,session_id,age_months,age_bin,utterance_clean
u1,Smoke,c1,s1,18,006-023,more milk
u2,Smoke,c1,s2,25,024-029,more cookie
u3,Smoke,c2,s1,18,006-023,go home
CSV
  cat > "${root}/complexity_manifest.json" <<JSON
{
  "run_id": "mila_smoke_complexity",
  "input_csv": "${root}/data/child.csv",
  "output_csv": "${root}/results/utterance_complexity.csv.gz",
  "trajectory_output_csv": "trajectory.csv.gz",
  "aggregate_output_csv": "age_bin_summary.csv.gz",
  "text_column": "utterance_clean",
  "id_columns": ["utterance_id"],
  "carry_columns": ["dataset", "child_id", "session_id", "age_months", "age_bin"],
  "child_column": "child_id",
  "age_column": "age_months",
  "age_bin_column": "age_bin",
  "session_column": "session_id"
}
JSON
  "${PYTHON_RUN[@]}" -m child_complexity_predictors validate-manifest --manifest "${root}/complexity_manifest.json" --check-inputs
  "${PYTHON_RUN[@]}" -m child_complexity_predictors extract --manifest "${root}/complexity_manifest.json"
  "${PYTHON_RUN[@]}" -m child_complexity_predictors trajectory --manifest "${root}/complexity_manifest.json"
}

run_generate_smoke
run_bayes_smoke
run_complexity_smoke

cat > "${SMOKE_ROOT}/cleanup_after_rsync.sh" <<SH
#!/bin/bash
set -euo pipefail
rm -rf -- '${SMOKE_ROOT}'
SH
chmod +x "${SMOKE_ROOT}/cleanup_after_rsync.sh"

find "${SMOKE_ROOT}" -maxdepth 4 -type f | sort > "${SMOKE_ROOT}/produced_files.txt"
echo "All modular repo smoke checks completed." > "${SMOKE_ROOT}/SUCCESS.txt"
echo "Outputs written under ${SMOKE_ROOT}"
echo "After rsyncing any needed outputs, clean scratch with:"
echo "  rm -rf -- '${SMOKE_ROOT}'"
