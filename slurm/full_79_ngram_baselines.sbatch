#!/bin/bash
#SBATCH --job-name=full79-ngram
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/src/generate_baselines_mila" ]]; then
  REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

if [[ $# -lt 1 ]]; then
  echo "Usage: sbatch $0 /path/to/default_naturalistic_merged_006_023 [output_root]" >&2
  exit 2
fi

BUNDLE_ROOT="$1"
JOB_ID="${SLURM_JOB_ID:-manual}"
SCRATCH_ROOT="${SCRATCH:-${HOME}/scratch}"
OUTPUT_ROOT="${2:-${SCRATCH_ROOT}/communicative_efficiency_runs/full79_ngram/${JOB_ID}}"
PYTHON_BIN="${PYTHON:-python3}"
RUN_ID="${RUN_ID:-full79_ngram_additive}"
SAMPLES_PER_TARGET="${SAMPLES_PER_TARGET:-1}"
SEED="${SEED:-13}"
CONTEXT_COLUMN="${CONTEXT_COLUMN:-context_k3}"
CONTEXT_TAIL_WORDS="${CONTEXT_TAIL_WORDS:-16}"

mkdir -p "${OUTPUT_ROOT}"

"${PYTHON_BIN}" -m generate_baselines_mila prepare-full79-ngram-manifest \
  --bundle-root "${BUNDLE_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --run-id "${RUN_ID}" \
  --samples-per-target "${SAMPLES_PER_TARGET}" \
  --seed "${SEED}" \
  --context-column "${CONTEXT_COLUMN}" \
  --context-tail-words "${CONTEXT_TAIL_WORDS}"

MANIFEST="${OUTPUT_ROOT}/manifests/full79_ngram_manifest.json"
"${PYTHON_BIN}" -m generate_baselines_mila validate-manifest --manifest "${MANIFEST}" --check-inputs
"${PYTHON_BIN}" -m generate_baselines_mila generate-ngram --manifest "${MANIFEST}"

cat > "${OUTPUT_ROOT}/cleanup_after_rsync.sh" <<EOF
#!/bin/bash
set -euo pipefail
rm -rf -- '${OUTPUT_ROOT}'
EOF
chmod +x "${OUTPUT_ROOT}/cleanup_after_rsync.sh"

echo "Outputs written under ${OUTPUT_ROOT}"
echo "After rsyncing needed outputs, clean scratch with:"
echo "  bash '${OUTPUT_ROOT}/cleanup_after_rsync.sh'"
