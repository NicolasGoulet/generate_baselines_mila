#!/bin/bash
#SBATCH --job-name=gen-ngram
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

MANIFEST="${1:-configs/ngram_additive_example.json}"
PYTHON_BIN="${PYTHON:-python3}"

mkdir -p logs
"${PYTHON_BIN}" -m generate_baselines_mila validate-manifest --manifest "${MANIFEST}"
"${PYTHON_BIN}" -m generate_baselines_mila generate-ngram --manifest "${MANIFEST}"
