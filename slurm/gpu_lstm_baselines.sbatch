#!/bin/bash
#SBATCH --job-name=gen-lstm
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

MANIFEST="${1:-configs/lstm_gpu_example.json}"
PYTHON_BIN="${PYTHON:-python3}"

mkdir -p logs
"${PYTHON_BIN}" -m generate_baselines_mila validate-manifest --manifest "${MANIFEST}"
"${PYTHON_BIN}" -m generate_baselines_mila generate-lstm --manifest "${MANIFEST}"
