#!/bin/bash
# Stella Embedding Base Model Evaluation
# Usage: bash run_stella.sh [dataset] [limit]

set -e

# Get the project root directory (3 levels up from this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# Change to project root
cd "${PROJECT_ROOT}"

DATASET=${1:-"hotpotqa"}
LIMIT=${2:-100}

# Log setup
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="${PROJECT_ROOT}/log/eval"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/stella_${DATASET}_${LIMIT}_${TIMESTAMP}.log"

exec > >(tee -a "${LOG_FILE}") 2>&1
echo "Log file: ${LOG_FILE}"

DATA_PATH="data/raw/${DATASET}.jsonl"
LLM_CONFIG="src/config/llm_inference.yaml"
RETRIEVAL_CONFIG="src/config/retrieval_model.yaml"
DEVICE="cuda:0"

echo "=================================================="
echo "ARK Evaluation - Stella Embedding (Base)"
echo "=================================================="
echo "Dataset: ${DATASET}"
echo "Limit: ${LIMIT} samples"
echo "=================================================="

PYTHONPATH="${PROJECT_ROOT}" python src/inference/run_inference.py \
    --dataset "${DATA_PATH}" \
    --retriever stella \
    --llm-config "${LLM_CONFIG}" \
    --retrieval-config "${RETRIEVAL_CONFIG}" \
    --device "${DEVICE}" \
    --limit "${LIMIT}"

echo ""
echo "✅ Stella evaluation completed!"
echo "Results saved to: results/raw/limit_${LIMIT}/${DATASET}/stella.jsonl"
echo "Scores saved to: results/score/limit_${LIMIT}/${DATASET}/stella.json"
