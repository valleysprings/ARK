#!/bin/bash
# Jina Embedding Base Model Evaluation
# Usage: bash run_jina.sh [eval_mode] [dataset] [limit]
# eval_mode: f1, llm, or both (default: f1)

set -e

# Get the project root directory (3 levels up from this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# Change to project root
cd "${PROJECT_ROOT}"

EVAL_MODE=${1:-"f1"}
DATASET=${2:-"hotpotqa"}
LIMIT=${3:-100}

# Log setup
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="${PROJECT_ROOT}/log/eval"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/jina_${DATASET}_${LIMIT}_${TIMESTAMP}.log"

exec > >(tee -a "${LOG_FILE}") 2>&1
echo "Log file: ${LOG_FILE}"

DATA_PATH="data/raw/${DATASET}.jsonl"
LLM_CONFIG="src/config/llm_inference.yaml"
RETRIEVAL_CONFIG="src/config/retrieval_model.yaml"
DEVICE="cuda:0"

echo "=================================================="
echo "ARK Evaluation - Jina Embedding (Base)"
echo "=================================================="
echo "Eval mode: ${EVAL_MODE}"
echo "Dataset: ${DATASET}"
echo "Limit: ${LIMIT} samples"
echo "=================================================="

PYTHONPATH="${PROJECT_ROOT}" python src/inference/run_inference.py \
    --dataset "${DATA_PATH}" \
    --retriever jina \
    --llm-config "${LLM_CONFIG}" \
    --retrieval-config "${RETRIEVAL_CONFIG}" \
    --device "${DEVICE}" \
    --limit "${LIMIT}" \
    --eval-mode "${EVAL_MODE}"

echo ""
echo "✅ Jina evaluation completed!"
echo "Results saved to: results/raw/limit_${LIMIT}/${DATASET}/jina.jsonl"
echo "Scores saved to: results/score/limit_${LIMIT}/${DATASET}/jina.json"
