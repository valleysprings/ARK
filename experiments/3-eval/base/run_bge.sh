#!/bin/bash
# BGE-M3 Base Model Evaluation (All Datasets)
# Usage: bash run_bge.sh [samples]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

SAMPLES=${1:-100}

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="${PROJECT_ROOT}/log/eval"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/bge_all_${SAMPLES}_${TIMESTAMP}.log"

echo "=================================================="
echo "ARK Evaluation - BGE-M3 (Base) - All Datasets"
echo "Samples: ${SAMPLES} samples per dataset"
echo "Log: ${LOG_FILE}"
echo "=================================================="

PYTHONPATH="${PROJECT_ROOT}" python src/inference/run_inference.py \
    --dataset "data/raw" \
    --retriever bge \
    --llm-config "src/config/llm.yaml" \
    --retrieval-config "src/config/retrieval_model.yaml" \
    --device "cuda:0" \
    --llm-device "cuda:1" \
    --limit "${SAMPLES}" \
    2>&1 | tee -a "${LOG_FILE}"

echo "✅ BGE all datasets evaluation completed!"
