#!/bin/bash
# Qwen3-Embedding Finetuned Model Evaluation (All Datasets)
# Usage: bash run_qwen.sh [samples]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

SAMPLES=${1:-100}

# Checkpoint path (uncomment one to use)
# CHECKPOINT_PATH="model/checkpoints/multi_sentence_1_o_0-stage1/checkpoint-96"
# SUFFIX="FT-stage1"
# CHECKPOINT_PATH="model/checkpoints/multi_sentence_1_o_0-stage2/checkpoint-128"
# SUFFIX="FT-stage2"
CHECKPOINT_PATH="model/checkpoints/multi_sentence_1_o_0/checkpoint-96"
SUFFIX="FT-stage3"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="${PROJECT_ROOT}/log/eval"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/qwen_${SUFFIX}_${SAMPLES}_${TIMESTAMP}.log"

echo "=================================================="
echo "ARK Evaluation - Qwen3 Embedding (Finetuned) - All Datasets"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Samples: ${SAMPLES} samples per dataset"
echo "Log: ${LOG_FILE}"
echo "=================================================="

# model suffix: FT-stage1, FT-stage2, FT-stage3
PYTHONPATH="${PROJECT_ROOT}" python src/inference/run_inference.py \
    --dataset "data/raw" \
    --retriever qwen \
    --model-path "${CHECKPOINT_PATH}" \
    --model-suffix "${SUFFIX}" \
    --llm-config "src/config/llm.yaml" \
    --retrieval-config "src/config/retrieval_model.yaml" \
    --device "cuda:0" \
    --llm-device "cuda:7" \
    --limit "${SAMPLES}" \
    2>&1 | tee -a "${LOG_FILE}"

echo "✅ Qwen finetuned (f1) all datasets evaluation comspleted!"
