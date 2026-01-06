#!/bin/bash
# Qwen Finetuned Model Evaluation
# Usage: bash run_qwen.sh [eval_dataset] [train_dataset] [limit] [gpu]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

EVAL_DATASET=${1:-"hotpotqa"}
TRAIN_DATASET=${2:-"legal"}
LIMIT=${3:-100}
GPU=${4:-"3"}

export CUDA_VISIBLE_DEVICES=$GPU

# Checkpoint path: find latest checkpoint
BASE_CHECKPOINT="model/checkpoints/${TRAIN_DATASET}"
CHECKPOINT_PATH=$(find "$BASE_CHECKPOINT" -maxdepth 1 -type d -name "checkpoint-*" | sort -V | tail -1)
if [ -z "$CHECKPOINT_PATH" ]; then
    CHECKPOINT_PATH="$BASE_CHECKPOINT"
fi

# Log setup
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="${PROJECT_ROOT}/log/eval"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/qwen_${TRAIN_DATASET}_${EVAL_DATASET}_${LIMIT}_${TIMESTAMP}.log"

exec > >(tee -a "${LOG_FILE}") 2>&1
echo "Log file: ${LOG_FILE}"

DATA_PATH="data/raw/${EVAL_DATASET}.jsonl"
LLM_CONFIG="src/config/llm_inference.yaml"
RETRIEVAL_CONFIG="src/config/retrieval_model.yaml"

echo "=================================================="
echo "ARK Evaluation - Qwen Finetuned"
echo "=================================================="
echo "Eval Dataset: ${EVAL_DATASET}"
echo "Train Dataset: ${TRAIN_DATASET}"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Limit: ${LIMIT} samples"
echo "GPU: ${GPU}"
echo "=================================================="

# Verify checkpoint exists
if [ ! -d "${CHECKPOINT_PATH}" ]; then
    echo "Error: Checkpoint not found: ${CHECKPOINT_PATH}"
    exit 1
fi

PYTHONPATH="${PROJECT_ROOT}" python src/inference/run_inference.py \
    --dataset "${DATA_PATH}" \
    --retriever qwen \
    --model-path "${CHECKPOINT_PATH}" \
    --model-suffix "${TRAIN_DATASET}" \
    --llm-config "${LLM_CONFIG}" \
    --retrieval-config "${RETRIEVAL_CONFIG}" \
    --limit "${LIMIT}"

echo ""
echo "✅ Qwen finetuned evaluation completed!"
echo "Results saved to: results/raw/limit_${LIMIT}/${EVAL_DATASET}/qwen_${TRAIN_DATASET}.jsonl"
