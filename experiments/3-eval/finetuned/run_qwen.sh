#!/bin/bash
# Qwen Finetuned Model Evaluation (Single Dataset)
# Usage: bash run_qwen.sh [eval_dataset] [train_dataset] [start_idx] [end_idx] [gpu]

set -e

EVAL_DATASET=${1:-"2wikimqa"}
TRAIN_DATASET=${2:-"fin"}
START_IDX=${3:-0}
END_IDX=${4:-100}
GPU=${5:-"0"}

export CUDA_VISIBLE_DEVICES=$GPU

# Checkpoint path: model/checkpoints/{train_dataset}
CHECKPOINT_PATH="model/checkpoints/${TRAIN_DATASET}"

OUTPUT_DIR="experiments/results/finetuned"
DATA_PATH="data/raw/${EVAL_DATASET}.jsonl"
LLM_CONFIG="src/config/llm_inference.yaml"
RETRIEVAL_CONFIG="src/config/retrieval_model.yaml"

mkdir -p "${OUTPUT_DIR}"

OUTPUT_FILE="${OUTPUT_DIR}/${EVAL_DATASET}_qwen_${TRAIN_DATASET}.jsonl"

echo "=================================================="
echo "ARK Evaluation - Qwen Finetuned (Single)"
echo "=================================================="
echo "Eval Dataset: ${EVAL_DATASET}"
echo "Train Dataset: ${TRAIN_DATASET}"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Range: ${START_IDX} to ${END_IDX}"
echo "GPU: ${GPU}"
echo "Output: ${OUTPUT_FILE}"
echo "=================================================="

# Verify checkpoint exists
if [ ! -d "${CHECKPOINT_PATH}" ]; then
    echo "Error: Checkpoint not found: ${CHECKPOINT_PATH}"
    exit 1
fi

python src/inference/run_inference.py \
    --dataset "${DATA_PATH}" \
    --output "${OUTPUT_FILE}" \
    --retriever qwen \
    --model-path "${CHECKPOINT_PATH}" \
    --llm-config "${LLM_CONFIG}" \
    --retrieval-config "${RETRIEVAL_CONFIG}" \
    --start-idx "${START_IDX}" \
    --end-idx "${END_IDX}"

echo ""
echo "Evaluation completed: ${OUTPUT_FILE}"
