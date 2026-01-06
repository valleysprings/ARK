#!/bin/bash
# Qwen Finetuned Model Evaluation (Multi Dataset)
# Usage: bash run_qwen_multi.sh [eval_dataset] [start_idx] [end_idx] [gpu]

set -e

EVAL_DATASET=${1:-"2wikimqa"}
START_IDX=${2:-0}
END_IDX=${3:-100}
GPU=${4:-"0"}

export CUDA_VISIBLE_DEVICES=$GPU

# Checkpoint path: model/checkpoints/multi
CHECKPOINT_PATH="model/checkpoints/multi"

OUTPUT_DIR="experiments/results/finetuned"
DATA_PATH="data/raw/${EVAL_DATASET}.jsonl"
LLM_CONFIG="src/config/llm_inference.yaml"
RETRIEVAL_CONFIG="src/config/retrieval_model.yaml"

mkdir -p "${OUTPUT_DIR}"

OUTPUT_FILE="${OUTPUT_DIR}/${EVAL_DATASET}_qwen_multi.jsonl"

echo "=================================================="
echo "ARK Evaluation - Qwen Finetuned (Multi)"
echo "=================================================="
echo "Eval Dataset: ${EVAL_DATASET}"
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
