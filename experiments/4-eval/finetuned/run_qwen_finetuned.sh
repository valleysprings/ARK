#!/bin/bash
# Qwen Finetuned Model Evaluation
# Usage: bash run_qwen_finetuned.sh [dataset] [checkpoint_path] [start_idx] [end_idx]
#
# Example checkpoint path format:
#   model/finetuned/qwen3/2wikimqa/nolabel/chunk64_32/checkpoint-70

set -e

DATASET=${1:-"2wikimqa"}
CHECKPOINT_PATH=${2:-"model/finetuned/qwen3/2wikimqa/nolabel/chunk64_32/checkpoint-70"}
START_IDX=${3:-0}
END_IDX=${4:-100}

OUTPUT_DIR="experiments/results/finetuned"
DATA_PATH="data/raw/${DATASET}.jsonl"
LLM_CONFIG="src/config/llm_inference.yaml"
RETRIEVAL_CONFIG="src/config/retrieval_model.yaml"
DEVICE="cuda:0"

mkdir -p "${OUTPUT_DIR}"

# Extract checkpoint info for output filename
CHECKPOINT_NAME=$(basename "${CHECKPOINT_PATH}")
OUTPUT_FILE="${OUTPUT_DIR}/${DATASET}_qwen_${CHECKPOINT_NAME}.jsonl"

echo "=================================================="
echo "ARK Evaluation - Qwen Finetuned Model"
echo "=================================================="
echo "Dataset: ${DATASET}"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Range: ${START_IDX} to ${END_IDX}"
echo "Output: ${OUTPUT_FILE}"
echo "=================================================="

# Verify checkpoint exists
if [ ! -d "${CHECKPOINT_PATH}" ]; then
    echo "❌ Error: Checkpoint path not found: ${CHECKPOINT_PATH}"
    echo ""
    echo "Available checkpoints:"
    if [ -d "model/finetuned/qwen3/${DATASET}" ]; then
        find "model/finetuned/qwen3/${DATASET}" -type d -name "checkpoint-*" | head -10
    else
        echo "  No checkpoints found for dataset: ${DATASET}"
    fi
    exit 1
fi

# Run inference with finetuned model
python src/inference/run_inference.py \
    --dataset "${DATA_PATH}" \
    --output "${OUTPUT_FILE}" \
    --retriever qwen \
    --model-path "${CHECKPOINT_PATH}" \
    --llm-config "${LLM_CONFIG}" \
    --retrieval-config "${RETRIEVAL_CONFIG}" \
    --device "${DEVICE}" \
    --start-idx "${START_IDX}" \
    --end-idx "${END_IDX}"

echo ""
echo "✅ Qwen finetuned evaluation completed!"
echo "Results: ${OUTPUT_FILE}"
echo ""
echo "To compare with base model:"
echo "  bash experiments/eval/base/run_qwen.sh ${DATASET} ${START_IDX} ${END_IDX}"
