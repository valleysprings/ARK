#!/bin/bash
# No Retrieval Baseline Evaluation (LLM only)
# Usage: bash run_no_retrieval.sh [dataset] [start_idx] [end_idx]

set -e

DATASET=${1:-"hotpotqa"}
START_IDX=${2:-0}
END_IDX=${3:-100}

OUTPUT_DIR="experiments/results/base"
DATA_PATH="data/raw/${DATASET}.jsonl"
LLM_CONFIG="src/config/llm_inference.yaml"
RETRIEVAL_CONFIG="src/config/retrieval_model.yaml"
DEVICE="cuda:0"

mkdir -p "${OUTPUT_DIR}"

echo "=================================================="
echo "ARK Evaluation - No Retrieval Baseline"
echo "=================================================="
echo "Dataset: ${DATASET}"
echo "Range: ${START_IDX} to ${END_IDX}"
echo "Output: ${OUTPUT_DIR}/${DATASET}_no_retrieval.jsonl"
echo "=================================================="
echo "Note: This baseline uses LLM only without any retrieval"
echo "=================================================="

python src/inference/run_inference.py \
    --dataset "${DATA_PATH}" \
    --output "${OUTPUT_DIR}/${DATASET}_no_retrieval.jsonl" \
    --retriever no_retrieval \
    --llm-config "${LLM_CONFIG}" \
    --retrieval-config "${RETRIEVAL_CONFIG}" \
    --device "${DEVICE}" \
    --start-idx "${START_IDX}" \
    --end-idx "${END_IDX}"

echo ""
echo "✅ No retrieval baseline evaluation completed!"
echo "Results: ${OUTPUT_DIR}/${DATASET}_no_retrieval.jsonl"
