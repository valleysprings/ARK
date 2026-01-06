#!/bin/bash
# Compute triple alignment scores for training data
# Usage: bash compute_alignment.sh [dataset] [start] [end]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${PROJECT_ROOT}"

DATASET=${1:-"fin"}
START_IDX=${2:-0}
END_IDX=${3:-10}

# Log setup
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="${PROJECT_ROOT}/log/alignment"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/alignment_${DATASET}_${START_IDX}_${END_IDX}_${TIMESTAMP}.log"

exec > >(tee -a "${LOG_FILE}") 2>&1
echo "Log file: ${LOG_FILE}"

INPUT_FILE="./data/raw/${DATASET}.jsonl"
OUTPUT_DIR="./data/preprocessed/${DATASET}/alignment"

echo "=================================================="
echo "Computing Triple Alignment Scores"
echo "=================================================="
echo "Dataset: $DATASET"
echo "Range: $START_IDX to $END_IDX"
echo "Output: $OUTPUT_DIR"
echo "=================================================="

mkdir -p "$OUTPUT_DIR"

PYTHONPATH="${PROJECT_ROOT}" python src/training/generate_pos.py \
    --input_jsonl "$INPUT_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --config "src/config/alignment.yaml" \
    --lm_model "./model/raw/Mistral-7B" \
    --embedding_model "./model/raw/bge-m3" \
    --top_k 1000 \
    --start_index "$START_IDX" \
    --end_index "$END_IDX"

echo ""
echo "=================================================="
echo "Alignment scoring completed!"
echo "Results saved to: $OUTPUT_DIR/"
echo "=================================================="
