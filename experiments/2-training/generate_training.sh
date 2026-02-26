#!/bin/bash
# Generate training data from alignment + matched chunks
# Usage: bash generate_training.sh [dataset] [stage] [llm] [embed]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

DATASET=${1:-"legal"}
STAGE=${2:-"all"}
LLM_NAME=${3:-"mistral"}
EMBED_NAME=${4:-"qwen"}
ALIGN_CONFIG="src/config/alignment.yaml"

# Chunking configurations (parallel arrays)
CHUNK_METHODS=("sentence" "sentence" "token" "token")
CHUNK_SIZES=("1" "5" "512" "64")
OVERLAPS=("0" "1" "12" "8")

LOG_DIR="${PROJECT_ROOT}/log/training_data"
mkdir -p "${LOG_DIR}"

for i in "${!CHUNK_METHODS[@]}"; do
    CHUNK_METHOD="${CHUNK_METHODS[$i]}"
    CHUNK_SIZE="${CHUNK_SIZES[$i]}"
    OVERLAP="${OVERLAPS[$i]}"
    CHUNK_TAG="${CHUNK_METHOD}_${CHUNK_SIZE}_o_${OVERLAP}"
    ALIGNMENT_DIR="./data/preprocessed/alignment/${LLM_NAME}_${EMBED_NAME}/${DATASET}/${CHUNK_METHOD}/${CHUNK_TAG}"

    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    LOG_FILE="${LOG_DIR}/training_${DATASET}_${CHUNK_TAG}_${TIMESTAMP}.log"

    echo "=================================================="
    echo "ARK Training Data Generation"
    echo "=================================================="
    echo "Dataset: $DATASET | Chunk: $CHUNK_TAG | Stage: $STAGE"
    echo "Alignment: $ALIGNMENT_DIR"
    echo "=================================================="

    PYTHONPATH="${PROJECT_ROOT}" python -m src.training.generate_neg \
        --mode training \
        --dataset "$DATASET" \
        --stage "$STAGE" \
        --align_config "$ALIGN_CONFIG" \
        --alignment_dir "$ALIGNMENT_DIR" \
        2>&1 | tee "$LOG_FILE"

    echo "[$CHUNK_TAG] Done."
    echo ""
done

echo "All chunking configurations completed."
