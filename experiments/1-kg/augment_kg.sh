#!/bin/bash
# KG Augmentation (Stage 2) - Add similarity edges using local embedding model
# Usage: bash augment_kg.sh <dataset> <start> <end> [OPTIONS]
#
# Options:
#   --embedding_provider <name>  bge | qwen (default: bge)
#   --embedding_model_path <path> Model path (default: auto from provider)
#   --model_name <name>          LLM model name used in Stage 1 (for locating KG files)
#   --similarity_threshold <f>   Similarity threshold (default: 0.8)
#   --max_concurrent <int>       Max concurrent documents (default: 5)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

DATASET=${1:-"legal"}
START_IDX=${2:-0}
END_IDX=${3:-100}
shift 3 2>/dev/null || true

# Defaults
EMBEDDING_PROVIDER="qwen"
EMBEDDING_MODEL_PATH=""
MODEL_NAME="gemini-2.5-flash"
# MODEL_NAME="gemini-2.5-flash-nothinking"
# MODEL_NAME="Qwen3-235B-A22B-Instruct-2507-FP8"
# MODEL_NAME="Qwen3-30B-A3B-Instruct"
SIMILARITY_THRESHOLD=0.8
MAX_CONCURRENT=5
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --embedding_provider)    EMBEDDING_PROVIDER="$2"; shift 2 ;;
        --embedding_model_path)  EMBEDDING_MODEL_PATH="$2"; shift 2 ;;
        --model_name)            MODEL_NAME="$2"; shift 2 ;;
        --similarity_threshold)  SIMILARITY_THRESHOLD="$2"; shift 2 ;;
        --max_concurrent)        MAX_CONCURRENT="$2"; shift 2 ;;
        *)                       EXTRA_ARGS+=("$1"); shift ;;
    esac
done

OUTPUT_DIR="${PROJECT_ROOT}/data/preprocessed/kg/${MODEL_NAME}/${DATASET}"

# Log setup
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="${PROJECT_ROOT}/log/kg/${MODEL_NAME}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/augment_${DATASET}_${START_IDX}_${END_IDX}_${TIMESTAMP}.log"

exec > >(tee -a "${LOG_FILE}") 2>&1
echo "Log file: ${LOG_FILE}"

echo "=================================================="
echo "KG Augmentation (Stage 2)"
echo "  Dataset: $DATASET"
echo "  Range: $START_IDX to $END_IDX"
echo "  Embedding: $EMBEDDING_PROVIDER"
echo "  Model name: $MODEL_NAME"
echo "  Output: $OUTPUT_DIR"
echo "=================================================="

# Build embedding args
EMB_ARGS=(--embedding_provider "$EMBEDDING_PROVIDER")
if [[ -n "$EMBEDDING_MODEL_PATH" ]]; then
    EMB_ARGS+=(--embedding_model_path "$EMBEDDING_MODEL_PATH")
fi

PYTHONPATH="${PROJECT_ROOT}" python src/kg/graph_builder.py \
    --dataset "$DATASET" \
    --start_index "$START_IDX" \
    --end_index "$END_IDX" \
    --config src/config/kg.yaml \
    --skip_build \
    --similarity_threshold "$SIMILARITY_THRESHOLD" \
    --max_concurrent "$MAX_CONCURRENT" \
    --output_dir "$OUTPUT_DIR" \
    "${EMB_ARGS[@]}" \
    "${EXTRA_ARGS[@]}"

echo ""
echo "=================================================="
echo "KG Augmentation Completed!"
echo "=================================================="
