#!/bin/bash
# KG Generation via OpenAI-compatible API (remote models)
# Usage: bash generate_kg_openai.sh <dataset> <start> <end> [OPTIONS]
#
# Options:
#   --provider <name>       LLM provider: gpt | deepseek | gemini (default: gpt)
#   --max_concurrent <int>  Max concurrent documents (default: 1)
#   --max_async_calls <int> Max concurrent LLM calls (default: 10)
#   --skip_build            Skip KG construction step
#   --skip_augment          Skip KG augmentation step

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

DATASET=${1:-"legal"}
START_IDX=${2:-0}
END_IDX=${3:-100}
shift 3 2>/dev/null || true

# Defaults
PROVIDER="gpt"
MAX_CONCURRENT=2
MAX_ASYNC_CALLS=20
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --provider)        PROVIDER="$2"; shift 2 ;;
        --max_concurrent)  MAX_CONCURRENT="$2"; shift 2 ;;
        --max_async_calls) MAX_ASYNC_CALLS="$2"; shift 2 ;;
        *)                 EXTRA_ARGS+=("$1"); shift ;;
    esac
done

# Resolve actual model name from llm.yaml
MODEL_NAME=$(python3 -c "
import yaml
with open('src/config/llm.yaml') as f:
    cfg = yaml.safe_load(f)
print(cfg.get('llm_api', {}).get('${PROVIDER}', {}).get('model', '${PROVIDER}'))
" 2>/dev/null || echo "$PROVIDER")

# Log setup
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="${PROJECT_ROOT}/log/kg/${MODEL_NAME}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/kg_${DATASET}_${START_IDX}_${END_IDX}_${TIMESTAMP}.log"

exec > >(tee -a "${LOG_FILE}") 2>&1
echo "Log file: ${LOG_FILE}"

echo "=================================================="
echo "KG Generation Pipeline (OpenAI API)"
echo "  Dataset: $DATASET"
echo "  Range: $START_IDX to $END_IDX"
echo "  Provider: $PROVIDER"
echo "  Model: $MODEL_NAME"
echo "=================================================="

# Output directory - use MODEL_NAME so log and data paths stay consistent
OUTPUT_DIR="${PROJECT_ROOT}/data/preprocessed/kg/${MODEL_NAME}/${DATASET}"

# Run pipeline (augmentation skipped by default — run separately with augment_kg.sh)
PYTHONPATH="${PROJECT_ROOT}" python src/kg/graph_builder.py \
    --dataset "$DATASET" \
    --start_index "$START_IDX" \
    --end_index "$END_IDX" \
    --config src/config/kg.yaml \
    --llm_provider "$PROVIDER" \
    --skip_augment \
    --max_concurrent "$MAX_CONCURRENT" \
    --max_async_calls "$MAX_ASYNC_CALLS" \
    --output_dir "$OUTPUT_DIR" \
    "${EXTRA_ARGS[@]}"

echo ""
echo "=================================================="
echo "KG Generation Completed!"
echo "=================================================="
