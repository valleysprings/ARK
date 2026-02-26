#!/bin/bash
# KG Generation via vLLM (local model serving)
# Usage: bash generate_kg_vllm.sh <dataset> <start> <end> [OPTIONS]
#
# Options:
#   --vllm_model <path>     Model path (default: model/llm/Qwen3-235B-A22B-Thinking-FP8)
#   --vllm_tp <int>         Tensor parallelism (default: 8)
#   --vllm_dp <int>         Data parallelism (default: 1)
#   --vllm_port <int>       Server port (default: 8000)
#   --vllm_max_model_len <int>  Max context length (default: 16384)
#   --max_concurrent <int>  Max concurrent documents (default: 5)
#   --max_async_calls <int> Max concurrent LLM calls (default: 64)
#   --skip_build            Skip KG construction step
#   --skip_augment          Skip KG augmentation step

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

DATASET=${1:-"legal"}
START_IDX=${2:-0}
END_IDX=${3:-200}
shift 3 2>/dev/null || true

# Defaults
# VLLM_MODEL="model/llm/Qwen3-30B-A3B-Instruct"
# VLLM_TP=1
# VLLM_DP=8
VLLM_MODEL="model/llm/Qwen3-235B-A22B-Instruct-2507-FP8"
VLLM_TP=2
VLLM_DP=4

VLLM_PORT=8000
VLLM_MAX_MODEL_LEN=8192
MAX_CONCURRENT=5
MAX_ASYNC_CALLS=64
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --vllm_model)      VLLM_MODEL="$2"; shift 2 ;;
        --vllm_tp)         VLLM_TP="$2"; shift 2 ;;
        --vllm_dp)         VLLM_DP="$2"; shift 2 ;;
        --vllm_port)       VLLM_PORT="$2"; shift 2 ;;
        --vllm_max_model_len) VLLM_MAX_MODEL_LEN="$2"; shift 2 ;;
        --max_concurrent)  MAX_CONCURRENT="$2"; shift 2 ;;
        --max_async_calls) MAX_ASYNC_CALLS="$2"; shift 2 ;;
        *)                 EXTRA_ARGS+=("$1"); shift ;;
    esac
done

# Model short name for log/output paths
MODEL_SHORT=$(basename "$VLLM_MODEL")

# Log setup
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="${PROJECT_ROOT}/log/kg/${MODEL_SHORT}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/kg_${DATASET}_${START_IDX}_${END_IDX}_${TIMESTAMP}.log"

exec > >(tee -a "${LOG_FILE}") 2>&1
echo "Log file: ${LOG_FILE}"

echo "=================================================="
echo "KG Generation Pipeline (vLLM)"
echo "  Dataset: $DATASET"
echo "  Range: $START_IDX to $END_IDX"
echo "  Model: $VLLM_MODEL"
echo "  TP: $VLLM_TP  DP: $VLLM_DP"
echo "  Port: $VLLM_PORT"
echo "  Max model len: $VLLM_MAX_MODEL_LEN"
echo "=================================================="

# Export for Python
export VLLM_BASE_URL="http://localhost:${VLLM_PORT}/v1"
export VLLM_MODEL="$VLLM_MODEL"

# Start vLLM server
VLLM_PID=""
cleanup() {
    if [[ -n "$VLLM_PID" ]]; then
        echo "Stopping vLLM server (PID: $VLLM_PID)..."
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
        echo "vLLM server stopped."
    fi
}
trap cleanup EXIT

VLLM_SERVE_ARGS=(
    "$VLLM_MODEL"
    --port "$VLLM_PORT"
    --tensor-parallel-size "$VLLM_TP"
    --max-model-len "$VLLM_MAX_MODEL_LEN"
    --trust-remote-code
    --enable-prefix-caching
    --enable-expert-parallel
    --disable-log-requests
)
if [[ "$VLLM_DP" -gt 1 ]]; then
    VLLM_SERVE_ARGS+=(--data-parallel-size "$VLLM_DP")
fi

vllm serve "${VLLM_SERVE_ARGS[@]}" &
VLLM_PID=$!

echo "Waiting for vLLM server to be ready..."
for i in $(seq 1 180); do
    if curl -s "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
        echo "vLLM server ready on port $VLLM_PORT (PID: $VLLM_PID)"
        break
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "ERROR: vLLM server process died"
        exit 1
    fi
    sleep 2
done

if ! curl -s "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
    echo "ERROR: vLLM server failed to start within timeout"
    exit 1
fi

# Output directory - use MODEL_SHORT so log and data paths stay consistent
OUTPUT_DIR="${PROJECT_ROOT}/data/preprocessed/kg/${MODEL_SHORT}/${DATASET}"

# Run pipeline (augmentation skipped by default — run separately with --embedding_provider bge/qwen)
PYTHONPATH="${PROJECT_ROOT}" python src/kg/graph_builder.py \
    --dataset "$DATASET" \
    --start_index "$START_IDX" \
    --end_index "$END_IDX" \
    --config src/config/kg.yaml \
    --llm_provider vllm \
    --skip_augment \
    --max_concurrent "$MAX_CONCURRENT" \
    --max_async_calls "$MAX_ASYNC_CALLS" \
    --output_dir "$OUTPUT_DIR" \
    "${EXTRA_ARGS[@]}"

echo ""
echo "=================================================="
echo "KG Generation Completed!"
echo "=================================================="
