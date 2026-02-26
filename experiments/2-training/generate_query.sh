#!/bin/bash
# Query generation from subgraphs (both large and small KG types)
# Usage: bash generate_query.sh [dataset] [start] [end] [max_async_calls] [query_num] [kg_model]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

DATASET=${1:-"legal"}
START=${2:-0}
END=${3:-200}
MAX_ASYNC=${4:-40}
QUERY_NUM=${5:-10}
KG_MODEL=${6:-"gemini-2.5-flash-nothinking"}

KG_DIR="./data/preprocessed/kg/${KG_MODEL}/${DATASET}"
DATA_PATH="./data/raw/${DATASET}.jsonl"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="${PROJECT_ROOT}/log/queries"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/query_${DATASET}_${START}_${END}_${TIMESTAMP}.log"

exec > >(tee -a "${LOG_FILE}") 2>&1
echo "Log: ${LOG_FILE}"

echo "=================================================="
echo "Query Generation (both large & small KG)"
echo "=================================================="
echo "Dataset: $DATASET | Range: $START-$END | Async: $MAX_ASYNC | Queries/subgraph: $QUERY_NUM"
echo "KG dir: $KG_DIR"
echo "=================================================="

PYTHONPATH="${PROJECT_ROOT}" python -m src.training.generate_neg \
    --mode queries \
    --dataset "$DATASET" \
    --start "$START" \
    --end "$END" \
    --kg_dir "$KG_DIR" \
    --data "$DATA_PATH" \
    --max_async_calls "$MAX_ASYNC" \
    --query_num "$QUERY_NUM"

echo ""
echo "=================================================="
echo "Query generation complete."
echo "Output: ${KG_DIR}/queries_{large,small}/"
echo "=================================================="
