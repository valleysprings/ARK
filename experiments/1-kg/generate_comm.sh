#!/bin/bash
# Community search: LLM entity extraction + PPR subgraph extraction
# Usage: bash generate_comm.sh [dataset] [start] [end] [max_async_calls] [kg_model]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

DATASET=${1:-"legal"}
START=${2:-0}
END=${3:-200}
MAX_ASYNC=${4:-20}
KG_MODEL=${5:-"gemini-2.5-flash-nothinking"}

KG_DIR="./data/preprocessed/kg/${KG_MODEL}/${DATASET}"
DATA_PATH="./data/raw/${DATASET}.jsonl"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="${PROJECT_ROOT}/log/community"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/comm_${DATASET}_${START}_${END}_${TIMESTAMP}.log"

exec > >(tee -a "${LOG_FILE}") 2>&1
echo "Log: ${LOG_FILE}"

echo "=================================================="
echo "Community Search (LLM extraction + PPR)"
echo "=================================================="
echo "Dataset: $DATASET | Range: $START-$END | Async: $MAX_ASYNC"
echo "KG dir: $KG_DIR"
echo "Data:   $DATA_PATH"
echo "=================================================="

PYTHONPATH="${PROJECT_ROOT}" python -m src.training.generate_neg \
    --mode community \
    --dataset "$DATASET" \
    --start "$START" \
    --end "$END" \
    --kg_dir "$KG_DIR" \
    --data "$DATA_PATH" \
    --max_async_calls "$MAX_ASYNC"

echo ""
echo "=================================================="
echo "Community search complete."
echo "Output: ${KG_DIR}/subgraphs_answer_{large,small}/"
echo "Entities: ${KG_DIR}/{answer,query}_entities.json"
echo "=================================================="
