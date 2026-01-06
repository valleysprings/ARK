#!/bin/bash
# KG Generation Pipeline
# Usage: bash generate_kg.sh <dataset> <start> <end> [--skip_build] [--skip_augment]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${PROJECT_ROOT}"

DATASET=${1:-"legal"}
START_IDX=${2:-0}
END_IDX=${3:-5}
shift 3 2>/dev/null || true  # Remove first 3 args, keep the rest (flags)

# Log setup
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="${PROJECT_ROOT}/log/kg"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/kg_${DATASET}_${START_IDX}_${END_IDX}_${TIMESTAMP}.log"

exec > >(tee -a "${LOG_FILE}") 2>&1
echo "Log file: ${LOG_FILE}"

echo "=================================================="
echo "KG Generation Pipeline"
echo "Dataset: $DATASET"
echo "Range: $START_IDX to $END_IDX"
echo "=================================================="

PYTHONPATH="${PROJECT_ROOT}" python src/kg/graph_builder.py \
    --dataset "$DATASET" \
    --start_index "$START_IDX" \
    --end_index "$END_IDX" \
    --config src/config/graph_builder.yaml \
    "$@"

echo ""
echo "=================================================="
echo "KG Generation Completed!"
echo "=================================================="
