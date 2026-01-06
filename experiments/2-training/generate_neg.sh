#!/bin/bash
# ARK Data Generation Pipeline
# Complete pipeline: subgraph extraction → query generation → training data
# Usage: bash generate_data.sh [dataset] [start] [end] [stage] [gpu]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${PROJECT_ROOT}"

DATASET=${1:-"fin"}
START=${2:-0}
END=${3:-5}
STAGE=${4:-"all"}
GPU=${5:-"3"}

export CUDA_VISIBLE_DEVICES=$GPU

# Log setup
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="${PROJECT_ROOT}/log/training"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/data_${DATASET}_${START}_${END}_${TIMESTAMP}.log"

exec > >(tee -a "${LOG_FILE}") 2>&1
echo "Log file: ${LOG_FILE}"

echo "=================================================="
echo "ARK Data Generation Pipeline"
echo "=================================================="
echo "Dataset: $DATASET"
echo "Range: $START to $END"
echo "Stage: $STAGE"
echo "=================================================="

PYTHONPATH="${PROJECT_ROOT}" python -m src.training.generate_neg \
    --dataset "$DATASET" \
    --start "$START" \
    --end "$END" \
    --stage "$STAGE"

echo ""
echo "=================================================="
echo "Pipeline Complete!"
echo "=================================================="
