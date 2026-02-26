#!/bin/bash
# Qwen Finetuned vs Baseline Pairwise Comparison
# Usage: bash run_qwen_comp.sh [dataset] [baseline] [samples] [train_dataset]
# baseline: bge, jina, stella, no_retrieval

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

DATASET=${1:-"hotpotqa"}
BASELINE=${2:-"bge"}
SAMPLES=${3:-100}
TRAIN_DATASET=${4:-"legal"}

# Paths
QWEN_RESULTS="results/raw/sample_${SAMPLES}/${DATASET}/qwen_${TRAIN_DATASET}.jsonl"
BASELINE_RESULTS="results/raw/sample_${SAMPLES}/${DATASET}/${BASELINE}.jsonl"

echo "=================================================="
echo "Pairwise Comparison - Qwen (${TRAIN_DATASET}) vs ${BASELINE}"
echo "=================================================="
echo "Dataset: ${DATASET}"
echo "Samples: ${SAMPLES}"
echo "Qwen results: ${QWEN_RESULTS}"
echo "Baseline results: ${BASELINE_RESULTS}"
echo "=================================================="

# Check files exist
if [ ! -f "${QWEN_RESULTS}" ]; then
    echo "Error: Qwen results not found: ${QWEN_RESULTS}"
    exit 1
fi

if [ ! -f "${BASELINE_RESULTS}" ]; then
    echo "Error: Baseline results not found: ${BASELINE_RESULTS}"
    exit 1
fi

PYTHONPATH="${PROJECT_ROOT}" python src/inference/run_pairwise_eval.py \
    --model1 "${QWEN_RESULTS}" \
    --model2 "${BASELINE_RESULTS}" \
    --output-dir "results/pairwise"

echo ""
echo "Comparison completed!"
