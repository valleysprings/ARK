#!/bin/bash
# KG Statistics
# Usage: bash experiments/4-stats/kg.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

# ---- Config (edit here) ----
MODEL="gemini-2.5-flash"
# MODEL="gemini-2.5-flash-nothinking"
# MODEL="Qwen3-235B-A22B-Instruct-2507-FP8"
# MODEL="Qwen3-30B-A3B-Instruct"
DATASET="legal"
AUGMENTED=1          # 0=full_kg, 1=full_kg_augmented
START_INDEX=0
END_INDEX=100
# -----------------------------

if [[ "$AUGMENTED" -eq 1 ]]; then
    KG_DIR="${PROJECT_ROOT}/data/preprocessed/kg/${MODEL}/${DATASET}/full_kg_augmented"
    SUFFIX="_augmented"
else
    KG_DIR="${PROJECT_ROOT}/data/preprocessed/kg/${MODEL}/${DATASET}/full_kg"
    SUFFIX=""
fi

OUTPUT_DIR="${PROJECT_ROOT}/log/kg_stats/${MODEL}"
mkdir -p "${OUTPUT_DIR}"
OUTPUT_FILE="${OUTPUT_DIR}/${DATASET}${SUFFIX}_${START_INDEX}_${END_INDEX}.json"

echo "KG Stats: model=${MODEL} dataset=${DATASET} augmented=${AUGMENTED}"
echo "Range: ${START_INDEX} to ${END_INDEX}"
echo "KG dir: ${KG_DIR}"
echo "Output: ${OUTPUT_FILE}"

PYTHONPATH="${PROJECT_ROOT}" python src/kg/utils/stats.py \
    --kg_dir "${KG_DIR}" \
    --output "${OUTPUT_FILE}" \
    --start_index "${START_INDEX}" \
    --end_index "${END_INDEX}"
