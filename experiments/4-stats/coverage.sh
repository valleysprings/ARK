#!/bin/bash
# Query-to-KG entity coverage test
# Usage: bash experiments/4-stats/coverage.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

# ---- Config (edit here) ----
MODEL="gemini-2.5-flash-nothinking"
DATASET="legal"
MAX_ASYNC_CALLS=20
MODE="answer"         # query or answer
START_INDEX=0
END_INDEX=200
# -----------------------------

KG_DIR="${PROJECT_ROOT}/data/preprocessed/${MODEL}/${DATASET}/full_kg"

DATA_FILE="${PROJECT_ROOT}/data/raw/additional/${DATASET}.jsonl"
if [[ ! -f "$DATA_FILE" ]]; then
    DATA_FILE="${PROJECT_ROOT}/data/raw/${DATASET}.jsonl"
fi

OUTPUT_DIR="${PROJECT_ROOT}/log/coverage/${MODEL}/${MODE}"
mkdir -p "${OUTPUT_DIR}"
OUTPUT_FILE="${OUTPUT_DIR}/${DATASET}_${START_INDEX}_${END_INDEX}.json"

echo "Coverage: model=${MODEL} dataset=${DATASET} mode=${MODE}"
echo "Range: ${START_INDEX} to ${END_INDEX}"
echo "Data: ${DATA_FILE}"
echo "KG dir: ${KG_DIR}"
echo "Output: ${OUTPUT_FILE}"

PYTHONPATH="${PROJECT_ROOT}" python src/kg/utils/coverage.py \
    --data "${DATA_FILE}" \
    --kg_dir "${KG_DIR}" \
    --dataset "${DATASET}" \
    --mode "${MODE}" \
    --start_index "${START_INDEX}" \
    --end_index "${END_INDEX}" \
    --max_async_calls "${MAX_ASYNC_CALLS}" \
    --output "${OUTPUT_FILE}"
