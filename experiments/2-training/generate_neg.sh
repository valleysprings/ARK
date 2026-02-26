#!/bin/bash
# Multi-GPU chunk matching (negative mining)
# Usage: bash generate_neg.sh [dataset] [start] [end] [num_gpus] [kg_model] [llm] [embed]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

DATASET=${1:-"legal"}
START=${2:-0}
END=${3:-200}
NUM_GPUS=${4:-8}
KG_MODEL=${5:-"gemini-2.5-flash-nothinking"}
LLM_NAME=${6:-"mistral"}
EMBED_NAME=${7:-"qwen"}
CONFIG="src/config/training.yaml"
ALIGN_CONFIG="src/config/alignment.yaml"

# Chunking configurations (parallel arrays)
CHUNK_METHODS=("sentence" "sentence" "token" "token")
CHUNK_SIZES=("1" "5" "512" "64")
OVERLAPS=("0" "1" "12" "8")

KG_DIR="./data/preprocessed/kg/${KG_MODEL}/${DATASET}"

for i in "${!CHUNK_METHODS[@]}"; do
    CHUNK_METHOD="${CHUNK_METHODS[$i]}"
    CHUNK_SIZE="${CHUNK_SIZES[$i]}"
    CHUNK_OVERLAP="${OVERLAPS[$i]}"
    CHUNK_TAG="${CHUNK_METHOD}_${CHUNK_SIZE}_o_${CHUNK_OVERLAP}"
    ALIGNMENT_DIR="./data/preprocessed/alignment/${LLM_NAME}_${EMBED_NAME}/${DATASET}/${CHUNK_METHOD}/${CHUNK_TAG}"

    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    LOG_DIR="${PROJECT_ROOT}/log/training_data"
    mkdir -p "${LOG_DIR}"

    echo "=================================================="
    echo "ARK Neg Mining: Multi-GPU Chunk Matching"
    echo "=================================================="
    echo "Dataset: $DATASET | Range: $START-$END | GPUs: $NUM_GPUS"
    echo "Chunk: method=$CHUNK_METHOD size=$CHUNK_SIZE overlap=$CHUNK_OVERLAP"
    echo "KG dir: $KG_DIR"
    echo "Alignment: $ALIGNMENT_DIR"
    echo "=================================================="

    TOTAL=$((END - START))
    SHARD_SIZE=$(( (TOTAL + NUM_GPUS - 1) / NUM_GPUS ))

    PIDS=()
    for GPU in $(seq 0 $((NUM_GPUS - 1))); do
        S=$((START + GPU * SHARD_SIZE))
        E=$((S + SHARD_SIZE))
        [ $E -gt $END ] && E=$END
        [ $S -ge $END ] && break

        LOG_FILE="${LOG_DIR}/match_${DATASET}_${CHUNK_TAG}_gpu${GPU}_${TIMESTAMP}.log"
        echo "GPU $GPU: matching items $S-$E -> $LOG_FILE"

        CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH="${PROJECT_ROOT}" python -m src.training.generate_neg \
            --mode matching \
            --dataset "$DATASET" \
            --start "$S" \
            --end "$E" \
            --config "$CONFIG" \
            --align_config "$ALIGN_CONFIG" \
            --kg_dir "$KG_DIR" \
            --alignment_dir "$ALIGNMENT_DIR" \
            > "$LOG_FILE" 2>&1 &

        PIDS+=($!)
    done

    echo "Launched ${#PIDS[@]} matching workers. Waiting..."

    FAILED=0
    for PID in "${PIDS[@]}"; do
        if ! wait "$PID"; then
            echo "Worker PID $PID failed!"
            FAILED=$((FAILED + 1))
        fi
    done

    if [ $FAILED -gt 0 ]; then
        echo "[$CHUNK_TAG] $FAILED matching worker(s) failed. Check logs in ${LOG_DIR}/"
        exit 1
    fi
    echo "[$CHUNK_TAG] All matching workers completed."
    echo ""
done

echo "All chunking configurations completed."
