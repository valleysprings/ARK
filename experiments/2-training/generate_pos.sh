#!/bin/bash
# Multi-GPU parallel alignment scoring via vLLM offline batch inference
# Usage: bash generate_pos.sh [dataset] [start] [end] [num_gpus] [llm] [embed] [delay_sec]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

DATASET=${1:-"fin"}
START_IDX=${2:-0}
END_IDX=${3:-200}
NUM_GPUS=${4:-8}
LLM_NAME=${5:-"mistral"}
EMBED_NAME=${6:-"qwen"}
DELAY=${7:-3600}

CONFIG="src/config/alignment.yaml"
LLM_CONFIG="src/config/llm.yaml"
LM_MODEL=$(python -c "import yaml; print(yaml.safe_load(open('${LLM_CONFIG}'))['generator']['model'])")
echo "LLM model (from ${LLM_CONFIG}): ${LM_MODEL}"

# Chunking configurations (parallel arrays)
CHUNK_METHODS=("sentence" "sentence" "token" "token")
CHUNK_SIZES=("1" "5" "512" "64")
OVERLAPS=("0" "1" "12" "8")

INPUT_FILE="./data/raw/${DATASET}.jsonl"

for i in "${!CHUNK_METHODS[@]}"; do
    CHUNK_METHOD="${CHUNK_METHODS[$i]}"
    CHUNK_SIZE="${CHUNK_SIZES[$i]}"
    CHUNK_OVERLAP="${OVERLAPS[$i]}"
    CHUNK_TAG="${CHUNK_METHOD}_${CHUNK_SIZE}_o_${CHUNK_OVERLAP}"
    OUTPUT_DIR="./data/preprocessed/alignment/${LLM_NAME}_${EMBED_NAME}/${DATASET}/${CHUNK_METHOD}/${CHUNK_TAG}"
    LOG_DIR="${PROJECT_ROOT}/log/alignment"
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

    TOTAL=$((END_IDX - START_IDX))
    SHARD_SIZE=$(( (TOTAL + NUM_GPUS - 1) / NUM_GPUS ))

    echo "=================================================="
    echo "Parallel Alignment Scoring (vLLM offline)"
    echo "=================================================="
    echo "Dataset: $DATASET | Range: $START_IDX-$END_IDX | GPUs: $NUM_GPUS | Shard: $SHARD_SIZE"
    echo "Chunk: method=$CHUNK_METHOD size=$CHUNK_SIZE overlap=$CHUNK_OVERLAP"
    echo "Models: llm=$LLM_NAME embed=$EMBED_NAME"
    echo "Output: $OUTPUT_DIR"
    echo "=================================================="

    if [ "$DELAY" -gt 0 ] 2>/dev/null; then
        echo "Sleeping ${DELAY}s before launch..."
        sleep "$DELAY"
    fi

    PIDS=()
    for GPU in $(seq 0 $((NUM_GPUS - 1))); do
        S=$((START_IDX + GPU * SHARD_SIZE))
        E=$((S + SHARD_SIZE))
        [ $E -gt $END_IDX ] && E=$END_IDX
        [ $S -ge $END_IDX ] && break

        LOG_FILE="${LOG_DIR}/alignment_${DATASET}_${CHUNK_TAG}_gpu${GPU}_${TIMESTAMP}.log"
        echo "GPU $GPU: items $S-$E -> $LOG_FILE"

        PYTHONPATH="${PROJECT_ROOT}" python src/training/generate_pos.py \
            --input_jsonl "$INPUT_FILE" \
            --output_dir "$OUTPUT_DIR" \
            --config "${CONFIG}" \
            --lm_model "./${LM_MODEL}" \
            --embedding_model "./model/raw/bge-m3" \
            --start_index "$S" \
            --end_index "$E" \
            --use_vllm_offline \
            --gpu_id "$GPU" \
            --chunk_method "$CHUNK_METHOD" \
            --chunk_size "$CHUNK_SIZE" \
            --chunk_overlap "$CHUNK_OVERLAP" \
            > "$LOG_FILE" 2>&1 &

        PIDS+=($!)
    done

    echo "Launched ${#PIDS[@]} workers. Waiting..."

    FAILED=0
    for PID in "${PIDS[@]}"; do
        if ! wait "$PID"; then
            echo "Worker PID $PID failed!"
            FAILED=$((FAILED + 1))
        fi
    done

    echo "=================================================="
    if [ $FAILED -eq 0 ]; then
        echo "[$CHUNK_TAG] All workers completed successfully."
    else
        echo "[$CHUNK_TAG] $FAILED worker(s) failed. Check logs in ${LOG_DIR}/"
        exit 1
    fi
    echo ""
done

echo "All chunking configurations completed."
