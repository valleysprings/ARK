#!/bin/bash
# Train Qwen Embedding on multiple datasets with the same chunking method
# Usage: bash train_qwen_multi.sh <datasets> <chunk_method> <chunk_size> <overlap> [stage] [ngpu]
#
# Arguments:
#   datasets: Comma-separated dataset names (e.g., "fin,legal")
#   chunk_method: Chunking method (e.g., sentence, token, or "all" for all chunking configs)
#   chunk_size: Chunk size (e.g., 1, 5) - ignored when chunk_method=all
#   overlap: Overlap size (e.g., 0, 1) - ignored when chunk_method=all
#   stage: all | stage1 | stage2 | stage3 (default: all)
#   ngpu: Number of GPUs (default: 8)

set +e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${PROJECT_ROOT}"

# Parse arguments
DATASETS=${1:-"fin,legal"}
CHUNK_METHOD=${2:-"token"}
CHUNK_SIZE=${3:-"512"}
OVERLAP=${4:-"12"}
STAGE=${5:-"all"}
NGPU=${6:-8}

TRAIN_CONFIG="src/config/training.yaml"

# Read training hyperparameters from config
read MODEL_PATH BATCH_SIZE LOSS_TYPE TASK_TYPE OVERWRITE \
    S1_EPOCHS S1_LR S1_WARMUP \
    S2_EPOCHS S2_LR S2_WARMUP \
    S3_EPOCHS S3_LR S3_WARMUP <<< $(python3 -c "
import yaml
with open('${TRAIN_CONFIG}') as f:
    c = yaml.safe_load(f)
cur = c['curriculum']
print(c['model_name_or_path'], c['per_device_train_batch_size'], 
      c['loss_type'], c['task_type'], str(c['overwrite_output_dir']).lower(),
      cur['stage1']['epochs'], cur['stage1']['learning_rate'], cur['stage1']['warmup_steps'],
      cur['stage2']['epochs'], cur['stage2']['learning_rate'], cur['stage2']['warmup_steps'],
      cur['stage3']['epochs'], cur['stage3']['learning_rate'], cur['stage3']['warmup_steps'])
")

if [ "$CHUNK_METHOD" == "all" ]; then
    OUTPUT_NAME="multi_all"
else
    OUTPUT_NAME="multi_${CHUNK_METHOD}_${CHUNK_SIZE}_o_${OVERLAP}"
fi

export NPROC_PER_NODE=$NGPU

# Set CUDA_HOME if not already set (needed by deepspeed)
if [ -z "$CUDA_HOME" ]; then
    if [ -d "/usr/local/cuda" ]; then
        export CUDA_HOME=/usr/local/cuda
    elif [ -f "${CONDA_PREFIX}/bin/nvcc" ]; then
        export CUDA_HOME=$CONDA_PREFIX
    fi
fi

echo "=================================================="
echo "Training Qwen Embedding Retriever (Multiple Datasets)"
echo "Datasets: $DATASETS"
echo "Chunking: ${CHUNK_METHOD}_${CHUNK_SIZE}_o_${OVERLAP}"
echo "Stage: $STAGE"
echo "Output Name: $OUTPUT_NAME"
echo "GPUs: $NGPU"
echo "=================================================="

# Activate virtual environment if exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Build dataset paths from comma-separated list
IFS=',' read -ra DATASET_ARRAY <<< "$DATASETS"

DATASET_PATH_STAGE1=""
DATASET_PATH_STAGE2=""
DATASET_PATH_STAGE3=""

for ds in "${DATASET_ARRAY[@]}"; do
    ds=$(echo "$ds" | xargs)  # trim whitespace
    BASE="./data/preprocessed/alignment/mistral_qwen/${ds}"

    if [ "$CHUNK_METHOD" == "all" ]; then
        # Auto-discover all chunking configs
        CHUNK_DIRS=$(find "$BASE" -mindepth 2 -maxdepth 2 -type d | sort)
    else
        CHUNK_DIRS="${BASE}/${CHUNK_METHOD}/${CHUNK_METHOD}_${CHUNK_SIZE}_o_${OVERLAP}"
    fi

    for DATA_DIR in $CHUNK_DIRS; do
        if [ -n "$DATASET_PATH_STAGE1" ]; then
            DATASET_PATH_STAGE1="${DATASET_PATH_STAGE1} "
            DATASET_PATH_STAGE2="${DATASET_PATH_STAGE2} "
            DATASET_PATH_STAGE3="${DATASET_PATH_STAGE3} "
        fi
        DATASET_PATH_STAGE1="${DATASET_PATH_STAGE1}${DATA_DIR}/training/stage1.jsonl"
        DATASET_PATH_STAGE2="${DATASET_PATH_STAGE2}${DATA_DIR}/training/stage2.jsonl"
        DATASET_PATH_STAGE3="${DATASET_PATH_STAGE3}${DATA_DIR}/training/stage3.jsonl"
    done
done

echo "Using training data:"
echo "  Stage 1: $DATASET_PATH_STAGE1"
echo "  Stage 2: $DATASET_PATH_STAGE2"
echo "  Stage 3: $DATASET_PATH_STAGE3"
echo ""

# Verify data exists
for f in $DATASET_PATH_STAGE1 $DATASET_PATH_STAGE2 $DATASET_PATH_STAGE3; do
    if [ ! -f "$f" ]; then
        echo "ERROR: Data file not found: $f"
        exit 1
    fi
done

if [ "$STAGE" == "all" ] || [ "$STAGE" == "stage1" ]; then
    echo "=================================================="
    echo "STAGE 1: Initial Contrastive Learning"
    echo "=================================================="
    swift sft \
        --model_type qwen3_emb \
        --model "$MODEL_PATH" \
        --train_type full \
        --add_version false \
        --dataset $DATASET_PATH_STAGE1 \
        --output_dir "./model/checkpoints/${OUTPUT_NAME}-stage1" \
        --num_train_epochs "$S1_EPOCHS" \
        --per_device_train_batch_size "$BATCH_SIZE" \
        --learning_rate "$S1_LR" \
        --warmup_steps "$S1_WARMUP" \
        --loss_type "$LOSS_TYPE" \
        --task_type "$TASK_TYPE" \
        --overwrite_output_dir "$OVERWRITE"
    echo "✅ Stage 1 completed!"
    echo ""
fi

if [ "$STAGE" == "all" ] || [ "$STAGE" == "stage2" ]; then
    echo "=================================================="
    echo "STAGE 2: Coarse-Grained Contrastive Learning"
    echo "=================================================="

    if [ "$STAGE" == "stage2" ]; then
        INPUT_MODEL="$MODEL_PATH"
    else
        STAGE1_DIR="./model/checkpoints/${OUTPUT_NAME}-stage1"
        INPUT_MODEL=$(find "$STAGE1_DIR" -maxdepth 1 -type d -name "checkpoint-*" | sort -V | tail -1)
        if [ -z "$INPUT_MODEL" ]; then
            INPUT_MODEL="$STAGE1_DIR"
        fi
    fi

    swift sft \
        --model_type qwen3_emb \
        --model "$INPUT_MODEL" \
        --train_type full \
        --add_version false \
        --dataset $DATASET_PATH_STAGE2 \
        --output_dir "./model/checkpoints/${OUTPUT_NAME}-stage2" \
        --num_train_epochs "$S2_EPOCHS" \
        --per_device_train_batch_size "$BATCH_SIZE" \
        --learning_rate "$S2_LR" \
        --warmup_steps "$S2_WARMUP" \
        --loss_type "$LOSS_TYPE" \
        --task_type "$TASK_TYPE" \
        --overwrite_output_dir "$OVERWRITE"
    echo "✅ Stage 2 completed!"
    echo ""
fi

if [ "$STAGE" == "all" ] || [ "$STAGE" == "stage3" ]; then
    echo "=================================================="
    echo "STAGE 3: Fine-Grained Contrastive Learning"
    echo "=================================================="

    if [ "$STAGE" == "stage3" ]; then
        INPUT_MODEL="$MODEL_PATH"
    else
        STAGE2_DIR="./model/checkpoints/${OUTPUT_NAME}-stage2"
        INPUT_MODEL=$(find "$STAGE2_DIR" -maxdepth 1 -type d -name "checkpoint-*" | sort -V | tail -1)
        if [ -z "$INPUT_MODEL" ]; then
            INPUT_MODEL="$STAGE2_DIR"
        fi
    fi

    swift sft \
        --model_type qwen3_emb \
        --model "$INPUT_MODEL" \
        --train_type full \
        --add_version false \
        --dataset $DATASET_PATH_STAGE3 \
        --output_dir "./model/checkpoints/${OUTPUT_NAME}" \
        --num_train_epochs "$S3_EPOCHS" \
        --per_device_train_batch_size "$BATCH_SIZE" \
        --learning_rate "$S3_LR" \
        --warmup_steps "$S3_WARMUP" \
        --loss_type "$LOSS_TYPE" \
        --task_type "$TASK_TYPE" \
        --overwrite_output_dir "$OVERWRITE"
    echo "✅ Stage 3 completed!"
    echo ""
fi

echo "=================================================="
echo "TRAINING COMPLETE"
echo "=================================================="
if [ "$STAGE" == "all" ]; then
    echo "Final model saved to: ./model/checkpoints/${OUTPUT_NAME}"
else
    echo "Model saved to: ./model/checkpoints/${OUTPUT_NAME}-${STAGE}"
fi
