#!/bin/bash
# Train Qwen Embedding on a single dataset
# Usage: bash train_qwen_single.sh <dataset> <stage> [output_name] [gpu]
#
# Arguments:
#   dataset: Dataset name (e.g., fin, hotpotqa, biology)
#   stage: all | stage1 | stage2 | stage3
#   output_name: Name for the output model (default: qwen-embedding-{dataset})
#   gpu: GPU device ID (default: 0)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${PROJECT_ROOT}"

# Parse arguments
DATASET=${1:-"legal"}
STAGE=${2:-"all"}
OUTPUT_NAME=${3:-"${DATASET}"}
GPU=${4:-"3"}

export CUDA_VISIBLE_DEVICES=$GPU

echo "=================================================="
echo "Training Qwen Embedding Retriever (Single Dataset)"
echo "Dataset: $DATASET"
echo "Stage: $STAGE"
echo "Output Name: $OUTPUT_NAME"
echo "GPU: $GPU"
echo "=================================================="

# Activate virtual environment if exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Training data paths
DATASET_PATH_STAGE1="./data/training/stage1/${DATASET}_train.jsonl"
DATASET_PATH_STAGE2="./data/training/stage2/${DATASET}_train.jsonl"
DATASET_PATH_STAGE3="./data/training/stage3/${DATASET}_train.jsonl"

if [ "$STAGE" == "all" ] || [ "$STAGE" == "stage1" ]; then
    echo "=================================================="
    echo "STAGE 1: Initial Chunk Alignment"
    echo "=================================================="
    swift sft \
        --model_type qwen3_emb \
        --model "./model/raw/qwen3" \
        --sft_type full \
        --dataset "$DATASET_PATH_STAGE1" \
        --output_dir "./model/checkpoints/${OUTPUT_NAME}-stage1" \
        --num_train_epochs 3 \
        --per_device_train_batch_size 2 \
        --gradient_accumulation_steps 16 \
        --learning_rate 6e-6 \
        --warmup_steps 100 \
        --loss_type infonce \
        --task_type embedding \
        --overwrite_output_dir true
    echo "✅ Stage 1 completed!"
    echo ""
fi

if [ "$STAGE" == "all" ] || [ "$STAGE" == "stage2" ]; then
    echo "=================================================="
    echo "STAGE 2: Coarse-Grained Contrastive Learning"
    echo "=================================================="

    # Determine input model path
    if [ "$STAGE" == "stage2" ]; then
        INPUT_MODEL="./model/raw/qwen3"
    else
        INPUT_MODEL="./model/checkpoints/${OUTPUT_NAME}-stage1"
    fi

    swift sft \
        --model_type qwen3_emb \
        --model "$INPUT_MODEL" \
        --sft_type full \
        --dataset "$DATASET_PATH_STAGE2" \
        --output_dir "./model/checkpoints/${OUTPUT_NAME}-stage2" \
        --num_train_epochs 4 \
        --per_device_train_batch_size 2 \
        --gradient_accumulation_steps 16 \
        --learning_rate 4e-6 \
        --warmup_steps 50 \
        --loss_type infonce \
        --task_type embedding \
        --overwrite_output_dir true
    echo "✅ Stage 2 completed!"
    echo ""
fi

if [ "$STAGE" == "all" ] || [ "$STAGE" == "stage3" ]; then
    echo "=================================================="
    echo "STAGE 3: Fine-Grained Contrastive Learning"
    echo "=================================================="

    # Determine input model path
    if [ "$STAGE" == "stage3" ]; then
        INPUT_MODEL="./model/raw/qwen3"
    else
        INPUT_MODEL="./model/checkpoints/${OUTPUT_NAME}-stage2"
    fi

    swift sft \
        --model_type qwen3_emb \
        --model "$INPUT_MODEL" \
        --sft_type full \
        --dataset "$DATASET_PATH_STAGE3" \
        --output_dir "./model/checkpoints/${OUTPUT_NAME}" \
        --num_train_epochs 3 \
        --per_device_train_batch_size 2 \
        --gradient_accumulation_steps 16 \
        --learning_rate 2e-6 \
        --warmup_steps 50 \
        --loss_type infonce \
        --task_type embedding \
        --overwrite_output_dir true
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
