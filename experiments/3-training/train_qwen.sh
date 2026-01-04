#!/bin/bash
# Train Qwen Embedding with triple alignment and curriculum learning
# Usage: bash train_qwen.sh <stage> <output_name> [epochs]

set -e

STAGE=${1:-"all"}  # all | stage1 | stage2 | stage3
OUTPUT_NAME=${2:-"qwen-embedding-finetuned"}
EPOCHS=${3:-10}

echo "=================================================="
echo "Training Qwen Embedding Retriever"
echo "Stage: $STAGE"
echo "Output Name: $OUTPUT_NAME"
echo "Epochs: $EPOCHS"
echo "=================================================="

# Activate virtual environment if exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

if [ "$STAGE" == "all" ]; then
    echo "Running all 3 stages of curriculum learning..."

    # Stage 1: Initial alignment
    echo "--- Stage 1: Initial Alignment ---"
    swift sft \
        --model_type qwen-embedding \
        --model_id_or_path "Qwen/Qwen3-Embedding-0.6B" \
        --dataset "./data/training/qwen_data/stage1" \
        --output_dir "./model/checkpoints/${OUTPUT_NAME}-stage1" \
        --num_train_epochs 3 \
        --per_device_train_batch_size 8 \
        --gradient_accumulation_steps 16 \
        --learning_rate 6e-6 \
        --warmup_steps 100 \
        --loss_type cosine_similarity \
        --task_type embedding

    # Stage 2: Coarse-grained alignment (large subgraph)
    echo "--- Stage 2: Coarse-Grained Alignment ---"
    swift sft \
        --model_type qwen-embedding \
        --model_id_or_path "./model/checkpoints/${OUTPUT_NAME}-stage1" \
        --dataset "./data/training/qwen_data/stage2" \
        --output_dir "./model/checkpoints/${OUTPUT_NAME}-stage2" \
        --num_train_epochs 4 \
        --per_device_train_batch_size 8 \
        --gradient_accumulation_steps 16 \
        --learning_rate 4e-6 \
        --warmup_steps 50 \
        --loss_type cosine_similarity \
        --task_type embedding

    # Stage 3: Fine-grained alignment (small subgraph)
    echo "--- Stage 3: Fine-Grained Alignment ---"
    swift sft \
        --model_type qwen-embedding \
        --model_id_or_path "./model/checkpoints/${OUTPUT_NAME}-stage2" \
        --dataset "./data/training/qwen_data/stage3" \
        --output_dir "./model/checkpoints/${OUTPUT_NAME}" \
        --num_train_epochs 3 \
        --per_device_train_batch_size 8 \
        --gradient_accumulation_steps 16 \
        --learning_rate 2e-6 \
        --warmup_steps 50 \
        --loss_type cosine_similarity \
        --task_type embedding

    echo "✅ All stages completed!"
else
    # Run specific stage
    python -m src.training.qwen_trainer \
        --stage "$STAGE" \
        --model_name_or_path "Qwen/Qwen3-Embedding-0.6B" \
        --train_data "./data/training/qwen_data/${STAGE}" \
        --output_dir "./model/checkpoints/${OUTPUT_NAME}-${STAGE}" \
        --num_train_epochs "$EPOCHS" \
        --per_device_train_batch_size 8 \
        --gradient_accumulation_steps 16 \
        --learning_rate 6e-6 \
        --alignment_forward_weight 1.0 \
        --alignment_backward_weight 0.3 \
        --alignment_parameter_weight 1.0

    echo "✅ Stage $STAGE completed!"
fi

echo "Model saved to: ./model/checkpoints/${OUTPUT_NAME}"
