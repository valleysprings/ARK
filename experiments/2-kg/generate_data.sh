#!/bin/bash
# ARK Data Generation Pipeline
# Generates subgraphs (both small and large), queries, and training data in one go
#
# Usage: bash generate_data.sh <dataset_type> <start_index> <end_index> [mode]
# Mode: all (default) | extract-only | queries-only | data-only

set -e

DATASET=${1:-"fin"}
START_IDX=${2:-0}
END_IDX=${3:-1}
MODE=${4:-"all"}  # all | extract-only | queries-only | data-only

echo "=================================================="
echo "ARK Data Generation Pipeline"
echo "=================================================="
echo "Dataset: $DATASET"
echo "Range: $START_IDX to $END_IDX"
echo "Mode: $MODE"
echo "=================================================="

# Set paths (now handles both small and large)
ALIGNMENT_DIR="./data/preprocessed/${DATASET}/alignment"

# Step 1: Extract Subgraphs (both small and large in one pass)
if [ "$MODE" == "all" ] || [ "$MODE" == "extract-only" ]; then
    echo ""
    echo "=================================================="
    echo "Step 1/3: Extracting Subgraphs (Small + Large)"
    echo "=================================================="

    python -m src.kg.generate_training_data extract-subgraphs \
        --dataset_type "$DATASET" \
        --start_index "$START_IDX" \
        --end_index "$END_IDX"

    echo "✅ Subgraph extraction completed!"
fi

# Step 2: Generate Queries (both small and large)
if [ "$MODE" == "all" ] || [ "$MODE" == "queries-only" ]; then
    echo ""
    echo "=================================================="
    echo "Step 2/3: Generating Queries (Small + Large)"
    echo "=================================================="

    python -m src.kg.generate_training_data generate-queries \
        --dataset_type "$DATASET" \
        --start_index "$START_IDX" \
        --end_index "$END_IDX"

    echo "✅ Query generation completed!"
fi

# Step 3: Generate Training Data (both small and large)
if [ "$MODE" == "all" ] || [ "$MODE" == "data-only" ]; then
    echo ""
    echo "=================================================="
    echo "Step 3/3: Generating Training Data (Small + Large)"
    echo "=================================================="

    # Generate training data for small
    echo "Generating training data for SMALL subgraphs..."
    python -m src.kg.generate_training_data generate-data \
        --alignment_dir "$ALIGNMENT_DIR" \
        --query_dir "./data/preprocessed/${DATASET}/queries_small" \
        --output_dir "./data/training/${DATASET}/preference_pairs_small" \
        --embedding_model "BAAI/bge-base-en-v1.5" \
        --device "cuda" \
        --top_n 3 \
        --top_m 10 \
        --start_index "$START_IDX" \
        --end_index "$END_IDX"

    # Generate training data for large
    echo "Generating training data for LARGE subgraphs..."
    python -m src.kg.generate_training_data generate-data \
        --alignment_dir "$ALIGNMENT_DIR" \
        --query_dir "./data/preprocessed/${DATASET}/queries_large" \
        --output_dir "./data/training/${DATASET}/preference_pairs_large" \
        --embedding_model "BAAI/bge-base-en-v1.5" \
        --device "cuda" \
        --top_n 3 \
        --top_m 10 \
        --start_index "$START_IDX" \
        --end_index "$END_IDX"

    echo "✅ Training data generation completed!"
fi

echo ""
echo "=================================================="
echo "✅ Pipeline Complete!"
echo "=================================================="

if [ "$MODE" == "all" ]; then
    echo "Generated files:"
    echo "  - Subgraphs Small: ./data/preprocessed/${DATASET}/subgraphs_answer_small/"
    echo "  - Subgraphs Large: ./data/preprocessed/${DATASET}/subgraphs_answer_large/"
    echo "  - Queries Small: ./data/preprocessed/${DATASET}/queries_small/"
    echo "  - Queries Large: ./data/preprocessed/${DATASET}/queries_large/"
    echo "  - Training Data Small: ./data/training/${DATASET}/preference_pairs_small/"
    echo "  - Training Data Large: ./data/training/${DATASET}/preference_pairs_large/"
fi
