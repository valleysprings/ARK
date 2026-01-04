#!/bin/bash
# Unified KG Generation Pipeline
# Replaces: build_kg.sh, augment_kg.sh, extract_subgraph.sh
#
# This script orchestrates the complete KG pipeline:
#   1. Build KG from documents
#   2. Augment with similarity edges
#   3. Extract subgraphs (optional, use-case specific)
#
# Usage: bash generate_kg.sh <dataset> <start_index> <end_index>
# Example: bash generate_kg.sh hotpotqa 0 10

set -e

DATASET=${1:-"fin"}
START_IDX=${2:-0}
END_IDX=${3:-1}

echo "=================================================="
echo "KG Generation Pipeline"
echo "Dataset: $DATASET"
echo "Range: $START_IDX to $END_IDX"
echo "=================================================="

# Activate virtual environment if exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Run KG builder as module
python -m src.kg.graph_builder \
    --dataset "$DATASET" \
    --start_index "$START_IDX" \
    --end_index "$END_IDX" \
    --chunk_size 512 \
    --chunk_overlap 12 \
    --similarity_threshold 0.8 \
    --config src/config/graph_builder.yaml

echo ""
echo "=================================================="
echo "✅ KG Generation Pipeline Completed!"
echo "=================================================="
echo ""
echo "Output structure:"
echo "  data/preprocessed/${DATASET}/"
echo "    ├── full_kg/              # Original KGs"
echo "    ├── full_kg_augmented/    # Augmented KGs with similarity edges"
echo "    └── all_entities_data/    # Entity metadata"
echo ""
echo "To skip specific steps, use flags:"
echo "  --skip_build      # Skip KG construction"
echo "  --skip_augment    # Skip augmentation"
echo "  --skip_extract    # Skip subgraph extraction"
