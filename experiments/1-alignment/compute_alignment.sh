#!/bin/bash
# Compute triple alignment scores for training data
# Usage: bash compute_alignment.sh <dataset_type> <start_index> <end_index>

set -e

DATASET=${1:-"fin"}
START_IDX=${2:-0}
END_IDX=${3:-1}

echo "=================================================="
echo "Computing Triple Alignment Scores"
echo "Dataset: $DATASET"
echo "Range: $START_IDX to $END_IDX"
echo "=================================================="

# Activate virtual environment if exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Set up paths
INPUT_FILE="./data/raw/${DATASET}.jsonl"
OUTPUT_DIR="./data/preprocessed/${DATASET}/alignment"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Run alignment scoring
# Note: Each sample will be saved as alignment_score_{index}.jsonl in OUTPUT_DIR
python -m src.alignment.scorer \
    --input_jsonl "$INPUT_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --config "src/config/alignment.yaml" \
    --lm_model "./model/Mistral-7B" \
    --embedding_model "./model/bge-m3" \
    --top_k 1000 \
    --start_index "$START_IDX" \
    --end_index "$END_IDX"

echo "✅ Alignment scoring completed!"
echo "📁 Results saved to: $OUTPUT_DIR/"
