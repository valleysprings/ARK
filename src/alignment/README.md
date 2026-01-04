# Alignment Module

Triple alignment scoring for Answer-Centric Retrieval.

## Overview

This module implements a triple alignment scoring mechanism that evaluates the quality of chunk-question-answer alignments using three complementary metrics:

1. **Forward alignment**: P(answer|chunk, question) - How well a chunk supports generating the answer
2. **Backward alignment**: P(question|chunk, answer) - How well a chunk supports reconstructing the question
3. **Parameter alignment**: cosine similarity between query and chunk embeddings

## Quick Start

### 1. Compute Alignment Scores

```bash
# Process samples 0-100 from hotpotqa dataset
bash experiments/1-alignment/compute_alignment.sh hotpotqa 0 100
```

**Output**: Each sample is saved as a separate file in `./data/alignment/scores/hotpotqa/`:
- `alignment_score_0.jsonl`
- `alignment_score_1.jsonl`
- `alignment_score_2.jsonl`
- ...
- `alignment_score_99.jsonl`

### 2. Generate Preference Pairs (Optional)

If you need to generate preference pairs for DPO/RLHF training:

```bash
python -m src.alignment.preference \
    --scored_data ./data/alignment/scores/hotpotqa \
    --new_queries ./data/preprocessed/hotpotqa/augmented_queries.json \
    --output ./data/training/preference_pairs.jsonl \
    --embedding_model BAAI/bge-base-en-v1.5
```

## File Format

### Input Format

Input JSONL file (`./data/raw/{dataset}.jsonl`) should contain:

```json
{
  "input": "What is the capital of France?",
  "answers": ["Paris"],
  "context": "France is a country in Europe. Paris is the capital and largest city of France..."
}
```

### Output Format

Each alignment score file (`alignment_score_{index}.jsonl`) contains:

```json
{
  "input": "What is the capital of France?",
  "answers": ["Paris"],
  "chunk_list": ["France is a country...", "Paris is the capital..."],
  "score_list": [1.85, 2.34],
  "forward_list": [0.65, 0.89],
  "reverse_list": [0.45, 0.52],
  "qwen_list": [0.75, 0.93]
}
```

Where:
- `chunk_list`: Top-k chunks sorted by alignment score (descending)
- `score_list`: Combined alignment scores (weighted sum of forward + backward + parameter)
- `forward_list`: Forward alignment scores (normalized)
- `reverse_list`: Backward alignment scores (normalized)
- `qwen_list`: Parameter alignment scores (embedding similarity)

## Configuration

Key parameters in `config.yaml`:

```yaml
training:
  common:
    device: "cuda"
  qwen:
    alignment:
      forward_weight: 1.0      # Weight for forward alignment
      backward_weight: 0.3     # Weight for backward alignment
      parameter_weight: 1.0    # Weight for parameter alignment
      batch_size: 64
      normalize_scores: true
kg:
  chunk_size: 512              # Size of text chunks in words
  chunk_overlap: 12            # Overlap between consecutive chunks
```

## Command-Line Arguments

### scorer.py

```bash
python -m src.alignment.scorer \
    --input_jsonl <path>       # Input JSONL file with questions/answers/contexts
    --output_dir <path>        # Output directory for score files
    --config <path>            # Path to config.yaml (default: ./config.yaml)
    --lm_model <path>          # Path to language model for forward/backward alignment
    --embedding_model <path>   # Path to embedding model for parameter alignment
    --top_k <int>              # Number of top chunks to keep (default: 1000)
    --start_index <int>        # Start index for processing (default: 0)
    --end_index <int>          # End index for processing (optional)
```

### preference.py

```bash
python -m src.alignment.preference \
    --scored_data <path>       # Path to scored data directory or JSONL file
    --new_queries <path>       # Path to augmented queries JSON file
    --output <path>            # Path to output preference pairs JSONL
    --embedding_model <path>   # Path to embedding model
    --config <path>            # Path to config.yaml (optional)
    --top_n <int>              # Number of top chunks for chosen response (default: 3)
    --top_m <int>              # Number of similar chunks to retrieve (default: 10)
    --max_items <int>          # Maximum number of items to process (optional)
    --validate                 # Validate output after generation
```

## Advanced Usage

### Parallel Processing

Process different ranges in parallel:

```bash
# Terminal 1
bash experiments/1-alignment/compute_alignment.sh hotpotqa 0 100

# Terminal 2
bash experiments/1-alignment/compute_alignment.sh hotpotqa 100 200

# Terminal 3
bash experiments/1-alignment/compute_alignment.sh hotpotqa 200 300
```

### Loading Scored Data

The `preference.py` module automatically detects whether `--scored_data` is a directory or file:

```python
from src.alignment.preference import PreferenceGenerator

generator = PreferenceGenerator(embedding_model_path="BAAI/bge-base-en-v1.5")

# Load from directory (multiple files)
generator.generate_from_scored_data(
    scored_data_path="./data/alignment/scores/hotpotqa",  # Directory
    new_queries_path="./queries.json",
    output_path="./output.jsonl"
)

# Load from single file (backward compatibility)
generator.generate_from_scored_data(
    scored_data_path="./scores.jsonl",  # Single file
    new_queries_path="./queries.json",
    output_path="./output.jsonl"
)
```

## Integration with KG Pipeline

The alignment scores are used in the KG training data generation pipeline:

```bash
# 1. Generate alignment scores
bash experiments/1-alignment/compute_alignment.sh hotpotqa 0 1000

# 2. Use in KG data generation
bash experiments/2-kg/generate_data.sh hotpotqa large 0 1000
```

## Performance Tips

1. **Batch Size**: Adjust `batch_size` in config based on GPU memory
2. **Top-k**: Use smaller `--top_k` values to reduce memory usage
3. **Chunking**: Tune `chunk_size` and `chunk_overlap` based on your dataset
4. **Model Selection**:
   - Use smaller LLMs (e.g., Mistral-7B) for faster processing
   - Use efficient embedding models (e.g., BGE-M3) for better quality

## Troubleshooting

### Out of Memory

Reduce batch size or top_k:
```bash
# In config.yaml
alignment:
  batch_size: 32  # Reduce from 64
```

### Too Many Files

Use larger index ranges:
```bash
# Process 1000 samples at once instead of 100
bash experiments/1-alignment/compute_alignment.sh hotpotqa 0 1000
```

### Reading Multiple Files

The module automatically handles both directory and file inputs, so you can pass either to `--scored_data`.
