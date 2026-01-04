# ARK Evaluation Scripts

Evaluation scripts are organized by model type for easier management and comparison.

## Directory Structure

```
experiments/eval/
├── base/                    # Base (pretrained) model evaluations
│   ├── run_qwen.sh         # Qwen2.5-Embedding
│   ├── run_stella.sh       # Stella Embedding
│   ├── run_bge.sh          # BGE-M3
│   ├── run_jina.sh         # Jina Embedding
│   └── run_no_retrieval.sh # No retrieval baseline (LLM only)
└── finetuned/              # Finetuned model evaluations
    └── run_qwen_finetuned.sh # Example: Qwen finetuned
```

## Usage

### Base Models

Each base model has its own script. All scripts accept the same arguments:

```bash
bash experiments/eval/base/run_<model>.sh [dataset] [start_idx] [end_idx]
```

**Examples:**

```bash
# Qwen2.5-Embedding
bash experiments/eval/base/run_qwen.sh hotpotqa 0 100

# Stella Embedding
bash experiments/eval/base/run_stella.sh hotpotqa 0 100

# BGE-M3
bash experiments/eval/base/run_bge.sh hotpotqa 0 100

# Jina Embedding
bash experiments/eval/base/run_jina.sh hotpotqa 0 100

# No retrieval baseline (LLM only)
bash experiments/eval/base/run_no_retrieval.sh hotpotqa 0 100
```

### Finetuned Models

Finetuned models require the checkpoint path:

```bash
bash experiments/eval/finetuned/run_qwen_finetuned.sh \
    [dataset] \
    [checkpoint_path] \
    [start_idx] \
    [end_idx]
```

**Example:**

```bash
bash experiments/eval/finetuned/run_qwen_finetuned.sh \
    2wikimqa \
    model/finetuned/qwen3/2wikimqa/nolabel/chunk64_32/checkpoint-70 \
    0 100
```

## Output

Results are saved as JSONL files:
- **Base models**: `experiments/results/base/`
- **Finetuned models**: `experiments/results/finetuned/`

Output filename format: `{dataset}_{retriever}.jsonl`

Each result entry contains:
- `question`: The input question
- `answer`: Ground truth answer
- `prediction`: Model's prediction
- `f1_score`: F1 score for this example
- `num_chunks_used`: Number of chunks retrieved
- `chunk_scores`: Similarity scores of retrieved chunks

## Batch Evaluation

### Run All Base Models

```bash
# Evaluate all base models on a dataset
for script in experiments/eval/base/*.sh; do
    bash "$script" hotpotqa 0 100
done
```

### Compare Results

```bash
# Compare results from all base models
python src/analysis/compare_results.py \
    experiments/results/base/hotpotqa_*.jsonl
```

## Dataset Examples

### Available Datasets

- **HotpotQA** - Multi-hop question answering
- **2WikiMQA** - Multi-hop QA with Wikipedia
- **Musique** - Multi-hop QA with diverse reasoning
- **NarrativeQA** - Long-form question answering

### Examples by Dataset

```bash
# HotpotQA (Multi-hop QA)
bash experiments/eval/base/run_qwen.sh hotpotqa 0 500

# NarrativeQA (Long-form QA)
bash experiments/eval/base/run_bge.sh narrativeqa 0 500

# 2WikiMQA (Multi-hop QA)
bash experiments/eval/base/run_jina.sh 2wikimqa 0 500

# Musique (Multi-hop QA)
bash experiments/eval/base/run_stella.sh musique 0 500
```

## Configuration

All scripts use the following default configs:
- **LLM Config**: `src/config/llm_inference.yaml`
- **Retrieval Config**: `src/config/retrieval_model.yaml`
- **Device**: `cuda:0`

To modify configs, edit the variables at the top of each script.

## Requirements

- Python 3.8+
- Required packages installed (see `requirements.txt`)
- Dataset files in `data/raw/` directory
- Model checkpoints or access to model repositories
- CUDA-capable GPU (recommended)
