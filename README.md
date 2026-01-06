# ARK: Answer-centric Retriever via KG-driven Curriculum Learning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2511.16326-b31b1b.svg)](https://arxiv.org/abs/2511.16326)

**Authors**: Jiawei Zhou*, Hang Ding*, Haiyun Jiang (*Equal Contribution)

## Overview

**Query Construction & Contrastive Finetuning**: ARK constructs a knowledge graph from documents, extracts query-based subgraphs, and generates augmented queries to mine hard negative chunks for contrastive learning.

![ARK Framework](asset/Framework.png)

**Answer-Centric Alignment & Curriculum Learning**: We rank chunks by forward/backward alignment scores (whether a chunk helps generate the correct answer), then progressively train the retriever from easy to hard negatives across three curriculum stages.

![ARK Training Pipeline](asset/FT.png)

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env  # Setup API keys
```

## Pipeline

```bash
# 1. Alignment
bash experiments/1-alignment/compute_alignment.sh <dataset> <start> <end> <gpu>

# 2. Knowledge Graph
bash experiments/2-kg/generate_kg.sh <dataset> <start> <end> <gpu>

# 3. Training Data
bash experiments/3-training/generate_data.sh <dataset> <start> <end> <stage> <gpu>

# 4. Train
bash experiments/3-training/train_qwen_single.sh <dataset> <stage> [output] [gpu]
bash experiments/3-training/train_qwen_multi.sh <datasets> <stage> [output] [gpu]

# 5. Evaluate
bash experiments/4-eval/finetuned/run_qwen.sh <eval_data> <train_data> <start> <end> <gpu>
```

## Config

Configuration files in `src/config/`:
- `training.yaml` - Training params (num_positives, num_negatives, retrieval_top_k)
- `alignment.yaml` - Alignment score computation
- `graph_builder.yaml` - Knowledge graph construction
- `query_generation.yaml` - Synthetic query generation
- `llm_api.yaml` - LLM API settings
- `llm_inference.yaml` - Inference settings
- `retrieval_model.yaml` - Retrieval model settings

## Results

**LongBench (F1 Score):**

| Model | NarrativeQA | Qasper | MuSiQue | 2WikiMQA | HotpotQA |
|-------|-------------|--------|---------|----------|----------|
| Qwen3-embedding | 19.58 | 23.90 | 14.19 | 21.24 | 35.27 |
| BGE-M3 | 18.37 | 23.33 | 21.13 | 22.86 | 38.64 |
| Stella-v5 | 20.90 | 23.39 | 17.08 | 22.13 | 35.45 |
| HippoRAG | 11.51 | 21.90 | 13.09 | 30.96 | 28.71 |
| **ARK (Ours)** | **21.57** | **24.04** | **20.60** | **23.41** | **42.35** |

**UltraDomain (F1 Score):**

| Model | Biology | Fiction | Music | Technology | Philosophy |
|-------|---------|---------|-------|------------|------------|
| Qwen3-embedding | 32.99 | 29.41 | 34.90 | 38.03 | 34.04 |
| BGE-M3 | 32.52 | 31.72 | 35.34 | 39.13 | 35.97 |
| Stella-v5 | 33.85 | 32.41 | 35.02 | 35.16 | 34.09 |
| HippoRAG | 36.13 | 29.23 | 32.94 | 27.15 | 29.06 |
| **ARK (Ours)** | **36.19** | **32.59** | **38.03** | **40.16** | **37.86** |

## Citation

```bibtex
@article{zhou2025ark,
  title={ARK: Answer-Centric Retriever Tuning via KG-augmented Curriculum Learning},
  author={Zhou, Jiawei and Ding, Hang and Jiang, Haiyun},
  journal={arXiv preprint arXiv:2511.16326},
  year={2025}
}
```

## Acknowledgments

**Models** (Base retriever and training framework):
- [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
- [MS-SWIFT](https://github.com/modelscope/ms-swift)
- LLM Providers: Gemini, DeepSeek, GPT

**Datasets** (Evaluation benchmarks):
- [LongBench](https://github.com/THUDM/LongBench)
- [UltraDomain](https://huggingface.co/datasets/MemoRAG/UltraDomain)

**Baselines** (Comparison methods):
- [HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG)
- [LightRAG](https://github.com/HKUDS/LightRAG)
- [MemoRAG](https://github.com/qhjqhj00/MemoRAG)
- [GraphRAG](https://github.com/microsoft/graphrag)
- [Nano-GraphRAG](https://github.com/gusye1234/nano-graphrag)

## License

MIT License
