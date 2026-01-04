# ARK: Answer-centric Retriever via KG-driven Curriculum Learning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2511.16326-b31b1b.svg)](https://arxiv.org/abs/2511.16326)

**Authors**: Jiawei Zhou*, Hang Ding*, Haiyun Jiang

*: Equal Contribution with random order.

ARK is a retrieval-augmented generation (RAG) framework that trains retrievers using **answer-centric alignment** and **knowledge graph-driven curriculum learning**. Unlike traditional query-document matching, ARK focuses on whether retrieved chunks contain sufficient information to generate correct answers.



## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd ARK

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. API Credentials Setup

**IMPORTANT**: ARK uses a `.env` file to keep API keys secure.

```bash
# Copy the example .env file
cp .env.example .env

# Edit .env and add your API keys
nano .env  # or use your preferred editor
```

**Note**: The `.env` file is gitignored and will never be committed to version control.

### 3. Ollama Setup (Optional but Recommended)

For local embedding and LLM inference:

```bash
# Install Ollama: https://ollama.ai
ollama pull bge-m3:latest
ollama pull mistral:latest
```

### 4. Data and Model Setup

**Required Model Structure**:
```
model/
├── raw/
│   ├── bge-m3/          # BGE-M3 multi-vector embedding model
│   └── qwen3/           # Qwen3-Embedding-0.6B base model
└── finetuned/
    └── qwen3/           # Fine-tuned checkpoints (created during training)
```

**Base Models**:
- `model/raw/bge-m3/`: BGE-M3 from [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding)
  - Used for graph augmentation (similarity edge detection)
  - Supports dense, sparse, and ColBERT embeddings
- `model/raw/qwen3/`: Qwen3-Embedding-0.6B from [HuggingFace](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
  - Used as base model for fine-tuning
  - Can be cached from HuggingFace downloads

## Usage

ARK follows a 4-stage experiment pipeline. Run the scripts in order:

### 1. Compute Triple Alignment Scores

```bash
bash experiments/1-alignment/compute_alignment.sh
```

Computes forward, backward, and embedding alignment scores for (question, chunk, answer) triples.

### 2. Build Knowledge Graphs

```bash
bash experiments/2-kg/generate_kg.sh
```

Constructs knowledge graphs from documents with entity extraction and graph augmentation.

### 3. Train Data Generation

```bash
bash experiments/2-kg/generate_data.sh
```

Generates training data with positive chunks and KG-augmented hard negatives for curriculum learning stages.

### 4. Train Retriever

```bash
bash experiments/3-training/train_qwen.sh
```

Fine-tunes the retriever using curriculum learning with KG-augmented hard negatives.

### 4. Evaluate

**Evaluate base models:**
```bash
bash experiments/4-eval/base/run_qwen.sh      # Qwen3-Embedding
bash experiments/4-eval/base/run_bge.sh       # BGE-M3
bash experiments/4-eval/base/run_stella.sh    # Stella-v5
bash experiments/4-eval/base/run_jina.sh      # Jina-v3
```

**Evaluate fine-tuned model:**
```bash
bash experiments/4-eval/finetuned/run_qwen_finetuned.sh
```

**Evaluate ablation:**
```bash
bash experiments/4-eval/base/run_no_retrieval.sh  # Full context (no retrieval)
```

Results are saved to `experiments/results/` with F1 scores and win rates.


### Baseline Methods

See [`baselines/README.md`](baselines/README.md) for details on:
- **MemoRAG**: Memory-augmented retrieval
- **HippoRAG**: Hierarchical retrieval with personalized PageRank
- **LightRAG**: Lightweight efficient RAG
- **NodeRAG**: Node-based graph RAG

## Datasets

We evaluate on 10 datasets from LongBench and UltraDomain:

**LongBench** (5 datasets):
- NarrativeQA, Qasper, MuSiQue, 2WikiMQA, HotpotQA

**UltraDomain** (5 datasets):
- Biology, Fiction, Music, Technology, Philosophy

For training, we use two datasets (Finance and Legal domains) with the first 200 entries each to calibrate the retriever model.

## Performance

### Main Evaluation Results

Evaluation metrics show **F1-score / Win Rate (%)** across multiple datasets. The improvement rate (↑%) is calculated relative to our base model Qwen3-embedding.

#### F1 Scores

**LongBench Datasets:**

| Model | NarrativeQA | Qasper | MuSiQue | 2WikiMQA | HotpotQA |
|-------|-------------|--------|---------|----------|----------|
| **Full** | 12.95 | 22.79 | 6.74 | 20.13 | 26.87 |
| **Qwen3-embedding** | 19.58 | **23.90** | 14.19 | 21.24 | 35.27 |
| **BGE-M3** | 18.37 | 23.33 | **21.13** | 22.86 | 38.64 |
| **Stella-v5** | **20.90** | 23.39 | 17.08 | 22.13 | 35.45 |
| **Jina-emb-v3** | 19.39 | 20.70 | 20.58 | 19.34 | **39.17** |
| **GraphRAG** | 4.21 | 7.69 | 2.15 | 5.52 | 3.03 |
| **LightRAG** | 2.65 | 3.25 | 1.95 | 3.67 | 2.74 |
| **HippoRAG** | 11.51 | 21.90 | 13.09 | **30.96** | 28.71 |
| **MemoRAG** | 15.49 | 17.96 | 8.74 | 16.57 | 22.79 |
| **ARK (Ours)** | **21.57** | **24.04** | **20.60** | **23.41** | **42.35** |
| **↑% vs Base** | +10.2% | +0.6% | +45.2% | +10.2% | +20.1% |

**UltraDomain Datasets:**

| Model | Biology | Fiction | Music | Technology | Philosophy |
|-------|---------|---------|-------|------------|------------|
| **Full** | 27.47 | 25.75 | 25.50 | 22.68 | 23.05 |
| **Qwen3-embedding** | 32.99 | 29.41 | 34.90 | 38.03 | 34.04 |
| **BGE-M3** | 32.52 | 31.72 | **35.34** | 39.13 | 35.97 |
| **Stella-v5** | 33.85 | **32.41** | 35.02 | 35.16 | 34.09 |
| **Jina-emb-v3** | 32.88 | 29.00 | 33.74 | 38.74 | **36.81** |
| **GraphRAG** | 18.87 | 16.92 | 14.97 | 21.93 | 20.01 |
| **LightRAG** | 16.06 | 14.13 | 15.08 | 12.19 | 14.04 |
| **HippoRAG** | **36.13** | 29.23 | 32.94 | 27.15 | 29.06 |
| **MemoRAG** | 31.08 | 27.87 | 33.26 | **39.14** | 31.98 |
| **ARK (Ours)** | **36.19** | **32.59** | **38.03** | **40.16** | **37.86** |
| **↑% vs Base** | +9.7% | +10.8% | +9.0% | +5.6% | +11.2% |

## Citation

If you find ARK useful in your research, please cite:

```bibtex
@article{zhou2025ark,
  title={ARK: Answer-Centric Retriever Tuning via KG-augmented Curriculum Learning},
  author={Zhou, Jiawei and Ding, Hang and Jiang, Haiyun},
  journal={arXiv preprint arXiv:2511.16326},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **BGE-M3**: [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding)
- **Qwen Embedding**: [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
- **LLM Providers**: Gemini, DeepSeek, GPT model families
- **Baseline Methods**:
  - [HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG)
  - [LightRAG](https://github.com/HKUDS/LightRAG)
  - [MemoRAG](https://github.com/qhjqhj00/MemoRAG)
  - [GraphRAG](https://github.com/microsoft/graphrag)
  - [Nano-GraphRAG](https://github.com/gusye1234/nano-graphrag)
- **Datasets**:
  - [LongBench](https://github.com/THUDM/LongBench)
  - [UltraDomain](https://huggingface.co/datasets/MemoRAG/UltraDomain)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or issues, please open an issue on GitHub or contact the authors.
