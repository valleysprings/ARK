# ARK Training Module - Structure Summary

## 完成的重构

### 1. 目录结构重组

```
src/training/
├── __init__.py              # 模块导入，暴露train()统一入口
├── trainer.py               # 基类 + train()调度器
├── README.md                # 简化的API文档
├── trainers/                # 具体trainer实现
│   ├── __init__.py
│   ├── bge_trainer.py      # BGE-M3训练器
│   └── qwen_trainer.py     # Qwen Embedding训练器
└── data/                    # 数据加载工具
    └── __init__.py

src/alignment/               # 三重对齐评分（独立模块）
├── __init__.py
├── scorer.py                # 对齐分数计算
└── preference.py            # 偏好学习
```

### 2. 配置文件统一管理

```
src/config/
├── training.yaml           # 训练配置（BGE和Qwen）
├── retrieval_model.yaml    # 模型推理配置
├── kg.yaml                 # KG构建配置
├── llm_inference.yaml      # LLM推理配置
└── experiments/            # 实验特定配置
    ├── exp_bge.yaml
    ├── exp_qwen.yaml
    ├── exp_qwen_base.yaml
    ├── exp_jina.yaml
    └── exp_stella.yaml
```

### 3. 训练脚本组织

```
experiments/
├── 1-alignment/            # Alignment计算
│   └── compute_alignment.sh # 计算三重对齐分数
├── 2-kg/                   # KG构建和处理
│   ├── build_kg.sh        # Stage 1: 构建KG
│   ├── augment_kg.sh      # Stage 2: 增强KG
│   ├── extract_subgraph.sh # Stage 3: 提取子图
│   └── generate_queries.sh # 生成查询（可选）
├── 3-training/             # 模型训练
│   ├── train_bge.sh        # 训练BGE-M3
│   └── train_qwen.sh       # 训练Qwen
└── 4-eval/                 # 评估
    └── run_evaluation.sh   # 运行评估
```

---

## 完整训练Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                   ARK Training Pipeline                         │
└─────────────────────────────────────────────────────────────────┘

Stage 1: KG Construction
┌──────────────────────────────────┐
│ experiments/2-kg/build_kg.sh     │
│   ↓                              │
│ src.kg.ops.construction          │
│   ↓                              │
│ data/preprocessed/{ds}/full_kg/  │
└──────────────────────────────────┘

Stage 2: KG Augmentation
┌──────────────────────────────────────────┐
│ experiments/2-kg/augment_kg.sh           │
│   ↓                                      │
│ src.kg.ops.augmentation                  │
│   ↓                                      │
│ data/preprocessed/{ds}/full_kg_augmented/│
└──────────────────────────────────────────┘

Stage 3: Subgraph Extraction (使用PPR)
┌────────────────────────────────────────────────┐
│ experiments/2-kg/extract_subgraph.sh           │
│   ↓                                            │
│ src.kg.ops.extraction                          │
│   ↓                                            │
│ ├─ subgraphs_question/ (大子图 - Stage 2训练) │
│ └─ subgraphs_answer/   (小子图 - Stage 3训练) │
└────────────────────────────────────────────────┘

Stage 4: Triple Alignment Scoring (三重对齐)
┌──────────────────────────────────────────────────┐
│ experiments/1-alignment/compute_alignment.sh     │
│   ↓                                              │
│ src.alignment.scorer                             │
│   ├─ Forward:  P(answer|chunk, question)  [LLM] │
│   ├─ Backward: P(question|chunk, answer)  [LLM] │
│   └─ Parameter: cosine(query, chunk)  [Embedding]│
│   ↓                                              │
│ data/alignment/scores/{ds}/                      │
└──────────────────────────────────────────────────┘

Stage 5: Model Training
┌──────────────────────────────────────────────────┐
│ Option A: BGE-M3                                 │
│ experiments/3-training/train_bge.sh              │
│   ↓                                              │
│ src.training.trainers.bge_trainer                │
│   ├─ Unified fine-tuning (dense+sparse+colbert) │
│   ├─ Knowledge distillation                     │
│   └─ Lambda scheduling                          │
│   ↓                                              │
│ model/checkpoints/bge-m3-finetuned/              │
│                                                  │
│ Option B: Qwen Embedding (Curriculum Learning)   │
│ experiments/3-training/train_qwen.sh             │
│   ↓                                              │
│ src.training.trainers.qwen_trainer               │
│   ├─ Stage 1: 基础QA (3 epochs)                 │
│   ├─ Stage 2: 粗粒度对齐 + 大子图 (4 epochs)    │
│   └─ Stage 3: 细粒度对齐 + 小子图 (3 epochs)    │
│   ↓                                              │
│ model/checkpoints/qwen-finetuned/                │
└──────────────────────────────────────────────────┘
```

---

## 调用关系

### KG构建阶段

```python
# build_kg.sh 调用
python -m src.kg.ops.construction \
    --dataset_name hotpotqa \
    --chunk_size 512 \
    --chunk_overlap 12 \
    --llm_provider gemini

# augment_kg.sh 调用
python -m src.kg.ops.augmentation \
    --dataset_type hotpotqa \
    --kg_dir ./data/preprocessed/hotpotqa/full_kg

# extract_subgraph.sh 调用
python -m src.kg.ops.extraction \
    --dataset_type hotpotqa \
    --mode answer \
    --ppr_alpha 0.85
```

### 训练数据准备

```python
# compute_alignment.sh 调用
# 输出：每个样本一个文件 (alignment_score_{index}.jsonl)
python -m src.alignment.scorer \
    --input_jsonl ./data/raw/hotpotqa.jsonl \
    --output_dir ./data/alignment/scores/hotpotqa \
    --config ./config.yaml \
    --lm_model ./model/Mistral-7B \
    --embedding_model bge-m3:latest \
    --top_k 1000 \
    --start_index 0 \
    --end_index 100
```

### 模型训练

```python
# train_bge.sh 调用
python -m src.training.trainers.bge_trainer \
    --model_name_or_path ./model/bge-m3 \
    --train_data ./data/training/bge_traindata \
    --unified_finetuning \
    --use_self_distill

# train_qwen.sh 调用 (通过MS-SWIFT)
swift sft \
    --model_type qwen-embedding \
    --dataset ./data/training/qwen_data/stage1 \
    --loss_type cosine_similarity \
    --task_type embedding

# 或者直接调用
python -m src.training.trainers.qwen_trainer \
    --stage stage1 \
    --train_data ./data/training/qwen_data/stage1
```

### 统一训练接口

```python
# 使用统一入口
python -m src.training.trainer \
    --model_type qwen \
    --config src/config/training.yaml \
    --output_dir ./model/checkpoints/qwen \
    --dataset_paths ./data/training/qwen_data/stage1

# Python API
from src.training import train

train(
    model_type='qwen',
    config='src/config/training.yaml',
    output_dir='./model/checkpoints/qwen'
)
```

---

## 关键组件

### 1. BaseRetrieverTrainer (trainer.py)
- 所有trainer的抽象基类
- 提供统一的训练接口
- 处理分布式训练、检查点管理

### 2. train() 调度器 (trainer.py)
- 统一训练入口点
- 根据model_type分派到具体trainer
- 支持从YAML配置加载

### 3. Triple Alignment Scorer (alignment/scorer.py)
- 计算三重对齐分数
- 使用LLM prompting (forward/backward)
- 使用embedding similarity (parameter)
- 输出用于选择正样本

### 4. KG Operations (src/kg/ops/)
- construction.py: 实体关系提取，构建KG
- augmentation.py: 增强实体描述
- extraction.py: PPR子图提取

---

## 数据流

```
Raw Data (.jsonl)
    ↓
[KG Construction] → Full KG (NetworkX)
    ↓
[KG Augmentation] → Augmented KG
    ↓
[Subgraph Extraction] → Question/Answer Subgraphs
    ↓
[Alignment Scoring] → Chunk-Question-Answer Triplets with Scores
    ↓
[Model Training] → Fine-tuned Retriever
```

---

## 配置继承关系

```
src/config/training.yaml (默认配置)
    ↓
src/config/experiments/exp_qwen.yaml (实验配置覆盖)
    ↓
命令行参数 (最高优先级)
```

---

## 文档

- `docs/TRAINING_PIPELINE.md`: 完整训练流程文档
- `src/training/README.md`: Training模块API文档
- `src/kg/README.md`: KG操作文档
- `experiments/README.md`: 实验管理文档

---

## 下一步

1. 准备训练数据：运行Stage 1-4
2. 配置训练参数：修改 `src/config/training.yaml`
3. 启动训练：使用 `train_qwen.sh` 或 `train_bge.sh`
4. 监控训练：查看 `experiments/logs/`
5. 评估模型：使用 `experiments/eval/run_evaluation.sh`
