# ARK Training Module

Training module for fine-tuning retrieval models (BGE-M3 and Qwen Embedding).

## Structure

```
src/training/
├── __init__.py           # Module exports
├── trainer.py            # Base trainer class
├── bge_trainer.py        # BGE-M3 trainer
├── qwen_trainer.py       # Qwen Embedding trainer
├── alignment/            # Triple alignment scoring
└── data/                 # Data loading utilities
```

## Quick Start

### BGE-M3 Training

```python
from training import BGEM3Model, BGETrainer, BGETrainerConfig
from transformers import AutoTokenizer

config = BGETrainerConfig(
    model_name_or_path="BAAI/bge-m3",
    output_dir="./model/checkpoints/bge-m3",
    learning_rate=1e-5,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    unified_finetuning=True,
    use_self_distill=True,
    fp16=True,
)

model = BGEM3Model(
    model_name=config.model_name_or_path,
    unified_finetuning=config.unified_finetuning,
)

tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)

trainer = BGETrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    config=config,
)

trainer.train()
```

### Qwen Embedding Training

```python
from training import create_qwen_trainer

trainer = create_qwen_trainer(
    dataset_paths=["data/training/dataset.jsonl"],
    output_dir="./model/checkpoints/qwen",
    learning_rate=6e-6,
    num_train_epochs=10,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=16,
)

trainer.train()
```

## Configuration

Configuration files are located in `src/config/`:
- `training.yaml` - Default training configuration
- `retrieval_model.yaml` - Model-specific settings
- `experiments/` - Experiment-specific configurations

## Training Scripts

Training scripts are located in `experiments/training/`:
- `train_bge.sh` - BGE-M3 training script
- `train_qwen.sh` - Qwen training script
- `compute_alignment.sh` - Triple alignment computation

## Features

- **Unified Training Interface**: All trainers inherit from `BaseRetrieverTrainer`
- **DeepSpeed Support**: Distributed training with DeepSpeed
- **Checkpoint Management**: Automatic checkpoint saving and loading
- **Knowledge Distillation**: Teacher-student learning for BGE-M3
- **Curriculum Learning**: Three-stage curriculum for Qwen Embedding
- **Triple Alignment**: Forward/backward/parameter alignment scoring

## Dependencies

```
torch >= 2.0.0
transformers >= 4.30.0
datasets >= 2.0.0
sentence-transformers >= 2.0.0  # For BGE-M3
ms-swift >= 2.0.0               # For Qwen Embedding
```
