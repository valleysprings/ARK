"""
Training module for ARK retrieval models.

This module provides trainers for different retrieval models:
- BGE-M3: Multi-vector retrieval with dense, sparse, and colbert representations
- Qwen Embedding: Dense retrieval with triple alignment scoring

All trainers inherit from BaseRetrieverTrainer and provide:
- Unified training interface
- DeepSpeed support
- Checkpoint management
- Distributed training support
"""

from .trainer import BaseRetrieverTrainer, TrainerConfig, train
from .trainers import (
    BGETrainer,
    BGETrainerConfig,
    BGEM3Model,
    QwenEmbeddingTrainer,
    QwenTrainerConfig,
    create_qwen_trainer,
)

__all__ = [
    # Base classes
    "BaseRetrieverTrainer",
    "TrainerConfig",
    "train",  # Unified training entry point
    # BGE-M3
    "BGETrainer",
    "BGETrainerConfig",
    "BGEM3Model",
    # Qwen Embedding
    "QwenEmbeddingTrainer",
    "QwenTrainerConfig",
    "create_qwen_trainer",
]

__version__ = "0.1.0"
