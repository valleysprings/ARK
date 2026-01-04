"""
Base trainer module for ARK retrieval models.

This module provides abstract base classes and utilities for training retrieval models,
including support for DeepSpeed, distributed training, and checkpoint management.
"""

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import Dataset
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_callback import TrainerCallback

logger = logging.getLogger(__name__)


@dataclass
class TrainerConfig:
    """Configuration for retrieval model training.

    Attributes:
        output_dir: Directory for saving checkpoints and logs
        learning_rate: Learning rate for optimizer
        num_train_epochs: Number of training epochs
        per_device_train_batch_size: Batch size per device for training
        per_device_eval_batch_size: Batch size per device for evaluation
        gradient_accumulation_steps: Number of steps to accumulate gradients
        warmup_steps: Number of warmup steps for learning rate scheduler
        logging_steps: Log every X updates steps
        save_steps: Save checkpoint every X updates steps
        eval_steps: Evaluate every X updates steps
        save_total_limit: Limit the total number of checkpoints
        seed: Random seed for reproducibility
        fp16: Whether to use fp16 16-bit (mixed) precision training
        bf16: Whether to use bf16 16-bit (mixed) precision training
        deepspeed: Path to deepspeed config file or dict
        local_rank: For distributed training: local_rank
        ddp_backend: The backend to use for distributed training
        gradient_checkpointing: Whether to use gradient checkpointing
    """
    output_dir: str
    learning_rate: float = 5e-5
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    warmup_steps: int = 0
    logging_steps: int = 100
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3
    seed: int = 42
    fp16: bool = False
    bf16: bool = False
    deepspeed: Optional[Union[str, Dict]] = None
    local_rank: int = -1
    ddp_backend: str = "nccl"
    gradient_checkpointing: bool = False


class BaseRetrieverTrainer(ABC):
    """Abstract base class for retrieval model trainers.

    This class provides a unified interface for training different types of retrieval models
    (e.g., BGE-M3, Qwen Embedding) with common functionality for data preparation,
    training, evaluation, and checkpoint management.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: PreTrainedTokenizer,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        config: Optional[TrainerConfig] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
    ):
        """Initialize the trainer.

        Args:
            model: The model to train
            tokenizer: Tokenizer for processing text
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset (optional)
            config: Training configuration
            callbacks: List of callback handlers
        """
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.config = config or TrainerConfig(output_dir="./output")
        self.callbacks = callbacks or []

        # Initialize distributed training if needed
        self._init_distributed()

        # Setup logging
        self._setup_logging()

    def _init_distributed(self) -> None:
        """Initialize distributed training environment."""
        if self.config.local_rank != -1:
            if not dist.is_initialized():
                torch.distributed.init_process_group(backend=self.config.ddp_backend)
            self.world_size = dist.get_world_size()
            self.process_rank = dist.get_rank()
        else:
            self.world_size = 1
            self.process_rank = 0

        logger.info(
            f"Process rank: {self.process_rank}, "
            f"World size: {self.world_size}, "
            f"Device: {torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'}"
        )

    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            level=logging.INFO if self.process_rank in [-1, 0] else logging.WARN,
        )

    def is_world_process_zero(self) -> bool:
        """Check if current process is the main process."""
        return self.process_rank in [-1, 0]

    @abstractmethod
    def prepare_data(self) -> Tuple[Dataset, Optional[Dataset]]:
        """Prepare training and evaluation datasets.

        Returns:
            Tuple of (train_dataset, eval_dataset)
        """
        raise NotImplementedError

    @abstractmethod
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary of training metrics
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate(self) -> Dict[str, float]:
        """Evaluate the model.

        Returns:
            Dictionary of evaluation metrics
        """
        raise NotImplementedError

    def train(self) -> None:
        """Main training loop."""
        logger.info("Starting training...")
        logger.info(f"Training configuration: {self.config}")

        # Prepare data
        train_dataset, eval_dataset = self.prepare_data()

        # Training loop
        for epoch in range(int(self.config.num_train_epochs)):
            logger.info(f"Epoch {epoch + 1}/{self.config.num_train_epochs}")

            # Train for one epoch
            train_metrics = self.train_epoch(epoch)
            logger.info(f"Training metrics: {train_metrics}")

            # Evaluate if eval dataset is provided
            if eval_dataset is not None:
                eval_metrics = self.evaluate()
                logger.info(f"Evaluation metrics: {eval_metrics}")

            # Save checkpoint
            if (epoch + 1) % self.config.save_steps == 0:
                checkpoint_path = os.path.join(
                    self.config.output_dir,
                    f"checkpoint-epoch-{epoch + 1}"
                )
                self.save_checkpoint(checkpoint_path)

        # Save final model
        self.save_model(self.config.output_dir)
        logger.info("Training completed!")

    def save_checkpoint(self, output_dir: str) -> None:
        """Save a training checkpoint.

        Args:
            output_dir: Directory to save checkpoint
        """
        if not self.is_world_process_zero():
            return

        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving checkpoint to {output_dir}")

        # Save model
        self._save_model_impl(output_dir)

        # Save tokenizer
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Save training config
        config_path = os.path.join(output_dir, "training_config.pt")
        torch.save(self.config, config_path)

        logger.info(f"Checkpoint saved to {output_dir}")

    def save_model(self, output_dir: str) -> None:
        """Save the trained model.

        Args:
            output_dir: Directory to save model
        """
        if not self.is_world_process_zero():
            return

        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model to {output_dir}")

        # Save model
        self._save_model_impl(output_dir)

        # Save tokenizer
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        logger.info(f"Model saved to {output_dir}")

    @abstractmethod
    def _save_model_impl(self, output_dir: str) -> None:
        """Implementation-specific model saving logic.

        Args:
            output_dir: Directory to save model
        """
        raise NotImplementedError

    def load_checkpoint(self, checkpoint_dir: str) -> None:
        """Load a training checkpoint.

        Args:
            checkpoint_dir: Directory containing checkpoint
        """
        logger.info(f"Loading checkpoint from {checkpoint_dir}")

        # Load model
        self._load_model_impl(checkpoint_dir)

        # Load training config
        config_path = os.path.join(checkpoint_dir, "training_config.pt")
        if os.path.exists(config_path):
            self.config = torch.load(config_path, map_location="cpu")
            logger.info(f"Loaded training config from {config_path}")

        logger.info(f"Checkpoint loaded from {checkpoint_dir}")

    @abstractmethod
    def _load_model_impl(self, checkpoint_dir: str) -> None:
        """Implementation-specific model loading logic.

        Args:
            checkpoint_dir: Directory containing checkpoint
        """
        raise NotImplementedError


class HFTrainerWrapper(BaseRetrieverTrainer):
    """Wrapper around HuggingFace Trainer for retrieval models.

    This class provides a bridge between the BaseRetrieverTrainer interface
    and HuggingFace's Trainer, enabling use of HF's training infrastructure.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: PreTrainedTokenizer,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        config: Optional[TrainerConfig] = None,
        data_collator: Optional[Any] = None,
        compute_metrics: Optional[Any] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
    ):
        """Initialize the HF trainer wrapper.

        Args:
            model: The model to train
            tokenizer: Tokenizer for processing text
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset (optional)
            config: Training configuration
            data_collator: Data collator for batching
            compute_metrics: Function to compute evaluation metrics
            callbacks: List of callback handlers
        """
        super().__init__(model, tokenizer, train_dataset, eval_dataset, config, callbacks)
        self.data_collator = data_collator
        self.compute_metrics = compute_metrics

        # Create HF training arguments
        self.training_args = self._create_training_arguments()

        # Create HF trainer
        self.trainer = self._create_trainer()

    def _create_training_arguments(self) -> TrainingArguments:
        """Create HuggingFace TrainingArguments from config."""
        return TrainingArguments(
            output_dir=self.config.output_dir,
            learning_rate=self.config.learning_rate,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            warmup_steps=self.config.warmup_steps,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            save_total_limit=self.config.save_total_limit,
            seed=self.config.seed,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            deepspeed=self.config.deepspeed,
            local_rank=self.config.local_rank,
            ddp_backend=self.config.ddp_backend,
            gradient_checkpointing=self.config.gradient_checkpointing,
            remove_unused_columns=False,
            evaluation_strategy="steps" if self.eval_dataset is not None else "no",
        )

    @abstractmethod
    def _create_trainer(self) -> Trainer:
        """Create HuggingFace Trainer instance.

        Returns:
            Configured Trainer instance
        """
        raise NotImplementedError

    def prepare_data(self) -> Tuple[Dataset, Optional[Dataset]]:
        """Prepare training and evaluation datasets."""
        return self.train_dataset, self.eval_dataset

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch using HF Trainer."""
        # HF Trainer handles epoch training internally
        return {}

    def evaluate(self) -> Dict[str, float]:
        """Evaluate the model using HF Trainer."""
        if self.eval_dataset is None:
            logger.warning("No evaluation dataset provided, skipping evaluation")
            return {}

        return self.trainer.evaluate()

    def train(self) -> None:
        """Train the model using HF Trainer."""
        logger.info("Starting training with HuggingFace Trainer...")
        self.trainer.train()
        self.save_model(self.config.output_dir)
        logger.info("Training completed!")

    def _save_model_impl(self, output_dir: str) -> None:
        """Save model using HF Trainer."""
        if hasattr(self.model, 'save'):
            self.model.save(output_dir)
        elif hasattr(self.model, 'save_pretrained'):
            self.model.save_pretrained(output_dir)
        else:
            # Fallback to torch.save
            model_path = os.path.join(output_dir, "pytorch_model.bin")
            torch.save(self.model.state_dict(), model_path)

    def _load_model_impl(self, checkpoint_dir: str) -> None:
        """Load model using HF Trainer."""
        if hasattr(self.model, 'load_pooler'):
            self.model.load_pooler(checkpoint_dir)

        # Load model state dict if available
        model_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location="cpu")
            self.model.load_state_dict(state_dict)


# ==================== Unified Training Entry Point ====================

def train(
    model_type: str,
    config: Optional[Union[str, Dict, TrainerConfig]] = None,
    **kwargs
) -> None:
    """Unified training entry point for all retrieval models.

    This function dispatches to the appropriate trainer based on model_type.

    Args:
        model_type: Type of model to train ('bge' or 'qwen')
        config: Training configuration. Can be:
            - Path to YAML config file
            - Dictionary of config values
            - TrainerConfig instance
        **kwargs: Additional arguments passed to the specific trainer

    Example:
        # Train BGE-M3
        train(
            model_type='bge',
            config='src/config/training.yaml',
            model_name_or_path='./model/bge-m3',
            output_dir='./model/checkpoints/bge-m3'
        )

        # Train Qwen Embedding
        train(
            model_type='qwen',
            config='src/config/training.yaml',
            dataset_paths=['./data/training/qwen_data/stage1'],
            output_dir='./model/checkpoints/qwen'
        )
    """
    from .trainers import BGETrainer, BGETrainerConfig, QwenEmbeddingTrainer, QwenTrainerConfig, create_qwen_trainer
    import yaml

    logger.info(f"Starting {model_type.upper()} model training...")

    # Load config if path or dict provided
    if isinstance(config, str):
        with open(config, 'r') as f:
            config_dict = yaml.safe_load(f)
        # Extract model-specific config
        if model_type == 'bge':
            config_dict = config_dict.get('training', {}).get('bge', {})
        elif model_type == 'qwen':
            config_dict = config_dict.get('training', {}).get('qwen', {})
        config_dict.update(kwargs)
    elif isinstance(config, dict):
        config_dict = {**config, **kwargs}
    else:
        config_dict = kwargs

    # Dispatch to appropriate trainer
    if model_type.lower() == 'bge':
        logger.info("Initializing BGE-M3 trainer...")
        trainer_config = BGETrainerConfig(**config_dict)
        # User needs to provide model and datasets
        raise NotImplementedError(
            "BGE training requires manual setup. Please use BGETrainer directly:\n"
            "  from training.trainers import BGETrainer, BGEM3Model\n"
            "  model = BGEM3Model(...)\n"
            "  trainer = BGETrainer(model=model, ...)\n"
            "  trainer.train()"
        )

    elif model_type.lower() == 'qwen':
        logger.info("Initializing Qwen Embedding trainer...")
        trainer = create_qwen_trainer(**config_dict)
        trainer.train()
        logger.info(f"Qwen training completed! Model saved to {config_dict.get('output_dir')}")

    else:
        raise ValueError(
            f"Unknown model_type: {model_type}. "
            f"Supported types: 'bge', 'qwen'"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train ARK retrieval models")
    parser.add_argument("--model_type", type=str, required=True, choices=['bge', 'qwen'],
                       help="Type of model to train")
    parser.add_argument("--config", type=str, default="src/config/training.yaml",
                       help="Path to training config file")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for checkpoints")

    args, unknown = parser.parse_known_args()

    # Parse additional kwargs
    kwargs = {}
    i = 0
    while i < len(unknown):
        if unknown[i].startswith('--'):
            key = unknown[i][2:]
            if i + 1 < len(unknown) and not unknown[i + 1].startswith('--'):
                value = unknown[i + 1]
                # Try to convert to appropriate type
                try:
                    value = eval(value)
                except:
                    pass
                kwargs[key] = value
                i += 2
            else:
                kwargs[key] = True
                i += 1
        else:
            i += 1

    # Add required args to kwargs
    kwargs['output_dir'] = args.output_dir

    # Run training
    train(model_type=args.model_type, config=args.config, **kwargs)
