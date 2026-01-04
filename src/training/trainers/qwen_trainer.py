"""
Qwen Embedding model trainer with curriculum learning support.

This module provides training functionality for Qwen Embedding models with:
- Three-stage curriculum learning
- MS-SWIFT framework integration
- Cosine similarity loss
- Full parameter fine-tuning
"""

import logging
import os
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from ..trainer import BaseRetrieverTrainer, TrainerConfig

logger = logging.getLogger(__name__)


class CurriculumStage(Enum):
    """Curriculum learning stages for Qwen embedding training.

    Stages progress from simple to complex tasks:
    - STAGE1: Basic retrieval tasks (HotpotQA, 2WikiMQA)
    - STAGE2: Domain-specific tasks (Biology, Music, Philosophy, Technology)
    - STAGE3: Advanced tasks (NarrativeQA, Qasper, Musique, Fiction, Legal, Finance)
    """
    STAGE1 = "stage1"
    STAGE2 = "stage2"
    STAGE3 = "stage3"


@dataclass
class QwenTrainerConfig(TrainerConfig):
    """Configuration for Qwen Embedding trainer.

    Attributes:
        model_name_or_path: Path to pretrained Qwen model
        train_type: Training type ('full', 'lora', 'qlora')
        task_type: Task type for MS-SWIFT ('embedding')
        model_type: Model type for MS-SWIFT ('qwen3_emb')
        loss_type: Loss function ('cosine_similarity', 'contrastive')
        dataset_paths: List of dataset paths for training
        dataset_num_proc: Number of processes for dataset loading
        split_dataset_ratio: Ratio for train/eval split
        eval_strategy: Evaluation strategy ('steps', 'epoch', 'no')
        label_names: Names of label fields in dataset
        dataloader_drop_last: Drop last incomplete batch
        add_version: Add version suffix to output directory
        use_curriculum: Enable curriculum learning
        curriculum_stage: Current curriculum stage
        chunk_size: Chunk size for text splitting
        chunk_overlap: Overlap size for text chunking
    """
    model_name_or_path: str = "Qwen/Qwen3-Embedding-0.6B"
    train_type: str = "full"
    task_type: str = "embedding"
    model_type: str = "qwen3_emb"
    loss_type: str = "cosine_similarity"
    dataset_paths: List[str] = field(default_factory=list)
    dataset_num_proc: int = 16
    split_dataset_ratio: float = 0.1
    eval_strategy: str = "steps"
    label_names: List[str] = field(default_factory=lambda: ["label"])
    dataloader_drop_last: bool = True
    add_version: bool = False
    use_curriculum: bool = False
    curriculum_stage: Optional[str] = None
    chunk_size: int = 512
    chunk_overlap: int = 384


class QwenEmbeddingTrainer(BaseRetrieverTrainer):
    """Trainer for Qwen Embedding models using MS-SWIFT framework.

    This trainer wraps the MS-SWIFT CLI for training Qwen embedding models
    with support for curriculum learning and multi-stage training.
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        config: Optional[QwenTrainerConfig] = None,
        **kwargs,
    ):
        """Initialize Qwen Embedding trainer.

        Args:
            model: Model instance (not used, MS-SWIFT loads model internally)
            tokenizer: Tokenizer instance (not used, MS-SWIFT loads tokenizer internally)
            train_dataset: Training dataset (not used, datasets loaded from files)
            eval_dataset: Evaluation dataset (optional)
            config: Training configuration
            **kwargs: Additional arguments
        """
        if config is None:
            config = QwenTrainerConfig(output_dir="./output")

        # Initialize base trainer (model/tokenizer can be None for MS-SWIFT)
        super().__init__(
            model=model or torch.nn.Module(),  # Placeholder
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            config=config,
        )

        self.config: QwenTrainerConfig = config

        # Setup curriculum learning if enabled
        if self.config.use_curriculum:
            self._setup_curriculum()

    def _setup_curriculum(self) -> None:
        """Setup curriculum learning stages and datasets."""
        self.curriculum_datasets = {
            CurriculumStage.STAGE1: [
                "hotpotqa",
                "2wikimqa",
            ],
            CurriculumStage.STAGE2: [
                "biology",
                "music",
                "philosophy",
                "technology",
            ],
            CurriculumStage.STAGE3: [
                "narrativeqa",
                "qasper",
                "musique",
                "fiction",
                "legal",
                "fin",
                "fin_legal",
            ],
        }

        logger.info("Curriculum learning enabled")
        logger.info(f"Stage 1 datasets: {self.curriculum_datasets[CurriculumStage.STAGE1]}")
        logger.info(f"Stage 2 datasets: {self.curriculum_datasets[CurriculumStage.STAGE2]}")
        logger.info(f"Stage 3 datasets: {self.curriculum_datasets[CurriculumStage.STAGE3]}")

    def prepare_data(self) -> tuple:
        """Prepare training and evaluation datasets.

        For MS-SWIFT, datasets are loaded from file paths, so this method
        primarily validates that dataset paths exist.

        Returns:
            Tuple of (train_dataset, eval_dataset) - both None for MS-SWIFT
        """
        if not self.config.dataset_paths:
            raise ValueError("No dataset paths provided")

        # Validate dataset paths
        for dataset_path in self.config.dataset_paths:
            if not os.path.exists(dataset_path):
                raise FileNotFoundError(f"Dataset not found: {dataset_path}")
            logger.info(f"Validated dataset: {dataset_path}")

        return None, None

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch.

        MS-SWIFT handles epoch training internally, so this is not used.

        Args:
            epoch: Current epoch number

        Returns:
            Empty dict
        """
        return {}

    def evaluate(self) -> Dict[str, float]:
        """Evaluate the model.

        MS-SWIFT handles evaluation internally during training.

        Returns:
            Empty dict
        """
        return {}

    def _build_swift_command(self, dataset_path: str, output_dir: str) -> List[str]:
        """Build MS-SWIFT command for training.

        Args:
            dataset_path: Path to dataset file
            output_dir: Directory for saving outputs

        Returns:
            List of command arguments
        """
        cmd = [
            "swift", "sft",
            "--model", self.config.model_name_or_path,
            "--task_type", self.config.task_type,
            "--model_type", self.config.model_type,
            "--train_type", self.config.train_type,
            "--dataset", dataset_path,
            "--dataset_num_proc", str(self.config.dataset_num_proc),
            "--split_dataset_ratio", str(self.config.split_dataset_ratio),
            "--eval_strategy", self.config.eval_strategy,
            "--output_dir", output_dir,
            "--add_version", str(self.config.add_version).lower(),
            "--eval_steps", str(self.config.eval_steps),
            "--num_train_epochs", str(int(self.config.num_train_epochs)),
            "--save_steps", str(self.config.save_steps),
            "--per_device_train_batch_size", str(self.config.per_device_train_batch_size),
            "--per_device_eval_batch_size", str(self.config.per_device_eval_batch_size),
            "--gradient_accumulation_steps", str(self.config.gradient_accumulation_steps),
            "--learning_rate", str(self.config.learning_rate),
            "--loss_type", self.config.loss_type,
            "--label_names", " ".join(self.config.label_names),
            "--dataloader_drop_last", str(self.config.dataloader_drop_last).lower(),
        ]

        # Add logging steps
        cmd.extend(["--logging_steps", str(self.config.logging_steps)])

        # Add seed
        cmd.extend(["--seed", str(self.config.seed)])

        # Add mixed precision training
        if self.config.fp16:
            cmd.append("--fp16")
        if self.config.bf16:
            cmd.append("--bf16")

        # Add DeepSpeed config if provided
        if self.config.deepspeed:
            if isinstance(self.config.deepspeed, str):
                cmd.extend(["--deepspeed", self.config.deepspeed])
            else:
                # Save deepspeed config to temp file
                import json
                import tempfile
                with tempfile.NamedTemporaryFile(
                    mode='w',
                    suffix='.json',
                    delete=False
                ) as f:
                    json.dump(self.config.deepspeed, f)
                    cmd.extend(["--deepspeed", f.name])

        return cmd

    def _train_single_dataset(
        self,
        dataset_path: str,
        dataset_name: str,
        log_file: Optional[str] = None
    ) -> None:
        """Train on a single dataset using MS-SWIFT.

        Args:
            dataset_path: Path to dataset file
            dataset_name: Name of dataset (for output directory)
            log_file: Path to log file (optional)
        """
        # Determine output directory
        chunk_str = f"chunk{self.config.chunk_size}_{self.config.chunk_overlap}"
        output_dir = os.path.join(
            self.config.output_dir,
            dataset_name,
            "nolabel",
            chunk_str
        )

        # Build MS-SWIFT command
        cmd = self._build_swift_command(dataset_path, output_dir)

        logger.info(f"Starting training for dataset: {dataset_name}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Command: {' '.join(cmd)}")

        # Run MS-SWIFT training
        try:
            if log_file:
                # Run with output redirection
                with open(log_file, 'w') as f:
                    process = subprocess.Popen(
                        cmd,
                        stdout=f,
                        stderr=subprocess.STDOUT,
                        text=True
                    )
                    process.wait()
            else:
                # Run with live output
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True
                )

                # Stream output
                for line in process.stdout:
                    print(line, end='')

                process.wait()

            if process.returncode != 0:
                raise RuntimeError(
                    f"MS-SWIFT training failed with return code {process.returncode}"
                )

            logger.info(f"Training completed for dataset: {dataset_name}")

        except Exception as e:
            logger.error(f"Error training dataset {dataset_name}: {e}")
            raise

    def train(self) -> None:
        """Train the model using MS-SWIFT framework.

        If curriculum learning is enabled, trains on datasets in stages.
        Otherwise, trains on all provided dataset paths.
        """
        logger.info("Starting Qwen Embedding training with MS-SWIFT")

        # Create log directory
        log_dir = os.path.join(self.config.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)

        if self.config.use_curriculum:
            # Curriculum learning: train stage by stage
            self._train_curriculum()
        else:
            # Standard training: train on all datasets
            self._train_standard()

        logger.info("Training completed!")

    def _train_curriculum(self) -> None:
        """Train using curriculum learning (three stages)."""
        logger.info("Training with curriculum learning")

        # Determine which stage to train
        if self.config.curriculum_stage:
            stages = [CurriculumStage(self.config.curriculum_stage)]
        else:
            stages = [
                CurriculumStage.STAGE1,
                CurriculumStage.STAGE2,
                CurriculumStage.STAGE3,
            ]

        for stage in stages:
            logger.info(f"Training {stage.value}")
            datasets = self.curriculum_datasets[stage]

            for dataset_name in datasets:
                # Find dataset file
                dataset_path = self._find_dataset_path(dataset_name)
                if dataset_path is None:
                    logger.warning(f"Dataset not found: {dataset_name}, skipping")
                    continue

                # Setup log file
                log_file = os.path.join(
                    self.config.output_dir,
                    "logs",
                    f"{dataset_name}_nolabel_chunk{self.config.chunk_size}_{self.config.chunk_overlap}.log"
                )

                # Train on dataset
                self._train_single_dataset(dataset_path, dataset_name, log_file)

            logger.info(f"Completed {stage.value}")

    def _train_standard(self) -> None:
        """Train on all provided datasets."""
        logger.info("Training in standard mode (no curriculum)")

        for dataset_path in self.config.dataset_paths:
            # Extract dataset name from path
            dataset_name = Path(dataset_path).stem

            # Setup log file
            log_file = os.path.join(
                self.config.output_dir,
                "logs",
                f"{dataset_name}_nolabel_chunk{self.config.chunk_size}_{self.config.chunk_overlap}.log"
            )

            # Train on dataset
            self._train_single_dataset(dataset_path, dataset_name, log_file)

    def _find_dataset_path(self, dataset_name: str) -> Optional[str]:
        """Find dataset path by name.

        Searches in configured dataset paths for a file matching the dataset name.

        Args:
            dataset_name: Name of dataset (without extension)

        Returns:
            Path to dataset file, or None if not found
        """
        # Check if dataset_paths contains directories
        for path in self.config.dataset_paths:
            if os.path.isdir(path):
                # Search in directory
                for ext in ['.jsonl', '.json']:
                    candidate = os.path.join(path, f"{dataset_name}{ext}")
                    if os.path.exists(candidate):
                        return candidate
            elif os.path.isfile(path):
                # Check if filename matches
                if Path(path).stem == dataset_name:
                    return path

        return None

    def _save_model_impl(self, output_dir: str) -> None:
        """Save model implementation.

        For MS-SWIFT, models are saved automatically during training.

        Args:
            output_dir: Directory to save model
        """
        logger.info(f"MS-SWIFT automatically saves models to {output_dir}")

    def _load_model_impl(self, checkpoint_dir: str) -> None:
        """Load model implementation.

        For MS-SWIFT, models are loaded via the --model argument.

        Args:
            checkpoint_dir: Directory containing checkpoint
        """
        logger.info(f"To resume from checkpoint, set model_name_or_path={checkpoint_dir}")


def create_qwen_trainer(
    config: Optional[QwenTrainerConfig] = None,
    dataset_paths: Optional[List[str]] = None,
    output_dir: str = "./output",
    use_curriculum: bool = False,
    curriculum_stage: Optional[str] = None,
    **kwargs,
) -> QwenEmbeddingTrainer:
    """Factory function to create Qwen Embedding trainer.

    Args:
        config: Training configuration
        dataset_paths: List of dataset paths
        output_dir: Output directory
        use_curriculum: Enable curriculum learning
        curriculum_stage: Specific curriculum stage to train
        **kwargs: Additional config arguments

    Returns:
        Configured QwenEmbeddingTrainer instance

    Example:
        >>> trainer = create_qwen_trainer(
        ...     dataset_paths=["/path/to/hotpotqa.jsonl"],
        ...     output_dir="./output",
        ...     learning_rate=6e-6,
        ...     num_train_epochs=10,
        ... )
        >>> trainer.train()
    """
    if config is None:
        config = QwenTrainerConfig(
            output_dir=output_dir,
            dataset_paths=dataset_paths or [],
            use_curriculum=use_curriculum,
            curriculum_stage=curriculum_stage,
            **kwargs,
        )

    return QwenEmbeddingTrainer(config=config)
