"""
BGE-M3 model trainer with unified fine-tuning support.

This module provides training functionality for BGE-M3 models with:
- Dense, sparse, and ColBERT representations
- Knowledge distillation from teacher models
- Self-distillation with cosine-scheduled mixing
- InfoNCE + KL Divergence loss combination
"""

import logging
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn, Tensor
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    PreTrainedTokenizer,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments as HFTrainingArguments,
)
from transformers.file_utils import ModelOutput

from ..trainer import HFTrainerWrapper, TrainerConfig

logger = logging.getLogger(__name__)


@dataclass
class EncoderOutput(ModelOutput):
    """Output from encoder model.

    Attributes:
        q_reps: Query representations
        p_reps: Passage representations
        loss: Training loss
        scores: Similarity scores
    """
    q_reps: Optional[Tensor] = None
    p_reps: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None


@dataclass
class BGETrainerConfig(TrainerConfig):
    """Configuration for BGE-M3 trainer.

    Attributes:
        model_name_or_path: Path to pretrained BGE-M3 model
        normalized: Whether to normalize embeddings
        sentence_pooling_method: Pooling method ('cls' or 'mean')
        negatives_cross_device: Share negatives across devices
        temperature: Temperature for contrastive loss
        enable_sub_batch: Enable sub-batch encoding for long sequences
        unified_finetuning: Enable unified fine-tuning (dense + sparse + colbert)
        use_self_distill: Use self-distillation
        colbert_dim: Dimension for ColBERT linear projection (-1 for hidden_size)
        self_distill_start_step: Step to start self-distillation (-1 to disable)
        lambda_min: Minimum lambda weight for loss mixing (favors InfoNCE)
        lambda_max: Maximum lambda weight for loss mixing (favors KL divergence)
        fix_position_embedding: Freeze position embeddings
        fix_encoder: Freeze encoder (only train sparse/colbert layers)
        knowledge_distillation: Use knowledge distillation from teacher scores
    """
    model_name_or_path: str = "BAAI/bge-m3"
    normalized: bool = True
    sentence_pooling_method: str = "cls"
    negatives_cross_device: bool = False
    temperature: float = 0.02
    enable_sub_batch: bool = True
    unified_finetuning: bool = True
    use_self_distill: bool = False
    colbert_dim: int = -1
    self_distill_start_step: int = -1
    lambda_min: float = 0.1
    lambda_max: float = 0.9
    fix_position_embedding: bool = False
    fix_encoder: bool = False
    knowledge_distillation: bool = False


class BGEM3Model(nn.Module):
    """BGE-M3 model with unified fine-tuning support.

    This model supports three types of representations:
    - Dense: Single vector representation (CLS or mean pooling)
    - Sparse: Lexical matching via learned term weights
    - ColBERT: Multi-vector representation with late interaction
    """

    def __init__(
        self,
        model_name: str = None,
        normalized: bool = True,
        sentence_pooling_method: str = 'cls',
        negatives_cross_device: bool = False,
        temperature: float = 1.0,
        enable_sub_batch: bool = True,
        unified_finetuning: bool = True,
        use_self_distill: bool = False,
        colbert_dim: int = -1,
        self_distill_start_step: int = -1,
        max_steps: int = 1000,
        lambda_min: float = 0.1,
        lambda_max: float = 0.9,
    ):
        """Initialize BGE-M3 model.

        Args:
            model_name: Path to pretrained model
            normalized: Whether to normalize embeddings
            sentence_pooling_method: Pooling method ('cls' or 'mean')
            negatives_cross_device: Share negatives across devices
            temperature: Temperature for contrastive loss
            enable_sub_batch: Enable sub-batch encoding
            unified_finetuning: Enable unified fine-tuning
            use_self_distill: Use self-distillation
            colbert_dim: Dimension for ColBERT projection
            self_distill_start_step: Step to start self-distillation
            max_steps: Maximum training steps for lambda scheduling
            lambda_min: Minimum lambda weight
            lambda_max: Maximum lambda weight
        """
        super().__init__()
        self.load_model(model_name, colbert_dim=colbert_dim)
        self.vocab_size = self.model.config.vocab_size
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

        self.unified_finetuning = unified_finetuning
        if not self.unified_finetuning:
            self.colbert_linear = None
            self.sparse_linear = None

        self.normalized = normalized
        self.sentence_pooling_method = sentence_pooling_method
        self.enable_sub_batch = enable_sub_batch
        self.temperature = temperature
        self.use_self_distill = use_self_distill
        self.self_distill_start_step = self_distill_start_step

        # Lambda scheduling parameters
        self.max_steps = max_steps
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.step = 0

        if not normalized:
            self.temperature = 1.0
            logger.info("Reset temperature = 1.0 due to using inner product")

        self.negatives_cross_device = negatives_cross_device
        if self.negatives_cross_device:
            if not dist.is_initialized():
                raise ValueError('Distributed training not initialized for cross-device negatives')
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

    def load_model(self, model_name: str, colbert_dim: int = -1) -> None:
        """Load pretrained model and initialize projection layers.

        Args:
            model_name: Path to pretrained model
            colbert_dim: Dimension for ColBERT projection
        """
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        hidden_size = self.model.config.hidden_size
        self.colbert_linear = nn.Linear(
            in_features=hidden_size,
            out_features=hidden_size if colbert_dim == -1 else colbert_dim
        )
        self.sparse_linear = nn.Linear(in_features=hidden_size, out_features=1)

        # Load projection layers if they exist
        colbert_path = os.path.join(model_name, 'colbert_linear.pt')
        sparse_path = os.path.join(model_name, 'sparse_linear.pt')

        if os.path.exists(colbert_path) and os.path.exists(sparse_path):
            logger.info('Loading existing colbert_linear and sparse_linear')
            self.load_pooler(model_dir=model_name)
        else:
            logger.info('Initializing new colbert_linear and sparse_linear (training mode)')

    def gradient_checkpointing_enable(self, **kwargs) -> None:
        """Enable gradient checkpointing for memory efficiency."""
        self.model.gradient_checkpointing_enable(**kwargs)

    def dense_embedding(self, hidden_state: Tensor, mask: Tensor) -> Tensor:
        """Compute dense embedding via pooling.

        Args:
            hidden_state: Last hidden state from encoder
            mask: Attention mask

        Returns:
            Dense embedding vector
        """
        if self.sentence_pooling_method == 'cls':
            return hidden_state[:, 0]
        elif self.sentence_pooling_method == 'mean':
            s = torch.sum(hidden_state * mask.unsqueeze(-1).float(), dim=1)
            d = mask.sum(axis=1, keepdim=True).float()
            return s / d

    def sparse_embedding(
        self,
        hidden_state: Tensor,
        input_ids: Tensor,
        return_embedding: bool = True
    ) -> Tensor:
        """Compute sparse embedding (learned term weights).

        Args:
            hidden_state: Last hidden state from encoder
            input_ids: Input token IDs
            return_embedding: Whether to return full vocabulary embedding

        Returns:
            Sparse embedding (vocab_size dimension) or token weights
        """
        token_weights = torch.relu(self.sparse_linear(hidden_state))
        if not return_embedding:
            return token_weights

        sparse_embedding = torch.zeros(
            input_ids.size(0),
            input_ids.size(1),
            self.vocab_size,
            dtype=token_weights.dtype,
            device=token_weights.device
        )
        sparse_embedding = torch.scatter(
            sparse_embedding,
            dim=-1,
            index=input_ids.unsqueeze(-1),
            src=token_weights
        )

        # Zero out special tokens
        unused_tokens = [
            self.tokenizer.cls_token_id,
            self.tokenizer.eos_token_id,
            self.tokenizer.pad_token_id,
            self.tokenizer.unk_token_id,
        ]
        sparse_embedding = torch.max(sparse_embedding, dim=1).values
        sparse_embedding[:, unused_tokens] *= 0.0
        return sparse_embedding

    def colbert_embedding(self, last_hidden_state: Tensor, mask: Tensor) -> Tensor:
        """Compute ColBERT multi-vector embedding.

        Args:
            last_hidden_state: Last hidden state from encoder
            mask: Attention mask

        Returns:
            ColBERT multi-vector representation
        """
        colbert_vecs = self.colbert_linear(last_hidden_state[:, 1:])
        colbert_vecs = colbert_vecs * mask[:, 1:][:, :, None].float()
        return colbert_vecs

    def dense_score(self, q_reps: Tensor, p_reps: Tensor) -> Tensor:
        """Compute dense similarity scores."""
        scores = self.compute_similarity(q_reps, p_reps) / self.temperature
        scores = scores.view(q_reps.size(0), -1)
        return scores

    def sparse_score(self, q_reps: Tensor, p_reps: Tensor) -> Tensor:
        """Compute sparse similarity scores."""
        scores = self.compute_similarity(q_reps, p_reps) / self.temperature
        scores = scores.view(q_reps.size(0), -1)
        return scores

    def colbert_score(self, q_reps: Tensor, p_reps: Tensor, q_mask: Tensor) -> Tensor:
        """Compute ColBERT late interaction scores.

        Args:
            q_reps: Query multi-vector representations
            p_reps: Passage multi-vector representations
            q_mask: Query attention mask

        Returns:
            ColBERT similarity scores
        """
        token_scores = torch.einsum('qin,pjn->qipj', q_reps, p_reps)
        scores, _ = token_scores.max(-1)
        scores = scores.sum(1) / q_mask[:, 1:].sum(-1, keepdim=True)
        scores = scores / self.temperature
        return scores

    def _encode(self, features: Dict[str, Tensor]) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        """Encode features into embeddings.

        Args:
            features: Input features dict

        Returns:
            Tuple of (dense_vecs, sparse_vecs, colbert_vecs)
        """
        last_hidden_state = self.model(**features, return_dict=True).last_hidden_state
        dense_vecs = self.dense_embedding(last_hidden_state, features['attention_mask'])

        sparse_vecs = None
        colbert_vecs = None
        if self.unified_finetuning:
            sparse_vecs = self.sparse_embedding(last_hidden_state, features['input_ids'])
            colbert_vecs = self.colbert_embedding(last_hidden_state, features['attention_mask'])

        if self.normalized:
            dense_vecs = F.normalize(dense_vecs, dim=-1)
            if self.unified_finetuning:
                colbert_vecs = F.normalize(colbert_vecs, dim=-1)

        return dense_vecs, sparse_vecs, colbert_vecs

    def encode(
        self,
        features: Optional[Dict[str, Tensor]],
        sub_batch_size: Optional[int] = None
    ) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        """Encode features with optional sub-batching.

        Args:
            features: Input features dict
            sub_batch_size: Sub-batch size for long sequences

        Returns:
            Tuple of (dense_vecs, sparse_vecs, colbert_vecs)
        """
        if features is None:
            return None, None, None

        if sub_batch_size is not None and sub_batch_size != -1:
            all_dense_vecs, all_sparse_vecs, all_colbert_vecs = [], [], []
            for i in range(0, len(features['attention_mask']), sub_batch_size):
                end_idx = min(i + sub_batch_size, len(features['attention_mask']))
                sub_features = {k: v[i:end_idx] for k, v in features.items()}

                dense_vecs, sparse_vecs, colbert_vecs = self._encode(sub_features)
                all_dense_vecs.append(dense_vecs)
                if self.unified_finetuning:
                    all_sparse_vecs.append(sparse_vecs)
                    all_colbert_vecs.append(colbert_vecs)

            dense_vecs = torch.cat(all_dense_vecs, 0)
            if self.unified_finetuning:
                sparse_vecs = torch.cat(all_sparse_vecs, 0)
                colbert_vecs = torch.cat(all_colbert_vecs, 0)
            else:
                sparse_vecs, colbert_vecs = None, None
        else:
            dense_vecs, sparse_vecs, colbert_vecs = self._encode(features)

        return (
            dense_vecs.contiguous() if dense_vecs is not None else None,
            sparse_vecs.contiguous() if sparse_vecs is not None else None,
            colbert_vecs.contiguous() if colbert_vecs is not None else None,
        )

    def compute_sub_batch_size(self, features: Dict[str, Tensor]) -> int:
        """Compute optimal sub-batch size based on sequence length."""
        mapping = [
            (6000, 1), (5000, 2), (4000, 3), (3000, 3),
            (2000, 5), (1000, 9), (512, 16), (0, 32)
        ]
        cur_len = features['input_ids'].size(-1)
        for length, batch_size in mapping:
            if cur_len >= length:
                return batch_size
        return 32

    def compute_similarity(self, q_reps: Tensor, p_reps: Tensor) -> Tensor:
        """Compute similarity scores between queries and passages."""
        if len(p_reps.size()) == 2:
            return torch.matmul(q_reps, p_reps.transpose(0, 1))
        return torch.matmul(q_reps, p_reps.transpose(-2, -1))

    def get_lambda_weight(self) -> float:
        """Get current lambda weight for loss mixing using cosine decay.

        Returns:
            Lambda weight (decays from lambda_max to lambda_min)
        """
        if self.step >= self.max_steps:
            return self.lambda_min

        cosine_decay = 0.5 * (1 + math.cos(math.pi * self.step / self.max_steps))
        lambda_weight = self.lambda_min + (self.lambda_max - self.lambda_min) * cosine_decay
        return lambda_weight

    def info_nce_loss(
        self,
        student_scores: Tensor,
        positive_indices: Tensor,
        temperature: float
    ) -> Tensor:
        """Compute InfoNCE loss.

        Args:
            student_scores: Student model scores [batch_size, num_samples]
            positive_indices: Positive sample indices [batch_size]
            temperature: Temperature parameter

        Returns:
            InfoNCE loss
        """
        logits = student_scores / temperature
        loss = F.cross_entropy(logits, positive_indices)
        return loss

    def contrastive_loss_with_teacher(
        self,
        student_scores: Tensor,
        teacher_scores: Tensor,
        temperature: float
    ) -> Tensor:
        """Compute mixed InfoNCE + KL divergence loss with cosine-scheduled mixing.

        Args:
            student_scores: Student model scores
            teacher_scores: Teacher model scores
            temperature: Temperature parameter

        Returns:
            Weighted combination of InfoNCE and KL divergence loss
        """
        lambda_weight = self.get_lambda_weight()

        # InfoNCE loss (assumes positive at index 0)
        positive_indices = torch.zeros(
            student_scores.size(0),
            dtype=torch.long,
            device=student_scores.device
        )
        info_nce_loss = self.info_nce_loss(student_scores, positive_indices, temperature)

        # KL divergence loss
        student_scores_scaled = student_scores / temperature
        teacher_scores_scaled = teacher_scores / temperature
        student_log_probs = F.log_softmax(student_scores_scaled, dim=-1)
        teacher_probs = F.softmax(teacher_scores_scaled, dim=-1)
        kl_loss = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction='batchmean'
        ) * (temperature ** 2)

        # Weighted combination: lambda * InfoNCE + (1-lambda) * KL
        weighted_loss = lambda_weight * info_nce_loss + (1 - lambda_weight) * kl_loss

        if self.step % 100 == 0:
            logger.info(
                f"Step {self.step}: lambda={lambda_weight:.4f}, "
                f"InfoNCE={info_nce_loss:.4f}, KL={kl_loss:.4f}"
            )

        return weighted_loss

    def forward(
        self,
        query: Optional[Dict[str, Tensor]] = None,
        passage: Optional[Dict[str, Tensor]] = None,
        teacher_scores: Optional[Tensor] = None,
        bi_directions: Optional[bool] = None,
    ) -> EncoderOutput:
        """Forward pass for training.

        Args:
            query: Query features
            passage: Passage features
            teacher_scores: Teacher model scores for distillation
            bi_directions: Whether to use bidirectional training

        Returns:
            EncoderOutput with loss and representations
        """
        if self.enable_sub_batch:
            q_dense, q_sparse, q_colbert = self.encode(
                query,
                sub_batch_size=self.compute_sub_batch_size(query)
            )
            p_dense, p_sparse, p_colbert = self.encode(
                passage,
                sub_batch_size=self.compute_sub_batch_size(passage)
            )
        else:
            q_dense, q_sparse, q_colbert = self.encode(query)
            p_dense, p_sparse, p_colbert = self.encode(passage)

        loss = None
        if self.training and teacher_scores is not None:
            # Dense loss
            dense_scores = self.dense_score(q_dense, p_dense)
            dense_loss = self.contrastive_loss_with_teacher(
                dense_scores,
                teacher_scores,
                self.temperature
            )

            if self.unified_finetuning:
                # Sparse loss
                sparse_scores = self.sparse_score(q_sparse, p_sparse)
                sparse_loss = self.contrastive_loss_with_teacher(
                    sparse_scores,
                    teacher_scores,
                    self.temperature
                )

                # ColBERT loss
                colbert_scores = self.colbert_score(
                    q_colbert,
                    p_colbert,
                    q_mask=query['attention_mask']
                )
                colbert_loss = self.contrastive_loss_with_teacher(
                    colbert_scores,
                    teacher_scores,
                    self.temperature
                )

                # Ensemble loss
                ensemble_scores = dense_scores + 0.3 * sparse_scores + colbert_scores
                ensemble_loss = self.contrastive_loss_with_teacher(
                    ensemble_scores,
                    teacher_scores,
                    self.temperature
                )

                # Weighted average of all losses
                loss = (dense_loss + ensemble_loss + 0.1 * sparse_loss + colbert_loss) / 4
            else:
                loss = dense_loss

            self.step += 1

        return EncoderOutput(loss=loss)

    def save(self, output_dir: str) -> None:
        """Save model to directory.

        Args:
            output_dir: Directory to save model
        """
        def _trans_state_dict(state_dict):
            return type(state_dict)({
                k: v.clone().cpu() for k, v in state_dict.items()
            })

        self.model.save_pretrained(
            output_dir,
            state_dict=_trans_state_dict(self.model.state_dict())
        )

        if self.unified_finetuning:
            torch.save(
                _trans_state_dict(self.colbert_linear.state_dict()),
                os.path.join(output_dir, 'colbert_linear.pt')
            )
            torch.save(
                _trans_state_dict(self.sparse_linear.state_dict()),
                os.path.join(output_dir, 'sparse_linear.pt')
            )

    def load_pooler(self, model_dir: str) -> None:
        """Load projection layers from directory.

        Args:
            model_dir: Directory containing projection layers
        """
        colbert_state_dict = torch.load(
            os.path.join(model_dir, 'colbert_linear.pt'),
            map_location='cpu'
        )
        sparse_state_dict = torch.load(
            os.path.join(model_dir, 'sparse_linear.pt'),
            map_location='cpu'
        )
        self.colbert_linear.load_state_dict(colbert_state_dict)
        self.sparse_linear.load_state_dict(sparse_state_dict)


class BGETrainer(HFTrainerWrapper):
    """Trainer for BGE-M3 models using HuggingFace Trainer."""

    def __init__(
        self,
        model: BGEM3Model,
        tokenizer: PreTrainedTokenizer,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        config: Optional[BGETrainerConfig] = None,
        data_collator: Optional[Any] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
    ):
        """Initialize BGE trainer.

        Args:
            model: BGE-M3 model
            tokenizer: Tokenizer
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            config: Training configuration
            data_collator: Data collator
            callbacks: Trainer callbacks
        """
        if config is None:
            config = BGETrainerConfig(output_dir="./output")

        super().__init__(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            config=config,
            data_collator=data_collator,
            callbacks=callbacks,
        )

        # Apply model-specific configurations
        self._configure_model()

    def _configure_model(self) -> None:
        """Apply model-specific configurations."""
        config: BGETrainerConfig = self.config

        # Enable gradient checkpointing if requested
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        # Freeze position embeddings if requested
        if config.fix_position_embedding:
            for name, param in self.model.named_parameters():
                if "position_embeddings" in name:
                    logger.info(f"Freezing {name}")
                    param.requires_grad = False

        # Freeze encoder if requested (only train projection layers)
        if config.fix_encoder:
            for name, param in self.model.named_parameters():
                if "colbert_linear" in name or "sparse_linear" in name:
                    logger.info(f"Training {name}")
                else:
                    param.requires_grad = False

    def _create_trainer(self) -> Trainer:
        """Create HuggingFace Trainer instance."""
        return BiTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=self.data_collator,
            tokenizer=self.tokenizer,
            callbacks=self.callbacks,
        )


class BiTrainer(Trainer):
    """Custom trainer for BGE-M3 bi-encoder models."""

    def _save(self, output_dir: Optional[str] = None, state_dict=None) -> None:
        """Save model checkpoint.

        Args:
            output_dir: Directory to save checkpoint
            state_dict: State dict to save (unused, model.save() is used instead)
        """
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        if not hasattr(self.model, 'save'):
            raise NotImplementedError(
                f'Model {self.model.__class__.__name__} does not support save interface'
            )

        self.model.save(output_dir)

        if self.tokenizer is not None and self.is_world_process_zero():
            self.tokenizer.save_pretrained(output_dir)

        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> Union[Tensor, Tuple[Tensor, EncoderOutput]]:
        """Compute training loss.

        Args:
            model: The model
            inputs: Input batch
            return_outputs: Whether to return model outputs
            num_items_in_batch: Number of items in batch (unused)

        Returns:
            Loss tensor, optionally with model outputs
        """
        outputs = model(**inputs)
        loss = outputs.loss

        return (loss, outputs) if return_outputs else loss
