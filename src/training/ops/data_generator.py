"""
Training data generation for Stage 1/2/3 with ms-swift InfoNCE format.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


def generate_stage1_data(
    alignment_data: List[Dict],
    output_file: str,
    num_positives: int = 3,
    num_negatives: Optional[int] = None
) -> int:
    """
    Generate Stage 1 training data (chunk alignment with random negatives from same entry).

    Format: {"query": "question", "response": "positive_chunk", "rejected_response": ["neg1", ...]}
    Each positive generates one sample. Negatives are randomly sampled from non-positive chunks.

    Args:
        alignment_data: List of alignment data dicts
        output_file: Output JSONL file path
        num_positives: Number of top chunks to use as positives (default 3)
        num_negatives: Number of negatives to sample (None = all remaining chunks)

    Returns:
        Number of training samples generated
    """
    import random
    logger.info(f"Generating Stage 1 training data (num_positives={num_positives})...")

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    num_samples = 0
    num_skipped = 0

    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in tqdm(alignment_data, desc="Stage 1"):
            query = sample.get('input', '')
            chunks = sample.get('chunk_list', [])
            scores = sample.get('score_list', [])

            if not query or not chunks:
                continue

            # Sort chunks by score (descending)
            sorted_indices = sorted(
                range(len(scores)),
                key=lambda i: scores[i],
                reverse=True
            )

            # Get top-k positive chunks
            top_k = min(num_positives, len(sorted_indices))
            positive_chunks = [chunks[sorted_indices[i]] for i in range(top_k)]
            positive_chunks_set = set(positive_chunks)

            # Get negative chunks (all non-positive chunks from this entry)
            negative_chunks = [c for c in chunks if c not in positive_chunks_set]

            if not negative_chunks:
                num_skipped += 1
                continue

            # Random sample if num_negatives specified
            if num_negatives is not None and len(negative_chunks) > num_negatives:
                negative_chunks = random.sample(negative_chunks, num_negatives)

            # Generate one sample per positive
            for positive_chunk in positive_chunks:
                training_sample = {
                    "query": query,
                    "response": positive_chunk,
                    "rejected_response": negative_chunks
                }

                f.write(json.dumps(training_sample, ensure_ascii=False) + '\n')
                num_samples += 1

    logger.info(f"Generated {num_samples} Stage 1 samples (skipped {num_skipped})")
    return num_samples


def generate_stage2_data(
    alignment_data: List[Dict],
    matched_chunks_data: List[List[str]],
    output_file: str,
    num_positives: int = 3,
    num_negatives: Optional[int] = None
) -> int:
    """
    Generate Stage 2 training data (large subgraph contrastive learning).

    Format: {"query": "question", "response": "positive_chunk", "rejected_response": ["neg1", "neg2", ...]}
    Each positive generates one sample with all negatives.

    Args:
        alignment_data: List of alignment data dicts
        matched_chunks_data: List of pre-computed matched chunk groups
        output_file: Output JSONL file path
        num_positives: Number of top chunks to use as positives (default 3)
        num_negatives: Number of negatives (None = all matched negatives)

    Returns:
        Number of training samples generated
    """
    logger.info(f"Generating Stage 2 training data (num_positives={num_positives})...")

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    num_samples = 0
    num_skipped = 0

    with open(output_path, 'w', encoding='utf-8') as f:
        for sample, matched_chunks in tqdm(
            zip(alignment_data, matched_chunks_data),
            desc="Stage 2",
            total=min(len(alignment_data), len(matched_chunks_data))
        ):
            query = sample.get('input', '')
            chunks = sample.get('chunk_list', [])
            scores = sample.get('score_list', [])

            if not query or not chunks or not matched_chunks:
                continue

            # Sort chunks by score (descending)
            sorted_indices = sorted(
                range(len(scores)),
                key=lambda i: scores[i],
                reverse=True
            )

            # Get top-k positive chunks
            top_k = min(num_positives, len(sorted_indices))
            positive_chunks = [chunks[sorted_indices[i]] for i in range(top_k)]
            positive_chunks_set = set(positive_chunks)

            # Filter out positive chunks from matched chunks to get negatives
            negative_chunks = [c for c in matched_chunks if c not in positive_chunks_set]

            if not negative_chunks:
                num_skipped += 1
                continue

            if num_negatives is not None:
                negative_chunks = negative_chunks[:num_negatives]

            # Generate one sample per positive
            for positive_chunk in positive_chunks:
                training_sample = {
                    "query": query,
                    "response": positive_chunk,
                    "rejected_response": negative_chunks
                }

                f.write(json.dumps(training_sample, ensure_ascii=False) + '\n')
                num_samples += 1

    logger.info(f"Generated {num_samples} Stage 2 samples (skipped {num_skipped})")
    return num_samples


def generate_stage3_data(
    alignment_data: List[Dict],
    matched_chunks_data: List[List[str]],
    output_file: str,
    num_positives: int = 3,
    num_negatives: Optional[int] = None
) -> int:
    """
    Generate Stage 3 training data (small subgraph contrastive learning).

    Format: {"query": "question", "response": "positive_chunk", "rejected_response": ["neg1", "neg2", ...]}
    Each positive generates one sample with all negatives.

    Args:
        alignment_data: List of alignment data dicts
        matched_chunks_data: List of pre-computed matched chunk groups
        output_file: Output JSONL file path
        num_positives: Number of top chunks to use as positives (default 3)
        num_negatives: Number of negatives (None = all matched negatives)

    Returns:
        Number of training samples generated
    """
    logger.info(f"Generating Stage 3 training data (num_positives={num_positives})...")

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    num_samples = 0
    num_skipped = 0

    with open(output_path, 'w', encoding='utf-8') as f:
        for sample, matched_chunks in tqdm(
            zip(alignment_data, matched_chunks_data),
            desc="Stage 3",
            total=min(len(alignment_data), len(matched_chunks_data))
        ):
            query = sample.get('input', '')
            chunks = sample.get('chunk_list', [])
            scores = sample.get('score_list', [])

            if not query or not chunks or not matched_chunks:
                continue

            # Sort chunks by score (descending)
            sorted_indices = sorted(
                range(len(scores)),
                key=lambda i: scores[i],
                reverse=True
            )

            # Get top-k positive chunks
            top_k = min(num_positives, len(sorted_indices))
            positive_chunks = [chunks[sorted_indices[i]] for i in range(top_k)]
            positive_chunks_set = set(positive_chunks)

            # Filter out positive chunks from matched chunks to get negatives
            negative_chunks = [c for c in matched_chunks if c not in positive_chunks_set]

            if not negative_chunks:
                num_skipped += 1
                continue

            if num_negatives is not None:
                negative_chunks = negative_chunks[:num_negatives]

            # Generate one sample per positive
            for positive_chunk in positive_chunks:
                training_sample = {
                    "query": query,
                    "response": positive_chunk,
                    "rejected_response": negative_chunks
                }

                f.write(json.dumps(training_sample, ensure_ascii=False) + '\n')
                num_samples += 1

    logger.info(f"Generated {num_samples} Stage 3 samples (skipped {num_skipped})")
    return num_samples
