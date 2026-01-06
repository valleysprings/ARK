"""
Data loading operations for training data generation.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


def load_alignment_data(
    dataset_name: str,
    preprocessed_dir: str = "./data/preprocessed",
    limit: Optional[int] = None
) -> List[Dict]:
    """
    Load alignment data for a single dataset.

    Args:
        dataset_name: Name of the dataset (e.g., 'hotpotqa', 'biology')
        preprocessed_dir: Path to preprocessed data directory
        limit: Maximum number of samples to load (None for all)

    Returns:
        List of alignment data dictionaries with keys:
        - input: question text
        - answers: list of answers
        - chunk_list: list of chunks
        - score_list: list of scores for each chunk
    """
    alignment_dir = Path(preprocessed_dir) / dataset_name / "alignment"

    if not alignment_dir.exists():
        logger.warning(f"Alignment directory not found: {alignment_dir}")
        return []

    alignment_data = []
    json_files = sorted(alignment_dir.glob("*.json"))

    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

                if isinstance(data, dict):
                    alignment_data.append(data)
                elif isinstance(data, list):
                    alignment_data.extend(data)

                if limit is not None and len(alignment_data) >= limit:
                    alignment_data = alignment_data[:limit]
                    break
        except Exception as e:
            logger.error(f"Error loading {json_file}: {e}")
            continue

    logger.info(f"Loaded {len(alignment_data)} samples from {dataset_name}")
    return alignment_data


def load_queries_data(
    dataset_name: str,
    query_type: str,
    preprocessed_dir: str = "./data/preprocessed",
    limit: Optional[int] = None
) -> List[List[str]]:
    """
    Load query data for a single dataset.

    Args:
        dataset_name: Name of the dataset
        query_type: 'large' or 'small' for subgraph size
        preprocessed_dir: Path to preprocessed data directory
        limit: Maximum number of query groups to load

    Returns:
        List of query groups (each group has ~10 queries)
    """
    query_dir = Path(preprocessed_dir) / dataset_name / f"queries_{query_type}"

    if not query_dir.exists():
        logger.warning(f"Query directory not found: {query_dir}")
        return []

    queries = []
    json_files = sorted(query_dir.glob("*.json"))

    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    queries.append(data)

                if limit is not None and len(queries) >= limit:
                    queries = queries[:limit]
                    break
        except Exception as e:
            logger.error(f"Error loading {json_file}: {e}")
            continue

    logger.info(f"Loaded {len(queries)} query groups from {dataset_name} ({query_type})")
    return queries


def load_matched_chunks_data(
    dataset_name: str,
    chunk_type: str,
    preprocessed_dir: str = "./data/preprocessed",
    limit: Optional[int] = None
) -> List[List[str]]:
    """
    Load matched chunks data for a single dataset.

    Args:
        dataset_name: Name of the dataset
        chunk_type: 'large' or 'small' for subgraph size
        preprocessed_dir: Path to preprocessed data directory
        limit: Maximum number of matched chunk groups to load

    Returns:
        List of matched chunk groups (each group is a list of chunks)
    """
    matched_dir = Path(preprocessed_dir) / dataset_name / f"matched_chunks_{chunk_type}"

    if not matched_dir.exists():
        logger.warning(f"Matched chunks directory not found: {matched_dir}")
        return []

    matched_chunks = []
    json_files = sorted(matched_dir.glob("*.json"))

    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    matched_chunks.append(data)

                if limit is not None and len(matched_chunks) >= limit:
                    matched_chunks = matched_chunks[:limit]
                    break
        except Exception as e:
            logger.error(f"Error loading {json_file}: {e}")
            continue

    logger.info(f"Loaded {len(matched_chunks)} matched chunk groups from {dataset_name} ({chunk_type})")
    return matched_chunks
