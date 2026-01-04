"""
Contrastive Pair Generation for Answer-Centric Retriever Training

This module generates contrastive training pairs (positive vs hard negative chunks) for
curriculum-based contrastive learning of the retriever using InfoNCE loss.

The contrastive pairs are generated based on:
1. Scored chunks from the AlignmentScorer (positives: answer-sufficient chunks)
2. Semantic similarity between augmented queries and chunks (hard negatives: similar but insufficient)
3. Filtering to ensure positive and negative chunks are distinct
"""

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple
import yaml

import jsonlines
import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm


@dataclass
class PreferenceConfig:
    """Configuration for contrastive pair generation.

    Attributes:
        top_n: Number of top chunks to use as positive samples (answer-sufficient)
        top_m: Number of similar chunks to retrieve as hard negative candidates
        min_similarity: Minimum similarity threshold for hard negative chunks
        max_similarity: Maximum similarity threshold (to avoid too similar chunks)
        batch_size: Batch size for embedding computation
        device: Device for model inference
        filter_duplicates: Whether to filter duplicate chunks
        ensure_distinct: Whether to ensure positives and negatives are distinct
    """
    top_n: int = 3
    top_m: int = 10
    min_similarity: float = 0.3
    max_similarity: float = 0.95
    batch_size: int = 64
    device: str = "cuda"
    filter_duplicates: bool = True
    ensure_distinct: bool = True

    @classmethod
    def from_yaml(cls, config_path: str) -> "PreferenceConfig":
        """Load configuration from YAML file.

        Args:
            config_path: Path to YAML configuration file

        Returns:
            PreferenceConfig instance
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        alignment_config = config.get('training', {}).get('qwen', {}).get('alignment', {})

        return cls(
            top_n=alignment_config.get('top_n', 3),
            top_m=alignment_config.get('top_m', 10),
            min_similarity=alignment_config.get('min_similarity', 0.3),
            max_similarity=alignment_config.get('max_similarity', 0.95),
            batch_size=alignment_config.get('batch_size', 64),
            device=config.get('training', {}).get('common', {}).get('device', 'cuda'),
            filter_duplicates=alignment_config.get('filter_duplicates', True),
            ensure_distinct=alignment_config.get('ensure_distinct', True)
        )


class PreferenceGenerator:
    """Generate contrastive training pairs for answer-centric retriever training.

    This class takes scored chunks and generates contrastive pairs by:
    1. Using top-scored chunks as positive samples (answer-sufficient chunks)
    2. Finding similar but lower-scored chunks as hard negatives (insufficient chunks)
    3. Ensuring positive and negative samples are distinct

    These pairs are used for curriculum-based contrastive learning with InfoNCE loss.
    """

    def __init__(
        self,
        embedding_model_path: str,
        config: Optional[PreferenceConfig] = None,
        max_workers: Optional[int] = None
    ):
        """Initialize the preference generator.

        Args:
            embedding_model_path: Path to sentence embedding model
            config: PreferenceConfig instance. If None, uses default config
            max_workers: Maximum number of workers for parallel execution
        """
        self.config = config or PreferenceConfig()

        # Load embedding model
        print(f"Loading embedding model from {embedding_model_path}...")
        self.embedding_model = SentenceTransformer(
            embedding_model_path,
            device=self.config.device
        )

        # Initialize executor for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        print("PreferenceGenerator initialized successfully.")

    def _deduplicate_chunks(self, chunks: List[str]) -> List[str]:
        """Remove duplicate chunks while preserving order.

        Args:
            chunks: List of chunk strings

        Returns:
            Deduplicated list of chunks
        """
        seen = set()
        result = []
        for chunk in chunks:
            if chunk not in seen:
                seen.add(chunk)
                result.append(chunk)
        return result

    def _compute_similarity_batch(
        self,
        query: str,
        chunks: List[str],
        show_progress: bool = False
    ) -> List[float]:
        """Compute cosine similarity between query and chunks in batches.

        Args:
            query: Query string
            chunks: List of chunk strings
            show_progress: Whether to show progress bar

        Returns:
            List of similarity scores
        """
        # Encode query once
        query_embedding = self.embedding_model.encode(
            query,
            convert_to_tensor=True,
            show_progress_bar=False
        )

        all_similarities = []

        # Process chunks in batches
        iterator = range(0, len(chunks), self.config.batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Computing similarities", leave=False)

        for i in iterator:
            batch_chunks = chunks[i:i + self.config.batch_size]

            # Encode batch
            chunk_embeddings = self.embedding_model.encode(
                batch_chunks,
                convert_to_tensor=True,
                show_progress_bar=False
            )

            # Compute similarities
            similarities = util.cos_sim(query_embedding, chunk_embeddings)[0]
            all_similarities.extend(similarities.cpu().tolist())

        return all_similarities

    def generate_single_preference(
        self,
        original_query: str,
        new_query: str,
        chunk_list: List[str],
        score_list: Optional[List[float]] = None
    ) -> Optional[Dict[str, Any]]:
        """Generate a single contrastive training pair.

        Args:
            original_query: Original query text
            new_query: Augmented query text (for hard negative mining)
            chunk_list: List of chunks (assumed to be sorted by alignment score)
            score_list: Optional list of alignment scores corresponding to chunks

        Returns:
            Contrastive pair dictionary with keys:
            - query: Original query
            - response: List of positive chunks (answer-sufficient)
            - rejected_response: List of hard negative chunks (similar but insufficient)
            Returns None if no valid contrastive pair can be generated
        """
        if not chunk_list:
            return None

        # Deduplicate if configured
        if self.config.filter_duplicates:
            chunk_list = self._deduplicate_chunks(chunk_list)

        # Get chosen response (top-N chunks)
        chosen_chunks = chunk_list[:self.config.top_n]
        chosen_set = set(chosen_chunks)

        # Compute similarities between new query and all chunks
        similarities = self._compute_similarity_batch(new_query, chunk_list)

        # Pair chunks with similarities
        chunk_sim_pairs = list(zip(chunk_list, similarities))

        # Sort by similarity (descending)
        chunk_sim_pairs.sort(key=lambda x: x[1], reverse=True)

        # Select top-M similar chunks
        top_m_similar = chunk_sim_pairs[:self.config.top_m]

        # Filter based on similarity thresholds and ensure distinct from chosen
        rejected_candidates = [
            chunk for chunk, sim in top_m_similar
            if (self.config.min_similarity <= sim <= self.config.max_similarity and
                chunk not in chosen_set)
        ]

        # If no valid rejected chunks found, return None
        if not rejected_candidates:
            return None

        return {
            "query": original_query,
            "response": chosen_chunks,
            "rejected_response": rejected_candidates
        }

    def generate_from_scored_data(
        self,
        scored_data_path: str,
        new_queries_path: str,
        output_path: str,
        max_items: Optional[int] = None,
        show_progress: bool = True
    ) -> int:
        """Generate preference pairs from scored data and new queries.

        Args:
            scored_data_path: Path to directory with scored files OR single JSONL file
            new_queries_path: Path to JSON file with augmented queries
            output_path: Path to output JSONL file
            max_items: Maximum number of items to process
            show_progress: Whether to show progress bar

        Returns:
            Number of preference pairs generated
        """
        # Load new queries
        print(f"Loading new queries from {new_queries_path}...")
        with open(new_queries_path, 'r', encoding='utf-8') as f:
            new_queries_data = json.load(f)

        # Process scored data
        scored_items = []
        scored_path = Path(scored_data_path)

        if scored_path.is_dir():
            # Load from directory (multiple files)
            print(f"Loading scored data from directory {scored_data_path}...")
            score_files = sorted(scored_path.glob("alignment_score_*.jsonl"))

            for score_file in score_files:
                with jsonlines.open(score_file) as reader:
                    for item in reader:
                        scored_items.append(item)
                        if max_items is not None and len(scored_items) >= max_items:
                            break
                if max_items is not None and len(scored_items) >= max_items:
                    break
        else:
            # Load from single file (backward compatibility)
            print(f"Loading scored data from file {scored_data_path}...")
            with jsonlines.open(scored_data_path) as reader:
                scored_items = list(reader)
                if max_items is not None:
                    scored_items = scored_items[:max_items]

        print(f"Loaded {len(scored_items)} scored items.")
        results = []

        iterator = enumerate(scored_items)
        if show_progress:
            iterator = tqdm(iterator, total=len(scored_items), desc="Generating preference pairs")

        for idx, item in iterator:
            original_query = item['input']
            chunk_list = item['chunk_list']
            score_list = item.get('score_list')

            if not chunk_list:
                continue

            # Get corresponding new queries
            new_queries = new_queries_data.get(str(idx))
            if not new_queries:
                continue

            # Generate preference pair for each new query
            for new_query in new_queries:
                preference = self.generate_single_preference(
                    original_query=original_query,
                    new_query=new_query,
                    chunk_list=chunk_list,
                    score_list=score_list
                )

                if preference is not None:
                    results.append(preference)

        # Write results
        print(f"\nGenerated {len(results)} preference pairs.")
        print(f"Writing results to {output_path}...")
        with jsonlines.open(output_path, 'w') as writer:
            writer.write_all(results)

        return len(results)

    async def generate_single_preference_async(
        self,
        original_query: str,
        new_query: str,
        chunk_list: List[str],
        score_list: Optional[List[float]] = None
    ) -> Optional[Dict[str, Any]]:
        """Async version of generate_single_preference.

        Args:
            original_query: Original query text
            new_query: New augmented query text
            chunk_list: List of chunks
            score_list: Optional list of scores

        Returns:
            Preference pair dictionary or None
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.generate_single_preference,
            original_query,
            new_query,
            chunk_list,
            score_list
        )

    async def generate_batch_async(
        self,
        items: List[Tuple[str, str, List[str], Optional[List[float]]]],
        max_concurrent: int = 10
    ) -> List[Optional[Dict[str, Any]]]:
        """Generate preference pairs asynchronously with concurrency control.

        Args:
            items: List of tuples (original_query, new_query, chunk_list, score_list)
            max_concurrent: Maximum number of concurrent operations

        Returns:
            List of preference pairs (may contain None values)
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def generate_with_semaphore(item):
            async with semaphore:
                original_query, new_query, chunk_list, score_list = item
                return await self.generate_single_preference_async(
                    original_query, new_query, chunk_list, score_list
                )

        tasks = [generate_with_semaphore(item) for item in items]
        return await asyncio.gather(*tasks)

    def generate_from_alignment_scorer_output(
        self,
        scorer_output_path: str,
        augmented_queries: Dict[str, List[str]],
        output_path: str,
        max_items: Optional[int] = None
    ) -> int:
        """Generate preference pairs directly from AlignmentScorer output.

        This is a convenience method that combines scored chunks with augmented queries.

        Args:
            scorer_output_path: Path to directory with scored files OR single JSONL file
            augmented_queries: Dictionary mapping item index to list of augmented queries
            output_path: Path to output JSONL file
            max_items: Maximum number of items to process

        Returns:
            Number of preference pairs generated
        """
        results = []
        scored_path = Path(scorer_output_path)

        # Load scored items
        if scored_path.is_dir():
            # Load from directory (multiple files)
            print(f"Loading scored data from directory {scorer_output_path}...")
            score_files = sorted(scored_path.glob("alignment_score_*.jsonl"))
            items = []

            for score_file in score_files:
                with jsonlines.open(score_file) as reader:
                    for item in reader:
                        items.append(item)
                        if max_items is not None and len(items) >= max_items:
                            break
                if max_items is not None and len(items) >= max_items:
                    break
        else:
            # Load from single file (backward compatibility)
            with jsonlines.open(scorer_output_path) as reader:
                items = list(reader)
                if max_items is not None:
                    items = items[:max_items]

        for idx, item in enumerate(tqdm(items, desc="Generating preferences")):
            original_query = item['input']
            chunk_list = item['chunk_list']
            score_list = item.get('score_list')

            # Get augmented queries for this item
            new_queries = augmented_queries.get(str(idx), [])

            for new_query in new_queries:
                preference = self.generate_single_preference(
                    original_query=original_query,
                    new_query=new_query,
                    chunk_list=chunk_list,
                    score_list=score_list
                )

                if preference is not None:
                    results.append(preference)

        # Write results
        with jsonlines.open(output_path, 'w') as writer:
            writer.write_all(results)

        return len(results)

    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()


class PreferenceDataset:
    """Utility class for working with contrastive training pair datasets.

    NOTE: This class is currently not used in the ARK pipeline.
    It provides utility functions for validation and filtering if needed in the future.
    """

    @staticmethod
    def load(path: str) -> List[Dict[str, Any]]:
        """Load preference pairs from JSONL file.

        Args:
            path: Path to JSONL file

        Returns:
            List of preference pair dictionaries
        """
        with jsonlines.open(path) as reader:
            return list(reader)

    @staticmethod
    def save(data: List[Dict[str, Any]], path: str):
        """Save preference pairs to JSONL file.

        Args:
            data: List of preference pair dictionaries
            path: Output path
        """
        with jsonlines.open(path, 'w') as writer:
            writer.write_all(data)

    @staticmethod
    def validate(data: List[Dict[str, Any]]) -> Tuple[int, List[str]]:
        """Validate preference pair data.

        Args:
            data: List of preference pair dictionaries

        Returns:
            Tuple of (num_valid, list_of_error_messages)
        """
        errors = []
        num_valid = 0

        required_keys = {'query', 'response', 'rejected_response'}

        for idx, item in enumerate(data):
            # Check required keys
            missing_keys = required_keys - set(item.keys())
            if missing_keys:
                errors.append(f"Item {idx}: Missing keys {missing_keys}")
                continue

            # Check types
            if not isinstance(item['query'], str):
                errors.append(f"Item {idx}: 'query' must be string")
                continue

            if not isinstance(item['response'], list):
                errors.append(f"Item {idx}: 'response' must be list")
                continue

            if not isinstance(item['rejected_response'], list):
                errors.append(f"Item {idx}: 'rejected_response' must be list")
                continue

            # Check non-empty
            if not item['response']:
                errors.append(f"Item {idx}: 'response' is empty")
                continue

            if not item['rejected_response']:
                errors.append(f"Item {idx}: 'rejected_response' is empty")
                continue

            # Check distinct
            response_set = set(item['response'])
            rejected_set = set(item['rejected_response'])
            overlap = response_set & rejected_set

            if overlap:
                errors.append(f"Item {idx}: Overlap between response and rejected_response: {overlap}")
                continue

            num_valid += 1

        return num_valid, errors

    @staticmethod
    def filter_by_length(
        data: List[Dict[str, Any]],
        min_response_length: int = 1,
        max_response_length: Optional[int] = None,
        min_rejected_length: int = 1,
        max_rejected_length: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Filter preference pairs by response length.

        Args:
            data: List of preference pairs
            min_response_length: Minimum number of chunks in response
            max_response_length: Maximum number of chunks in response
            min_rejected_length: Minimum number of chunks in rejected_response
            max_rejected_length: Maximum number of chunks in rejected_response

        Returns:
            Filtered list of preference pairs
        """
        filtered = []

        for item in data:
            response_len = len(item['response'])
            rejected_len = len(item['rejected_response'])

            if response_len < min_response_length:
                continue

            if max_response_length and response_len > max_response_length:
                continue

            if rejected_len < min_rejected_length:
                continue

            if max_rejected_length and rejected_len > max_rejected_length:
                continue

            filtered.append(item)

        return filtered


def main():
    """Example usage of PreferenceGenerator."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate Preference Pairs")
    parser.add_argument('--scored_data', type=str, required=True,
                        help='Path to scored data directory OR JSONL file (from AlignmentScorer)')
    parser.add_argument('--new_queries', type=str, required=True,
                        help='Path to augmented queries JSON file')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to output preference pairs JSONL')
    parser.add_argument('--embedding_model', type=str, required=True,
                        help='Path to embedding model')
    parser.add_argument('--config', type=str, default='/home/jiawei/ARK/config.yaml',
                        help='Path to config.yaml')
    parser.add_argument('--top_n', type=int, default=3,
                        help='Number of top chunks for chosen response')
    parser.add_argument('--top_m', type=int, default=10,
                        help='Number of similar chunks to retrieve')
    parser.add_argument('--max_items', type=int, default=None,
                        help='Maximum number of items to process')
    parser.add_argument('--validate', action='store_true',
                        help='Validate output after generation')

    args = parser.parse_args()

    # Load or create config
    if Path(args.config).exists():
        config = PreferenceConfig.from_yaml(args.config)
    else:
        config = PreferenceConfig()

    # Override config with command-line arguments
    config.top_n = args.top_n
    config.top_m = args.top_m

    # Generate preference pairs
    with PreferenceGenerator(args.embedding_model, config) as generator:
        num_pairs = generator.generate_from_scored_data(
            scored_data_path=args.scored_data,
            new_queries_path=args.new_queries,
            output_path=args.output,
            max_items=args.max_items
        )

    print(f"\nGenerated {num_pairs} preference pairs.")

    # Validate if requested
    if args.validate:
        print("\nValidating preference pairs...")
        data = PreferenceDataset.load(args.output)
        num_valid, errors = PreferenceDataset.validate(data)

        print(f"Valid pairs: {num_valid}/{len(data)}")
        if errors:
            print(f"\nFound {len(errors)} errors:")
            for error in errors[:10]:  # Show first 10 errors
                print(f"  - {error}")
            if len(errors) > 10:
                print(f"  ... and {len(errors) - 10} more errors")

    print("\nDone!")


if __name__ == '__main__':
    main()
