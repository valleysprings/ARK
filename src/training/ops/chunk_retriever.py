"""
Chunk retrieval operations for Stage 2 & 3 negative sampling.
"""

import numpy as np
from typing import List, Set
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)


class ChunkRetriever:
    """Retrieves chunks using semantic similarity with queries."""

    def __init__(
        self,
        model_name_or_path: str,
        device: str = "cuda",
        batch_size: int = 64
    ):
        """
        Initialize the chunk retriever.

        Args:
            model_name_or_path: Path to the embedding model
            device: Device to use for encoding
            batch_size: Batch size for encoding
        """
        self.device = device
        self.batch_size = batch_size

        logger.info(f"Loading model: {model_name_or_path}")
        self.model = SentenceTransformer(
            model_name_or_path,
            device=device,
            trust_remote_code=True
        )

        self.corpus_chunks = []
        self.corpus_embeddings = None

    def build_corpus(self, alignment_data: List[dict]):
        """
        Build chunk corpus from alignment data.

        Args:
            alignment_data: List of alignment data dicts with 'chunk_list'
        """
        logger.info("Building chunk corpus from alignment data...")

        # Collect all unique chunks
        chunk_set = set()
        for sample in alignment_data:
            chunks = sample.get('chunk_list', [])
            chunk_set.update(chunks)

        self.build_corpus_from_chunks(list(chunk_set))

    def build_corpus_from_chunks(self, chunks: List[str]):
        """
        Build chunk corpus from a list of chunks.

        Args:
            chunks: List of chunk strings
        """
        self.corpus_chunks = list(set(chunks))  # deduplicate
        logger.info(f"Corpus built with {len(self.corpus_chunks)} unique chunks")

        # Encode corpus
        self.corpus_embeddings = self.model.encode(
            self.corpus_chunks,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            device=self.device
        )

    def retrieve_for_queries(
        self,
        queries: List[str],
        top_k: int = 100
    ) -> Set[str]:
        """
        Retrieve top-k chunks for queries and return their union.

        Args:
            queries: List of query strings
            top_k: Number of top chunks to retrieve per query

        Returns:
            Set of retrieved chunk strings (union across all queries)
        """
        if self.corpus_embeddings is None:
            raise ValueError("Corpus not built. Call build_corpus() first.")

        # Encode queries
        query_embeddings = self.model.encode(
            queries,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            device=self.device
        )

        # Compute similarities
        similarities = np.dot(query_embeddings, self.corpus_embeddings.T)

        # Get top-k for each query and take union
        retrieved_chunks = set()
        for sim_scores in similarities:
            top_indices = np.argsort(sim_scores)[-top_k:]
            for idx in top_indices:
                retrieved_chunks.add(self.corpus_chunks[idx])

        return retrieved_chunks
