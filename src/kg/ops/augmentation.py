"""
Knowledge Graph Augmentation Module

This module provides functionality for augmenting knowledge graphs with similarity-based edges
using entity embeddings. It enhances the graph by connecting semantically similar entities that
may not have explicit relationships in the source text.

Key Components:
    - GraphAugmentor: Main class that orchestrates the graph augmentation pipeline
    - compute_entity_embeddings(): Generates entity embeddings using vector database
    - augment_graph_with_similarity(): Adds similarity-based edges to the graph
    - Helper functions for tuple generation and edge creation

The augmentation process follows these steps:
    1. Extract entity data from the knowledge graph
    2. Generate embeddings for each entity (name + description)
    3. Compute pairwise similarity matrix
    4. Apply threshold to filter weak similarities
    5. Add new edges to the graph with similarity metadata

Usage:
    >>> from src.kg.augmentation import GraphAugmentor
    >>> from src.kg.core.nx_graph import nx_graph
    >>>
    >>> # Initialize augmentor
    >>> augmentor = GraphAugmentor(config_path="config.yaml")
    >>>
    >>> # Augment graph
    >>> augmented_kg = await augmentor.augment(
    ...     kg=knowledge_graph,
    ...     entities_data=all_entities_data,
    ...     threshold=0.8
    ... )
"""

import os
import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass

import yaml
import numpy as np

# Import from src.kg.utils
from src.kg.utils.embeddings import ollama_embedding, create_local_embedding
from src.kg.utils.text_processing import compute_mdhash_id

# Import from src.kg
from src.kg.core.nx_graph import nx_graph
from src.kg.indexing.vector_index import NanoVectorDBStorage

# Configure logger
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class AugmentationConfig:
    """
    Configuration for graph augmentation

    Attributes:
        similarity_threshold: Minimum cosine similarity to create edge (0.0-1.0)
        enable_augmentation: Whether to enable graph augmentation
        embedding_batch_size: Batch size for embedding generation
        vectordb_namespace: Namespace for vector database
        working_dir: Base directory for storing vector database files
        dataset_name: Dataset name for organizing files
        entry_id: Entry ID for per-entry storage
    """
    similarity_threshold: float = 0.8
    enable_augmentation: bool = True
    embedding_batch_size: int = 128
    vectordb_namespace: str = "entities"
    working_dir: str = "./data/preprocessed"
    dataset_name: str = "default"
    entry_id: str = None

    def get_entity_embeddings_dir(self) -> str:
        """Get the directory for entity embeddings based on dataset name."""
        return f"{self.working_dir}/{self.dataset_name}/entity_embeddings"

    def get_entity_embeddings_file(self) -> str:
        """Get the file path for entity embeddings based on entry_id."""
        base_dir = self.get_entity_embeddings_dir()
        return f"{base_dir}/{self.entry_id}.json"

    @classmethod
    def from_config(cls, config_path: str) -> "AugmentationConfig":
        """
        Load configuration from YAML file

        Args:
            config_path: Path to config.yaml file

        Returns:
            AugmentationConfig instance

        Example:
            >>> config = AugmentationConfig.from_config("config.yaml")
            >>> config.similarity_threshold
            0.8
        """
        config_path = Path(config_path)
        if not config_path.exists():
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return cls()

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        return cls.from_dict(config)

    @classmethod
    def from_dict(cls, config: Dict) -> "AugmentationConfig":
        """
        Create AugmentationConfig from configuration dictionary

        Args:
            config: Configuration dictionary (from YAML or other source)

        Returns:
            AugmentationConfig instance
        """
        graph_config = config['kg']['graph']

        return cls(
            similarity_threshold=graph_config['similarity_threshold'],
            enable_augmentation=graph_config['enable_augmentation'],
            embedding_batch_size=graph_config['embedding_batch_size'],
            vectordb_namespace="entities",
            working_dir=config.get('cache_root', './data/preprocessed'),
            dataset_name=config.get('dataset_name', 'default'),
            entry_id=config.get('entry_id', None)
        )


# ============================================================================
# Entity Embedding Functions
# ============================================================================

async def compute_entity_embeddings(
    entities_data: List[Dict[str, Any]],
    embedding_func: Any = None,
    namespace: str = "entities",
    working_dir: str = "./data/preprocessed/entity_embeddings",
    batch_size: int = 128
) -> NanoVectorDBStorage:
    """
    Generate entity embeddings and store in vector database

    This function creates a vector database storage with embeddings for all entities
    in the knowledge graph. Each entity is represented by its name and description
    concatenated together.

    Args:
        entities_data: List of entity dictionaries from graph generation
            Expected format: [{"entity_name": str, "description": str, ...}, ...]
        embedding_func: Embedding function to use (defaults to ollama_embedding)
        namespace: Namespace for vector database
        working_dir: Directory to store vector database files
        batch_size: Maximum batch size for embedding generation

    Returns:
        NanoVectorDBStorage instance with entity embeddings

    Example:
        >>> entities = [
        ...     {"entity_name": "APPLE INC.", "description": "Technology company"},
        ...     {"entity_name": "GOOGLE", "description": "Search engine company"}
        ... ]
        >>> vectorizer = await compute_entity_embeddings(entities)
        >>> similarity_matrix = vectorizer.get_similarity_matrix()

    Note:
        - Creates unique IDs using MD5 hash with "ent-" prefix
        - Combines entity name and description for richer embeddings
        - Results are cached in working_dir for future use
        - Embeddings are generated in batches to avoid memory issues
    """
    if embedding_func is None:
        embedding_func = ollama_embedding

    logger.info(f"Computing embeddings for {len(entities_data)} entities")

    # Create working directory if it doesn't exist
    os.makedirs(working_dir, exist_ok=True)

    # Initialize vector database storage
    vectorizer = NanoVectorDBStorage(
        embedding_func=embedding_func,
        namespace=namespace,
        working_dir=working_dir,
        _max_batch_size=batch_size
    )

    # Prepare data for vector database
    # Format: {unique_id: {"content": str, "entity_name": str}}
    data_for_vdb = {
        compute_mdhash_id(entity["entity_name"], prefix="ent-"): {
            "content": entity["entity_name"] + " " + entity.get("description", ""),
            "entity_name": entity["entity_name"],
        }
        for entity in entities_data
    }

    logger.info(f"Inserting {len(data_for_vdb)} entities into vector database")

    # Upsert entities to vector database (generates embeddings)
    await vectorizer.upsert(data_for_vdb)

    logger.info(f"Entity embeddings computed and stored in {working_dir}")

    return vectorizer


# ============================================================================
# Similarity-based Edge Generation
# ============================================================================

def _generate_similarity_tuples(
    entity_names: List[str],
    similarity_matrix: np.ndarray,
    threshold: float = 0.8
) -> List[Tuple[str, str]]:
    """
    Generate edge tuples from similarity matrix for graph augmentation

    Creates edges between entities based on embedding similarity above threshold.
    Only generates edges for the upper triangle of the similarity matrix to avoid
    duplicates in undirected graphs.

    Args:
        entity_names: List of entity names (order matches similarity matrix)
        similarity_matrix: Pairwise similarity matrix (NxN)
        threshold: Minimum similarity to create edge (0.0-1.0)

    Returns:
        List of (entity_i, entity_j) tuples for edges to add

    Example:
        >>> entity_names = ["COMPANY_A", "COMPANY_B", "PRODUCT_X"]
        >>> similarity_matrix = np.array([
        ...     [1.0, 0.85, 0.3],
        ...     [0.85, 1.0, 0.4],
        ...     [0.3, 0.4, 1.0]
        ... ])
        >>> tuples = _generate_similarity_tuples(entity_names, similarity_matrix, threshold=0.8)
        >>> tuples
        [('COMPANY_A', 'COMPANY_B')]

    Note:
        - Diagonal values (self-similarity) are ignored
        - Only upper triangle is processed to avoid duplicate edges
        - Returns empty list if no similarities exceed threshold
    """
    all_tuples = []

    for i in range(similarity_matrix.shape[0]):
        for j in range(i + 1, similarity_matrix.shape[1]):
            if similarity_matrix[i, j] >= threshold:
                all_tuples.append((entity_names[i], entity_names[j]))

    logger.debug(f"Generated {len(all_tuples)} similarity-based edge tuples")

    return all_tuples


async def _add_similarity_edges(
    kg: nx_graph,
    edge_tuples: List[Tuple[str, str]],
    similarity_matrix: np.ndarray,
    entity_names: List[str],
    threshold: float
) -> int:
    """
    Add similarity-based edges to the knowledge graph

    Augments the graph with edges based on entity embedding similarity.
    These edges represent semantic similarity rather than explicit relationships
    mentioned in the text.

    Args:
        kg: Knowledge graph instance (nx_graph)
        edge_tuples: List of (entity_i, entity_j) tuples to add as edges
        similarity_matrix: Full similarity matrix for weight lookup
        entity_names: List of entity names (for indexing)
        threshold: Similarity threshold used (for metadata)

    Returns:
        Number of edges added

    Note:
        - New edges have order=0 (highest priority in some ranking algorithms)
        - Weight is the cosine similarity score (0.0-1.0)
        - Description indicates these are embedding-based similarity edges
        - source_id is set to concatenation of both entity names
    """
    edges_added = 0

    for entity_i, entity_j in edge_tuples:
        # Find indices in entity_names list
        i = entity_names.index(entity_i)
        j = entity_names.index(entity_j)

        # Get similarity score
        similarity = float(similarity_matrix[i, j])

        # Create edge data
        edge_data = {
            "weight": similarity,
            "description": f"based on embedding similarity with threshold {threshold}",
            "source_id": f"{entity_i}<SEP>{entity_j}",
            "order": 0  # Highest priority
        }

        # Add edge to graph
        await kg.upsert_edge(entity_i, entity_j, edge_data=edge_data)
        edges_added += 1

    logger.info(f"Added {edges_added} similarity-based edges to the graph")

    return edges_added


async def augment_graph_with_similarity(
    kg: nx_graph,
    vectorizer: NanoVectorDBStorage,
    threshold: float = 0.8
) -> Tuple[nx_graph, int]:
    """
    Augment knowledge graph with similarity-based edges

    This is the main function for graph augmentation. It computes the similarity
    matrix from entity embeddings, applies threshold cutoff, generates edge tuples,
    and adds them to the graph.

    Args:
        kg: Knowledge graph to augment (nx_graph instance)
        vectorizer: Vector database with entity embeddings
        threshold: Minimum cosine similarity to create edge (0.0-1.0)

    Returns:
        Tuple of (augmented_graph, num_edges_added)

    Example:
        >>> # After building KG and computing embeddings
        >>> augmented_kg, num_edges = await augment_graph_with_similarity(
        ...     kg=knowledge_graph,
        ...     vectorizer=entity_vectorizer,
        ...     threshold=0.8
        ... )
        >>> print(f"Added {num_edges} similarity edges")

    Process:
        1. Compute similarity matrix from embeddings (cosine similarity)
        2. Apply threshold cutoff (set values below threshold to 0)
        3. Generate edge tuples from filtered matrix
        4. Add edges to graph with similarity metadata

    Note:
        - Original graph is modified in-place
        - Returns the same graph instance (for chaining)
        - Similarity matrix is NxN where N is number of entities
        - Higher threshold = fewer, more confident edges
    """
    logger.info(f"Starting graph augmentation with threshold={threshold}")

    # Get all entity data from vector database
    all_entities = vectorizer.get_all_data()
    entity_names = [entity["entity_name"] for entity in all_entities]

    logger.info(f"Processing {len(entity_names)} entities for augmentation")

    # Compute similarity matrix
    similarity_matrix = vectorizer.get_similarity_matrix()
    logger.info(f"Computed similarity matrix with shape {similarity_matrix.shape}")

    # Apply threshold cutoff
    cut_off_matrix = vectorizer.cut_off_similarity_matrix(
        similarity_matrix,
        threshold=threshold
    )

    # Count how many similarities exceed threshold
    num_similarities = np.sum(cut_off_matrix > 0) // 2  # Divide by 2 for symmetric matrix
    logger.info(f"Found {num_similarities} entity pairs above threshold {threshold}")

    # Generate edge tuples
    edge_tuples = _generate_similarity_tuples(
        entity_names,
        cut_off_matrix,
        threshold=threshold
    )

    if not edge_tuples:
        logger.warning("No entity pairs exceed similarity threshold, no edges added")
        return kg, 0

    # Add edges to graph
    num_edges_added = await _add_similarity_edges(
        kg=kg,
        edge_tuples=edge_tuples,
        similarity_matrix=cut_off_matrix,
        entity_names=entity_names,
        threshold=threshold
    )

    # Log graph statistics
    all_nodes = await kg.get_all_nodes()
    all_edges = await kg.get_all_edges()
    logger.info(f"Graph augmentation complete: {len(all_nodes)} nodes, {len(all_edges)} edges")

    return kg, num_edges_added


# ============================================================================
# GraphAugmentor Class
# ============================================================================

class GraphAugmentor:
    """
    Main class for orchestrating knowledge graph augmentation

    This class provides a high-level interface for the complete graph augmentation
    pipeline, including configuration loading, embedding generation, and edge creation.

    Attributes:
        config: AugmentationConfig instance
        embedding_func: Embedding function to use
        vectorizer: Optional cached NanoVectorDBStorage instance

    Example:
        >>> # Initialize with config file
        >>> augmentor = GraphAugmentor(config_path="config.yaml")
        >>>
        >>> # Augment graph
        >>> augmented_kg, stats = await augmentor.augment(
        ...     kg=knowledge_graph,
        ...     entities_data=all_entities_data
        ... )
        >>>
        >>> print(f"Added {stats['edges_added']} edges")
        >>> print(f"Final graph: {stats['total_nodes']} nodes, {stats['total_edges']} edges")
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        config: Optional[AugmentationConfig] = None,
        embedding_func: Any = None,
        embedding_provider: Optional[str] = None,
        embedding_model_path: Optional[str] = None,
    ):
        """
        Initialize GraphAugmentor

        Args:
            config_path: Path to config.yaml file (overrides config parameter)
            config: AugmentationConfig instance (used if config_path not provided)
            embedding_func: Custom embedding function (overrides provider)
            embedding_provider: "ollama" | "bge" | "qwen" (default: ollama)
            embedding_model_path: Model path for bge/qwen provider
        """
        if config_path is not None:
            self.config = AugmentationConfig.from_config(config_path)
        elif config is not None:
            if isinstance(config, dict):
                self.config = AugmentationConfig.from_dict(config)
            else:
                self.config = config
        else:
            self.config = AugmentationConfig()

        if embedding_func is not None:
            self.embedding_func = embedding_func
        elif embedding_provider in ("bge", "qwen"):
            defaults = {"bge": "model/raw/bge-m3", "qwen": "model/raw/qwen3"}
            path = embedding_model_path or defaults[embedding_provider]
            self.embedding_func = create_local_embedding(path, embedding_provider)
        else:
            self.embedding_func = ollama_embedding
        self.vectorizer: Optional[NanoVectorDBStorage] = None

        logger.info(f"GraphAugmentor initialized with threshold={self.config.similarity_threshold}")

    async def compute_embeddings(
        self,
        entities_data: List[Dict[str, Any]],
        namespace: Optional[str] = None,
        force_recompute: bool = False
    ) -> NanoVectorDBStorage:
        """
        Compute or retrieve entity embeddings

        Args:
            entities_data: List of entity dictionaries
            namespace: Custom namespace (overrides config)
            force_recompute: If True, recompute even if cached

        Returns:
            NanoVectorDBStorage instance with embeddings
        """
        # Use entry_id as namespace if available
        if self.config.entry_id:
            namespace = self.config.entry_id
        else:
            namespace = namespace or self.config.vectordb_namespace

        # Check if we already have a vectorizer and don't need to recompute
        if self.vectorizer is not None and not force_recompute:
            logger.info("Using cached vectorizer")
            return self.vectorizer

        # Use dataset-specific directory
        working_dir = self.config.get_entity_embeddings_dir()

        # Compute embeddings
        self.vectorizer = await compute_entity_embeddings(
            entities_data=entities_data,
            embedding_func=self.embedding_func,
            namespace=namespace,
            working_dir=working_dir,
            batch_size=self.config.embedding_batch_size
        )

        return self.vectorizer

    async def augment(
        self,
        kg: nx_graph,
        entities_data: List[Dict[str, Any]],
        threshold: Optional[float] = None,
        namespace: Optional[str] = None
    ) -> Tuple[nx_graph, Dict[str, Any]]:
        """
        Augment knowledge graph with similarity-based edges

        This is the main entry point for graph augmentation. It handles the complete
        pipeline: embedding computation, similarity calculation, and edge addition.

        Args:
            kg: Knowledge graph to augment
            entities_data: List of entity dictionaries
            threshold: Custom similarity threshold (overrides config)
            namespace: Custom namespace for vector database

        Returns:
            Tuple of (augmented_graph, statistics_dict)

        Statistics Dictionary:
            {
                "edges_added": int,
                "total_nodes": int,
                "total_edges": int,
                "threshold": float,
                "num_entities": int
            }

        Example:
            >>> augmentor = GraphAugmentor(config_path="config.yaml")
            >>> kg, stats = await augmentor.augment(
            ...     kg=knowledge_graph,
            ...     entities_data=all_entities_data,
            ...     threshold=0.85
            ... )
        """
        if not self.config.enable_augmentation:
            logger.warning("Graph augmentation is disabled in config")
            all_nodes = await kg.get_all_nodes()
            all_edges = await kg.get_all_edges()
            return kg, {
                "edges_added": 0,
                "total_nodes": len(all_nodes),
                "total_edges": len(all_edges),
                "threshold": 0.0,
                "num_entities": len(entities_data)
            }

        threshold = threshold or self.config.similarity_threshold

        logger.info(f"Starting graph augmentation for {len(entities_data)} entities")

        # Step 1: Compute embeddings
        vectorizer = await self.compute_embeddings(
            entities_data=entities_data,
            namespace=namespace
        )

        # Step 2: Augment graph with similarity edges
        augmented_kg, edges_added = await augment_graph_with_similarity(
            kg=kg,
            vectorizer=vectorizer,
            threshold=threshold
        )

        # Step 3: Collect statistics
        all_nodes = await augmented_kg.get_all_nodes()
        all_edges = await augmented_kg.get_all_edges()

        stats = {
            "edges_added": edges_added,
            "total_nodes": len(all_nodes),
            "total_edges": len(all_edges),
            "threshold": threshold,
            "num_entities": len(entities_data)
        }

        logger.info(f"Augmentation complete: {stats}")

        return augmented_kg, stats

    def get_vectorizer(self) -> Optional[NanoVectorDBStorage]:
        """
        Get the cached vectorizer instance

        Returns:
            NanoVectorDBStorage instance or None if not computed yet
        """
        return self.vectorizer

    def set_threshold(self, threshold: float) -> None:
        """
        Update similarity threshold

        Args:
            threshold: New similarity threshold (0.0-1.0)
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be between 0.0 and 1.0, got {threshold}")

        self.config.similarity_threshold = threshold
        logger.info(f"Similarity threshold updated to {threshold}")


# ============================================================================
# Convenience Functions
# ============================================================================

async def quick_augment(
    kg: nx_graph,
    entities_data: List[Dict[str, Any]],
    threshold: float = 0.8,
    config_path: Optional[str] = None
) -> Tuple[nx_graph, Dict[str, Any]]:
    """
    Quick one-liner for graph augmentation

    Convenience function that creates a GraphAugmentor instance and
    performs augmentation in one call.

    Args:
        kg: Knowledge graph to augment
        entities_data: List of entity dictionaries
        threshold: Similarity threshold
        config_path: Optional path to config.yaml

    Returns:
        Tuple of (augmented_graph, statistics_dict)

    Example:
        >>> kg, stats = await quick_augment(
        ...     kg=knowledge_graph,
        ...     entities_data=all_entities_data,
        ...     threshold=0.8
        ... )
    """
    augmentor = GraphAugmentor(config_path=config_path)
    return await augmentor.augment(kg, entities_data, threshold=threshold)


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    # Main class
    "GraphAugmentor",

    # Core functions
    "compute_entity_embeddings",
    "augment_graph_with_similarity",

    # Configuration
    "AugmentationConfig",

    # Convenience
    "quick_augment",
]
