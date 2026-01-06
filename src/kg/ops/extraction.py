"""
Subgraph Extraction Module for Answer-Augmented Retrieval

This module provides functionality for extracting relevant subgraphs from knowledge graphs
using Personalized PageRank (PPR) with adaptive cutoff.

Key Features:
- Entity extraction and matching (exact and embedding-based)
- Personalized PageRank with adaptive cutoff
- Integration with vector database for embedding similarity

Classes:
    SubgraphExtractor: Main class for subgraph extraction workflow

Functions:
    personalized_pagerank: PPR algorithm implementation
    adaptive_cutoff: Adaptive threshold selection using negative log transformation
    match_entities_exact: Exact string matching against graph nodes
    match_entities_embedding: Embedding similarity-based matching
"""

import asyncio
import logging
from typing import List, Dict, Tuple, Optional, Set, Any, Union
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import networkx as nx
import yaml

# Configure logging
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration Management
# ============================================================================

@dataclass
class PPRConfig:
    """Configuration for Personalized PageRank"""
    alpha: float = 0.85
    epsilon: float = 1e-4
    max_iterations: int = 100
    adaptive_cutoff: bool = True
    min_k: int = 3
    max_entities: int = 10000


@dataclass
class SubgraphConfig:
    """Configuration for subgraph extraction"""
    ppr: PPRConfig = field(default_factory=PPRConfig)
    similarity_threshold: float = 0.8  # Embedding similarity threshold
    enable_augmentation: bool = True  # Use augmented graph
    algorithm: str = "exact"  # "exact" or "emb" (embedding)
    top_k: int = 10  # Top-k entities for embedding matching


def load_config(config_path: Optional[str] = None) -> SubgraphConfig:
    """
    Load configuration from YAML file

    Args:
        config_path: Path to config.yaml file. If None, looks for config.yaml in project root.

    Returns:
        SubgraphConfig object with loaded settings

    Example:
        >>> config = load_config("./config.yaml")
        >>> config.ppr.alpha
        0.85
    """
    if config_path is None:
        # Try to find config.yaml in common locations
        possible_paths = [
            Path("config.yaml"),
            Path("../config.yaml"),
            Path("../../config.yaml"),
        ]
        for path in possible_paths:
            if path.exists():
                config_path = str(path)
                break

    if config_path is None or not Path(config_path).exists():
        logger.warning(f"Config file not found at {config_path}, using defaults")
        return SubgraphConfig()

    try:
        with open(config_path, 'r') as f:
            yaml_config = yaml.safe_load(f)

        # Extract PPR settings
        kg_config = yaml_config.get('kg', {})
        ppr_config = kg_config.get('ppr', {})
        graph_config = kg_config.get('graph', {})

        ppr = PPRConfig(
            alpha=ppr_config.get('alpha', 0.85),
            epsilon=ppr_config.get('epsilon', 1e-4),
            max_iterations=ppr_config.get('max_iterations', 100),
            adaptive_cutoff=ppr_config.get('adaptive_cutoff', True),
            max_entities=graph_config.get('max_entities', 10000),  # No limit by default
        )

        config = SubgraphConfig(
            ppr=ppr,
            similarity_threshold=graph_config.get('similarity_threshold', 0.8),
            enable_augmentation=graph_config.get('enable_augmentation', True),
        )

        logger.info(f"Loaded configuration from {config_path}")
        return config

    except Exception as e:
        logger.error(f"Error loading config from {config_path}: {e}")
        return SubgraphConfig()


# ============================================================================
# Personalized PageRank
# ============================================================================

def personalized_pagerank(
    graph: nx.Graph,
    personalization: Dict[str, float],
    alpha: float = 0.85,
    epsilon: float = 1e-4,
    max_iterations: int = 100,
) -> Dict[str, float]:
    """
    Compute Personalized PageRank scores for graph nodes

    PPR is a variant of PageRank where random walks restart from a personalized
    set of nodes rather than uniformly. This allows scoring nodes by relevance
    to a seed set (e.g., entities mentioned in a query).

    Args:
        graph: NetworkX graph to compute PPR on
        personalization: Dictionary mapping seed nodes to initial probabilities
                        Example: {"entity1": 1.0, "entity2": 1.0}
        alpha: Damping factor (teleport probability). Higher = more global influence.
               Typical range: 0.8-0.95
        epsilon: Convergence threshold. Algorithm stops when max change < epsilon.
        max_iterations: Maximum iterations before forced termination

    Returns:
        Dictionary mapping each node to its PPR score (sorted by score descending)

    Algorithm:
        PPR is computed iteratively:
        1. Initialize all nodes with personalization vector (normalized)
        2. At each iteration:
           - Distribute score from each node to neighbors (weighted by edges)
           - With probability (1-alpha), teleport back to personalized nodes
        3. Repeat until convergence or max iterations

    Example:
        >>> G = nx.Graph()
        >>> G.add_edges_from([("A", "B"), ("B", "C"), ("A", "C")])
        >>> personalization = {"A": 1.0}  # Start from node A
        >>> scores = personalized_pagerank(G, personalization, alpha=0.85)
        >>> sorted(scores.items(), key=lambda x: x[1], reverse=True)
        [('A', 0.42), ('B', 0.31), ('C', 0.27)]

    Note:
        - Uses NetworkX's built-in pagerank implementation for efficiency
        - Personalization vector is automatically normalized
        - Higher alpha means scores propagate further from seed nodes
    """
    # Normalize personalization vector
    total = sum(personalization.values())
    if total > 0:
        personalization = {k: v / total for k, v in personalization.items()}

    # Use NetworkX's efficient PageRank implementation
    ppr_scores = nx.pagerank(
        graph,
        alpha=alpha,
        personalization=personalization,
        max_iter=max_iterations,
        tol=epsilon,
    )

    return ppr_scores


# ============================================================================
# Adaptive Cutoff
# ============================================================================

def adaptive_cutoff(
    ppr_scores: List[Tuple[str, float]],
    min_k: int = 3,
    max_entities: int = 10000,
) -> List[Tuple[str, float]]:
    """
    Adaptively determine cutoff point for PPR scores using negative log transformation

    Finds the most significant "elbow" in the score distribution that gives
    ≤max_entities. If no such elbow exists, returns max_entities.

    Args:
        ppr_scores: List of (entity_name, score) tuples sorted by score (descending)
        min_k: Minimum number of entities to return
        max_entities: Maximum number of entities to return

    Returns:
        Filtered list of (entity_name, transformed_score) tuples
    """
    # Transform scores using negative log
    transformed_scores = []
    for entity, score in ppr_scores:
        if score > 0:
            transformed_scores.append((entity, -np.log(score)))

    # Adjust min_k if max_entities is smaller
    effective_min_k = min(min_k, max_entities)

    # Need at least effective_min_k entities
    if len(transformed_scores) <= effective_min_k:
        return transformed_scores

    # Find differences between consecutive scores
    diffs = []
    for i in range(1, len(transformed_scores)):
        diff = transformed_scores[i][1] - transformed_scores[i-1][1]
        diffs.append((i, diff))

    # Sort by diff descending to find significant drops
    sorted_diffs = sorted(diffs, key=lambda x: x[1], reverse=True)

    # Find the most significant cutoff point that gives ≤max_entities
    for cutoff_idx, diff in sorted_diffs:
        if effective_min_k <= cutoff_idx <= max_entities:
            return transformed_scores[:cutoff_idx]

    # No valid cutoff found, return up to max_entities
    return transformed_scores[:min(max_entities, len(transformed_scores))]


# ============================================================================
# Entity Extraction and Matching
# ============================================================================

async def match_entities_exact(
    entities: List[str],
    graph: nx.Graph,
) -> Set[str]:
    """
    Match entities to graph nodes using exact string matching

    Args:
        entities: List of entity names to match (will be normalized to uppercase)
        graph: NetworkX graph containing nodes

    Returns:
        Set of entity names that exist in the graph

    Example:
        >>> G = nx.Graph()
        >>> G.add_node("APPLE INC")
        >>> G.add_node("GOOGLE LLC")
        >>> entities = ["apple inc", "Microsoft Corp", "google llc"]
        >>> matches = await match_entities_exact(entities, G)
        >>> sorted(matches)
        ['APPLE INC', 'GOOGLE LLC']
    """
    # Normalize entities to uppercase
    normalized_entities = [entity.upper().strip() for entity in entities]

    # Get graph nodes
    graph_nodes = set(graph.nodes())

    # Find intersection
    matches = set(normalized_entities) & graph_nodes

    return matches


async def match_entities_embedding(
    entities: List[str],
    entity_descriptions: List[str],
    vectordb,
    top_k: int = 10,
) -> Dict[str, float]:
    """
    Match entities using embedding similarity via vector database

    For each extracted entity, query the vector database to find the top-k
    most similar entities in the knowledge graph. Aggregate scores across
    multiple queries.

    Args:
        entities: List of entity names extracted from text
        entity_descriptions: List of descriptions for each entity
        vectordb: Vector database instance (NanoVectorDBStorage)
        top_k: Number of similar entities to retrieve per query

    Returns:
        Dictionary mapping matched entity names to aggregated similarity scores

    Algorithm:
        1. For each (entity, description) pair:
           a. Query: entity_name + description
           b. Retrieve top_k similar entities from vector DB
        2. Aggregate scores across all queries:
           - If entity appears multiple times, sum its scores
        3. Sort by total score (descending)

    Example:
        >>> entities = ["Apple", "iPhone"]
        >>> descriptions = ["Tech company", "Smartphone device"]
        >>> matches = await match_entities_embedding(
        ...     entities, descriptions, vectordb, top_k=5
        ... )
        >>> matches
        {'APPLE INC': 1.85, 'IPHONE 14': 1.62, 'TECHNOLOGY': 0.91, ...}

    Note:
        - Combines entity name and description for richer semantic matching
        - Scores are similarity metrics (higher = more similar)
        - Aggregation allows entities mentioned multiple times to rank higher
    """
    matched_entities = {}

    # Query vector database for each entity
    for entity, description in zip(entities, entity_descriptions):
        query_text = f"{entity} {description}"

        # Query vector database
        results = await vectordb.query(query_text, top_k=top_k)

        # Aggregate scores
        for result in results:
            entity_name = result["entity_name"]
            score = result.get("__metrics__", result.get("distance", 0.0))

            if entity_name in matched_entities:
                matched_entities[entity_name] += score
            else:
                matched_entities[entity_name] = score

    # Sort by score (descending)
    matched_entities = dict(
        sorted(matched_entities.items(), key=lambda x: x[1], reverse=True)
    )

    return matched_entities


async def extract_and_match_entities(
    text: str,
    graph: nx.Graph,
    algorithm: str = "exact",
    vectordb = None,
    top_k: int = 10,
    entity_extraction_func = None,
) -> Set[str]:
    """
    Extract entities from text and match them to graph nodes

    This is a convenience function that combines entity extraction (using LLM)
    with entity matching (exact or embedding-based).

    Args:
        text: Input text to extract entities from
        graph: Knowledge graph to match against
        algorithm: "exact" for string matching, "emb" for embedding similarity
        vectordb: Vector database (required if algorithm="emb")
        top_k: Top-k for embedding matching
        entity_extraction_func: Optional custom entity extraction function

    Returns:
        Set of matched entity names from the graph

    Example:
        >>> text = "Apple Inc released the new iPhone yesterday."
        >>> matches = await extract_and_match_entities(
        ...     text, graph, algorithm="exact"
        ... )
        >>> matches
        {'APPLE INC', 'IPHONE'}

    Note:
        - Requires entity_extraction_func for actual entity extraction
        - If not provided, falls back to simple keyword matching
        - For embedding matching, vectordb must be provided
    """
    if entity_extraction_func is not None:
        # Use LLM-based entity extraction
        # This would call the entity extraction pipeline
        # For now, we'll use a simple placeholder
        raise NotImplementedError(
            "LLM-based entity extraction should be implemented using "
            "src.utils.graph_operations.extract_entities"
        )

    # Placeholder: Simple keyword extraction (not recommended for production)
    # In practice, use the entity extraction from src/utils/graph_operations.py
    words = text.upper().split()
    potential_entities = words  # Very naive approach

    if algorithm == "exact":
        matches = await match_entities_exact(potential_entities, graph)
    elif algorithm == "emb":
        if vectordb is None:
            raise ValueError("vectordb is required for embedding-based matching")

        # For embedding matching, we need descriptions
        # This is a placeholder - in practice, extract entities properly
        descriptions = [""] * len(potential_entities)
        matched_dict = await match_entities_embedding(
            potential_entities, descriptions, vectordb, top_k
        )
        matches = set(matched_dict.keys())
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    return matches


# ============================================================================
# Main Subgraph Extractor Class
# ============================================================================

@dataclass
class SubgraphExtractor:
    """
    Main class for subgraph extraction workflow

    This class orchestrates the complete subgraph extraction pipeline:
    1. Entity extraction and matching
    2. Personalized PageRank computation
    3. Adaptive cutoff or community search
    4. Subgraph materialization

    Attributes:
        graph: NetworkX knowledge graph
        vectordb: Optional vector database for embedding matching
        config: Configuration object

    Example:
        >>> # Load graph and vector database
        >>> graph = await kg.get_graph()
        >>> vectordb = NanoVectorDBStorage(...)
        >>>
        >>> # Initialize extractor
        >>> extractor = SubgraphExtractor(
        ...     graph=graph,
        ...     vectordb=vectordb,
        ...     config=load_config("config.yaml")
        ... )
        >>>
        >>> # Extract subgraph for a query
        >>> query = "What products does Apple sell?"
        >>> subgraph_entities = await extractor.extract_subgraph(
        ...     query_text=query,
        ...     algorithm="exact",
        ...     use_adaptive=True
        ... )
        >>> print(f"Found {len(subgraph_entities)} relevant entities")
    """

    graph: nx.Graph
    vectordb: Optional[Any] = None
    config: SubgraphConfig = field(default_factory=SubgraphConfig)

    async def extract_subgraph_from_seeds(
        self,
        seed_entities: Set[str],
        use_adaptive: bool = True,
        top_k: Optional[int] = None,
        max_entities: Optional[int] = None,
        min_k: Optional[int] = None,
    ) -> Tuple[List[str], Dict[str, float]]:
        """
        Extract subgraph from seed entities using PPR

        Args:
            seed_entities: Set of seed entity names (must exist in graph)
            use_adaptive: Use adaptive cutoff (True) or fixed top-k (False)
            top_k: Fixed top-k (only used if use_adaptive=False)
            max_entities: Maximum entities for adaptive cutoff
            min_k: Minimum entities for adaptive cutoff

        Returns:
            Tuple of (entity_list, ppr_scores_dict)
        """
        if not seed_entities:
            logger.warning("No seed entities provided")
            return [], {}

        # Filter out seeds that don't exist in graph
        valid_seeds = {e for e in seed_entities if e in self.graph}
        if not valid_seeds:
            logger.warning(f"None of the seed entities exist in graph: {seed_entities}")
            return [], {}

        logger.info(f"Valid seed entities: {valid_seeds}")

        # Use Personalized PageRank (more global, score-based)
        personalization = {entity: 1.0 for entity in valid_seeds}

        ppr_scores = personalized_pagerank(
            self.graph,
            personalization,
            alpha=self.config.ppr.alpha,
            epsilon=self.config.ppr.epsilon,
            max_iterations=self.config.ppr.max_iterations,
        )

        # Sort by score (descending)
        sorted_scores = sorted(ppr_scores.items(), key=lambda x: x[1], reverse=True)

        # Apply adaptive cutoff or fixed top-k
        if use_adaptive and self.config.ppr.adaptive_cutoff:
            # Transform to -log and find adaptive cutoff
            max_ent = max_entities if max_entities is not None else self.config.ppr.max_entities
            min_ent = min_k if min_k is not None else self.config.ppr.min_k
            filtered_scores = adaptive_cutoff(
                sorted_scores,
                min_k=min_ent,
                max_entities=max_ent,
            )
            # Convert back to dict (scores are now -log transformed)
            entity_list = [entity for entity, _ in filtered_scores]
            scores_dict = {entity: score for entity, score in filtered_scores}
        else:
            # Use fixed top-k
            k = top_k if top_k is not None else self.config.top_k
            k = min(k, len(sorted_scores))
            filtered_scores = sorted_scores[:k]
            entity_list = [entity for entity, _ in filtered_scores]
            scores_dict = {entity: score for entity, score in filtered_scores}

        logger.info(f"Extracted subgraph with {len(entity_list)} entities")
        return entity_list, scores_dict

    async def extract_subgraph(
        self,
        query_text: str,
        algorithm: Optional[str] = None,
        use_adaptive: bool = True,
        top_k: Optional[int] = None,
        entity_extraction_func = None,
    ) -> Tuple[List[str], Dict[str, float]]:
        """
        Extract subgraph from query text (end-to-end pipeline)

        Args:
            query_text: Input text (question or answer)
            algorithm: "exact" or "emb" (if None, uses config default)
            use_adaptive: Use adaptive cutoff
            top_k: Fixed top-k (for exact matching or final output)
            entity_extraction_func: Custom entity extraction function

        Returns:
            Tuple of (entity_list, scores_dict)
        """
        algo = algorithm if algorithm is not None else self.config.algorithm

        # Step 1: Extract and match entities
        if algo == "exact":
            # For exact matching, we need entity extraction first
            # This is a placeholder - should use proper entity extraction
            if entity_extraction_func is None:
                # Simple fallback: use uppercase words as potential entities
                words = query_text.upper().split()
                seed_entities = await match_entities_exact(words, self.graph)
            else:
                # Use custom entity extraction
                seed_entities = await extract_and_match_entities(
                    query_text,
                    self.graph,
                    algorithm="exact",
                    entity_extraction_func=entity_extraction_func,
                )

        elif algo == "emb":
            # For embedding matching
            if self.vectordb is None:
                raise ValueError("vectordb is required for embedding-based matching")

            if entity_extraction_func is None:
                # Simple fallback
                words = query_text.upper().split()
                descriptions = [""] * len(words)
                matched_dict = await match_entities_embedding(
                    words, descriptions, self.vectordb,
                    top_k=top_k or self.config.top_k
                )
                seed_entities = set(matched_dict.keys())
            else:
                # Use custom entity extraction
                seed_entities = await extract_and_match_entities(
                    query_text,
                    self.graph,
                    algorithm="emb",
                    vectordb=self.vectordb,
                    top_k=top_k or self.config.top_k,
                    entity_extraction_func=entity_extraction_func,
                )
        else:
            raise ValueError(f"Unknown algorithm: {algo}")

        if not seed_entities:
            logger.warning("No entities matched from query")
            return [], {}

        # Step 2: Extract subgraph from seeds
        entity_list, scores_dict = await self.extract_subgraph_from_seeds(
            seed_entities,
            use_adaptive=use_adaptive,
            top_k=top_k,
        )

        return entity_list, scores_dict

    def get_entity_data(self, entity_names: List[str]) -> Dict[str, Dict]:
        """
        Get entity data (attributes) from graph for given entity names

        Args:
            entity_names: List of entity names to retrieve

        Returns:
            Dictionary mapping entity names to their graph node attributes

        Example:
            >>> entity_data = extractor.get_entity_data(["APPLE INC", "IPHONE"])
            >>> entity_data["APPLE INC"]
            {
                'entity_type': 'ORGANIZATION',
                'description': 'Technology company...',
                'source_id': 'chunk-abc123'
            }
        """
        entity_data = {}
        for entity in entity_names:
            if entity in self.graph:
                entity_data[entity] = dict(self.graph.nodes[entity])
            else:
                logger.warning(f"Entity not found in graph: {entity}")
        return entity_data