"""
Knowledge Graph Construction Module

This module provides the core functionality for building knowledge graphs from text documents.
It handles entity extraction, relationship identification, graph construction, and entity merging.

Key Components:
    - KGBuilder: Main class that orchestrates the complete KG construction pipeline
    - extract_entities_and_relations(): Extracts entities and relationships from text chunks
    - build_knowledge_graph(): Constructs a NetworkX graph from extracted entities and relationships
"""

import re
import asyncio
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from collections import Counter, defaultdict
from pathlib import Path

import yaml

# Import from src.kg.utils
from src.kg.utils.text_processing import (
    split_string_by_multi_markers,
    clean_str,
    is_float_regex,
    encode_string_by_tiktoken,
    decode_tokens_by_tiktoken,
    compute_mdhash_id,
    doc_to_chunks,
)
from src.kg.utils.llm_client import google_model, gpt_model, deepseek_model
from src.kg.utils.file_operations import (
    cache_exists,
    load_cache,
    save_cache,
    get_cache_path,
)

# Import from src.kg
from src.kg.core.nx_graph import nx_graph
from src.kg.prompts.entity_extraction import PROMPTS, GRAPH_FIELD_SEP

# Configure logger
logger = logging.getLogger(__name__)


# ============================================================================
# Entity and Relationship Extraction
# ============================================================================

async def _handle_single_entity_extraction(
    record_attributes: List[str],
    chunk_key: str,
) -> Optional[Dict[str, Any]]:
    """
    Parse and validate a single entity record from LLM output

    This function processes raw entity attributes extracted from LLM responses
    and validates them according to the expected schema.

    Args:
        record_attributes: Parsed attributes from LLM response
            Expected format: ['"entity"', 'entity_name', 'entity_type', 'description', ...]
        chunk_key: Source chunk identifier (hash-based unique ID)

    Returns:
        Entity dictionary with normalized fields, or None if invalid

        Entity Schema:
            {
                "entity_name": str (uppercase, normalized),
                "entity_type": str (uppercase),
                "description": str (cleaned),
                "source_id": str (chunk_key)
            }

    Example:
        >>> attrs = ['"entity"', 'Apple Inc.', 'ORGANIZATION', 'A technology company']
        >>> entity = await _handle_single_entity_extraction(attrs, 'chunk-abc123')
        >>> entity['entity_name']
        'APPLE INC.'
    """
    # Validate minimum required fields
    if len(record_attributes) < 4 or record_attributes[0] != '"entity"':
        return None

    # Extract and normalize entity name
    entity_name = clean_str(record_attributes[1].upper())
    if not entity_name.strip():
        return None

    # Extract other fields
    entity_type = clean_str(record_attributes[2].upper())
    entity_description = clean_str(record_attributes[3])
    entity_source_id = chunk_key

    return dict(
        entity_name=entity_name,
        entity_type=entity_type,
        description=entity_description,
        source_id=entity_source_id,
    )


async def _handle_single_relationship_extraction(
    record_attributes: List[str],
    chunk_key: str,
) -> Optional[Dict[str, Any]]:
    """
    Parse and validate a single relationship record from LLM output

    This function processes raw relationship attributes extracted from LLM responses
    and validates them according to the expected schema.

    Args:
        record_attributes: Parsed attributes from LLM response
            Expected format: ['"relationship"', 'source', 'target', 'description', 'weight']
        chunk_key: Source chunk identifier

    Returns:
        Relationship dictionary with normalized fields, or None if invalid

        Relationship Schema:
            {
                "src_id": str (uppercase, normalized),
                "tgt_id": str (uppercase, normalized),
                "weight": float (1.0-10.0),
                "description": str (cleaned),
                "source_id": str (chunk_key)
            }

    Example:
        >>> attrs = ['"relationship"', 'Apple', 'iPhone', 'manufactures', '9']
        >>> rel = await _handle_single_relationship_extraction(attrs, 'chunk-abc123')
        >>> rel['weight']
        9.0
    """
    # Validate minimum required fields
    if len(record_attributes) < 5 or record_attributes[0] != '"relationship"':
        return None

    # Extract and normalize entity names
    source = clean_str(record_attributes[1].upper())
    target = clean_str(record_attributes[2].upper())
    edge_description = clean_str(record_attributes[3])
    edge_source_id = chunk_key

    # Parse weight (default to 1.0 if not a valid float)
    weight = (
        float(record_attributes[-1]) if is_float_regex(record_attributes[-1]) else 1.0
    )

    return dict(
        src_id=source,
        tgt_id=target,
        weight=weight,
        description=edge_description,
        source_id=edge_source_id,
    )


async def extract_entities_and_relations(
    chunks: Dict[str, Dict],
    llm_func: callable = google_model,
    show_progress: bool = True,
) -> List[Tuple[Dict, Dict, List]]:
    """
    Extract entities and relationships from text chunks using LLM

    This is the core extraction function that:
    1. Sends each chunk to the LLM with an entity extraction prompt
    2. Parses the structured response (entities and relationships)
    3. Returns aggregated results with deduplication

    The function processes all chunks in parallel for efficiency and tracks progress.

    Args:
        chunks: Dictionary of text chunks {chunk_id: {content, tokens, ...}}
            Each chunk should have a 'content' field with the text to process
        llm_func: LLM function to use for extraction (default: google_model)
            Supported: google_model, gpt_model, deepseek_model
        show_progress: Whether to display progress during processing (default: True)

    Returns:
        List of tuples, each containing:
            - entities_dict: {entity_name: [entity_data, ...]} - grouped by entity name
            - relations_dict: {(src, tgt): [relation_data, ...]} - grouped by entity pair
            - raw_records: List of unparsed LLM output records

    Example:
        >>> chunks = {"chunk-abc123": {"content": "John works at Google..."}}
        >>> results = await extract_entities_and_relations(chunks)
        >>> entities_dict, relations_dict, _ = results[0]
        >>> list(entities_dict.keys())
        ['JOHN', 'GOOGLE']
        >>> list(relations_dict.keys())
        [('JOHN', 'GOOGLE')]

    Note:
        - All entities and relationships are normalized to uppercase
        - Duplicates are grouped by entity name or (source, target) tuple
        - Invalid records are silently skipped
        - Progress ticker is displayed if show_progress=True
    """
    ordered_chunks = list(chunks.items())

    # Get extraction prompt template
    entity_extract_prompt = PROMPTS["entity_extraction"]
    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
    )

    # Tracking variables for progress display
    already_processed = 0
    already_entities = 0
    already_relations = 0

    async def _process_single_content(chunk_key_dp: Tuple[str, Dict]) -> Tuple[Dict, Dict, List]:
        """Process a single chunk to extract entities and relationships"""
        nonlocal already_processed, already_entities, already_relations

        chunk_key = chunk_key_dp[0]
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp["content"]

        # Format prompt with chunk content
        hint_prompt = entity_extract_prompt.format(**context_base, input_text=content)

        # Call LLM to extract entities and relationships
        final_result = await llm_func(
            hint_prompt,
            operation_name="entity_extraction"
        )

        # Parse LLM response
        records = split_string_by_multi_markers(
            final_result,
            [context_base["record_delimiter"], context_base["completion_delimiter"]],
        )

        maybe_nodes = defaultdict(list)
        maybe_edges = defaultdict(list)

        for record in records:
            # Extract content from parentheses
            record_match = re.search(r"\((.*)\)", record)
            if record_match is None:
                continue
            record = record_match.group(1)

            # Split by tuple delimiter
            record_attributes = split_string_by_multi_markers(
                record, [context_base["tuple_delimiter"]]
            )

            # Try to parse as entity
            if_entities = await _handle_single_entity_extraction(
                record_attributes, chunk_key
            )
            if if_entities is not None:
                maybe_nodes[if_entities["entity_name"]].append(if_entities)
                continue

            # Try to parse as relationship
            if_relation = await _handle_single_relationship_extraction(
                record_attributes, chunk_key
            )
            if if_relation is not None:
                maybe_edges[(if_relation["src_id"], if_relation["tgt_id"])].append(
                    if_relation
                )

        # Update progress tracking
        already_processed += 1
        already_entities += len(maybe_nodes)
        already_relations += len(maybe_edges)

        if show_progress:
            now_ticks = PROMPTS["process_tickers"][
                already_processed % len(PROMPTS["process_tickers"])
            ]
            print(
                f"{now_ticks} Processed {already_processed}/{len(ordered_chunks)} "
                f"({already_processed*100//len(ordered_chunks)}%) chunks, "
                f"{already_entities} entities (duplicated), "
                f"{already_relations} relations (duplicated)\r",
                end="",
                flush=True,
            )

        return dict(maybe_nodes), dict(maybe_edges), records

    # Process all chunks in parallel
    results = await asyncio.gather(
        *[_process_single_content(c) for c in ordered_chunks]
    )

    if show_progress:
        print()  # New line after progress

    return results


# ============================================================================
# Entity and Relationship Summarization
# ============================================================================

async def _handle_entity_relation_summary(
    entity_or_relation_name: Union[str, Tuple[str, str]],
    description: str,
    llm_func: callable = google_model,
) -> str:
    """
    Summarize entity or relationship descriptions when they become too long

    When multiple chunks mention the same entity, descriptions are concatenated.
    If the combined description exceeds the token limit, this function
    generates a concise summary using an LLM.

    Args:
        entity_or_relation_name: Name of entity or (src, tgt) tuple for relation
        description: Concatenated descriptions (joined by GRAPH_FIELD_SEP)
        llm_func: LLM function to use for summarization (default: google_model)

    Returns:
        Original description if short enough, otherwise LLM-generated summary

    Note:
        - Summary is only triggered if description exceeds 1024 tokens
        - Uses first 16384 tokens if description is extremely long
        - Summary maintains entity name context

    Example:
        >>> long_desc = "Desc1<SEP>Desc2<SEP>..." * 100  # Very long
        >>> summary = await _handle_entity_relation_summary("APPLE", long_desc)
        >>> len(encode_string_by_tiktoken(summary)) < 1024
        True
    """
    llm_max_tokens = 16384
    tiktoken_model_name = "gpt-4o"
    summary_max_tokens = 1024

    # Check if summary is needed
    tokens = encode_string_by_tiktoken(description, model_name=tiktoken_model_name)
    if len(tokens) < summary_max_tokens:
        return description

    # Truncate description if too long
    prompt_template = PROMPTS["summarize_entity_descriptions"]
    use_description = decode_tokens_by_tiktoken(
        tokens[:llm_max_tokens], model_name=tiktoken_model_name
    )

    # Format summarization prompt
    context_base = dict(
        entity_name=entity_or_relation_name,
        description_list=use_description.split(GRAPH_FIELD_SEP),
    )
    use_prompt = prompt_template.format(**context_base)

    logger.info(f"Generating summary for: {entity_or_relation_name}")

    # Generate summary
    summary = await llm_func(
        use_prompt,
        operation_name="entity_summarization"
    )

    return summary


# ============================================================================
# Graph Construction
# ============================================================================

async def _merge_nodes_then_upsert(
    entity_name: str,
    nodes_data: List[Dict],
    kg: nx_graph,
    llm_func: callable = google_model,
) -> Dict[str, Any]:
    """
    Merge duplicate entity nodes and upsert to graph

    When multiple chunks mention the same entity, this function:
    1. Combines all entity types (uses most common)
    2. Concatenates descriptions
    3. Merges source IDs
    4. Summarizes if description is too long
    5. Upserts to the knowledge graph

    Args:
        entity_name: Name of the entity (uppercase, normalized)
        nodes_data: List of entity data dictionaries from different chunks
        kg: Knowledge graph instance (nx_graph)
        llm_func: LLM function for summarization (default: google_model)

    Returns:
        Merged entity data dictionary with all metadata

    Note:
        - Entity type is determined by majority vote
        - Descriptions are deduplicated and sorted
        - Source IDs track all chunks mentioning this entity
        - If entity already exists in graph, data is merged with existing

    Example:
        >>> nodes = [
        ...     {"entity_name": "APPLE", "entity_type": "ORGANIZATION", ...},
        ...     {"entity_name": "APPLE", "entity_type": "COMPANY", ...}
        ... ]
        >>> merged = await _merge_nodes_then_upsert("APPLE", nodes, kg)
        >>> merged['entity_type']  # Most common type
        'ORGANIZATION'
    """
    already_entity_types = []
    already_source_ids = []
    already_description = []

    # Check if entity already exists in graph
    already_node = await kg.get_node(entity_name)
    if already_node is not None:
        already_entity_types.append(already_node["entity_type"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_node["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_node["description"])

    # Determine most common entity type (majority vote)
    entity_type = sorted(
        Counter(
            [dp["entity_type"] for dp in nodes_data] + already_entity_types
        ).items(),
        key=lambda x: x[1],
        reverse=True,
    )[0][0]

    # Merge descriptions (deduplicate and sort for consistency)
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in nodes_data] + already_description))
    )

    # Merge source IDs
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in nodes_data] + already_source_ids)
    )

    # Summarize if description is too long
    description = await _handle_entity_relation_summary(
        entity_name, description, llm_func
    )

    # Prepare node data
    node_data = dict(
        entity_type=entity_type,
        description=description,
        source_id=source_id,
    )

    # Upsert to graph
    await kg.upsert_node(
        entity_name,
        node_data=node_data,
    )

    node_data["entity_name"] = entity_name
    return node_data


async def _merge_edges_then_upsert(
    src_id: str,
    tgt_id: str,
    edges_data: List[Dict],
    kg: nx_graph,
    llm_func: callable = google_model,
) -> None:
    """
    Merge duplicate relationship edges and upsert to graph

    When multiple chunks mention the same relationship, this function:
    1. Sums relationship weights
    2. Concatenates descriptions
    3. Merges source IDs
    4. Ensures both source and target nodes exist
    5. Summarizes if description is too long
    6. Upserts to the knowledge graph

    Args:
        src_id: Source entity name (uppercase, normalized)
        tgt_id: Target entity name (uppercase, normalized)
        edges_data: List of relationship data dictionaries
        kg: Knowledge graph instance (nx_graph)
        llm_func: LLM function for summarization (default: google_model)

    Returns:
        None (edge is upserted to graph)

    Note:
        - Weights are summed (higher weight = stronger relationship)
        - If nodes don't exist, creates placeholder nodes with "UNKNOWN" type
        - Order field is used for relationship precedence (minimum across all mentions)
        - Order=0 is reserved for augmentation edges (embedding similarity)

    Example:
        >>> edges = [
        ...     {"src_id": "APPLE", "tgt_id": "IPHONE", "weight": 8, ...},
        ...     {"src_id": "APPLE", "tgt_id": "IPHONE", "weight": 9, ...}
        ... ]
        >>> await _merge_edges_then_upsert("APPLE", "IPHONE", edges, kg)
        >>> edge = await kg.get_edge("APPLE", "IPHONE")
        >>> edge['weight']
        17.0  # Sum of weights
    """
    already_weights = []
    already_source_ids = []
    already_description = []
    already_order = []

    # Check if edge already exists
    if await kg.has_edge(src_id, tgt_id):
        already_edge = await kg.get_edge(src_id, tgt_id)
        already_weights.append(already_edge["weight"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_edge["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_edge["description"])
        already_order.append(already_edge.get("order", 1))

    # Merge edge properties
    order = min([dp.get("order", 1) for dp in edges_data] + already_order)
    weight = sum([dp["weight"] for dp in edges_data] + already_weights)
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in edges_data] + already_description))
    )
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in edges_data] + already_source_ids)
    )

    # Ensure both nodes exist (create placeholder if missing)
    for need_insert_id in [src_id, tgt_id]:
        if not (await kg.has_node(need_insert_id)):
            await kg.upsert_node(
                need_insert_id,
                node_data={
                    "source_id": source_id,
                    "description": description,
                    "entity_type": '"UNKNOWN"',
                },
            )

    # Summarize if description is too long
    description = await _handle_entity_relation_summary(
        (src_id, tgt_id), description, llm_func
    )

    # Upsert edge
    await kg.upsert_edge(
        src_id,
        tgt_id,
        edge_data=dict(
            weight=weight,
            description=description,
            source_id=source_id,
            order=order
        ),
    )


async def build_knowledge_graph(
    extraction_results: List[Tuple[Dict, Dict, List]],
    kg: Optional[nx_graph] = None,
    llm_func: callable = google_model,
) -> Tuple[nx_graph, List[Dict]]:
    """
    Build knowledge graph from extracted entities and relationships

    Takes the output from extract_entities_and_relations() and constructs a knowledge graph
    by merging duplicate entities/relationships and upserting to the graph.

    This function handles:
    - Aggregating entities and relationships from all chunks
    - Treating relationships as undirected (sorting node pairs)
    - Merging duplicates with conflict resolution
    - Parallel processing for efficiency

    Args:
        extraction_results: List of extraction results from extract_entities_and_relations()
            Each tuple contains (entities_dict, relations_dict, raw_records)
        kg: Knowledge graph instance (nx_graph). If None, creates a new graph.
        llm_func: LLM function for summarization (default: google_model)

    Returns:
        Tuple of (kg, all_entities_data):
            - kg: Updated knowledge graph (nx_graph instance)
            - all_entities_data: List of all entity dictionaries with metadata

    Example:
        >>> results = await extract_entities_and_relations(chunks)
        >>> kg, entities = await build_knowledge_graph(results)
        >>> print(f"Graph has {len(await kg.get_all_nodes())} nodes")
        Graph has 245 nodes
        >>> print(f"Graph has {len(await kg.get_all_edges())} edges")
        Graph has 512 edges

    Note:
        - Entities are processed first (nodes before edges)
        - Relationships are treated as undirected (edges sorted by name)
        - All operations are performed in parallel for efficiency
        - Returns None if no entities were extracted
    """
    # Create new graph if not provided
    if kg is None:
        kg = nx_graph()

    maybe_nodes = defaultdict(list)
    maybe_edges = defaultdict(list)

    # Aggregate all entities and relationships from extraction results
    for m_nodes, m_edges, _ in extraction_results:
        for k, v in m_nodes.items():
            maybe_nodes[k].extend(v)
        for k, v in m_edges.items():
            # Treat as undirected graph (sort node names)
            maybe_edges[tuple(sorted(k))].extend(v)

    # Merge and upsert all entities (parallel processing)
    logger.info(f"Merging {len(maybe_nodes)} unique entities...")
    all_entities_data = await asyncio.gather(
        *[
            _merge_nodes_then_upsert(k, v, kg, llm_func)
            for k, v in maybe_nodes.items()
        ]
    )

    # Merge and upsert all relationships (parallel processing)
    logger.info(f"Merging {len(maybe_edges)} unique relationships...")
    await asyncio.gather(
        *[
            _merge_edges_then_upsert(k[0], k[1], v, kg, llm_func)
            for k, v in maybe_edges.items()
        ]
    )

    # Validate that entities were extracted
    if not len(all_entities_data):
        logger.warning("No entities extracted - LLM may not be working properly")
        return None, []

    logger.info(
        f"Graph construction complete: {len(await kg.get_all_nodes())} nodes, "
        f"{len(await kg.get_all_edges())} edges"
    )

    return kg, all_entities_data


# ============================================================================
# KGBuilder Class - High-Level Interface
# ============================================================================

class KGBuilder:
    """
    High-level class for orchestrating knowledge graph construction

    This class provides a clean interface for the complete KG construction pipeline,
    with support for configuration files, caching, and progress tracking.

    Key features:
    - Automatic configuration loading from YAML
    - Built-in caching for expensive operations
    - Support for multiple LLM providers
    - Progress tracking and logging
    - Batch processing of multiple documents

    Attributes:
        config: Configuration dictionary loaded from YAML
        llm_func: LLM function to use for extraction
        cache_dir: Directory for caching intermediate results
        enable_cache: Whether to use caching

    Example:
        >>> builder = KGBuilder(config_path="config.yaml")
        >>>
        >>> # Option 1: Build from text chunks
        >>> chunks = doc_to_chunks(document_text, chunk_size=512)
        >>> kg, entities = await builder.build_from_chunks(chunks)
        >>>
        >>> # Option 2: Build from document with caching
        >>> kg, entities = await builder.build_from_document(
        ...     document_text,
        ...     doc_id="doc-123",
        ...     use_cache=True
        ... )
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        config: Optional[Dict] = None,
        llm_provider: Optional[str] = None,
        cache_dir: Optional[str] = None,
        enable_cache: bool = True,
    ):
        """
        Initialize KGBuilder

        Args:
            config_path: Path to YAML configuration file (optional)
            config: Configuration dictionary (overrides config_path)
            llm_provider: LLM provider to use ("gpt", "deepseek", "gemini", "ollama")
                         If None, reads from config (default: gpt)
            cache_dir: Directory for caching (uses config if not provided)
            enable_cache: Whether to enable caching of intermediate results
        """
        # Load configuration
        if config is not None:
            self.config = config
        elif config_path is not None:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            # Use default configuration
            self.config = self._get_default_config()

        # Set LLM function based on provider
        llm_provider = llm_provider or self.config.get('services', {}).get('llm', {}).get('provider', 'gpt')
        self.llm_func = self._get_llm_function(llm_provider)

        # Set cache directory
        self.cache_dir = cache_dir or self.config.get('paths', {}).get('cache_root', './cache')
        self.enable_cache = enable_cache

        # Create cache directory if needed
        if self.enable_cache:
            Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

        logger.info(f"KGBuilder initialized with {llm_provider} LLM provider")

    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            'kg': {
                'chunk_size': 512,
                'chunk_overlap': 64,
            },
            'services': {
                'llm': {
                    'provider': 'gpt',
                }
            },
            'paths': {
                'cache_root': './cache'
            }
        }

    def _get_llm_function(self, provider: str) -> callable:
        """Get LLM function based on provider name"""
        provider_map = {
            'gemini': google_model,
            'google': google_model,
            'gpt': gpt_model,
            'openai': gpt_model,
            'deepseek': deepseek_model,
        }

        llm_func = provider_map.get(provider.lower())
        if llm_func is None:
            logger.warning(f"Unknown provider '{provider}', defaulting to gpt_model")
            llm_func = gpt_model

        return llm_func

    async def build_from_chunks(
        self,
        chunks: Dict[str, Dict],
        show_progress: bool = True,
    ) -> Tuple[nx_graph, List[Dict]]:
        """
        Build knowledge graph from text chunks

        Args:
            chunks: Dictionary of text chunks {chunk_id: {content, tokens, ...}}
            show_progress: Whether to show progress during extraction

        Returns:
            Tuple of (kg, entities):
                - kg: Constructed knowledge graph
                - entities: List of all entity dictionaries

        Example:
            >>> chunks = doc_to_chunks(text, chunk_size=512)
            >>> kg, entities = await builder.build_from_chunks(chunks)
        """
        logger.info(f"Extracting entities from {len(chunks)} chunks...")

        # Step 1: Extract entities and relationships
        extraction_results = await extract_entities_and_relations(
            chunks,
            llm_func=self.llm_func,
            show_progress=show_progress,
        )

        # Step 2: Build knowledge graph
        logger.info("Building knowledge graph...")
        kg, entities = await build_knowledge_graph(
            extraction_results,
            llm_func=self.llm_func,
        )

        return kg, entities

    async def build_from_document(
        self,
        document: str,
        doc_id: Optional[str] = None,
        chunk_size: Optional[int] = None,
        overlap_size: Optional[int] = None,
        use_cache: bool = True,
        show_progress: bool = True,
    ) -> Tuple[nx_graph, List[Dict]]:
        """
        Build knowledge graph from a document with automatic chunking

        This method handles the complete pipeline:
        1. Chunks the document
        2. Caches chunks if enabled
        3. Extracts entities and relationships
        4. Builds the knowledge graph
        5. Caches results if enabled

        Args:
            document: Text document to process
            doc_id: Document identifier for caching (auto-generated if None)
            chunk_size: Token size for chunks (uses config if None)
            overlap_size: Overlap tokens between chunks (uses config if None)
            use_cache: Whether to use cached results if available
            show_progress: Whether to show progress during extraction

        Returns:
            Tuple of (kg, entities):
                - kg: Constructed knowledge graph
                - entities: List of all entity dictionaries

        Example:
            >>> kg, entities = await builder.build_from_document(
            ...     document_text,
            ...     doc_id="finance-report-2024",
            ...     use_cache=True
            ... )
        """
        # Generate doc_id if not provided
        if doc_id is None:
            doc_id = compute_mdhash_id(document, prefix="doc-")

        # Get chunk sizes from config if not provided
        if chunk_size is None:
            chunk_size = self.config.get('kg', {}).get('chunk_size', 512)
        if overlap_size is None:
            overlap_size = self.config.get('kg', {}).get('chunk_overlap', 64)

        # Check cache for entities
        cache_path = None
        if self.enable_cache and use_cache:
            cache_path = Path(self.cache_dir) / f"{doc_id}_entities.pkl"
            if cache_path.exists():
                logger.info(f"Loading cached extraction results for {doc_id}")
                extraction_results = load_cache(
                    self.cache_dir, "entities", doc_id, 0, extension="pkl"
                )
                kg, entities = await build_knowledge_graph(
                    extraction_results,
                    llm_func=self.llm_func,
                )
                return kg, entities

        # Chunk document
        logger.info(f"Chunking document (chunk_size={chunk_size}, overlap={overlap_size})...")
        chunks = doc_to_chunks(document, chunk_size=chunk_size, overlap_size=overlap_size)

        # Build graph
        kg, entities = await self.build_from_chunks(chunks, show_progress=show_progress)

        # Cache results
        if self.enable_cache and cache_path is not None:
            logger.info(f"Caching extraction results to {cache_path}")
            # Note: We would cache extraction_results here, but we need to refactor
            # build_from_chunks to return them

        return kg, entities

    def get_chunk_config(self) -> Dict[str, int]:
        """
        Get chunking configuration

        Returns:
            Dictionary with chunk_size and chunk_overlap
        """
        return {
            'chunk_size': self.config.get('kg', {}).get('chunk_size', 512),
            'chunk_overlap': self.config.get('kg', {}).get('chunk_overlap', 64),
        }
