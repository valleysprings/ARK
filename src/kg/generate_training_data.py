"""
ARK Data Generation Interface

Unified interface for:
1. Extracting subgraphs from full KG based on answers
2. Generating augmented queries from KG subgraphs
3. Generating training data (preference pairs) from queries

Usage:
    # Extract subgraphs
    python -m src.kg.generate_training_data extract-subgraphs --dataset hotpotqa --subgraph_size large --start 0 --end 100

    # Generate queries
    python -m src.kg.generate_training_data generate-queries --dataset hotpotqa --start 0 --end 100

    # Generate training data
    python -m src.kg.generate_training_data generate-data --dataset hotpotqa
"""

import argparse
import json
import os
import yaml
import jsonlines
import pickle
import asyncio
import networkx as nx
from pathlib import Path
from typing import Dict, List, Any, Set
from tqdm import tqdm
import torch

# Import ops
from src.kg.ops.query_generation import QueryGenerator
from src.kg.ops.training_data import load_generated_queries, generate_preference_pairs
from src.kg.ops.extraction import SubgraphExtractor, load_config as load_extraction_config


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_original_data(dataset_path: str, index: int) -> Dict[str, Any]:
    """Load original question-answer data at specific index"""
    with jsonlines.open(dataset_path) as reader:
        for i, line in enumerate(reader):
            if i == index:
                return line
    raise ValueError(f"Index {index} not found in {dataset_path}")


def extract_seed_entities_from_text(text: str, graph: nx.Graph, max_entities: int = 20) -> Set[str]:
    """
    Extract seed entities from text by matching uppercase words to graph nodes

    Args:
        text: Input text (e.g., answer)
        graph: NetworkX graph containing entity nodes
        max_entities: Maximum number of seed entities to return

    Returns:
        Set of matched entity names from the graph
    """
    # Simple approach: extract words and match to graph nodes
    # Normalize to uppercase (assuming graph nodes are uppercase)
    words = text.upper().split()
    graph_nodes = set(graph.nodes())

    # Create a mapping of stripped node names to original node names
    # (to handle quoted nodes like '"MILLER V. CALIFORNIA"')
    node_mapping = {}
    for node in graph_nodes:
        stripped = node.strip('"').strip("'")
        node_mapping[stripped] = node

    # Try different n-gram lengths (3-gram, 2-gram, 1-gram)
    matched_entities = set()

    # Try 3-grams
    for i in range(len(words) - 2):
        trigram = " ".join(words[i:i+3])
        if trigram in node_mapping:
            matched_entities.add(node_mapping[trigram])
        elif trigram in graph_nodes:
            matched_entities.add(trigram)

    # Try 2-grams
    for i in range(len(words) - 1):
        bigram = " ".join(words[i:i+2])
        if bigram in node_mapping:
            matched_entities.add(node_mapping[bigram])
        elif bigram in graph_nodes:
            matched_entities.add(bigram)

    # Try 1-grams
    for word in words:
        if word in node_mapping:
            matched_entities.add(node_mapping[word])
        elif word in graph_nodes:
            matched_entities.add(word)

    # Return top max_entities (prioritize longer n-grams)
    return set(list(matched_entities)[:max_entities])


async def cmd_extract_subgraphs(args):
    """Extract subgraphs from full KG based on answers

    This function extracts BOTH small and large subgraphs in one pass:
    - Runs PPR once per sample
    - Applies top-K cutoff with K=10 for small, K=20 for large
    - Saves to both small and large directories
    """

    # Load config
    config = load_config(args.config)

    # Resolve parameters
    dataset_type = args.dataset_type or config['dataset']['type']

    # Get both min_k values
    k_small = config['subgraph'].get('min_k_small', 10)
    k_large = config['subgraph'].get('min_k_large', 20)

    # Paths
    kg_dir = args.kg_dir or f"./data/preprocessed/{dataset_type}/full_kg_augmented"
    output_dir_small = f"./data/preprocessed/{dataset_type}/subgraphs_answer_small"
    output_dir_large = f"./data/preprocessed/{dataset_type}/subgraphs_answer_large"

    if args.dataset_path:
        dataset_path = args.dataset_path
    else:
        # Try both formats: dataset_type.jsonl and dataset_type/train.jsonl
        dataset_path_simple = os.path.join(
            config['dataset']['raw_data_dir'],
            f'{dataset_type}.jsonl'
        )
        dataset_path_nested = os.path.join(
            config['dataset']['raw_data_dir'],
            dataset_type,
            'train.jsonl'
        )

        # Use whichever exists
        if os.path.exists(dataset_path_simple):
            dataset_path = dataset_path_simple
        elif os.path.exists(dataset_path_nested):
            dataset_path = dataset_path_nested
        else:
            # Default to simple format
            dataset_path = dataset_path_simple

    start_index = args.start_index if args.start_index is not None else config['range']['start_index']
    end_index = args.end_index if args.end_index is not None else config['range']['end_index']

    # Create both output directories
    os.makedirs(output_dir_small, exist_ok=True)
    os.makedirs(output_dir_large, exist_ok=True)

    print("=" * 60)
    print("Subgraph Extraction (Small + Large)")
    print("=" * 60)
    print(f"Dataset: {dataset_type}")
    print(f"KG Directory: {kg_dir}")
    print(f"Output Small: {output_dir_small} (K={k_small})")
    print(f"Output Large: {output_dir_large} (K={k_large})")
    print(f"Range: {start_index} to {end_index}")
    print("=" * 60)

    # Load extraction config
    extraction_config = load_extraction_config(args.extraction_config if hasattr(args, 'extraction_config') else None)

    # Process each index
    for idx in tqdm(range(start_index, end_index), desc="Extracting subgraphs"):
        kg_file = os.path.join(kg_dir, f"{dataset_type}_{idx}.pkl")
        output_file_small = os.path.join(output_dir_small, f"{dataset_type}_{idx}.pkl")
        output_file_large = os.path.join(output_dir_large, f"{dataset_type}_{idx}.pkl")

        # Skip if both already exist
        if (os.path.exists(output_file_small) and os.path.exists(output_file_large)
            and not args.overwrite):
            print(f"\nSkipping {idx}: both subgraphs already exist")
            continue

        if not os.path.exists(kg_file):
            print(f"\nWarning: KG not found for index {idx}")
            continue

        try:
            # Load KG (pickle format)
            with open(kg_file, 'rb') as f:
                graph = pickle.load(f)

            # Handle different graph formats
            # First check if it's already a NetworkX graph
            if hasattr(graph, 'nodes') and hasattr(graph, 'edges'):
                # Already a NetworkX graph, use as-is
                pass
            elif hasattr(graph, 'graph') and not hasattr(graph, 'nodes'):
                # Wrapper object with 'graph' attribute
                graph = graph.graph
            elif isinstance(graph, dict):
                # If graph is a dict, try to convert to NetworkX Graph
                if 'graph' in graph:
                    graph = graph['graph']
                else:
                    # Try to create NetworkX graph from dict
                    import networkx as nx
                    graph = nx.Graph(graph)

            # Verify we have a NetworkX graph
            if not hasattr(graph, 'nodes'):
                print(f"\nWarning: Invalid graph type for index {idx}: {type(graph)}")
                continue

            # Load original data to get answer
            original_data = load_original_data(dataset_path, idx)
            answer = original_data.get("answer", original_data.get("answers", original_data.get("output", "")))

            # Handle list format (e.g., hotpotqa uses ["answer"])
            if isinstance(answer, list) and answer:
                answer = answer[0]

            if not answer:
                print(f"\nWarning: No answer found for index {idx}")
                continue

            # Extract seed entities from answer
            seed_entities = extract_seed_entities_from_text(answer, graph)

            if not seed_entities:
                print(f"\nWarning: No seed entities found for index {idx}")
                continue

            # Create SubgraphExtractor
            extractor = SubgraphExtractor(
                graph=graph,
                vectordb=None,
                config=extraction_config
            )

            # Run PPR once to get ranked entities
            # Use adaptive=False with top_k=k_large to get top-20 entities
            entity_list_full, scores_dict = await extractor.extract_subgraph_from_seeds(
                seed_entities=seed_entities,
                use_adaptive=False,  # Use fixed top-k instead of adaptive cutoff
                use_community_search=False,
                top_k=k_large,  # Get top-20 entities (for large subgraph)
            )

            # Apply top-K cutoff for small (K=10) - take first 10 from the 20
            entity_list_small = entity_list_full[:k_small]

            # Large subgraph uses all entities returned (top-20)
            entity_list_large = entity_list_full

            # Get entity data for small subgraph
            entity_data_dict_small = extractor.get_entity_data(entity_list_small)
            entity_data_list_small = [
                {
                    'entity_name': entity_name,
                    'entity_type': entity_info.get('entity_type', 'Unknown'),
                    'description': entity_info.get('description', ''),
                    'source_id': entity_info.get('source_id', '')
                }
                for entity_name, entity_info in entity_data_dict_small.items()
            ]

            # Get entity data for large subgraph
            entity_data_dict_large = extractor.get_entity_data(entity_list_large)
            entity_data_list_large = [
                {
                    'entity_name': entity_name,
                    'entity_type': entity_info.get('entity_type', 'Unknown'),
                    'description': entity_info.get('description', ''),
                    'source_id': entity_info.get('source_id', '')
                }
                for entity_name, entity_info in entity_data_dict_large.items()
            ]

            # Save small subgraph
            with open(output_file_small, 'wb') as f:
                pickle.dump(entity_data_list_small, f)

            # Save large subgraph
            with open(output_file_large, 'wb') as f:
                pickle.dump(entity_data_list_large, f)

            print(f"\nIndex {idx}: Extracted {len(entity_list_small)} entities (small) and {len(entity_list_large)} entities (large) from {len(seed_entities)} seeds")

        except Exception as e:
            print(f"\nError at index {idx}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("=" * 60)
    print(f"✅ Extracted subgraphs for {end_index - start_index} samples")
    print(f"📁 Small subgraphs: {output_dir_small}")
    print(f"📁 Large subgraphs: {output_dir_large}")
    print("=" * 60)


def cmd_generate_queries(args):
    """Generate augmented queries from subgraphs

    This function generates queries for BOTH small and large subgraphs:
    - Reads both small and large subgraphs
    - Generates queries for each
    - Saves to separate directories (queries_small and queries_large)
    """

    # Load config
    config = load_config(args.config)

    # Resolve parameters
    dataset_type = args.dataset_type or config['dataset']['type']

    # Define both subgraph directories
    subgraph_dir_small = f"./data/preprocessed/{dataset_type}/subgraphs_answer_small"
    subgraph_dir_large = f"./data/preprocessed/{dataset_type}/subgraphs_answer_large"

    # Define both output directories
    output_dir_small = f"./data/preprocessed/{dataset_type}/queries_small"
    output_dir_large = f"./data/preprocessed/{dataset_type}/queries_large"

    if args.dataset_path:
        dataset_path = args.dataset_path
    else:
        # Try both formats: dataset_type.jsonl and dataset_type/train.jsonl
        dataset_path_simple = os.path.join(
            config['dataset']['raw_data_dir'],
            f'{dataset_type}.jsonl'
        )
        dataset_path_nested = os.path.join(
            config['dataset']['raw_data_dir'],
            dataset_type,
            'train.jsonl'
        )

        # Use whichever exists
        if os.path.exists(dataset_path_simple):
            dataset_path = dataset_path_simple
        elif os.path.exists(dataset_path_nested):
            dataset_path = dataset_path_nested
        else:
            # Default to simple format
            dataset_path = dataset_path_simple

    start_index = args.start_index if args.start_index is not None else config['range']['start_index']
    end_index = args.end_index if args.end_index is not None else config['range']['end_index']

    # LLM config
    if args.llm_config and args.llm_config in config.get('llm_configs', {}):
        llm_cfg = config['llm_configs'][args.llm_config]
        llm_provider = llm_cfg['provider']
        llm_model = llm_cfg['model']
    else:
        llm_provider = args.llm_provider or config['llm']['provider']
        llm_model = args.llm_model or config['llm']['model']

    num_queries = args.num_queries if args.num_queries is not None else config['generation']['num_queries']

    # Load LLM API config
    llm_api_config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'llm_api.yaml')
    try:
        with open(llm_api_config_path, 'r') as f:
            llm_api_config = yaml.safe_load(f).get('llm_api', {})
    except FileNotFoundError:
        print(f"Warning: LLM API config not found at {llm_api_config_path}, using environment variables")
        llm_api_config = {}

    # Create both output directories
    os.makedirs(output_dir_small, exist_ok=True)
    os.makedirs(output_dir_large, exist_ok=True)

    # Initialize query generator
    generator = QueryGenerator(
        llm_provider=llm_provider,
        llm_model=llm_model,
        num_queries=num_queries,
        llm_config=llm_api_config,
    )

    print("=" * 60)
    print("Query Generation (Small + Large)")
    print("=" * 60)
    print(f"Dataset: {dataset_type}")
    print(f"Subgraph Small: {subgraph_dir_small}")
    print(f"Subgraph Large: {subgraph_dir_large}")
    print(f"Output Small: {output_dir_small}")
    print(f"Output Large: {output_dir_large}")
    print(f"Range: {start_index} to {end_index}")
    print(f"LLM: {llm_provider}/{llm_model}")
    print(f"Queries per subgraph: {num_queries}")
    print("=" * 60)

    # Process each index
    num_generated_small = 0
    num_generated_large = 0

    for idx in tqdm(range(start_index, end_index), desc="Generating queries"):
        subgraph_file_small = os.path.join(subgraph_dir_small, f"{dataset_type}_{idx}.pkl")
        subgraph_file_large = os.path.join(subgraph_dir_large, f"{dataset_type}_{idx}.pkl")

        # Load original data once
        try:
            original_data = load_original_data(dataset_path, idx)
        except Exception as e:
            print(f"\nError loading original data for index {idx}: {e}")
            continue

        # Process small subgraph
        if os.path.exists(subgraph_file_small):
            try:
                queries_small = generator.process_subgraph_file(subgraph_file_small, original_data)
                if queries_small:
                    output_file_small = os.path.join(output_dir_small, f"{dataset_type}_{idx}.pkl")
                    with open(output_file_small, 'wb') as f:
                        pickle.dump(queries_small, f)
                    num_generated_small += 1
            except Exception as e:
                print(f"\nError generating small queries for index {idx}: {e}")

        # Process large subgraph
        if os.path.exists(subgraph_file_large):
            try:
                queries_large = generator.process_subgraph_file(subgraph_file_large, original_data)
                if queries_large:
                    output_file_large = os.path.join(output_dir_large, f"{dataset_type}_{idx}.pkl")
                    with open(output_file_large, 'wb') as f:
                        pickle.dump(queries_large, f)
                    num_generated_large += 1
            except Exception as e:
                print(f"\nError generating large queries for index {idx}: {e}")

        # Print summary for this index
        if os.path.exists(subgraph_file_small) or os.path.exists(subgraph_file_large):
            print(f"\nIndex {idx}: Generated queries (small: {num_generated_small}, large: {num_generated_large})")

    print("=" * 60)
    print(f"✅ Generated queries for {num_generated_small} small samples and {num_generated_large} large samples")
    print(f"📁 Small queries: {output_dir_small}/")
    print(f"📁 Large queries: {output_dir_large}/")
    print("=" * 60)


def cmd_generate_data(args):
    """Generate training data (preference pairs) from queries"""

    print("=" * 60)
    print("Training Data Generation")
    print("=" * 60)
    print(f"Alignment dir: {args.alignment_dir}")
    print(f"Query dir: {args.query_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Embedding model: {args.embedding_model}")
    print(f"Top-N: {args.top_n}, Top-M: {args.top_m}")
    if args.start_index is not None or args.end_index is not None:
        print(f"Index range: {args.start_index} to {args.end_index}")
    print("=" * 60)

    # Load generated queries
    print("\nLoading generated queries...")
    generated_queries = load_generated_queries(args.query_dir)
    print(f"Loaded queries for {len(generated_queries)} samples")

    # Filter by index range if specified
    if args.start_index is not None or args.end_index is not None:
        start_idx = args.start_index if args.start_index is not None else 0
        end_idx = args.end_index if args.end_index is not None else len(generated_queries)

        # Filter queries by index range
        filtered_queries = {
            str(idx): queries
            for idx, queries in generated_queries.items()
            if start_idx <= int(idx) < end_idx
        }
        print(f"Filtered to {len(filtered_queries)} samples (index {start_idx} to {end_idx})")
        generated_queries = filtered_queries

    # Generate preference pairs
    print("\nGenerating preference pairs...")
    preference_pairs = generate_preference_pairs(
        alignment_dir=args.alignment_dir,
        generated_queries=generated_queries,
        embedding_model_path=args.embedding_model,
        device=args.device,
        top_n=args.top_n,
        top_m=args.top_m,
    )

    # Save results
    print(f"\nGenerated {len(preference_pairs)} preference pairs")
    os.makedirs(args.output_dir, exist_ok=True)

    # Extract dataset name from alignment_dir path
    # e.g., ./data/preprocessed/hotpotqa/alignment -> hotpotqa
    dataset_name = os.path.basename(os.path.dirname(args.alignment_dir))

    # Write each preference pair to a separate pkl file
    for idx, pair in preference_pairs.items():
        output_file = os.path.join(args.output_dir, f"{dataset_name}_{idx}.pkl")
        with open(output_file, 'wb') as f:
            pickle.dump(pair, f)

    print("=" * 60)
    print(f"✅ Training data generation complete!")
    print(f"📁 Saved to: {args.output_dir}/")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="ARK Data Generation Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # ===== Extract Subgraphs Command =====
    extract_parser = subparsers.add_parser(
        'extract-subgraphs',
        help='Extract subgraphs from full KG based on answers'
    )

    extract_parser.add_argument('--config', type=str, default='src/config/query_generation.yaml')
    extract_parser.add_argument('--dataset_type', type=str, default=None)
    extract_parser.add_argument('--dataset_path', type=str, default=None)
    extract_parser.add_argument('--kg_dir', type=str, default=None,
                               help='Directory containing full KG files (graphml)')
    extract_parser.add_argument('--start_index', type=int, default=None)
    extract_parser.add_argument('--end_index', type=int, default=None)
    extract_parser.add_argument('--extraction_config', type=str, default=None,
                               help='Path to extraction config (graph_builder.yaml)')
    extract_parser.add_argument('--overwrite', action='store_true',
                               help='Overwrite existing subgraph files')

    # ===== Generate Queries Command =====
    queries_parser = subparsers.add_parser(
        'generate-queries',
        help='Generate augmented queries from KG subgraphs'
    )

    queries_parser.add_argument('--config', type=str, default='src/config/query_generation.yaml')
    queries_parser.add_argument('--dataset_type', type=str, default=None)
    queries_parser.add_argument('--dataset_path', type=str, default=None)
    queries_parser.add_argument('--output_dir', type=str, default=None)
    queries_parser.add_argument('--start_index', type=int, default=None)
    queries_parser.add_argument('--end_index', type=int, default=None)
    queries_parser.add_argument('--num_queries', type=int, default=None)
    queries_parser.add_argument('--llm_provider', type=str, default=None, choices=['gemini', 'openai'])
    queries_parser.add_argument('--llm_model', type=str, default=None)
    queries_parser.add_argument('--llm_config', type=str, default=None)

    # ===== Generate Training Data Command =====
    data_parser = subparsers.add_parser(
        'generate-data',
        help='Generate training data (preference pairs) from queries'
    )

    data_parser.add_argument('--alignment_dir', type=str, required=True,
                            help='Directory with alignment score PKL files')
    data_parser.add_argument('--query_dir', type=str, required=True,
                            help='Directory with generated query JSON files')
    data_parser.add_argument('--output_dir', type=str, required=True,
                            help='Output directory for preference pairs PKL files')
    data_parser.add_argument('--embedding_model', type=str, default='BAAI/bge-base-en-v1.5',
                            help='Sentence embedding model')
    data_parser.add_argument('--device', type=str,
                            default='cuda' if torch.cuda.is_available() else 'cpu')
    data_parser.add_argument('--top_n', type=int, default=3,
                            help='Number of top chunks for chosen response')
    data_parser.add_argument('--top_m', type=int, default=10,
                            help='Number of similar chunks for rejected response')
    data_parser.add_argument('--start_index', type=int, default=None,
                            help='Start index for processing (inclusive)')
    data_parser.add_argument('--end_index', type=int, default=None,
                            help='End index for processing (exclusive)')

    args = parser.parse_args()

    if args.command == 'extract-subgraphs':
        asyncio.run(cmd_extract_subgraphs(args))
    elif args.command == 'generate-queries':
        cmd_generate_queries(args)
    elif args.command == 'generate-data':
        cmd_generate_data(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
