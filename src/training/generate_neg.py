"""
Complete data generation pipeline for ARK training.
Handles: subgraph extraction → query generation → training data generation
"""

import argparse
import json
import logging
import sys
import yaml
from pathlib import Path
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.kg.ops.extraction import SubgraphExtractor
from src.kg.ops.query_generation import QueryGenerator
from src.training.ops.data_loader import load_alignment_data, load_queries_data, load_matched_chunks_data
from src.training.ops.chunk_retriever import ChunkRetriever
from src.training.ops.data_generator import generate_stage1_data, generate_stage2_data, generate_stage3_data

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "./src/config/training.yaml"):
    """Load training configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def prepare_stage1(config, dataset_name: str):
    """Prepare Stage 1 training data."""
    logger.info("=" * 60)
    logger.info("PREPARING STAGE 1 DATA")
    logger.info("=" * 60)

    preprocessed_dir = config['paths']['data_root'] + '/preprocessed'
    stage1_config = config['training']['qwen']['curriculum']['stage1']

    # Load alignment data for specified dataset
    alignment_data = load_alignment_data(dataset_name, preprocessed_dir, limit=None)

    if not alignment_data:
        logger.error(f"No alignment data loaded for {dataset_name}!")
        return 0

    # Output filename
    filename = f"{dataset_name}_train.jsonl"
    output_file = f"{config['paths']['data_root']}/training/stage1/{filename}"

    num_samples = generate_stage1_data(
        alignment_data=alignment_data,
        output_file=output_file,
        num_positives=stage1_config.get('num_positives', 3),
        num_negatives=stage1_config.get('num_negatives', None)
    )

    return num_samples


def prepare_stage2(config, dataset_name: str):
    """Prepare Stage 2 training data."""
    logger.info("=" * 60)
    logger.info("PREPARING STAGE 2 DATA")
    logger.info("=" * 60)

    preprocessed_dir = config['paths']['data_root'] + '/preprocessed'
    stage2_config = config['training']['qwen']['curriculum']['stage2']

    # Load data for specified dataset
    alignment_data = load_alignment_data(dataset_name, preprocessed_dir, limit=None)
    matched_chunks_data = load_matched_chunks_data(dataset_name, 'large', preprocessed_dir, limit=None)

    if not alignment_data or not matched_chunks_data:
        logger.error(f"No data loaded for Stage 2 ({dataset_name})!")
        return 0

    # Output filename
    filename = f"{dataset_name}_train.jsonl"
    output_file = f"{config['paths']['data_root']}/training/stage2/{filename}"

    num_samples = generate_stage2_data(
        alignment_data=alignment_data,
        matched_chunks_data=matched_chunks_data,
        output_file=output_file,
        num_positives=stage2_config.get('num_positives', 3),
        num_negatives=stage2_config.get('num_negatives', None)
    )

    return num_samples


def prepare_stage3(config, dataset_name: str):
    """Prepare Stage 3 training data."""
    logger.info("=" * 60)
    logger.info("PREPARING STAGE 3 DATA")
    logger.info("=" * 60)

    preprocessed_dir = config['paths']['data_root'] + '/preprocessed'
    stage3_config = config['training']['qwen']['curriculum']['stage3']

    # Load data for specified dataset
    alignment_data = load_alignment_data(dataset_name, preprocessed_dir, limit=None)
    matched_chunks_data = load_matched_chunks_data(dataset_name, 'small', preprocessed_dir, limit=None)

    if not alignment_data or not matched_chunks_data:
        logger.error(f"No data loaded for Stage 3 ({dataset_name})!")
        return 0

    # Output filename
    filename = f"{dataset_name}_train.jsonl"
    output_file = f"{config['paths']['data_root']}/training/stage3/{filename}"

    num_samples = generate_stage3_data(
        alignment_data=alignment_data,
        matched_chunks_data=matched_chunks_data,
        output_file=output_file,
        num_positives=stage3_config.get('num_positives', 5),
        num_negatives=stage3_config.get('num_negatives', None)
    )

    return num_samples


def extract_subgraphs(dataset_name: str, start_idx: int, end_idx: int):
    """Extract subgraphs (both small and large) from KG."""
    import pickle
    import json
    import asyncio
    from pathlib import Path
    from src.kg.ops.extraction import SubgraphExtractor, SubgraphConfig
    from src.kg.indexing import NanoVectorDBStorage

    logger.info("=" * 60)
    logger.info("STEP 1: EXTRACTING SUBGRAPHS")
    logger.info("=" * 60)

    base_dir = Path(f'./data/preprocessed/{dataset_name}')

    for idx in range(start_idx, end_idx):
        doc_id = f"{dataset_name}_{idx}"
        logger.info(f"Processing {doc_id}...")

        # Load KG
        kg_path = base_dir / "full_kg_augmented" / f"{doc_id}.pkl"
        if not kg_path.exists():
            kg_path = base_dir / "full_kg" / f"{doc_id}.pkl"
        if not kg_path.exists():
            logger.warning(f"KG not found for {doc_id}, skipping")
            continue

        with open(kg_path, 'rb') as f:
            graph = pickle.load(f)

        # Load alignment data to get questions
        alignment_path = base_dir / "alignment" / f"{doc_id}.json"
        if not alignment_path.exists():
            logger.warning(f"Alignment not found for {doc_id}, skipping")
            continue

        with open(alignment_path, 'r') as f:
            alignment_data = json.load(f)

        # Initialize extractor
        config = SubgraphConfig()
        extractor = SubgraphExtractor(graph=graph, config=config)

        # Get question and answer
        question = alignment_data.get('input', '')
        answer = alignment_data.get('answers', [''])[0] if alignment_data.get('answers') else ''

        # Extract subgraphs using answer as seed
        async def extract():
            # Find seed entities from answer
            seed_entities = set()
            for node in graph.nodes():
                if node.lower() in answer.lower() or answer.lower() in node.lower():
                    seed_entities.add(node)

            if not seed_entities:
                # Use top degree nodes as fallback
                degrees = dict(graph.degree())
                top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:5]
                seed_entities = set(top_nodes)

            # Extract large subgraph (min=20, max=50)
            large_entities, large_scores = await extractor.extract_subgraph_from_seeds(
                seed_entities=seed_entities,
                min_k=20,
                max_entities=50
            )

            # Extract small subgraph (min=3, max=10)
            small_entities, small_scores = await extractor.extract_subgraph_from_seeds(
                seed_entities=seed_entities,
                min_k=3,
                max_entities=10
            )

            return large_entities, large_scores, small_entities, small_scores

        large_entities, large_scores, small_entities, small_scores = asyncio.run(extract())

        # Save subgraphs
        large_dir = base_dir / "subgraphs_answer_large"
        small_dir = base_dir / "subgraphs_answer_small"
        large_dir.mkdir(parents=True, exist_ok=True)
        small_dir.mkdir(parents=True, exist_ok=True)

        # Convert to list of dicts for compatibility
        large_data = [{"entity_name": e, "score": large_scores.get(e, 0.0)} for e in large_entities]
        small_data = [{"entity_name": e, "score": small_scores.get(e, 0.0)} for e in small_entities]

        with open(large_dir / f"{doc_id}.json", 'w') as f:
            json.dump(large_data, f)
        with open(small_dir / f"{doc_id}.json", 'w') as f:
            json.dump(small_data, f)

        logger.info(f"  Large: {len(large_entities)} entities, Small: {len(small_entities)} entities")

    logger.info("Subgraph extraction completed")


def generate_queries(dataset_name: str, start_idx: int, end_idx: int):
    """Generate queries from subgraphs (both small and large) in parallel."""
    import json
    import yaml
    from pathlib import Path
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from src.kg.utils.llm_client import load_llm_config

    logger.info("=" * 60)
    logger.info("STEP 2: GENERATING QUERIES")
    logger.info("=" * 60)

    # Load query generation config
    config_path = Path('./src/config/query_generation.yaml')
    with open(config_path, 'r') as f:
        query_config = yaml.safe_load(f)
    max_workers = query_config.get('generation', {}).get('max_workers', 10)

    # Load LLM config from llm_api.yaml
    llm_config = load_llm_config()
    gpt_config = llm_config.get('gpt', {})

    # Initialize query generator with GPT config (OpenAI-compatible API)
    query_generator = QueryGenerator(
        llm_provider="openai",
        llm_model=gpt_config.get('model', 'gemini-2.5-flash'),
        llm_config={'gpt': gpt_config}
    )

    base_dir = Path(f'./data/preprocessed/{dataset_name}')
    logger.info(f"Generating queries for {dataset_name} from {start_idx} to {end_idx}")

    def process_single(subgraph_type: str, idx: int):
        """Process a single subgraph file."""
        doc_id = f"{dataset_name}_{idx}"
        subgraph_dir = base_dir / f"subgraphs_answer_{subgraph_type}"
        output_dir = base_dir / f"queries_{subgraph_type}"
        output_dir.mkdir(parents=True, exist_ok=True)

        subgraph_path = subgraph_dir / f"{doc_id}.json"
        alignment_path = base_dir / "alignment" / f"{doc_id}.json"

        if not subgraph_path.exists() or not alignment_path.exists():
            return (doc_id, subgraph_type, 0, "missing")

        with open(alignment_path, 'r') as f:
            alignment_data = json.load(f)

        queries = query_generator.process_subgraph_file(
            subgraph_path=str(subgraph_path),
            original_data=alignment_data
        )

        output_path = output_dir / f"{doc_id}.json"
        with open(output_path, 'w') as f:
            json.dump(queries, f, indent=2)

        return (doc_id, subgraph_type, len(queries) if queries else 0, "ok")

    # Build task list: (subgraph_type, idx) for all combinations
    tasks = []
    for subgraph_type in ['large', 'small']:
        for idx in range(start_idx, end_idx):
            tasks.append((subgraph_type, idx))

    # Process in parallel (max_workers from config)
    workers = min(max_workers, len(tasks))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_single, t[0], t[1]): t for t in tasks}
        for future in as_completed(futures):
            doc_id, subgraph_type, num_queries, status = future.result()
            if status == "missing":
                logger.warning(f"Missing files for {doc_id} ({subgraph_type})")
            else:
                logger.info(f"  {doc_id} ({subgraph_type}): {num_queries} queries")

    logger.info("Query generation completed")


def match_chunks(dataset_name: str, config_path: str):
    """Match queries to chunks and save results (both small and large).

    For each entry:
    1. Build corpus from that entry's chunks only
    2. Retrieve top-k chunks for each synthetic query
    3. Take union of retrieved chunks (deduplicate)
    4. Save results
    """
    logger.info("=" * 60)
    logger.info("STEP 3: MATCHING CHUNKS")
    logger.info("=" * 60)

    config = load_config(config_path)
    preprocessed_dir = config['paths']['data_root'] + '/preprocessed'

    # Load alignment data
    logger.info(f"Loading alignment data for {dataset_name}...")
    alignment_data = load_alignment_data(dataset_name, preprocessed_dir, limit=None)

    if not alignment_data:
        logger.error(f"No alignment data found for {dataset_name}")
        return

    # Load queries data
    large_queries = load_queries_data(dataset_name, 'large', preprocessed_dir, limit=None)
    small_queries = load_queries_data(dataset_name, 'small', preprocessed_dir, limit=None)

    # Initialize chunk retriever (model only, no corpus yet)
    model_path = config['training']['qwen']['model_name_or_path']
    device = config.get('device', 'cuda:0')
    logger.info(f"Initializing chunk retriever with model: {model_path}")
    retriever = ChunkRetriever(model_name_or_path=model_path, device=device)

    # Get retrieval top-k settings
    large_top_k = config['training']['qwen']['curriculum']['stage2'].get('retrieval_top_k', 100)
    small_top_k = config['training']['qwen']['curriculum']['stage3'].get('retrieval_top_k', 100)

    # Process each entry separately
    large_output_dir = Path(f'{preprocessed_dir}/{dataset_name}/matched_chunks_large')
    small_output_dir = Path(f'{preprocessed_dir}/{dataset_name}/matched_chunks_small')
    large_output_dir.mkdir(parents=True, exist_ok=True)
    small_output_dir.mkdir(parents=True, exist_ok=True)

    for idx, alignment in enumerate(tqdm(alignment_data, desc="Matching chunks per entry")):
        entry_chunks = alignment.get('chunk_list', [])
        if not entry_chunks:
            continue

        # Build corpus for this entry only
        retriever.build_corpus_from_chunks(entry_chunks)

        # Match large queries for this entry
        if large_queries and idx < len(large_queries) and large_queries[idx]:
            matched = retriever.retrieve_for_queries(large_queries[idx], top_k=large_top_k)
            output_file = large_output_dir / f'{dataset_name}_{idx}.json'
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(list(matched), f, ensure_ascii=False, indent=2)

        # Match small queries for this entry
        if small_queries and idx < len(small_queries) and small_queries[idx]:
            matched = retriever.retrieve_for_queries(small_queries[idx], top_k=small_top_k)
            output_file = small_output_dir / f'{dataset_name}_{idx}.json'
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(list(matched), f, ensure_ascii=False, indent=2)

    logger.info("Chunk matching completed")


def generate_training_data(stage: str, dataset_name: str, config_path: str):
    """Generate training data (Stage 1/2/3)."""
    logger.info("=" * 60)
    logger.info("STEP 4: GENERATING TRAINING DATA")
    logger.info("=" * 60)

    config = load_config(config_path)

    if stage in ['stage1', 'all']:
        prepare_stage1(config, dataset_name)

    if stage in ['stage2', 'all']:
        prepare_stage2(config, dataset_name)

    if stage in ['stage3', 'all']:
        prepare_stage3(config, dataset_name)


def main():
    parser = argparse.ArgumentParser(description="Complete ARK data generation pipeline")
    parser.add_argument('--dataset', type=str, default='fin',
                        help='Dataset name (e.g., hotpotqa, fin, biology)')
    parser.add_argument('--start', type=int, default=0,
                        help='Start index for processing')
    parser.add_argument('--end', type=int, default=100,
                        help='End index for processing')
    parser.add_argument('--stage', type=str, choices=['stage1', 'stage2', 'stage3', 'all'],
                        default='all', help='Which training stage to generate data for')
    parser.add_argument('--config', type=str, default='./src/config/training.yaml',
                        help='Path to training configuration')
    parser.add_argument('--skip-subgraphs', action='store_true',
                        help='Skip subgraph extraction')
    parser.add_argument('--skip-queries', action='store_true',
                        help='Skip query generation')
    parser.add_argument('--skip-matching', action='store_true',
                        help='Skip chunk matching')
    args = parser.parse_args()

    # Step 1: Extract subgraphs
    if not args.skip_subgraphs:
        extract_subgraphs(args.dataset, args.start, args.end)

    # Step 2: Generate queries
    if not args.skip_queries:
        generate_queries(args.dataset, args.start, args.end)

    # Step 3: Match chunks
    if not args.skip_matching:
        match_chunks(args.dataset, args.config)

    # Step 4: Generate training data
    generate_training_data(args.stage, args.dataset, args.config)

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
