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
from src.training.ops.data_loader import load_alignment_data, load_queries_data, load_matched_chunks_data, load_full_chunks_data
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


def prepare_stage1(config, dataset_name: str, alignment_dir: str = None, num_positives: int = None, num_negatives: int = None):
    """Prepare Stage 1 training data."""
    logger.info("=" * 60)
    logger.info("PREPARING STAGE 1 DATA")
    logger.info("=" * 60)

    stage_cfg = config.get('training_data', {}).get('stage1', {})
    n_pos = num_positives or stage_cfg['num_positives']
    n_neg = num_negatives or stage_cfg['num_negatives']

    alignment_data = load_alignment_data(dataset_name, alignment_dir=alignment_dir)

    full_chunks_data = []
    if alignment_dir:
        full_chunks_data = load_full_chunks_data(dataset_name, alignment_dir)

    if not alignment_data:
        logger.error(f"No alignment data loaded for {dataset_name}!")
        return 0

    if alignment_dir:
        output_file = f"{alignment_dir}/training/stage1.jsonl"
    else:
        output_file = f"./data/training/stage1/{dataset_name}_train.jsonl"

    num_samples = generate_stage1_data(
        alignment_data=alignment_data,
        output_file=output_file,
        num_positives=n_pos,
        num_negatives=n_neg,
        full_chunks_data=full_chunks_data
    )

    return num_samples


def prepare_stage2(config, dataset_name: str, alignment_dir: str = None, num_positives: int = None, num_negatives: int = None):
    """Prepare Stage 2 training data."""
    logger.info("=" * 60)
    logger.info("PREPARING STAGE 2 DATA")
    logger.info("=" * 60)

    stage_cfg = config.get('training_data', {}).get('stage2', {})
    n_pos = num_positives or stage_cfg['num_positives']
    n_neg = num_negatives or stage_cfg['num_negatives']

    alignment_data = load_alignment_data(dataset_name, alignment_dir=alignment_dir)
    matched_chunks_data = load_matched_chunks_data(dataset_name, 'large', alignment_dir=alignment_dir)

    if not alignment_data or not matched_chunks_data:
        logger.error(f"No data loaded for Stage 2 ({dataset_name})!")
        return 0

    if alignment_dir:
        output_file = f"{alignment_dir}/training/stage2.jsonl"
    else:
        output_file = f"./data/training/stage2/{dataset_name}_train.jsonl"

    num_samples = generate_stage2_data(
        alignment_data=alignment_data,
        matched_chunks_data=matched_chunks_data,
        output_file=output_file,
        num_positives=n_pos,
        num_negatives=n_neg
    )

    return num_samples


def prepare_stage3(config, dataset_name: str, alignment_dir: str = None, num_positives: int = None, num_negatives: int = None):
    """Prepare Stage 3 training data."""
    logger.info("=" * 60)
    logger.info("PREPARING STAGE 3 DATA")
    logger.info("=" * 60)

    stage_cfg = config.get('training_data', {}).get('stage3', {})
    n_pos = num_positives or stage_cfg['num_positives']
    n_neg = num_negatives or stage_cfg['num_negatives']

    alignment_data = load_alignment_data(dataset_name, alignment_dir=alignment_dir)
    matched_chunks_data = load_matched_chunks_data(dataset_name, 'small', alignment_dir=alignment_dir)

    if not alignment_data or not matched_chunks_data:
        logger.error(f"No data loaded for Stage 3 ({dataset_name})!")
        return 0

    if alignment_dir:
        output_file = f"{alignment_dir}/training/stage3.jsonl"
    else:
        output_file = f"./data/training/stage3/{dataset_name}_train.jsonl"

    num_samples = generate_stage3_data(
        alignment_data=alignment_data,
        matched_chunks_data=matched_chunks_data,
        output_file=output_file,
        num_positives=n_pos,
        num_negatives=n_neg
    )

    return num_samples


def extract_subgraphs(dataset_name: str, start_idx: int, end_idx: int,
                      kg_dir: str, data_path: str,
                      align_config_path: str = "./src/config/alignment.yaml"):
    """Extract subgraphs from KG using async LLM entity extraction.

    Args:
        kg_dir: KG directory (e.g. data/preprocessed/kg/gemini-2.5-flash-nothinking/fin)
        data_path: Raw JSONL file with 'input' and 'answers' fields
    """
    import pickle
    import asyncio
    from src.kg.ops.extraction import (
        SubgraphExtractor, SubgraphConfig, PPRConfig,
        extract_and_match_entities, _llm_extract_entities,
    )

    logger.info("=" * 60)
    logger.info("STEP 1: EXTRACTING SUBGRAPHS (LLM-based)")
    logger.info("=" * 60)

    with open(align_config_path, 'r') as f:
        ppr_cfg = yaml.safe_load(f)['ppr']

    base_dir = Path(kg_dir)

    # Load raw data for answers
    raw_data = {}
    with open(data_path) as f:
        for line_idx, line in enumerate(f):
            item = json.loads(line)
            ans = item.get('answers', [''])
            raw_data[line_idx] = ans[0] if isinstance(ans, list) else ans

    # Load context map for dedup resolution
    context_map_path = base_dir / "context_map.json"
    context_map = {}
    hash_to_representative = {}
    if context_map_path.exists():
        with open(context_map_path) as f:
            context_map = json.load(f)
        for k, h in sorted(context_map.items(), key=lambda x: int(x[0])):
            if h not in hash_to_representative:
                hash_to_representative[h] = int(k)
        logger.info(f"Loaded context_map: {len(context_map)} docs -> {len(hash_to_representative)} unique")

    # Collect valid (idx, graph, answer) tuples
    items = []
    for idx in range(start_idx, end_idx):
        if idx not in raw_data:
            continue
        doc_id = f"{dataset_name}_{idx}"
        kg_doc_id = doc_id
        if str(idx) in context_map:
            rep_idx = hash_to_representative[context_map[str(idx)]]
            if rep_idx != idx:
                kg_doc_id = f"{dataset_name}_{rep_idx}"

        kg_path = base_dir / "full_kg_augmented" / f"{kg_doc_id}.pkl"
        if not kg_path.exists():
            kg_path = base_dir / "full_kg" / f"{kg_doc_id}.pkl"
        if not kg_path.exists():
            logger.warning(f"KG not found for {kg_doc_id}, skipping")
            continue

        with open(kg_path, 'rb') as f:
            graph = pickle.load(f)

        items.append((idx, doc_id, graph, raw_data[idx]))

    logger.info(f"Collected {len(items)} valid items for entity extraction")

    # Try reuse existing answer_entities.json
    ans_ent_path = base_dir / "answer_entities.json"
    cached_ans = {}
    if ans_ent_path.exists():
        with open(ans_ent_path) as f:
            cached_ans = json.load(f)
        logger.info(f"Loaded cached answer_entities.json ({len(cached_ans)} entries)")

    # Split items into cached vs need-extraction
    need_extract = [(i, it) for i, it in enumerate(items) if str(it[0]) not in cached_ans]
    logger.info(f"Need LLM extraction: {len(need_extract)}, cached: {len(items) - len(need_extract)}")

    # Extract only missing ones
    if need_extract:
        async def run_missing():
            return await asyncio.gather(
                *[_llm_extract_entities(items[i][3]) for i, _ in need_extract]
            )
        new_ents = asyncio.run(run_missing())
        for (i, _), ents in zip(need_extract, new_ents):
            cached_ans[str(items[i][0])] = ents
        with open(ans_ent_path, 'w') as f:
            json.dump(cached_ans, f, indent=2, ensure_ascii=False)
        logger.info(f"Updated answer_entities.json ({len(cached_ans)} entries)")

    # Match extracted entities to graph nodes (with Jaccard fallback)
    async def run_matching():
        from src.kg.ops.extraction import match_entities_exact, _jaccard_similarity
        results = []
        for idx, doc_id, graph, answer in items:
            extracted = cached_ans.get(str(idx), [])
            if not extracted:
                results.append(set())
                continue
            matched = await match_entities_exact(extracted, graph)
            if not matched:
                graph_nodes = list(graph.nodes())
                scored = []
                for ent in extracted:
                    for node in graph_nodes:
                        sim = _jaccard_similarity(ent, node)
                        if sim > 0:
                            scored.append((node, sim))
                scored.sort(key=lambda x: x[1], reverse=True)
                matched = {n for n, _ in scored[:5]}
                if matched:
                    logger.info(f"Jaccard fallback matched {len(matched)} for {doc_id}")
            results.append(matched)
        return results

    seed_sets = asyncio.run(run_matching())

    # Try reuse existing query_entities.json
    qry_ent_path = base_dir / "query_entities.json"
    cached_qry = {}
    if qry_ent_path.exists():
        with open(qry_ent_path) as f:
            cached_qry = json.load(f)
        logger.info(f"Loaded cached query_entities.json ({len(cached_qry)} entries)")

    with open(data_path) as f:
        all_lines = f.readlines()

    need_qry = [idx for idx in range(start_idx, end_idx)
                 if idx < len(all_lines) and str(idx) not in cached_qry]
    logger.info(f"Query entities: need LLM={len(need_qry)}, cached={end_idx - start_idx - len(need_qry)}")

    if need_qry:
        async def run_qry_missing():
            return await asyncio.gather(
                *[_llm_extract_entities(json.loads(all_lines[i]).get('input', ''))
                  for i in need_qry]
            )
        new_qry = asyncio.run(run_qry_missing())
        for idx, ents in zip(need_qry, new_qry):
            cached_qry[str(idx)] = ents
        with open(qry_ent_path, 'w') as f:
            json.dump(cached_qry, f, indent=2, ensure_ascii=False)
        logger.info(f"Updated query_entities.json ({len(cached_qry)} entries)")

    # Now run PPR and save for each item
    for (idx, doc_id, graph, answer), seed_entities in zip(items, seed_sets):
        ppr_config = PPRConfig(
            alpha=ppr_cfg['alpha'], epsilon=ppr_cfg['epsilon'],
            max_iterations=ppr_cfg['max_iterations'],
            adaptive_cutoff=ppr_cfg['adaptive_cutoff'],
        )
        config = SubgraphConfig(ppr=ppr_config)
        extractor = SubgraphExtractor(graph=graph, config=config)

        if not seed_entities:
            logger.warning(f"No entities matched for {doc_id}, skipping")
            continue

        async def extract_both():
            large_ent, large_sc = await extractor.extract_subgraph_from_seeds(
                seed_entities, min_k=ppr_cfg['large']['min_k'],
                max_entities=ppr_cfg['large']['max_k'],
            )
            small_ent, small_sc = await extractor.extract_subgraph_from_seeds(
                seed_entities, min_k=ppr_cfg['small']['min_k'],
                max_entities=ppr_cfg['small']['max_k'],
            )
            return large_ent, large_sc, small_ent, small_sc

        large_entities, large_scores, small_entities, small_scores = asyncio.run(extract_both())

        # Save subgraphs with entity attributes
        for size_tag, entities, scores in [
            ("large", large_entities, large_scores),
            ("small", small_entities, small_scores),
        ]:
            entity_set = set(entities)
            base_out = base_dir / f"subgraphs_answer_{size_tag}"
            for variant in ("subgraph", "full"):
                out_dir = base_out / variant
                out_dir.mkdir(parents=True, exist_ok=True)
                with open(out_dir / f"{doc_id}.jsonl", 'w') as f:
                    for e in entities:
                        node_attrs = dict(graph.nodes[e]) if e in graph else {}
                        edges = []
                        if e in graph:
                            for nb in graph.neighbors(e):
                                if variant == "full" or nb in entity_set:
                                    edges.append({"target": nb, **dict(graph.edges[e, nb])})
                        row = {
                            "entity_name": e,
                            "score": scores.get(e, 0.0),
                            "entity_type": node_attrs.get("entity_type", ""),
                            "description": node_attrs.get("description", ""),
                            "edges": edges,
                        }
                        f.write(json.dumps(row, ensure_ascii=False, indent=2) + '\n')

        logger.info(f"  {doc_id}: Large={len(large_entities)}, Small={len(small_entities)}")

    logger.info("Subgraph extraction completed")


def generate_queries(dataset_name: str, start_idx: int, end_idx: int,
                     kg_dir: str, data_path: str, query_num: int = None):
    """Generate queries from subgraphs (both large and small) in parallel.

    Args:
        kg_dir: KG directory containing subgraphs_answer_{large,small}/
        data_path: Raw JSONL file with 'input' and 'answers' fields
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from src.kg.utils.llm_client import load_llm_config

    logger.info("=" * 60)
    logger.info("STEP 2: GENERATING QUERIES")
    logger.info("=" * 60)

    config_path = Path('./src/config/alignment.yaml')
    with open(config_path, 'r') as f:
        query_config = yaml.safe_load(f)
    max_workers = query_config['query_generation']['max_workers']
    num_queries = query_num or query_config['query_generation']['num_queries']

    llm_config = load_llm_config()
    gpt_config = llm_config.get('gpt', {})

    query_generator = QueryGenerator(
        llm_provider="openai",
        llm_model=gpt_config['model'],
        num_queries=num_queries,
        llm_config={'gpt': gpt_config}
    )

    base_dir = Path(kg_dir)

    # Load raw data for questions/answers
    raw_lines = {}
    with open(data_path) as f:
        for i, line in enumerate(f):
            raw_lines[i] = json.loads(line)

    logger.info(f"Generating queries for {dataset_name} [{start_idx},{end_idx})")

    def process_single(subgraph_type: str, idx: int):
        doc_id = f"{dataset_name}_{idx}"
        subgraph_path = base_dir / f"subgraphs_answer_{subgraph_type}" / "subgraph" / f"{doc_id}.jsonl"
        output_dir = base_dir / f"queries_{subgraph_type}"
        output_dir.mkdir(parents=True, exist_ok=True)

        if not subgraph_path.exists() or idx not in raw_lines:
            return (doc_id, subgraph_type, 0, "missing")

        original_data = raw_lines[idx]
        queries = query_generator.process_subgraph_file(
            subgraph_path=str(subgraph_path),
            original_data=original_data
        )

        with open(output_dir / f"{doc_id}.json", 'w') as f:
            json.dump(queries, f, indent=2)

        return (doc_id, subgraph_type, len(queries) if queries else 0, "ok")

    tasks = [(st, idx) for st in ['large', 'small']
             for idx in range(start_idx, end_idx)]

    workers = min(max_workers, len(tasks))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_single, t[0], t[1]): t for t in tasks}
        for future in as_completed(futures):
            doc_id, subgraph_type, nq, status = future.result()
            if status == "missing":
                logger.warning(f"Missing files for {doc_id} ({subgraph_type})")
            else:
                logger.info(f"  {doc_id} ({subgraph_type}): {nq} queries")

    logger.info("Query generation completed")


def match_chunks(dataset_name: str, config_path: str,
                 start_idx: int = 0, end_idx: int = None,
                 kg_dir: str = None, alignment_dir: str = None,
                 align_config_path: str = "./src/config/alignment.yaml"):
    """Match queries to chunks, using same chunking as pos pipeline.

    Args:
        kg_dir: KG directory containing queries_{large,small}/
        alignment_dir: Base alignment dir (e.g. .../sentence_5_o_1/),
                       contains chunk/, topk/score/, matched_chunks output goes here too
    """
    logger.info("=" * 60)
    logger.info(f"MATCHING CHUNKS (shard {start_idx}-{end_idx})")
    logger.info("=" * 60)

    config = load_config(config_path)

    with open(align_config_path, 'r') as f:
        align_cfg = yaml.safe_load(f)
    top_k = align_cfg.get('top_k', 10)

    base_path = Path(alignment_dir) if alignment_dir else (
        Path("./data/preprocessed") / dataset_name / "alignment")
    chunk_dir = base_path / "chunk"
    pos_dir = base_path / "topk" / "score"

    if not chunk_dir.exists():
        logger.error(f"Chunk directory not found: {chunk_dir}")
        return

    # Count available files for end_idx
    chunk_files = sorted(chunk_dir.glob("*.json"))
    if not chunk_files:
        logger.error(f"No chunk files in {chunk_dir}")
        return
    max_idx = max(int(f.stem.split('_')[-1]) for f in chunk_files) + 1
    if end_idx is None:
        end_idx = max_idx
    end_idx = min(end_idx, max_idx)

    # Load queries from kg_dir
    kg_path = Path(kg_dir) if kg_dir else Path("./data/preprocessed") / dataset_name
    large_queries, small_queries = {}, {}
    for qt, qdict in [('large', large_queries), ('small', small_queries)]:
        qdir = kg_path / f"queries_{qt}"
        if qdir.exists():
            for f in sorted(qdir.glob("*.json")):
                idx = int(f.stem.split('_')[-1])
                with open(f) as fh:
                    qdict[idx] = json.load(fh)
    logger.info(f"Loaded queries: large={len(large_queries)}, small={len(small_queries)}")

    model_path = config['model_name_or_path']
    device = config['device']
    retriever = ChunkRetriever(model_name_or_path=model_path, device=device)

    large_output_dir = base_path / "matched_chunks_large"
    small_output_dir = base_path / "matched_chunks_small"
    large_output_dir.mkdir(parents=True, exist_ok=True)
    small_output_dir.mkdir(parents=True, exist_ok=True)

    for idx in tqdm(range(start_idx, end_idx), desc=f"Matching chunks [{start_idx},{end_idx})"):
        doc_id = f"{dataset_name}_{idx}"

        # Load full corpus from chunk/ dir
        corpus_file = chunk_dir / f"{doc_id}.json"
        if not corpus_file.exists():
            continue
        with open(corpus_file) as fh:
            corpus_chunks = json.load(fh).get('chunk_list', [])
        if not corpus_chunks:
            continue

        # Load pos chunks from topk/score/ dir
        pos_file = pos_dir / f"{doc_id}.json"
        pos_set = set()
        if pos_file.exists():
            with open(pos_file) as fh:
                for c in json.load(fh).get('chunk_list', []):
                    pos_set.add(c['content'] if isinstance(c, dict) else c)

        retriever.build_corpus_from_chunks(corpus_chunks)

        for qt, queries, out_dir in [
            ('large', large_queries, large_output_dir),
            ('small', small_queries, small_output_dir),
        ]:
            if idx not in queries or not queries[idx]:
                continue
            matched_raw = retriever.retrieve_for_queries(queries[idx], top_k=top_k)
            # Exclude pos chunks (all raw, cleaning deferred to training data generation)
            neg_chunks = [c for c in matched_raw if c not in pos_set]
            with open(out_dir / f'{doc_id}.json', 'w', encoding='utf-8') as f:
                json.dump(neg_chunks, f, ensure_ascii=False, indent=2)

    logger.info("Chunk matching completed")


def generate_training_data(stage: str, dataset_name: str, align_config_path: str,
                           alignment_dir: str = None, num_positives: int = None,
                           num_negatives: int = None):
    """Generate training data (Stage 1/2/3)."""
    logger.info("=" * 60)
    logger.info("STEP 4: GENERATING TRAINING DATA")
    logger.info("=" * 60)

    with open(align_config_path, 'r') as f:
        config = yaml.safe_load(f)

    if stage in ['stage1', 'all']:
        prepare_stage1(config, dataset_name, alignment_dir=alignment_dir, num_positives=num_positives, num_negatives=num_negatives)

    if stage in ['stage2', 'all']:
        prepare_stage2(config, dataset_name, alignment_dir=alignment_dir, num_positives=num_positives, num_negatives=num_negatives)

    if stage in ['stage3', 'all']:
        prepare_stage3(config, dataset_name, alignment_dir=alignment_dir, num_positives=num_positives, num_negatives=num_negatives)


def main():
    parser = argparse.ArgumentParser(description="ARK data generation pipeline")
    parser.add_argument('--mode', type=str,
                        choices=['community', 'queries', 'matching', 'training'],
                        required=True, help='Pipeline mode to run')
    parser.add_argument('--dataset', type=str, default='fin')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=100)
    parser.add_argument('--kg_dir', type=str, default=None,
                        help='KG directory (e.g. data/preprocessed/kg/gemini-2.5-flash-nothinking/fin)')
    parser.add_argument('--data', type=str, default=None,
                        help='Raw JSONL data file path')
    parser.add_argument('--config', type=str, default='./src/config/training.yaml')
    parser.add_argument('--align_config', type=str, default='./src/config/alignment.yaml')
    parser.add_argument('--max_async_calls', type=int, default=10,
                        help='Max concurrent async LLM calls')
    parser.add_argument('--query_num', type=int, default=None,
                        help='Override num_queries per subgraph')
    parser.add_argument('--alignment_dir', type=str, default=None,
                        help='Alignment data directory')
    parser.add_argument('--num_positives', type=int, default=None,
                        help='Override num_positives (top-k chunks as positives)')
    parser.add_argument('--num_negatives', type=int, default=None,
                        help='Override num_negatives per sample')
    parser.add_argument('--stage', type=str,
                        choices=['stage1', 'stage2', 'stage3', 'all'],
                        default='all')
    args = parser.parse_args()

    # Set async concurrency
    from src.kg.utils import llm_client
    llm_client.MAX_ASYNC_CALL_SIZE = args.max_async_calls

    # Resolve data path: try raw/ then raw/additional/
    if args.data and not Path(args.data).exists():
        alt = args.data.replace("/data/raw/", "/data/raw/additional/")
        if Path(alt).exists():
            args.data = alt
        else:
            raise FileNotFoundError(f"Not found: {args.data} or {alt}")

    if args.mode == 'community':
        extract_subgraphs(args.dataset, args.start, args.end,
                         kg_dir=args.kg_dir, data_path=args.data,
                         align_config_path=args.align_config)

    elif args.mode == 'queries':
        generate_queries(args.dataset, args.start, args.end,
                         kg_dir=args.kg_dir, data_path=args.data,
                         query_num=args.query_num)

    elif args.mode == 'matching':
        match_chunks(args.dataset, args.config, args.start, args.end,
                     kg_dir=args.kg_dir, alignment_dir=args.alignment_dir,
                     align_config_path=args.align_config)

    elif args.mode == 'training':
        generate_training_data(args.stage, args.dataset, args.align_config,
                               alignment_dir=args.alignment_dir,
                               num_positives=args.num_positives,
                               num_negatives=args.num_negatives)

    logger.info("DONE")


if __name__ == '__main__':
    main()
