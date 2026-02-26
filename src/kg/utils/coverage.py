"""Query-to-KG entity coverage test using LLM entity extraction."""

import asyncio
import pickle
import json
import argparse
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.kg.utils.llm_client import gpt_model
from src.kg.utils import llm_client
from src.kg.prompts.entity_extraction import PROMPTS


def _clean_node(name: str) -> str:
    s = name.strip()
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        s = s[1:-1]
    return s


async def extract_query_entities(query: str) -> list[str]:
    """Call LLM to extract entities from a query, return uppercased list."""
    prompt = PROMPTS["query_entity_extraction"].format(input_text=query)
    response = await gpt_model(prompt)
    # Parse JSON list from response
    text = response.strip()
    # Find JSON array in response
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        return []
    entities = json.loads(text[start:end + 1])
    return [e.upper().strip() for e in entities if isinstance(e, str)]


def coverage_single(entities: list[str], graph) -> dict:
    """Check how many extracted entities match KG nodes via .upper() exact match."""
    kg_nodes = {_clean_node(n).upper() for n in graph.nodes()}
    matched = [e for e in entities if e in kg_nodes]
    unmatched = [e for e in entities if e not in kg_nodes]
    return {
        "num_entities": len(entities),
        "num_matched": len(matched),
        "coverage": round(len(matched) / len(entities), 6) if entities else 0.0,
        "has_match": len(matched) > 0,
        "matched": matched,
        "unmatched": unmatched,
        "kg_nodes_count": len(kg_nodes),
    }


async def coverage_test(data_path, kg_dir, dataset, mode="query",
                        start_index=0, end_index=100):
    # Load texts (query or answer)
    texts = {}
    with open(data_path) as f:
        for idx, line in enumerate(f):
            if idx < start_index:
                continue
            if idx >= end_index:
                break
            item = json.loads(line)
            if mode == "answer":
                ans = item["answers"]
                texts[idx] = ans[0] if isinstance(ans, list) else ans
            else:
                texts[idx] = item["input"]

    kg_dir = Path(kg_dir)
    base_dir = kg_dir.parent  # data/preprocessed/{MODEL}/{DATASET}/

    # Load context_map for dedup resolution
    context_map_path = base_dir / "context_map.json"
    hash_to_rep = {}
    context_map = {}
    if context_map_path.exists():
        with open(context_map_path) as f:
            context_map = json.load(f)
        for k, h in sorted(context_map.items(), key=lambda x: int(x[0])):
            if h not in hash_to_rep:
                hash_to_rep[h] = int(k)
        print(f"Loaded context_map: {len(context_map)} docs -> {len(hash_to_rep)} unique")

    def resolve_kg_path(idx):
        """Resolve idx to representative KG pkl via context_map."""
        rep_idx = idx
        if str(idx) in context_map:
            rep_idx = hash_to_rep[context_map[str(idx)]]
        return kg_dir / f"{dataset}_{rep_idx}.pkl"

    # Filter to indices that have both text and resolved KG
    valid = [(idx, texts[idx]) for idx in range(start_index, end_index)
             if idx in texts and resolve_kg_path(idx).exists()]

    # Parallel LLM entity extraction
    print(f"Extracting entities for {len(valid)} {mode}s in parallel...")
    all_entities = await asyncio.gather(
        *[extract_query_entities(t) for _, t in valid])

    entities_map = {str(idx): ents for (idx, _), ents in zip(valid, all_entities)}

    # Match against KG
    per_sample = []
    for (idx, text), entities in zip(valid, all_entities):
        with open(resolve_kg_path(idx), "rb") as f:
            G = pickle.load(f)
        cov = coverage_single(entities, G)
        cov["index"] = idx
        cov["text"] = text
        cov["entities"] = entities
        per_sample.append(cov)
        print(f"  [{idx}] matched={cov['num_matched']}/{cov['num_entities']} "
              f"coverage={cov['coverage']:.4f}  "
              f"matched={cov['matched']}  unmatched={cov['unmatched']}")

    # Save {mode}_entities.json alongside context_map.json
    entities_path = base_dir / f"{mode}_entities.json"
    with open(entities_path, "w") as f:
        json.dump(entities_map, f, indent=2, ensure_ascii=False)
    print(f"Saved {mode} entities to {entities_path}")

    if not per_sample:
        print("No samples processed.")
        return {"aggregate": {}, "per_sample": []}

    hit = sum(1 for s in per_sample if s["has_match"])
    coverages = [s["coverage"] for s in per_sample]
    num_ents = [s["num_entities"] for s in per_sample]
    agg = {
        "num_samples": len(per_sample),
        "hit_rate": round(hit / len(per_sample), 4),
        "avg_coverage": round(float(np.mean(coverages)), 6),
        "median_coverage": round(float(np.median(coverages)), 6),
        "avg_entities_per_query": round(float(np.mean(num_ents)), 2),
        "total_entities": int(np.sum(num_ents)),
        "zero_match_count": len(per_sample) - hit,
    }

    print(f"\n{'='*60}")
    print(f"Samples={agg['num_samples']}, hit_rate={agg['hit_rate']}")
    print(f"Avg coverage={agg['avg_coverage']}, median={agg['median_coverage']}")
    print(f"Avg entities/query={agg['avg_entities_per_query']}, "
          f"total={agg['total_entities']}, zero_match={agg['zero_match_count']}")
    print(f"{'='*60}")

    return {"aggregate": agg, "per_sample": per_sample}


def main():
    parser = argparse.ArgumentParser(description="Query-to-KG entity coverage test")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--kg_dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=["query", "answer"], default="query")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=100)
    parser.add_argument("--max_async_calls", type=int, default=10)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    llm_client.MAX_ASYNC_CALL_SIZE = args.max_async_calls

    result = asyncio.run(coverage_test(
        args.data, args.kg_dir, args.dataset,
        args.mode, args.start_index, args.end_index))

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"Saved to {out}")


if __name__ == "__main__":
    main()
