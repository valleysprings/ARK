#!/usr/bin/env python3
"""
ARK Unified Inference Script
Runs inference using UnifiedRetriever with experiment config
"""

import argparse
import json
import yaml
import torch
from pathlib import Path
from typing import Dict, Any

from src.inference.unified_retriever import (
    UnifiedRetriever,
    VLLMGenerator,
    text_to_chunks,
    llm_generate
)
from src.inference.metrics import qa_f1_score
from src.inference.prompts.dataset_prompts import DATASET2PROMPT, ULTRADOMAIN_PROMPT


def load_config(llm_config_path: str = None, retrieval_config_path: str = None, exp_config_path: str = None) -> Dict[str, Any]:
    """
    Load and merge configs

    Args:
        llm_config_path: Path to LLM config (src/config/llm.yaml)
        retrieval_config_path: Path to retrieval config (src/config/retrieval_model.yaml)
        exp_config_path: Path to experiment config (experiments/exp_*.yaml)

    Returns:
        Merged config dictionary
    """
    # Default paths
    if llm_config_path is None:
        llm_config_path = "src/config/llm.yaml"
    if retrieval_config_path is None:
        retrieval_config_path = "src/config/retrieval_model.yaml"

    # Load LLM config
    with open(llm_config_path, 'r') as f:
        llm_config = yaml.safe_load(f)

    # Load retrieval config
    with open(retrieval_config_path, 'r') as f:
        retrieval_config = yaml.safe_load(f)

    # Merge into unified structure
    config = {
        "paths": llm_config.get("paths", {}),
        "services": llm_config.get("services", {}),
        "inference": {
            "retriever": retrieval_config.get("retriever", {}),
            "generator": llm_config.get("generator", {}),
            "pipeline": llm_config.get("pipeline", {}),
        },
    }

    # Merge experiment config if provided
    if exp_config_path:
        with open(exp_config_path, 'r') as f:
            exp_config = yaml.safe_load(f)

        # Deep merge (simple implementation)
        if "inference" in exp_config:
            for key, value in exp_config["inference"].items():
                if key in config["inference"] and isinstance(config["inference"][key], dict):
                    config["inference"][key].update(value)
                else:
                    config["inference"][key] = value
        if "experiment" in exp_config:
            config["experiment"] = exp_config["experiment"]

    return config


def run_inference_single(config: Dict[str, Any], dataset_path: str, args, retriever, generator):
    """
    Run inference on a single dataset (models already loaded).
    """
    retriever_config = config["inference"]["retriever"]
    model_type = retriever_config["type"]
    model_config = retriever_config[model_type]
    chunk_size = model_config["chunk_size"]
    overlap = model_config["overlap"]

    dataset_name = Path(dataset_path).stem
    prompt_template = DATASET2PROMPT.get(dataset_name, ULTRADOMAIN_PROMPT)

    generator_config = config["inference"]["generator"]
    llm_model = generator_config["model"]

    print(f"\n" + "=" * 60)
    print(f"ARK Unified Inference")
    print(f"=" * 60)
    print(f"Retriever: {model_type.upper()}")
    print(f"Dataset: {dataset_name}")
    print(f"Chunk size: {chunk_size}, Overlap: {overlap}")
    print(f"LLM: {llm_model} (vLLM on {args.llm_device})")
    print(f"=" * 60)

    with open(dataset_path, 'r') as f:
        dataset = [json.loads(line) for line in f]

    if args.limit:
        dataset = dataset[:args.limit]

    results = []
    total_f1 = 0.0

    for idx, entry in enumerate(dataset, 1):
        context = entry.get("context", "")
        question = entry.get("input", entry.get("question", ""))
        answer = entry.get("answers", entry.get("ideal", entry.get("answer", "")))
        if isinstance(answer, list) and len(answer) > 0:
            answer = answer[0]

        if model_type == "no_retrieval":
            # Truncate context to fit max_model_len * 0.5 (reserve space for generation)
            max_model_len = generator_config["max_model_len"]
            max_context_words = int(max_model_len * 0.5)
            words = context.split()
            if len(words) > max_context_words:
                context = ' '.join(words[:max_context_words])
            context_text = context
            prompt = prompt_template.format(context=context_text, input=question)
            prediction = llm_generate(prompt, generator=generator)
            f1 = qa_f1_score(prediction, answer)
            total_f1 += f1
            result = {
                "question": question, "answer": answer, "prediction": prediction,
                "num_chunks_used": 0, "chunk_scores": [], "f1_score": f1
            }
            results.append(result)
            if idx % 10 == 0 or idx == len(dataset):
                print(f"[{idx}/{len(dataset)}] Avg F1: {total_f1 / idx:.4f}")
            continue

        chunks = text_to_chunks(context, chunk_size, overlap)
        if not chunks:
            print(f"[{idx}/{len(dataset)}] No chunks for entry, skipping")
            continue

        top_chunks = retriever.retrieve(question, chunks)
        selected_chunks = [chunks[chunk_idx] for chunk_idx, score in top_chunks]
        context_text = "\n\n".join(selected_chunks)
        prompt = prompt_template.format(context=context_text, input=question)
        prediction = llm_generate(prompt, generator=generator)
        f1 = qa_f1_score(prediction, answer)
        total_f1 += f1
        result = {
            "question": question, "answer": answer, "prediction": prediction,
            "num_chunks_used": len(selected_chunks),
            "chunk_scores": [score for _, score in top_chunks],
            "retrieved_chunks": selected_chunks, "f1_score": f1
        }
        results.append(result)
        if idx % 10 == 0 or idx == len(dataset):
            print(f"[{idx}/{len(dataset)}] Avg F1: {total_f1 / idx:.4f}")

    # Save results
    limit_prefix = f"sample_{args.limit}" if args.limit else "full"
    raw_dir = Path("results/raw") / limit_prefix / dataset_name
    score_dir = Path("results/score") / limit_prefix / dataset_name
    raw_dir.mkdir(parents=True, exist_ok=True)
    score_dir.mkdir(parents=True, exist_ok=True)

    output_name = f"{model_type}_{args.model_suffix}" if args.model_suffix else model_type
    raw_output = raw_dir / f"{output_name}.jsonl"
    with open(raw_output, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

    avg_f1 = total_f1 / len(results) if results else 0
    score_data = {
        "model": model_type, "dataset": dataset_name,
        "limit": args.limit if args.limit else "full",
        "total_samples": len(results), "f1_score": round(avg_f1, 4)
    }
    score_output = score_dir / f"{output_name}.json"
    with open(score_output, 'w') as f:
        json.dump(score_data, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"[{dataset_name}] Done! Samples: {len(results)}, Avg F1: {avg_f1:.4f}")
    print(f"  Raw: {raw_output}")
    print(f"  Score: {score_output}")
    print(f"{'=' * 60}")

    return score_data


def run_inference(config: Dict[str, Any], dataset_path: str, output_path: str, args=None):
    """
    Run inference on one or multiple datasets.
    If dataset_path is a directory, iterate over all .jsonl files (models loaded once).
    """
    # Collect dataset files
    dp = Path(dataset_path)
    if dp.is_dir():
        dataset_files = sorted(dp.glob("*.jsonl"))
    else:
        dataset_files = [dp]

    if not dataset_files:
        print(f"No .jsonl files found in {dataset_path}")
        return

    # Initialize models once
    retriever = UnifiedRetriever(config, device=args.device)

    generator_config = config["inference"]["generator"]
    llm_model = generator_config["model"]
    generator = VLLMGenerator(
        model_path=llm_model,
        device=args.llm_device,
        max_tokens=generator_config["max_new_tokens"],
        max_model_len=generator_config["max_model_len"],
        temperature=generator_config["temperature"],
        top_p=generator_config["top_p"],
    )

    print(f"Loaded models. Running on {len(dataset_files)} dataset(s)...")

    # Run inference on each dataset
    all_scores = []
    for ds_file in dataset_files:
        score = run_inference_single(config, str(ds_file), args, retriever, generator)
        all_scores.append(score)

    # Clean up
    generator.shutdown()
    del retriever
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU memory cleared")

    # Print overall summary
    if len(all_scores) > 1:
        print(f"\n" + "=" * 60)
        print(f"All Datasets Complete!")
        print(f"=" * 60)
        for s in all_scores:
            print(f"  {s['dataset']:20s} F1: {s['f1_score']:.4f}  ({s['total_samples']} samples)")
        overall_f1 = sum(s['f1_score'] for s in all_scores) / len(all_scores)
        print(f"  {'AVERAGE':20s} F1: {overall_f1:.4f}")
        print(f"=" * 60)


def main():
    parser = argparse.ArgumentParser(description="ARK Unified Inference")
    parser.add_argument(
        "--exp-config",
        type=str,
        required=False,
        default=None,
        help="Experiment config (e.g., experiments/exp_qwen.yaml)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to dataset JSONL file"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=False,
        default=None,
        help="(Deprecated) Output path - now auto-determined from dataset and model"
    )
    parser.add_argument(
        "--llm-config",
        type=str,
        default="src/config/llm.yaml",
        help="LLM config file (default: src/config/llm.yaml)"
    )
    parser.add_argument(
        "--retrieval-config",
        type=str,
        default="src/config/retrieval_model.yaml",
        help="Retrieval model config file (default: src/config/retrieval_model.yaml)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="CUDA device for retriever"
    )
    parser.add_argument(
        "--llm-device",
        type=str,
        default="cuda:1",
        help="CUDA device for vLLM LLM server"
    )
    parser.add_argument(
        "--retriever",
        type=str,
        required=False,
        default=None,
        help="Retriever type (e.g., qwen, bge, jina, stella) - overrides config"
    )
    parser.add_argument(
        "--limit",
        type=int,
        required=False,
        default=None,
        help="Limit number of samples to process (for testing)"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=False,
        default=None,
        help="Override model path (e.g., model/checkpoints/legal/checkpoint-3)"
    )
    parser.add_argument(
        "--model-suffix",
        type=str,
        required=False,
        default=None,
        help="Suffix for output filename (e.g., 'legal' -> qwen_legal.jsonl)"
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.llm_config, args.retrieval_config, args.exp_config)

    # Override retriever type if specified
    if args.retriever:
        config["inference"]["retriever"]["type"] = args.retriever

    # Override model path if specified
    if args.model_path:
        retriever_type = config["inference"]["retriever"]["type"]
        config["inference"]["retriever"][retriever_type]["model_path"] = args.model_path

    # Run inference
    run_inference(config, args.dataset, args.output, args)


if __name__ == "__main__":
    main()
