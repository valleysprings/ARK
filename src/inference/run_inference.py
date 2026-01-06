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
    text_to_chunks,
    llm_generate
)
from src.inference.metrics import qa_f1_score
from src.inference.prompts.dataset_prompts import DATASET2PROMPT, ULTRADOMAIN_PROMPT


def load_config(llm_config_path: str = None, retrieval_config_path: str = None, exp_config_path: str = None) -> Dict[str, Any]:
    """
    Load and merge configs

    Args:
        llm_config_path: Path to LLM config (src/config/llm_inference.yaml)
        retrieval_config_path: Path to retrieval config (src/config/retrieval_model.yaml)
        exp_config_path: Path to experiment config (experiments/exp_*.yaml)

    Returns:
        Merged config dictionary
    """
    # Default paths
    if llm_config_path is None:
        llm_config_path = "src/config/llm_inference.yaml"
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
        "logging": llm_config.get("logging", {}),
        "token_tracking": llm_config.get("token_tracking", {}),
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


def run_inference(config: Dict[str, Any], dataset_path: str, output_path: str, args=None):
    """
    Run inference on dataset

    Args:
        config: Configuration dictionary
        dataset_path: Path to JSONL dataset
        output_path: Path to save results
    """
    # Initialize retriever
    retriever = UnifiedRetriever(config, device=args.device)

    # Get configuration
    retriever_config = config["inference"]["retriever"]
    model_type = retriever_config["type"]
    model_config = retriever_config[model_type]

    chunk_size = model_config.get("chunk_size", 512)
    overlap = model_config.get("overlap", 12)

    generator_config = config["inference"]["generator"]
    llm_endpoint = generator_config["endpoint"]
    llm_model = generator_config["model"]
    llm_num_ctx = generator_config.get("num_ctx", 16384)  # Default: 16384

    # Get dataset name for prompt
    dataset_name = Path(dataset_path).stem
    prompt_template = DATASET2PROMPT.get(dataset_name, ULTRADOMAIN_PROMPT)

    print(f"=" * 60)
    print(f"ARK Unified Inference")
    print(f"=" * 60)
    print(f"Retriever: {model_type.upper()}")
    print(f"Dataset: {dataset_name}")
    print(f"Chunk size: {chunk_size}, Overlap: {overlap}")
    print(f"LLM: {llm_model} @ {llm_endpoint}")
    print(f"=" * 60)

    # Load dataset
    with open(dataset_path, 'r') as f:
        dataset = [json.loads(line) for line in f]

    # Apply limit if specified
    if args.limit:
        dataset = dataset[:args.limit]

    results = []
    total_f1 = 0.0

    for idx, entry in enumerate(dataset, 1):
        context = entry.get("context", "")
        question = entry.get("input", entry.get("question", ""))
        # Handle different answer field names: answers (list), ideal, answer
        answer = entry.get("answers", entry.get("ideal", entry.get("answer", "")))
        # If answers is a list, take the first one
        if isinstance(answer, list) and len(answer) > 0:
            answer = answer[0]

        # Handle no_retrieval mode: skip chunking and use full context
        if model_type == "no_retrieval":
            # Use full context directly without chunking or retrieval
            context_text = context
            prompt = prompt_template.format(
                context=context_text,
                input=question
            )
            # Generate answer
            prediction = llm_generate(prompt, llm_endpoint, llm_model, llm_num_ctx)

            # Compute F1 score
            f1 = qa_f1_score(prediction, answer)
            total_f1 += f1

            # Save result
            result = {
                "question": question,
                "answer": answer,
                "prediction": prediction,
                "num_chunks_used": 0,
                "chunk_scores": [],
                "f1_score": f1
            }
            results.append(result)

            # Print progress
            if idx % 10 == 0 or idx == len(dataset):
                print(f"[{idx}/{len(dataset)}] Avg F1: {total_f1 / idx:.4f}")

            continue

        # Standard retrieval mode: chunk and retrieve
        # Split into chunks
        chunks = text_to_chunks(context, chunk_size, overlap)

        if not chunks:
            print(f"[{idx}/{len(dataset)}] No chunks for entry, skipping")
            continue

        # Retrieve top chunks
        top_chunks = retriever.retrieve(question, chunks)

        # Get chunk texts
        selected_chunks = [chunks[chunk_idx] for chunk_idx, score in top_chunks]

        # Build prompt
        context_text = "\n\n".join(selected_chunks)
        prompt = prompt_template.format(
            context=context_text,
            input=question
        )

        # Generate answer
        prediction = llm_generate(prompt, llm_endpoint, llm_model, llm_num_ctx)

        # Compute F1 score
        f1 = qa_f1_score(prediction, answer)
        total_f1 += f1

        # Save result
        result = {
            "question": question,
            "answer": answer,
            "prediction": prediction,
            "num_chunks_used": len(selected_chunks),
            "chunk_scores": [score for _, score in top_chunks],
            "retrieved_chunks": selected_chunks,
            "f1_score": f1
        }
        results.append(result)

        # Print progress
        if idx % 10 == 0 or idx == len(dataset):
            print(f"[{idx}/{len(dataset)}] Avg F1: {total_f1 / idx:.4f}")

    # Organize output paths
    dataset_name = Path(dataset_path).stem

    # Determine limit prefix: limit_X or full
    limit_prefix = f"limit_{args.limit}" if args.limit else "full"

    # Create results directory structure: results/raw/[limit_X|full]/[dataset]/ and results/score/[limit_X|full]/[dataset]/
    raw_dir = Path("results/raw") / limit_prefix / dataset_name
    score_dir = Path("results/score") / limit_prefix / dataset_name
    raw_dir.mkdir(parents=True, exist_ok=True)
    score_dir.mkdir(parents=True, exist_ok=True)

    # Save raw results to results/raw/[limit_X|full]/[dataset]/[model].jsonl
    output_name = f"{model_type}_{args.model_suffix}" if args.model_suffix else model_type
    raw_output = raw_dir / f"{output_name}.jsonl"
    with open(raw_output, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

    # Calculate and save scores to results/score/[limit_X|full]/[dataset]/[model].json
    score_output = score_dir / f"{output_name}.json"
    avg_f1 = total_f1 / len(results) if results else 0
    score_data = {
        "model": model_type,
        "dataset": dataset_name,
        "limit": args.limit if args.limit else "full",
        "total_samples": len(results),
        "f1_score": round(avg_f1, 4)
    }
    with open(score_output, 'w') as f:
        json.dump(score_data, f, indent=2)

    # Clean up GPU memory after retrieval
    del retriever
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU memory cleared for device: {args.device}")

    # Print summary
    print(f"\n" + "=" * 60)
    print(f"Inference Complete!")
    print(f"=" * 60)
    print(f"Total samples: {len(results)}")
    print(f"Average F1: {score_data.get('f1_score', 0):.4f}")
    print(f"Raw results saved to: {raw_output}")
    print(f"Scores saved to: {score_output}")
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
        default="src/config/llm_inference.yaml",
        help="LLM inference config file (default: src/config/llm_inference.yaml)"
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
        help="CUDA device"
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
