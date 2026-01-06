"""
Pairwise evaluation script for comparing two retrieval models
Usage: python run_pairwise_eval.py --model1 results/.../bge.jsonl --model2 results/.../qwen.jsonl
"""

import argparse
import asyncio
import json
import re
from pathlib import Path
from tqdm import tqdm

from src.inference.prompts.llm_eval_prompt import LLM_EVAL_PROMPT
from src.kg.utils.llm_client import gpt_model


def load_results(filepath: str) -> list:
    """Load results from jsonl file"""
    results = []
    with open(filepath, 'r') as f:
        for line in f:
            results.append(json.loads(line.strip()))
    return results


async def pairwise_eval(question: str, ground_truth: str, answer1: str, answer2: str) -> dict:
    """
    Use GPT to compare two model answers.
    Returns dict with winner info.
    """
    eval_prompt = LLM_EVAL_PROMPT.format(
        question=question,
        ground_truth=ground_truth,
        answer1=answer1,
        answer2=answer2
    )

    response = await gpt_model(eval_prompt, operation_name="pairwise_eval")

    # Parse JSON response
    result = {
        "winner": "None",
        "faithfulness_winner": "None",
        "conciseness_winner": "None",
        "raw_response": response
    }

    try:
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            parsed = json.loads(json_match.group())
            overall = parsed.get("Overall Winner", {})
            result["winner"] = overall.get("Winner", "None")
            result["faithfulness_winner"] = parsed.get("Faithfulness", {}).get("Winner", "None")
            result["conciseness_winner"] = parsed.get("Conciseness", {}).get("Winner", "None")
    except (json.JSONDecodeError, AttributeError):
        pass

    return result


async def run_evaluation(args):
    """Run the pairwise evaluation"""
    # Load results
    results1 = load_results(args.model1)
    results2 = load_results(args.model2)

    # Get model names from paths
    model1_name = Path(args.model1).stem
    model2_name = Path(args.model2).stem

    # Get dataset and limit from path
    path_parts = Path(args.model1).parts
    limit_str = [p for p in path_parts if p.startswith("limit_")]
    limit_dir = limit_str[0] if limit_str else "limit_all"
    dataset_name = Path(args.model1).parent.name

    # Ensure same length
    min_len = min(len(results1), len(results2))
    if args.limit:
        min_len = min(min_len, args.limit)

    print(f"=" * 60)
    print(f"Pairwise Evaluation: {model1_name} vs {model2_name}")
    print(f"=" * 60)
    print(f"Dataset: {dataset_name}")
    print(f"Samples: {min_len}")
    print(f"LLM: GPT-4.1 (via llm_client)")
    print(f"=" * 60)

    # Run pairwise evaluation
    eval_results = []
    wins_model1 = 0
    wins_model2 = 0
    ties = 0

    for i in tqdm(range(min_len), desc="Evaluating"):
        r1 = results1[i]
        r2 = results2[i]

        question = r1.get("question", "")
        ground_truth = r1.get("answer", "")
        answer1 = r1.get("prediction", "")
        answer2 = r2.get("prediction", "")

        eval_result = await pairwise_eval(question, ground_truth, answer1, answer2)

        # Count wins
        winner = eval_result["winner"]
        if "Answer 1" in winner:
            wins_model1 += 1
        elif "Answer 2" in winner:
            wins_model2 += 1
        else:
            ties += 1

        eval_results.append({
            "question": question,
            "ground_truth": ground_truth,
            "answer1": answer1,
            "answer2": answer2,
            "winner": winner,
            "faithfulness_winner": eval_result["faithfulness_winner"],
            "conciseness_winner": eval_result["conciseness_winner"]
        })

    # Calculate win rates
    total = len(eval_results)
    win_rate_model1 = wins_model1 / total if total > 0 else 0
    win_rate_model2 = wins_model2 / total if total > 0 else 0
    tie_rate = ties / total if total > 0 else 0

    # Output path
    output_dir = Path(args.output_dir) / limit_dir / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{model1_name}_vs_{model2_name}.json"

    # Save summary
    summary = {
        "model1": model1_name,
        "model2": model2_name,
        "dataset": dataset_name,
        "total_samples": total,
        "wins_model1": wins_model1,
        "wins_model2": wins_model2,
        "ties": ties,
        "win_rate_model1": round(win_rate_model1, 4),
        "win_rate_model2": round(win_rate_model2, 4),
        "tie_rate": round(tie_rate, 4),
        "details": eval_results
    }

    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n" + "=" * 60)
    print(f"Results")
    print(f"=" * 60)
    print(f"{model1_name} wins: {wins_model1} ({win_rate_model1:.2%})")
    print(f"{model2_name} wins: {wins_model2} ({win_rate_model2:.2%})")
    print(f"Ties: {ties} ({tie_rate:.2%})")
    print(f"Output: {output_file}")
    print(f"=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Pairwise evaluation between two models")
    parser.add_argument("--model1", type=str, required=True, help="Path to model 1 results (jsonl)")
    parser.add_argument("--model2", type=str, required=True, help="Path to model 2 results (jsonl)")
    parser.add_argument("--output-dir", type=str, default="results/pairwise")
    parser.add_argument("--limit", type=int, default=None, help="Limit samples")
    args = parser.parse_args()

    asyncio.run(run_evaluation(args))


if __name__ == "__main__":
    main()
