"""
Triple Alignment Scorer for Answer-Centric Retrieval

This module implements a triple alignment scoring mechanism that evaluates
the quality of chunk-question-answer alignments using three complementary metrics:
1. Forward alignment: P(answer|chunk, question)
2. Backward alignment: P(question|chunk, answer)
3. Parameter alignment: cosine similarity between query and chunk embeddings
"""

import asyncio
import math
import pickle
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Callable
import yaml

import torch
from torch.nn.functional import log_softmax
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

@dataclass
class AlignmentConfig:
    """Configuration for alignment scoring.

    Attributes:
        forward_weight: Weight for forward alignment score P(answer|chunk, question)
        backward_weight: Weight for backward alignment score P(question|chunk, answer)
        parameter_weight: Weight for parameter alignment (embedding similarity)
        chunk_size: Size of text chunks in words
        overlap: Overlap size between consecutive chunks
        batch_size: Batch size for embedding computation
        max_context_length: Maximum context length for language model
        device: Device for model inference (e.g., 'cuda:0', 'cpu')
        normalize_scores: Whether to normalize individual alignment scores
    """
    forward_weight: float = 1.0
    backward_weight: float = 0.3
    parameter_weight: float = 1.0
    chunk_size: int = 512
    overlap: int = 12
    batch_size: int = 64
    max_context_length: Optional[int] = None
    device: str = "cuda"
    normalize_scores: bool = True

    @classmethod
    def from_yaml(cls, config_path: str) -> "AlignmentConfig":
        """Load configuration from YAML file.

        Args:
            config_path: Path to YAML configuration file

        Returns:
            AlignmentConfig instance
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # Extract alignment config from nested structure
        alignment_config = config.get('training', {}).get('qwen', {}).get('alignment', {})

        return cls(
            forward_weight=alignment_config.get('forward_weight', 1.0),
            backward_weight=alignment_config.get('backward_weight', 0.3),
            parameter_weight=alignment_config.get('parameter_weight', 1.0),
            chunk_size=config.get('kg', {}).get('chunk_size', 512),
            overlap=config.get('kg', {}).get('chunk_overlap', 12),
            batch_size=alignment_config.get('batch_size', 64),
            device=config.get('training', {}).get('common', {}).get('device', 'cuda'),
            normalize_scores=alignment_config.get('normalize_scores', True)
        )


class AlignmentScorer:
    """Triple alignment scorer for chunk-question-answer triplets.

    This class implements three complementary alignment metrics:
    1. Forward alignment: Measures how well a chunk supports generating the answer
    2. Backward alignment: Measures how well a chunk supports reconstructing the question
    3. Parameter alignment: Measures semantic similarity via embeddings

    Attributes:
        config: AlignmentConfig instance
        lm_model: Language model for computing forward/backward alignments
        lm_tokenizer: Tokenizer for the language model
        embedding_model: Sentence transformer for parameter alignment
        executor: Thread/process pool executor for parallel scoring
    """

    def __init__(
        self,
        lm_model_path: str,
        embedding_model_path: str,
        config: Optional[AlignmentConfig] = None,
        use_multiprocessing: bool = False,
        max_workers: Optional[int] = None
    ):
        """Initialize the alignment scorer.

        Args:
            lm_model_path: Path to language model for forward/backward alignment
            embedding_model_path: Path to embedding model for parameter alignment
            config: AlignmentConfig instance. If None, uses default config
            use_multiprocessing: Whether to use ProcessPoolExecutor (True) or ThreadPoolExecutor (False)
            max_workers: Maximum number of workers for parallel execution
        """
        self.config = config or AlignmentConfig()

        # Load language model
        print(f"Loading language model from {lm_model_path}...")
        self.lm_tokenizer = AutoTokenizer.from_pretrained(lm_model_path, use_fast=True)
        if self.lm_tokenizer.pad_token is None:
            self.lm_tokenizer.pad_token = self.lm_tokenizer.eos_token

        self.lm_model = AutoModelForCausalLM.from_pretrained(
            lm_model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            attn_implementation="eager"
        )
        self.lm_model = self.lm_model.to(self.config.device).eval()

        # Store max context length
        if self.config.max_context_length is None:
            self.config.max_context_length = self.lm_model.config.max_position_embeddings

        # Load embedding model
        print(f"Loading embedding model from {embedding_model_path}...")
        self.embedding_model = SentenceTransformer(
            embedding_model_path,
            device=self.config.device,
            trust_remote_code=True
        )

        # Initialize executor for parallel processing
        executor_class = ProcessPoolExecutor if use_multiprocessing else ThreadPoolExecutor
        self.executor = executor_class(max_workers=max_workers)

        print("AlignmentScorer initialized successfully.")

    def text_to_chunks(self, text: str) -> List[str]:
        """Split text into overlapping chunks.

        Args:
            text: Input text to be chunked

        Returns:
            List of text chunks
        """
        words = text.split(' ')
        chunks = []
        step = self.config.chunk_size - self.config.overlap

        for i in range(0, len(words), step):
            chunk = ' '.join(words[i:i + self.config.chunk_size]).strip()
            if chunk:
                chunks.append(chunk)

        return chunks

    @staticmethod
    def normalize_scores(scores: List[Optional[float]]) -> List[Optional[float]]:
        """Normalize scores to [0, 1] range using min-max normalization.

        Args:
            scores: List of scores (may contain None values)

        Returns:
            Normalized scores
        """
        valid = [s for s in scores if s is not None]
        if not valid:
            return scores

        mn, mx = min(valid), max(valid)
        if math.isclose(mx, mn):
            return [0.5 if s is not None else None for s in scores]

        return [
            None if s is None else (s - mn) / (mx - mn)
            for s in scores
        ]

    def _compute_lm_logprob(
        self,
        prompts: List[str],
        targets: List[str],
        desc: str = "Computing"
    ) -> List[float]:
        """Compute log probabilities of targets given prompts.

        This is an internal method used by forward_alignment and backward_alignment.

        Args:
            prompts: List of prompt strings
            targets: List of target strings (one per prompt)
            desc: Description for progress bar

        Returns:
            List of average log probabilities
        """
        results = []
        device = self.config.device
        max_ctx = self.config.max_context_length

        # Assume all targets are the same (as in the original implementation)
        tgt_ids = self.lm_tokenizer(targets[0], add_special_tokens=False)["input_ids"]

        for prompt, target in tqdm(zip(prompts, targets), total=len(prompts), desc=desc, leave=False):
            prom_ids = self.lm_tokenizer(prompt, add_special_tokens=False)["input_ids"]
            total_len = len(prom_ids) + len(tgt_ids)

            # Truncate prompt if necessary
            if total_len > max_ctx:
                avail = max_ctx - len(tgt_ids)
                prom_ids = prom_ids[:max(avail, 0)]

            ctx_len = len(prom_ids)
            full_text = self.lm_tokenizer.decode(prom_ids, skip_special_tokens=False) + target

            encodings = self.lm_tokenizer(
                full_text,
                add_special_tokens=False,
                return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                logits = self.lm_model(**encodings).logits
                logp = log_softmax(logits, dim=-1)

            ids = encodings.input_ids[0, ctx_len:ctx_len+len(tgt_ids)]
            pred_logp = logp[0, ctx_len-1:ctx_len-1+len(tgt_ids)]
            tok_logp = pred_logp.gather(1, ids.unsqueeze(1))

            results.append(tok_logp.mean().item())

        return results

    def forward_alignment(
        self,
        chunks: List[str],
        question: str,
        answer: str
    ) -> List[float]:
        """Compute forward alignment: P(answer|chunk, question).

        This measures how well each chunk supports generating the answer
        given the question.

        Args:
            chunks: List of text chunks
            question: The question text
            answer: The expected answer text

        Returns:
            List of forward alignment scores (log probabilities)
        """
        prompts = [
            f"You are given a context and a question. Answer the question as concisely as you can, "
            f"using a single phrase or sentence if possible. If the question cannot be answered based "
            f"on the information in the context, write \"unanswerable\". If the question is a yes/no "
            f"question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\n"
            f"Context: {chunk}\n\n"
            f"Question: {question}\n\n"
            f"Answer:"
            for chunk in chunks
        ]
        targets = [answer] * len(chunks)
        return self._compute_lm_logprob(prompts, targets, desc="Forward alignment")

    def backward_alignment(
        self,
        chunks: List[str],
        question: str,
        answer: str
    ) -> List[float]:
        """Compute backward alignment: P(question|chunk, answer).

        This measures how well each chunk supports reconstructing the question
        given the answer.

        Args:
            chunks: List of text chunks
            question: The question text
            answer: The given answer text

        Returns:
            List of backward alignment scores (log probabilities)
        """
        prompts = [
            f"You are given a context and an answer. Infer the original question that proposed for "
            f"the given answer, ensuring the question is clear, specific, and self-contained. If the "
            f"answer could address multiple plausible questions, choose the one most directly supported "
            f"by the context. Do not provide any explanation.\n\n"
            f"Context: {chunk}\n\n"
            f"Answer: {answer}\n\n"
            f"Question:"
            for chunk in chunks
        ]
        targets = [question] * len(chunks)
        return self._compute_lm_logprob(prompts, targets, desc="Backward alignment")

    def parameter_alignment(
        self,
        chunks: List[str],
        question: str
    ) -> List[float]:
        """Compute parameter alignment: cosine similarity between embeddings.

        This measures the semantic similarity between the question and each chunk
        using dense embeddings.

        Args:
            chunks: List of text chunks
            question: The question text

        Returns:
            List of cosine similarity scores
        """
        scores = []
        device = self.config.device

        # Encode question once
        question_emb = self.embedding_model.encode(
            [question],
            normalize_embeddings=True,
            device=device
        )

        # Batch encode chunks
        for i in tqdm(range(0, len(chunks), self.config.batch_size),
                      desc="Computing embeddings", leave=False):
            batch_chunks = chunks[i:i + self.config.batch_size]

            # Compute similarity for this batch
            similarities = self.embedding_model.similarity(
                question_emb,
                self.embedding_model.encode(
                    batch_chunks,
                    normalize_embeddings=True,
                    device=device
                )
            )

            if similarities is not None:
                scores.extend(similarities[0].tolist())
            else:
                scores.extend([0.0] * len(batch_chunks))

        return scores

    def compute_score(
        self,
        chunks: List[str],
        question: str,
        answer: str,
        return_components: bool = False
    ) -> Tuple[List[float], Optional[Dict[str, List[float]]]]:
        """Compute comprehensive alignment score combining all three metrics.

        Args:
            chunks: List of text chunks
            question: The question text
            answer: The answer text
            return_components: Whether to return individual component scores

        Returns:
            Tuple of (final_scores, component_scores_dict)
            - final_scores: List of weighted combined scores
            - component_scores_dict: Dictionary with keys 'forward', 'backward', 'parameter'
              containing normalized component scores (only if return_components=True)
        """
        # Compute all three alignment scores
        forward_scores = self.forward_alignment(chunks, question, answer)
        backward_scores = self.backward_alignment(chunks, question, answer)
        parameter_scores = self.parameter_alignment(chunks, question)

        # Normalize scores if configured
        if self.config.normalize_scores:
            forward_scores_norm = self.normalize_scores(forward_scores)
            backward_scores_norm = self.normalize_scores(backward_scores)
            parameter_scores_norm = self.normalize_scores(parameter_scores)
        else:
            forward_scores_norm = forward_scores
            backward_scores_norm = backward_scores
            parameter_scores_norm = parameter_scores

        # Compute weighted combination
        final_scores = [
            self.config.forward_weight * f +
            self.config.backward_weight * b +
            self.config.parameter_weight * p
            for f, b, p in zip(
                forward_scores_norm,
                backward_scores_norm,
                parameter_scores_norm
            )
        ]

        components = None
        if return_components:
            components = {
                'forward': forward_scores_norm,
                'backward': backward_scores_norm,
                'parameter': parameter_scores_norm
            }

        return final_scores, components

    def score_single_item(
        self,
        question: str,
        answer: str,
        context: str,
        top_k: Optional[int] = None
    ) -> Dict[str, Any]:
        """Score a single question-answer-context triplet.

        Args:
            question: The question text
            answer: The answer text (can be string or list)
            context: The full context text to be chunked
            top_k: If specified, return only top-k chunks

        Returns:
            Dictionary containing:
            - chunks: List of chunks (sorted by score if top_k specified)
            - scores: List of alignment scores
            - forward_scores: List of forward alignment scores
            - backward_scores: List of backward alignment scores
            - parameter_scores: List of parameter alignment scores
        """
        # Handle answer format
        if isinstance(answer, list) and answer:
            answer = answer[0]

        # Chunk the context
        chunks = self.text_to_chunks(context)
        if not chunks:
            return {
                'chunks': [],
                'scores': [],
                'forward_scores': [],
                'backward_scores': [],
                'parameter_scores': []
            }

        print(f"  → Split into {len(chunks)} chunks")

        # Compute scores
        print(f"  → Computing alignment scores...")
        final_scores, components = self.compute_score(
            chunks, question, answer, return_components=True
        )

        # Sort by score and optionally select top-k
        sorted_indices = sorted(
            range(len(final_scores)),
            key=lambda i: final_scores[i],
            reverse=True
        )

        if top_k is not None:
            original_count = len(sorted_indices)
            sorted_indices = sorted_indices[:top_k]
            print(f"  → Selected top-{len(sorted_indices)} chunks (from {original_count} total)")
        else:
            print(f"  → Keeping all {len(sorted_indices)} chunks")

        return {
            'chunks': [chunks[i] for i in sorted_indices],
            'scores': [final_scores[i] for i in sorted_indices],
            'forward_scores': [components['forward'][i] for i in sorted_indices],
            'backward_scores': [components['backward'][i] for i in sorted_indices],
            'parameter_scores': [components['parameter'][i] for i in sorted_indices]
        }

    async def score_single_item_async(
        self,
        question: str,
        answer: str,
        context: str,
        top_k: Optional[int] = None
    ) -> Dict[str, Any]:
        """Async version of score_single_item.

        Args:
            question: The question text
            answer: The answer text
            context: The full context text
            top_k: If specified, return only top-k chunks

        Returns:
            Same as score_single_item
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.score_single_item,
            question,
            answer,
            context,
            top_k
        )

    def score_batch(
        self,
        items: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """Score a batch of items sequentially.

        Args:
            items: List of dictionaries with keys 'question', 'answer', 'context'
            top_k: If specified, return only top-k chunks per item
            show_progress: Whether to show progress bar

        Returns:
            List of scoring results (one per input item)
        """
        results = []
        iterator = tqdm(items, desc="Scoring items") if show_progress else items

        for item in iterator:
            result = self.score_single_item(
                question=item['question'],
                answer=item['answer'],
                context=item['context'],
                top_k=top_k
            )
            results.append(result)

        return results

    async def score_batch_async(
        self,
        items: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        max_concurrent: int = 10
    ) -> List[Dict[str, Any]]:
        """Score a batch of items asynchronously with concurrency control.

        Args:
            items: List of dictionaries with keys 'question', 'answer', 'context'
            top_k: If specified, return only top-k chunks per item
            max_concurrent: Maximum number of concurrent scoring operations

        Returns:
            List of scoring results (one per input item)
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def score_with_semaphore(item):
            async with semaphore:
                return await self.score_single_item_async(
                    question=item['question'],
                    answer=item['answer'],
                    context=item['context'],
                    top_k=top_k
                )

        tasks = [score_with_semaphore(item) for item in items]
        return await asyncio.gather(*tasks)

    def cleanup(self):
        """Clean up resources (executor, models)."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)

        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()


def main():
    """Example usage of AlignmentScorer."""
    import argparse
    import jsonlines

    parser = argparse.ArgumentParser(description="Triple Alignment Scoring")
    parser.add_argument('--input_jsonl', type=str, required=True,
                        help='Input JSONL file with questions, answers, and contexts')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for individual score files (one per sample)')
    parser.add_argument('--config', type=str, default='/home/jiawei/ARK/config.yaml',
                        help='Path to config.yaml')
    parser.add_argument('--lm_model', type=str, required=True,
                        help='Path to language model')
    parser.add_argument('--embedding_model', type=str, required=True,
                        help='Path to embedding model')
    parser.add_argument('--top_k', type=int, default=1000,
                        help='Number of top chunks to keep')
    parser.add_argument('--start_index', type=int, default=0,
                        help='Start index for processing (inclusive)')
    parser.add_argument('--end_index', type=int, default=None,
                        help='End index for processing (exclusive)')
    parser.add_argument('--use_async', action='store_true',
                        help='Use async processing')
    parser.add_argument('--max_concurrent', type=int, default=10,
                        help='Maximum concurrent async operations')

    args = parser.parse_args()

    # Load configuration
    config = AlignmentConfig.from_yaml(args.config)

    # Initialize scorer
    with AlignmentScorer(args.lm_model, args.embedding_model, config) as scorer:
        # Load data
        print(f"Loading data from {args.input_jsonl}...")
        items = []
        with jsonlines.open(args.input_jsonl) as reader:
            for idx, item in enumerate(reader):
                # Skip items before start_index
                if idx < args.start_index:
                    continue

                # Stop if we've reached end_index
                if args.end_index is not None and idx >= args.end_index:
                    break

                items.append({
                    'question': item['input'],
                    'answer': item['answers'],
                    'context': item['context'],
                    'original_item': item
                })

        print(f"Loaded {len(items)} items (index {args.start_index} to {args.end_index or args.start_index + len(items)}).")

        # Extract dataset name from input file path
        # e.g., ./data/raw/hotpotqa.jsonl -> hotpotqa
        import os
        dataset_name = os.path.basename(args.input_jsonl).replace('.jsonl', '')

        # Prepare output directory
        from pathlib import Path
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Score and write items one by one
        print(f"Processing and writing results to {output_dir}/...")
        iterator = enumerate(items, start=args.start_index)
        if not args.use_async:
            iterator = tqdm(iterator, total=len(items), desc="Scoring items")

        for idx, item in iterator:
            # Score single item
            result = scorer.score_single_item(
                question=item['question'],
                answer=item['answer'],
                context=item['context'],
                top_k=args.top_k
            )

            # Prepare output
            output = {
                'input': item['question'],
                'answers': item['answer'],
                'chunk_list': result['chunks'],
                'score_list': result['scores'],
                'forward_list': result['forward_scores'],
                'reverse_list': result['backward_scores'],
                'qwen_list': result['parameter_scores']
            }

            # Write to individual file with format: {dataset}_{index}.pkl
            output_file = output_dir / f"{dataset_name}_{idx}.pkl"
            with open(output_file, 'wb') as f:
                pickle.dump(output, f)

        print(f"Done! Wrote {len(items)} files to {output_dir}/")


if __name__ == '__main__':
    main()
