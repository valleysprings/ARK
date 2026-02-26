"""
Triple Alignment Scorer for Answer-Centric Retrieval

This module implements a triple alignment scoring mechanism that evaluates
the quality of chunk-question-answer alignments using three complementary metrics:
1. Forward alignment: P(answer|chunk, question)
2. Backward alignment: P(question|chunk, answer)
3. Parameter alignment: cosine similarity between query and chunk embeddings
"""

import asyncio
import json
import math
import os
import re
import subprocess
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Callable
import yaml

import requests
import torch
from torch.nn.functional import log_softmax
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from src.kg.utils.text_processing import (
    chunking_by_token_size, chunking_by_sentence,
    TextChunkSchema, ENCODER as TIKTOKEN_ENCODER,
)

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
    chunk_size: int = 1
    overlap: int = 0
    chunk_method: str = "sentence"
    batch_size: int = 64
    max_context_length: Optional[int] = None
    device: str = "cuda"
    normalize_scores: bool = True
    gpu_memory_utilization: float = 0.85
    vllm_max_model_len: int = 8192
    top_k: int = 10
    min_chunk_tokens: int = 10

    @classmethod
    def from_yaml(cls, config_path: str) -> "AlignmentConfig":
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        return cls(
            forward_weight=config['forward_weight'],
            backward_weight=config['backward_weight'],
            parameter_weight=config['parameter_weight'],
            chunk_size=config['chunk_size'],
            overlap=config['chunk_overlap'],
            chunk_method=config['chunk_method'],
            batch_size=config['batch_size'],
            normalize_scores=config['normalize_scores'],
            gpu_memory_utilization=config['gpu_memory_utilization'],
            vllm_max_model_len=config['vllm_max_model_len'],
            top_k=config['top_k'],
            min_chunk_tokens=config['min_chunk_tokens'],
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
        max_workers: Optional[int] = None,
        use_vllm: bool = False,
        vllm_port: int = 8199,
        vllm_device: str = "cuda:0",
        use_vllm_offline: bool = False,
    ):
        self.config = config or AlignmentConfig()
        self.use_vllm = use_vllm
        self.use_vllm_offline = use_vllm_offline
        self.lm_model_path = lm_model_path
        self.vllm_port = vllm_port
        self._vllm_proc = None
        self.vllm_llm = None

        # Load tokenizer (needed for both modes)
        print(f"Loading tokenizer from {lm_model_path}...")
        self.lm_tokenizer = AutoTokenizer.from_pretrained(lm_model_path, use_fast=True)
        if self.lm_tokenizer.pad_token is None:
            self.lm_tokenizer.pad_token = self.lm_tokenizer.eos_token

        if use_vllm_offline:
            from vllm import LLM, SamplingParams
            print(f"Loading vLLM offline model from {lm_model_path}...")
            self.vllm_llm = LLM(
                model=lm_model_path,
                max_model_len=self.config.vllm_max_model_len,
                trust_remote_code=True,
                dtype="float16",
                gpu_memory_utilization=self.config.gpu_memory_utilization,
            )
            self.vllm_sampling_params = SamplingParams(
                prompt_logprobs=1, max_tokens=1, temperature=0,
            )
            if self.config.max_context_length is None:
                self.config.max_context_length = self.config.vllm_max_model_len
        elif use_vllm:
            self._start_vllm_server(vllm_device, self.config.vllm_max_model_len)
            if self.config.max_context_length is None:
                self.config.max_context_length = self.config.vllm_max_model_len
        else:
            # Load model locally
            print(f"Loading language model from {lm_model_path}...")
            self.lm_model = AutoModelForCausalLM.from_pretrained(
                lm_model_path,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                attn_implementation="eager"
            )
            self.lm_model = self.lm_model.to(self.config.device).eval()
            if self.config.max_context_length is None:
                self.config.max_context_length = self.lm_model.config.max_position_embeddings

        # Load embedding model
        print(f"Loading embedding model from {embedding_model_path}...")
        self.embedding_model = SentenceTransformer(
            embedding_model_path,
            device=self.config.device,
            trust_remote_code=True
        )

        executor_class = ProcessPoolExecutor if use_multiprocessing else ThreadPoolExecutor
        self.executor = executor_class(max_workers=max_workers)
        print("AlignmentScorer initialized successfully.")

    def _start_vllm_server(self, device: str, max_model_len: int):
        """Start vLLM server on specified GPU."""
        gpu_id = device.split(":")[-1] if ":" in device else "0"
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu_id}
        vllm_log = "/tmp/vllm_alignment_stderr.log"
        self._vllm_log_f = open(vllm_log, "w")
        print(f"Starting vLLM server: {self.lm_model_path} on GPU {gpu_id}, port {self.vllm_port}...")
        self._vllm_proc = subprocess.Popen(
            ["python", "-m", "vllm.entrypoints.openai.api_server",
             "--model", self.lm_model_path, "--port", str(self.vllm_port),
             "--trust-remote-code", "--max-model-len", str(max_model_len)],
            env=env, stdout=self._vllm_log_f, stderr=self._vllm_log_f
        )
        for _ in range(120):
            try:
                r = requests.get(f"http://localhost:{self.vllm_port}/health", timeout=2)
                if r.status_code == 200:
                    print(f"vLLM server ready on port {self.vllm_port}")
                    return
            except Exception:
                pass
            time.sleep(2)
        try:
            with open(vllm_log, "r") as f:
                print("=== vLLM server log ===")
                print(f.read()[-3000:])
                print("=== end vLLM log ===")
        except Exception:
            pass
        raise RuntimeError("vLLM server failed to start")

    def _shutdown_vllm(self):
        if self._vllm_proc:
            self._vllm_proc.terminate()
            self._vllm_proc.wait()
            self._vllm_proc = None
            print("vLLM server stopped")

    def text_to_chunks(self, text: str) -> List[TextChunkSchema]:
        """Split text into chunks, returning TextChunkSchema list."""
        if self.config.chunk_method == "sentence":
            return chunking_by_sentence(
                text, max_sentences=self.config.chunk_size,
                overlap_sentences=self.config.overlap,
            )
        tokens = TIKTOKEN_ENCODER.encode(text)
        return chunking_by_token_size(
            [tokens], tiktoken_model=TIKTOKEN_ENCODER,
            overlap_token_size=self.config.overlap,
            max_token_size=self.config.chunk_size,
        )

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
        if self.use_vllm_offline:
            return self._compute_lm_logprob_offline(prompts, targets, desc)
        if self.use_vllm:
            return self._compute_lm_logprob_vllm(prompts, targets, desc)
        return self._compute_lm_logprob_local(prompts, targets, desc)

    def _compute_lm_logprob_offline(
        self,
        prompts: List[str],
        targets: List[str],
        desc: str = "Computing"
    ) -> List[float]:
        """Compute log probabilities via vLLM offline batch inference."""
        max_ctx = self.config.max_context_length
        tgt_ids = self.lm_tokenizer(targets[0], add_special_tokens=False)["input_ids"]
        tgt_len = len(tgt_ids)

        full_texts = []
        prefix_lens = []
        for prompt, target in zip(prompts, targets):
            prom_ids = self.lm_tokenizer(prompt, add_special_tokens=False)["input_ids"]
            if len(prom_ids) + tgt_len > max_ctx:
                prom_ids = prom_ids[:max(max_ctx - tgt_len, 0)]
            prefix_lens.append(len(prom_ids))
            full_texts.append(
                self.lm_tokenizer.decode(prom_ids, skip_special_tokens=False) + target
            )

        outputs = self.vllm_llm.generate(full_texts, self.vllm_sampling_params)

        results = []
        for out, plen in zip(outputs, prefix_lens):
            plp = out.prompt_logprobs
            token_lps = []
            for pos in range(plen, min(plen + tgt_len, len(plp))):
                if plp[pos] is not None:
                    # Each position is a dict {token_id: Logprob}; get the actual token's logprob
                    token_id = out.prompt_token_ids[pos]
                    if token_id in plp[pos]:
                        token_lps.append(plp[pos][token_id].logprob)
            results.append(sum(token_lps) / len(token_lps) if token_lps else float('-inf'))

        return results

    def _compute_lm_logprob_vllm(
        self,
        prompts: List[str],
        targets: List[str],
        desc: str = "Computing"
    ) -> List[float]:
        """Compute log probabilities via vLLM server (echo + logprobs)."""
        results = []
        max_ctx = self.config.max_context_length
        tgt_ids = self.lm_tokenizer(targets[0], add_special_tokens=False)["input_ids"]

        for prompt, target in tqdm(zip(prompts, targets), total=len(prompts), desc=desc, leave=False):
            prom_ids = self.lm_tokenizer(prompt, add_special_tokens=False)["input_ids"]
            if len(prom_ids) + len(tgt_ids) > max_ctx:
                avail = max_ctx - len(tgt_ids)
                prom_ids = prom_ids[:max(avail, 0)]

            ctx_len = len(prom_ids)
            full_text = self.lm_tokenizer.decode(prom_ids, skip_special_tokens=False) + target

            try:
                resp = requests.post(
                    f"http://localhost:{self.vllm_port}/v1/completions",
                    json={
                        "model": self.lm_model_path,
                        "prompt": full_text,
                        "max_tokens": 1,
                        "echo": True,
                        "logprobs": 1,
                        "temperature": 0.0,
                    },
                    timeout=120
                )
                resp.raise_for_status()
                token_logprobs = resp.json()["choices"][0]["logprobs"]["token_logprobs"]
                # Extract logprobs for target tokens (positions after prompt)
                target_lps = token_logprobs[ctx_len:ctx_len + len(tgt_ids)]
                valid = [lp for lp in target_lps if lp is not None]
                results.append(sum(valid) / len(valid) if valid else float('-inf'))
            except Exception as e:
                print(f"vLLM logprob error: {e}")
                results.append(float('-inf'))

        return results

    def _compute_lm_logprob_local(
        self,
        prompts: List[str],
        targets: List[str],
        desc: str = "Computing"
    ) -> List[float]:
        """Compute log probabilities locally with loaded model."""
        results = []
        device = self.config.device
        max_ctx = self.config.max_context_length
        tgt_ids = self.lm_tokenizer(targets[0], add_special_tokens=False)["input_ids"]

        for prompt, target in tqdm(zip(prompts, targets), total=len(prompts), desc=desc, leave=False):
            prom_ids = self.lm_tokenizer(prompt, add_special_tokens=False)["input_ids"]
            if len(prom_ids) + len(tgt_ids) > max_ctx:
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
    ) -> Dict[str, Any]:
        """Score a single question-answer-context triplet.

        Returns unsorted:
            {chunks, scores, forward_scores, backward_scores, parameter_scores}
        """
        if isinstance(answer, list) and answer:
            answer = answer[0]

        chunk_schemas = self.text_to_chunks(context)
        if not chunk_schemas:
            return {
                'chunks': [], 'scores': [], 'forward_scores': [],
                'backward_scores': [], 'parameter_scores': [],
            }

        chunk_texts = [c["content"] for c in chunk_schemas]
        print(f"  → Split into {len(chunk_texts)} chunks")

        print(f"  → Computing alignment scores...")
        final_scores, components = self.compute_score(
            chunk_texts, question, answer, return_components=True
        )

        # Apply short-phrase penalty
        min_tok = self.config.min_chunk_tokens
        for i, c in enumerate(chunk_schemas):
            if c["tokens"] < min_tok:
                penalty = c["tokens"] / min_tok
                final_scores[i] *= penalty

        return {
            'chunks': chunk_schemas,
            'scores': final_scores,
            'forward_scores': components['forward'],
            'backward_scores': components['backward'],
            'parameter_scores': components['parameter'],
        }

    async def score_single_item_async(
        self,
        question: str,
        answer: str,
        context: str,
    ) -> Dict[str, Any]:
        """Async version of score_single_item."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.score_single_item,
            question,
            answer,
            context,
        )

    def score_batch(
        self,
        items: List[Dict[str, Any]],
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """Score a batch of items sequentially."""
        results = []
        iterator = tqdm(items, desc="Scoring items") if show_progress else items
        for item in iterator:
            result = self.score_single_item(
                question=item['question'],
                answer=item['answer'],
                context=item['context'],
            )
            results.append(result)
        return results

    async def score_batch_async(
        self,
        items: List[Dict[str, Any]],
        max_concurrent: int = 10
    ) -> List[Dict[str, Any]]:
        """Score a batch of items asynchronously with concurrency control."""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def score_with_semaphore(item):
            async with semaphore:
                return await self.score_single_item_async(
                    question=item['question'],
                    answer=item['answer'],
                    context=item['context'],
                )

        tasks = [score_with_semaphore(item) for item in items]
        return await asyncio.gather(*tasks)

    def cleanup(self):
        """Clean up resources (executor, models, vLLM server)."""
        self._shutdown_vllm()
        if self.vllm_llm is not None:
            del self.vllm_llm
            self.vllm_llm = None
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()


def clean_content(text: str) -> str:
    """Remove newlines, dash sequences, and decorative non-alphanumeric symbols.
    Keep letters, digits, spaces, and basic punctuation."""
    text = text.replace('\n', ' ')
    text = re.sub(r'-{2,}', ' ', text)
    text = re.sub(r'[^\w\s.,;:!?\'\"()\[\]{}/\\@#$%&*+=<>-]', '', text)
    return re.sub(r'\s+', ' ', text).strip()


def main():
    import argparse
    import jsonlines

    parser = argparse.ArgumentParser(description="Triple Alignment Scoring")
    parser.add_argument('--input_jsonl', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--config', type=str, default='./config.yaml')
    parser.add_argument('--lm_model', type=str, required=True)
    parser.add_argument('--embedding_model', type=str, required=True)
    parser.add_argument('--start_index', type=int, default=0)
    parser.add_argument('--end_index', type=int, default=None)
    parser.add_argument('--use_async', action='store_true')
    parser.add_argument('--max_concurrent', type=int, default=10)
    parser.add_argument('--use_vllm', action='store_true')
    parser.add_argument('--vllm_port', type=int, default=8199)
    parser.add_argument('--vllm_device', type=str, default='cuda:0')
    parser.add_argument('--use_vllm_offline', action='store_true')
    parser.add_argument('--gpu_id', type=int, default=None)
    parser.add_argument('--chunk_method', type=str, default=None)
    parser.add_argument('--chunk_size', type=int, default=None)
    parser.add_argument('--chunk_overlap', type=int, default=None)
    args = parser.parse_args()

    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    config = AlignmentConfig.from_yaml(args.config)
    if args.chunk_method is not None:
        config.chunk_method = args.chunk_method
    if args.chunk_size is not None:
        config.chunk_size = args.chunk_size
    if args.chunk_overlap is not None:
        config.overlap = args.chunk_overlap

    with AlignmentScorer(
        args.lm_model, args.embedding_model, config,
        use_vllm=args.use_vllm,
        vllm_port=args.vllm_port,
        vllm_device=args.vllm_device,
        use_vllm_offline=args.use_vllm_offline,
    ) as scorer:
        input_path = args.input_jsonl
        if not Path(input_path).exists():
            alt_path = input_path.replace("/data/raw/", "/data/raw/additional/")
            if Path(alt_path).exists():
                input_path = alt_path
            else:
                raise FileNotFoundError(f"Not found: {input_path} or {alt_path}")

        print(f"Loading data from {input_path}...")
        items = []
        with jsonlines.open(input_path) as reader:
            for idx, item in enumerate(reader):
                if idx < args.start_index:
                    continue
                if args.end_index is not None and idx >= args.end_index:
                    break
                items.append({
                    'question': item['input'],
                    'answer': item['answers'],
                    'context': item['context'],
                })

        print(f"Loaded {len(items)} items (index {args.start_index} to "
              f"{args.end_index or args.start_index + len(items)}).")

        dataset_name = os.path.basename(input_path).replace('.jsonl', '')
        output_dir = Path(args.output_dir)

        # Create all output directories
        dirs = {}
        for name in ['chunk', 'full-rank', 'topk-rank']:
            dirs[name] = output_dir / name
            dirs[name].mkdir(parents=True, exist_ok=True)
        for prefix in ['full', 'topk']:
            for sub in ['score', 'forward', 'backward', 'parameter']:
                key = f"{prefix}/{sub}"
                dirs[key] = output_dir / prefix / sub
                dirs[key].mkdir(parents=True, exist_ok=True)

        print(f"Processing and writing results to {output_dir}/...")
        k = config.top_k

        for idx, item in tqdm(enumerate(items, start=args.start_index),
                              total=len(items), desc="Scoring items"):
            result = scorer.score_single_item(
                question=item['question'],
                answer=item['answer'],
                context=item['context'],
            )

            n = len(result['chunks'])
            fname = f"{dataset_name}_{idx}.json"
            q, a = item['question'], item['answer']
            chunk_contents = [c['content'] for c in result['chunks']]

            # 1) chunk/ — original order, no scores
            _write_json(dirs['chunk'] / fname, {
                'input': q, 'answers': a, 'chunk_list': chunk_contents,
            })

            # Helper: build sorted output for a given sort key
            def _sorted_output(sort_key, limit=None):
                indices = sorted(range(n), key=lambda i: result[sort_key][i], reverse=True)
                if limit:
                    indices = indices[:limit]
                return {
                    'input': q, 'answers': a,
                    'chunk_list': [chunk_contents[i] for i in indices],
                    'score_list': [result['scores'][i] for i in indices],
                    'forward_list': [result['forward_scores'][i] for i in indices],
                    'backward_list': [result['backward_scores'][i] for i in indices],
                    'parameter_list': [result['parameter_scores'][i] for i in indices],
                }

            # 2) full/ — sub-dirs sorted by each score type
            for sub, sort_key in [('score', 'scores'), ('forward', 'forward_scores'),
                                  ('backward', 'backward_scores'), ('parameter', 'parameter_scores')]:
                _write_json(dirs[f'full/{sub}'] / fname, _sorted_output(sort_key))

            # 3) topk/ — same as full but top-k, with cleaned_content
            for sub, sort_key in [('score', 'scores'), ('forward', 'forward_scores'),
                                  ('backward', 'backward_scores'), ('parameter', 'parameter_scores')]:
                out = _sorted_output(sort_key, limit=k)
                out['chunk_list'] = [
                    {'content': c, 'cleaned_content': clean_content(c)}
                    for c in out['chunk_list']
                ]
                _write_json(dirs[f'topk/{sub}'] / fname, out)

            # 4) full-rank/ — ranking indices (unsorted)
            def _rank(key):
                return sorted(range(n), key=lambda i: result[key][i], reverse=True)

            _write_json(dirs['full-rank'] / fname, {
                'input': q, 'answers': a,
                'forward_rank': _rank('forward_scores'),
                'backward_rank': _rank('backward_scores'),
                'parameter_rank': _rank('parameter_scores'),
            })

            # 5) topk-rank/ — top-k ranking indices
            _write_json(dirs['topk-rank'] / fname, {
                'input': q, 'answers': a,
                'forward_rank': _rank('forward_scores')[:k],
                'backward_rank': _rank('backward_scores')[:k],
                'parameter_rank': _rank('parameter_scores')[:k],
            })

        print(f"Done! Wrote {len(items)} items to {output_dir}/{{chunk,full,topk,full-rank,topk-rank}}/")


def _write_json(path, obj):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
