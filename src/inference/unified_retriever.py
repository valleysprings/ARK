"""
Unified Retriever for ARK - Supports multiple embedding models
"""

import os
import torch
import warnings
from typing import List, Tuple, Dict, Any
import yaml
import re
import string

warnings.filterwarnings("ignore")

# Import different embedding models
from FlagEmbedding import BGEM3FlagModel
from sentence_transformers import SentenceTransformer
import requests

# Import prompts
from .prompts import DATASET2PROMPT, ULTRADOMAIN_PROMPT


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def normalize_zh_answer(s):
    """Lower text and remove punctuation, extra whitespace."""

    def white_space_fix(text):
        return "".join(text.split())

    def remove_punc(text):
        cn_punctuation = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—''‛""„‟…‧﹏."
        all_punctuation = set(string.punctuation + cn_punctuation)
        return "".join(ch for ch in text if ch not in all_punctuation)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))


class UnifiedRetriever:
    """
    Unified retriever supporting multiple embedding models:
    - Qwen3-Embedding (fine-tuned or base)
    - BGE-M3 (multi-vector: dense + sparse + colbert)
    - Jina Embeddings v3
    - Stella v5
    """

    SUPPORTED_MODELS = ["qwen", "bge", "jina", "stella"]

    def __init__(self, config: Dict[str, Any], device: str = "cuda:0"):
        """
        Initialize retriever from config

        Args:
            config: Configuration dictionary (from config.yaml)
            device: CUDA device to use
        """
        self.config = config
        self.device = torch.device(device) if torch.cuda.is_available() else torch.device("cpu")

        # Get retriever config
        retriever_config = config.get("inference", {}).get("retriever", {})
        self.model_type = retriever_config.get("type", "qwen")

        if self.model_type not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model type: {self.model_type}. "
                           f"Choose from: {self.SUPPORTED_MODELS}")

        print(f"Initializing {self.model_type.upper()} retriever...")
        self._load_model()

    def _load_model(self):
        """Load the appropriate model based on config"""
        retriever_config = self.config["inference"]["retriever"]
        model_config = retriever_config.get(self.model_type, {})

        if self.model_type == "qwen":
            model_path = model_config.get("model_path", "Qwen/Qwen3-Embedding-0.6B")
            self.model = SentenceTransformer(
                model_path,
                device=str(self.device),
                model_kwargs={"torch_dtype": "bfloat16"}
            )
            self.top_k = model_config.get("top_k", 5)

        elif self.model_type == "bge":
            model_path = model_config.get("model_path", "model/raw/bge-m3")
            self.model = BGEM3FlagModel(
                model_path,
                use_fp16=True,
                device=str(self.device)
            )
            self.top_k = model_config.get("top_k", 10)
            self.use_reranker = model_config.get("use_reranker", False)
            self.dense_weight = model_config.get("dense_weight", 1.0)
            self.sparse_weight = model_config.get("sparse_weight", 0.3)
            self.colbert_weight = model_config.get("colbert_weight", 1.0)

        elif self.model_type == "jina":
            model_path = model_config.get("model_path", "jinaai/jina-embeddings-v3")
            trust_remote_code = model_config.get("trust_remote_code", True)
            self.model = SentenceTransformer(
                model_path,
                device=str(self.device),
                trust_remote_code=trust_remote_code,
                model_kwargs={"torch_dtype": torch.bfloat16}
            )
            self.top_k = model_config.get("top_k", 5)

        elif self.model_type == "stella":
            model_path = model_config.get("model_path", "NovaSearch/stella_en_1.5B_v5")
            self.model = SentenceTransformer(
                model_path,
                device=str(self.device),
                model_kwargs={"torch_dtype": torch.bfloat16}
            )
            self.top_k = model_config.get("top_k", 5)

        print(f"Model loaded: {self.model_type} (top_k={self.top_k})")

    def retrieve(self, query: str, chunks: List[str]) -> List[Tuple[int, float]]:
        """
        Retrieve top-k most relevant chunks for query

        Args:
            query: Query string
            chunks: List of text chunks

        Returns:
            List of (chunk_index, score) tuples, sorted by score descending
        """
        if self.model_type == "qwen":
            return self._retrieve_qwen(query, chunks)
        elif self.model_type == "bge":
            return self._retrieve_bge(query, chunks)
        elif self.model_type == "jina":
            return self._retrieve_jina(query, chunks)
        elif self.model_type == "stella":
            return self._retrieve_stella(query, chunks)

    def _retrieve_qwen(self, query: str, chunks: List[str]) -> List[Tuple[int, float]]:
        """Qwen3-Embedding retrieval"""
        query_emb = self.model.encode(query, prompt_name="query")
        chunk_embs = self.model.encode(chunks)

        similarities = self.model.similarity(query_emb, chunk_embs)[0]

        # Get top-k indices
        top_indices = torch.argsort(similarities, descending=True)[:self.top_k]

        results = [
            (idx.item(), similarities[idx].item())
            for idx in top_indices
        ]
        return results

    def _retrieve_bge(self, query: str, chunks: List[str]) -> List[Tuple[int, float]]:
        """BGE-M3 multi-vector retrieval"""
        # Encode query
        query_emb = self.model.encode(
            [query],
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=True
        )

        # Encode chunks
        chunk_embs = self.model.encode(
            chunks,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=True
        )

        # Compute scores for each embedding type
        dense_scores = query_emb['dense_vecs'] @ chunk_embs['dense_vecs'].T

        # Sparse scores (lexical matching)
        sparse_scores = self.model.compute_lexical_matching_score(
            [query_emb['lexical_weights'][0]],
            chunk_embs['lexical_weights']
        )[0]  # Extract first row since query is wrapped in list

        # ColBERT scores - compute for each chunk individually
        import numpy as np
        colbert_scores = np.array([
            self.model.colbert_score(
                query_emb['colbert_vecs'][0],
                chunk_vec
            )
            for chunk_vec in chunk_embs['colbert_vecs']
        ])

        # Weighted combination
        combined_scores = (
            self.dense_weight * dense_scores[0] +
            self.sparse_weight * sparse_scores +
            self.colbert_weight * colbert_scores
        )

        # Get top-k
        top_indices = torch.argsort(torch.tensor(combined_scores), descending=True)[:self.top_k]

        results = [
            (idx.item(), combined_scores[idx].item())
            for idx in top_indices
        ]
        return results

    def _retrieve_jina(self, query: str, chunks: List[str]) -> List[Tuple[int, float]]:
        """Jina Embeddings retrieval"""
        query_emb = self.model.encode(query)
        chunk_embs = self.model.encode(chunks)

        # Cosine similarity
        similarities = torch.nn.functional.cosine_similarity(
            torch.tensor(query_emb).unsqueeze(0),
            torch.tensor(chunk_embs),
            dim=1
        )

        top_indices = torch.argsort(similarities, descending=True)[:self.top_k]

        results = [
            (idx.item(), similarities[idx].item())
            for idx in top_indices
        ]
        return results

    def _retrieve_stella(self, query: str, chunks: List[str]) -> List[Tuple[int, float]]:
        """Stella Embeddings retrieval"""
        query_emb = self.model.encode(query)
        chunk_embs = self.model.encode(chunks)

        # Cosine similarity
        similarities = torch.nn.functional.cosine_similarity(
            torch.tensor(query_emb).unsqueeze(0),
            torch.tensor(chunk_embs),
            dim=1
        )

        top_indices = torch.argsort(similarities, descending=True)[:self.top_k]

        results = [
            (idx.item(), similarities[idx].item())
            for idx in top_indices
        ]
        return results


def text_to_chunks(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Split text into overlapping chunks

    Args:
        text: Input text
        chunk_size: Number of words per chunk
        overlap: Number of overlapping words

    Returns:
        List of text chunks
    """
    words = text.split(' ')
    chunks = []
    step = chunk_size - overlap

    for i in range(0, len(words), step):
        chunk = ' '.join(words[i:i + chunk_size]).strip()
        if chunk:
            chunks.append(chunk)

    return chunks


def llm_generate(prompt: str, endpoint: str = "http://localhost:11434",
                model: str = "mistral:latest") -> str:
    """
    Generate answer using Ollama LLM

    Args:
        prompt: Full prompt including context and question
        endpoint: Ollama endpoint
        model: Model name

    Returns:
        Generated answer
    """
    try:
        response = requests.post(
            f"{endpoint}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9
                }
            },
            timeout=60
        )
        response.raise_for_status()
        return response.json()["response"].strip()
    except Exception as e:
        print(f"LLM generation error: {e}")
        return ""


# Convenience function to load retriever from config file
def load_retriever_from_config(config_path: str, device: str = "cuda:0") -> UnifiedRetriever:
    """
    Load retriever from YAML config file

    Args:
        config_path: Path to config.yaml
        device: CUDA device

    Returns:
        UnifiedRetriever instance
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return UnifiedRetriever(config, device=device)
