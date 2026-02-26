"""
Embedding Utilities for Answer Augmented Retrieval

Provides embedding functions and wrappers for entity vectorization.
Supports Ollama (BGE-M3) and SentenceTransformer backends.
"""

import asyncio
from dataclasses import dataclass
from functools import wraps
from typing import Any
import numpy as np

try:
    from ollama import Client as OllamaClient
    _ollama_available = True
except ImportError:
    _ollama_available = False

try:
    from .text_processing import compute_mdhash_id
except ImportError:
    from text_processing import compute_mdhash_id


# ============================================================================
# Configuration
# ============================================================================

# Ollama Client Configuration
OLLAMA_HOST = "http://localhost:11434"
client = OllamaClient(host=OLLAMA_HOST) if _ollama_available else None

# Embedding Model Configuration
EMBEDDING_MODEL = "bge-m3:latest"
EMBEDDING_MODEL_DIM = 1024
EMBEDDING_MODEL_MAX_TOKENS = 8000
MAX_ASYNC_CALL_SIZE = 10

# SentenceTransformer singleton
_st_model = None
_st_model_path = None


# ============================================================================
# Embedding Function Wrapper
# ============================================================================

@dataclass
class EmbeddingFunc:
    """
    Wrapper class for embedding functions with metadata

    Stores embedding dimension and max token size alongside the function.
    Provides a callable interface for async embedding generation.
    """
    embedding_dim: int
    max_token_size: int
    func: callable

    async def __call__(self, *args, **kwargs) -> np.ndarray:
        """Call the underlying embedding function"""
        return await self.func(*args, **kwargs)


def wrap_embedding_func_with_attrs(**kwargs):
    """
    Decorator to wrap embedding functions with metadata

    Adds embedding_dim and max_token_size attributes to the function.

    Args:
        **kwargs: Metadata to attach (embedding_dim, max_token_size)

    Returns:
        Decorator function

    Example:
        >>> @wrap_embedding_func_with_attrs(embedding_dim=1024, max_token_size=8000)
        ... async def my_embedding_func(texts):
        ...     return embeddings
    """
    def decorator(func) -> EmbeddingFunc:
        new_func = EmbeddingFunc(**kwargs, func=func)
        return new_func

    return decorator


def limit_async_func_call(max_size: int, waiting_time: float = 0.0001):
    """
    Decorator to limit concurrent async function calls

    Prevents overwhelming the embedding service with too many simultaneous requests.

    Args:
        max_size: Maximum number of concurrent calls allowed
        waiting_time: Time to wait between checks (in seconds)

    Returns:
        Decorated async function with concurrency limiting
    """
    def decorator(func):
        """Not using asyncio.Semaphore to avoid nest-asyncio issues"""
        __current_size = 0

        @wraps(func)
        async def wait_func(*args, **kwargs):
            nonlocal __current_size
            while __current_size >= max_size:
                await asyncio.sleep(waiting_time)
            __current_size += 1
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                __current_size -= 1

        return wait_func

    return decorator


# ============================================================================
# Embedding Functions
# ============================================================================

@limit_async_func_call(max_size=MAX_ASYNC_CALL_SIZE, waiting_time=0.01)
@wrap_embedding_func_with_attrs(
    embedding_dim=EMBEDDING_MODEL_DIM,
    max_token_size=EMBEDDING_MODEL_MAX_TOKENS,
)
async def ollama_embedding(texts: list[str]) -> np.ndarray:
    """
    Generate embeddings using Ollama with BGE-M3 model

    This is the primary embedding function for the system. It uses a locally
    hosted Ollama server with the BGE-M3 model (1024-dimensional embeddings).

    Args:
        texts: List of text strings to embed

    Returns:
        NumPy array of embeddings with shape (len(texts), 1024)

    Example:
        >>> texts = ["Entity description 1", "Entity description 2"]
        >>> embeddings = await ollama_embedding(texts)
        >>> embeddings.shape
        (2, 1024)

    Note:
        - Requires Ollama server running at localhost:11434
        - Uses BGE-M3:latest model (must be pulled: ollama pull bge-m3:latest)
        - Max context: 8192 tokens
        - Embedding dimension: 1024
        - Concurrency limited to MAX_ASYNC_CALL_SIZE to prevent overload
    """
    data = client.embed(
        model=EMBEDDING_MODEL,
        input=texts,
        options={"num_ctx": 8192}
    )
    embed_text = data["embeddings"]

    return np.array(embed_text)


# ============================================================================
# Local GPU Embedding (BGE-M3 / Qwen3-Embedding)
# ============================================================================

_local_model = None
_local_model_path = None
_local_model_type = None  # "bge" or "qwen"


def _get_local_model(model_path: str, model_type: str):
    """Lazy-load local embedding model singleton."""
    global _local_model, _local_model_path, _local_model_type
    if _local_model is not None and _local_model_path == model_path:
        return _local_model, _local_model_type

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if model_type == "bge":
        from FlagEmbedding import BGEM3FlagModel
        _local_model = BGEM3FlagModel(model_path, use_fp16=True, device=device)
    else:  # qwen
        from sentence_transformers import SentenceTransformer
        _local_model = SentenceTransformer(
            model_path, device=device,
            model_kwargs={"dtype": "bfloat16"}
        )

    _local_model_path = model_path
    _local_model_type = model_type
    return _local_model, _local_model_type


def create_local_embedding(model_path: str = "model/raw/bge-m3",
                           model_type: str = "bge") -> "EmbeddingFunc":
    """
    Create a local GPU embedding function (BGE-M3 or Qwen3-Embedding).

    Args:
        model_path: Path to embedding model
        model_type: "bge" or "qwen"
    """
    @wrap_embedding_func_with_attrs(
        embedding_dim=EMBEDDING_MODEL_DIM,
        max_token_size=EMBEDDING_MODEL_MAX_TOKENS,
    )
    async def _local_embedding(texts: list[str]) -> np.ndarray:
        model, mtype = _get_local_model(model_path, model_type)
        if mtype == "bge":
            result = model.encode(texts, return_dense=True,
                                  return_sparse=False, return_colbert_vecs=False)
            return np.array(result['dense_vecs'])
        else:
            return np.array(model.encode(texts))

    return _local_embedding


# ============================================================================
# Vector Database Utilities
# ============================================================================

async def vdb_upsert(entity_vdb: Any, all_entities_data: list[dict]) -> Any:
    """
    Upsert entities to vector database

    Converts entity data to the format required by the vector database
    and performs batch upsert operation.

    Args:
        entity_vdb: NanoVectorDBStorage instance
        all_entities_data: List of entity dictionaries from graph generation

    Returns:
        Result from vector database upsert operation

    Example:
        >>> from vectordb.vdb import NanoVectorDBStorage
        >>> vectorizer = NanoVectorDBStorage(
        ...     embedding_func=ollama_embedding,
        ...     namespace="entities",
        ...     working_dir="./embeddings"
        ... )
        >>> entities = [{"entity_name": "JOHN", "description": "CEO..."}]
        >>> await vdb_upsert(vectorizer, entities)

    Note:
        - Creates unique IDs using MD5 hash with "ent-" prefix
        - Combines entity name and description for embedding
        - Embeddings are generated in batches by the vector database
    """
    data_for_vdb = {
        compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
            "content": dp["entity_name"] + dp["description"],
            "entity_name": dp["entity_name"],
        }
        for dp in all_entities_data
    }
    return await entity_vdb.upsert(data_for_vdb)


# ============================================================================
# Configuration Utilities
# ============================================================================

def get_embedding_config() -> dict:
    """
    Get current embedding configuration

    Returns:
        Dictionary with embedding configuration details
    """
    return {
        "model": EMBEDDING_MODEL,
        "dimension": EMBEDDING_MODEL_DIM,
        "max_tokens": EMBEDDING_MODEL_MAX_TOKENS,
        "host": OLLAMA_HOST,
        "max_concurrent_calls": MAX_ASYNC_CALL_SIZE,
    }


def set_ollama_host(host: str) -> None:
    """
    Set Ollama host URL

    Args:
        host: URL of Ollama server (e.g., "http://localhost:11434")

    Example:
        >>> set_ollama_host("http://192.168.1.100:11434")
    """
    global client, OLLAMA_HOST
    OLLAMA_HOST = host
    client = Client(host=host)
