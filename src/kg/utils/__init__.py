"""
KG Utilities Module

Utilities specifically for Knowledge Graph operations.
"""

# LLM Client
from .llm_client import (
    google_model,
    gpt_model,
    deepseek_model,
    GPT_API_KEY,
    GPT_MODEL,
    GPT_BASE_URL,
    DEEPSEEK_API_KEY,
    DEEPSEEK_MODEL,
    DEEPSEEK_BASE_URL,
    GOOGLE_API_KEY,
    GOOGLE_MODEL,
)

# Token Tracking
from .token_tracker import (
    TokenCostTracker,
    SimpleTokenTracker,
    get_simple_tracker,
    print_token_summary,
    print_token_details,
    reset_token_tracker,
    get_global_session_summary,
    reset_global_session_stats,
)

# File Operations
from .file_operations import (
    read_source_document,
    parse_jsonl,
    cache_exists,
    load_cache,
    save_cache,
    get_cache_path,
)

# Text Processing
from .text_processing import (
    doc_to_chunks,
    split_string_by_multi_markers,
    clean_str,
    is_float_regex,
    encode_string_by_tiktoken,
    decode_tokens_by_tiktoken,
    compute_mdhash_id,
)

# Embeddings
from .embeddings import (
    ollama_embedding,
)

__all__ = [
    # LLM Client
    "google_model",
    "gpt_model",
    "deepseek_model",
    "GPT_API_KEY",
    "GPT_MODEL",
    "GPT_BASE_URL",
    "DEEPSEEK_API_KEY",
    "DEEPSEEK_MODEL",
    "DEEPSEEK_BASE_URL",
    "GOOGLE_API_KEY",
    "GOOGLE_MODEL",

    # Token Tracking
    "TokenCostTracker",
    "SimpleTokenTracker",
    "get_simple_tracker",
    "print_token_summary",
    "print_token_details",
    "reset_token_tracker",
    "get_global_session_summary",
    "reset_global_session_stats",

    # File Operations
    "read_source_document",
    "parse_jsonl",
    "cache_exists",
    "load_cache",
    "save_cache",
    "get_cache_path",

    # Text Processing
    "doc_to_chunks",
    "split_string_by_multi_markers",
    "clean_str",
    "is_float_regex",
    "encode_string_by_tiktoken",
    "decode_tokens_by_tiktoken",
    "compute_mdhash_id",

    # Embeddings
    "ollama_embedding",
]
