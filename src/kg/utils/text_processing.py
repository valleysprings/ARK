"""
Text Processing Utilities for Answer Augmented Retrieval

Provides functions for text encoding, chunking, and document processing.
Uses tiktoken for efficient token-level operations.
"""

import os
os.environ.setdefault("TIKTOKEN_CACHE_DIR", os.path.expanduser("~/.cache/tiktoken_cache"))

import re
import tiktoken
from typing import List, Dict, TypedDict
from hashlib import md5


# ============================================================================
# Type Definitions
# ============================================================================

TextChunkSchema = TypedDict(
    "TextChunkSchema",
    {"tokens": int, "content": str, "chunk_order_index": int},
)


# ============================================================================
# Global Encoder
# ============================================================================

# Global tiktoken encoder (initialized on first use)
ENCODER = tiktoken.encoding_for_model("gpt-4o")


# ============================================================================
# Hash Functions
# ============================================================================

def compute_mdhash_id(content: str, prefix: str = "") -> str:
    """
    Compute MD5 hash for content with optional prefix

    Creates a unique identifier for text content using MD5 hashing.
    Useful for deduplication and content tracking.

    Args:
        content: Text content to hash
        prefix: Optional prefix for the hash ID (e.g., "doc-", "chunk-")

    Returns:
        MD5 hash string with prefix

    Example:
        >>> compute_mdhash_id("hello world", prefix="doc-")
        'doc-5eb63bbbe01eeed093cb22bb8f5acdc3'
    """
    return prefix + md5(content.encode()).hexdigest()


# ============================================================================
# Token Encoding/Decoding
# ============================================================================

def encode_string_by_tiktoken(content: str, model_name: str = "gpt-4o") -> List[int]:
    """
    Encode a string into tokens using tiktoken

    Args:
        content: Text content to encode
        model_name: Model name for the tokenizer (default: "gpt-4o")

    Returns:
        List of token IDs

    Example:
        >>> tokens = encode_string_by_tiktoken("Hello world")
        >>> len(tokens)
        2
    """
    global ENCODER
    if ENCODER is None:
        ENCODER = tiktoken.encoding_for_model(model_name)
    tokens = ENCODER.encode(content)
    return tokens


def decode_tokens_by_tiktoken(tokens: List[int], model_name: str = "gpt-4o") -> str:
    """
    Decode tokens back to string using tiktoken

    Args:
        tokens: List of token IDs
        model_name: Model name for the tokenizer (default: "gpt-4o")

    Returns:
        Decoded text string

    Example:
        >>> tokens = [9906, 1917]
        >>> decode_tokens_by_tiktoken(tokens)
        'Hello world'
    """
    global ENCODER
    if ENCODER is None:
        ENCODER = tiktoken.encoding_for_model(model_name)
    content = ENCODER.decode(tokens)
    return content


# ============================================================================
# Text Chunking
# ============================================================================

def _split_long_sentence(text: str, max_tokens: int = 256, min_tokens: int = 30) -> List[str]:
    global ENCODER

    tok_cache = {}
    def tok_len(s: str) -> int:
        v = tok_cache.get(s)
        if v is None:
            v = len(ENCODER.encode(s))
            tok_cache[s] = v
        return v

    if tok_len(text) <= max_tokens:
        return [text]

    parts = re.split(r'(\n\s*-{10,}\s*\n|\n\n+)', text)

    segs: List[str] = []
    for i in range(0, len(parts), 2):
        seg = parts[i]
        if i + 1 < len(parts):
            seg += parts[i + 1]
        if seg != "":
            segs.append(seg)

    out: List[str] = []
    cur = ""
    cur_tok = 0

    for seg in segs:
        st = tok_len(seg)
        if cur and cur_tok + st > max_tokens:
            if out and cur_tok < min_tokens and tok_len(out[-1]) + cur_tok <= max_tokens:
                out[-1] += cur
            else:
                out.append(cur)
            cur, cur_tok = seg, st
            continue

        cur += seg
        cur_tok += st

    if cur:
        if out and cur_tok < min_tokens and tok_len(out[-1]) + cur_tok <= max_tokens:
            out[-1] += cur
        else:
            out.append(cur)

    return out if out else [text]


def chunking_by_sentence(
    text: str,
    max_sentences: int = 5,
    overlap_sentences: int = 1,
    max_token_per_sentence: int = 256,
) -> List[TextChunkSchema]:
    """Chunk text by sentence count with overlap.

    Uses nltk sent_tokenize, then further splits any sentence exceeding
    max_token_per_sentence on paragraph/section boundaries and semicolons
    before enumeration markers (common in legal text).
    """
    from nltk.tokenize import sent_tokenize
    global ENCODER

    raw_sentences = sent_tokenize(text)
    sentences = []
    for s in raw_sentences:
        sentences.extend(_split_long_sentence(s, max_token_per_sentence))

    chunks = []
    step = max(max_sentences - overlap_sentences, 1)
    for i in range(0, len(sentences), step):
        group = sentences[i:i + max_sentences]
        content = ' '.join(group)
        chunks.append({
            "tokens": len(ENCODER.encode(content)),
            "content": content,
            "chunk_order_index": len(chunks),
        })
    return chunks


def chunking_by_token_size(
    tokens_list: List[List[int]],
    tiktoken_model: tiktoken.Encoding,
    overlap_token_size: int = 128,
    max_token_size: int = 1024,
) -> List[TextChunkSchema]:
    """
    Chunk documents by token size with overlap

    Args:
        tokens_list: List of tokenized documents (each doc is a list of token IDs)
        tiktoken_model: Tiktoken encoding model for decoding
        overlap_token_size: Number of tokens to overlap between chunks (default: 128)
        max_token_size: Maximum tokens per chunk (default: 1024)

    Returns:
        List of chunk dictionaries with metadata:
            - tokens: Number of tokens in chunk
            - content: Decoded text content
            - chunk_order_index: Position in document

    Example:
        >>> encoder = tiktoken.encoding_for_model("gpt-4o")
        >>> doc_tokens = [encoder.encode("Your long document text...")]
        >>> chunks = chunking_by_token_size(
        ...     doc_tokens, encoder, overlap_token_size=100, max_token_size=500
        ... )
    """
    results = []

    for index, tokens in enumerate(tokens_list):
        chunk_token = []
        lengths = []

        # Create overlapping chunks
        for start in range(0, len(tokens), max_token_size - overlap_token_size):
            chunk_token.append(tokens[start : start + max_token_size])
            lengths.append(min(max_token_size, len(tokens) - start))

        # Decode chunks (batch decoding for efficiency)
        # Note: chunk_token is list[list[int]], so decode_batch is used
        chunk_token = tiktoken_model.decode_batch(chunk_token)

        for i, chunk in enumerate(chunk_token):
            results.append(
                {
                    "tokens": lengths[i],
                    "content": chunk.strip(),
                    "chunk_order_index": i,
                }
            )

    return results


def get_chunks(
    new_docs: Dict[str, Dict],
    chunk_func=chunking_by_token_size,
    **chunk_func_params
) -> Dict[str, TextChunkSchema]:
    """
    Process documents into chunks using specified chunking function

    High-level function that handles the complete chunking pipeline:
    1. Extracts content from document dictionary
    2. Encodes text to tokens (batch operation for efficiency)
    3. Applies chunking function
    4. Returns chunks with unique hash IDs

    Args:
        new_docs: Dictionary of documents {doc_id: {"content": text, ...}}
        chunk_func: Chunking function to use (default: chunking_by_token_size)
        **chunk_func_params: Additional parameters for the chunking function
            (e.g., overlap_token_size, max_token_size)

    Returns:
        Dictionary of chunks {chunk_id: chunk_data}
        where chunk_id is MD5 hash with "chunk-" prefix

    Example:
        >>> docs = {"doc-1": {"content": "Long text..."}}
        >>> chunks = get_chunks(docs, overlap_token_size=100, max_token_size=500)
        >>> len(chunks)
        5  # Number of chunks created
    """
    inserting_chunks = {}

    # Extract documents and keys
    new_docs_list = list(new_docs.items())
    docs = [new_doc[1]["content"] for new_doc in new_docs_list]
    # Encode documents in batch (more efficient than one-by-one)
    ENCODER = tiktoken.encoding_for_model("gpt-4o")
    tokens = ENCODER.encode_batch(docs, num_threads=16)

    # Apply chunking function
    chunks = chunk_func(
        tokens, tiktoken_model=ENCODER, **chunk_func_params
    )

    # Create hash IDs for chunks
    for chunk in chunks:
        inserting_chunks.update(
            {compute_mdhash_id(chunk["content"], prefix="chunk-"): chunk}
        )

    return inserting_chunks


def doc_to_chunks(
    doc: str | List[str],
    chunk_size: int = 600,
    overlap_size: int = 100
) -> Dict[str, TextChunkSchema]:
    """
    Convert document(s) to chunks with default settings

    Convenience function that handles the complete document-to-chunks pipeline.
    Creates document IDs, chunks text, and returns ready-to-use chunks.

    Args:
        doc: Single document string or list of document strings
        chunk_size: Maximum tokens per chunk (default: 600)
        overlap_size: Overlap tokens between chunks (default: 100)

    Returns:
        Dictionary of chunks {chunk_id: chunk_data}

    Example:
        >>> text = "Your long document text here..."
        >>> chunks = doc_to_chunks(text, chunk_size=500, overlap_size=100)
        >>> print(f"Created {len(chunks)} chunks")
        Created 3 chunks
    """
    # Normalize input to list
    if isinstance(doc, str):
        doc = [doc]

    # Create document dictionary with hash IDs
    new_docs = {
        compute_mdhash_id(c.strip(), prefix="doc-"): {"content": c.strip()}
        for c in doc
    }

    _add_doc_keys = set(new_docs.keys())
    new_docs = {k: v for k, v in new_docs.items() if k in _add_doc_keys}

    if not len(new_docs):
        print(f"All docs are already in the storage")
        return {}

    print(f"[New Docs] inserting {len(new_docs)} docs")

    # Chunk documents
    inserting_chunks = get_chunks(
        new_docs=new_docs,
        chunk_func=chunking_by_token_size,
        overlap_token_size=overlap_size,
        max_token_size=chunk_size,
    )

    print(f"[New Chunks] inserting {len(inserting_chunks)} chunks")

    return inserting_chunks


# ============================================================================
# String Processing
# ============================================================================

def clean_str(input_str: str) -> str:
    """
    Clean an input string by removing HTML escapes and control characters

    Args:
        input_str: String to clean

    Returns:
        Cleaned string

    Example:
        >>> clean_str("Hello&nbsp;World\\x00")
        'Hello World'
    """
    import html
    import re

    # If we get non-string input, just give it back
    if not isinstance(input_str, str):
        return input_str

    result = html.unescape(input_str.strip())
    # Remove control characters
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python
    return re.sub(r"[\x00-\x1f\x7f-\x9f]", "", result)


def split_string_by_multi_markers(content: str, markers: List[str]) -> List[str]:
    """
    Split a string by multiple delimiter markers

    Used for parsing LLM outputs with multiple delimiters.

    Args:
        content: String to split
        markers: List of delimiter strings

    Returns:
        List of split strings (stripped and non-empty)

    Example:
        >>> split_string_by_multi_markers("a##b||c", ["##", "||"])
        ['a', 'b', 'c']
    """
    import re

    if not markers:
        return [content]

    # Create regex pattern from markers
    results = re.split("|".join(re.escape(marker) for marker in markers), content)
    return [r.strip() for r in results if r.strip()]


def is_float_regex(value: str) -> bool:
    """
    Check if a string represents a valid float using regex

    Args:
        value: String to check

    Returns:
        True if string is a valid float, False otherwise

    Example:
        >>> is_float_regex("3.14")
        True
        >>> is_float_regex("abc")
        False
    """
    import re
    return bool(re.match(r"^[-+]?[0-9]*\.?[0-9]+$", value))
