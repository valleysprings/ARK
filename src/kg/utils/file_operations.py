"""
File Operations Utilities for Answer Augmented Retrieval

Provides functions for file I/O, JSONL parsing, and pickle serialization.
"""

import os
import json
import pickle
from typing import List, Dict, Any


# ============================================================================
# File Removal
# ============================================================================

def remove_if_exist(file: str) -> None:
    """
    Remove a file if it exists

    Safely removes a file without raising an error if it doesn't exist.

    Args:
        file: Path to the file to remove

    Example:
        >>> remove_if_exist("/tmp/old_data.pkl")
    """
    if os.path.exists(file):
        os.remove(file)


# ============================================================================
# JSONL Parsing
# ============================================================================

def parse_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """
    Parse a JSONL (JSON Lines) file into a list of dictionaries

    JSONL format has one JSON object per line. This function reads and parses
    each line, handling errors gracefully and skipping malformed lines.

    Args:
        file_path: Path to the JSONL file

    Returns:
        List of parsed JSON objects (dictionaries)

    Example:
        >>> documents = parse_jsonl("dataset/UltraDomain/fin.jsonl")
        >>> len(documents)
        150
        >>> documents[0].keys()
        dict_keys(['context', 'input', 'ideal'])

    Note:
        - Empty lines are skipped
        - Malformed JSON lines are logged and skipped
        - File not found errors are caught and reported
    """
    results = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:  # skip empty line
                    try:
                        json_obj = json.loads(line)
                        results.append(json_obj)
                    except json.JSONDecodeError as e:
                        print(f"Error parsing line {line_num}: {line[:50]}...")
                        print(f"JSON error: {str(e)}")
                        continue

    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"Error reading file: {str(e)}")

    return results


# ============================================================================
# Pickle Serialization
# ============================================================================

def pickle_dump(obj: Any, path: str) -> None:
    """
    Serialize an object to a pickle file

    Creates the directory structure if it doesn't exist. Useful for caching
    processed data (graphs, embeddings, entities, etc.).

    Args:
        obj: Object to serialize
        path: Path where the pickle file will be saved

    Example:
        >>> data = {"entities": [...], "relations": [...]}
        >>> pickle_dump(data, "./preprocessed/entities/fin_0.pkl")

    Note:
        - Automatically creates parent directories if they don't exist
        - Overwrites existing files
    """
    # Create directory if it doesn't exist
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(obj, f)


def pickle_read(path: str) -> Any:
    """
    Deserialize an object from a pickle file

    Args:
        path: Path to the pickle file

    Returns:
        Deserialized object

    Example:
        >>> data = pickle_read("./preprocessed/entities/fin_0.pkl")
        >>> type(data)
        <class 'dict'>

    Raises:
        FileNotFoundError: If the pickle file doesn't exist
    """
    with open(path, "rb") as f:
        return pickle.load(f)


# ============================================================================
# Source Document Reading
# ============================================================================

def read_source_document(path: str, index: int = 0) -> Dict[str, Any]:
    """
    Read a specific document from a JSONL file by index

    Convenience function for accessing a single document from a dataset.

    Args:
        path: Path to the JSONL file
        index: Index of the document to retrieve (default: 0)

    Returns:
        Dictionary containing the document data

    Example:
        >>> doc = read_source_document("./dataset/UltraDomain/fin.jsonl", index=5)
        >>> doc.keys()
        dict_keys(['context', 'input', 'ideal'])
        >>> len(doc['context'])
        15234

    Raises:
        IndexError: If index is out of range
    """
    source_documents = parse_jsonl(path)

    if not source_documents:
        raise ValueError(f"No documents found in {path}")

    if index >= len(source_documents):
        raise IndexError(
            f"Index {index} out of range for {len(source_documents)} documents"
        )

    return source_documents[index]


# ============================================================================
# Directory Helpers
# ============================================================================

def ensure_dir_exists(directory: str) -> None:
    """
    Ensure a directory exists, creating it if necessary

    Args:
        directory: Path to the directory

    Example:
        >>> ensure_dir_exists("./preprocessed/embeddings")
    """
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def get_cache_path(
    processed_dir: str,
    cache_type: str,
    dataset_type: str,
    index: int,
    extension: str = "pkl"
) -> str:
    """
    Generate a standardized cache file path

    Creates consistent cache paths following the pattern:
    {processed_dir}/{cache_type}/{dataset_type}_{index}.{extension}

    Args:
        processed_dir: Base directory for processed data
        cache_type: Type of cache (e.g., "entities", "full_kg", "embeddings")
        dataset_type: Dataset type identifier (e.g., "fin", "med")
        index: Document index
        extension: File extension (default: "pkl")

    Returns:
        Full path to the cache file

    Example:
        >>> path = get_cache_path(
        ...     "./preprocessed/UltraDomain",
        ...     "entities",
        ...     "fin",
        ...     0
        ... )
        >>> path
        './preprocessed/UltraDomain/entities/fin_0.pkl'
    """
    cache_dir = os.path.join(processed_dir, cache_type)
    ensure_dir_exists(cache_dir)
    filename = f"{dataset_type}_{index}.{extension}"
    return os.path.join(cache_dir, filename)


def cache_exists(
    processed_dir: str,
    cache_type: str,
    dataset_type: str,
    index: int,
    extension: str = "pkl"
) -> bool:
    """
    Check if a cache file exists

    Args:
        processed_dir: Base directory for processed data
        cache_type: Type of cache
        dataset_type: Dataset type identifier
        index: Document index
        extension: File extension (default: "pkl")

    Returns:
        True if cache file exists, False otherwise

    Example:
        >>> if cache_exists("./preprocessed/UltraDomain", "entities", "fin", 0):
        ...     print("Using cached entities")
        ... else:
        ...     print("Need to extract entities")
    """
    path = get_cache_path(processed_dir, cache_type, dataset_type, index, extension)
    return os.path.exists(path)


def load_cache(
    processed_dir: str,
    cache_type: str,
    dataset_type: str,
    index: int,
    extension: str = "pkl"
) -> Any:
    """
    Load data from cache

    Args:
        processed_dir: Base directory for processed data
        cache_type: Type of cache
        dataset_type: Dataset type identifier
        index: Document index
        extension: File extension (default: "pkl")

    Returns:
        Cached object

    Example:
        >>> entities = load_cache(
        ...     "./preprocessed/UltraDomain",
        ...     "entities",
        ...     "fin",
        ...     0
        ... )
    """
    path = get_cache_path(processed_dir, cache_type, dataset_type, index, extension)
    return pickle_read(path)


def save_cache(
    obj: Any,
    processed_dir: str,
    cache_type: str,
    dataset_type: str,
    index: int,
    extension: str = "pkl"
) -> str:
    """
    Save data to cache

    Args:
        obj: Object to cache
        processed_dir: Base directory for processed data
        cache_type: Type of cache
        dataset_type: Dataset type identifier
        index: Document index
        extension: File extension (default: "pkl")

    Returns:
        Path where the cache was saved

    Example:
        >>> save_cache(
        ...     my_entities,
        ...     "./preprocessed/UltraDomain",
        ...     "entities",
        ...     "fin",
        ...     0
        ... )
        './preprocessed/UltraDomain/entities/fin_0.pkl'
    """
    path = get_cache_path(processed_dir, cache_type, dataset_type, index, extension)
    pickle_dump(obj, path)
    return path
