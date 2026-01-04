"""
Training Data Generation Operations

This module provides functions for generating training data (preference pairs)
from generated queries and document chunks.
"""

import json
import os
import pickle
import jsonlines
from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer, util
from typing import List, Dict, Any


def load_generated_queries(query_dir: str) -> Dict[str, List[str]]:
    """
    Load all generated queries from pkl files in a directory

    Args:
        query_dir: Directory containing query pkl files (format: {dataset}_{index}.pkl)

    Returns:
        Dictionary mapping index to list of queries
    """
    all_queries = {}

    # Find all query pkl files
    if not os.path.exists(query_dir):
        raise ValueError(f"Query directory not found: {query_dir}")

    query_files = [f for f in os.listdir(query_dir) if f.endswith('.pkl')]

    for query_file in query_files:
        # Extract index from filename (e.g., hotpotqa_0.pkl -> 0)
        try:
            index = query_file.rsplit('_', 1)[1].replace('.pkl', '')
            with open(os.path.join(query_dir, query_file), 'rb') as f:
                queries = pickle.load(f)
                all_queries[index] = queries
        except Exception as e:
            print(f"Warning: Failed to load {query_file}: {e}")
            continue

    return all_queries


def generate_preference_pairs(
    alignment_dir: str,
    generated_queries: Dict[str, List[str]],
    embedding_model_path: str,
    device: str = "cuda",
    top_n: int = 3,
    top_m: int = 10,
) -> Dict[str, Dict[str, Any]]:
    """
    Generate preference pairs for training

    Args:
        alignment_dir: Directory containing alignment score PKL files
        generated_queries: Dictionary mapping index to list of generated queries
        embedding_model_path: Path to sentence embedding model
        device: Device for model inference
        top_n: Number of top chunks for chosen response
        top_m: Number of similar chunks to retrieve for rejected response

    Returns:
        Dictionary mapping index to preference pair data
    """
    # Load embedding model
    print(f"Loading embedding model from: {embedding_model_path}")
    model = SentenceTransformer(embedding_model_path, device=device)
    print("Embedding model loaded.")

    results = {}

    # Load alignment data from pkl files
    import glob
    alignment_files = sorted(glob.glob(os.path.join(alignment_dir, "*.pkl")))

    print(f"Found {len(alignment_files)} alignment files")

    for alignment_file in tqdm(alignment_files, desc="Processing data"):
        # Extract index from filename (e.g., hotpotqa_0.pkl -> 0)
        filename = os.path.basename(alignment_file)
        try:
            idx = int(filename.rsplit('_', 1)[1].replace('.pkl', ''))
        except (ValueError, IndexError):
            print(f"Warning: Could not extract index from {filename}")
            continue

        # Load alignment data
        with open(alignment_file, 'rb') as f:
            item = pickle.load(f)

        # Get original query and chunks
        original_query = item.get('input', item.get('question', ''))
        original_chunk_list = item.get('chunk_list', [])

        if not original_chunk_list:
            continue

        # Top N chunks as chosen response
        chosen_response_chunks = original_chunk_list[:top_n]
        chosen_chunks_set = set(chosen_response_chunks)

        # Get generated queries for this index
        new_queries = generated_queries.get(str(idx), [])
        if not new_queries:
            continue

        # Encode all chunks once
        chunk_embeddings = model.encode(
            original_chunk_list,
            convert_to_tensor=True,
            show_progress_bar=False
        )

        # Process each generated query
        for new_query in new_queries:
                # Encode query
                query_embedding = model.encode(new_query, convert_to_tensor=True)

                # Compute similarities
                similarities = util.cos_sim(query_embedding, chunk_embeddings)[0]

                # Get top-M similar chunks
                chunk_sim_pairs = list(zip(original_chunk_list, similarities.cpu().tolist()))
                chunk_sim_pairs.sort(key=lambda x: x[1], reverse=True)
                top_m_similar_chunks = [chunk for chunk, sim in chunk_sim_pairs[:top_m]]

                # Rejected chunks: top-M chunks that are not in chosen set
                rejected_response_chunks = [
                    chunk for chunk in top_m_similar_chunks
                    if chunk not in chosen_chunks_set
                ]

                # Only add if we have rejected chunks
                if rejected_response_chunks:
                    if idx not in results:
                        results[idx] = []
                    results[idx].append({
                        "query": new_query,  # Use generated query
                        "original_query": original_query,  # Keep original for reference
                        "response": chosen_response_chunks,
                        "rejected_response": rejected_response_chunks
                    })

    return results
