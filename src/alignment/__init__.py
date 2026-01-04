"""
ARK Alignment Module

This module provides triple alignment scoring and contrastive pair generation
for answer-centric retriever training via curriculum-based contrastive learning.

Components:
-----------
1. AlignmentScorer: Triple alignment scoring system
   - Forward alignment: P(answer|chunk, question)
   - Backward alignment: P(question|chunk, answer)
   - Parameter alignment: cosine(query_embedding, chunk_embedding)

2. PreferenceGenerator: Contrastive pair generation for retriever training
   - Generates positive (answer-sufficient) vs hard negative (insufficient) pairs
   - Based on semantic similarity and alignment scores
   - Used for curriculum-based contrastive learning with InfoNCE loss

Usage Examples:
--------------

Basic Alignment Scoring:
```python
from alignment import AlignmentScorer, AlignmentConfig

# Load configuration
config = AlignmentConfig.from_yaml('config.yaml')

# Initialize scorer
scorer = AlignmentScorer(
    lm_model_path='path/to/language_model',
    embedding_model_path='path/to/embedding_model',
    config=config
)

# Score a single item
result = scorer.score_single_item(
    question="What is the capital of France?",
    answer="Paris",
    context="France is a country in Europe. Its capital is Paris...",
    top_k=10
)

# Access results
print(f"Top chunks: {result['chunks']}")
print(f"Scores: {result['scores']}")
```

Batch Processing with Async:
```python
import asyncio
from alignment import AlignmentScorer

async def score_dataset():
    scorer = AlignmentScorer(lm_model_path, embedding_model_path)

    items = [
        {'question': q1, 'answer': a1, 'context': c1},
        {'question': q2, 'answer': a2, 'context': c2},
        # ... more items
    ]

    results = await scorer.score_batch_async(
        items,
        top_k=10,
        max_concurrent=5
    )

    return results

# Run async scoring
results = asyncio.run(score_dataset())
```

Preference Pair Generation:
```python
from alignment import PreferenceGenerator, PreferenceConfig

# Initialize generator
config = PreferenceConfig(top_n=3, top_m=10)
generator = PreferenceGenerator(
    embedding_model_path='path/to/embedding_model',
    config=config
)

# Generate preference pairs
num_pairs = generator.generate_from_scored_data(
    scored_data_path='scored_output.jsonl',
    new_queries_path='augmented_queries.json',
    output_path='preference_pairs.jsonl'
)

print(f"Generated {num_pairs} preference pairs")
```

Validate Preference Pairs:
```python
from alignment import PreferenceDataset

# Load and validate
data = PreferenceDataset.load('preference_pairs.jsonl')
num_valid, errors = PreferenceDataset.validate(data)

print(f"Valid: {num_valid}/{len(data)}")
if errors:
    for error in errors:
        print(error)

# Filter by length
filtered = PreferenceDataset.filter_by_length(
    data,
    min_response_length=1,
    max_response_length=5
)

# Save filtered data
PreferenceDataset.save(filtered, 'filtered_pairs.jsonl')
```

Author: ARK Project
Date: 2026-01-02
"""

from .scorer import (
    AlignmentScorer,
    AlignmentConfig,
)

from .preference import (
    PreferenceGenerator,
    PreferenceConfig,
    PreferenceDataset,
)

__version__ = "2.0.0"

__all__ = [
    # Scorer exports
    "AlignmentScorer",
    "AlignmentConfig",

    # Preference exports
    "PreferenceGenerator",
    "PreferenceConfig",
    "PreferenceDataset",
]


# Module-level convenience functions

def create_scorer_from_config(
    config_path: str,
    lm_model_path: str,
    embedding_model_path: str
) -> AlignmentScorer:
    """Create an AlignmentScorer from a YAML config file.

    Args:
        config_path: Path to config.yaml
        lm_model_path: Path to language model
        embedding_model_path: Path to embedding model

    Returns:
        Configured AlignmentScorer instance
    """
    config = AlignmentConfig.from_yaml(config_path)
    return AlignmentScorer(lm_model_path, embedding_model_path, config)


def create_preference_generator_from_config(
    config_path: str,
    embedding_model_path: str
) -> PreferenceGenerator:
    """Create a PreferenceGenerator from a YAML config file.

    Args:
        config_path: Path to config.yaml
        embedding_model_path: Path to embedding model

    Returns:
        Configured PreferenceGenerator instance
    """
    config = PreferenceConfig.from_yaml(config_path)
    return PreferenceGenerator(embedding_model_path, config)


# Add convenience functions to __all__
__all__.extend([
    "create_scorer_from_config",
    "create_preference_generator_from_config",
])
