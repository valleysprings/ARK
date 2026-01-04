"""
ARK Knowledge Graph Module

统一的知识图谱接口，提供：
- Construction: 从文本构建知识图谱
- Augmentation: 基于嵌入的图增强
- Extraction: 子图提取和检索

Example:
    >>> from src.kg import KnowledgeGraph
    >>>
    >>> kg = KnowledgeGraph(config_path="config.yaml")
    >>> await kg.build_from_document(document)
    >>> await kg.augment(threshold=0.8)
    >>> entities, scores = await kg.extract_subgraph("query")
"""

from .graph_builder import KnowledgeGraph

# 向后兼容：保留旧的导入方式
from .ops.construction import KGBuilder, extract_entities_and_relations, build_knowledge_graph
from .ops.augmentation import GraphAugmentor, compute_entity_embeddings, augment_graph_with_similarity
from .ops.extraction import SubgraphExtractor, personalized_pagerank, adaptive_cutoff, community_search
from .ops.query_generation import QueryGenerator

__all__ = [
    # 主类（推荐使用）
    "KnowledgeGraph",

    # 向后兼容
    "KGBuilder",
    "GraphAugmentor",
    "SubgraphExtractor",
    "QueryGenerator",

    # 函数
    "extract_entities_and_relations",
    "build_knowledge_graph",
    "compute_entity_embeddings",
    "augment_graph_with_similarity",
    "personalized_pagerank",
    "adaptive_cutoff",
    "community_search",
]
