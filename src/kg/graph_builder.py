"""
KnowledgeGraph main class - unified KG operation entry point
"""

from typing import Optional, Dict, List, Tuple, Any
from pathlib import Path
import yaml
import logging
import asyncio

from src.kg.core.nx_graph import nx_graph
from src.kg.ops.construction import KGBuilder
from src.kg.ops.augmentation import GraphAugmentor
from src.kg.ops.extraction import SubgraphExtractor

logger = logging.getLogger(__name__)


class KnowledgeGraph:
    """Unified knowledge graph operation interface."""

    def __init__(
        self,
        config_path: Optional[str] = None,
        config: Optional[Dict] = None,
        working_dir: Optional[str] = None
    ):
        # Load config
        if config is not None:
            self.config = config
        elif config_path is not None:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self._get_default_config()

        # Set working directory
        self.working_dir = working_dir or self.config.get('paths', {}).get('cache_root', './cache')
        Path(self.working_dir).mkdir(parents=True, exist_ok=True)

        # Initialize core components
        self._graph = nx_graph()
        self._vector_index = None

        # Initialize operators (lazy initialization)
        self._builder: Optional[KGBuilder] = None
        self._augmentor: Optional[GraphAugmentor] = None
        self._extractor: Optional[SubgraphExtractor] = None

        logger.info(f"KnowledgeGraph initialized with working_dir={self.working_dir}")

    # ========================================================================
    # Construction Operations
    # ========================================================================

    async def build_from_chunks(
        self,
        chunks: Dict[str, Dict],
        show_progress: bool = True
    ) -> Tuple[int, int]:
        if self._builder is None:
            self._builder = KGBuilder(config=self.config, cache_dir=self.working_dir)

        kg, entities = await self._builder.build_from_chunks(chunks, show_progress=show_progress)
        self._graph = kg
        self._entities_data = entities

        num_nodes = len(await self._graph.get_all_nodes())
        num_edges = len(await self._graph.get_all_edges())

        logger.info(f"Graph built: {num_nodes} nodes, {num_edges} edges")
        return num_nodes, num_edges

    async def build_from_document(
        self,
        document: str,
        doc_id: Optional[str] = None,
        chunk_size: Optional[int] = None,
        overlap_size: Optional[int] = None,
        use_cache: bool = True,
        show_progress: bool = True
    ) -> Tuple[int, int]:
        if self._builder is None:
            self._builder = KGBuilder(config=self.config, cache_dir=self.working_dir)

        kg, entities = await self._builder.build_from_document(
            document,
            doc_id=doc_id,
            chunk_size=chunk_size,
            overlap_size=overlap_size,
            use_cache=use_cache,
            show_progress=show_progress
        )
        self._graph = kg
        self._entities_data = entities

        num_nodes = len(await self._graph.get_all_nodes())
        num_edges = len(await self._graph.get_all_edges())

        logger.info(f"Graph built: {num_nodes} nodes, {num_edges} edges")
        return num_nodes, num_edges

    # ========================================================================
    # Augmentation Operations
    # ========================================================================

    async def augment(
        self,
        threshold: Optional[float] = None,
        force_recompute_embeddings: bool = False,
        dataset_name: str = None,
        entry_id: str = None
    ) -> int:
        # Set dataset_name and entry_id in config
        if dataset_name:
            self.config['dataset_name'] = dataset_name
        if entry_id:
            self.config['entry_id'] = entry_id

        if self._augmentor is None:
            self._augmentor = GraphAugmentor(config=self.config)

        if not hasattr(self, '_entities_data'):
            raise ValueError("No entities data available. Please build the graph first.")

        _, stats = await self._augmentor.augment(
            kg=self._graph,
            entities_data=self._entities_data,
            threshold=threshold
        )

        # Cache vector index
        self._vector_index = self._augmentor.get_vectorizer()

        logger.info(f"Graph augmented: {stats['edges_added']} edges added")
        return stats['edges_added']

    # ========================================================================
    # Extraction Operations
    # ========================================================================

    async def extract_subgraph(
        self,
        query: str,
        algorithm: str = "exact",
        use_adaptive: bool = True,
        top_k: Optional[int] = None
    ) -> Tuple[List[str], Dict[str, float]]:
        if self._extractor is None:
            graph = await self._graph.get_graph()

            # Load config for extractor
            try:
                from src.kg.ops.extraction import load_config
                extractor_config = load_config(None)  # Will use defaults
            except:
                extractor_config = None

            self._extractor = SubgraphExtractor(
                graph=graph,
                vectordb=self._vector_index,
                config=extractor_config
            )

        entities, scores = await self._extractor.extract_subgraph(
            query_text=query,
            algorithm=algorithm,
            use_adaptive=use_adaptive,
            top_k=top_k
        )

        logger.info(f"Extracted subgraph: {len(entities)} entities")
        return entities, scores

    # ========================================================================
    # Access Methods
    # ========================================================================

    async def get_graph(self):
        """Get underlying NetworkX graph."""
        return await self._graph.get_graph()

    def get_vector_index(self):
        """Get vector index."""
        return self._vector_index

    async def get_node(self, node_id: str):
        """Get node data."""
        return await self._graph.get_node(node_id)

    async def get_edge(self, src_id: str, tgt_id: str):
        """Get edge data."""
        return await self._graph.get_edge(src_id, tgt_id)

    async def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        nodes = await self._graph.get_all_nodes()
        edges = await self._graph.get_all_edges()

        return {
            "num_nodes": len(nodes),
            "num_edges": len(edges),
            "has_vector_index": self._vector_index is not None
        }

    # ========================================================================
    # Persistence
    # ========================================================================

    def save(self, file_path: str):
        """Save graph to file (pkl format)."""
        import pickle
        graph = self._graph._graph

        # Ensure .pkl extension for consistency with old format
        if not file_path.endswith('.pkl'):
            file_path = file_path.replace('.graphml', '.pkl')

        with open(file_path, 'wb') as f:
            pickle.dump(graph, f)

        logger.info(f"Graph saved to {file_path}")

    def load(self, file_path: str):
        """Load graph from file (pkl format)."""
        import pickle

        with open(file_path, 'rb') as f:
            graph = pickle.load(f)

        self._graph._graph = graph
        logger.info(f"Graph loaded from {file_path}")

    # ========================================================================
    # Utilities
    # ========================================================================

    def _get_default_config(self) -> Dict:
        """Default configuration."""
        return {
            'kg': {
                'chunk_size': 512,
                'chunk_overlap': 12,
                'graph': {
                    'similarity_threshold': 0.8,
                    'enable_augmentation': True,
                },
                'ppr': {
                    'alpha': 0.85,
                    'epsilon': 1e-4,
                    'max_iterations': 100,
                    'adaptive_cutoff': True
                }
            },
            'services': {
                'llm': {'provider': 'gpt'}
            },
            'paths': {
                'cache_root': './data/preprocessed'
            }
        }


# ========================================================================
# CLI Interface
# ========================================================================

async def process_document(
    kg: KnowledgeGraph,
    input_path: str,
    doc_index: int,
    dataset_type: str,
    output_dir: Path,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    similarity_threshold: float = 0.8,
    skip_build: bool = False,
    skip_augment: bool = False,
) -> None:
    """
    Process a single document through the complete KG pipeline

    Args:
        kg: KnowledgeGraph instance
        input_path: Path to input JSONL file
        doc_index: Document index
        dataset_type: Dataset type identifier
        output_dir: Output directory
        chunk_size: Token size for chunks
        chunk_overlap: Overlap tokens between chunks
        similarity_threshold: Similarity threshold for augmentation
        skip_build: Skip construction step
        skip_augment: Skip augmentation step
    """
    import pickle
    from src.kg.utils.file_operations import read_source_document

    doc_id = f"{dataset_type}_{doc_index}"

    logger.info(f"\n{'='*70}")
    logger.info(f"Processing document {doc_index}: {doc_id}")
    logger.info(f"{'='*70}")

    # ========================================================================
    # Step 1: Build KG (Construction)
    # ========================================================================
    if not skip_build:
        logger.info("\n[Step 1/3] Building Knowledge Graph...")

        # Check if already built
        kg_path = output_dir / "full_kg" / f"{doc_id}.pkl"
        if kg_path.exists():
            logger.info(f"KG already exists at {kg_path}, loading...")
            kg.load(str(kg_path))
        else:
            # Read document
            doc = read_source_document(input_path, doc_index)
            document_text = doc.get("context", "")

            if not document_text:
                logger.warning(f"Empty document at index {doc_index}, skipping")
                return

            # Build KG
            num_nodes, num_edges = await kg.build_from_document(
                document=document_text,
                doc_id=doc_id,
                chunk_size=chunk_size,
                overlap_size=chunk_overlap,
                use_cache=True,
                show_progress=True
            )

            # Save KG
            kg_path.parent.mkdir(parents=True, exist_ok=True)
            kg.save(str(kg_path))

            # Save entities data for augmentation
            entities_path = output_dir / "entities" / f"{doc_id}.json"
            entities_path.parent.mkdir(parents=True, exist_ok=True)
            if hasattr(kg, '_entities_data'):
                import json
                with open(entities_path, 'w', encoding='utf-8') as f:
                    json.dump(kg._entities_data, f, ensure_ascii=False, indent=2)

            logger.info(f"✓ Built KG: {num_nodes} nodes, {num_edges} edges")
            logger.info(f"✓ Saved to {kg_path}")

    # ========================================================================
    # Step 2: Augment KG (Add similarity edges)
    # ========================================================================
    if not skip_augment:
        logger.info("\n[Step 2/3] Augmenting Knowledge Graph...")

        # Check if already augmented
        aug_kg_path = output_dir / "full_kg_augmented" / f"{doc_id}.pkl"
        if aug_kg_path.exists():
            logger.info(f"Augmented KG already exists at {aug_kg_path}, loading...")
            kg.load(str(aug_kg_path))
        else:
            # Load entities data if not in memory
            if not hasattr(kg, '_entities_data'):
                import json
                entities_path = output_dir / "entities" / f"{doc_id}.json"
                if entities_path.exists():
                    with open(entities_path, 'r', encoding='utf-8') as f:
                        kg._entities_data = json.load(f)
                else:
                    logger.warning("Entities data not found, extracting from graph...")
                    # Extract from graph (fallback)
                    all_nodes = await kg.get_all_nodes()
                    entities_data = []
                    for node_id in all_nodes:
                        node_data = await kg.get_node(node_id)
                        entities_data.append({
                            "entity_name": node_id,
                            **node_data
                        })
                    kg._entities_data = entities_data

            # Augment
            edges_added = await kg.augment(
                threshold=similarity_threshold,
                dataset_name=dataset_type,
                entry_id=doc_id
            )

            # Save augmented KG
            aug_kg_path.parent.mkdir(parents=True, exist_ok=True)
            kg.save(str(aug_kg_path))

            logger.info(f"✓ Augmented KG: {edges_added} similarity edges added")
            logger.info(f"✓ Saved to {aug_kg_path}")

    # ========================================================================
    logger.info(f"\n{'='*70}")
    logger.info(f"✅ Completed processing document {doc_index}")
    logger.info(f"{'='*70}\n")


def main():
    import argparse
    import asyncio
    import time
    from src.kg.utils.token_tracker import print_token_summary, print_token_details

    parser = argparse.ArgumentParser(
        description="Generate Knowledge Graphs from documents",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Dataset arguments
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name (e.g., hotpotqa, biology, fin)")
    parser.add_argument("--start_index", type=int, default=0,
                        help="Start index for processing")
    parser.add_argument("--end_index", type=int, default=10,
                        help="End index for processing")

    # KG construction arguments
    parser.add_argument("--chunk_size", type=int, default=512,
                        help="Token size for chunks")
    parser.add_argument("--chunk_overlap", type=int, default=12,
                        help="Overlap tokens between chunks")

    # Augmentation arguments
    parser.add_argument("--similarity_threshold", type=float, default=0.8,
                        help="Similarity threshold for augmentation (0.0-1.0)")

    # I/O arguments
    parser.add_argument("--input_file", type=str,
                        help="Input JSONL file path (default: ./data/raw/{dataset}.jsonl)")
    parser.add_argument("--output_dir", type=str,
                        help="Output directory for all KG data (default: ./data/preprocessed/{dataset})")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config.yaml file")

    # Pipeline control
    parser.add_argument("--skip_build", action="store_true",
                        help="Skip KG construction step")
    parser.add_argument("--skip_augment", action="store_true",
                        help="Skip KG augmentation step")

    # Token tracking options
    parser.add_argument("--show_token_details", action="store_true",
                        help="Show detailed per-call token breakdown at the end")

    # Parallelism control
    parser.add_argument("--max_concurrent", type=int, default=1,
                        help="Max concurrent documents to process (default: 1, sequential)")

    args = parser.parse_args()

    async def run_pipeline():
        # Start timing
        start_time = time.time()

        # Determine input file
        if args.input_file:
            input_path = args.input_file
        else:
            input_path = f"./data/raw/{args.dataset}.jsonl"

        # Create output directory
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            output_dir = Path(f"./data/preprocessed/{args.dataset}")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load config
        config = None
        if args.config and Path(args.config).exists():
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)

        # Process each document
        logger.info(f"\n{'='*70}")
        logger.info(f"Processing {args.dataset} dataset")
        logger.info(f"Range: {args.start_index} to {args.end_index}")
        logger.info(f"Output: {output_dir}")
        logger.info(f"{'='*70}\n")

        # Process documents with concurrency control
        semaphore = asyncio.Semaphore(args.max_concurrent)

        async def process_with_semaphore(idx):
            async with semaphore:
                try:
                    # Create independent KG instance for each document
                    doc_kg = KnowledgeGraph(config=config, working_dir=str(output_dir))
                    await process_document(
                        kg=doc_kg,
                        input_path=input_path,
                        doc_index=idx,
                        dataset_type=args.dataset,
                        output_dir=output_dir,
                        chunk_size=args.chunk_size,
                        chunk_overlap=args.chunk_overlap,
                        similarity_threshold=args.similarity_threshold,
                        skip_build=args.skip_build,
                        skip_augment=args.skip_augment,
                    )
                except Exception as e:
                    logger.error(f"Error processing document {idx}: {e}")
                    import traceback
                    traceback.print_exc()

        # Run all documents with controlled concurrency
        await asyncio.gather(*[
            process_with_semaphore(idx)
            for idx in range(args.start_index, args.end_index)
        ])

        # End timing
        end_time = time.time()
        elapsed_time = end_time - start_time

        logger.info(f"\n{'='*70}")
        logger.info("🎉 KG Generation Pipeline Completed!")
        logger.info(f"{'='*70}")

        # Print execution time
        logger.info(f"\n{'='*70}")
        logger.info("⏱️  EXECUTION TIME")
        logger.info(f"{'='*70}")
        logger.info(f"Total Time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
        logger.info(f"Documents Processed: {args.end_index - args.start_index}")
        logger.info(f"Average Time per Document: {elapsed_time/(args.end_index - args.start_index):.2f} seconds")
        logger.info(f"{'='*70}\n")

        # Print token usage summary
        print_token_summary()

        # Print detailed breakdown if requested
        if args.show_token_details:
            print_token_details()

    asyncio.run(run_pipeline())


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    main()
