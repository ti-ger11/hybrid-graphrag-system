"""Three-way hybrid search pipeline: Dense + BM25 (via Qdrant RRF) + Knowledge Graph.

Combines three retrieval strategies:
  1. **Dense vector search** — Semantic similarity via sentence embeddings.
  2. **BM25 sparse search** — Exact keyword/token matching.
  3. **Knowledge graph traversal** — Structured entity-relationship context.

Steps 1 and 2 are fused inside Qdrant via Reciprocal Rank Fusion (RRF).
The fused results are then reranked by a cross-encoder for maximum precision.
Knowledge graph context is appended as supplementary structured information.
"""

from __future__ import annotations

import logging
from typing import Sequence

from src.graph.store import Neo4jGraphStore
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.vector_store import QdrantVectorStore, RetrievedDocument

logger = logging.getLogger(__name__)


class HybridSearchPipeline:
    """Three-way hybrid search with RRF fusion and cross-encoder reranking.

    Pipeline flow:
        1. Qdrant hybrid search (dense + BM25 → RRF fusion) → top-N candidates
        2. Cross-encoder reranking → top-K refined results
        3. Knowledge graph context augmentation (entity neighbors/relations)

    The pipeline returns `RetrievedDocument` objects that include both the
    text content and rich metadata (page numbers, section titles, etc.)
    needed for citation generation downstream.

    Usage:
        pipeline = HybridSearchPipeline(vector_store, graph_store, reranker)
        results = pipeline.search("What is the company's revenue?", entities=["Company X"])
    """

    def __init__(
        self,
        vector_store: QdrantVectorStore,
        graph_store: Neo4jGraphStore,
        reranker: CrossEncoderReranker,
    ) -> None:
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.reranker = reranker

    def search(
        self,
        query: str,
        entities: list[str] | None = None,
        initial_top_k: int = 20,
        rerank_top_n: int | None = None,
    ) -> list[RetrievedDocument]:
        """Execute the full hybrid search + rerank + KG augmentation pipeline.

        Args:
            query: The user's natural language query.
            entities: Entity names extracted from the query for KG traversal.
            initial_top_k: Number of candidates to retrieve from Qdrant before reranking.
            rerank_top_n: Override the cross-encoder's default top-N.

        Returns:
            Ordered list of RetrievedDocument objects (most relevant first),
            optionally augmented with knowledge graph context.
        """
        entities = entities or []

        logger.info(
            "Hybrid search: query=%.60s..., entities=%s, top_k=%d",
            query,
            entities[:5],
            initial_top_k,
        )

        # ── Step 1: Hybrid dense + sparse search via Qdrant RRF ─────────
        try:
            rrf_candidates = self.vector_store.hybrid_search(
                query=query,
                limit=initial_top_k,
            )
        except Exception as e:
            logger.error("Hybrid search failed: %s", e)
            rrf_candidates = []

        if not rrf_candidates:
            logger.warning("No candidates from hybrid search")
            # Still try KG context as fallback
            kg_docs = self._get_kg_documents(entities)
            return kg_docs

        logger.info("RRF fusion returned %d candidates", len(rrf_candidates))

        # ── Step 2: Cross-encoder reranking ──────────────────────────────
        try:
            reranked = self.reranker.rerank(
                query=query,
                documents=rrf_candidates,
                top_n=rerank_top_n,
            )
        except Exception as e:
            logger.error("Reranking failed, using RRF order: %s", e)
            reranked = list(rrf_candidates[: (rerank_top_n or 5)])

        logger.info("Cross-encoder reranking kept %d documents", len(reranked))

        # ── Step 3: Augment with Knowledge Graph context ─────────────────
        kg_docs = self._get_kg_documents(entities)

        # Combine: reranked vector results + KG context
        final_results = reranked + kg_docs

        logger.info(
            "Final results: %d vector docs + %d KG docs",
            len(reranked),
            len(kg_docs),
        )
        return final_results

    def _get_kg_documents(self, entities: list[str]) -> list[RetrievedDocument]:
        """Retrieve knowledge graph context as RetrievedDocument objects.

        Args:
            entities: Entity names to look up in the graph.

        Returns:
            List containing a single RetrievedDocument with KG context,
            or an empty list if no context is found.
        """
        if not entities:
            return []

        try:
            kg_context = self.graph_store.query_related_context(entities)
        except Exception as e:
            logger.error("KG context retrieval failed: %s", e)
            return []

        if not kg_context:
            return []

        return [
            RetrievedDocument(
                text=kg_context,
                score=1.0,
                metadata={
                    "source": "knowledge_graph",
                    "entities_queried": entities,
                    "page_numbers": [],
                },
            )
        ]

    def deduplicate(
        self, documents: Sequence[RetrievedDocument]
    ) -> list[RetrievedDocument]:
        """Remove near-duplicate documents based on chunk_id.

        If multiple retrieval paths return the same chunk, keep the one
        with the highest score.

        Args:
            documents: List of potentially duplicated documents.

        Returns:
            Deduplicated list preserving the highest-scoring version of each chunk.
        """
        seen: dict[str, RetrievedDocument] = {}
        for doc in documents:
            cid = doc.chunk_id or id(doc)
            if cid not in seen or doc.score > seen[cid].score:
                seen[cid] = doc

        deduped = sorted(seen.values(), key=lambda d: d.score, reverse=True)
        if len(deduped) < len(documents):
            logger.debug(
                "Deduplicated %d → %d documents",
                len(documents),
                len(deduped),
            )
        return deduped
