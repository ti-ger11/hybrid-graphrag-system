"""Qdrant vector store with dual dense + sparse (BM25) vector indexing.

Manages a Qdrant collection with two named vector spaces:
  - **dense**: Semantic embeddings from a sentence-transformer model.
  - **sparse**: BM25-style keyword vectors via FastEmbed's sparse model.

This dual-vector design enables the hybrid search pipeline to run
parallel dense and sparse searches in a single Qdrant `query_points`
call with Reciprocal Rank Fusion (RRF).

The store also maintains a parent-chunk lookup: child chunks are indexed
for retrieval precision, but their parent chunk text is stored in the
payload so the LLM receives maximum context.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence
import uuid

from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.http import models

from src.config import Settings
from src.ingestion.chunker import DocumentChunk

logger = logging.getLogger(__name__)


# ── Data Structures ──────────────────────────────────────────────────────


@dataclass
class RetrievedDocument:
    """A document retrieved from the vector store or knowledge graph."""

    text: str
    score: float
    metadata: dict

    @property
    def chunk_id(self) -> str | None:
        return self.metadata.get("chunk_id")

    @property
    def document_id(self) -> str | None:
        return self.metadata.get("document_id")

    @property
    def page_numbers(self) -> list[int]:
        return self.metadata.get("page_numbers", [])

    @property
    def source_label(self) -> str:
        """Human-readable source label for citations."""
        pages = self.page_numbers
        if pages:
            return f"Pages {min(pages)}–{max(pages)}"
        return "Unknown source"


# ── Vector Store ─────────────────────────────────────────────────────────


class QdrantVectorStore:
    """Qdrant collection with dense + sparse (BM25) dual-vector indexing.

    Attributes:
        client: Qdrant client instance.
        collection_name: Name of the Qdrant collection.
        dense_model: FastEmbed dense embedding model.
        sparse_model: FastEmbed sparse BM25 model.
    """

    def __init__(self, settings: Settings) -> None:
        try:
            self.client = QdrantClient(url=settings.qdrant_url, timeout=2.0)
            self.client.get_collections()
            logger.info("Connected to remote Qdrant at %s", settings.qdrant_url)
        except Exception as e:
            logger.warning(
                "Could not connect to remote Qdrant at %s (%s). Using embedded local Qdrant storage at 'data/qdrant_storage'.",
                settings.qdrant_url,
                e,
            )
            self.client = QdrantClient(path="data/qdrant_storage")

        self.collection_name = settings.qdrant_collection_name
        self._embedding_dim = settings.embedding_dimension

        logger.info("Initializing embedding models...")
        self.dense_model = TextEmbedding(settings.embedding_model_name)
        self.sparse_model = SparseTextEmbedding(settings.sparse_model_name)
        logger.info(
            "Models loaded: dense=%s, sparse=%s",
            settings.embedding_model_name,
            settings.sparse_model_name,
        )

    def ensure_collection(self) -> None:
        """Create the collection with named dense + sparse vector configs.

        Safe to call multiple times — checks if the collection already exists.
        """
        collections = self.client.get_collections().collections
        existing = [c.name for c in collections]

        if self.collection_name in existing:
            logger.info("Collection '%s' already exists", self.collection_name)
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                "dense": models.VectorParams(
                    size=self._embedding_dim,
                    distance=models.Distance.COSINE,
                )
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(
                    modifier=models.Modifier.IDF,
                )
            },
        )
        logger.info(
            "Created collection '%s' with dense(%d) + sparse vectors",
            self.collection_name,
            self._embedding_dim,
        )

    def upsert_chunks(
        self,
        child_chunks: Sequence[DocumentChunk],
        parent_chunks: Sequence[DocumentChunk] | None = None,
    ) -> None:
        """Generate both dense + sparse vectors and upsert child chunks.

        Each child chunk point stores:
          - Dense vector (semantic embedding)
          - Sparse vector (BM25 keyword embedding)
          - Payload: text, parent_text (for LLM context expansion),
            document_id, page_numbers, section_title, etc.

        Args:
            child_chunks: Child chunks to embed and index.
            parent_chunks: Optional parent chunks to build a parent-text
                          lookup for the "small-to-big" retrieval pattern.
        """
        if not child_chunks:
            logger.warning("No child chunks to upsert")
            return

        self.ensure_collection()

        # Build parent text lookup for context expansion
        parent_text_map: dict[str, str] = {}
        if parent_chunks:
            for pc in parent_chunks:
                parent_text_map[pc.chunk_id] = pc.text

        texts = [c.text for c in child_chunks]
        logger.info("Generating embeddings for %d child chunks...", len(texts))

        dense_embeddings = list(self.dense_model.embed(texts))
        sparse_embeddings = list(self.sparse_model.embed(texts))

        points: list[models.PointStruct] = []
        for chunk, dense_emb, sparse_emb in zip(
            child_chunks, dense_embeddings, sparse_embeddings
        ):
            # Ensure point ID is a valid RFC 4122 UUID for Qdrant
            try:
                point_id = str(uuid.UUID(str(chunk.chunk_id)))
            except ValueError:
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(chunk.chunk_id)))

            # Resolve parent text for context expansion
            parent_text = parent_text_map.get(chunk.parent_id, "") if chunk.parent_id else ""

            point = models.PointStruct(
                id=point_id,
                vector={
                    "dense": dense_emb.tolist(),
                    "sparse": models.SparseVector(
                        indices=sparse_emb.indices.tolist(),
                        values=sparse_emb.values.tolist(),
                    ),
                },
                payload={
                    "text": chunk.text,
                    "parent_text": parent_text,
                    "parent_id": chunk.parent_id,
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "page_numbers": chunk.page_numbers,
                    "section_title": chunk.metadata.get("section_title", ""),
                    "chunk_type": chunk.chunk_type,
                },
            )
            points.append(point)

        # Batch upsert (Qdrant handles batching internally for large sets)
        BATCH_SIZE = 100
        for i in range(0, len(points), BATCH_SIZE):
            batch = points[i : i + BATCH_SIZE]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch,
            )

        logger.info("Upserted %d points into '%s'", len(points), self.collection_name)

    def hybrid_search(
        self,
        query: str,
        limit: int = 20,
        score_threshold: float | None = None,
    ) -> list[RetrievedDocument]:
        """Execute hybrid dense + sparse search with RRF fusion.

        Uses Qdrant's native `query_points` API with parallel prefetches
        to run dense and sparse searches concurrently, then merges results
        via Reciprocal Rank Fusion.

        Args:
            query: Natural language search query.
            limit: Maximum number of fused results to return.
            score_threshold: Optional minimum RRF score cutoff.

        Returns:
            List of RetrievedDocument objects sorted by RRF score.
        """
        # Generate query vectors
        query_dense = list(self.dense_model.embed([query]))[0].tolist()
        query_sparse_raw = list(self.sparse_model.embed([query]))[0]

        response = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                models.Prefetch(
                    query=query_dense,
                    using="dense",
                    limit=limit,
                ),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=query_sparse_raw.indices.tolist(),
                        values=query_sparse_raw.values.tolist(),
                    ),
                    using="sparse",
                    limit=limit,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
        )

        results: list[RetrievedDocument] = []
        for hit in response.points:
            if score_threshold and hit.score < score_threshold:
                continue

            # Use parent_text for context expansion if available
            text = hit.payload.get("parent_text") or hit.payload.get("text", "")

            results.append(
                RetrievedDocument(
                    text=text,
                    score=hit.score,
                    metadata={
                        "chunk_id": hit.payload.get("chunk_id"),
                        "child_text": hit.payload.get("text", ""),
                        "document_id": hit.payload.get("document_id"),
                        "page_numbers": hit.payload.get("page_numbers", []),
                        "section_title": hit.payload.get("section_title", ""),
                        "parent_id": hit.payload.get("parent_id"),
                    },
                )
            )

        logger.debug("Hybrid search returned %d results for: %.50s...", len(results), query)
        return results

    def get_collection_info(self) -> dict[str, Any]:
        """Return collection statistics."""
        try:
            info = self.client.get_collection(self.collection_name)
            points_count = getattr(info, "points_count", 0)
            vectors_count = getattr(
                info, "vectors_count", getattr(info, "indexed_vectors_count", points_count)
            )
            status = getattr(info, "status", "green")
            return {
                "name": self.collection_name,
                "points_count": points_count,
                "vectors_count": vectors_count,
                "status": str(status),
            }
        except Exception as e:
            logger.error("Failed to get collection info: %s", e)
            return {"name": self.collection_name, "error": str(e)}
