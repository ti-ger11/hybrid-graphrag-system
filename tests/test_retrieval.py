"""Unit tests for the retrieval pipeline: vector store, reranker, and hybrid search.

These tests validate retrieval data structures, RRF integration, reranker sorting,
and hybrid search augmentation without requiring live Qdrant or Neo4j instances.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.config import Settings
from src.retrieval.hybrid_search import HybridSearchPipeline
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.vector_store import RetrievedDocument


# ── RetrievedDocument Tests ──────────────────────────────────────────────


class TestRetrievedDocument:
    """Tests for the RetrievedDocument data class and properties."""

    def test_properties_with_full_metadata(self):
        doc = RetrievedDocument(
            text="Revenue increased by 15% in Q3.",
            score=0.89,
            metadata={
                "chunk_id": "chunk-101",
                "document_id": "doc-annual-report",
                "parent_chunk_id": "parent-1",
                "page_numbers": [3, 4],
                "section_title": "Financial Highlights",
            },
        )
        assert doc.text == "Revenue increased by 15% in Q3."
        assert doc.score == 0.89
        assert doc.chunk_id == "chunk-101"
        assert doc.document_id == "doc-annual-report"
        assert doc.page_numbers == [3, 4]
        assert doc.source_label == "Pages 3–4"

    def test_single_page_label(self):
        doc = RetrievedDocument(
            text="Summary text",
            score=0.75,
            metadata={"page_numbers": [5]},
        )
        assert doc.page_numbers == [5]
        assert doc.source_label == "Pages 5–5"

    def test_missing_metadata_defaults(self):
        doc = RetrievedDocument(
            text="Standalone content",
            score=0.5,
            metadata={},
        )
        assert doc.chunk_id is None
        assert doc.document_id is None
        assert doc.page_numbers == []
        assert doc.source_label == "Unknown source"


# ── CrossEncoderReranker Tests ────────────────────────────────────────────


class TestCrossEncoderReranker:
    """Tests for the CrossEncoderReranker scoring and sorting logic."""

    @patch("src.retrieval.reranker.CrossEncoder")
    def test_reranker_reordering(self, mock_hf_encoder):
        # Mock cross-encoder predict: returns scores [0.2, 0.9, 0.5]
        mock_model_instance = MagicMock()
        mock_model_instance.predict.return_value = [0.2, 0.9, 0.5]
        mock_model_instance.score.return_value = [0.2, 0.9, 0.5]
        mock_hf_encoder.return_value = mock_model_instance

        mock_settings = MagicMock(spec=Settings)
        mock_settings.reranker_model_name = "test-reranker"
        mock_settings.reranker_top_n = 2

        reranker = CrossEncoderReranker(mock_settings)

        docs = [
            RetrievedDocument(text="Low relevance", score=0.99, metadata={"id": 1}),
            RetrievedDocument(text="High relevance", score=0.10, metadata={"id": 2}),
            RetrievedDocument(text="Medium relevance", score=0.50, metadata={"id": 3}),
        ]

        reranked = reranker.rerank(query="What is high relevance?", documents=docs, top_n=2)

        # Expected order: id 2 (0.9), id 3 (0.5). Top_n=2 so id 1 (0.2) is cut off.
        assert len(reranked) == 2
        assert reranked[0].metadata["id"] == 2
        assert reranked[0].score == 0.9
        assert reranked[1].metadata["id"] == 3
        assert reranked[1].score == 0.5

    @patch("src.retrieval.reranker.CrossEncoder")
    def test_empty_documents_handling(self, mock_hf_encoder):
        mock_settings = MagicMock(spec=Settings)
        mock_settings.reranker_model_name = "test-reranker"
        mock_settings.reranker_top_n = 5
        reranker = CrossEncoderReranker(mock_settings)

        result = reranker.rerank("Query", [])
        assert result == []


# ── HybridSearchPipeline Tests ────────────────────────────────────────────


class TestHybridSearchPipeline:
    """Tests for the full hybrid search orchestration."""

    def test_hybrid_search_end_to_end_flow(self):
        mock_vector_store = MagicMock()
        mock_graph_store = MagicMock()
        mock_reranker = MagicMock()

        # Mock vector store returning initial candidate docs
        initial_docs = [
            RetrievedDocument(text="Chunk A", score=0.8, metadata={"chunk_id": "a"}),
            RetrievedDocument(text="Chunk B", score=0.7, metadata={"chunk_id": "b"}),
        ]
        mock_vector_store.hybrid_search.return_value = initial_docs

        # Mock reranker returning re-scored docs
        reranked_docs = [
            RetrievedDocument(text="Chunk B", score=0.95, metadata={"chunk_id": "b"}),
            RetrievedDocument(text="Chunk A", score=0.60, metadata={"chunk_id": "a"}),
        ]
        mock_reranker.rerank.return_value = reranked_docs

        # Mock graph store returning context
        mock_graph_store.query_related_context.return_value = (
            "Entity: Acme Corp (type: Organization). Related to: Project Titan via OWNS."
        )

        pipeline = HybridSearchPipeline(
            vector_store=mock_vector_store,
            graph_store=mock_graph_store,
            reranker=mock_reranker,
        )

        results = pipeline.search(
            query="What is Project Titan?",
            entities=["Acme Corp"],
            initial_top_k=10,
            rerank_top_n=2,
        )

        # Vector store called with initial_top_k
        mock_vector_store.hybrid_search.assert_called_once_with(
            query="What is Project Titan?",
            limit=10,
        )

        # Reranker called with returned candidates
        mock_reranker.rerank.assert_called_once_with(
            query="What is Project Titan?",
            documents=initial_docs,
            top_n=2,
        )

        # Graph store queried for entity context
        mock_graph_store.query_related_context.assert_called_once_with(["Acme Corp"])

        # Result should include reranked docs + graph context doc
        assert len(results) == 3
        assert results[0].metadata["chunk_id"] == "b"
        assert results[1].metadata["chunk_id"] == "a"
        assert results[2].metadata["source"] == "knowledge_graph"
        assert "Acme Corp" in results[2].text
