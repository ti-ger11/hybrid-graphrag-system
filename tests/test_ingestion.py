"""Unit tests for the ingestion pipeline: parser and chunker.

These tests validate parsing and chunking logic without requiring
external services (Qdrant, Neo4j) or actual PDF files.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.ingestion.chunker import DocumentChunk, ParentChildChunker
from src.ingestion.parser import ParsedElement, UnstructuredPDFParser


# ── ParsedElement Tests ──────────────────────────────────────────────────


class TestParsedElement:
    """Tests for the ParsedElement data class."""

    def test_creation(self):
        el = ParsedElement(
            text="Hello World",
            element_type="NarrativeText",
            page_number=1,
            metadata={"key": "value"},
        )
        assert el.text == "Hello World"
        assert el.element_type == "NarrativeText"
        assert el.page_number == 1
        assert el.metadata == {"key": "value"}

    def test_defaults(self):
        el = ParsedElement(text="Test", element_type="Title")
        assert el.page_number is None
        assert el.metadata == {}


# ── UnstructuredPDFParser Tests ──────────────────────────────────────────


class TestUnstructuredPDFParser:
    """Tests for the UnstructuredPDFParser (error handling only — no real PDFs)."""

    def test_file_not_found(self):
        parser = UnstructuredPDFParser()
        with pytest.raises(FileNotFoundError, match="PDF file not found"):
            parser.parse(Path("/nonexistent/file.pdf"))

    def test_invalid_file_extension(self, tmp_path: Path):
        # Create a non-PDF file
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("Hello")

        parser = UnstructuredPDFParser()
        with pytest.raises(ValueError, match="Expected a .pdf file"):
            parser.parse(txt_file)


# ── ParentChildChunker Tests ─────────────────────────────────────────────


class TestParentChildChunker:
    """Tests for the ParentChildChunker."""

    @pytest.fixture
    def chunker(self) -> ParentChildChunker:
        return ParentChildChunker(
            parent_chunk_size=500,
            child_chunk_size=150,
            chunk_overlap=20,
        )

    @pytest.fixture
    def sample_elements(self) -> list[ParsedElement]:
        """Create a set of realistic parsed elements."""
        return [
            ParsedElement(text="Introduction", element_type="Title", page_number=1),
            ParsedElement(
                text="This is the first paragraph of the introduction. It provides "
                "an overview of the document and sets the context for the reader.",
                element_type="NarrativeText",
                page_number=1,
            ),
            ParsedElement(
                text="The second paragraph continues with more details about the "
                "subject matter and expands on the key themes.",
                element_type="NarrativeText",
                page_number=1,
            ),
            ParsedElement(text="Methodology", element_type="Title", page_number=2),
            ParsedElement(
                text="We employed a mixed-methods approach combining quantitative "
                "surveys with qualitative interviews. The sample included 500 "
                "participants from diverse backgrounds.",
                element_type="NarrativeText",
                page_number=2,
            ),
            ParsedElement(
                text="| Metric | Value |\n| --- | --- |\n| Sample Size | 500 |",
                element_type="Table",
                page_number=2,
                metadata={"html": "<table>...</table>"},
            ),
        ]

    def test_invalid_chunk_sizes(self):
        """Child chunk size must be smaller than parent chunk size."""
        with pytest.raises(ValueError, match="child_chunk_size.*must be smaller"):
            ParentChildChunker(parent_chunk_size=100, child_chunk_size=200)

    def test_empty_elements(self, chunker: ParentChildChunker):
        """Chunking empty elements should return empty list."""
        result = chunker.chunk([], "doc-1")
        assert result == []

    def test_chunk_types(self, chunker: ParentChildChunker, sample_elements):
        """All chunks should be either 'parent' or 'child'."""
        chunks = chunker.chunk(sample_elements, "doc-1")
        for chunk in chunks:
            assert chunk.chunk_type in ("parent", "child")

    def test_parent_child_relationship(self, chunker: ParentChildChunker, sample_elements):
        """Every child chunk should have a valid parent_id."""
        chunks = chunker.chunk(sample_elements, "doc-1")
        parent_ids = {c.chunk_id for c in chunks if c.chunk_type == "parent"}
        child_chunks = [c for c in chunks if c.chunk_type == "child"]

        for child in child_chunks:
            assert child.parent_id is not None, "Child must have a parent_id"
            assert child.parent_id in parent_ids, (
                f"Child's parent_id '{child.parent_id}' not found in parent chunks"
            )

    def test_parents_have_no_parent_id(self, chunker: ParentChildChunker, sample_elements):
        """Parent chunks should have parent_id=None."""
        chunks = chunker.chunk(sample_elements, "doc-1")
        parents = [c for c in chunks if c.chunk_type == "parent"]
        for parent in parents:
            assert parent.parent_id is None

    def test_section_boundary_creates_new_parent(
        self, chunker: ParentChildChunker, sample_elements
    ):
        """Title elements should trigger new parent chunks."""
        chunks = chunker.chunk(sample_elements, "doc-1")
        parents = [c for c in chunks if c.chunk_type == "parent"]

        # We have two Title elements, so at least 2 parent chunks
        assert len(parents) >= 2

    def test_document_id_propagated(self, chunker: ParentChildChunker, sample_elements):
        """All chunks should carry the correct document_id."""
        doc_id = "test-doc-123"
        chunks = chunker.chunk(sample_elements, doc_id)
        for chunk in chunks:
            assert chunk.document_id == doc_id

    def test_deterministic_ids(self, chunker: ParentChildChunker, sample_elements):
        """Same input should produce same chunk IDs (idempotent)."""
        chunks_1 = chunker.chunk(sample_elements, "doc-1")
        chunks_2 = chunker.chunk(sample_elements, "doc-1")

        ids_1 = [c.chunk_id for c in chunks_1]
        ids_2 = [c.chunk_id for c in chunks_2]
        assert ids_1 == ids_2

    def test_unique_chunk_ids(self, chunker: ParentChildChunker, sample_elements):
        """All chunk IDs within a document should be unique."""
        chunks = chunker.chunk(sample_elements, "doc-1")
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Duplicate chunk IDs found"

    def test_child_text_within_parent(self, chunker: ParentChildChunker, sample_elements):
        """Each child's text should be a substring of its parent's text (minus overlap)."""
        chunks = chunker.chunk(sample_elements, "doc-1")
        parent_map = {c.chunk_id: c for c in chunks if c.chunk_type == "parent"}

        for child in chunks:
            if child.chunk_type == "child" and child.parent_id:
                parent = parent_map[child.parent_id]
                # Due to splitting, each child's core text should appear in the parent
                # Use first 50 chars as a substring check (overlap may cause edge cases)
                core_text = child.text[:50]
                assert core_text in parent.text, (
                    f"Child text not found in parent. "
                    f"Child: {core_text!r}, Parent starts: {parent.text[:100]!r}"
                )

    def test_page_numbers_propagated(self, chunker: ParentChildChunker, sample_elements):
        """Chunks should carry page number information."""
        chunks = chunker.chunk(sample_elements, "doc-1")
        # At least some chunks should have page numbers
        chunks_with_pages = [c for c in chunks if c.page_numbers]
        assert len(chunks_with_pages) > 0

    def test_section_title_in_metadata(self, chunker: ParentChildChunker, sample_elements):
        """Parent chunks starting with a Title should have section_title in metadata."""
        chunks = chunker.chunk(sample_elements, "doc-1")
        parents_with_title = [
            c
            for c in chunks
            if c.chunk_type == "parent" and c.metadata.get("section_title")
        ]
        assert len(parents_with_title) >= 1
