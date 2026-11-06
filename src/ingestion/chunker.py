"""Parent-child hierarchical chunking strategy.

Implements a two-tier chunking approach:
  1. **Parent chunks** — Large contextual windows (~1500 chars) that preserve
     section-level coherence. These are stored in Qdrant's payload and returned
     to the LLM for maximum context.
  2. **Child chunks** — Smaller retrieval units (~400 chars) derived from each
     parent. These are embedded and indexed for high-precision vector search.

Each child carries a `parent_id` reference, enabling a "small-to-big" retrieval
pattern: search over precise children, but expand to the richer parent context
before passing to the LLM.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Sequence
import uuid

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.ingestion.parser import ParsedElement

logger = logging.getLogger(__name__)


# ── Data Structures ──────────────────────────────────────────────────────


@dataclass
class DocumentChunk:
    """A chunk with parent-child lineage tracking.

    Attributes:
        chunk_id: Globally unique identifier for this chunk.
        text: The textual content of the chunk.
        parent_id: ID of the parent chunk (None if this IS the parent).
        document_id: ID of the source document.
        page_numbers: List of pages this chunk spans.
        chunk_type: Either "parent" or "child".
        element_types: Original structural element types that compose this chunk.
        metadata: Additional metadata (source file, section title, etc.).
    """

    chunk_id: str
    text: str
    parent_id: str | None
    document_id: str
    page_numbers: list[int]
    chunk_type: str  # "parent" or "child"
    element_types: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


# ── Chunker ──────────────────────────────────────────────────────────────


class ParentChildChunker:
    """Creates hierarchical parent-child chunks from parsed document elements.

    The chunking respects structural boundaries: section titles trigger new
    parent chunks, and tables are kept whole whenever possible. This prevents
    the common failure mode of naive character-based splitting that breaks
    mid-sentence or mid-table.

    Usage:
        chunker = ParentChildChunker(parent_chunk_size=1500, child_chunk_size=400)
        chunks = chunker.chunk(parsed_elements, document_id="doc-123")
    """

    # Element types that signal a new section boundary
    SECTION_BREAK_TYPES = {"Title", "Header"}

    def __init__(
        self,
        parent_chunk_size: int = 1500,
        child_chunk_size: int = 400,
        chunk_overlap: int = 100,
    ) -> None:
        if child_chunk_size >= parent_chunk_size:
            raise ValueError(
                f"child_chunk_size ({child_chunk_size}) must be smaller than "
                f"parent_chunk_size ({parent_chunk_size})"
            )

        self.parent_chunk_size = parent_chunk_size
        self.child_chunk_size = child_chunk_size
        self.chunk_overlap = chunk_overlap

        self._child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    def chunk(
        self,
        elements: Sequence[ParsedElement],
        document_id: str,
    ) -> list[DocumentChunk]:
        """Split parsed elements into parent-child chunks.

        Args:
            elements: Ordered list of parsed document elements.
            document_id: Identifier for the source document.

        Returns:
            List of DocumentChunk objects (parents + children interleaved).
        """
        if not elements:
            logger.warning("No elements to chunk for document %s", document_id)
            return []

        # Step 1: Group elements into section-aware parent blocks
        parent_blocks = self._build_parent_blocks(elements)

        # Step 2: Create parent chunks, then split each into children
        all_chunks: list[DocumentChunk] = []

        for block in parent_blocks:
            parent_text = "\n\n".join(el.text for el in block)
            parent_pages = sorted(
                {el.page_number for el in block if el.page_number is not None}
            )
            element_types = [el.element_type for el in block]

            # Generate a deterministic parent ID for idempotent re-ingestion
            parent_id = self._deterministic_id(document_id, parent_text)

            parent_chunk = DocumentChunk(
                chunk_id=parent_id,
                text=parent_text,
                parent_id=None,
                document_id=document_id,
                page_numbers=parent_pages,
                chunk_type="parent",
                element_types=element_types,
                metadata={"section_title": self._find_section_title(block)},
            )
            all_chunks.append(parent_chunk)

            # Split parent text into children
            child_texts = self._child_splitter.split_text(parent_text)
            for child_text in child_texts:
                child_id = self._deterministic_id(parent_id, child_text)
                child_chunk = DocumentChunk(
                    chunk_id=child_id,
                    text=child_text,
                    parent_id=parent_id,
                    document_id=document_id,
                    page_numbers=parent_pages,
                    chunk_type="child",
                    element_types=element_types,
                    metadata={"section_title": parent_chunk.metadata.get("section_title")},
                )
                all_chunks.append(child_chunk)

        parent_count = sum(1 for c in all_chunks if c.chunk_type == "parent")
        child_count = sum(1 for c in all_chunks if c.chunk_type == "child")
        logger.info(
            "Document %s: created %d parent chunks and %d child chunks",
            document_id,
            parent_count,
            child_count,
        )
        return all_chunks

    # ── Private Helpers ──────────────────────────────────────────────────

    def _build_parent_blocks(
        self, elements: Sequence[ParsedElement]
    ) -> list[list[ParsedElement]]:
        """Group elements into blocks respecting section boundaries and size.

        Rules:
          - A Title/Header element starts a new block.
          - Blocks are split when accumulated text exceeds `parent_chunk_size`.
          - Table elements are kept in their own block if they're large.
        """
        blocks: list[list[ParsedElement]] = []
        current_block: list[ParsedElement] = []
        current_size = 0

        for el in elements:
            el_size = len(el.text)

            # Start a new block on section boundary
            is_section_break = el.element_type in self.SECTION_BREAK_TYPES
            would_overflow = (current_size + el_size) > self.parent_chunk_size

            if current_block and (is_section_break or would_overflow):
                blocks.append(current_block)
                current_block = []
                current_size = 0

            # Large tables get their own block
            if el.element_type == "Table" and el_size > self.parent_chunk_size * 0.5:
                if current_block:
                    blocks.append(current_block)
                    current_block = []
                    current_size = 0
                blocks.append([el])
                continue

            current_block.append(el)
            current_size += el_size

        if current_block:
            blocks.append(current_block)

        return blocks

    @staticmethod
    def _find_section_title(block: list[ParsedElement]) -> str | None:
        """Extract the first Title/Header element text from a block."""
        for el in block:
            if el.element_type in ("Title", "Header"):
                return el.text
        return None

    @staticmethod
    def _deterministic_id(namespace: str, content: str) -> str:
        """Generate a deterministic UUID from namespace + content.

        This ensures idempotent re-ingestion: the same content always
        produces the same valid UUID chunk ID, allowing safe upserts in Qdrant.
        """
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{namespace}::{content}"))
