"""End-to-end ingestion pipeline: PDF → Parse → Chunk → Embed → Store.

Orchestrates the full document processing flow:
  1. Parse the PDF into layout-aware structural elements.
  2. Create parent-child chunks respecting section boundaries.
  3. Generate dense + sparse embeddings for child chunks and upsert into Qdrant.
  4. Extract entities/relations from parent chunks and store in Neo4j.

The pipeline is designed to be idempotent: re-ingesting the same document
produces the same chunk IDs and safely upserts over existing data.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from src.config import Settings, get_settings
from src.ingestion.chunker import DocumentChunk, ParentChildChunker
from src.ingestion.parser import DocumentParser, ParsedElement

logger = logging.getLogger(__name__)


# ── Result Type ──────────────────────────────────────────────────────────


@dataclass
class IngestionResult:
    """Summary of a completed ingestion run."""

    document_id: str
    file_name: str
    num_elements: int
    num_parent_chunks: int
    num_child_chunks: int
    num_entities: int
    num_relations: int
    errors: list[str]

    @property
    def success(self) -> bool:
        return len(self.errors) == 0


# ── Pipeline ─────────────────────────────────────────────────────────────


class IngestionPipeline:
    """End-to-end PDF → Vector Store + Knowledge Graph pipeline.

    Dependencies are injected through the constructor, making it easy to
    swap implementations (e.g. different parsers, vector stores) and test
    individual components in isolation.

    Usage:
        pipeline = IngestionPipeline(
            parser=UnstructuredPDFParser(),
            chunker=ParentChildChunker(),
            vector_store=qdrant_store,
            graph_extractor=graph_extractor,
            graph_store=neo4j_store,
        )
        result = await pipeline.ingest(Path("report.pdf"))
    """

    def __init__(
        self,
        parser: DocumentParser,
        chunker: ParentChildChunker,
        vector_store,  # QdrantVectorStore — imported at runtime to avoid circular deps
        graph_extractor,  # GraphExtractor
        graph_store,  # Neo4jGraphStore
        data_dir: Path | None = None,
    ) -> None:
        self.parser = parser
        self.chunker = chunker
        self.vector_store = vector_store
        self.graph_extractor = graph_extractor
        self.graph_store = graph_store
        self.data_dir = data_dir or Path("data")
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def ingest(
        self,
        file_path: Path,
        document_id: str | None = None,
    ) -> IngestionResult:
        """Ingest a PDF document through the full pipeline (synchronous).

        Args:
            file_path: Path to the PDF file to ingest.
            document_id: Optional document identifier. If not provided,
                         a UUID is generated.

        Returns:
            IngestionResult with statistics about the ingestion.
        """
        document_id = document_id or uuid4().hex[:12]
        file_name = file_path.name
        errors: list[str] = []

        logger.info("Starting ingestion: %s (id=%s)", file_name, document_id)

        # ── Step 1: Copy file to data directory ──────────────────────────
        stored_path = self.data_dir / f"{document_id}_{file_name}"
        if not stored_path.exists():
            shutil.copy2(file_path, stored_path)
            logger.info("Stored document to: %s", stored_path)

        # ── Step 2: Parse PDF ────────────────────────────────────────────
        try:
            elements = self.parser.parse(stored_path)
        except Exception as e:
            logger.error("Parsing failed for %s: %s", file_name, e)
            return IngestionResult(
                document_id=document_id,
                file_name=file_name,
                num_elements=0,
                num_parent_chunks=0,
                num_child_chunks=0,
                num_entities=0,
                num_relations=0,
                errors=[f"Parsing failed: {e}"],
            )

        # ── Step 3: Create parent-child chunks ───────────────────────────
        all_chunks = self.chunker.chunk(elements, document_id)
        parent_chunks = [c for c in all_chunks if c.chunk_type == "parent"]
        child_chunks = [c for c in all_chunks if c.chunk_type == "child"]

        logger.info(
            "Chunking complete: %d parents, %d children",
            len(parent_chunks),
            len(child_chunks),
        )

        # ── Step 4: Embed and store child chunks in Qdrant ───────────────
        num_entities = 0
        num_relations = 0

        try:
            self.vector_store.upsert_chunks(child_chunks, parent_chunks)
            logger.info("Upserted %d child chunks into Qdrant", len(child_chunks))
        except Exception as e:
            err = f"Vector store upsert failed: {e}"
            logger.error(err)
            errors.append(err)

        # ── Step 5: Extract KG entities and store in Neo4j (if available) ─
        if self.graph_extractor and self.graph_store:
            try:
                graph_docs = self.graph_extractor.extract(parent_chunks)
                self.graph_store.add_graph_documents(graph_docs)

                # Count entities and relations for reporting
                for gd in graph_docs:
                    num_entities += len(gd.nodes)
                    num_relations += len(gd.relationships)

                logger.info(
                    "Stored %d entities and %d relations in Neo4j",
                    num_entities,
                    num_relations,
                )
            except Exception as e:
                err = f"Knowledge graph extraction skipped/failed: {e}"
                logger.warning(err)
        else:
            logger.info("Neo4j graph store not connected; continuing with vector indexing only.")

        result = IngestionResult(
            document_id=document_id,
            file_name=file_name,
            num_elements=len(elements),
            num_parent_chunks=len(parent_chunks),
            num_child_chunks=len(child_chunks),
            num_entities=num_entities,
            num_relations=num_relations,
            errors=errors,
        )

        if result.success:
            logger.info("Ingestion complete: %s", result)
        else:
            logger.warning("Ingestion completed with errors: %s", result.errors)

        return result

    async def aingest(
        self,
        file_path: Path,
        document_id: str | None = None,
    ) -> IngestionResult:
        """Asynchronous wrapper for ingest."""
        return self.ingest(file_path, document_id)
