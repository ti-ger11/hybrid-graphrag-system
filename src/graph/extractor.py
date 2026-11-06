"""LLM-powered entity and relationship extraction from document chunks.

Uses LangChain's LLMGraphTransformer to convert unstructured text into
structured graph documents. Entity types are discovered dynamically by
the LLM and then normalized to canonical labels via post-hoc normalization.

This open-ended approach is ideal for general-purpose document Q&A where
the entity schema isn't known in advance.
"""

from __future__ import annotations

import logging
from typing import Sequence

from langchain_core.documents import Document
from langchain_neo4j import LLMGraphTransformer
from langchain_neo4j.graphs.graph_document import GraphDocument
from src.config import Settings
from src.graph.schemas import normalize_entity_type, normalize_relationship_type
from src.ingestion.chunker import DocumentChunk

logger = logging.getLogger(__name__)


class GraphExtractor:
    """Extracts entities and relations from text using LLM-powered graph transformation.

    The extractor operates in open-ended mode (no allowed_nodes / allowed_relationships
    constraints), letting the LLM discover entity types dynamically from the text.
    After extraction, types are normalized to canonical labels using the mappings
    in `src.graph.schemas`.

    Attributes:
        llm: The language model used for structured extraction.
        transformer: LangChain's LLMGraphTransformer configured for open-ended extraction.
    """

    def __init__(self, settings: Settings) -> None:
        if settings.llm_provider == "google":
            from langchain_google_genai import ChatGoogleGenerativeAI
            self.llm = ChatGoogleGenerativeAI(
                model=settings.google_model,
                temperature=0,
                google_api_key=settings.google_api_key,
            )
        else:
            from langchain_openai import ChatOpenAI
            self.llm = ChatOpenAI(
                model=settings.openai_model,
                temperature=0,
                api_key=settings.openai_api_key,
            )
        self.transformer = LLMGraphTransformer(
            llm=self.llm,
            # Open-ended: no allowed_nodes or allowed_relationships
            # The LLM discovers entity types from the document text
            node_properties=True,
            relationship_properties=True,
        )

    def extract(self, chunks: Sequence[DocumentChunk]) -> list[GraphDocument]:
        """Extract entities and relationships from document chunks.

        Each chunk is converted to a LangChain Document and processed by the
        LLMGraphTransformer. The resulting graph documents have their entity
        and relationship types post-hoc normalized.

        Args:
            chunks: Parent chunks to extract entities from. Parent chunks
                    are preferred over children for richer context.

        Returns:
            List of GraphDocument objects with normalized types, ready for
            storage in Neo4j.
        """
        if not chunks:
            logger.warning("No chunks provided for entity extraction")
            return []

        # Convert chunks to LangChain Documents
        documents = [
            Document(
                page_content=chunk.text,
                metadata={
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "page_numbers": chunk.page_numbers,
                    "section_title": chunk.metadata.get("section_title", ""),
                },
            )
            for chunk in chunks
        ]

        logger.info("Extracting entities from %d chunks...", len(documents))

        try:
            graph_documents = self.transformer.convert_to_graph_documents(documents)
        except Exception as e:
            logger.error("Graph extraction failed: %s", e)
            raise

        # Post-hoc normalization of entity and relationship types
        normalized_docs = [
            self._normalize_graph_document(gd) for gd in graph_documents
        ]

        total_nodes = sum(len(gd.nodes) for gd in normalized_docs)
        total_rels = sum(len(gd.relationships) for gd in normalized_docs)
        logger.info(
            "Extracted and normalized %d entities and %d relationships",
            total_nodes,
            total_rels,
        )

        return normalized_docs

    @staticmethod
    def _normalize_graph_document(graph_doc: GraphDocument) -> GraphDocument:
        """Apply post-hoc type normalization to a GraphDocument.

        Normalizes node types and relationship types to canonical labels,
        ensuring consistent naming across documents and LLM runs.
        """
        # Normalize node types
        for node in graph_doc.nodes:
            if hasattr(node, "type") and node.type:
                node.type = normalize_entity_type(node.type)

        # Normalize relationship types
        for rel in graph_doc.relationships:
            if hasattr(rel, "type") and rel.type:
                rel.type = normalize_relationship_type(rel.type)

        return graph_doc
