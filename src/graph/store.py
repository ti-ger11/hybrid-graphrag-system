"""Neo4j graph database storage and querying.

Manages the lifecycle of the Neo4j connection, storing graph documents
produced by the GraphExtractor, and executing structured Cypher queries
for knowledge-graph-based retrieval.

The graph store links extracted entities back to their source document
chunks (via `include_source=True`), enabling provenance tracking and
citation generation.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_neo4j import Neo4jGraph
from langchain_neo4j.graphs.graph_document import GraphDocument

from src.config import Settings

logger = logging.getLogger(__name__)


class Neo4jGraphStore:
    """Manages storage and retrieval of knowledge graph data in Neo4j.

    Provides methods for:
    - Storing LLMGraphTransformer output as nodes/edges in Neo4j.
    - Querying for entity context via neighborhood traversal.
    - Formatting graph traversal results as natural language context
      suitable for injection into an LLM prompt.

    Attributes:
        graph: LangChain's Neo4jGraph connection wrapper.
    """

    def __init__(self, settings: Settings) -> None:
        try:
            self.graph = Neo4jGraph(
                url=settings.neo4j_uri,
                username=settings.neo4j_username,
                password=settings.neo4j_password,
            )
            logger.info("Connected to Neo4j at %s", settings.neo4j_uri)
        except Exception as e:
            logger.error("Failed to connect to Neo4j: %s", e)
            raise

    def add_graph_documents(self, graph_docs: list[GraphDocument]) -> None:
        """Store extracted graph documents in Neo4j.

        Each graph document's nodes and relationships are merged into
        the database. The `baseEntityLabel=True` flag adds a common
        `__Entity__` label to all nodes for unified querying. The
        `include_source=True` flag creates `MENTIONS` relationships
        from `Document` nodes to entity nodes, preserving provenance.

        Args:
            graph_docs: List of GraphDocument objects from the extractor.
        """
        if not graph_docs:
            logger.warning("No graph documents to store")
            return

        try:
            self.graph.add_graph_documents(
                graph_docs,
                baseEntityLabel=True,
                include_source=True,
            )
            total_nodes = sum(len(gd.nodes) for gd in graph_docs)
            total_rels = sum(len(gd.relationships) for gd in graph_docs)
            logger.info(
                "Stored %d nodes and %d relationships in Neo4j",
                total_nodes,
                total_rels,
            )
        except Exception as e:
            logger.error("Failed to store graph documents: %s", e)
            raise

    def query_related_context(
        self,
        entities: list[str],
        depth: int = 2,
        max_triples: int = 30,
    ) -> str:
        """Retrieve structured context about entities from the knowledge graph.

        Performs a neighborhood traversal around the given entity names,
        collecting related nodes and relationships up to the specified depth.
        Results are formatted as natural language triples for injection into
        LLM prompts.

        Args:
            entities: List of entity names to look up.
            depth: Maximum traversal depth from each entity.
            max_triples: Maximum number of triples to return.

        Returns:
            Formatted string of knowledge graph context, or empty string
            if no entities are found.
        """
        if not entities:
            return ""

        # Use case-insensitive matching for entity lookup
        cypher = """
        UNWIND $entities AS entity_name
        MATCH (e:__Entity__)
        WHERE toLower(e.id) = toLower(entity_name)
        WITH e
        MATCH path = (e)-[r*1..{depth}]-(connected:__Entity__)
        UNWIND relationships(path) AS rel
        WITH DISTINCT startNode(rel) AS source, rel, endNode(rel) AS target
        RETURN
            source.id AS source_name,
            type(rel) AS relationship,
            target.id AS target_name
        LIMIT $max_triples
        """.replace("{depth}", str(depth))

        try:
            results = self.graph.query(
                cypher,
                {"entities": entities, "max_triples": max_triples},
            )
        except Exception as e:
            logger.error("Knowledge graph query failed: %s", e)
            return ""

        if not results:
            logger.debug("No graph context found for entities: %s", entities)
            return ""

        return self._format_graph_context(results)

    def get_entity_neighbors(self, entity_name: str) -> list[dict[str, Any]]:
        """Get immediate neighbors of a specific entity.

        Useful for exploratory queries or building entity-centric context.

        Args:
            entity_name: The entity ID/name to look up.

        Returns:
            List of dicts with neighbor information.
        """
        cypher = """
        MATCH (e:__Entity__)-[r]-(neighbor:__Entity__)
        WHERE toLower(e.id) = toLower($entity_name)
        RETURN
            e.id AS entity,
            type(r) AS relationship,
            neighbor.id AS neighbor_name,
            labels(neighbor) AS neighbor_labels
        LIMIT 20
        """
        try:
            return self.graph.query(cypher, {"entity_name": entity_name})
        except Exception as e:
            logger.error("Neighbor query failed for %s: %s", entity_name, e)
            return []

    def get_graph_stats(self) -> dict[str, int]:
        """Return basic statistics about the knowledge graph."""
        try:
            node_count = self.graph.query("MATCH (n:__Entity__) RETURN count(n) AS count")
            rel_count = self.graph.query("MATCH ()-[r]->() RETURN count(r) AS count")
            return {
                "nodes": node_count[0]["count"] if node_count else 0,
                "relationships": rel_count[0]["count"] if rel_count else 0,
            }
        except Exception as e:
            logger.error("Graph stats query failed: %s", e)
            return {"nodes": 0, "relationships": 0}

    @staticmethod
    def _format_graph_context(results: list[dict[str, Any]]) -> str:
        """Format graph query results as natural language triples.

        Produces lines like:
            - Microsoft HEADQUARTERED_IN Redmond
            - Satya Nadella WORKS_AT Microsoft
        """
        lines = ["Knowledge Graph Context:"]
        seen = set()

        for row in results:
            source = row.get("source_name", "Unknown")
            rel = row.get("relationship", "RELATED_TO")
            target = row.get("target_name", "Unknown")

            triple = f"{source} → {rel} → {target}"
            if triple not in seen:
                seen.add(triple)
                lines.append(f"  - {triple}")

        return "\n".join(lines)

    def close(self) -> None:
        """Close the Neo4j driver connection."""
        try:
            if hasattr(self.graph, "_driver") and self.graph._driver:
                self.graph._driver.close()
                logger.info("Neo4j connection closed")
        except Exception as e:
            logger.warning("Error closing Neo4j connection: %s", e)
