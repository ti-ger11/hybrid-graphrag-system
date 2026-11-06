"""Unit tests for the knowledge graph schemas, normalization, and store operations.

These tests validate entity and relationship normalization, data models,
and Cypher query generation without requiring a running Neo4j database.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.graph.schemas import (
    GraphEntity,
    GraphRelationship,
    normalize_entity_type,
    normalize_relationship_type,
)
from src.graph.store import Neo4jGraphStore


# ── Normalization Tests ──────────────────────────────────────────────────


class TestNormalization:
    """Tests for entity and relationship type canonicalization."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("company", "Organization"),
            ("university", "Organization"),
            ("author", "Person"),
            ("researcher", "Person"),
            ("city", "Location"),
            ("framework", "Technology"),
            ("report", "Document"),
            ("custom_new_type", "Custom_New_Type"),
        ],
    )
    def test_normalize_entity_type(self, raw: str, expected: str):
        assert normalize_entity_type(raw) == expected

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("works_at", "WORKS_AT"),
            ("employed_by", "WORKS_AT"),
            ("headquartered_in", "LOCATED_IN"),
            ("developed", "CREATED"),
            ("published_in", "PUBLISHED_IN"),
            ("collaborates with", "COLLABORATES_WITH"),
        ],
    )
    def test_normalize_relationship_type(self, raw: str, expected: str):
        assert normalize_relationship_type(raw) == expected


# ── Pydantic Models Tests ────────────────────────────────────────────────


class TestGraphModels:
    """Tests for GraphEntity and GraphRelationship Pydantic models."""

    def test_entity_creation(self):
        entity = GraphEntity(
            id="org-google",
            name="Google",
            entity_type="Organization",
            properties={"country": "USA"},
        )
        assert entity.id == "org-google"
        assert entity.name == "Google"
        assert entity.entity_type == "Organization"
        assert entity.properties["country"] == "USA"

    def test_relationship_creation(self):
        rel = GraphRelationship(
            source_id="person-alice",
            target_id="org-google",
            relationship_type="WORKS_AT",
            properties={"role": "Engineer"},
        )
        assert rel.source_id == "person-alice"
        assert rel.target_id == "org-google"
        assert rel.relationship_type == "WORKS_AT"
        assert rel.properties["role"] == "Engineer"


# ── Neo4jGraphStore Mock Tests ───────────────────────────────────────────


class TestNeo4jGraphStore:
    """Tests for graph store query and context operations using mocks."""

    @patch("src.graph.store.Neo4jGraph")
    def test_store_entities_and_relationships(self, mock_neo4j_cls):
        mock_graph = MagicMock()
        mock_neo4j_cls.return_value = mock_graph

        mock_settings = MagicMock()
        mock_settings.neo4j_uri = "bolt://localhost:7687"
        mock_settings.neo4j_username = "neo4j"
        mock_settings.neo4j_password = "password"

        store = Neo4jGraphStore(mock_settings)

        mock_doc = MagicMock()
        mock_doc.nodes = [MagicMock()]
        mock_doc.relationships = [MagicMock()]

        store.add_graph_documents([mock_doc])

        # Assert add_graph_documents was invoked on graph
        assert mock_graph.add_graph_documents.called

    @patch("src.graph.store.Neo4jGraph")
    def test_get_entity_context_formatting(self, mock_neo4j_cls):
        mock_graph = MagicMock()
        mock_neo4j_cls.return_value = mock_graph

        # Mock result records returned from query
        mock_record_1 = {
            "source_name": "Alice",
            "relationship": "WORKS_AT",
            "target_name": "Acme",
        }
        mock_graph.query.return_value = [mock_record_1]

        mock_settings = MagicMock()
        mock_settings.neo4j_uri = "bolt://localhost:7687"
        mock_settings.neo4j_username = "neo4j"
        mock_settings.neo4j_password = "password"

        store = Neo4jGraphStore(mock_settings)
        context = store.query_related_context(["Alice"])

        assert "Alice" in context
        assert "WORKS_AT" in context
        assert "Acme" in context
