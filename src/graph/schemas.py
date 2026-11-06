"""Graph node and relationship type definitions for post-hoc normalization.

The system uses open-ended entity extraction (the LLM discovers types
dynamically), but we normalize extracted types to canonical forms during
storage. This module defines the normalization mappings and Pydantic models
for type-safe graph data handling.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ── Normalization Mappings ───────────────────────────────────────────────

# Maps common LLM-generated entity type variants to canonical labels.
# Extend this as you observe new patterns in your document domain.
ENTITY_TYPE_NORMALIZATION: dict[str, str] = {
    # Person variants
    "person": "Person",
    "people": "Person",
    "individual": "Person",
    "author": "Person",
    "researcher": "Person",
    "scientist": "Person",
    # Organization variants
    "organization": "Organization",
    "organisation": "Organization",
    "company": "Organization",
    "corporation": "Organization",
    "institution": "Organization",
    "university": "Organization",
    "agency": "Organization",
    # Location variants
    "location": "Location",
    "place": "Location",
    "city": "Location",
    "country": "Location",
    "region": "Location",
    "state": "Location",
    # Concept variants
    "concept": "Concept",
    "topic": "Concept",
    "theory": "Concept",
    "method": "Concept",
    "methodology": "Concept",
    "technique": "Concept",
    "technology": "Technology",
    "tool": "Technology",
    "software": "Technology",
    "framework": "Technology",
    # Document variants
    "document": "Document",
    "paper": "Document",
    "article": "Document",
    "report": "Document",
    "publication": "Document",
    # Event variants
    "event": "Event",
    "conference": "Event",
    "meeting": "Event",
    # Metric / Data variants
    "metric": "Metric",
    "measurement": "Metric",
    "statistic": "Metric",
    "percentage": "Metric",
    "value": "Metric",
    "number": "Metric",
    # Date variants
    "date": "Date",
    "year": "Date",
    "time_period": "Date",
}

RELATIONSHIP_TYPE_NORMALIZATION: dict[str, str] = {
    "works_at": "WORKS_AT",
    "works_for": "WORKS_AT",
    "employed_by": "WORKS_AT",
    "affiliated_with": "AFFILIATED_WITH",
    "associated_with": "AFFILIATED_WITH",
    "located_in": "LOCATED_IN",
    "based_in": "LOCATED_IN",
    "headquartered_in": "LOCATED_IN",
    "founded": "FOUNDED",
    "created": "CREATED",
    "developed": "CREATED",
    "authored": "AUTHORED",
    "wrote": "AUTHORED",
    "published": "PUBLISHED",
    "published_in": "PUBLISHED_IN",
    "part_of": "PART_OF",
    "belongs_to": "PART_OF",
    "contains": "CONTAINS",
    "includes": "CONTAINS",
    "related_to": "RELATED_TO",
    "refers_to": "REFERS_TO",
    "mentions": "MENTIONS",
    "uses": "USES",
    "utilizes": "USES",
    "applied_to": "APPLIED_TO",
}


# ── Pydantic Models ──────────────────────────────────────────────────────


class GraphEntity(BaseModel):
    """A normalized entity extracted from document text."""

    id: str = Field(..., description="Unique identifier for the entity")
    name: str = Field(..., description="Human-readable entity name")
    entity_type: str = Field(..., description="Normalized entity type label")
    properties: dict = Field(default_factory=dict, description="Additional properties")


class GraphRelationship(BaseModel):
    """A normalized relationship between two entities."""

    source_id: str = Field(..., description="Source entity ID")
    target_id: str = Field(..., description="Target entity ID")
    relationship_type: str = Field(..., description="Normalized relationship type")
    properties: dict = Field(default_factory=dict, description="Additional properties")


# ── Normalization Functions ──────────────────────────────────────────────


def normalize_entity_type(raw_type: str) -> str:
    """Normalize an LLM-generated entity type to a canonical label.

    Falls back to title-casing the raw type if no mapping exists,
    ensuring consistent casing even for unmapped types.
    """
    key = raw_type.lower().strip().replace(" ", "_")
    return ENTITY_TYPE_NORMALIZATION.get(key, raw_type.strip().title())


def normalize_relationship_type(raw_type: str) -> str:
    """Normalize an LLM-generated relationship type to a canonical label.

    Falls back to uppercasing with underscores for unmapped types,
    matching Neo4j relationship naming conventions.
    """
    key = raw_type.lower().strip().replace(" ", "_")
    return RELATIONSHIP_TYPE_NORMALIZATION.get(
        key, raw_type.strip().upper().replace(" ", "_")
    )
