"""Centralized application configuration with environment validation.

All settings are loaded from environment variables (with .env file support)
and validated at startup using pydantic-settings. Supports both Google Gemini
and OpenAI providers.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any
from typing_extensions import Self

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Validated application configuration loaded from environment variables.

    Supports both Google Gemini and OpenAI as LLM backends.
    """

    # ── LLM Provider & Keys ──────────────────────────────────────────────
    llm_provider: str = Field(
        default="google",
        description="Active LLM provider: 'google' or 'openai'",
    )
    google_api_key: str | None = Field(
        default=None,
        description="Google Gemini API key for LLM operations",
    )
    google_model: str = Field(
        default="gemini-2.0-flash",
        description="Gemini model identifier (e.g. gemini-2.0-flash, gemini-1.5-flash)",
    )
    openai_api_key: str | None = Field(
        default=None,
        description="OpenAI API key for LLM operations",
    )
    openai_model: str = Field(
        default="gpt-4o",
        description="OpenAI model identifier",
    )

    # ── Neo4j ────────────────────────────────────────────────────────────
    neo4j_uri: str = Field(
        default="bolt://localhost:7687",
        description="Neo4j Bolt connection URI",
    )
    neo4j_username: str = Field(default="neo4j")
    neo4j_password: str = Field(default="graphrag_password")

    # ── Qdrant ───────────────────────────────────────────────────────────
    qdrant_url: str = Field(
        default="http://localhost:6333",
        description="Qdrant REST API URL (falls back to local storage if unreachable)",
    )
    qdrant_collection_name: str = Field(default="document_chunks")

    # ── Embedding ────────────────────────────────────────────────────────
    embedding_model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="HuggingFace model ID for dense embeddings",
    )
    embedding_dimension: int = Field(
        default=384,
        description="Dimensionality of the dense embedding model output",
    )

    # ── Sparse Embedding (BM25) ─────────────────────────────────────────
    sparse_model_name: str = Field(
        default="Qdrant/bm25",
        description="FastEmbed sparse model for BM25-style keyword vectors",
    )

    # ── Cross-Encoder Reranker ───────────────────────────────────────────
    reranker_model_name: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="HuggingFace cross-encoder model for reranking",
    )
    reranker_top_n: int = Field(
        default=5,
        description="Number of documents to keep after cross-encoder reranking",
    )

    # ── Agent ────────────────────────────────────────────────────
    max_retrieval_retries: int = Field(
        default=3,
        description="Maximum self-correction retrieval loops before fallback",
    )

    # ── Chunking ─────────────────────────────────────────────────────────
    parent_chunk_size: int = Field(
        default=1500,
        description="Maximum character length of parent (context) chunks",
    )
    child_chunk_size: int = Field(
        default=400,
        description="Maximum character length of child (retrieval) chunks",
    )
    chunk_overlap: int = Field(
        default=100,
        description="Character overlap between consecutive chunks",
    )

    # ── Validators ───────────────────────────────────────────────────────

    @field_validator("openai_api_key", mode="before")
    @classmethod
    def empty_str_to_none_openai(cls, v: Any) -> Any:
        if v == "" or (isinstance(v, str) and v.startswith("sk-your")):
            return None
        return v

    @field_validator("google_api_key", mode="before")
    @classmethod
    def empty_str_to_none_google(cls, v: Any) -> Any:
        if v == "":
            return None
        return v

    @model_validator(mode="after")
    def validate_llm_config(self) -> Self:
        # Auto-detect provider if one key is provided
        if self.google_api_key and not self.openai_api_key:
            self.llm_provider = "google"
        elif self.openai_api_key and not self.google_api_key:
            self.llm_provider = "openai"

        if self.llm_provider == "google" and not self.google_api_key:
            raise ValueError(
                "GOOGLE_API_KEY must be configured in .env to use Google Gemini."
            )
        if self.llm_provider == "openai" and not self.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY must be configured in .env to use OpenAI."
            )
        return self

    @field_validator("embedding_dimension")
    @classmethod
    def validate_dimension(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("embedding_dimension must be a positive integer")
        return v

    @field_validator("max_retrieval_retries")
    @classmethod
    def validate_retries(cls, v: int) -> int:
        if v < 1 or v > 10:
            raise ValueError("max_retrieval_retries must be between 1 and 10")
        return v

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",
    }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    settings = Settings()  # type: ignore[call-arg]
    logger.info("Configuration loaded: provider=%s", settings.llm_provider)
    return settings
