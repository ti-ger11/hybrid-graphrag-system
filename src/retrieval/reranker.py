"""Cross-Encoder reranking module.

Implements a retrieve-then-rerank pattern: after the initial hybrid search
(dense + BM25 via RRF) returns a broad set of candidates, this module
scores each (query, document) pair jointly through a cross-encoder
transformer, producing far more accurate relevance scores than
bi-encoder cosine similarity alone.

The cross-encoder processes the query and document simultaneously in a
single transformer forward pass, capturing deep token-level interactions
that bi-encoders miss.
"""

from __future__ import annotations

import logging
from typing import Sequence

from sentence_transformers import CrossEncoder

from src.config import Settings
from src.retrieval.vector_store import RetrievedDocument

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Reranks retrieved documents using a cross-encoder model.

    The reranker scores each (query, document) pair jointly, then returns
    the top-N highest-scoring documents. This dramatically improves
    precision over raw vector similarity scores.

    Attributes:
        model: sentence_transformers cross-encoder model wrapper.
        top_n: Number of documents to keep after reranking.
    """

    def __init__(self, settings: Settings) -> None:
        self.top_n = settings.reranker_top_n
        logger.info("Loading cross-encoder model: %s", settings.reranker_model_name)

        self.model = CrossEncoder(settings.reranker_model_name)
        logger.info("Cross-encoder model loaded successfully")

    def rerank(
        self,
        query: str,
        documents: Sequence[RetrievedDocument],
        top_n: int | None = None,
    ) -> list[RetrievedDocument]:
        """Score and rerank documents using the cross-encoder.

        Args:
            query: The user's search query.
            documents: Candidate documents from the hybrid search.
            top_n: Override the default top-N cutoff.

        Returns:
            Top-N documents reranked by cross-encoder relevance score,
            in descending order.
        """
        if not documents:
            return []

        effective_top_n = top_n or self.top_n

        # If fewer documents than top_n, just score and return all
        if len(documents) <= effective_top_n:
            logger.debug(
                "Document count (%d) <= top_n (%d), scoring all without cutoff",
                len(documents),
                effective_top_n,
            )

        # Create (query, document_text) pairs for joint scoring
        pairs = [(query, doc.text) for doc in documents]

        try:
            if hasattr(self.model, "predict"):
                scores = self.model.predict(pairs)
            else:
                scores = self.model.score(pairs)
        except Exception as e:
            logger.error("Cross-encoder scoring failed: %s", e)
            # Fallback: return documents in original order
            return list(documents[:effective_top_n])

        # Sort by cross-encoder score (descending)
        scored_pairs = sorted(
            zip(documents, scores),
            key=lambda x: float(x[1]),
            reverse=True,
        )

        reranked = []
        for doc, ce_score in scored_pairs[:effective_top_n]:
            # Create a new RetrievedDocument with the cross-encoder score
            reranked_doc = RetrievedDocument(
                text=doc.text,
                score=float(ce_score),
                metadata={
                    **doc.metadata,
                    "original_rrf_score": doc.score,
                    "cross_encoder_score": float(ce_score),
                },
            )
            reranked.append(reranked_doc)

        logger.debug(
            "Reranked %d → %d documents. Score range: [%.4f, %.4f]",
            len(documents),
            len(reranked),
            reranked[-1].score if reranked else 0,
            reranked[0].score if reranked else 0,
        )

        return reranked
