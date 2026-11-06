"""Individual node functions for the LangGraph agentic workflow.

Each function takes the current AgentState, performs its task, and returns
a partial state update dict. LangGraph merges these updates into the
accumulated state automatically.

Node responsibilities:
  - analyze_query:        Extract intent, entities, optimize the search query.
  - hybrid_retrieve:      Execute the hybrid search pipeline.
  - grade_documents:      LLM-as-Judge evaluates document relevance.
  - rewrite_query:        Reformulate the query for better retrieval on retry.
  - generate_answer:      Produce a grounded answer with inline citations.
  - check_hallucination:  Verify the answer is grounded in the retrieved docs.
"""

from __future__ import annotations

import logging

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from src.agent.prompts import (
    DOCUMENT_GRADING_PROMPT,
    GENERATION_PROMPT,
    HALLUCINATION_CHECK_PROMPT,
    QUERY_ANALYSIS_PROMPT,
    QUERY_REWRITE_PROMPT,
    CitedAnswer,
    GradeResult,
    HallucinationCheck,
    QueryAnalysis,
    RewrittenQuery,
)
from src.agent.state import AgentState
from src.config import get_settings
from src.retrieval.hybrid_search import HybridSearchPipeline

logger = logging.getLogger(__name__)

# ── Shared LLM Instance ─────────────────────────────────────────────────
# Lazily initialized on first use to avoid import-time config loading.
_llm = None
_search_pipeline: HybridSearchPipeline | None = None


def set_search_pipeline(pipeline: HybridSearchPipeline) -> None:
    """Inject the search pipeline dependency (called during app startup)."""
    global _search_pipeline
    _search_pipeline = pipeline


def _get_llm():
    """Get or create the shared LLM instance supporting Google Gemini and OpenAI."""
    global _llm
    if _llm is None:
        settings = get_settings()
        if settings.llm_provider == "google":
            from langchain_google_genai import ChatGoogleGenerativeAI
            _llm = ChatGoogleGenerativeAI(
                model=settings.google_model,
                temperature=0,
                google_api_key=settings.google_api_key,
            )
        else:
            from langchain_openai import ChatOpenAI
            _llm = ChatOpenAI(
                model=settings.openai_model,
                temperature=0,
                api_key=settings.openai_api_key,
            )
    return _llm


# ── Helper Functions ─────────────────────────────────────────────────────


def _format_documents(documents: list[dict]) -> str:
    """Format retrieved documents for injection into prompts.

    Each document is numbered for citation reference and includes
    metadata about page numbers and section titles.
    """
    if not documents:
        return "No documents retrieved."

    formatted = []
    for i, doc in enumerate(documents, 1):
        pages = doc.get("page_numbers", [])
        section = doc.get("section_title", "")
        source = doc.get("source", "vector_store")

        header_parts = [f"[Source {i}]"]
        if pages:
            header_parts.append(f"Pages {min(pages)}–{max(pages)}")
        if section:
            header_parts.append(f"Section: {section}")
        if source == "knowledge_graph":
            header_parts.append("(Knowledge Graph)")

        header = " | ".join(header_parts)
        text = doc.get("text", "")

        formatted.append(f"{header}\n{text}")

    return "\n\n---\n\n".join(formatted)


def _format_chat_history(messages: list) -> str:
    """Format message history for context resolution."""
    if not messages:
        return "No previous conversation."

    history_lines = []
    for msg in messages[-6:]:  # Last 3 turns max
        if isinstance(msg, HumanMessage):
            history_lines.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            content = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
            history_lines.append(f"Assistant: {content}")

    return "\n".join(history_lines)


# ── Node Functions ───────────────────────────────────────────────────────


async def analyze_query(state: AgentState) -> dict:
    """Extract entities, classify intent, and optimize the search query.

    This is the entry point of the agentic loop. It analyzes the user's
    raw question and produces an optimized search query plus extracted
    entities for knowledge graph lookup.

    Returns:
        Partial state update with rewritten_query, extracted_entities,
        and initialized counters.
    """
    llm = _get_llm()
    query = state["query"]

    logger.info("Analyzing query: %.80s...", query)

    prompt = ChatPromptTemplate.from_template(QUERY_ANALYSIS_PROMPT)
    chain = prompt | llm.with_structured_output(QueryAnalysis)

    try:
        result = await chain.ainvoke({
            "query": query,
            "chat_history": _format_chat_history(state.get("messages", [])),
        })

        logger.info(
            "Query analysis: type=%s, entities=%s, optimized=%.80s...",
            result.query_type,
            result.entities[:5],
            result.optimized_query,
        )

        return {
            "rewritten_query": result.optimized_query,
            "extracted_entities": result.entities,
            "retrieval_attempts": 0,
            "generation_attempts": 0,
        }

    except Exception as e:
        logger.error("Query analysis failed: %s. Using raw query.", e)
        return {
            "rewritten_query": query,
            "extracted_entities": [],
            "retrieval_attempts": 0,
            "generation_attempts": 0,
        }


async def hybrid_retrieve(state: AgentState) -> dict:
    """Execute the hybrid search pipeline with the current query.

    Runs dense + BM25 search via Qdrant RRF fusion, applies cross-encoder
    reranking, and augments with knowledge graph context.

    Returns:
        Partial state update with documents and incremented retrieval_attempts.
    """
    if _search_pipeline is None:
        raise RuntimeError("Search pipeline not initialized. Call set_search_pipeline() first.")

    query = state.get("rewritten_query") or state["query"]
    entities = state.get("extracted_entities", [])
    attempt = state.get("retrieval_attempts", 0) + 1

    logger.info("Hybrid retrieval attempt %d: %.80s...", attempt, query)

    try:
        results = _search_pipeline.search(
            query=query,
            entities=entities,
        )

        documents = [
            {
                "text": doc.text,
                "score": doc.score,
                "chunk_id": doc.metadata.get("chunk_id"),
                "document_id": doc.metadata.get("document_id"),
                "page_numbers": doc.metadata.get("page_numbers", []),
                "section_title": doc.metadata.get("section_title", ""),
                "source": doc.metadata.get("source", "vector_store"),
            }
            for doc in results
        ]

        logger.info("Retrieved %d documents", len(documents))

        return {
            "documents": documents,
            "retrieval_attempts": attempt,
        }

    except Exception as e:
        logger.error("Hybrid retrieval failed: %s", e)
        return {
            "documents": [],
            "retrieval_attempts": attempt,
        }


async def grade_documents(state: AgentState) -> dict:
    """LLM-as-Judge: evaluate whether retrieved documents are relevant.

    Uses structured output to classify documents as relevant or irrelevant
    to the user's query. This determination drives the self-correction
    loop: relevant → generate, irrelevant → rewrite and retry.

    Returns:
        Partial state update with is_relevant boolean.
    """
    llm = _get_llm()
    documents = state.get("documents", [])
    query = state["query"]

    if not documents:
        logger.warning("No documents to grade")
        return {"is_relevant": False}

    logger.info("Grading %d documents for relevance...", len(documents))

    prompt = ChatPromptTemplate.from_template(DOCUMENT_GRADING_PROMPT)
    chain = prompt | llm.with_structured_output(GradeResult)

    try:
        result = await chain.ainvoke({
            "query": query,
            "documents": _format_documents(documents),
        })

        logger.info(
            "Grading result: relevant=%s, reasoning=%.100s...",
            result.is_relevant,
            result.reasoning,
        )

        return {"is_relevant": result.is_relevant}

    except Exception as e:
        logger.error("Document grading failed: %s. Assuming relevant.", e)
        # Fail-open: if grading fails, proceed to generation
        return {"is_relevant": True}


async def rewrite_query(state: AgentState) -> dict:
    """Reformulate the query to improve retrieval on the next attempt.

    Called when the grader determines retrieved documents are irrelevant.
    Uses the original query, previous search query, and attempt number
    to generate a better reformulation.

    Returns:
        Partial state update with rewritten_query.
    """
    llm = _get_llm()
    query = state["query"]
    previous_query = state.get("rewritten_query", query)
    attempt = state.get("retrieval_attempts", 1)
    max_retries = state.get("max_retries", 3)

    logger.info("Rewriting query (attempt %d/%d)...", attempt, max_retries)

    prompt = ChatPromptTemplate.from_template(QUERY_REWRITE_PROMPT)
    chain = prompt | llm.with_structured_output(RewrittenQuery)

    try:
        result = await chain.ainvoke({
            "query": query,
            "previous_query": previous_query,
            "attempt_number": attempt,
            "max_attempts": max_retries,
        })

        logger.info(
            "Rewritten query: %.80s... (reason: %.80s...)",
            result.rewritten_query,
            result.reasoning,
        )

        return {"rewritten_query": result.rewritten_query}

    except Exception as e:
        logger.error("Query rewriting failed: %s. Using original query.", e)
        return {"rewritten_query": query}


async def generate_answer(state: AgentState) -> dict:
    """Generate a grounded answer with strict inline citations.

    Produces a comprehensive answer based solely on the retrieved documents,
    with every factual claim backed by an inline citation. Uses structured
    output to ensure citations are machine-parseable.

    Returns:
        Partial state update with generation, citations, generation_attempts,
        and a new AI message appended to messages.
    """
    llm = _get_llm()
    query = state["query"]
    documents = state.get("documents", [])
    gen_attempt = state.get("generation_attempts", 0) + 1

    logger.info("Generating answer (attempt %d)...", gen_attempt)

    prompt = ChatPromptTemplate.from_template(GENERATION_PROMPT)
    chain = prompt | llm.with_structured_output(CitedAnswer)

    try:
        result = await chain.ainvoke({
            "query": query,
            "context": _format_documents(documents),
        })

        citations = [
            {
                "chunk_id": c.chunk_id,
                "page_numbers": c.page_numbers,
                "text_snippet": c.text_snippet,
                "section_title": c.section_title,
            }
            for c in result.citations
        ]

        logger.info(
            "Generated answer with %d citations (%.80s...)",
            len(citations),
            result.answer,
        )

        return {
            "generation": result.answer,
            "citations": citations,
            "generation_attempts": gen_attempt,
            "messages": [AIMessage(content=result.answer)],
        }

    except Exception as e:
        logger.error("Answer generation failed: %s", e)
        fallback = (
            "I apologize, but I encountered an error generating an answer. "
            "Please try rephrasing your question."
        )
        return {
            "generation": fallback,
            "citations": [],
            "generation_attempts": gen_attempt,
            "messages": [AIMessage(content=fallback)],
        }


async def check_hallucination(state: AgentState) -> dict:
    """Verify that the generated answer is grounded in retrieved documents.

    Acts as a guardrail against hallucination. If the answer contains
    claims not traceable to the source documents, it's flagged and the
    generation node is retried with awareness of the ungrounded claims.

    Returns:
        Partial state update with is_grounded boolean.
    """
    llm = _get_llm()
    generation = state.get("generation", "")
    documents = state.get("documents", [])

    if not generation or not documents:
        logger.warning("No generation or documents to check")
        return {"is_grounded": True}

    logger.info("Checking hallucination...")

    prompt = ChatPromptTemplate.from_template(HALLUCINATION_CHECK_PROMPT)
    chain = prompt | llm.with_structured_output(HallucinationCheck)

    try:
        result = await chain.ainvoke({
            "generation": generation,
            "documents": _format_documents(documents),
        })

        if not result.is_grounded:
            logger.warning(
                "Hallucination detected! Ungrounded claims: %s",
                result.ungrounded_claims,
            )
        else:
            logger.info("Answer verified as grounded")

        return {"is_grounded": result.is_grounded}

    except Exception as e:
        logger.error("Hallucination check failed: %s. Assuming grounded.", e)
        return {"is_grounded": True}
