"""LangGraph workflow compilation — the agentic RAG state machine.

Defines the directed cyclic graph that orchestrates the self-correcting
retrieval loop:

    analyze_query → hybrid_retrieve → grade_documents
        ├── relevant → generate_answer → check_hallucination
        │                                   ├── grounded → END
        │                                   └── not grounded → generate_answer (retry)
        └── not relevant → rewrite_query → hybrid_retrieve (loop)
                       └── max retries → generate_answer (fallback)

The graph enforces hard limits on retry cycles (retrieval and generation)
to prevent infinite loops and runaway token costs.
"""

from __future__ import annotations

import logging

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph as CompiledGraph

from src.agent.nodes import (
    analyze_query,
    check_hallucination,
    generate_answer,
    grade_documents,
    hybrid_retrieve,
    rewrite_query,
    set_search_pipeline,
)
from src.agent.state import AgentState
from src.config import Settings, get_settings
from src.graph.extractor import GraphExtractor
from src.graph.store import Neo4jGraphStore
from src.retrieval.hybrid_search import HybridSearchPipeline
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.vector_store import QdrantVectorStore

logger = logging.getLogger(__name__)


# ── Routing Functions ────────────────────────────────────────────────────


def route_after_grading(state: AgentState) -> str:
    """Decide what to do after the document grader runs.

    Decision tree:
      - If documents are relevant → proceed to generation.
      - If not relevant AND we haven't exceeded max retries → rewrite and retry.
      - If not relevant AND max retries exceeded → fallback to generation
        with whatever documents we have (with a caveat to the user).

    Args:
        state: Current agent state.

    Returns:
        Next node key: "generate", "rewrite", or "fallback_generate".
    """
    is_relevant = state.get("is_relevant", False)
    attempts = state.get("retrieval_attempts", 0)
    max_retries = state.get("max_retries", 3)

    if is_relevant:
        logger.info("Documents graded as relevant → generating answer")
        return "generate"

    if attempts < max_retries:
        logger.info(
            "Documents not relevant (attempt %d/%d) → rewriting query",
            attempts,
            max_retries,
        )
        return "rewrite"

    logger.warning(
        "Max retrieval retries (%d) exceeded → fallback generation",
        max_retries,
    )
    return "fallback_generate"


def route_after_hallucination_check(state: AgentState) -> str:
    """Decide what to do after the hallucination check.

    Decision tree:
      - If grounded → we're done, return to the user.
      - If not grounded AND generation_attempts < 2 → retry generation.
      - If not grounded AND too many generation attempts → accept and return.

    Args:
        state: Current agent state.

    Returns:
        Next node key: "grounded" or "retry_generation".
    """
    is_grounded = state.get("is_grounded", True)
    gen_attempts = state.get("generation_attempts", 0)
    max_gen_retries = 2  # Hard limit on generation retries

    if is_grounded:
        logger.info("Answer verified as grounded → returning to user")
        return "grounded"

    if gen_attempts < max_gen_retries:
        logger.warning(
            "Answer not grounded (attempt %d/%d) → retrying generation",
            gen_attempts,
            max_gen_retries,
        )
        return "retry_generation"

    logger.warning(
        "Max generation retries (%d) exceeded → accepting answer",
        max_gen_retries,
    )
    return "grounded"


# ── Graph Builder ────────────────────────────────────────────────────────


def build_agent_graph(
    vector_store: QdrantVectorStore | None = None,
    graph_store: Neo4jGraphStore | None = None,
    reranker: CrossEncoderReranker | None = None,
    settings: Settings | None = None,
) -> CompiledGraph:
    """Build and compile the agentic RAG state machine.

    This function wires all nodes together with conditional edges,
    injects the search pipeline into the nodes module, and compiles
    the graph with an in-memory checkpointer.

    Args:
        vector_store: Qdrant vector store instance.
        graph_store: Neo4j graph store instance.
        reranker: Cross-encoder reranker instance.
        settings: Application settings (used if components not provided).

    Returns:
        Compiled LangGraph state machine ready for invocation.
    """
    settings = settings or get_settings()

    # ── Initialize components if not provided ────────────────────────
    if vector_store is None:
        vector_store = QdrantVectorStore(settings)
    if graph_store is None:
        try:
            graph_store = Neo4jGraphStore(settings)
        except Exception as e:
            logger.warning(
                "Neo4j graph store unreachable (%s). Continuing with hybrid vector search.",
                e,
            )
            graph_store = None
    if reranker is None:
        reranker = CrossEncoderReranker(settings)

    # ── Create and inject the search pipeline ────────────────────────
    search_pipeline = HybridSearchPipeline(
        vector_store=vector_store,
        graph_store=graph_store,
        reranker=reranker,
    )
    set_search_pipeline(search_pipeline)

    # ── Build the state graph ────────────────────────────────────────
    workflow = StateGraph(AgentState)

    # Register nodes
    workflow.add_node("analyze_query", analyze_query)
    workflow.add_node("hybrid_retrieve", hybrid_retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("generate_answer", generate_answer)
    workflow.add_node("check_hallucination", check_hallucination)

    # ── Wire the graph edges ─────────────────────────────────────────

    # Entry point
    workflow.set_entry_point("analyze_query")

    # Linear edges
    workflow.add_edge("analyze_query", "hybrid_retrieve")
    workflow.add_edge("hybrid_retrieve", "grade_documents")

    # Conditional: grade → generate or rewrite
    workflow.add_conditional_edges(
        "grade_documents",
        route_after_grading,
        {
            "generate": "generate_answer",
            "rewrite": "rewrite_query",
            "fallback_generate": "generate_answer",
        },
    )

    # Rewrite loops back to retrieval
    workflow.add_edge("rewrite_query", "hybrid_retrieve")

    # Generation → hallucination check
    workflow.add_edge("generate_answer", "check_hallucination")

    # Conditional: hallucination check → end or retry
    workflow.add_conditional_edges(
        "check_hallucination",
        route_after_hallucination_check,
        {
            "grounded": END,
            "retry_generation": "generate_answer",
        },
    )

    # ── Compile with checkpointer ────────────────────────────────────
    checkpointer = MemorySaver()
    compiled = workflow.compile(checkpointer=checkpointer)

    logger.info("Agent graph compiled successfully")
    return compiled


# ── Convenience: Run a query ─────────────────────────────────────────────


async def run_query(
    graph: CompiledGraph,
    query: str,
    thread_id: str = "default",
    max_retries: int | None = None,
) -> dict:
    """Run a single query through the agentic RAG pipeline.

    This is the main public API for executing queries. It initializes
    the agent state, invokes the compiled graph, and returns the final
    state containing the answer, citations, and metadata.

    Args:
        graph: Compiled LangGraph state machine.
        query: User's natural language question.
        thread_id: Session/conversation thread ID for checkpointing.
        max_retries: Override the default max retrieval retries.

    Returns:
        Final agent state dict with generation, citations, etc.
    """
    settings = get_settings()

    initial_state: AgentState = {
        "messages": [HumanMessage(content=query)],
        "query": query,
        "rewritten_query": "",
        "extracted_entities": [],
        "documents": [],
        "generation": "",
        "retrieval_attempts": 0,
        "max_retries": max_retries or settings.max_retrieval_retries,
        "is_relevant": False,
        "is_grounded": False,
        "generation_attempts": 0,
        "citations": [],
    }

    config = {"configurable": {"thread_id": thread_id}}

    logger.info("Running query through agent graph: %.80s...", query)

    final_state = await graph.ainvoke(initial_state, config)

    logger.info(
        "Query complete. Retrieval attempts: %d, Generation attempts: %d, Citations: %d",
        final_state.get("retrieval_attempts", 0),
        final_state.get("generation_attempts", 0),
        len(final_state.get("citations", [])),
    )

    return final_state


def run_query_sync(
    graph: CompiledGraph,
    query: str,
    thread_id: str = "default",
    max_retries: int | None = None,
) -> dict:
    """Run a single query synchronously through the agentic RAG pipeline.

    Spawns an isolated worker thread with a fresh event loop to execute the
    asynchronous agent workflow via `graph.ainvoke`, avoiding both LangGraph's
    async node restrictions and Streamlit event loop collisions.
    """
    import asyncio
    import concurrent.futures

    def _worker():
        return asyncio.run(
            run_query(
                graph=graph,
                query=query,
                thread_id=thread_id,
                max_retries=max_retries,
            )
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_worker)
        return future.result()


# Import HumanMessage for run_query
from langchain_core.messages import HumanMessage  # noqa: E402
