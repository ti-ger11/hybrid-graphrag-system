"""LangGraph agent state schema.

Defines the typed state that flows through the agentic retrieval loop.
Every node reads from and writes to this state, enabling LangGraph to
manage transitions, checkpointing, and self-correction cycles.

Using TypedDict + Annotated with LangGraph's message reducer ensures
proper message accumulation across node transitions.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """Immutable state flowing through the agentic RAG retrieval loop.

    This state is read/written by all nodes in the LangGraph workflow.
    LangGraph serializes and checkpoints this state between every node
    transition for fault tolerance and session persistence.

    Attributes:
        messages: Accumulated conversation messages (user + assistant).
                  Uses LangGraph's add_messages reducer to append rather
                  than overwrite.
        query: Original user question (immutable throughout the loop).
        rewritten_query: Query after reformulation (updated by rewriter node).
        extracted_entities: Entity names identified from the query for KG lookup.
        documents: Retrieved + reranked documents (list of dicts with text,
                   metadata, and scores).
        generation: The generated answer text.
        retrieval_attempts: Loop counter preventing infinite self-correction
                           cycles. Incremented each time the retriever runs.
        max_retries: Maximum allowed retrieval attempts (from settings).
        is_relevant: Grader output — whether retrieved docs answer the question.
        is_grounded: Hallucination guard output — whether the answer is
                     grounded in retrieved documents.
        generation_attempts: Counter for generation retry cycles (hallucination).
        citations: Source citations for the generated answer. Each citation is
                   a dict with keys: chunk_id, page_numbers, text_snippet,
                   section_title.
    """

    messages: Annotated[list[BaseMessage], add_messages]
    query: str
    rewritten_query: str
    extracted_entities: list[str]
    documents: list[dict]
    generation: str
    retrieval_attempts: int
    max_retries: int
    is_relevant: bool
    is_grounded: bool
    generation_attempts: int
    citations: list[dict]
