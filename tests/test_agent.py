"""Unit tests for the LangGraph agent state machine and routing logic.

These tests validate state transitions, routing logic, loop termination guards,
and graph compilation without requiring live LLM calls or databases.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage

from src.agent.graph import (
    build_agent_graph,
    route_after_grading,
    route_after_hallucination_check,
)
from src.agent.state import AgentState


# ── Routing Logic Tests ──────────────────────────────────────────────────


class TestAgentRouting:
    """Tests for conditional edge routing functions in the agent graph."""

    def test_route_after_grading_relevant(self):
        state: AgentState = {
            "messages": [HumanMessage(content="What are the terms?")],
            "query": "What are the terms?",
            "rewritten_query": "",
            "extracted_entities": [],
            "documents": [{"text": "doc1"}],
            "generation": "",
            "retrieval_attempts": 1,
            "max_retries": 3,
            "is_relevant": True,
            "is_grounded": True,
            "generation_attempts": 0,
            "citations": [],
        }
        route = route_after_grading(state)
        assert route == "generate"

    def test_route_after_grading_not_relevant_under_limit(self):
        state: AgentState = {
            "messages": [HumanMessage(content="Query")],
            "query": "Query",
            "rewritten_query": "",
            "extracted_entities": [],
            "documents": [{"text": "irrelevant doc"}],
            "generation": "",
            "retrieval_attempts": 1,
            "max_retries": 3,
            "is_relevant": False,
            "is_grounded": True,
            "generation_attempts": 0,
            "citations": [],
        }
        route = route_after_grading(state)
        assert route == "rewrite"

    def test_route_after_grading_not_relevant_max_retries_exceeded(self):
        state: AgentState = {
            "messages": [HumanMessage(content="Query")],
            "query": "Query",
            "rewritten_query": "",
            "extracted_entities": [],
            "documents": [],
            "generation": "",
            "retrieval_attempts": 3,
            "max_retries": 3,
            "is_relevant": False,
            "is_grounded": True,
            "generation_attempts": 0,
            "citations": [],
        }
        route = route_after_grading(state)
        assert route == "fallback_generate"

    def test_route_after_hallucination_grounded(self):
        state: AgentState = {
            "messages": [],
            "query": "Query",
            "rewritten_query": "",
            "extracted_entities": [],
            "documents": [],
            "generation": "Accurate answer",
            "retrieval_attempts": 1,
            "max_retries": 3,
            "is_relevant": True,
            "is_grounded": True,
            "generation_attempts": 1,
            "citations": [],
        }
        route = route_after_hallucination_check(state)
        assert route == "grounded"

    def test_route_after_hallucination_ungrounded_retry(self):
        state: AgentState = {
            "messages": [],
            "query": "Query",
            "rewritten_query": "",
            "extracted_entities": [],
            "documents": [],
            "generation": "Hallucinated claim",
            "retrieval_attempts": 1,
            "max_retries": 3,
            "is_relevant": True,
            "is_grounded": False,
            "generation_attempts": 1,  # First attempt failed
            "citations": [],
        }
        route = route_after_hallucination_check(state)
        assert route == "retry_generation"

    def test_route_after_hallucination_ungrounded_max_attempts(self):
        state: AgentState = {
            "messages": [],
            "query": "Query",
            "rewritten_query": "",
            "extracted_entities": [],
            "documents": [],
            "generation": "Still uncertain claim",
            "retrieval_attempts": 1,
            "max_retries": 3,
            "is_relevant": True,
            "is_grounded": False,
            "generation_attempts": 2,  # Max generation attempts hit
            "citations": [],
        }
        route = route_after_hallucination_check(state)
        assert route == "grounded"


# ── Graph Compilation Tests ──────────────────────────────────────────────


class TestAgentGraphCompilation:
    """Tests that the LangGraph workflow compiles cleanly with all nodes and transitions."""

    def test_build_agent_graph_structure(self):
        mock_vs = MagicMock()
        mock_gs = MagicMock()
        mock_reranker = MagicMock()
        mock_settings = MagicMock()

        compiled_graph = build_agent_graph(
            vector_store=mock_vs,
            graph_store=mock_gs,
            reranker=mock_reranker,
            settings=mock_settings,
        )

        assert compiled_graph is not None

        # Verify all nodes are registered in the graph
        expected_nodes = {
            "analyze_query",
            "hybrid_retrieve",
            "grade_documents",
            "rewrite_query",
            "generate_answer",
            "check_hallucination",
        }
        # In LangGraph compiled graphs, nodes can be inspected via .nodes
        for node in expected_nodes:
            assert node in compiled_graph.nodes
