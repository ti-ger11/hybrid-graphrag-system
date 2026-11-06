"""Centralized prompt templates for all LangGraph agent nodes.

All prompts are defined here to keep node logic clean and enable
easy iteration on prompt engineering without touching business logic.

Each prompt uses LangChain's ChatPromptTemplate format with explicit
variable placeholders. The structured output schemas (QueryAnalysis,
GradeResult, CitedAnswer, HallucinationCheck) are defined alongside
their prompts for co-location.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ── Structured Output Schemas ────────────────────────────────────────────


class QueryAnalysis(BaseModel):
    """Structured output from the query analysis node."""

    optimized_query: str = Field(
        description="An optimized version of the user's question, "
        "rephrased for better retrieval performance. Should be "
        "self-contained (no pronouns referencing chat history)."
    )
    entities: list[str] = Field(
        default_factory=list,
        description="Key named entities mentioned in or implied by the query "
        "(people, organizations, concepts, technologies, etc.).",
    )
    query_type: str = Field(
        default="factual",
        description="Classification of query type: 'factual', 'analytical', "
        "'comparative', or 'procedural'.",
    )


class GradeResult(BaseModel):
    """Structured output from the document grading node."""

    is_relevant: bool = Field(
        description="True if the retrieved documents contain information "
        "sufficient to answer the user's question. False if the "
        "documents are off-topic or lack necessary detail."
    )
    reasoning: str = Field(
        description="Brief explanation of why the documents were judged "
        "relevant or irrelevant."
    )


class Citation(BaseModel):
    """A single source citation."""

    chunk_id: str = Field(description="ID of the source chunk")
    page_numbers: list[int] = Field(
        default_factory=list, description="Page numbers where the information appears"
    )
    text_snippet: str = Field(
        description="Brief verbatim excerpt from the source that supports the claim"
    )
    section_title: str = Field(
        default="", description="Section title of the source chunk"
    )


class CitedAnswer(BaseModel):
    """Structured output from the answer generation node."""

    answer: str = Field(
        description="The complete answer to the user's question. Every factual "
        "claim must include an inline citation in the format "
        "[Source N, Page P]. If the documents don't contain enough "
        "information, say so explicitly."
    )
    citations: list[Citation] = Field(
        default_factory=list,
        description="List of all sources cited in the answer.",
    )


class HallucinationCheck(BaseModel):
    """Structured output from the hallucination guard node."""

    is_grounded: bool = Field(
        description="True if every factual claim in the answer can be traced "
        "back to a specific retrieved document. False if any claim "
        "appears to be fabricated or unsupported."
    )
    ungrounded_claims: list[str] = Field(
        default_factory=list,
        description="List of specific claims in the answer that are not "
        "supported by the retrieved documents.",
    )


class RewrittenQuery(BaseModel):
    """Structured output from the query rewriter node."""

    rewritten_query: str = Field(
        description="A reformulated version of the original query designed to "
        "improve retrieval recall. Use synonyms, expand abbreviations, "
        "and add contextual terms."
    )
    reasoning: str = Field(
        description="Brief explanation of what was changed and why."
    )


# ── Prompt Templates ─────────────────────────────────────────────────────

QUERY_ANALYSIS_PROMPT = """You are a query analysis specialist for a document Q&A system.

Analyze the user's question and produce:
1. An optimized search query that will maximize retrieval accuracy.
2. A list of key named entities (people, organizations, concepts, technologies, etc.).
3. A classification of the query type.

Rules:
- The optimized query should be self-contained (resolve any pronouns from context).
- Extract entities even if they are implied, not just explicitly stated.
- If the query is ambiguous, resolve it in the most likely interpretation.

User's Question: {query}

Conversation History (for context resolution):
{chat_history}"""


DOCUMENT_GRADING_PROMPT = """You are a relevance grading specialist. Your job is to evaluate
whether the retrieved documents contain sufficient information to answer the user's question.

User's Question: {query}

Retrieved Documents:
{documents}

Rules:
- A document is relevant if it contains specific facts, data, or explanations that
  directly address the question.
- Partial relevance counts as relevant — even if the documents don't fully answer
  the question, if they provide useful information, grade as relevant.
- Grade as irrelevant ONLY if the documents are completely off-topic or contain
  no information related to the question.
- Consider the relationship between documents — together they may answer a question
  that individually they cannot."""


GENERATION_PROMPT = """You are a precise, citation-grounded document Q&A assistant.
Generate a comprehensive answer to the user's question based ONLY on the provided context.

STRICT CITATION RULES:
1. Every factual claim MUST include an inline citation: [Source N, Page P]
2. Source numbers correspond to the order of the provided context documents.
3. If a claim spans multiple sources, cite all of them.
4. If the provided context does NOT contain enough information to answer the question,
   explicitly state: "The available documents do not contain sufficient information to
   fully answer this question."
5. NEVER fabricate information that is not in the provided context.
6. Use direct quotes sparingly — prefer paraphrasing with citations.

User's Question: {query}

Context Documents:
{context}

Generate a well-structured, comprehensive answer with proper citations."""


HALLUCINATION_CHECK_PROMPT = """You are a hallucination detection specialist. Your job is to
verify that every factual claim in the generated answer is grounded in the retrieved documents.

Generated Answer:
{generation}

Source Documents Used:
{documents}

Rules:
1. Check each factual claim in the answer against the source documents.
2. A claim is "grounded" if it can be directly traced to or reasonably inferred from
   a specific source document.
3. General knowledge statements (e.g., "the sky is blue") do NOT need grounding.
4. If ANY specific factual claim about the documents' content is not traceable to
   a source, mark it as ungrounded.
5. Hedged statements like "the documents suggest..." or "based on the available
   information..." are acceptable."""


QUERY_REWRITE_PROMPT = """You are a query reformulation specialist. The previous search
did not find sufficiently relevant documents for the user's question.

Reformulate the query to improve retrieval performance:
1. Use synonyms and alternative phrasings.
2. Expand abbreviations and acronyms.
3. Add contextual terms that might appear in relevant documents.
4. Consider both technical and layperson phrasings.
5. Make the query more specific if it was too broad, or more general if too narrow.

Original Question: {query}
Previous Search Query: {previous_query}
Attempt Number: {attempt_number} of {max_attempts}

Generate a better search query."""
