# 📋 Enterprise Hybrid GraphRAG Document Q&A System — Interview Preparation Guide

> A comprehensive guide covering **basic concepts → architecture deep-dives → production-grade interview questions** for every component of this system. Includes free resources to study each concept.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Core Concepts — Basics](#2-core-concepts--basics)
3. [Ingestion Pipeline — Questions & Answers](#3-ingestion-pipeline--questions--answers)
4. [Knowledge Graph — Questions & Answers](#4-knowledge-graph--questions--answers)
5. [Retrieval Pipeline — Questions & Answers](#5-retrieval-pipeline--questions--answers)
6. [Agentic RAG Loop — Questions & Answers](#6-agentic-rag-loop--questions--answers)
7. [Production & System Design — Questions & Answers](#7-production--system-design--questions--answers)
8. [LLM & AI Engineering — Questions & Answers](#8-llm--ai-engineering--questions--answers)
9. [Code-Level Deep Dive Questions](#9-code-level-deep-dive-questions)
10. [Scenario-Based / Behavioral Questions](#10-scenario-based--behavioral-questions)
11. [Free Resources to Study](#11-free-resources-to-study)

---

## 1. Project Overview

This project is an **Enterprise Hybrid GraphRAG Document Q&A System** — a production-grade system that allows users to upload multi-page PDF documents and ask questions about their content. It combines multiple AI/ML techniques:

### Architecture at a Glance

```
PDF Upload → Layout-Aware Parsing → Parent-Child Chunking
    ├─→ Dense + BM25 Embedding → Qdrant Vector DB (Hybrid Search)
    └─→ LLM Entity Extraction → Type Normalization → Neo4j Knowledge Graph

User Question → Query Analysis → 3-Way Hybrid Retrieval (Dense + BM25 + KG)
    → Cross-Encoder Reranking → LLM-as-Judge Grading
    → Grounded Generation with Citations → Hallucination Verification → Answer
```

### Key Technologies Used

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Orchestration | **LangGraph** | Stateful agentic workflow with conditional routing |
| Vector Database | **Qdrant** | Dense + sparse (BM25) dual-vector storage with RRF |
| Knowledge Graph | **Neo4j** | Entity-relationship storage and multi-hop traversal |
| Embeddings | **sentence-transformers/all-MiniLM-L6-v2** | 384-dim dense embeddings |
| Sparse Embeddings | **Qdrant/bm25** via FastEmbed | BM25 term-frequency sparse vectors |
| Reranker | **cross-encoder/ms-marco-MiniLM-L-6-v2** | Cross-attention reranking |
| LLM | **OpenAI GPT / Google Gemini** | Generation, grading, extraction |
| PDF Parsing | **Unstructured / pypdf** | Layout-aware document extraction |
| UI | **Streamlit** | Interactive web chat interface |
| Config | **Pydantic Settings** | Type-safe configuration from `.env` |
| Containerization | **Docker Compose** | Neo4j + Qdrant services |

---

## 2. Core Concepts — Basics

### What is RAG (Retrieval-Augmented Generation)?

**Q: Explain RAG in simple terms.**

**A:** RAG is a technique where instead of relying solely on an LLM's training data, you first **retrieve** relevant documents from an external knowledge base, then **augment** the LLM's prompt with those documents, and finally **generate** an answer grounded in the retrieved context. This reduces hallucinations and allows the LLM to answer questions about private/recent data it wasn't trained on.

---

**Q: What problem does RAG solve that fine-tuning doesn't?**

**A:** 
- **Cost**: RAG doesn't require expensive GPU training cycles.
- **Freshness**: Documents can be updated without retraining the model.
- **Hallucination control**: Answers are grounded in retrieved evidence with citations.
- **Privacy**: Sensitive documents stay in your infrastructure, not baked into model weights.
- **Auditability**: You can trace exactly which source documents the answer came from.

---

### What is GraphRAG?

**Q: How does GraphRAG differ from standard RAG?**

**A:** Standard RAG retrieves flat text chunks via vector similarity. **GraphRAG** additionally builds a **knowledge graph** of entities and relationships extracted from documents, then uses graph traversal (e.g., multi-hop Cypher queries) to retrieve structured relational context that pure vector search would miss.

**Example:** If a document says "Microsoft is headquartered in Redmond" and another says "Satya Nadella is CEO of Microsoft", standard RAG might miss the connection. GraphRAG can traverse: `Satya Nadella → CEO_OF → Microsoft → HEADQUARTERED_IN → Redmond` to answer "Where does the CEO of Microsoft work?"

---

### What is Hybrid Search?

**Q: Why combine dense and sparse search instead of using just one?**

**A:**
- **Dense (semantic) search** captures meaning: "automobile" matches "car". But it can miss exact terms.
- **Sparse (BM25/keyword) search** excels at exact term matching: searching "GPT-4" returns documents with that exact string. But it misses synonyms.
- **Hybrid** combines both to get the best of both worlds — semantic understanding + keyword precision.

---

### What are Embeddings?

**Q: What are vector embeddings and why are they important in RAG?**

**A:** Embeddings are dense numerical vectors (e.g., 384 dimensions) that represent the semantic meaning of text. Texts with similar meanings have vectors that are close together in the embedding space (measured by cosine similarity). In RAG, we embed document chunks and queries into the same vector space, then find the most semantically similar chunks to answer a question.

In this project, we use `sentence-transformers/all-MiniLM-L6-v2` which produces 384-dimensional vectors optimized for semantic similarity tasks.

---

## 3. Ingestion Pipeline — Questions & Answers

### Basic Level

**Q: Walk me through the document ingestion pipeline in this system.**

**A:** The pipeline has 5 stages:
1. **PDF Upload**: The Streamlit UI accepts a PDF, which is copied to a local `data/` directory.
2. **Layout-Aware Parsing** (`parser.py`): Uses Unstructured library with `hi_res` strategy (YOLO-based layout detection) to extract structured elements — titles, narrative text, tables, headers — preserving document hierarchy. Falls back to pypdf if Unstructured fails.
3. **Parent-Child Chunking** (`chunker.py`): Groups parsed elements into section-aware parent blocks (~1500 chars), then splits each into child chunks (~400 chars). Children carry a `parent_id` reference back to their parent.
4. **Dual-Vector Indexing**: Child chunks are embedded with both dense (sentence-transformers) and sparse (BM25) vectors and upserted into Qdrant.
5. **Knowledge Graph Extraction**: Parent chunks are processed by an LLM to extract entities and relationships, which are type-normalized and stored in Neo4j.

---

**Q: What is parent-child chunking and why is it used?**

**A:** Parent-child chunking is a hierarchical retrieval strategy:
- **Child chunks (~400 chars)** are small and precise — they're what gets indexed and searched. Small chunks have better embedding precision because they represent a focused concept.
- **Parent chunks (~1500 chars)** are larger and provide full context — when a child chunk matches, we "hydrate" the parent into the LLM context window so the model sees the surrounding paragraphs.

**Why?** It solves the chunk-size dilemma: small chunks give better search precision, but large chunks give the LLM better context for generation. Parent-child gives you both.

**Code reference:** In `chunker.py`, each `DocumentChunk` has a `parent_id` field. Child chunks link to their parent via UUID5 deterministic IDs.

---

**Q: Why use deterministic UUIDs (UUID5) for chunk IDs?**

**A:** Deterministic IDs enable **idempotent re-ingestion**. If you re-ingest the same document, the same chunks get the same IDs, so Qdrant can upsert (update-or-insert) rather than creating duplicates. This is crucial for production systems where documents may be re-processed after updates.

```python
# From chunker.py — UUID5 with a fixed namespace ensures same input → same ID
chunk_id = str(uuid.uuid5(NAMESPACE_CHUNKS, f"{document_id}:{chunk_type}:{index}:{text[:100]}"))
```

---

### Intermediate Level

**Q: How does the layout-aware parser handle different PDF structures (tables, multi-column, headers)?**

**A:** The `UnstructuredPDFParser` uses the `hi_res` strategy which employs a YOLO-based document layout model to detect:
- **Tables**: Extracted as structured HTML, preserving rows/columns
- **Multi-column layouts**: Detected and linearized into reading order
- **Titles/Headers**: Classified as `Title` or `Header` elements, used as section boundaries by the chunker
- **Headers/Footers**: Detected and excluded from content chunking

The parser returns `ParsedElement` objects with `element_type`, `page_number`, and `metadata` fields. The chunker then uses `element_type == "Title"` as section break signals.

---

**Q: How does the chunker create section boundaries?**

**A:** The chunker iterates through parsed elements and starts a **new parent block** when:
1. It encounters a `Title` or `Header` element type
2. The current block exceeds the parent chunk size (1500 chars)
3. A large table is encountered (gets its own dedicated block)

```python
# From chunker.py
if element.element_type in ("Title", "Header") and current_block:
    parent_blocks.append(current_block)
    current_block = []
current_block.append(element)
```

This ensures parent chunks align with document sections rather than splitting mid-paragraph.

---

### Advanced / Production Level

**Q: What would you change if the system needed to handle 10,000+ page technical manuals?**

**A:** Several optimizations:
1. **Streaming parsing**: Process page-by-page instead of loading entire PDF into memory.
2. **Async batch embedding**: Parallelize embedding generation with `asyncio.gather()`.
3. **Incremental ingestion**: Use content hashing to detect changed pages and only re-process those.
4. **Hierarchical chunking expansion**: Add a third "grandchild" level for very fine-grained retrieval.
5. **Table-specific embedding**: Use specialized table embedding models (e.g., TAPAS) for table chunks.
6. **Distributed processing**: Use Celery/Ray for parallel chunk processing.
7. **Chunk deduplication**: Use MinHash/LSH to detect and deduplicate near-identical chunks across re-ingestions.

---

**Q: How would you handle multi-format document ingestion (DOCX, HTML, Markdown) beyond PDF?**

**A:** The system already uses the **Strategy/Protocol pattern** via the `DocumentParser` Protocol:

```python
class DocumentParser(Protocol):
    def parse(self, file_path: str) -> list[ParsedElement]: ...
```

To add new formats:
1. Create `DocxParser`, `HTMLParser`, `MarkdownParser` classes implementing the same Protocol.
2. Use a factory function that selects the parser based on file extension.
3. All parsers output the same `ParsedElement` objects, so downstream chunking/indexing is unchanged.

This is the Open/Closed Principle (SOLID) in action — open for extension, closed for modification.

---

## 4. Knowledge Graph — Questions & Answers

### Basic Level

**Q: What is a knowledge graph and how is it used in this system?**

**A:** A knowledge graph stores information as **entities (nodes)** connected by **relationships (edges)**. For example:
- Entity: `Microsoft` (type: Organization)
- Entity: `Redmond` (type: Location)
- Relationship: `Microsoft → HEADQUARTERED_IN → Redmond`

In this system, the knowledge graph (Neo4j) is used to:
1. **Extract** entities and relationships from document chunks using an LLM
2. **Augment** vector search results with structured relational context
3. **Enable multi-hop reasoning** that pure vector search can't do

---

**Q: What is post-hoc type normalization and why is it needed?**

**A:** When the LLM extracts entities, it generates inconsistent type labels:
- "company", "corporation", "firm", "business" → all mean `Organization`
- "works_for", "employed_by", "works_at" → all mean `WORKS_AT`

**Post-hoc normalization** maps these LLM-generated labels to a canonical set after extraction. This is done in `schemas.py` with dictionaries:

```python
ENTITY_TYPE_NORMALIZATION = {
    "company": "Organization",
    "corporation": "Organization",
    "firm": "Organization",
    # ...
}
```

**Why post-hoc?** Because we use **open-ended extraction** (no schema constraints), which gives the LLM freedom to discover domain-specific entities. Normalizing afterwards gives us consistent graph topology without losing extraction flexibility.

---

### Intermediate Level

**Q: How does the system query the knowledge graph during retrieval?**

**A:** The `Neo4jGraphStore.query_related_context()` method:

1. Takes a list of entity names extracted from the user's query
2. Runs a Cypher query that case-insensitively matches nodes by name
3. Traverses 1-degree neighborhood (configurable depth) to find related entities
4. Returns formatted natural language triples:

```
Microsoft → HEADQUARTERED_IN → Redmond
Microsoft → CEO → Satya Nadella
```

These triples are appended as a supplementary "document" to the vector search results before generation, giving the LLM structured relational context.

```cypher
MATCH (e:__Entity__)-[r]-(neighbor)
WHERE toLower(e.id) IN $entity_names
RETURN e.id, type(r), neighbor.id
LIMIT $max_triples
```

---

**Q: Why use `LLMGraphTransformer` in open-ended mode rather than providing a fixed schema?**

**A:** 
- **Open-ended mode** allows the LLM to discover entity types and relationships naturally from the document content. A legal document will produce different entity types (Clause, Statute, Court) than a biomedical paper (Gene, Protein, Disease).
- **Fixed schema** would require predefined domain knowledge, limiting the system to known entity types.
- The **normalization layer** provides consistency without constraining extraction, giving us the best of both approaches.

---

### Advanced / Production Level

**Q: How would you handle entity resolution (deduplication) in the knowledge graph?**

**A:** The current system stores entities by their LLM-extracted names, which can lead to duplicates ("Microsoft Corp" vs "Microsoft" vs "MSFT"). In production:

1. **Embedding-based matching**: Embed all entity names and cluster by cosine similarity threshold.
2. **Fuzzy string matching**: Use Levenshtein distance or Jaro-Winkler to detect near-matches.
3. **Canonical entity linking**: Link extracted entities to a canonical knowledge base (Wikidata, company registries).
4. **MERGE in Cypher**: The system already uses `MERGE` (not `CREATE`) in Neo4j, which prevents exact-name duplicates.
5. **Periodic reconciliation**: Batch job to merge entities with high embedding similarity.

---

**Q: What are the trade-offs of using Neo4j Community Edition vs Enterprise for this system?**

**A:**
| Feature | Community | Enterprise |
|---------|-----------|-----------|
| Clustering/HA | ❌ | ✅ Causal clustering |
| Role-based access | Basic | Full RBAC |
| Hot backups | ❌ | ✅ |
| Performance | Good | Optimized query planner |
| Cost | Free | Licensed |

For this project, Community Edition is sufficient for development and small-scale production. For enterprise deployment, you'd want clustering for high availability and hot backups for data safety.

---

## 5. Retrieval Pipeline — Questions & Answers

### Basic Level

**Q: Explain the three-way hybrid retrieval strategy.**

**A:** The system retrieves relevant documents using three complementary methods:

1. **Dense Vector Search** (Qdrant): Embeds the query using sentence-transformers and finds semantically similar chunks via cosine similarity. Catches semantic matches ("automobile" ↔ "car").

2. **Sparse BM25 Search** (Qdrant): Uses BM25 term-frequency vectors for keyword-based matching. Catches exact term matches ("GPT-4" ↔ "GPT-4").

3. **Knowledge Graph Context** (Neo4j): Extracts entities from the query, traverses Neo4j for related triples, and appends as structured context. Catches relational/multi-hop answers.

The dense and sparse results are fused using **Reciprocal Rank Fusion (RRF)** at the database level inside Qdrant, then **cross-encoder reranked** before KG context is appended.

---

**Q: What is Reciprocal Rank Fusion (RRF)?**

**A:** RRF is a rank-aggregation method that combines ranked lists from different retrieval methods. For each document, it computes:

$$\text{RRF}(d) = \sum_{r \in \text{rankers}} \frac{1}{k + \text{rank}_r(d)}$$

Where $k$ is a constant (typically 60) that smooths the contribution of each ranker. A document ranked #1 by dense search and #5 by BM25 gets a higher combined score than one ranked #3 by both.

**Why RRF over simple score fusion?** Different retrieval methods produce scores on different scales (cosine similarity 0-1 vs BM25 unbounded). RRF is **score-agnostic** — it only uses rank positions, making it robust to score distribution differences.

---

**Q: What is a Cross-Encoder and how does it differ from a Bi-Encoder?**

**A:**

| | Bi-Encoder | Cross-Encoder |
|---|-----------|--------------|
| **Architecture** | Encodes query and document separately | Encodes query-document pair jointly |
| **Speed** | Fast (pre-compute doc embeddings) | Slow (must score each pair) |
| **Accuracy** | Good | Excellent (full cross-attention) |
| **Use case** | Initial retrieval (thousands of docs) | Reranking (top-N candidates) |

In this system:
- **Bi-Encoder** (`all-MiniLM-L6-v2`): Used in initial retrieval to search thousands of chunks quickly.
- **Cross-Encoder** (`ms-marco-MiniLM-L-6-v2`): Used to rerank the top-N retrieved candidates with higher accuracy.

This two-stage architecture (retrieve-then-rerank) is the standard pattern used by Google, Microsoft, and other search companies.

---

### Intermediate Level

**Q: How does Qdrant implement hybrid search with RRF internally?**

**A:** Qdrant supports hybrid search via its `query_points` API with **named vectors** and **prefetch**:

```python
# From vector_store.py
prefetch = [
    models.Prefetch(query=dense_vector, using="dense", limit=limit * 2),
    models.Prefetch(query=sparse_vector, using="bm25", limit=limit * 2),
]
results = client.query_points(
    collection_name=collection,
    prefetch=prefetch,
    query=models.FusionQuery(fusion=models.Fusion.RRF),
    limit=limit,
)
```

The collection is created with two named vector spaces:
- `"dense"`: 384-dim cosine vectors from sentence-transformers
- `"bm25"`: Sparse vectors from FastEmbed BM25 model (with IDF modifier)

Each prefetch runs independently, then Qdrant fuses the results server-side using RRF.

---

**Q: Why does the system store `parent_text` in Qdrant payloads alongside child chunk text?**

**A:** This implements the **small-to-big retrieval** pattern:
1. **Search** uses child chunks (small, precise embeddings)
2. **Generation** uses parent chunks (large, contextually rich)

By storing `parent_text` in the Qdrant payload, we can retrieve it in a single query without a separate lookup. When a child chunk matches, we return its `parent_text` for the LLM context window, giving the model complete paragraphs and cross-sentence context.

---

### Advanced / Production Level

**Q: How would you evaluate retrieval quality? What metrics would you use?**

**A:** Key metrics for retrieval evaluation:

1. **Recall@K**: What fraction of relevant documents are in the top-K results?
2. **MRR (Mean Reciprocal Rank)**: How high is the first relevant result ranked on average?
3. **NDCG (Normalized Discounted Cumulative Gain)**: Measures ranking quality considering position and relevance grades.
4. **Hit Rate**: For what percentage of queries does the top-K contain at least one relevant document?
5. **Latency (P50/P95/P99)**: How fast are retrieval results returned?

**Evaluation approach:**
1. Create a **golden dataset** of (question, relevant_passages) pairs.
2. Run queries through the retrieval pipeline.
3. Measure retrieval metrics.
4. A/B test different configurations (chunk sizes, RRF parameters, reranker models).
5. Use **RAGAS** or **LangSmith** for automated RAG evaluation.

---

**Q: What would you do if the cross-encoder reranker becomes a latency bottleneck?**

**A:** Several mitigation strategies:

1. **Reduce candidates**: Lower the number of documents sent to the reranker (from 20 → 10).
2. **Distill the cross-encoder**: Train a smaller bi-encoder to mimic cross-encoder scores.
3. **ONNX Runtime**: Export the cross-encoder to ONNX and use optimized inference.
4. **GPU acceleration**: Move reranking to GPU with batched inference.
5. **Caching**: Cache reranker scores for frequently asked query-document pairs.
6. **ColBERT**: Use late-interaction models (ColBERT) which pre-compute document token embeddings and only compute interaction at query time — much faster than full cross-encoders while maintaining most of the accuracy.
7. **Cohere/Jina Reranker APIs**: Use hosted reranker APIs with optimized infrastructure.

---

## 6. Agentic RAG Loop — Questions & Answers

### Basic Level

**Q: What is LangGraph and why is it used instead of LangChain's simple chain?**

**A:** LangGraph is a framework for building **stateful, multi-step AI workflows** as directed graphs. Unlike LangChain's linear chains:

- **Cycles are allowed**: The system can loop back (e.g., rewrite query → re-retrieve → re-grade).
- **Conditional routing**: Different paths based on document relevance or hallucination checks.
- **State management**: TypedDict state flows through nodes and persists via checkpointers.
- **Error recovery**: Nodes can fail and the graph can route to fallback paths.

In this project, LangGraph enables the **self-correcting loop**: if retrieved documents are irrelevant, the system rewrites the query and retries (up to 3 times). If the generated answer hallucinates, it regenerates (up to 2 times).

---

**Q: Walk through the agent loop step-by-step for a user question.**

**A:**

1. **`analyze_query`**: Extracts entities from the question, classifies intent (factual/analytical/comparative/procedural), optimizes the search query using structured LLM output.

2. **`hybrid_retrieve`**: Runs 3-way hybrid retrieval (dense + BM25 via Qdrant RRF + Neo4j KG traversal), then cross-encoder reranking.

3. **`grade_documents`**: LLM-as-Judge evaluates each retrieved chunk for relevance. Outputs a structured `GradeResult` with `is_relevant` boolean and reasoning.

4. **Conditional routing**:
   - If relevant → go to `generate_answer`
   - If not relevant AND retries < 3 → go to `rewrite_query` (which loops back to `hybrid_retrieve`)
   - If not relevant AND retries >= 3 → go to `fallback_generate` (best-effort answer with caveats)

5. **`generate_answer`**: Produces a grounded answer with inline citations `[Source N, Page P]` using structured `CitedAnswer` output.

6. **`check_hallucination`**: Verifies every claim is supported by retrieved evidence. Outputs structured `HallucinationCheck`.

7. **Conditional routing**:
   - If grounded → return final answer
   - If not grounded AND generation attempts < 2 → regenerate
   - If max attempts reached → return current answer

---

### Intermediate Level

**Q: Explain the state management in the LangGraph agent.**

**A:** The agent state is defined as a `TypedDict` called `AgentState` (in `state.py`):

```python
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]  # Chat history with LangGraph reducer
    query: str                                # Original user question
    rewritten_query: str                      # Reformulated query for retries
    extracted_entities: list[str]             # Entities from query analysis
    documents: list[dict]                    # Retrieved & ranked documents
    generation: str                          # Generated answer text
    retrieval_attempts: int                  # Counter for retry loop (max 3)
    generation_attempts: int                 # Counter for hallucination retries (max 2)
    max_retries: int                         # Configurable retry limit
    is_relevant: bool                        # Grading result
    is_grounded: bool                        # Hallucination check result
    citations: list[dict]                    # Structured citation objects
```

The `add_messages` annotation is a LangGraph **reducer** — instead of replacing the messages list, new messages are appended. This enables conversation history accumulation.

Each node function receives the full state, modifies specific fields, and returns a partial update dict. LangGraph merges the updates into the state.

---

**Q: How do the routing functions work?**

**A:** Two routing functions control the graph's conditional edges:

```python
def route_after_grading(state: AgentState) -> str:
    if state.get("is_relevant"):
        return "generate"                # Documents are relevant → generate answer
    if state.get("retrieval_attempts", 0) >= state.get("max_retries", 3):
        return "fallback_generate"       # Max retries hit → best-effort answer
    return "rewrite"                     # Try again with rewritten query

def route_after_hallucination_check(state: AgentState) -> str:
    if state.get("is_grounded"):
        return "grounded"                # Answer is faithful → return to user
    if state.get("generation_attempts", 0) < 2:
        return "retry_generation"        # Not grounded → regenerate
    return "grounded"                    # Max gen retries → accept current answer
```

These are passed to `graph.add_conditional_edges()` to create branching paths in the workflow.

---

### Advanced / Production Level

**Q: What are the failure modes of this agentic loop and how does the system handle them?**

**A:** 

| Failure Mode | How It's Handled |
|---|---|
| Irrelevant retrieval results | Query rewriting + retry (max 3 attempts) |
| Hallucinated answer | Regeneration with explicit grounding instruction (max 2 attempts) |
| LLM structured output parsing failure | Try/except with fallback to raw text extraction |
| Neo4j connection down | Graceful degradation to vector-only retrieval |
| Qdrant connection down | Falls back to local embedded Qdrant instance |
| Infinite loops | Hard-coded retry limits (MAX_RETRIEVAL_RETRIES=3, MAX_GEN_RETRIES=2) |
| Token budget explosion | Parent chunk size limits + bounded retry counts |
| Empty retrieval results | KG-only fallback, then best-effort generation with caveats |

---

**Q: How would you add streaming responses to this agentic system?**

**A:** LangGraph supports streaming natively:

1. **Node-level streaming**: Use `astream_events()` to stream LLM token generation from the `generate_answer` node.
2. **Step-level streaming**: Use `astream()` to stream state updates after each node completes, showing progress (e.g., "Analyzing query...", "Retrieving documents...", "Checking hallucination...").
3. **Streamlit integration**: Use `st.write_stream()` with an async generator that yields tokens.

```python
async for event in graph.astream_events(state, config):
    if event["event"] == "on_chat_model_stream":
        yield event["data"]["chunk"].content
```

---

**Q: How would you implement multi-turn conversation memory in this system?**

**A:** The system already has foundations for this:

1. **LangGraph's `add_messages` reducer** accumulates messages across invocations.
2. **`MemorySaver` checkpointer** persists state between turns.
3. **`_format_chat_history()`** in `nodes.py` formats the last 3 turns for context.

To enhance:
- Use **thread-based sessions** with unique `thread_id` in the config.
- Implement **context window management** — summarize old turns when context gets long.
- Add **coreference resolution** — resolve "it", "they", "that company" to entities from previous turns.
- Consider **PostgreSQL checkpointer** (LangGraph's `PostgresSaver`) for production persistence.

---

## 7. Production & System Design — Questions & Answers

**Q: How would you deploy this system to handle 1000 concurrent users?**

**A:**

1. **Application layer**: Containerize the Streamlit app (or replace with FastAPI for API-first). Deploy on Kubernetes with horizontal pod autoscaling (HPA) based on CPU/request count.

2. **Vector database**: Qdrant supports distributed mode with sharding and replication. Use a managed Qdrant Cloud cluster or self-hosted multi-node setup.

3. **Knowledge graph**: Neo4j Enterprise Edition with causal clustering (3+ node cluster). Use read replicas for query load distribution.

4. **LLM calls**: Use async batch processing, request queuing, and multiple API key rotation. Consider deploying a self-hosted LLM (vLLM/TGI) for cost control at scale.

5. **Caching**: 
   - Redis for query result caching (semantic cache using embedding similarity).
   - CDN for static assets.
   - LLM response caching for repeated/similar queries.

6. **Load balancing**: Nginx/HAProxy in front of app instances.

7. **Monitoring**: Prometheus + Grafana for metrics, ELK stack for logs, LangSmith/Langfuse for LLM observability.

---

**Q: How would you monitor and observe this system in production?**

**A:**

| What to Monitor | Tool | Metrics |
|---|---|---|
| LLM calls | LangSmith / Langfuse | Latency, tokens, cost, success rate |
| Retrieval quality | Custom metrics | Recall@K, MRR, relevance scores |
| Application health | Prometheus + Grafana | CPU, memory, request rate, error rate |
| Vector DB | Qdrant metrics endpoint | Query latency, collection size, index stats |
| Knowledge graph | Neo4j monitoring | Query time, cache hit ratio, node/rel counts |
| User experience | Custom analytics | Query-to-answer latency, user satisfaction |
| Errors | Sentry / ELK | Exception tracking, stack traces |

---

**Q: How would you implement A/B testing for different RAG configurations?**

**A:**
1. Define **experiment variants**: different chunk sizes, embedding models, reranker models, prompt templates.
2. Use **feature flags** (LaunchDarkly/Unleash) to route users to variants.
3. Log all retrieval/generation data with variant IDs to a data warehouse.
4. Measure **key metrics**: answer quality (human eval + LLM-as-Judge), latency, cost, user satisfaction.
5. Use statistical significance testing (chi-squared, Mann-Whitney U) to determine winner.
6. Gradually roll out the winning variant.

---

**Q: Explain the Docker Compose setup and why those specific services are used.**

**A:** The `docker-compose.yml` defines two services:

```yaml
qdrant:
  image: qdrant/qdrant:latest
  ports: ["6333:6333", "6334:6334"]  # REST API + gRPC
  volumes: [qdrant_storage:/qdrant/storage]
  healthcheck: curl -f http://localhost:6333/healthz

neo4j:
  image: neo4j:community
  ports: ["7474:7474", "7687:7687"]  # Browser UI + Bolt protocol
  volumes: [neo4j_data:/data, neo4j_logs:/logs]
  environment:
    - NEO4J_AUTH=neo4j/graphrag_password
    - NEO4J_PLUGINS=["apoc"]  # APOC procedures for advanced operations
```

- **Persistent volumes** ensure data survives container restarts.
- **Health checks** allow Docker to monitor service availability.
- **APOC plugin** enables advanced Neo4j operations (path expansion, text processing).

---

**Q: What security considerations would you address before deploying this to production?**

**A:**
1. **API Key Management**: Use a secrets manager (AWS Secrets Manager, HashiCorp Vault) instead of `.env` files.
2. **Input Sanitization**: Validate PDF uploads (size limits, malware scanning, file type verification).
3. **Prompt Injection**: Sanitize user queries to prevent prompt injection attacks.
4. **Authentication/Authorization**: Add user authentication (OAuth2/OIDC) and role-based document access.
5. **Network Security**: TLS for all service communication, private VPC for databases.
6. **Data Encryption**: Encrypt data at rest in Neo4j and Qdrant.
7. **Rate Limiting**: Prevent abuse of LLM API calls.
8. **Audit Logging**: Log all document access and queries for compliance.
9. **PII Detection**: Scan uploaded documents for PII and handle appropriately.
10. **Neo4j/Qdrant Auth**: Change default credentials and use strong passwords.

---

## 8. LLM & AI Engineering — Questions & Answers

**Q: How does the system support multiple LLM providers (OpenAI and Google Gemini)?**

**A:** The `config.py` uses Pydantic Settings to detect available API keys:

```python
# If GOOGLE_API_KEY is set, use Google Gemini
# If OPENAI_API_KEY is set, use OpenAI
# Priority: Google > OpenAI (configurable)
```

In `nodes.py`, the LLM is lazily initialized based on which key is available:

```python
if settings.GOOGLE_API_KEY:
    llm = ChatGoogleGenerativeAI(model="gemini-pro", ...)
elif settings.OPENAI_API_KEY:
    llm = ChatOpenAI(model="gpt-4", ...)
```

Both providers use LangChain's unified `BaseChatModel` interface, so all downstream code (prompts, structured output, etc.) is provider-agnostic.

---

**Q: What is structured output and why is it used throughout the agent?**

**A:** Structured output forces the LLM to respond in a predefined Pydantic schema rather than free-form text. This project defines 6 structured output schemas in `prompts.py`:

```python
class GradeResult(BaseModel):
    is_relevant: bool
    reasoning: str

class CitedAnswer(BaseModel):
    answer: str
    citations: list[Citation]

class HallucinationCheck(BaseModel):
    is_grounded: bool
    ungrounded_claims: list[str]
```

**Benefits:**
- **Reliable parsing**: No regex/string parsing of LLM output.
- **Type safety**: Boolean fields are actual booleans, not "yes"/"no" strings.
- **Validation**: Pydantic validates the output structure automatically.
- **Routing**: The agent can make deterministic routing decisions based on structured fields.

Used via LangChain's `.with_structured_output(Schema)` method.

---

**Q: How would you reduce the cost of LLM calls in this system?**

**A:**

1. **Smaller models for grading**: Use GPT-4o-mini or Gemini Flash for document grading and hallucination checks (simpler tasks).
2. **Caching**: Cache LLM responses for identical or semantically similar queries.
3. **Prompt optimization**: Shorter, more focused prompts reduce input tokens.
4. **Early termination**: Skip hallucination check if confidence is high.
5. **Batch grading**: Grade all documents in a single LLM call instead of one-per-document.
6. **Local models**: Use open-source models (Llama, Mistral) for simpler tasks via Ollama.
7. **Output token limits**: Set `max_tokens` to prevent unnecessarily verbose answers.
8. **Filter before grading**: Use embedding similarity threshold to pre-filter obviously irrelevant docs.

---

**Q: What is "LLM-as-Judge" and what are its limitations?**

**A:** LLM-as-Judge uses an LLM to evaluate the quality of outputs (document relevance, hallucination, answer quality). In this system:
- **Document Grading**: The LLM judges whether a retrieved chunk is relevant to the query.
- **Hallucination Check**: The LLM verifies whether claims in the answer are supported by evidence.

**Limitations:**
- **Self-consistency bias**: The same LLM may rate its own outputs more favorably.
- **Prompt sensitivity**: Small prompt changes can significantly affect judgments.
- **Cost**: Every evaluation requires an additional LLM call.
- **Calibration**: LLMs may be overconfident in their judgments.
- **Adversarial robustness**: Adversarial inputs can fool the judge.

**Mitigations used in this project:**
- Structured output prevents ambiguous responses.
- Bounded retries prevent infinite grading loops.
- Fallback generation ensures the user always gets an answer.

---

## 9. Code-Level Deep Dive Questions

**Q: Explain the Pydantic Settings pattern used in `config.py`.**

**A:** The `Settings` class inherits from `BaseSettings` (pydantic-settings), which automatically:
1. Loads values from environment variables
2. Falls back to values from a `.env` file
3. Validates types (str, int, etc.) at startup
4. Provides a singleton via `@lru_cache` on `get_settings()`

```python
class Settings(BaseSettings):
    OPENAI_API_KEY: str = ""
    NEO4J_URI: str = "bolt://localhost:7687"
    QDRANT_URL: str = "http://localhost:6333"
    EMBEDDING_DIM: int = 384
    PARENT_CHUNK_SIZE: int = 1500
    CHILD_CHUNK_SIZE: int = 400
    
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
```

**Why this pattern?**
- **Type safety**: Catches misconfiguration at startup (e.g., `EMBEDDING_DIM="abc"` fails validation).
- **Centralized**: All config in one place, not scattered across modules.
- **Testable**: Can override settings in tests without modifying `.env`.
- **12-factor compliance**: Configuration from environment, not code.

---

**Q: How does the `run_query_sync()` function handle Streamlit's event loop conflict?**

**A:** Streamlit runs its own asyncio event loop, which conflicts with LangGraph's async execution. The solution in `graph.py`:

```python
def run_query_sync(query, ...):
    def _run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(run_query(query, ...))
        finally:
            loop.close()
    
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_run)
        return future.result()
```

This creates a **new event loop in a separate thread**, avoiding the "event loop already running" error that occurs when calling `asyncio.run()` inside Streamlit's existing event loop.

---

**Q: Explain the dependency injection pattern in the ingestion pipeline.**

**A:** The `IngestionPipeline` constructor accepts its dependencies as parameters:

```python
class IngestionPipeline:
    def __init__(
        self,
        parser: DocumentParser,          # Protocol — any parser implementation
        vector_store: QdrantVectorStore,  # Injected, not created internally
        graph_extractor: GraphExtractor,
        graph_store: Neo4jGraphStore,
        chunker: Optional[ParentChildChunker] = None,  # Optional with default
    ):
```

**Benefits:**
- **Testability**: Tests inject mocks instead of real database clients.
- **Flexibility**: Swap parsers, vector stores, or graph stores without changing pipeline code.
- **Single Responsibility**: The pipeline orchestrates; it doesn't configure dependencies.
- **Composition Root**: Dependencies are wired together at the top level (in `app.py` or tests).

---

## 10. Scenario-Based / Behavioral Questions

**Q: A user reports that the system gives wrong answers for questions about tables in their PDF. How would you debug this?**

**A:**
1. **Check parsing**: Look at the parsed elements — are tables being extracted as structured HTML or flattened to text? Run the parser on the PDF and inspect `ParsedElement` objects with `element_type == "Table"`.
2. **Check chunking**: Are table chunks being split across parent boundaries? Tables may need custom chunking logic.
3. **Check embeddings**: Embed a table-specific query and check cosine similarity with table chunks. Dense embeddings may not capture tabular structure well.
4. **Check retrieval**: Run the hybrid search and see if table chunks appear in results. BM25 may help if the query contains exact cell values.
5. **Fix**: Consider table-specific embeddings (TAPAS), structured table extraction to markdown, or separate table indexing with schema-aware search.

---

**Q: The system is taking 15+ seconds per query. How would you reduce latency?**

**A:**
1. **Profile each step**: Instrument the LangGraph nodes with timing.
   - Query analysis: ~1-2s (LLM call)
   - Hybrid retrieval: ~0.5-1s (Qdrant + Neo4j)
   - Cross-encoder reranking: ~0.5-2s (model inference)
   - Document grading: ~1-2s (LLM call)
   - Generation: ~2-5s (LLM call)
   - Hallucination check: ~1-2s (LLM call)

2. **Optimize LLM calls**: Use faster models (GPT-4o-mini, Gemini Flash) for grading/hallucination. Parallelize independent LLM calls.
3. **Cache**: Implement semantic query caching.
4. **Skip steps conditionally**: If embeddings similarity is very high (>0.9), skip grading.
5. **Reduce reranking candidates**: Lower the number from 20 to 10.
6. **GPU reranker**: Move cross-encoder to GPU.
7. **Streaming**: Stream the generation to improve perceived latency.

---

**Q: A new team member asks why you chose Qdrant over Pinecone or Weaviate. How do you explain?**

**A:**
| Criteria | Qdrant | Pinecone | Weaviate |
|---|---|---|---|
| **Hybrid search** | ✅ Native dense+sparse with RRF | ❌ Dense only (or alpha param) | ✅ BM25 + vector |
| **Self-hosted** | ✅ Docker, binary, cloud | ❌ Cloud-only | ✅ Docker, cloud |
| **Named vectors** | ✅ Multiple vector spaces per point | ❌ | ❌ |
| **Sparse vectors** | ✅ Native BM25 support | ✅ (sparse-dense) | ✅ (BM25 built-in) |
| **Local dev** | ✅ In-memory / embedded mode | ❌ | ✅ |
| **Open source** | ✅ Apache 2.0 | ❌ Proprietary | ✅ BSD-3 |
| **Performance** | Excellent (Rust-based) | Excellent | Good |

Key differentiators for this project: Native RRF fusion, named vector spaces (dense + sparse in one collection), embedded mode for development, and fully open-source.

---

## 11. Free Resources to Study

### RAG (Retrieval-Augmented Generation)

| Resource | Link | Description |
|----------|------|-------------|
| LangChain RAG Tutorial | [python.langchain.com/docs/tutorials/rag](https://python.langchain.com/docs/tutorials/rag/) | Official step-by-step RAG tutorial |
| RAG From Scratch (DeepLearning.AI) | [youtube.com/playlist?list=PLfaIDFEXuae2LXbO1_PKyVJiQ23ZztA0x](https://www.youtube.com/playlist?list=PLfaIDFEXuae2LXbO1_PKyVJiQ23ZztA0x) | Lance Martin's 14-part video series covering RAG from basics to advanced |
| Building RAG Agents with LLMs (NVIDIA) | [learn.nvidia.com/courses/course?course_id=course-v1:NVIDIA+S-FX-16+v1](https://learn.nvidia.com/courses/course?course_id=course-v1:NVIDIA+S-FX-16+v1) | Free NVIDIA course on building RAG agents |
| Anthropic RAG Cookbook | [github.com/anthropics/anthropic-cookbook](https://github.com/anthropics/anthropic-cookbook) | Practical RAG patterns and recipes |

### LangGraph & Agentic AI

| Resource | Link | Description |
|----------|------|-------------|
| LangGraph Official Docs | [langchain-ai.github.io/langgraph](https://langchain-ai.github.io/langgraph/) | Official documentation with tutorials |
| LangGraph Academy (Free) | [academy.langchain.com](https://academy.langchain.com/) | Free courses: Introduction to LangGraph, Agentic RAG |
| AI Agents in LangGraph (DeepLearning.AI) | [deeplearning.ai/short-courses/ai-agents-in-langgraph](https://www.deeplearning.ai/short-courses/ai-agents-in-langgraph/) | Free short course by Harrison Chase |
| LangGraph Corrective RAG Example | [langchain-ai.github.io/langgraph/tutorials/rag/langgraph_crag](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_crag/) | Self-correcting RAG tutorial (similar to this project) |

### Vector Databases & Embeddings

| Resource | Link | Description |
|----------|------|-------------|
| Qdrant Documentation | [qdrant.tech/documentation](https://qdrant.tech/documentation/) | Official Qdrant docs with hybrid search guide |
| Qdrant Hybrid Search Tutorial | [qdrant.tech/articles/hybrid-search](https://qdrant.tech/articles/hybrid-search/) | Detailed hybrid search + RRF explanation |
| Sentence-Transformers Docs | [sbert.net](https://www.sbert.net/) | Official docs for embedding models & cross-encoders |
| MTEB Leaderboard | [huggingface.co/spaces/mteb/leaderboard](https://huggingface.co/spaces/mteb/leaderboard) | Benchmark leaderboard for embedding models |
| Pinecone Learning Center | [pinecone.io/learn](https://www.pinecone.io/learn/) | Free articles on vector search, embeddings, and RAG |

### Knowledge Graphs & Neo4j

| Resource | Link | Description |
|----------|------|-------------|
| Neo4j GraphAcademy (Free) | [graphacademy.neo4j.com](https://graphacademy.neo4j.com/) | Free certified courses on Cypher, graph modeling, and GDS |
| GraphRAG Paper (Microsoft) | [arxiv.org/abs/2404.16130](https://arxiv.org/abs/2404.16130) | Original GraphRAG research paper |
| LangChain Neo4j Integration | [python.langchain.com/docs/integrations/graphs/neo4j_cypher](https://python.langchain.com/docs/integrations/graphs/neo4j_cypher/) | LangChain + Neo4j integration guide |
| Knowledge Graphs for RAG (DeepLearning.AI) | [deeplearning.ai/short-courses/knowledge-graphs-rag](https://www.deeplearning.ai/short-courses/knowledge-graphs-rag/) | Free short course on KG-enhanced RAG |

### Cross-Encoder Reranking

| Resource | Link | Description |
|----------|------|-------------|
| SBERT Cross-Encoders Guide | [sbert.net/examples/applications/cross-encoder](https://www.sbert.net/examples/applications/cross-encoder/README.html) | Official guide on bi-encoder vs cross-encoder |
| MS MARCO Cross-Encoder Models | [huggingface.co/cross-encoder](https://huggingface.co/cross-encoder) | Pre-trained cross-encoder models on HuggingFace |
| Retrieve & Re-Rank Tutorial | [sbert.net/examples/applications/retrieve_rerank](https://www.sbert.net/examples/applications/retrieve_rerank/README.html) | Two-stage retrieval pipeline tutorial |

### LLM Engineering & Prompt Engineering

| Resource | Link | Description |
|----------|------|-------------|
| Prompt Engineering Guide | [promptingguide.ai](https://www.promptingguide.ai/) | Comprehensive free guide to prompt engineering |
| OpenAI Prompt Engineering | [platform.openai.com/docs/guides/prompt-engineering](https://platform.openai.com/docs/guides/prompt-engineering) | Official OpenAI best practices |
| LLM University (Cohere) | [cohere.com/llmu](https://cohere.com/llmu) | Free comprehensive LLM course |
| Structured Output (OpenAI) | [platform.openai.com/docs/guides/structured-outputs](https://platform.openai.com/docs/guides/structured-outputs) | Guide to JSON schema/structured output from LLMs |

### Python & Software Engineering

| Resource | Link | Description |
|----------|------|-------------|
| Pydantic v2 Documentation | [docs.pydantic.dev](https://docs.pydantic.dev/) | Data validation and settings management |
| FastAPI + Pydantic Tutorial | [fastapi.tiangolo.com/tutorial](https://fastapi.tiangolo.com/tutorial/) | Modern Python API development |
| Python Design Patterns | [refactoring.guru/design-patterns/python](https://refactoring.guru/design-patterns/python) | Free design patterns guide with Python examples |
| Real Python — Async IO | [realpython.com/async-io-python](https://realpython.com/async-io-python/) | Understanding asyncio in Python |
| Pytest Documentation | [docs.pytest.org](https://docs.pytest.org/) | Testing framework used in this project |

### Docker & DevOps

| Resource | Link | Description |
|----------|------|-------------|
| Docker Official Getting Started | [docs.docker.com/get-started](https://docs.docker.com/get-started/) | Official Docker tutorial |
| Docker Compose Docs | [docs.docker.com/compose](https://docs.docker.com/compose/) | Compose file reference and tutorials |
| 12-Factor App | [12factor.net](https://12factor.net/) | Best practices for cloud-native applications |

### Streamlit

| Resource | Link | Description |
|----------|------|-------------|
| Streamlit Documentation | [docs.streamlit.io](https://docs.streamlit.io/) | Official docs and API reference |
| Streamlit Chat Elements | [docs.streamlit.io/library/api-reference/chat](https://docs.streamlit.io/library/api-reference/chat) | Chat UI components used in this project |
| 30 Days of Streamlit | [30days.streamlit.app](https://30days.streamlit.app/) | Free daily challenges to learn Streamlit |

### System Design & ML System Design

| Resource | Link | Description |
|----------|------|-------------|
| ML System Design (Stanford) | [stanford-cs329s.github.io](https://stanford-cs329s.github.io/) | Stanford's ML Systems Design course materials |
| Designing ML Systems (Chip Huyen) | [huyenchip.com/machine-learning-systems-design](https://huyenchip.com/machine-learning-systems-design/toc.html) | Free online book on ML systems |
| System Design Primer | [github.com/donnemartin/system-design-primer](https://github.com/donnemartin/system-design-primer) | Comprehensive system design study resource |
| MLOps Guide | [ml-ops.org](https://ml-ops.org/) | Free MLOps best practices and patterns |

### Research Papers (Free on arXiv)

| Paper | Link | Why Read It |
|-------|------|-------------|
| Retrieval-Augmented Generation (Lewis et al.) | [arxiv.org/abs/2005.11401](https://arxiv.org/abs/2005.11401) | The original RAG paper |
| From Local to Global: A Graph RAG Approach | [arxiv.org/abs/2404.16130](https://arxiv.org/abs/2404.16130) | Microsoft's GraphRAG paper |
| Reciprocal Rank Fusion | [plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) | Original RRF paper |
| Sentence-BERT (Reimers & Gurevych) | [arxiv.org/abs/1908.10084](https://arxiv.org/abs/1908.10084) | Foundation of sentence embeddings |
| ColBERT: Late Interaction | [arxiv.org/abs/2004.12832](https://arxiv.org/abs/2004.12832) | Efficient reranking alternative |
| Self-RAG (Asai et al.) | [arxiv.org/abs/2310.11511](https://arxiv.org/abs/2310.11511) | Self-reflective RAG (similar to this project's approach) |

### YouTube Channels & Playlists

| Channel | Focus |
|---------|-------|
| [James Briggs](https://www.youtube.com/@jamesbriggs) | RAG, vector DBs, LangChain tutorials |
| [Sam Witteveen](https://www.youtube.com/@samwitteveen) | LangChain, agents, RAG deep dives |
| [Underfitted](https://www.youtube.com/@underfitted) | ML engineering and MLOps |
| [ArXiv Dives (Yannic Kilcher)](https://www.youtube.com/@YannicKilcher) | Research paper explanations |
| [DeepLearning.AI](https://www.youtube.com/@Deeplearningai) | Free short courses on AI/ML |

---

## 📝 Quick Revision Cheat Sheet

```
System Flow:
PDF → Parse → Chunk (Parent-Child) → Embed (Dense+BM25) → Index (Qdrant+Neo4j)
Query → Analyze → Retrieve (3-way) → Rerank → Grade → Generate → Verify → Answer

Key Design Decisions:
✅ Parent-child chunking → search precision + generation context
✅ Open-ended KG extraction + post-hoc normalization → domain flexibility
✅ RRF fusion → score-agnostic rank aggregation
✅ Cross-encoder reranking → eliminate false-positive similarities
✅ Bounded retry loops → prevent infinite recursion / token burn
✅ Structured LLM output → reliable routing decisions
✅ Dependency injection → testability + flexibility
✅ Dual LLM provider support → vendor independence
✅ Graceful degradation → works even if Neo4j/Qdrant is down
✅ Deterministic UUIDs → idempotent re-ingestion

Numbers to Remember:
• Dense embedding dim: 384 (all-MiniLM-L6-v2)
• Parent chunk: ~1500 chars
• Child chunk: ~400 chars
• Chunk overlap: 100 chars
• Max retrieval retries: 3
• Max generation retries: 2
• Cross-encoder top_n: 5
• Qdrant RRF k parameter: 60 (default)
```

---

> **Tip:** For interview preparation, practice explaining the system architecture on a whiteboard. Start with the high-level flow, then drill into any component the interviewer asks about. The most impressive answers connect design decisions to trade-offs (e.g., "We chose parent-child chunking because it solves the chunk-size dilemma: small chunks for search precision, large chunks for generation context").

---

*Last updated: November 2026*

