"""Streamlit application for the Hybrid GraphRAG Document Q&A system.

Provides:
  - Sidebar: PDF upload with drag-and-drop, ingestion progress, system status.
  - Main area: Chat-style Q&A interface with citation rendering.
  - Session state: Maintains conversation history and LangGraph thread IDs.
  - Error handling: Graceful degradation if Neo4j/Qdrant are unreachable.

Run with:
    streamlit run src/ui/app.py
"""

from __future__ import annotations

import asyncio
import logging
import sys
import tempfile
from pathlib import Path
from uuid import uuid4

import streamlit as st

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Settings, get_settings  # noqa: E402

logger = logging.getLogger(__name__)

# ── Page Configuration ───────────────────────────────────────────────────

st.set_page_config(
    page_title="GraphRAG Document Q&A",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global font */
    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main header gradient */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .sub-header {
        color: #6b7280;
        font-size: 1rem;
        font-weight: 400;
        margin-bottom: 2rem;
    }

    /* Chat message styling */
    .stChatMessage {
        border-radius: 12px;
        margin-bottom: 1rem;
    }

    /* Citation expander */
    .citation-box {
        background: linear-gradient(135deg, #f0f4ff 0%, #f5f0ff 100%);
        border-left: 3px solid #667eea;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
        font-size: 0.85rem;
    }

    .citation-header {
        color: #667eea;
        font-weight: 600;
        margin-bottom: 4px;
    }

    .citation-snippet {
        color: #4b5563;
        font-style: italic;
    }

    /* Status indicators */
    .status-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 6px;
    }
    .status-online { background-color: #10b981; }
    .status-offline { background-color: #ef4444; }

    /* Sidebar styling */
    .sidebar-section {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 16px;
    }

    /* Ingestion stats cards */
    .stat-card {
        background: white;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .stat-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #667eea;
    }
    .stat-label {
        font-size: 0.75rem;
        color: #9ca3af;
        text-transform: uppercase;
    }

    /* Smooth animations */
    .stButton > button {
        transition: all 0.2s ease;
        border-radius: 8px;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# ── Session State Initialization ─────────────────────────────────────────


def init_session_state() -> None:
    """Initialize all session state variables."""
    defaults = {
        "chat_history": [],
        "thread_id": uuid4().hex[:12],
        "agent_graph": None,
        "ingestion_pipeline": None,
        "settings_loaded": False,
        "services_status": {"qdrant": False, "neo4j": False},
        "ingestion_results": [],
        "components_initialized": False,
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


init_session_state()


# ── Component Initialization ─────────────────────────────────────────────


@st.cache_resource
def initialize_components(_settings: Settings):
    """Initialize all system components (cached across reruns).

    Returns a dict with all initialized components, or None values
    for components that failed to initialize.
    """
    components = {
        "vector_store": None,
        "graph_store": None,
        "graph_extractor": None,
        "reranker": None,
        "agent_graph": None,
        "ingestion_pipeline": None,
    }

    # Vector Store
    try:
        from src.retrieval.vector_store import QdrantVectorStore
        components["vector_store"] = QdrantVectorStore(_settings)
        components["vector_store"].ensure_collection()
    except Exception as e:
        logger.error("Failed to initialize Qdrant: %s", e)

    # Graph Store
    try:
        from src.graph.store import Neo4jGraphStore
        components["graph_store"] = Neo4jGraphStore(_settings)
    except Exception as e:
        logger.error("Failed to initialize Neo4j: %s", e)

    # Graph Extractor
    try:
        from src.graph.extractor import GraphExtractor
        components["graph_extractor"] = GraphExtractor(_settings)
    except Exception as e:
        logger.error("Failed to initialize GraphExtractor: %s", e)

    # Reranker
    try:
        from src.retrieval.reranker import CrossEncoderReranker
        components["reranker"] = CrossEncoderReranker(_settings)
    except Exception as e:
        logger.error("Failed to initialize Reranker: %s", e)

    # Agent Graph
    try:
        from src.agent.graph import build_agent_graph
        components["agent_graph"] = build_agent_graph(
            vector_store=components["vector_store"],
            graph_store=components["graph_store"],
            reranker=components["reranker"],
            settings=_settings,
        )
    except Exception as e:
        logger.error("Failed to build agent graph: %s", e)

    # Ingestion Pipeline
    try:
        from src.ingestion.chunker import ParentChildChunker
        from src.ingestion.parser import UnstructuredPDFParser
        from src.ingestion.pipeline import IngestionPipeline

        components["ingestion_pipeline"] = IngestionPipeline(
            parser=UnstructuredPDFParser(),
            chunker=ParentChildChunker(
                parent_chunk_size=_settings.parent_chunk_size,
                child_chunk_size=_settings.child_chunk_size,
                chunk_overlap=_settings.chunk_overlap,
            ),
            vector_store=components["vector_store"],
            graph_extractor=components["graph_extractor"],
            graph_store=components["graph_store"],
        )
    except Exception as e:
        logger.error("Failed to initialize IngestionPipeline: %s", e)

    return components


def check_services(components: dict) -> dict[str, bool]:
    """Check which backend services are available."""
    status = {"qdrant": False, "neo4j": False}

    if components.get("vector_store"):
        try:
            info = components["vector_store"].get_collection_info()
            status["qdrant"] = "error" not in info
        except Exception:
            pass

    if components.get("graph_store"):
        try:
            components["graph_store"].get_graph_stats()
            status["neo4j"] = True
        except Exception:
            pass

    return status


# ── Sidebar ──────────────────────────────────────────────────────────────


def render_sidebar(components: dict) -> None:
    """Render the sidebar with document upload, status, and stats."""
    with st.sidebar:
        st.markdown("### 📄 Document Management")

        # ── Service Status ───────────────────────────────────────────
        services = check_services(components)
        st.session_state.services_status = services

        col1, col2 = st.columns(2)
        with col1:
            dot = "status-online" if services["qdrant"] else "status-offline"
            label = "Online" if services["qdrant"] else "Offline"
            st.markdown(
                f'<span class="status-dot {dot}"></span>Qdrant: {label}',
                unsafe_allow_html=True,
            )
        with col2:
            dot = "status-online" if services["neo4j"] else "status-offline"
            label = "Online" if services["neo4j"] else "Offline"
            st.markdown(
                f'<span class="status-dot {dot}"></span>Neo4j: {label}',
                unsafe_allow_html=True,
            )

        if not services["neo4j"]:
            st.caption("ℹ️ *Running in Hybrid Vector mode (Dense + BM25 RRF + Reranker). Start Neo4j in Docker to enable graph triples.*")

        st.divider()

        # ── Document Upload ──────────────────────────────────────────
        uploaded_file = st.file_uploader(
            "Upload a PDF document",
            type=["pdf"],
            help="Upload a PDF to ingest into the knowledge base.",
            key="pdf_uploader",
        )

        if uploaded_file is not None:
            st.info(f"📎 **{uploaded_file.name}** ({uploaded_file.size / 1024:.1f} KB)")

            if st.button("🚀 Ingest Document", use_container_width=True, type="primary"):
                pipeline = components.get("ingestion_pipeline")
                if pipeline is None:
                    st.error("❌ Ingestion pipeline not available. Check service connections.")
                    return

                with st.spinner("Parsing & indexing document..."):
                    progress_bar = st.progress(0, text="Parsing PDF...")

                    # Save uploaded file to temp location
                    with tempfile.NamedTemporaryFile(
                        suffix=".pdf", delete=False
                    ) as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = Path(tmp.name)

                    try:
                        progress_bar.progress(20, text="Extracting structure...")

                        # Run ingestion (synchronous)
                        result = pipeline.ingest(tmp_path, document_id=uuid4().hex[:12])

                        progress_bar.progress(100, text="Complete!")

                        if result.success:
                            st.success(
                                f"✅ **Ingestion complete!**\n"
                                f"- {result.num_parent_chunks} sections\n"
                                f"- {result.num_child_chunks} search chunks\n"
                                f"- {result.num_entities} entities\n"
                                f"- {result.num_relations} relationships"
                            )
                            st.session_state.ingestion_results.append(result)
                        else:
                            st.warning(
                                f"⚠️ Ingestion completed with errors:\n"
                                + "\n".join(f"- {e}" for e in result.errors)
                            )

                    except Exception as e:
                        st.error(f"❌ Ingestion failed: {e}")
                    finally:
                        tmp_path.unlink(missing_ok=True)

        st.divider()

        # ── Ingestion History ────────────────────────────────────────
        if st.session_state.ingestion_results:
            st.markdown("### 📊 Ingested Documents")
            for result in st.session_state.ingestion_results:
                with st.expander(f"📄 {result.file_name}", expanded=False):
                    cols = st.columns(3)
                    cols[0].metric("Chunks", result.num_child_chunks)
                    cols[1].metric("Entities", result.num_entities)
                    cols[2].metric("Relations", result.num_relations)

        st.divider()

        # ── Session Controls ─────────────────────────────────────────
        st.markdown("### ⚙️ Session")
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.thread_id = uuid4().hex[:12]
            st.rerun()

        st.caption(f"Thread: `{st.session_state.thread_id}`")


# ── Citation Renderer ────────────────────────────────────────────────────


def render_citations(citations: list[dict]) -> None:
    """Render citations in an expandable section with styled boxes."""
    if not citations:
        return

    with st.expander(f"📎 Sources ({len(citations)} citations)", expanded=False):
        for i, cite in enumerate(citations, 1):
            pages = cite.get("page_numbers", [])
            page_str = f"Pages {min(pages)}–{max(pages)}" if pages else "Page unknown"
            section = cite.get("section_title", "")
            snippet = cite.get("text_snippet", "")

            header = f"Source {i}"
            if section:
                header += f" — {section}"

            st.markdown(
                f"""<div class="citation-box">
                    <div class="citation-header">📖 {header} | {page_str}</div>
                    <div class="citation-snippet">"{snippet[:200]}{'...' if len(snippet) > 200 else ''}"</div>
                </div>""",
                unsafe_allow_html=True,
            )


# ── Main Chat Interface ─────────────────────────────────────────────────


def render_chat(components: dict) -> None:
    """Render the main chat interface."""
    # Header
    st.markdown('<div class="main-header">🔍 GraphRAG Document Q&A</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Ask questions about your documents — powered by hybrid retrieval, '
        "knowledge graphs, and self-correcting AI agents.</div>",
        unsafe_allow_html=True,
    )

    # Chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("citations"):
                render_citations(msg["citations"])

    # Chat input
    if query := st.chat_input("Ask a question about your documents...", key="chat_input"):
        # Display user message
        with st.chat_message("user"):
            st.markdown(query)
        st.session_state.chat_history.append({"role": "user", "content": query})

        # Generate response
        agent_graph = components.get("agent_graph")
        if agent_graph is None:
            with st.chat_message("assistant"):
                st.error(
                    "⚠️ The agent is not initialized. Please check that your `.env` has a valid `GOOGLE_API_KEY`."
                )
            return

        with st.chat_message("assistant"):
            with st.spinner("🧠 Thinking..."):
                try:
                    from src.agent.graph import run_query_sync

                    final_state = run_query_sync(
                        graph=agent_graph,
                        query=query,
                        thread_id=st.session_state.thread_id,
                    )

                    answer = final_state.get("generation", "No answer generated.")
                    citations = final_state.get("citations", [])
                    attempts = final_state.get("retrieval_attempts", 0)

                    st.markdown(answer)
                    render_citations(citations)

                    # Show retrieval metadata
                    if attempts > 1:
                        st.caption(f"ℹ️ Query required {attempts} retrieval attempts.")

                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer,
                        "citations": citations,
                    })

                except Exception as e:
                    error_msg = f"❌ Error processing your query: {str(e)}"
                    st.error(error_msg)
                    logger.error("Query processing error: %s", e, exc_info=True)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": error_msg,
                    })


# ── Main App ─────────────────────────────────────────────────────────────


def main() -> None:
    """Main application entry point."""
    # Load settings
    try:
        settings = get_settings()
    except Exception as e:
        st.error(
            f"⚠️ **Configuration Error**: {e}\n\n"
            "Please copy `.env.example` to `.env` and fill in your `OPENAI_API_KEY`."
        )
        st.stop()

    # Initialize components
    with st.spinner("Initializing system components..."):
        components = initialize_components(settings)

    # Render UI
    render_sidebar(components)
    render_chat(components)


if __name__ == "__main__":
    main()
