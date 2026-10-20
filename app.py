"""
PDF Document Summarization & Insight Agent (UC05)
Application Entry Point & Modern Streamlit User Interface.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

from src.config import config
from src.utils.logger import setup_logger

logger = setup_logger("app_entrypoint", config.log_level)


def load_ingestion_manifest_summary() -> dict:
    """Loads brief summary of the ingestion manifest if generated."""
    manifest_file = config.extracted_data_dir / "manifest.json"
    if not manifest_file.exists():
        return {
            "ingested": False,
            "total_processed": 0,
            "total_pages": 0,
            "total_ocr_needed": 0,
            "manifest_file": str(manifest_file),
        }
    try:
        data = json.loads(manifest_file.read_text(encoding="utf-8"))
        return {
            "ingested": True,
            "total_processed": data.get("total_processed", 0) + data.get("total_skipped", 0),
            "total_pages": data.get("total_pages", 0),
            "total_ocr_needed": data.get("total_ocr_needed", 0),
            "manifest_file": str(manifest_file),
        }
    except Exception:
        return {
            "ingested": False,
            "total_processed": 0,
            "total_pages": 0,
            "total_ocr_needed": 0,
            "manifest_file": str(manifest_file),
        }


def load_catalog_summary() -> dict:
    """Loads brief summary of the document catalog if generated."""
    if not config.catalog_file.exists():
        return {"catalog_built": False, "analyzed_count": 0, "categories": {}}
    try:
        data = json.loads(config.catalog_file.read_text(encoding="utf-8"))
        return {
            "catalog_built": True,
            "analyzed_count": data.get("analyzed_count", 0),
            "categories": data.get("categories", {}),
            "total_documents": data.get("total_documents", 0),
        }
    except Exception:
        return {"catalog_built": False, "analyzed_count": 0, "categories": {}}


def load_index_summary_info() -> dict:
    """Loads vector index status if generated."""
    manifest_file = config.chroma_persist_dir / "index_manifest.json"
    if not manifest_file.exists():
        return {"indexed": False, "total_chunks": 0, "indexed_docs": 0}
    try:
        data = json.loads(manifest_file.read_text(encoding="utf-8"))
        total_chunks = sum(item.get("chunk_count", 0) for item in data.values())
        return {
            "indexed": True,
            "total_chunks": total_chunks,
            "indexed_docs": len(data),
        }
    except Exception:
        return {"indexed": False, "total_chunks": 0, "indexed_docs": 0}


def render_streamlit_ui():
    """Renders the complete Streamlit multi-view interface."""
    import streamlit as st

    from src.ui.analysis_view import render_analysis_view
    from src.ui.chat_view import render_chat_view
    from src.ui.dashboard_view import render_dashboard_view
    from src.ui.explorer_view import render_explorer_view
    from src.ui.state import init_session_state, load_system_metrics
    from src.ui.styles import get_custom_css, render_header

    st.set_page_config(
        page_title="PDF Insight Agent (UC05)",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Inject modern theme styling
    st.markdown(get_custom_css(), unsafe_allow_html=True)

    # Initialize persistent state
    init_session_state()
    metrics = load_system_metrics()

    # Top Brand Header
    render_header(
        title="📄 PDF Document Summarization & Insight Agent",
        subtitle="UC05 — AI-Powered Multi-Domain Document Analysis, Semantic Retrieval & Grounded Insights",
    )

    # -------------------------------------------------------------------------
    # Sidebar Navigation & Settings
    # -------------------------------------------------------------------------
    with st.sidebar:
        st.markdown("### 🧭 Navigation")

        views = [
            "📊 Dashboard",
            "📑 Document Explorer",
            "💬 AI Chat",
            "📈 Corpus Analysis",
        ]

        # Sync selection with session state
        nav_keys = {
            "📊 Dashboard": "Dashboard",
            "📑 Document Explorer": "Document Explorer",
            "💬 AI Chat": "AI Chat",
            "📈 Corpus Analysis": "Corpus Analysis",
        }
        rev_keys = {v: k for k, v in nav_keys.items()}

        current_view_label = rev_keys.get(st.session_state.current_view, "📊 Dashboard")
        selected_nav = st.radio(
            "Select View",
            options=views,
            index=views.index(current_view_label) if current_view_label in views else 0,
            label_visibility="collapsed",
        )
        st.session_state.current_view = nav_keys[selected_nav]

        st.markdown("---")
        st.markdown("### ⚙️ System Status")

        if metrics["api_key_configured"]:
            st.success("🟢 Gemini API: Connected")
        else:
            st.warning("🟡 Gemini API: Missing Key")

        st.caption(f"**Embedding:** `{metrics['embedding_model']}`")
        st.caption(f"**ChromaDB:** `{metrics['indexed_chunks']:,} chunks`")
        st.caption(f"**Corpus:** `{metrics['processed']} PDFs ({metrics['total_pages']} pages)`")

        # Optional session key input (kept in-memory, never persisted to disk or logged)
        with st.expander("🔑 Session API Key Override"):
            st.caption("Provide an API key for this browser session without modifying `.env`:")
            session_key = st.text_input(
                "Gemini API Key",
                type="password",
                placeholder="AIzaSy...",
                key="session_api_key_input",
            )
            if st.button("Apply Key to Session", use_container_width=True):
                if session_key.strip():
                    import os
                    os.environ["GEMINI_API_KEY"] = session_key.strip()
                    config.gemini_api_key = session_key.strip()
                    st.cache_resource.clear()
                    st.cache_data.clear()
                    st.success("Session API key applied!")
                    st.rerun()

        st.markdown("---")
        st.caption("v0.1.0 • Antigravity AI Engine")

    # -------------------------------------------------------------------------
    # View Router
    # -------------------------------------------------------------------------
    if st.session_state.current_view == "Dashboard":
        render_dashboard_view()
    elif st.session_state.current_view == "Document Explorer":
        render_explorer_view()
    elif st.session_state.current_view == "AI Chat":
        render_chat_view()
    elif st.session_state.current_view == "Corpus Analysis":
        render_analysis_view()


def main():
    """CLI and execution entry point."""
    status = config.get_setup_status()
    ingest_info = load_ingestion_manifest_summary()
    catalog_info = load_catalog_summary()
    index_info = load_index_summary_info()

    print("=" * 65)
    print(f" {status['app_name']} (v{status['version']})")
    print("=" * 65)
    print(f"Environment:         {status['environment']}")
    print(f"PDF Dataset Dir:     {status['pdf_dataset_dir']}")
    print(f"PDFs Detected:       {status['pdf_count']}")
    if ingest_info["ingested"]:
        print(f"Ingestion Status:    Extracted ({ingest_info['total_processed']} PDFs, {ingest_info['total_pages']} pages)")
    else:
        print("Ingestion Status:    Pending (run: python -m scripts.ingest_pdfs)")
    if index_info["indexed"]:
        print(f"Vector Store Status: Indexed ({index_info['total_chunks']} chunks across {index_info['indexed_docs']} PDFs)")
    else:
        print("Vector Store Status: Pending (run: python -m scripts.build_index)")
    if catalog_info["catalog_built"]:
        print(f"Catalog Status:      Analyzed ({catalog_info['analyzed_count']} PDFs in {len(catalog_info['categories'])} domains)")
    else:
        print("Catalog Status:      Pending (run: python -m scripts.analyze_documents)")
    print(f"LLM Provider:        {status['llm_provider']} ({status['gemini_model']})")
    print(f"LLM Key Configured:  {status['gemini_api_key_configured']}")
    print(f"Status Note:         {status['gemini_status_message']}")
    print(f"Embedding Model:     {status['embedding_model']}")
    print(f"Chroma Storage:      {status['chroma_persist_dir']}")
    print(f"Catalog Storage:     {config.catalog_dir}")
    print("=" * 65)
    print("To launch the Streamlit interface, run:")
    print("  streamlit run app.py")
    print("=" * 65)


if __name__ == "__main__":
    if "streamlit" in sys.modules or (len(sys.argv) > 0 and sys.argv[0].endswith("streamlit")):
        render_streamlit_ui()
    else:
        try:
            import streamlit.runtime
            if streamlit.runtime.exists():
                render_streamlit_ui()
            else:
                main()
        except ImportError:
            main()
