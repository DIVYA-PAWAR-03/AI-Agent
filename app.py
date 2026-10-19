"""
PDF Document Summarization & Insight Agent (UC05)
Application Entry Point & Setup Dashboard.
"""

from __future__ import annotations

import sys
from src.config import config
from src.utils.logger import setup_logger

logger = setup_logger("app_entrypoint", config.log_level)


def render_streamlit_ui():
    """Renders the Streamlit setup status interface."""
    import streamlit as st

    st.set_page_config(
        page_title="PDF Insight Agent (UC05)",
        page_icon="📄",
        layout="wide",
    )

    st.title("📄 PDF Document Summarization & Insight Agent")
    st.caption("UC05 — AI-Powered Multi-Domain Document Analysis & Grounded Insights")

    st.markdown("---")
    st.subheader("System Setup & Environment Status")

    status = config.get_setup_status()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("App Version", status["version"])
        st.write(f"**Environment:** `{status['environment']}`")
    with col2:
        pdf_label = "Available" if status["pdf_dataset_found"] else "Missing"
        st.metric("PDF Documents Detected", f"{status['pdf_count']} ({pdf_label})")
        st.write(f"**Dataset Directory:** `{status['pdf_dataset_dir']}`")
    with col3:
        api_status = "Configured" if status["gemini_api_key_configured"] else "Key Needed"
        st.metric("LLM Provider", f"{status['llm_provider']} ({api_status})")
        st.write(f"**Target Model:** `{status['gemini_model']}`")

    st.markdown("### Component Readiness Checklist")

    # Dataset Status
    if status["pdf_dataset_found"] and status["pdf_count"] > 0:
        st.success(f"Dataset verified: Found {status['pdf_count']} PDF documents in dataset directory.")
    else:
        st.warning(
            f"Dataset not detected at `{status['pdf_dataset_dir']}`. "
            "Please ensure PDFs are placed in the configured directory."
        )

    # LLM Key Status
    if status["gemini_api_key_configured"]:
        st.success(f"LLM API Key: {status['gemini_status_message']}")
    else:
        st.info(
            f"ℹ️ {status['gemini_status_message']}\n\n"
            "Create a `.env` file from `.env.example` and set `GEMINI_API_KEY`."
        )

    # Vector Storage & Embedding Status
    st.info(
        f"**Embeddings Model:** `{status['embedding_model']}`\n\n"
        f"**ChromaDB Directory:** `{status['chroma_persist_dir']}`"
    )

    st.markdown("---")
    st.info(
        "Phase 01 setup complete. Document ingestion, vector indexing, "
        "and agent reasoning will activate in subsequent phases."
    )


def main():
    """CLI and execution entry point."""
    status = config.get_setup_status()
    print("=" * 60)
    print(f" {status['app_name']} (v{status['version']})")
    print("=" * 60)
    print(f"Environment:         {status['environment']}")
    print(f"PDF Dataset Dir:     {status['pdf_dataset_dir']}")
    print(f"PDFs Detected:       {status['pdf_count']}")
    print(f"LLM Provider:        {status['llm_provider']} ({status['gemini_model']})")
    print(f"LLM Key Configured:  {status['gemini_api_key_configured']}")
    print(f"Status Note:         {status['gemini_status_message']}")
    print(f"Embedding Model:     {status['embedding_model']}")
    print(f"Chroma Storage:      {status['chroma_persist_dir']}")
    print("=" * 60)
    print("To launch the Streamlit interface, run:")
    print("  streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    # If run through streamlit, render UI; otherwise execute main CLI summary
    if "streamlit" in sys.modules or (len(sys.argv) > 0 and sys.argv[0].endswith("streamlit")):
        render_streamlit_ui()
    else:
        # Check if called via streamlit runner
        try:
            import streamlit.runtime
            if streamlit.runtime.exists():
                render_streamlit_ui()
            else:
                main()
        except ImportError:
            main()
