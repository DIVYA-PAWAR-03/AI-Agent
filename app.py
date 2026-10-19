"""
PDF Document Summarization & Insight Agent (UC05)
Application Entry Point & Setup Dashboard.
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
    ingest_info = load_ingestion_manifest_summary()
    catalog_info = load_catalog_summary()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("App Version", status["version"])
        st.write(f"**Environment:** `{status['environment']}`")
    with col2:
        pdf_label = "Available" if status["pdf_dataset_found"] else "Missing"
        st.metric("PDFs Discovered", f"{status['pdf_count']} ({pdf_label})")
        st.write(f"**Input Dir:** `{status['pdf_dataset_dir']}`")
    with col3:
        if ingest_info["ingested"]:
            st.metric("Extracted Corpus", f"{ingest_info['total_pages']} Pages")
            st.write(f"**Indexed Docs:** {ingest_info['total_processed']} PDFs")
        else:
            st.metric("Extracted Corpus", "Not Ingested")
            st.write("Run `python -m scripts.ingest_pdfs`")
    with col4:
        if catalog_info["catalog_built"]:
            st.metric("Analyzed Catalog", f"{catalog_info['analyzed_count']} Docs")
            st.write(f"**Domains:** {len(catalog_info['categories'])} Categories")
        else:
            st.metric("Analyzed Catalog", "Pending")
            st.write("Run `python -m scripts.analyze_documents`")

    st.markdown("### Component Readiness Checklist")

    # Dataset Status
    if status["pdf_dataset_found"] and status["pdf_count"] > 0:
        st.success(f"Dataset verified: Found {status['pdf_count']} PDF documents.")
    else:
        st.warning(
            f"Dataset not detected at `{status['pdf_dataset_dir']}`. "
            "Please ensure PDFs are placed in the configured directory."
        )

    # Ingestion Pipeline Status
    if ingest_info["ingested"]:
        st.success(
            f"Ingestion Pipeline: Extracted {ingest_info['total_pages']} pages across "
            f"{ingest_info['total_processed']} documents into `{config.extracted_data_dir}`."
        )
    else:
        st.info(
            "Ingestion Pipeline: Text extraction pending. "
            "Run `python -m scripts.ingest_pdfs` to process PDFs."
        )

    # Catalog & Insight Status
    if catalog_info["catalog_built"]:
        st.success(
            f"Catalog Analysis: {catalog_info['analyzed_count']} documents analyzed with 2-line insights "
            f"across {len(catalog_info['categories'])} domains."
        )
    else:
        st.info(
            "Document Catalog: Analysis pending. "
            "Configure GEMINI_API_KEY in `.env` and run `python -m scripts.analyze_documents`."
        )

    # LLM Key Status
    if status["gemini_api_key_configured"]:
        st.success(f"LLM API Key: {status['gemini_status_message']}")
    else:
        st.info(
            f"ℹ️ {status['gemini_status_message']}\n\n"
            "Create a `.env` file from `.env.example` and set `GEMINI_API_KEY`."
        )

    st.markdown("---")
    st.info(
        "Phase 03 Document Analysis & Insight Generation module complete. "
        "Vector search, multi-doc reasoning, and grounded conversational QA will activate in Phase 04."
    )


def main():
    """CLI and execution entry point."""
    status = config.get_setup_status()
    ingest_info = load_ingestion_manifest_summary()
    catalog_info = load_catalog_summary()

    print("=" * 60)
    print(f" {status['app_name']} (v{status['version']})")
    print("=" * 60)
    print(f"Environment:         {status['environment']}")
    print(f"PDF Dataset Dir:     {status['pdf_dataset_dir']}")
    print(f"PDFs Detected:       {status['pdf_count']}")
    if ingest_info["ingested"]:
        print(f"Ingestion Status:    Extracted ({ingest_info['total_processed']} PDFs, {ingest_info['total_pages']} pages)")
    else:
        print("Ingestion Status:    Pending (run: python -m scripts.ingest_pdfs)")
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
    print("=" * 60)
    print("To launch the Streamlit interface, run:")
    print("  streamlit run app.py")
    print("=" * 60)


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
