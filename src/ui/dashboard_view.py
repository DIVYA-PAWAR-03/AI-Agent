"""
Home Dashboard view for PDF Document Summarization & Insight Agent.
Displays volume metrics, system health, readiness checklist, and pipeline actions.
"""

from __future__ import annotations

import streamlit as st
from src.config import config
from src.ingestion.pipeline import IngestionPipeline
from src.ui.state import get_catalog_builder, load_all_documents, load_system_metrics
from src.vectorstore.indexer import IndexManager


def render_dashboard_view() -> None:
    """Renders the Home Dashboard screen."""
    metrics = load_system_metrics()

    # Introduction & Executive Briefing
    st.markdown(
        """
        Welcome to the **PDF Document Summarization & Insight Agent (UC05)**. This platform ingests multi-domain PDF research 
        corpora, extracts page-level text, indexes semantic vectors via ChromaDB, generates dense 2-line insights and 
        structured executive summaries, and provides an evidence-grounded AI conversational reasoning engine.
        """
    )

    # 1. High-Level Metrics Row (6 Columns)
    st.markdown("### 📊 Corpus & Processing Overview")
    c1, c2, c3, c4, c5, c6 = st.columns(6)

    with c1:
        st.metric(
            label="Discovered PDFs",
            value=metrics["discovered"],
            help="Total PDF files discovered in the configured input directory.",
        )
    with c2:
        st.metric(
            label="Processed / Extracted",
            value=metrics["processed"],
            delta=f"{metrics['total_pages']} Pages" if metrics["total_pages"] > 0 else None,
            help="PDFs successfully parsed page-by-page using PyMuPDF.",
        )
    with c3:
        st.metric(
            label="Failed Extractions",
            value=metrics["failed"],
            delta="0 Errors" if metrics["failed"] == 0 else f"{metrics['failed']} Needs Review",
            delta_color="normal" if metrics["failed"] == 0 else "inverse",
            help="Unreadable or corrupt PDFs encountered during ingestion.",
        )
    with c4:
        st.metric(
            label="Indexed Chunks",
            value=metrics["indexed_chunks"],
            delta=f"{metrics['indexed_docs']} Docs" if metrics["indexed_docs"] > 0 else None,
            help="Semantic text chunks indexed into ChromaDB vector store.",
        )
    with c5:
        st.metric(
            label="Analyzed Catalog",
            value=metrics["analyzed_count"],
            delta=f"{len(metrics['categories'])} Domains" if metrics["categories"] else "Pending",
            help="Documents analyzed with AI summaries and 2-line insights.",
        )
    with c6:
        st.metric(
            label="OCR Required",
            value=metrics["ocr_needed"],
            delta="Scanned" if metrics["ocr_needed"] > 0 else "0 Scanned",
            delta_color="off",
            help="Documents flagged with low character density requiring OCR.",
        )

    st.markdown("---")

    # 2. Component Readiness & Pipeline Architecture
    st.markdown("### 🛡️ System Readiness & Component Status")

    col_left, col_right = st.columns(2)

    with col_left:
        # Ingestion Status Card
        if metrics["processed"] > 0:
            st.success(
                f"**Ingestion Pipeline Active:** {metrics['processed']} PDFs successfully extracted "
                f"({metrics['total_pages']} pages total). Storage location: `{config.extracted_data_dir}`"
            )
        else:
            st.warning(
                f"**Ingestion Pending:** No processed text found in `{config.extracted_data_dir}`. "
                "Trigger the ingestion pipeline below to extract text from `data/pdfs/`."
            )

        # ChromaDB Vector Store Card
        if metrics["indexed_chunks"] > 0:
            st.success(
                f"**ChromaDB Vector Store Ready:** {metrics['indexed_chunks']:,} semantic chunks indexed across "
                f"{metrics['indexed_docs']} documents using `{metrics['embedding_model']}`."
            )
        else:
            st.info(
                f"**Vector Store Empty:** Run indexing to embed extracted document text into ChromaDB "
                f"at `{config.chroma_persist_dir}`."
            )

    with col_right:
        # LLM Credentials Status Card
        if metrics["api_key_configured"]:
            st.success(
                f"**LLM Provider Connected:** {metrics['api_status_message']} "
                f"Model: `{metrics['gemini_model']}`"
            )
        else:
            st.info(
                f"**LLM API Notice:** {metrics['api_status_message']}\n\n"
                "To enable live multi-turn synthesis and automated document cataloging, configure `GEMINI_API_KEY` in your `.env` file."
            )

        # Document Catalog Status Card
        if metrics["analyzed_count"] > 0:
            st.success(
                f"**Document Catalog Ready:** {metrics['analyzed_count']} documents analyzed across "
                f"{len(metrics['categories'])} distinct domains in `{config.catalog_file.name}`."
            )
        else:
            st.info(
                "**Document Catalog Pending:** AI summaries, 2-line insights, and domain classifications "
                "have not yet been generated. Run Document Analysis below once GEMINI_API_KEY is configured."
            )

    st.markdown("---")

    # 3. Pipeline Maintenance & Execution Hub
    st.markdown("### ⚡ Pipeline Action Hub")
    st.caption("Safely trigger pipeline stages. All underlying services are idempotent and skip duplicate work.")

    act_col1, act_col2, act_col3 = st.columns(3)

    with act_col1:
        st.markdown("#### 1. PDF Ingestion")
        st.write("Scan `data/pdfs/` and extract page-level text into JSON documents.")
        if st.button("🚀 Run Ingestion Pipeline", use_container_width=True, key="btn_run_ingest"):
            with st.spinner("Executing PyMuPDF ingestion pipeline..."):
                try:
                    pipeline = IngestionPipeline()
                    manifest = pipeline.run()
                    st.cache_data.clear()
                    st.success(
                        f"Ingestion Complete: Processed {manifest.total_processed} PDFs "
                        f"({manifest.total_pages} total pages, {manifest.total_skipped} skipped)."
                    )
                    st.rerun()
                except Exception as exc:
                    st.error(f"Ingestion Pipeline Failed: {exc}")

    with act_col2:
        st.markdown("#### 2. Vector Indexing")
        st.write("Chunk extracted text, generate embeddings, and store in ChromaDB.")
        if st.button("⚡ Build / Update Index", use_container_width=True, key="btn_run_index"):
            with st.spinner("Chunking text and updating ChromaDB index..."):
                try:
                    indexer = IndexManager()
                    summary = indexer.index_all()
                    st.cache_data.clear()
                    st.success(
                        f"Vector Index Updated: {summary.indexed_chunks_count} chunks indexed across "
                        f"{len(summary.indexed_doc_ids)} documents."
                    )
                    st.rerun()
                except Exception as exc:
                    st.error(f"Vector Indexing Failed: {exc}")

    with act_col3:
        st.markdown("#### 3. AI Document Catalog")
        st.write("Generate dense 2-line insights, domain categories, and structured summaries.")
        can_analyze = metrics["api_key_configured"]
        if st.button(
            "🧠 Analyze Documents",
            use_container_width=True,
            disabled=not can_analyze,
            key="btn_run_analysis",
            help="Requires valid GEMINI_API_KEY in .env" if not can_analyze else "Run LLM analysis on extracted documents",
        ):
            builder = get_catalog_builder()
            progress_bar = st.progress(0, text="Initializing analysis...")
            status_box = st.empty()

            try:
                def on_progress(current: int, total: int, doc_id: str):
                    pct = int((current / max(1, total)) * 100)
                    progress_bar.progress(pct, text=f"Analyzing {doc_id} ({current}/{total})...")

                catalog = builder.build_catalog(progress_callback=on_progress)
                st.cache_data.clear()
                progress_bar.progress(100, text="Analysis Complete!")
                st.success(f"Analysis Complete! Catalog contains {catalog.analyzed_count} analyzed documents.")
                st.rerun()
            except Exception as exc:
                st.error(f"Analysis Execution Failed: {exc}")
