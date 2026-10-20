"""
Session state initialization, data loaders, and cached backend resources.
Ensures zero redundant re-processing across page rerenders.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import streamlit as st

from src.agent.catalog_builder import CatalogBuilder
from src.agent.conversation import ConversationSession
from src.agent.llm_service import LLMService
from src.agent.qa_engine import DocumentAgent
from src.config import config
from src.models.schemas import DocumentCatalog, DocumentCatalogEntry, DocumentExtractionRecord
from src.vectorstore.chroma import ChromaVectorStore


@st.cache_resource
def get_llm_service() -> LLMService:
    """Returns singleton LLMService."""
    return LLMService(config=config)


@st.cache_resource
def get_vector_store() -> ChromaVectorStore:
    """Returns singleton ChromaVectorStore."""
    return ChromaVectorStore(
        persist_dir=config.chroma_persist_dir,
        collection_name=config.chroma_collection_name,
    )


@st.cache_resource
def get_catalog_builder() -> CatalogBuilder:
    """Returns singleton CatalogBuilder."""
    return CatalogBuilder(
        config=config,
        llm_service=get_llm_service(),
        catalog_path=config.catalog_file,
    )


def get_agent(session: ConversationSession) -> DocumentAgent:
    """Returns a DocumentAgent configured with the current conversation session."""
    return DocumentAgent(
        llm_service=get_llm_service(),
        vector_store=get_vector_store(),
        catalog_builder=get_catalog_builder(),
        session=session,
        extracted_dir=config.extracted_data_dir,
    )


def init_session_state() -> None:
    """Initializes Streamlit session state keys if not already present."""
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = "default_session"

    if "conv_session" not in st.session_state:
        st.session_state["conv_session"] = ConversationSession(max_turns=10)

    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []

    if "selected_chat_doc" not in st.session_state:
        st.session_state["selected_chat_doc"] = None

    if "explorer_selected_doc" not in st.session_state:
        st.session_state["explorer_selected_doc"] = None

    if "current_view" not in st.session_state:
        st.session_state["current_view"] = "Dashboard"

    if "search_query" not in st.session_state:
        st.session_state["search_query"] = ""

    if "category_filter" not in st.session_state:
        st.session_state["category_filter"] = "All Categories"


def clear_chat() -> None:
    """Clears chat messages and resets conversation memory."""
    st.session_state["chat_messages"] = []
    if "conv_session" in st.session_state:
        st.session_state["conv_session"].clear()


@st.cache_data(show_spinner=False)
def load_all_documents() -> List[Dict[str, Any]]:
    """
    Loads unified list of all available documents from catalog and extraction manifest.
    Gracefully handles both analyzed and unanalyzed states.
    """
    docs: Dict[str, Dict[str, Any]] = {}

    # 1. Load from extraction manifest
    manifest_file = config.extracted_data_dir / "manifest.json"
    if manifest_file.exists():
        try:
            m_data = json.loads(manifest_file.read_text(encoding="utf-8"))
            for doc in m_data.get("documents", []):
                doc_id = doc.get("doc_id")
                if not doc_id:
                    continue
                # Derive title from filename
                raw_filename = doc.get("filename", "")
                clean_name = raw_filename.replace(".pdf", "").replace("PDF_", "")
                if "_" in clean_name:
                    clean_name = " ".join(clean_name.split("_")[1:])
                clean_title = clean_name.replace("_", " ").title() if clean_name else doc_id

                docs[doc_id] = {
                    "doc_id": doc_id,
                    "filename": raw_filename,
                    "title": clean_title,
                    "page_count": doc.get("page_count", 0),
                    "total_char_count": doc.get("total_char_count", 0),
                    "total_word_count": doc.get("total_word_count", 0),
                    "status": doc.get("status", "pending"),
                    "needs_ocr": doc.get("needs_ocr", False),
                    "category": "Analysis Pending",
                    "purpose": "Awaiting AI analysis with GEMINI_API_KEY.",
                    "two_line_insight": None,
                    "summary": None,
                    "key_findings": [],
                    "conclusions": [],
                    "facts_and_metrics": [],
                    "recommendations": [],
                    "risks": [],
                    "is_analyzed": False,
                }
        except Exception:
            pass

    # 2. Overlay catalog data if available
    if config.catalog_file.exists():
        try:
            c_data = json.loads(config.catalog_file.read_text(encoding="utf-8"))
            entries = c_data.get("entries", {})
            for doc_id, entry in entries.items():
                insight_obj = entry.get("two_line_insight")
                insight_lines = []
                if isinstance(insight_obj, dict):
                    if insight_obj.get("line_1"):
                        insight_lines.append(insight_obj["line_1"])
                    if insight_obj.get("line_2"):
                        insight_lines.append(insight_obj["line_2"])

                if doc_id not in docs:
                    docs[doc_id] = {
                        "doc_id": doc_id,
                        "filename": entry.get("filename", f"{doc_id}.pdf"),
                        "page_count": 0,
                        "total_char_count": 0,
                        "total_word_count": 0,
                        "status": "analyzed",
                        "needs_ocr": False,
                    }

                docs[doc_id].update({
                    "title": entry.get("title") or docs[doc_id].get("title", doc_id),
                    "category": entry.get("category", "General"),
                    "purpose": entry.get("purpose", ""),
                    "two_line_insight": insight_lines,
                    "summary": entry.get("summary", ""),
                    "key_findings": entry.get("key_findings", []),
                    "conclusions": entry.get("conclusions", []),
                    "facts_and_metrics": entry.get("facts_and_metrics", []),
                    "recommendations": entry.get("recommendations", []),
                    "risks": entry.get("risks_and_limitations", []),
                    "is_analyzed": True,
                })
        except Exception:
            pass

    # Return sorted by natural document key (PDF_01, PDF_02, ...)
    sorted_docs = sorted(docs.values(), key=lambda x: x["doc_id"])
    return sorted_docs


@st.cache_data(show_spinner=False)
def load_system_metrics() -> Dict[str, Any]:
    """Computes comprehensive health and volume metrics for the dashboard."""
    status = config.get_setup_status()
    manifest_file = config.extracted_data_dir / "manifest.json"
    index_file = config.chroma_persist_dir / "index_manifest.json"

    discovered = status.get("pdf_count", 0)
    processed = 0
    total_pages = 0
    ocr_needed = 0
    failed = 0

    if manifest_file.exists():
        try:
            m = json.loads(manifest_file.read_text(encoding="utf-8"))
            processed = m.get("total_processed", 0) + m.get("total_skipped", 0)
            total_pages = m.get("total_pages", 0)
            ocr_needed = m.get("total_ocr_needed", 0)
            failed = m.get("total_failed", 0)
            discovered = max(discovered, m.get("total_discovered", 0))
        except Exception:
            pass

    indexed_chunks = 0
    indexed_docs = 0
    if index_file.exists():
        try:
            idx = json.loads(index_file.read_text(encoding="utf-8"))
            indexed_chunks = sum(item.get("chunk_count", 0) for item in idx.values())
            indexed_docs = len(idx)
        except Exception:
            pass

    analyzed_count = 0
    categories: Dict[str, int] = {}
    if config.catalog_file.exists():
        try:
            cat = json.loads(config.catalog_file.read_text(encoding="utf-8"))
            analyzed_count = cat.get("analyzed_count", 0)
            categories = cat.get("categories", {})
        except Exception:
            pass

    return {
        "discovered": discovered,
        "processed": processed,
        "failed": failed,
        "total_pages": total_pages,
        "ocr_needed": ocr_needed,
        "indexed_chunks": indexed_chunks,
        "indexed_docs": indexed_docs,
        "analyzed_count": analyzed_count,
        "categories": categories,
        "api_key_configured": status.get("gemini_api_key_configured", False),
        "api_status_message": status.get("gemini_status_message", ""),
        "embedding_model": status.get("embedding_model", ""),
        "gemini_model": status.get("gemini_model", ""),
        "environment": status.get("environment", "development"),
        "version": status.get("version", "0.1.0"),
    }


def load_extracted_page_text(doc_id: str, page_number: int) -> Optional[str]:
    """Loads raw extracted text for a specific page of a document."""
    doc_file = config.extracted_data_dir / "documents" / f"{doc_id}.json"
    if not doc_file.exists():
        return None
    try:
        data = json.loads(doc_file.read_text(encoding="utf-8"))
        for p in data.get("pages", []):
            if p.get("page_number") == page_number:
                return p.get("text", "")
    except Exception:
        pass
    return None
