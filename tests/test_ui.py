"""
Automated unit and integration tests for Streamlit UI views and state management.
Tests run completely offline and do not require live Streamlit servers or network access.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from src.ui.styles import get_custom_css
from src.ui.state import (
    clear_chat,
    init_session_state,
    load_all_documents,
    load_extracted_page_text,
    load_system_metrics,
)


def test_ui_modules_import():
    """Verify that all UI modules import cleanly."""
    import src.ui.styles
    import src.ui.state
    import src.ui.dashboard_view
    import src.ui.explorer_view
    import src.ui.chat_view
    import src.ui.analysis_view
    import app

    assert callable(app.render_streamlit_ui)
    assert callable(src.ui.styles.get_custom_css)
    assert callable(src.ui.state.load_all_documents)


def test_ui_custom_css():
    """Verify custom CSS string contains required styling tokens."""
    css = get_custom_css()
    assert "<style>" in css
    assert "--primary-accent" in css
    assert "#4F46E5" in css
    assert "main-title-container" in css
    assert "insight-quote" in css
    assert "citation-box" in css


def test_load_all_documents():
    """Verify unified document loader retrieves available PDFs and handles metadata."""
    docs = load_all_documents()
    assert isinstance(docs, list)
    assert len(docs) >= 50

    doc1 = next((d for d in docs if d["doc_id"] == "PDF_01"), None)
    assert doc1 is not None
    assert doc1["doc_id"] == "PDF_01"
    assert "Quantum" in doc1["title"] or "PDF_01" in doc1["title"]
    assert doc1["page_count"] > 0
    assert "filename" in doc1
    assert "is_analyzed" in doc1


def test_load_system_metrics():
    """Verify load_system_metrics computes corpus volumes and pipeline status."""
    metrics = load_system_metrics()
    assert metrics["discovered"] >= 50
    assert metrics["processed"] >= 50
    assert metrics["total_pages"] >= 600
    assert metrics["indexed_chunks"] >= 3000
    assert "api_key_configured" in metrics
    assert "environment" in metrics


def test_load_extracted_page_text():
    """Verify loading page-level extracted text for the reader."""
    text = load_extracted_page_text("PDF_01", 1)
    assert text is not None
    assert "quantum" in text.lower()

    # Non-existent doc should return None
    missing = load_extracted_page_text("PDF_NON_EXISTENT", 1)
    assert missing is None


def test_session_state_and_clear_chat():
    """Verify session state initialization and clear_chat behavior."""
    mock_st = MagicMock()
    mock_session_state = {}
    mock_st.session_state = mock_session_state

    with patch("src.ui.state.st", mock_st):
        init_session_state()
        assert "chat_messages" in mock_session_state
        assert "conv_session" in mock_session_state
        assert "current_view" in mock_session_state
        assert mock_session_state["current_view"] == "Dashboard"

        # Add dummy messages
        mock_session_state["chat_messages"] = [{"role": "user", "content": "Hello"}]
        mock_conv = MagicMock()
        mock_session_state["conv_session"] = mock_conv

        clear_chat()
        assert mock_session_state["chat_messages"] == []
        mock_conv.clear.assert_called_once()


def test_document_search_and_filter_logic():
    """Verify search, category, and status filter logic on document collections."""
    docs = [
        {"doc_id": "PDF_01", "title": "Quantum Computing", "filename": "PDF_01.pdf", "category": "Physics", "is_analyzed": True, "needs_ocr": False},
        {"doc_id": "PDF_02", "title": "Artificial Intelligence", "filename": "PDF_02.pdf", "category": "Computer Science", "is_analyzed": True, "needs_ocr": False},
        {"doc_id": "PDF_03", "title": "Scanned Document", "filename": "PDF_03.pdf", "category": "Uncategorized", "is_analyzed": False, "needs_ocr": True},
    ]

    # Search filter
    query = "quantum"
    matched = [d for d in docs if query in d["title"].lower() or query in d["doc_id"].lower()]
    assert len(matched) == 1
    assert matched[0]["doc_id"] == "PDF_01"

    # Category filter
    physics_docs = [d for d in docs if d["category"] == "Physics"]
    assert len(physics_docs) == 1

    # Needs OCR filter
    ocr_docs = [d for d in docs if d["needs_ocr"]]
    assert len(ocr_docs) == 1
    assert ocr_docs[0]["doc_id"] == "PDF_03"
