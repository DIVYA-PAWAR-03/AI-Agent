"""
Tests for configuration loading, credential validation, and setup status.
"""

from pathlib import Path
from src.config import AppConfig


def test_default_config_instantiation():
    """Verify that AppConfig can be instantiated without requiring environment variables."""
    cfg = AppConfig()
    assert cfg.app_name == "PDF Document Summarization & Insight Agent"
    assert cfg.app_version == "0.1.0"
    assert cfg.llm_provider == "gemini"
    assert cfg.gemini_model == "gemini-1.5-flash"


def test_validate_llm_credentials_missing():
    """Verify friendly error message when Gemini API key is missing or empty."""
    cfg = AppConfig(gemini_api_key=None)
    is_valid, msg = cfg.validate_llm_credentials()
    assert is_valid is False
    assert "GEMINI_API_KEY is not configured" in msg


def test_validate_llm_credentials_placeholder():
    """Verify friendly error message when placeholder text is present."""
    cfg = AppConfig(gemini_api_key="your_gemini_api_key_here")
    is_valid, msg = cfg.validate_llm_credentials()
    assert is_valid is False
    assert "GEMINI_API_KEY is not configured or contains placeholder text" in msg


def test_validate_llm_credentials_valid():
    """Verify positive validation when a real key format is supplied."""
    cfg = AppConfig(gemini_api_key="AIzaSyDummyValidKey123456789")
    is_valid, msg = cfg.validate_llm_credentials()
    assert is_valid is True
    assert "is configured" in msg


def test_setup_status_structure(tmp_path: Path):
    """Verify setup status output dictionary keys and values."""
    # Create temporary PDF folder
    test_pdf_dir = tmp_path / "test_pdfs"
    test_pdf_dir.mkdir()
    (test_pdf_dir / "doc1.pdf").write_bytes(b"%PDF-1.4 dummy")
    (test_pdf_dir / "doc2.pdf").write_bytes(b"%PDF-1.4 dummy")

    cfg = AppConfig(pdf_dataset_dir=test_pdf_dir)
    status = cfg.get_setup_status()

    assert status["app_name"] == "PDF Document Summarization & Insight Agent"
    assert status["pdf_dataset_found"] is True
    assert status["pdf_count"] == 2
    assert "gemini_status_message" in status
    assert "embedding_model" in status


def test_ensure_directories(tmp_path: Path):
    """Verify that ensure_directories creates the target persistence directory."""
    chroma_dir = tmp_path / "test_chroma" / "sub"
    cfg = AppConfig(chroma_persist_dir=chroma_dir)
    assert not chroma_dir.exists()
    cfg.ensure_directories()
    assert chroma_dir.exists()
