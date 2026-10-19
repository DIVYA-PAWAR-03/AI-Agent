"""
Unit tests for Document Analysis, Insight Generation, LLM Service, and Catalog Builder.
Covers successful analysis, invalid model output, missing API configuration, caching, and rate limiting.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock
import pytest

from src.agent.catalog_builder import CatalogBuilder, format_catalog_summary
from src.agent.llm_service import (
    LLMConfigurationError,
    LLMOutputParsingError,
    LLMRateLimitError,
    LLMService,
)
from src.config import AppConfig
from src.models.schemas import (
    DocumentCatalog,
    DocumentCatalogEntry,
    DocumentExtractionRecord,
    FactOrMetric,
    PageExtraction,
    TwoLineInsight,
)


@pytest.fixture
def sample_doc_record() -> DocumentExtractionRecord:
    """Fixture providing a realistic extracted document record."""
    return DocumentExtractionRecord(
        doc_id="PDF_01",
        filename="PDF_01_Quantum_computing.pdf",
        relative_path="PDF_01_Quantum_computing.pdf",
        file_size_bytes=43716,
        file_hash="dummyhash1234567890abcdef",
        page_count=2,
        extracted_pages_count=2,
        total_char_count=500,
        total_word_count=80,
        status="success",
        needs_ocr=False,
        extracted_at="2026-10-19T10:00:00Z",
        pages=[
            PageExtraction(
                page_number=1,
                text="Quantum computing utilizes quantum mechanics principles such as superposition.",
                char_count=78,
                word_count=10,
                has_images=False,
                needs_ocr=False,
            ),
            PageExtraction(
                page_number=2,
                text="In 2024, experimental tests demonstrated a 100x speedup in specific algorithms.",
                char_count=78,
                word_count=11,
                has_images=False,
                needs_ocr=False,
            ),
        ],
    )


@pytest.fixture
def valid_llm_json_response() -> str:
    """Fixture providing valid structured JSON conforming to the analysis schema."""
    return json.dumps({
        "title": "Quantum Computing Fundamentals",
        "main_topic": "Quantum Computing Principles and Applications",
        "purpose": "Provide a comprehensive survey of quantum computation mechanisms and algorithms.",
        "category": "Physics & Quantum Computing",
        "two_line_insight": {
            "doc_id": "PDF_01",
            "line_1": "Quantum computing harnesses superposition and entanglement to surpass classical computational limits.",
            "line_2": "Algorithmic speedups of up to 100x have been experimentally observed in targeted optimization tasks.",
        },
        "summary": "This document explores the fundamental mechanisms of quantum computing, outlining qubits and superposition. It reviews empirical benchmarks and algorithm performance.",
        "key_findings": [
            "Superposition allows qubits to evaluate multiple problem states concurrently.",
            "Algorithmic benchmarks in 2024 revealed significant speedups over classical counterparts.",
        ],
        "conclusions": [
            "Quantum computational systems hold transformative potential for specialized domain problems.",
        ],
        "facts_and_metrics": [
            {
                "fact": "100x algorithmic speedup demonstrated in 2024 experimental benchmarks.",
                "page_number": 2,
                "is_explicit_fact": True,
            }
        ],
        "recommendations": [
            "Standardize error-mitigation protocols across cryogenic quantum processors."
        ],
        "risks_and_limitations": [
            "Decoherence and thermal noise remain major barriers to fault-tolerant scaling."
        ],
        "explicit_vs_interpretation_notes": "The 100x speedup is explicitly measured in the text; domain scalability is an informed projection.",
    })


def test_missing_api_configuration_raises_clear_error(sample_doc_record):
    """Verify that attempting to analyze without a configured API key raises LLMConfigurationError."""
    cfg = AppConfig(gemini_api_key=None)
    service = LLMService(config=cfg)

    assert service.is_configured is False
    with pytest.raises(LLMConfigurationError) as exc_info:
        service.analyze_document(sample_doc_record)
    assert "GEMINI_API_KEY is not configured" in str(exc_info.value)


def test_successful_analysis_with_mock_client(sample_doc_record, valid_llm_json_response):
    """Verify successful parsing and validation when LLM returns valid structured JSON."""
    cfg = AppConfig(gemini_api_key="AIzaSyMockValidKey")
    mock_model = MagicMock()
    mock_response = MagicMock()
    mock_response.text = valid_llm_json_response
    mock_model.generate_content.return_value = mock_response

    service = LLMService(config=cfg)
    service._model = mock_model

    entry = service.analyze_document(sample_doc_record)

    assert entry.doc_id == "PDF_01"
    assert entry.filename == "PDF_01_Quantum_computing.pdf"
    assert entry.category == "Physics & Quantum Computing"
    assert "superposition" in entry.two_line_insight.line_1.lower()
    assert len(entry.facts_and_metrics) == 1
    assert entry.facts_and_metrics[0].page_number == 2
    assert entry.status == "success"
    assert entry.error is None


def test_invalid_model_output_handling(sample_doc_record):
    """Verify that malformed / non-JSON model output is handled gracefully without uncaught crashes."""
    cfg = AppConfig(gemini_api_key="AIzaSyMockValidKey")
    mock_model = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "This is plain prose and not JSON at all."
    mock_model.generate_content.return_value = mock_response

    service = LLMService(config=cfg)
    service._model = mock_model

    # Should retry and return a failed entry record rather than crashing
    entry = service.analyze_document(sample_doc_record, max_retries=2, retry_delay_seconds=0.01)
    assert entry.status == "failed"
    assert entry.doc_id == "PDF_01"
    assert "Failed to decode LLM response as JSON" in entry.error


def test_invalid_schema_output_handling(sample_doc_record):
    """Verify that JSON output violating the schema types is caught and handled gracefully."""
    cfg = AppConfig(gemini_api_key="AIzaSyMockValidKey")
    mock_model = MagicMock()
    mock_response = MagicMock()
    # Invalid: facts_and_metrics is a string instead of a list
    mock_response.text = json.dumps({
        "title": "Quantum Computing",
        "main_topic": "Quantum",
        "purpose": "Survey",
        "category": "Physics",
        "two_line_insight": "not a dict",
        "summary": "Summary text",
        "facts_and_metrics": "invalid type",
    })
    mock_model.generate_content.return_value = mock_response

    service = LLMService(config=cfg)
    service._model = mock_model

    entry = service.analyze_document(sample_doc_record, max_retries=2, retry_delay_seconds=0.01)
    assert entry.status == "failed"
    assert "Failed to validate catalog entry against Pydantic schema" in entry.error


def test_rate_limit_backoff_and_retry(sample_doc_record, valid_llm_json_response):
    """Verify exponential backoff retry when rate limit (429) occurs."""
    cfg = AppConfig(gemini_api_key="AIzaSyMockValidKey")
    mock_model = MagicMock()
    mock_response = MagicMock()
    mock_response.text = valid_llm_json_response

    # First call raises rate limit exception, second call succeeds
    mock_model.generate_content.side_effect = [
        Exception("ResourceExhausted: 429 Rate limit exceeded"),
        mock_response,
    ]

    service = LLMService(config=cfg)
    service._model = mock_model

    entry = service.analyze_document(sample_doc_record, max_retries=2, retry_delay_seconds=0.01)
    assert entry.status == "success"
    assert mock_model.generate_content.call_count == 2


def test_catalog_builder_incremental_caching(tmp_path: Path, sample_doc_record, valid_llm_json_response):
    """Verify that unchanged documents are skipped and not re-analyzed by LLM."""
    extracted_dir = tmp_path / "extracted"
    docs_dir = extracted_dir / "documents"
    docs_dir.mkdir(parents=True)
    catalog_dir = tmp_path / "catalog"

    # Write sample extracted record to disk
    (docs_dir / "PDF_01.json").write_text(sample_doc_record.model_dump_json(), encoding="utf-8")

    mock_service = MagicMock(spec=LLMService)
    mock_service.is_configured = True
    mock_entry = DocumentCatalogEntry(
        doc_id="PDF_01",
        filename=sample_doc_record.filename,
        file_hash=sample_doc_record.file_hash,
        title="Quantum Computing",
        main_topic="Quantum",
        purpose="Survey",
        category="Physics & Quantum Computing",
        two_line_insight=TwoLineInsight(doc_id="PDF_01", line_1="Insight 1", line_2="Insight 2"),
        summary="Summary text",
        key_findings=["Finding 1"],
        conclusions=["Conclusion 1"],
        facts_and_metrics=[],
        recommendations=[],
        risks_and_limitations=[],
        explicit_vs_interpretation_notes=None,
        analyzed_at="2026-10-19T10:00:00Z",
        model_used="gemini-1.5-flash",
        status="success",
        error=None,
    )
    mock_service.analyze_document.return_value = mock_entry

    builder = CatalogBuilder(
        llm_service=mock_service,
        extracted_dir=extracted_dir,
        catalog_dir=catalog_dir,
    )

    # First build: should call LLM once
    cat1 = builder.build_catalog(force=False)
    assert cat1.total_documents == 1
    assert cat1.analyzed_count == 1
    assert mock_service.analyze_document.call_count == 1

    # Second build with force=False: should use cached entry and NOT call LLM
    cat2 = builder.build_catalog(force=False)
    assert cat2.total_documents == 1
    assert mock_service.analyze_document.call_count == 1  # Unchanged!

    # Third build with force=True: should re-analyze
    cat3 = builder.build_catalog(force=True)
    assert cat3.total_documents == 1
    assert mock_service.analyze_document.call_count == 2


def test_catalog_builder_missing_prerequisite(tmp_path: Path):
    """Verify clear FileNotFoundError when extracted documents directory is absent."""
    empty_dir = tmp_path / "empty_extracted"
    builder = CatalogBuilder(extracted_dir=empty_dir)

    with pytest.raises(FileNotFoundError) as exc_info:
        builder.build_catalog()
    assert "Extracted documents directory not found" in str(exc_info.value)


def test_format_catalog_summary():
    """Verify catalog summary formatting."""
    cat = DocumentCatalog(
        generated_at="2026-10-19T10:00:00Z",
        total_documents=2,
        analyzed_count=2,
        failed_count=0,
        categories={"Physics & Quantum Computing": 1, "Economics & Finance": 1},
        entries={},
    )
    summary = format_catalog_summary(cat)
    assert "Document Analysis & Catalog Summary" in summary
    assert "Physics & Quantum Computing" in summary
    assert "Economics & Finance" in summary
