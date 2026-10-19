"""
Tests verifying application import, setup execution, and data schemas.
"""

import sys
from io import StringIO
import pytest
from src.models.schemas import (
    DocumentMetadata,
    TwoLineInsight,
    DocumentSummary,
    Citation,
    GroundedAnswer,
)


def test_import_modules_without_api_key():
    """Verify core modules can be imported without an API key set."""
    import src
    import src.config
    import src.models.schemas
    import src.utils.logger
    import app

    assert src.__version__ == "0.1.0"
    assert app.config is not None


def test_cli_entrypoint_execution(monkeypatch):
    """Verify that calling app.main() prints status and returns without error."""
    import app

    captured_out = StringIO()
    monkeypatch.setattr(sys, "stdout", captured_out)

    app.main()
    output = captured_out.getvalue()

    assert "PDF Document Summarization & Insight Agent" in output
    assert "Environment:" in output
    assert "PDFs Detected:" in output
    assert "streamlit run app.py" in output


def test_pydantic_schemas_validity():
    """Verify that domain schemas serialize and validate accurately."""
    doc_meta = DocumentMetadata(
        doc_id="PDF_01",
        filename="PDF_01_Quantum_computing.pdf",
        title="Quantum Computing Fundamentals",
        category="Physics & Computing",
        page_count=5,
    )
    assert doc_meta.doc_id == "PDF_01"

    insight = TwoLineInsight(
        doc_id="PDF_01",
        line_1="Quantum computing leverages qubits and superposition to solve complex problems.",
        line_2="Recent breakthroughs demonstrate significant speedups over classical algorithms.",
    )
    assert len(insight.line_1) > 0

    citation = Citation(doc_id="PDF_01", page_number=2, snippet="Superposition allows qubits...")
    answer = GroundedAnswer(
        query="What allows quantum speedup?",
        answer="Superposition and entanglement enable quantum computers to process multiple states simultaneously.",
        citations=[citation],
        is_directly_supported=True,
    )
    assert answer.is_directly_supported is True
    assert len(answer.citations) == 1
