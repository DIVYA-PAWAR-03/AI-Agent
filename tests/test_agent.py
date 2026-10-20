"""
Unit tests for Evidence-Grounded Document Agent, Multi-turn Reasoning, and Guardrails.
Covers all required test scenarios using offline deterministic test fixtures.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock
import pytest

from src.agent.conversation import ConversationSession
from src.agent.intent import OperationType, classify_operation, extract_document_references
from src.agent.llm_service import LLMService
from src.agent.qa_engine import INSUFFICIENT_EVIDENCE_MESSAGE, DocumentAgent
from src.config import AppConfig
from src.models.schemas import (
    DocumentCatalog,
    DocumentCatalogEntry,
    DocumentExtractionRecord,
    FactOrMetric,
    PageExtraction,
    TextChunk,
    TwoLineInsight,
)
from src.vectorstore.chroma import ChromaVectorStore
from src.vectorstore.embeddings import DeterministicMockEmbeddings


@pytest.fixture
def test_env(tmp_path: Path):
    """Sets up a complete offline test environment with mock vector store, catalog, and extraction records."""
    mock_embeddings = DeterministicMockEmbeddings(dimension=64)
    persist_dir = tmp_path / "chroma"
    extracted_dir = tmp_path / "extracted"
    docs_dir = extracted_dir / "documents"
    docs_dir.mkdir(parents=True)
    catalog_dir = tmp_path / "catalog"
    catalog_dir.mkdir(parents=True)

    vector_store = ChromaVectorStore(
        persist_dir=persist_dir,
        collection_name="test_agent_col",
        embedding_provider=mock_embeddings,
    )

    # 1. Document 1: Quantum Computing (PDF_01)
    rec1 = DocumentExtractionRecord(
        doc_id="PDF_01",
        filename="PDF_01_Quantum_computing.pdf",
        relative_path="PDF_01_Quantum_computing.pdf",
        file_size_bytes=4000,
        file_hash="hash_01",
        page_count=2,
        extracted_pages_count=2,
        total_char_count=800,
        total_word_count=120,
        status="success",
        needs_ocr=False,
        extracted_at="2026-10-19T10:00:00Z",
        pages=[
            PageExtraction(
                page_number=1,
                text="A quantum computer processes information using quantum states. Phenomena include superposition and entanglement. Qubits represent linear superpositions.",
                char_count=160,
                word_count=20,
            ),
            PageExtraction(
                page_number=2,
                text="In 2024, experimental benchmarks demonstrated a 100x algorithmic speedup over classical algorithms in specialized tasks. Decoherence remains an obstacle.",
                char_count=155,
                word_count=20,
            ),
        ],
    )
    (docs_dir / "PDF_01.json").write_text(rec1.model_dump_json(), encoding="utf-8")

    # 2. Document 2: Artificial Intelligence (PDF_02)
    rec2 = DocumentExtractionRecord(
        doc_id="PDF_02",
        filename="PDF_02_Artificial_intelligence.pdf",
        relative_path="PDF_02_Artificial_intelligence.pdf",
        file_size_bytes=3500,
        file_hash="hash_02",
        page_count=2,
        extracted_pages_count=2,
        total_char_count=700,
        total_word_count=110,
        status="success",
        needs_ocr=False,
        extracted_at="2026-10-19T10:00:00Z",
        pages=[
            PageExtraction(
                page_number=1,
                text="Artificial intelligence systems emulate human cognitive capabilities, focusing on neural network architectures and pattern recognition.",
                char_count=145,
                word_count=17,
            ),
            PageExtraction(
                page_number=2,
                text="Transformers utilize attention mechanisms to process sequential data, achieving breakthrough performance in natural language processing.",
                char_count=145,
                word_count=17,
            ),
        ],
    )
    (docs_dir / "PDF_02.json").write_text(rec2.model_dump_json(), encoding="utf-8")

    # 3. Document 3: Empty / Scanned Document with insufficient text (PDF_EMPTY)
    rec_empty = DocumentExtractionRecord(
        doc_id="PDF_EMPTY",
        filename="PDF_EMPTY_Scanned.pdf",
        relative_path="PDF_EMPTY_Scanned.pdf",
        file_size_bytes=1000,
        file_hash="hash_empty",
        page_count=1,
        extracted_pages_count=1,
        total_char_count=10,
        total_word_count=2,
        status="needs_ocr",
        needs_ocr=True,
        extracted_at="2026-10-19T10:00:00Z",
        pages=[PageExtraction(page_number=1, text="[Scanned]", char_count=9, word_count=1, needs_ocr=True)],
    )
    (docs_dir / "PDF_EMPTY.json").write_text(rec_empty.model_dump_json(), encoding="utf-8")

    # Add chunks to vector store
    vector_store.add_chunks([
        TextChunk(
            chunk_id="PDF_01_p1_c0",
            doc_id="PDF_01",
            filename=rec1.filename,
            page_number=1,
            chunk_index=0,
            text=rec1.pages[0].text,
        ),
        TextChunk(
            chunk_id="PDF_01_p2_c0",
            doc_id="PDF_01",
            filename=rec1.filename,
            page_number=2,
            chunk_index=0,
            text=rec1.pages[1].text,
        ),
        TextChunk(
            chunk_id="PDF_02_p1_c0",
            doc_id="PDF_02",
            filename=rec2.filename,
            page_number=1,
            chunk_index=0,
            text=rec2.pages[0].text,
        ),
        TextChunk(
            chunk_id="PDF_02_p2_c0",
            doc_id="PDF_02",
            filename=rec2.filename,
            page_number=2,
            chunk_index=0,
            text=rec2.pages[1].text,
        ),
    ])

    # Catalog Entry for PDF_01 with NO recommendations
    catalog_entry = DocumentCatalogEntry(
        doc_id="PDF_01",
        filename=rec1.filename,
        file_hash=rec1.file_hash,
        title="Quantum Computing Principles",
        main_topic="Quantum Computing and Qubit Superposition",
        purpose="Analyze quantum computation models and empirical speedup benchmarks.",
        category="Physics & Quantum Computing",
        two_line_insight=TwoLineInsight(
            doc_id="PDF_01",
            line_1="Quantum computing leverages superposition and entanglement to solve specialized complex problems.",
            line_2="Experimental benchmarks demonstrate a 100x speedup in targeted optimization tasks.",
        ),
        summary="A survey of quantum computing architectures and superposition capabilities.",
        key_findings=["Superposition allows multi-state evaluation.", "100x speedup observed in 2024."],
        conclusions=["Quantum processors offer breakthrough potential for specialized workloads."],
        facts_and_metrics=[
            FactOrMetric(fact="100x speedup observed in 2024", page_number=2, is_explicit_fact=True)
        ],
        recommendations=[],  # Explicitly empty list!
        risks_and_limitations=["Decoherence causes error escalation."],
        explicit_vs_interpretation_notes="The 100x speedup is an explicit metric from page 2.",
        analyzed_at="2026-10-19T10:00:00Z",
        model_used="gemini-1.5-flash",
        status="success",
    )

    catalog = DocumentCatalog(
        generated_at="2026-10-19T10:00:00Z",
        total_documents=1,
        analyzed_count=1,
        failed_count=0,
        categories={"Physics & Quantum Computing": 1},
        entries={"PDF_01": catalog_entry},
    )
    (catalog_dir / "document_catalog.json").write_text(catalog.model_dump_json(), encoding="utf-8")

    # Mock LLM Service
    mock_llm = MagicMock(spec=LLMService)
    mock_llm.is_configured = True
    mock_model = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "Based on page 2 of PDF_01, experimental benchmarks in 2024 revealed a 100x speedup."
    mock_model.generate_content.return_value = mock_response
    mock_llm._model = mock_model

    # Catalog Builder mock
    mock_cb = MagicMock()
    mock_cb.load_catalog.return_value = catalog

    agent = DocumentAgent(
        llm_service=mock_llm,
        vector_store=vector_store,
        catalog_builder=mock_cb,
        session=ConversationSession(),
        extracted_dir=extracted_dir,
    )

    return {"agent": agent, "mock_llm": mock_llm, "mock_model": mock_model}


# -----------------------------------------------------------------------------
# Test Scenario 1: Detailed Summary
# -----------------------------------------------------------------------------
def test_scenario_1_detailed_summary(test_env):
    """Test Scenario 1: Ask for a detailed summary of a specific PDF."""
    agent: DocumentAgent = test_env["agent"]
    resp = agent.process_query("Give me a detailed summary of PDF_01")

    assert resp.operation == "detailed_summary"
    assert resp.active_doc_id == "PDF_01"
    assert resp.is_directly_supported is True
    assert "Quantum Computing" in resp.answer
    assert "Key Findings" in resp.answer
    assert len(resp.citations) >= 1
    assert resp.citations[0].doc_id == "PDF_01"


# -----------------------------------------------------------------------------
# Test Scenario 2: Finding Supported by Specific Page
# -----------------------------------------------------------------------------
def test_scenario_2_finding_supported_by_specific_page(test_env):
    """Test Scenario 2: Ask for a finding supported by a specific page."""
    agent: DocumentAgent = test_env["agent"]
    test_env["mock_response"] = "The document explicitly states a 100x speedup on [Page 2]."

    resp = agent.process_query("What speedup finding is reported in PDF_01 on page 2?")

    assert resp.operation in ("document_qa", "facts_and_recommendations")
    assert resp.active_doc_id == "PDF_01"
    assert resp.is_directly_supported is True
    assert len(resp.citations) >= 1
    # Check that citations include page 2
    pages = [c.page_number for c in resp.citations]
    assert 2 in pages


# -----------------------------------------------------------------------------
# Test Scenario 3: Follow-Up Questions Using Pronouns
# -----------------------------------------------------------------------------
def test_scenario_3_follow_up_question_using_pronouns(test_env):
    """Test Scenario 3: Ask a follow-up using 'it', 'that finding', preserving active context."""
    agent: DocumentAgent = test_env["agent"]

    # Turn 1: Discuss PDF_01
    resp1 = agent.process_query("What does PDF_01 say about qubits?")
    assert resp1.active_doc_id == "PDF_01"
    assert agent.session.active_doc_id == "PDF_01"

    # Turn 2: Follow-up using 'it'
    resp2 = agent.process_query("What are the primary obstacles facing it?")
    assert resp2.active_doc_id == "PDF_01"
    assert resp2.operation in ("follow_up_qa", "document_qa", "facts_and_recommendations")
    # Verify that conversation history has both turns
    assert len(agent.session.messages) == 4


# -----------------------------------------------------------------------------
# Test Scenario 4: Cross-Document Comparison
# -----------------------------------------------------------------------------
def test_scenario_4_compare_two_documents(test_env):
    """Test Scenario 4: Compare two documents on related topics."""
    agent: DocumentAgent = test_env["agent"]
    mock_model = test_env["mock_model"]
    mock_model.generate_content.return_value.text = (
        "**Key Similarities:** Both examine advanced computational paradigms.\n"
        "**Contrasts:** PDF_01 focuses on quantum physics and qubits, while PDF_02 focuses on neural networks."
    )

    resp = agent.process_query("Compare PDF_01 and PDF_02")

    assert resp.operation == "compare_documents"
    assert "PDF_01" in resp.target_doc_ids
    assert "PDF_02" in resp.target_doc_ids
    assert resp.is_directly_supported is True
    # Citations should reference both documents
    citation_docs = {c.doc_id for c in resp.citations}
    assert "PDF_01" in citation_docs
    assert "PDF_02" in citation_docs


# -----------------------------------------------------------------------------
# Test Scenario 5: Question Not Supported by Documents
# -----------------------------------------------------------------------------
def test_scenario_5_unsupported_question(test_env):
    """Test Scenario 5: Ask a question not supported by the documents."""
    agent: DocumentAgent = test_env["agent"]
    test_env["mock_model"].generate_content.return_value.text = INSUFFICIENT_EVIDENCE_MESSAGE

    resp = agent.process_query("What does PDF_01 say about the Roman Colosseum and Julius Caesar?")

    assert resp.is_directly_supported is False
    assert INSUFFICIENT_EVIDENCE_MESSAGE in resp.answer


# -----------------------------------------------------------------------------
# Test Scenario 6: Recommendations Check (None Invented)
# -----------------------------------------------------------------------------
def test_scenario_6_recommendation_check_none_invented(test_env):
    """Test Scenario 6: Ask for recommendations and verify none are fabricated."""
    agent: DocumentAgent = test_env["agent"]

    resp = agent.process_query("What are the main recommendations in PDF_01?")

    assert resp.operation == "facts_and_recommendations"
    assert "does not explicitly state any recommendations" in resp.answer.lower()
    assert resp.is_directly_supported is True


# -----------------------------------------------------------------------------
# Test Scenario 7: Document with Insufficient Extracted Text
# -----------------------------------------------------------------------------
def test_scenario_7_insufficient_text_document(test_env):
    """Test Scenario 7: Ask for a summary of a document with insufficient extracted text."""
    agent: DocumentAgent = test_env["agent"]

    resp = agent.process_query("Summarize PDF_EMPTY")

    assert resp.operation == "detailed_summary"
    assert resp.is_directly_supported is False
    assert "insufficient extracted text" in resp.answer.lower()


# -----------------------------------------------------------------------------
# Test Scenario 8: Intent Classification & Reference Extraction
# -----------------------------------------------------------------------------
def test_intent_classification_and_extraction():
    """Verify regex and keyword intent routing."""
    refs1 = extract_document_references("What is quantum computing in PDF_01?")
    assert refs1 == ["PDF_01"]

    refs2 = extract_document_references("Compare artificial intelligence and deep learning")
    assert "PDF_02" in refs2
    assert "PDF_03" in refs2

    op1 = classify_operation("Summarize PDF_01 in two lines", ["PDF_01"])
    assert op1 == OperationType.TWO_LINE_INSIGHT

    op2 = classify_operation("What category is PDF_05?", ["PDF_05"])
    assert op2 == OperationType.CATEGORIZATION_LOOKUP

    op3 = classify_operation("Compare PDF_01 and PDF_02", ["PDF_01", "PDF_02"])
    assert op3 == OperationType.COMPARE_DOCUMENTS


# -----------------------------------------------------------------------------
# Test Scenario 9: Session Reset
# -----------------------------------------------------------------------------
def test_session_clear():
    """Verify session reset removes all history and active documents."""
    session = ConversationSession()
    session.set_active_document("PDF_01")
    session.add_user_query("Hello")
    session.add_assistant_response("Hi", operation="document_qa")

    assert len(session.messages) == 2
    assert session.active_doc_id == "PDF_01"

    session.clear()
    assert len(session.messages) == 0
    assert session.active_doc_id is None
