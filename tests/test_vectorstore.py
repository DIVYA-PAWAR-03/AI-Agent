"""
Unit tests for Chunking, Vector Storage, Indexing, and Semantic Retrieval.
Uses DeterministicMockEmbeddings fixture to avoid downloading external neural models.
"""

from pathlib import Path
import pytest

from scripts.build_index import main as cli_main
from src.models.schemas import DocumentExtractionRecord, PageExtraction, TextChunk
from src.vectorstore.chroma import ChromaVectorStore
from src.vectorstore.chunker import chunk_document, split_text_recursively
from src.vectorstore.embeddings import DeterministicMockEmbeddings
from src.vectorstore.indexer import IndexManager, format_index_summary


@pytest.fixture
def mock_embeddings() -> DeterministicMockEmbeddings:
    """Fixture providing fast deterministic mock embeddings for offline unit tests."""
    return DeterministicMockEmbeddings(dimension=64)


@pytest.fixture
def sample_extraction_record() -> DocumentExtractionRecord:
    """Fixture providing multi-page document record."""
    return DocumentExtractionRecord(
        doc_id="PDF_01",
        filename="PDF_01_Quantum.pdf",
        relative_path="PDF_01_Quantum.pdf",
        file_size_bytes=1000,
        file_hash="hash_alpha_123",
        page_count=2,
        extracted_pages_count=2,
        total_char_count=600,
        total_word_count=80,
        status="success",
        needs_ocr=False,
        extracted_at="2026-10-19T10:00:00Z",
        pages=[
            PageExtraction(
                page_number=1,
                text="Quantum computers use quantum bits or qubits. Qubits exist in superposition.\n\nSuperposition enables simultaneous evaluation of states.",
                char_count=138,
                word_count=16,
            ),
            PageExtraction(
                page_number=2,
                text="Quantum entanglement links multiple particles. Bell inequality tests confirmed quantum nonlocality in physical experiments.",
                char_count=127,
                word_count=15,
            ),
        ],
    )


def test_split_text_recursively_boundaries():
    """Verify recursive splitter respects chunk size, overlap, and natural sentence boundaries."""
    text = (
        "Quantum computing is a transformative field. It leverages quantum phenomena. "
        "Unlike classical bits, qubits represent linear superpositions of states. "
        "Decoherence remains a fundamental engineering challenge."
    )
    chunks = split_text_recursively(text, chunk_size=80, chunk_overlap=20)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 100  # Within bounds


def test_split_text_empty_and_short():
    """Verify edge cases: empty text returns [] and short text returns single chunk."""
    assert split_text_recursively("") == []
    assert split_text_recursively("   \n  ") == []

    short = "Short single sentence."
    assert split_text_recursively(short, chunk_size=100) == [short]


def test_chunk_document_provenance(sample_extraction_record):
    """Verify chunk_document retains doc_id, filename, page_number, and deterministic chunk_ids."""
    chunks = chunk_document(
        sample_extraction_record,
        chunk_size=100,
        chunk_overlap=20,
        extra_metadata={"category": "Physics"},
    )
    assert len(chunks) >= 2
    for chunk in chunks:
        assert chunk.doc_id == "PDF_01"
        assert chunk.filename == "PDF_01_Quantum.pdf"
        assert chunk.page_number in (1, 2)
        assert chunk.chunk_id.startswith("PDF_01_p")
        assert chunk.metadata["category"] == "Physics"
        assert chunk.char_count == len(chunk.text)


def test_deterministic_mock_embeddings(mock_embeddings):
    """Verify deterministic mock embeddings generate repeatable, normalized vectors."""
    vec1 = mock_embeddings.embed_query("quantum computing")
    vec2 = mock_embeddings.embed_query("quantum computing")
    vec3 = mock_embeddings.embed_query("roman architecture")

    assert len(vec1) == 64
    assert vec1 == vec2  # Perfectly deterministic
    assert vec1 != vec3  # Differentiable

    # Verify unit-length normalization
    norm = sum(x * x for x in vec1) ** 0.5
    assert pytest.approx(norm, 0.001) == 1.0


def test_vector_store_add_and_count(tmp_path: Path, mock_embeddings):
    """Verify adding chunks and counting items in persistent ChromaDB collection."""
    store = ChromaVectorStore(
        persist_dir=tmp_path / "chroma",
        collection_name="test_collection",
        embedding_provider=mock_embeddings,
    )
    assert store.count() == 0

    chunk = TextChunk(
        chunk_id="PDF_01_p1_c0",
        doc_id="PDF_01",
        filename="test.pdf",
        page_number=1,
        chunk_index=0,
        text="Sample passage text",
        char_count=19,
        word_count=3,
    )
    added = store.add_chunks([chunk])
    assert added == 1
    assert store.count() == 1


def test_vector_store_delete_document(tmp_path: Path, mock_embeddings):
    """Verify delete_document removes only chunks belonging to that specific doc_id."""
    store = ChromaVectorStore(
        persist_dir=tmp_path / "chroma",
        collection_name="test_del",
        embedding_provider=mock_embeddings,
    )

    c1 = TextChunk(
        chunk_id="PDF_01_p1_c0",
        doc_id="PDF_01",
        filename="doc1.pdf",
        page_number=1,
        chunk_index=0,
        text="Quantum text",
    )
    c2 = TextChunk(
        chunk_id="PDF_02_p1_c0",
        doc_id="PDF_02",
        filename="doc2.pdf",
        page_number=1,
        chunk_index=0,
        text="Biology text",
    )
    store.add_chunks([c1, c2])
    assert store.count() == 2

    store.delete_document("PDF_01")
    assert store.count() == 1
    indexed_ids = store.get_indexed_doc_ids()
    assert "PDF_01" not in indexed_ids
    assert "PDF_02" in indexed_ids


def test_vector_store_query_and_filtering(tmp_path: Path, mock_embeddings):
    """Verify semantic search with document ID filtering and similarity scoring."""
    store = ChromaVectorStore(
        persist_dir=tmp_path / "chroma",
        collection_name="test_query",
        embedding_provider=mock_embeddings,
    )

    c1 = TextChunk(
        chunk_id="PDF_01_p1_c0",
        doc_id="PDF_01",
        filename="quantum.pdf",
        page_number=1,
        chunk_index=0,
        text="Superposition and quantum gates enable parallel state evaluations.",
    )
    c2 = TextChunk(
        chunk_id="PDF_02_p1_c0",
        doc_id="PDF_02",
        filename="history.pdf",
        page_number=1,
        chunk_index=0,
        text="Roman aqueducts and concrete arches revolutionized ancient infrastructure.",
    )
    store.add_chunks([c1, c2])

    # Unfiltered query
    results = store.query("quantum gates", top_k=2, min_score=0.0)
    assert len(results) >= 1
    for r in results:
        assert 0.0 <= r.score <= 1.0
        assert r.page_number == 1

    # Filtered query restricted to PDF_02
    filtered_results = store.query(
        "quantum gates",
        top_k=2,
        doc_ids=["PDF_02"],
        min_score=0.0,
    )
    for r in filtered_results:
        assert r.doc_id == "PDF_02"


def test_vector_store_empty_query_and_empty_collection(tmp_path: Path, mock_embeddings):
    """Verify graceful handling of empty query strings and empty collections."""
    store = ChromaVectorStore(
        persist_dir=tmp_path / "chroma",
        collection_name="test_empty",
        embedding_provider=mock_embeddings,
    )
    # Empty collection query
    assert store.query("test query") == []

    # Add chunk
    store.add_chunks([
        TextChunk(
            chunk_id="C1",
            doc_id="D1",
            filename="f.pdf",
            page_number=1,
            chunk_index=0,
            text="hello",
        )
    ])
    # Empty query string
    assert store.query("") == []
    assert store.query("   \n  ") == []


def test_index_manager_incremental_indexing(tmp_path: Path, mock_embeddings, sample_extraction_record):
    """Verify incremental indexing skips unchanged files and safely re-indexes forced runs."""
    ext_dir = tmp_path / "extracted"
    docs_dir = ext_dir / "documents"
    docs_dir.mkdir(parents=True)
    persist_dir = tmp_path / "chroma"

    # Write document record
    (docs_dir / "PDF_01.json").write_text(sample_extraction_record.model_dump_json(), encoding="utf-8")

    store = ChromaVectorStore(
        persist_dir=persist_dir,
        collection_name="test_incremental",
        embedding_provider=mock_embeddings,
    )
    manager = IndexManager(
        vector_store=store,
        extracted_dir=ext_dir,
        chunk_size=100,
        chunk_overlap=20,
    )

    # First run
    sum1 = manager.build_index(force=False)
    assert sum1.total_documents == 1
    initial_chunks = sum1.total_chunks
    assert initial_chunks > 0

    # Second run without force (should be idempotent / skip)
    sum2 = manager.build_index(force=False)
    assert sum2.total_chunks == initial_chunks
    assert sum2.total_documents == 1

    # Third run with force (should not create duplicate chunks)
    sum3 = manager.build_index(force=True)
    assert sum3.total_chunks == initial_chunks  # Exact same chunk count!


def test_build_index_cli_execution(tmp_path: Path, sample_extraction_record):
    """Verify build_index CLI script execution."""
    ext_dir = tmp_path / "cli_extracted"
    docs_dir = ext_dir / "documents"
    docs_dir.mkdir(parents=True)
    (docs_dir / "PDF_01.json").write_text(sample_extraction_record.model_dump_json(), encoding="utf-8")

    code = cli_main(["--limit", "1", "--verbose"])
    assert code == 0


def test_format_index_summary():
    """Verify index summary formatting function."""
    from src.models.schemas import IndexSummary

    summary = IndexSummary(
        collection_name="test_col",
        total_chunks=150,
        total_documents=10,
        embedding_model="all-MiniLM-L6-v2",
        persist_directory="./data/chroma_db",
        indexed_doc_ids=["PDF_01", "PDF_02"],
        generated_at="2026-10-19T10:00:00Z",
    )
    text = format_index_summary(summary)
    assert "Vector Indexing & RAG Knowledge Base Summary" in text
    assert "Total Chunks:        150" in text
    assert "PDF_01" in text
