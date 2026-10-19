"""
Unit tests for PDF Ingestion Pipeline using small temporary PDF fixtures.
Does not depend on the private 50-document dataset.
"""

from pathlib import Path
import pymupdf
import pytest

from scripts.ingest_pdfs import main as cli_main
from src.config import resolve_dataset_dir
from src.ingestion.loader import (
    clean_extracted_text,
    compute_file_hash,
    extract_pdf_document,
)
from src.ingestion.pipeline import (
    IngestionPipeline,
    discover_pdfs,
    format_ingestion_summary,
    generate_doc_id,
)


def create_sample_pdf(file_path: Path, pages_text: list[str]) -> Path:
    """Helper to generate a clean synthetic PDF for test fixtures."""
    doc = pymupdf.open()
    for text in pages_text:
        page = doc.new_page(width=612, height=792)
        if text:
            page.insert_text((50, 72), text)
    doc.save(str(file_path))
    doc.close()
    return file_path


def create_image_only_pdf(file_path: Path) -> Path:
    """Helper to generate a PDF with a raster image and zero extractable text to test OCR detection."""
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    # Create a small 100x100 RGB pixmap
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 100, 100), False)
    pix.clear_with(128)
    page.insert_image(pymupdf.Rect(50, 50, 200, 200), pixmap=pix)
    doc.save(str(file_path))
    doc.close()
    return file_path


def test_clean_extracted_text():
    """Verify text cleaner strips nulls, normalizes spaces, and collapses excessive line breaks."""
    raw = "Header text\x00 with null.\r\nLine 2 with \u00a0 non-breaking space.   \n\n\n\nLine 3."
    cleaned = clean_extracted_text(raw)
    assert "\x00" not in cleaned
    assert "\r" not in cleaned
    assert "Line 3." in cleaned
    assert "\n\n\n" not in cleaned


def test_extract_valid_multipage_pdf(tmp_path: Path):
    """Verify standard extraction from a multi-page PDF."""
    pdf_path = create_sample_pdf(
        tmp_path / "PDF_01_Test.pdf",
        ["Page 1: Artificial Intelligence overview.", "Page 2: Deep Learning concepts."],
    )

    record = extract_pdf_document(pdf_path, doc_id="PDF_01")
    assert record.doc_id == "PDF_01"
    assert record.filename == "PDF_01_Test.pdf"
    assert record.page_count == 2
    assert record.extracted_pages_count == 2
    assert record.status == "success"
    assert record.needs_ocr is False
    assert "Artificial Intelligence" in record.pages[0].text
    assert record.pages[0].page_number == 1
    assert "Deep Learning" in record.pages[1].text
    assert record.pages[1].page_number == 2


def test_extract_zero_byte_file(tmp_path: Path):
    """Verify graceful failure on 0-byte unreadable file."""
    empty_file = tmp_path / "empty.pdf"
    empty_file.touch()

    record = extract_pdf_document(empty_file, doc_id="DOC_EMPTY")
    assert record.status == "failed"
    assert "0 bytes" in record.error_message


def test_extract_corrupted_file(tmp_path: Path):
    """Verify graceful failure on invalid non-PDF file."""
    corrupt_file = tmp_path / "corrupt.pdf"
    corrupt_file.write_bytes(b"This is not a PDF file header at all.")

    record = extract_pdf_document(corrupt_file, doc_id="DOC_CORRUPT")
    assert record.status == "failed"
    assert "Failed to open/parse PDF" in record.error_message


def test_extract_ocr_detection_for_scanned_pdf(tmp_path: Path):
    """Verify OCR requirement is flagged when pages contain images but 0 extractable text."""
    scanned_pdf = create_image_only_pdf(tmp_path / "scanned_doc.pdf")

    record = extract_pdf_document(scanned_pdf, doc_id="DOC_SCAN")
    assert record.needs_ocr is True
    assert record.status == "needs_ocr"
    assert "requires OCR" in record.error_message


def test_discover_pdfs_recursive(tmp_path: Path):
    """Verify recursive PDF discovery finds only PDF files and ignores non-PDFs."""
    dir_a = tmp_path / "sub_a"
    dir_b = tmp_path / "sub_b"
    dir_a.mkdir()
    dir_b.mkdir()

    create_sample_pdf(dir_a / "doc1.pdf", ["Page 1"])
    create_sample_pdf(dir_b / "doc2.PDF", ["Page 1"])
    (dir_a / "notes.txt").write_text("not a pdf")
    (dir_b / "data.docx").write_bytes(b"PK dummy")

    discovered = discover_pdfs(tmp_path)
    assert len(discovered) == 2
    names = {p.name.lower() for p in discovered}
    assert names == {"doc1.pdf", "doc2.pdf"}


def test_generate_doc_id_and_disambiguation():
    """Verify stable document ID generation and duplicate disambiguation."""
    existing = set()
    base = Path("/root")

    id1 = generate_doc_id(Path("/root/PDF_05_Blockchain.pdf"), base, 1, existing)
    assert id1 == "PDF_05"

    id2 = generate_doc_id(Path("/root/custom_report.pdf"), base, 2, existing)
    assert id2 == "DOC_002_custom_report"

    # Duplicate filename in different folder
    id3 = generate_doc_id(Path("/root/sub/PDF_05_Blockchain.pdf"), base, 3, existing)
    assert id3 == "PDF_05_2"


def test_pipeline_execution_and_idempotency(tmp_path: Path):
    """Verify full pipeline execution, manifest creation, and idempotent skipping."""
    in_dir = tmp_path / "input"
    out_dir = tmp_path / "output"
    in_dir.mkdir()

    create_sample_pdf(in_dir / "PDF_01_Alpha.pdf", ["Alpha content."])
    create_sample_pdf(in_dir / "PDF_02_Beta.pdf", ["Beta content."])

    pipeline = IngestionPipeline(input_dir=in_dir, output_dir=out_dir)

    # First run: should extract both documents
    manifest_1 = pipeline.run(force=False)
    assert manifest_1.total_discovered == 2
    assert manifest_1.total_processed == 2
    assert manifest_1.total_skipped == 0
    assert (out_dir / "manifest.json").exists()
    assert (out_dir / "documents" / "PDF_01.json").exists()
    assert (out_dir / "documents" / "PDF_02.json").exists()

    # Second run without force: should skip both (up-to-date)
    manifest_2 = pipeline.run(force=False)
    assert manifest_2.total_discovered == 2
    assert manifest_2.total_processed == 0
    assert manifest_2.total_skipped == 2

    # Third run with force: should re-extract both
    manifest_3 = pipeline.run(force=True)
    assert manifest_3.total_discovered == 2
    assert manifest_3.total_processed == 2
    assert manifest_3.total_skipped == 0


def test_cli_script_execution(tmp_path: Path):
    """Verify CLI command executes and produces formatted summary."""
    in_dir = tmp_path / "cli_input"
    out_dir = tmp_path / "cli_output"
    in_dir.mkdir()
    create_sample_pdf(in_dir / "PDF_01_Demo.pdf", ["Sample demo text."])

    code = cli_main(["--input-dir", str(in_dir), "--output-dir", str(out_dir)])
    assert code == 0
    assert (out_dir / "manifest.json").exists()


def test_missing_dataset_resolution(tmp_path: Path, monkeypatch):
    """Verify clear FileNotFoundError when dataset directory cannot be found."""
    non_existent = tmp_path / "does_not_exist"
    monkeypatch.setenv("PDF_DATASET_DIR", str(non_existent))

    with pytest.raises(FileNotFoundError) as exc_info:
        resolve_dataset_dir(custom_path=non_existent)
    assert "does not exist" in str(exc_info.value).lower()
