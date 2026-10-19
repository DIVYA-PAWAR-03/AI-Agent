"""
PyMuPDF document loader, text extraction, and text cleaning utilities.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
import re
from typing import Optional

import pymupdf

from src.models.schemas import DocumentExtractionRecord, PageExtraction
from src.utils.logger import setup_logger

logger = setup_logger("pdf_loader")


def compute_file_hash(file_path: Path) -> str:
    """Computes SHA256 checksum of a file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def clean_extracted_text(text: str) -> str:
    """
    Cleans raw extracted text from PDF pages:
    - Normalizes unicode whitespace (non-breaking spaces, form feeds).
    - Normalizes line breaks (\r\n -> \n).
    - Removes null bytes and unprintable ASCII control characters.
    - Collapses excessive blank lines (3+ consecutive newlines -> 2).
    """
    if not text:
        return ""

    # Remove null bytes
    cleaned = text.replace("\x00", "")

    # Normalize line breaks
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")

    # Replace special unicode spaces with standard space
    cleaned = re.sub(r"[\u00a0\u1680\u2000-\u200b\u202f\u205f\u3000]", " ", cleaned)

    # Collapse multiple spaces on the same line (preserve newlines)
    cleaned = re.sub(r"[^\S\n]+", " ", cleaned)

    # Collapse excessive blank lines
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    return cleaned.strip()


def extract_pdf_page(page: pymupdf.Page, page_number: int) -> PageExtraction:
    """
    Extracts text and metadata from a single PDF page with error handling.
    """
    try:
        raw_text = page.get_text("text") or ""
        cleaned = clean_extracted_text(raw_text)

        # Inspect if page contains embedded raster images
        images = page.get_images(full=True)
        has_images = len(images) > 0

        # Flag OCR need if page has virtually no text but contains images
        char_count = len(cleaned)
        word_count = len(cleaned.split()) if cleaned else 0
        needs_ocr = char_count == 0 and has_images

        return PageExtraction(
            page_number=page_number,
            text=cleaned,
            char_count=char_count,
            word_count=word_count,
            has_images=has_images,
            needs_ocr=needs_ocr,
            error=None,
        )
    except Exception as exc:
        logger.warning("Error extracting page %d: %s", page_number, exc)
        return PageExtraction(
            page_number=page_number,
            text="",
            char_count=0,
            word_count=0,
            has_images=False,
            needs_ocr=False,
            error=str(exc),
        )


def extract_pdf_document(
    pdf_path: Path, doc_id: str, relative_path: str = ""
) -> DocumentExtractionRecord:
    """
    Extracts complete text and metadata for a PDF document page-by-page.
    Detects empty PDFs, unreadable files, corrupt pages, and OCR requirements.
    """
    now_iso = datetime.now(timezone.utc).isoformat()
    rel_path = relative_path or pdf_path.name

    # Check file existence
    if not pdf_path.exists():
        return DocumentExtractionRecord(
            doc_id=doc_id,
            filename=pdf_path.name,
            relative_path=rel_path,
            file_size_bytes=0,
            file_hash="",
            page_count=0,
            extracted_pages_count=0,
            status="failed",
            error_message="File does not exist",
            extracted_at=now_iso,
        )

    file_size = pdf_path.stat().st_size
    if file_size == 0:
        return DocumentExtractionRecord(
            doc_id=doc_id,
            filename=pdf_path.name,
            relative_path=rel_path,
            file_size_bytes=0,
            file_hash="",
            page_count=0,
            extracted_pages_count=0,
            status="failed",
            error_message="Empty file (0 bytes)",
            extracted_at=now_iso,
        )

    file_hash = compute_file_hash(pdf_path)

    # Attempt opening document with PyMuPDF
    doc: Optional[pymupdf.Document] = None
    try:
        doc = pymupdf.open(pdf_path)
    except Exception as exc:
        logger.error("Failed to open PDF %s: %s", pdf_path.name, exc)
        return DocumentExtractionRecord(
            doc_id=doc_id,
            filename=pdf_path.name,
            relative_path=rel_path,
            file_size_bytes=file_size,
            file_hash=file_hash,
            page_count=0,
            extracted_pages_count=0,
            status="failed",
            error_message=f"Failed to open/parse PDF: {exc}",
            extracted_at=now_iso,
        )

    try:
        page_count = len(doc)
        if page_count == 0:
            return DocumentExtractionRecord(
                doc_id=doc_id,
                filename=pdf_path.name,
                relative_path=rel_path,
                file_size_bytes=file_size,
                file_hash=file_hash,
                page_count=0,
                extracted_pages_count=0,
                status="failed",
                error_message="PDF contains 0 pages",
                extracted_at=now_iso,
            )

        extracted_pages = []
        total_chars = 0
        total_words = 0
        ocr_flagged_pages = 0

        for idx, page in enumerate(doc):
            page_num = idx + 1
            page_rec = extract_pdf_page(page, page_num)
            extracted_pages.append(page_rec)
            total_chars += page_rec.char_count
            total_words += page_rec.word_count
            if page_rec.needs_ocr:
                ocr_flagged_pages += 1

        # Determine overall document status and OCR requirement
        doc_needs_ocr = False
        error_msg: Optional[str] = None

        if total_chars == 0:
            doc_needs_ocr = True
            status = "needs_ocr"
            error_msg = (
                "No extractable text detected across any pages. "
                "Document appears to be scanned or image-based and requires OCR."
            )
        elif ocr_flagged_pages > 0 and (total_chars / max(1, page_count)) < 50:
            doc_needs_ocr = True
            status = "needs_ocr"
            error_msg = (
                f"{ocr_flagged_pages}/{page_count} pages contain images with zero extractable text. "
                "OCR may be required for complete extraction."
            )
        else:
            status = "success"

        return DocumentExtractionRecord(
            doc_id=doc_id,
            filename=pdf_path.name,
            relative_path=rel_path,
            file_size_bytes=file_size,
            file_hash=file_hash,
            page_count=page_count,
            extracted_pages_count=len(extracted_pages),
            total_char_count=total_chars,
            total_word_count=total_words,
            status=status,
            needs_ocr=doc_needs_ocr,
            error_message=error_msg,
            extracted_at=now_iso,
            pages=extracted_pages,
        )
    finally:
        if doc is not None:
            doc.close()
