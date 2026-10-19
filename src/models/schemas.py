"""
Pydantic schemas for structured outputs in the PDF Insight Agent.
"""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class PageExtraction(BaseModel):
    """Extracted text and metadata for an individual PDF page."""

    page_number: int = Field(description="1-indexed page number")
    text: str = Field(description="Cleaned extracted text of the page")
    char_count: int = Field(default=0, description="Number of characters in the page text")
    word_count: int = Field(default=0, description="Number of words in the page text")
    has_images: bool = Field(default=False, description="Whether the page contains embedded images")
    needs_ocr: bool = Field(default=False, description="Flag indicating page may require OCR")
    error: Optional[str] = Field(default=None, description="Error message if page extraction failed")


class DocumentExtractionRecord(BaseModel):
    """Complete extraction record for a single PDF document."""

    doc_id: str = Field(description="Stable document identifier, e.g. PDF_01")
    filename: str = Field(description="PDF file name")
    relative_path: str = Field(description="Relative path from ingestion root")
    file_size_bytes: int = Field(description="File size in bytes")
    file_hash: str = Field(description="SHA256 checksum of the file content")
    page_count: int = Field(default=0, description="Total number of pages reported by PDF")
    extracted_pages_count: int = Field(default=0, description="Number of pages successfully parsed")
    total_char_count: int = Field(default=0, description="Total characters extracted across all pages")
    total_word_count: int = Field(default=0, description="Total words extracted across all pages")
    status: str = Field(
        default="pending",
        description="Extraction status: 'success', 'failed', 'needs_ocr', or 'skipped'",
    )
    needs_ocr: bool = Field(default=False, description="True if document likely requires OCR")
    error_message: Optional[str] = Field(default=None, description="Failure details if applicable")
    extracted_at: str = Field(description="ISO timestamp of extraction")
    pages: List[PageExtraction] = Field(default_factory=list, description="Extracted pages")


class IngestionManifest(BaseModel):
    """Manifest summarizing the entire PDF ingestion run."""

    generated_at: str
    input_directory: str
    output_directory: str
    total_discovered: int = 0
    total_processed: int = 0
    total_skipped: int = 0
    total_failed: int = 0
    total_pages: int = 0
    total_ocr_needed: int = 0
    documents: List[DocumentExtractionRecord] = Field(default_factory=list)


class DocumentMetadata(BaseModel):
    """Metadata extracted or inferred for a PDF document."""

    doc_id: str = Field(description="Unique document identifier, e.g. PDF_01")
    filename: str = Field(description="Name of the PDF file")
    title: str = Field(description="Extracted or inferred title of the document")
    category: str = Field(description="High-level domain category")
    page_count: int = Field(default=1, description="Number of pages in the PDF")


class TwoLineInsight(BaseModel):
    """Concise 2-line executive insight representing key essence."""

    doc_id: str
    line_1: str = Field(description="First insight sentence summarizing core topic/problem")
    line_2: str = Field(description="Second insight sentence highlighting key finding/impact")


class DocumentSummary(BaseModel):
    """Detailed structured summary of a single document."""

    doc_id: str
    title: str
    category: str
    main_topic: str
    purpose: str
    executive_summary: str
    key_findings: List[str] = Field(default_factory=list)
    facts_and_metrics: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    conclusions: List[str] = Field(default_factory=list)


class Citation(BaseModel):
    """Evidence citation linking an answer directly to document contents."""

    doc_id: str
    page_number: Optional[int] = None
    snippet: str = Field(description="Verbatim or close excerpt from the source PDF")


class GroundedAnswer(BaseModel):
    """Grounded answer to user query with explicit evidence support."""

    query: str
    answer: str
    citations: List[Citation] = Field(default_factory=list)
    is_directly_supported: bool = Field(
        default=True,
        description="True if the answer is grounded in PDF documents, False if information is lacking",
    )
    unsupported_elements_or_notes: Optional[str] = None
