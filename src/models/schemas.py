"""
Pydantic schemas for structured outputs in the PDF Insight Agent.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
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


class FactOrMetric(BaseModel):
    """Numerical fact, statistic, date, or named entity with source attribution."""

    fact: str = Field(description="The extracted factual statement, metric, date, or statistic")
    page_number: Optional[int] = Field(default=None, description="Source page number where available")
    is_explicit_fact: bool = Field(
        default=True,
        description="True if explicitly stated in text; False if inferred or calculated",
    )


class TwoLineInsight(BaseModel):
    """Concise 2-line executive insight representing key essence."""

    doc_id: str
    line_1: str = Field(description="First insight sentence summarizing core topic/problem")
    line_2: str = Field(description="Second insight sentence highlighting key finding/impact")


class DocumentCatalogEntry(BaseModel):
    """Complete structured catalog entry for an analyzed document."""

    doc_id: str = Field(description="Document ID, e.g. PDF_01")
    filename: str = Field(description="Original PDF filename")
    file_hash: str = Field(description="SHA256 checksum of source PDF for cache invalidation")
    title: str = Field(description="Document title extracted or inferred from text")
    main_topic: str = Field(description="Main subject matter of the document")
    purpose: str = Field(description="Core purpose or objective of the document")
    category: str = Field(
        description="Primary domain category (e.g. AI & Machine Learning, Physics, Finance)"
    )
    two_line_insight: TwoLineInsight = Field(
        description="Executive 2-line high-density insight"
    )
    summary: str = Field(description="Concise, structured, and informative summary")
    key_findings: List[str] = Field(
        default_factory=list, description="Primary empirical or research findings"
    )
    conclusions: List[str] = Field(
        default_factory=list, description="Main conclusions drawn in the document"
    )
    facts_and_metrics: List[FactOrMetric] = Field(
        default_factory=list,
        description="Important statistics, dates, numerical values, and named entities",
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Explicit recommendations only when supported by the text",
    )
    risks_and_limitations: List[str] = Field(
        default_factory=list,
        description="Identified risks, caveats, or limitations supported by the text",
    )
    explicit_vs_interpretation_notes: Optional[str] = Field(
        default=None,
        description="Notes distinguishing confirmed factual findings from analytical inferences",
    )
    analyzed_at: str = Field(description="ISO timestamp of LLM analysis")
    model_used: str = Field(description="LLM model name used for analysis")
    status: str = Field(
        default="success", description="Analysis status: 'success' or 'failed'"
    )
    error: Optional[str] = Field(
        default=None, description="Error message if analysis failed"
    )


class DocumentCatalog(BaseModel):
    """Persistent catalog collection containing all analyzed document entries."""

    generated_at: str = Field(description="ISO timestamp of catalog generation")
    total_documents: int = Field(default=0, description="Total documents in corpus")
    analyzed_count: int = Field(default=0, description="Count of successfully analyzed docs")
    failed_count: int = Field(default=0, description="Count of failed analyses")
    categories: Dict[str, int] = Field(
        default_factory=dict, description="Category distribution counts"
    )
    entries: Dict[str, DocumentCatalogEntry] = Field(
        default_factory=dict, description="Catalog entries indexed by doc_id"
    )


class DocumentMetadata(BaseModel):
    """Metadata extracted or inferred for a PDF document."""

    doc_id: str = Field(description="Unique document identifier, e.g. PDF_01")
    filename: str = Field(description="Name of the PDF file")
    title: str = Field(description="Extracted or inferred title of the document")
    category: str = Field(description="High-level domain category")
    page_count: int = Field(default=1, description="Number of pages in the PDF")


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


class TextChunk(BaseModel):
    """A semantic text chunk with source provenance and metadata."""

    chunk_id: str = Field(description="Unique chunk identifier, e.g. PDF_01_p1_c0")
    doc_id: str = Field(description="Parent document identifier")
    filename: str = Field(description="Source PDF filename")
    page_number: int = Field(description="1-based page number")
    chunk_index: int = Field(description="0-based chunk index on the page")
    text: str = Field(description="Chunk text content")
    char_count: int = Field(default=0, description="Character length of chunk")
    word_count: int = Field(default=0, description="Word count of chunk")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary additional metadata (e.g. domain, category)"
    )


class RetrievedPassage(BaseModel):
    """A retrieved text passage with similarity score and citation metadata."""

    chunk_id: str = Field(description="Chunk identifier")
    doc_id: str = Field(description="Source document identifier")
    filename: str = Field(description="Source PDF filename")
    page_number: int = Field(description="1-based page number")
    text: str = Field(description="Passage text")
    score: float = Field(description="Normalized similarity score (0.0 to 1.0)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Chunk metadata")


class IndexSummary(BaseModel):
    """Metadata summary of the ChromaDB vector index."""

    collection_name: str
    total_chunks: int = 0
    total_documents: int = 0
    embedding_model: str
    persist_directory: str
    indexed_doc_ids: List[str] = Field(default_factory=list)
    generated_at: str


class AgentResponse(BaseModel):
    """Structured response from the document AI agent."""

    operation: str = Field(description="Classified operation type")
    query: str = Field(description="The original user query")
    answer: str = Field(description="The evidence-grounded response text")
    active_doc_id: Optional[str] = Field(default=None, description="The active or resolved document ID")
    target_doc_ids: List[str] = Field(default_factory=list, description="All documents referenced in the operation")
    citations: List[Citation] = Field(default_factory=list, description="Grounding source citations")
    is_directly_supported: bool = Field(
        default=True,
        description="True if the response is directly supported by evidence; False if evidence is insufficient",
    )
    distinction_notes: Optional[str] = Field(
        default=None,
        description="Distinction between direct factual statements and analytical inferences",
    )
    retrieved_passages_count: int = Field(default=0, description="Number of passages retrieved")


