"""
Pydantic schemas for structured outputs in the PDF Insight Agent.
"""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


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
