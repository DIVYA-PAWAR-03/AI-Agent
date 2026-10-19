"""
PDF Document Ingestion & Text Extraction Package.
"""

from src.ingestion.loader import (
    clean_extracted_text,
    compute_file_hash,
    extract_pdf_document,
    extract_pdf_page,
)
from src.ingestion.pipeline import (
    IngestionPipeline,
    discover_pdfs,
    format_ingestion_summary,
    generate_doc_id,
)

__all__ = [
    "clean_extracted_text",
    "compute_file_hash",
    "extract_pdf_document",
    "extract_pdf_page",
    "IngestionPipeline",
    "discover_pdfs",
    "format_ingestion_summary",
    "generate_doc_id",
]
