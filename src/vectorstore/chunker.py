"""
Semantic text chunker for PDF pages with provenance preservation and overlap.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from src.config import config
from src.models.schemas import DocumentExtractionRecord, PageExtraction, TextChunk
from src.utils.logger import setup_logger

logger = setup_logger("text_chunker")


def split_text_recursively(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
    separators: Optional[List[str]] = None,
) -> List[str]:
    """
    Splits text into chunks respecting natural linguistic boundaries (paragraphs, sentences, words).
    Maintains an overlap window between consecutive chunks.
    """
    if not text or not text.strip():
        return []

    if separators is None:
        separators = ["\n\n", "\n", ". ", "? ", "! ", "; ", " ", ""]

    if chunk_overlap >= chunk_size:
        chunk_overlap = max(0, chunk_size // 4)

    # Base case: text is already smaller than chunk_size
    if len(text) <= chunk_size:
        return [text.strip()]

    # Find the highest priority separator present in the text
    chosen_separator = ""
    for sep in separators:
        if sep in text:
            chosen_separator = sep
            break

    # Split text using the chosen separator
    if chosen_separator:
        parts = text.split(chosen_separator)
    else:
        # Fallback character slice
        parts = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size - chunk_overlap)]
        return [p.strip() for p in parts if p.strip()]

    chunks: List[str] = []
    current_chunk: List[str] = []
    current_length = 0

    for part in parts:
        part_len = len(part) + (len(chosen_separator) if current_chunk else 0)

        if current_length + part_len <= chunk_size:
            current_chunk.append(part)
            current_length += part_len
        else:
            if current_chunk:
                chunk_str = chosen_separator.join(current_chunk).strip()
                if chunk_str:
                    chunks.append(chunk_str)

                # Build overlap from the end of the previous chunk
                overlap_tokens: List[str] = []
                overlap_len = 0
                for prev_part in reversed(current_chunk):
                    if overlap_len + len(prev_part) <= chunk_overlap:
                        overlap_tokens.insert(0, prev_part)
                        overlap_len += len(prev_part) + len(chosen_separator)
                    else:
                        break
                current_chunk = overlap_tokens
                current_length = overlap_len

            # If a single part exceeds chunk_size, split it with sub-separators
            if len(part) > chunk_size:
                sub_seps = [s for s in separators if s != chosen_separator]
                sub_chunks = split_text_recursively(
                    part,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    separators=sub_seps,
                )
                chunks.extend(sub_chunks)
                current_chunk = []
                current_length = 0
            else:
                current_chunk.append(part)
                current_length += len(part)

    if current_chunk:
        chunk_str = chosen_separator.join(current_chunk).strip()
        if chunk_str and (not chunks or chunk_str != chunks[-1]):
            chunks.append(chunk_str)

    return chunks


def chunk_document(
    record: DocumentExtractionRecord,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> List[TextChunk]:
    """
    Chunks all pages of an extracted document, retaining document ID, filename,
    page number, chunk ID, and metadata.
    """
    size = chunk_size or config.chunk_size
    overlap = chunk_overlap or config.chunk_overlap
    meta_base = extra_metadata or {}

    chunks: List[TextChunk] = []

    for page in record.pages:
        page_text = page.text.strip()
        if not page_text:
            continue

        raw_chunks = split_text_recursively(
            page_text,
            chunk_size=size,
            chunk_overlap=overlap,
        )

        for idx, text_segment in enumerate(raw_chunks):
            chunk_id = f"{record.doc_id}_p{page.page_number}_c{idx}"
            chunk = TextChunk(
                chunk_id=chunk_id,
                doc_id=record.doc_id,
                filename=record.filename,
                page_number=page.page_number,
                chunk_index=idx,
                text=text_segment,
                char_count=len(text_segment),
                word_count=len(text_segment.split()),
                metadata={
                    **meta_base,
                    "doc_id": record.doc_id,
                    "filename": record.filename,
                    "page_number": page.page_number,
                    "chunk_id": chunk_id,
                },
            )
            chunks.append(chunk)

    return chunks
