"""
Index Manager: Orchestrates corpus chunking, embedding generation, incremental indexing, and manifest tracking.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from src.config import config
from src.models.schemas import DocumentExtractionRecord, IndexSummary, TextChunk
from src.utils.logger import setup_logger
from src.vectorstore.chroma import ChromaVectorStore
from src.vectorstore.chunker import chunk_document

logger = setup_logger("index_manager")


class IndexManager:
    """
    Coordinates semantic chunking, embedding indexing into ChromaDB,
    incremental caching via content hashes, and safe re-indexing.
    """

    def __init__(
        self,
        vector_store: Optional[ChromaVectorStore] = None,
        extracted_dir: Optional[Path | str] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ):
        self.vector_store = vector_store or ChromaVectorStore()
        self.extracted_dir = (
            Path(extracted_dir).resolve()
            if extracted_dir
            else config.extracted_data_dir.resolve()
        )
        self.chunk_size = chunk_size or config.chunk_size
        self.chunk_overlap = chunk_overlap or config.chunk_overlap
        self.manifest_file = self.vector_store.persist_dir / "index_manifest.json"

    def load_index_manifest(self) -> Dict[str, Any]:
        """Loads index manifest tracking indexed document hashes."""
        if not self.manifest_file.exists():
            return {}
        try:
            return json.loads(self.manifest_file.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Could not read index manifest: %s", exc)
            return {}

    def save_index_manifest(self, manifest_data: Dict[str, Any]) -> None:
        """Saves index manifest to vector store persistence directory."""
        try:
            self.manifest_file.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.warning("Could not write index manifest: %s", exc)

    def load_extracted_documents(self) -> List[DocumentExtractionRecord]:
        """Loads extracted documents from ingestion storage."""
        docs_dir = self.extracted_dir / "documents"
        if not docs_dir.exists():
            raise FileNotFoundError(
                f"Extracted documents directory not found at '{docs_dir}'. "
                "Please run 'python -m scripts.ingest_pdfs' before indexing."
            )

        records: List[DocumentExtractionRecord] = []
        for file_path in sorted(docs_dir.glob("*.json")):
            try:
                data = json.loads(file_path.read_text(encoding="utf-8"))
                records.append(DocumentExtractionRecord(**data))
            except Exception as exc:
                logger.warning("Skipping unreadable extraction file %s: %s", file_path.name, exc)

        if not records:
            raise FileNotFoundError(
                f"No extracted documents found in '{docs_dir}'. "
                "Please run 'python -m scripts.ingest_pdfs' before indexing."
            )

        return records

    def build_index(
        self,
        force: bool = False,
        limit: Optional[int] = None,
        target_doc_id: Optional[str] = None,
    ) -> IndexSummary:
        """
        Processes extracted documents into chunks and indexes them into ChromaDB.
        Idempotent: skips unchanged documents based on file_hash unless force=True.
        """
        doc_records = self.load_extracted_documents()
        manifest = self.load_index_manifest()

        if target_doc_id:
            doc_records = [r for r in doc_records if r.doc_id.lower() == target_doc_id.lower()]
            if not doc_records:
                raise ValueError(f"Document ID '{target_doc_id}' not found in extracted corpus.")

        if limit and limit > 0:
            doc_records = doc_records[:limit]

        total_docs = len(doc_records)
        logger.info(
            "Starting vector indexing for %d documents (force=%s, chunk_size=%d, overlap=%d)",
            total_docs,
            force,
            self.chunk_size,
            self.chunk_overlap,
        )

        indexed_docs = 0
        skipped_docs = 0
        total_chunks_added = 0

        for idx, record in enumerate(doc_records, start=1):
            doc_id = record.doc_id

            # Incremental check: if previously indexed and hash matches
            if (
                not force
                and doc_id in manifest
                and manifest[doc_id].get("file_hash") == record.file_hash
            ):
                logger.debug("Skipping unchanged, indexed document: %s", doc_id)
                skipped_docs += 1
                continue

            logger.info("[%d/%d] Chunking and indexing: %s (%s)", idx, total_docs, doc_id, record.filename)

            # Generate chunks
            chunks = chunk_document(
                record,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )

            # Clean existing chunks for this document to avoid duplication
            self.vector_store.delete_document(doc_id)

            # Add to vector store
            if chunks:
                added = self.vector_store.add_chunks(chunks)
                total_chunks_added += added

            # Update manifest
            manifest[doc_id] = {
                "filename": record.filename,
                "file_hash": record.file_hash,
                "chunk_count": len(chunks),
                "indexed_at": datetime.now(timezone.utc).isoformat(),
            }
            indexed_docs += 1

        self.save_index_manifest(manifest)

        summary = IndexSummary(
            collection_name=self.vector_store.collection_name,
            total_chunks=self.vector_store.count(),
            total_documents=len(manifest),
            embedding_model=getattr(self.vector_store.embedding_provider, "model_name", "custom"),
            persist_directory=str(self.vector_store.persist_dir),
            indexed_doc_ids=sorted(list(manifest.keys())),
            generated_at=datetime.now(timezone.utc).isoformat(),
        )

        logger.info(
            "Indexing complete: %d newly indexed, %d skipped. Total chunks in collection: %d",
            indexed_docs,
            skipped_docs,
            summary.total_chunks,
        )

        return summary


def format_index_summary(summary: IndexSummary) -> str:
    """Formats human-readable summary of the vector store index."""
    lines = [
        "=" * 65,
        " Vector Indexing & RAG Knowledge Base Summary",
        "=" * 65,
        f"Collection Name:     {summary.collection_name}",
        f"Total Chunks:        {summary.total_chunks}",
        f"Total Documents:     {summary.total_documents}",
        f"Embedding Model:     {summary.embedding_model}",
        f"Storage Directory:   {summary.persist_directory}",
        f"Generated At:        {summary.generated_at}",
        f"Sample Indexed IDs:  {', '.join(summary.indexed_doc_ids[:8])}{'...' if len(summary.indexed_doc_ids) > 8 else ''}",
        "=" * 65,
    ]
    return "\n".join(lines)
