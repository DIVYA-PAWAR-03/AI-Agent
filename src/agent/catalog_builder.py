"""
Catalog Builder: Orchestrates corpus analysis, caching/idempotency, and catalog persistence.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Dict, List, Optional

from src.agent.llm_service import LLMConfigurationError, LLMService
from src.config import config
from src.models.schemas import (
    DocumentCatalog,
    DocumentCatalogEntry,
    DocumentExtractionRecord,
    IngestionManifest,
)
from src.utils.logger import setup_logger

logger = setup_logger("catalog_builder")


class CatalogBuilder:
    """
    Manages the lifecycle of document insight analysis and catalog caching.
    """

    def __init__(
        self,
        llm_service: Optional[LLMService] = None,
        extracted_dir: Optional[Path | str] = None,
        catalog_dir: Optional[Path | str] = None,
    ):
        self.llm_service = llm_service or LLMService()
        self.extracted_dir = (
            Path(extracted_dir).resolve()
            if extracted_dir
            else config.extracted_data_dir.resolve()
        )
        self.catalog_dir = (
            Path(catalog_dir).resolve()
            if catalog_dir
            else config.catalog_dir.resolve()
        )
        self.catalog_file = self.catalog_dir / "document_catalog.json"
        self.entries_dir = self.catalog_dir / "entries"

    def _ensure_dirs(self) -> None:
        """Ensures storage directories exist."""
        self.catalog_dir.mkdir(parents=True, exist_ok=True)
        self.entries_dir.mkdir(parents=True, exist_ok=True)

    def load_catalog(self) -> DocumentCatalog:
        """Loads existing catalog from disk or initializes an empty catalog."""
        if not self.catalog_file.exists():
            return DocumentCatalog(
                generated_at=datetime.now(timezone.utc).isoformat(),
                total_documents=0,
                analyzed_count=0,
                failed_count=0,
                categories={},
                entries={},
            )
        try:
            data = json.loads(self.catalog_file.read_text(encoding="utf-8"))
            return DocumentCatalog(**data)
        except Exception as exc:
            logger.warning("Could not parse existing catalog from %s: %s", self.catalog_file, exc)
            return DocumentCatalog(
                generated_at=datetime.now(timezone.utc).isoformat(),
                total_documents=0,
                analyzed_count=0,
                failed_count=0,
                categories={},
                entries={},
            )

    def save_catalog(self, catalog: DocumentCatalog) -> None:
        """Saves master catalog and individual document entry JSON files."""
        self._ensure_dirs()
        catalog.generated_at = datetime.now(timezone.utc).isoformat()

        # Recalculate category distribution and counts
        categories: Counter[str] = Counter()
        analyzed_count = 0
        failed_count = 0

        for entry in catalog.entries.values():
            if entry.status == "success":
                categories[entry.category] += 1
                analyzed_count += 1
            else:
                failed_count += 1

            # Save individual document entry for modular access
            entry_file = self.entries_dir / f"{entry.doc_id}.json"
            entry_file.write_text(entry.model_dump_json(indent=2), encoding="utf-8")

        catalog.categories = dict(categories)
        catalog.analyzed_count = analyzed_count
        catalog.failed_count = failed_count
        catalog.total_documents = len(catalog.entries)

        # Write master catalog
        self.catalog_file.write_text(catalog.model_dump_json(indent=2), encoding="utf-8")
        logger.info("Saved catalog to %s (%d entries)", self.catalog_file, len(catalog.entries))

    def load_extracted_documents(self) -> List[DocumentExtractionRecord]:
        """Loads extracted documents from ingestion storage."""
        docs_dir = self.extracted_dir / "documents"
        if not docs_dir.exists():
            raise FileNotFoundError(
                f"Extracted documents directory not found at '{docs_dir}'. "
                "Please run 'python -m scripts.ingest_pdfs' before analyzing documents."
            )

        records: List[DocumentExtractionRecord] = []
        for file_path in sorted(docs_dir.glob("*.json")):
            try:
                data = json.loads(file_path.read_text(encoding="utf-8"))
                records.append(DocumentExtractionRecord(**data))
            except Exception as exc:
                logger.warning("Skipping invalid extraction record %s: %s", file_path.name, exc)

        if not records:
            raise FileNotFoundError(
                f"No extracted document JSON files found in '{docs_dir}'. "
                "Please run 'python -m scripts.ingest_pdfs' before analyzing documents."
            )

        return records

    def build_catalog(
        self,
        force: bool = False,
        limit: Optional[int] = None,
        target_doc_id: Optional[str] = None,
        rate_limit_delay: float = 1.0,
    ) -> DocumentCatalog:
        """
        Executes document analysis over extracted records with incremental caching.
        """
        self._ensure_dirs()
        catalog = self.load_catalog()
        doc_records = self.load_extracted_documents()

        if target_doc_id:
            doc_records = [r for r in doc_records if r.doc_id.lower() == target_doc_id.lower()]
            if not doc_records:
                raise ValueError(f"Document ID '{target_doc_id}' not found in extracted corpus.")

        if limit and limit > 0:
            doc_records = doc_records[:limit]

        total_to_process = len(doc_records)
        logger.info("Building catalog for %d documents (force=%s)", total_to_process, force)

        skipped_count = 0
        analyzed_count = 0
        failed_count = 0

        for idx, record in enumerate(doc_records, start=1):
            doc_id = record.doc_id

            # Incremental caching check: if previously analyzed with identical hash and success
            if not force and doc_id in catalog.entries:
                existing_entry = catalog.entries[doc_id]
                if (
                    existing_entry.file_hash == record.file_hash
                    and existing_entry.status == "success"
                ):
                    logger.debug("Skipping unchanged, cached document: %s", doc_id)
                    skipped_count += 1
                    continue

            logger.info("[%d/%d] Analyzing document: %s (%s)", idx, total_to_process, doc_id, record.filename)

            try:
                entry = self.llm_service.analyze_document(record)
                catalog.entries[doc_id] = entry
                if entry.status == "success":
                    analyzed_count += 1
                else:
                    failed_count += 1

                # Persist incrementally to avoid losing progress
                self.save_catalog(catalog)

            except LLMConfigurationError:
                # Bubble up configuration errors immediately to inform user
                raise
            except Exception as exc:
                logger.error("Error analyzing %s: %s", doc_id, exc)
                failed_count += 1

        self.save_catalog(catalog)
        return catalog


def format_catalog_summary(catalog: DocumentCatalog) -> str:
    """Formats a human-readable summary of the document catalog."""
    lines = [
        "=" * 65,
        " Document Analysis & Catalog Summary",
        "=" * 65,
        f"Generated At:       {catalog.generated_at}",
        f"Total In Catalog:   {catalog.total_documents}",
        f"Successfully Analyzed: {catalog.analyzed_count}",
        f"Failed Analyses:    {catalog.failed_count}",
        "",
        "Domain Category Breakdown:",
    ]
    for cat, count in sorted(catalog.categories.items(), key=lambda x: -x[1]):
        lines.append(f"  • {cat:40s} : {count} documents")

    if catalog.failed_count > 0:
        lines.append("\n⚠️ Failed Document Analyses:")
        for doc_id, entry in catalog.entries.items():
            if entry.status != "success":
                lines.append(f"  - {doc_id} ({entry.filename}): {entry.error}")

    lines.append("=" * 65)
    return "\n".join(lines)
