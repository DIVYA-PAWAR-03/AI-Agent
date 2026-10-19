"""
PDF Ingestion Pipeline: discovery, stable ID assignment, extraction, and manifest generation.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Dict, List, Optional, Set, Tuple

from src.config import config, resolve_dataset_dir
from src.ingestion.loader import compute_file_hash, extract_pdf_document
from src.models.schemas import DocumentExtractionRecord, IngestionManifest
from src.utils.logger import setup_logger

logger = setup_logger("ingestion_pipeline")


def discover_pdfs(input_dir: Path) -> List[Path]:
    """
    Recursively discovers all PDF files within the specified directory.
    Sorts files for deterministic processing order.
    """
    if not input_dir.exists() or not input_dir.is_dir():
        return []
    # Discover case-insensitively for *.pdf
    discovered = [
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() == ".pdf"
    ]
    return sorted(discovered, key=lambda p: str(p).lower())


def generate_doc_id(
    pdf_path: Path,
    base_dir: Path,
    index: int,
    existing_ids: Set[str],
) -> str:
    """
    Generates a human-friendly, deterministic, unique document identifier.
    Preserves existing naming patterns like PDF_01 from the dataset.
    Disambiguates duplicate filenames from different subdirectories.
    """
    filename = pdf_path.stem
    match = re.match(r"^(PDF_\d+)", filename, re.IGNORECASE)
    if match:
        candidate_id = match.group(1).upper()
    else:
        # Fallback to sanitized stem or indexed ID
        sanitized = re.sub(r"[^\w\-]", "_", filename)[:24].strip("_")
        candidate_id = f"DOC_{index:03d}_{sanitized}"

    # Ensure uniqueness across duplicate filenames in nested paths
    if candidate_id not in existing_ids:
        existing_ids.add(candidate_id)
        return candidate_id

    # If collision occurs, disambiguate with directory path or counter
    disambiguator = 2
    while f"{candidate_id}_{disambiguator}" in existing_ids:
        disambiguator += 1

    final_id = f"{candidate_id}_{disambiguator}"
    existing_ids.add(final_id)
    return final_id


class IngestionPipeline:
    """
    Orchestrates the discovery, extraction, storage, and manifest creation for PDFs.
    """

    def __init__(
        self,
        input_dir: Optional[Path | str] = None,
        output_dir: Optional[Path | str] = None,
    ):
        if input_dir:
            self.input_dir = Path(input_dir).resolve()
        else:
            self.input_dir = resolve_dataset_dir(config.pdf_dataset_dir)

        self.output_dir = (
            Path(output_dir).resolve()
            if output_dir
            else config.extracted_data_dir.resolve()
        )
        self.docs_dir = self.output_dir / "documents"

    def _ensure_output_dirs(self) -> None:
        """Creates output folders if they do not exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.docs_dir.mkdir(parents=True, exist_ok=True)

    def _load_existing_record(self, doc_id: str) -> Optional[DocumentExtractionRecord]:
        """Loads a previously saved extraction record if available."""
        doc_path = self.docs_dir / f"{doc_id}.json"
        if not doc_path.exists():
            return None
        try:
            data = json.loads(doc_path.read_text(encoding="utf-8"))
            return DocumentExtractionRecord(**data)
        except Exception as exc:
            logger.warning("Could not read cached document %s: %s", doc_id, exc)
            return None

    def run(self, force: bool = False) -> IngestionManifest:
        """
        Executes the ingestion pipeline.
        - Discovers all PDFs recursively.
        - Checks idempotency: skips unchanged files unless force=True.
        - Extracts text page-by-page.
        - Detects unreadable files and OCR needs.
        - Saves document records and master manifest.
        """
        self._ensure_output_dirs()
        discovered_paths = discover_pdfs(self.input_dir)
        total_discovered = len(discovered_paths)

        logger.info(
            "Starting ingestion from %s (Discovered %d PDFs)",
            self.input_dir,
            total_discovered,
        )

        existing_ids: Set[str] = set()
        document_records: List[DocumentExtractionRecord] = []

        total_processed = 0
        total_skipped = 0
        total_failed = 0
        total_pages = 0
        total_ocr_needed = 0

        for idx, pdf_path in enumerate(discovered_paths, start=1):
            rel_path = str(pdf_path.relative_to(self.input_dir))
            doc_id = generate_doc_id(pdf_path, self.input_dir, idx, existing_ids)

            # Idempotency check: if already processed and hash unchanged
            if not force:
                current_hash = compute_file_hash(pdf_path) if pdf_path.stat().st_size > 0 else ""
                existing_record = self._load_existing_record(doc_id)
                if (
                    existing_record is not None
                    and existing_record.file_hash == current_hash
                    and existing_record.status in ("success", "needs_ocr")
                ):
                    logger.debug("Skipping unchanged document %s (%s)", doc_id, rel_path)
                    existing_record.status = "skipped"
                    document_records.append(existing_record)
                    total_skipped += 1
                    total_pages += existing_record.page_count
                    if existing_record.needs_ocr:
                        total_ocr_needed += 1
                    continue

            # Perform extraction
            logger.info("Extracting [%d/%d] %s -> %s", idx, total_discovered, rel_path, doc_id)
            record = extract_pdf_document(pdf_path, doc_id=doc_id, relative_path=rel_path)

            # Persist individual document record
            doc_file = self.docs_dir / f"{doc_id}.json"
            doc_file.write_text(record.model_dump_json(indent=2), encoding="utf-8")

            document_records.append(record)

            if record.status == "success":
                total_processed += 1
                total_pages += record.page_count
            elif record.status == "needs_ocr":
                total_processed += 1
                total_ocr_needed += 1
                total_pages += record.page_count
            else:
                total_failed += 1

        manifest = IngestionManifest(
            generated_at=datetime.now(timezone.utc).isoformat(),
            input_directory=str(self.input_dir),
            output_directory=str(self.output_dir),
            total_discovered=total_discovered,
            total_processed=total_processed,
            total_skipped=total_skipped,
            total_failed=total_failed,
            total_pages=total_pages,
            total_ocr_needed=total_ocr_needed,
            documents=document_records,
        )

        # Save master manifest
        manifest_path = self.output_dir / "manifest.json"
        manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        logger.info("Saved ingestion manifest to %s", manifest_path)

        return manifest


def format_ingestion_summary(manifest: IngestionManifest) -> str:
    """Formats a human-readable summary of the ingestion run."""
    lines = [
        "=" * 60,
        " PDF Ingestion Pipeline Summary",
        "=" * 60,
        f"Input Directory:         {manifest.input_directory}",
        f"Output Directory:        {manifest.output_directory}",
        f"Total PDFs Discovered:   {manifest.total_discovered}",
        f"Successfully Extracted:  {manifest.total_processed}",
        f"Skipped (Up-to-Date):    {manifest.total_skipped}",
        f"Failed / Unreadable:     {manifest.total_failed}",
        f"Total Pages Extracted:   {manifest.total_pages}",
        f"Documents Requiring OCR: {manifest.total_ocr_needed}",
        f"Manifest File:           {manifest.output_directory}/manifest.json",
        "=" * 60,
    ]
    if manifest.total_ocr_needed > 0:
        lines.append("\n⚠️ Warning: The following documents may require OCR:")
        for doc in manifest.documents:
            if doc.needs_ocr:
                lines.append(f"  - {doc.doc_id}: {doc.filename} ({doc.error_message})")
        lines.append("=" * 60)

    if manifest.total_failed > 0:
        lines.append("\n❌ The following documents failed extraction:")
        for doc in manifest.documents:
            if doc.status == "failed":
                lines.append(f"  - {doc.doc_id}: {doc.filename} ({doc.error_message})")
        lines.append("=" * 60)

    return "\n".join(lines)
