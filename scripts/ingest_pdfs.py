"""
Repeatable PDF Ingestion CLI Command:
Executes recursive discovery, text extraction, OCR detection, and manifest generation.

Usage:
  python -m scripts.ingest_pdfs [--input-dir PATH] [--output-dir PATH] [--force] [--verbose]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.ingestion.pipeline import IngestionPipeline, format_ingestion_summary
from src.utils.logger import setup_logger

logger = setup_logger("scripts.ingest_pdfs")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PDF Document Ingestion & Extraction Pipeline (UC05)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Path to directory containing PDF files (defaults to data/pdfs or configured path)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Path to output storage directory for extracted text and manifest (defaults to data/extracted)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-extraction of all documents even if previously cached",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed debug logging during extraction",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logger("ingestion_pipeline", log_level)
    setup_logger("pdf_loader", log_level)

    try:
        pipeline = IngestionPipeline(
            input_dir=Path(args.input_dir) if args.input_dir else None,
            output_dir=Path(args.output_dir) if args.output_dir else None,
        )
        manifest = pipeline.run(force=args.force)
        summary = format_ingestion_summary(manifest)
        print(summary)
        return 0
    except FileNotFoundError as err:
        print("\n" + "=" * 60, file=sys.stderr)
        print("❌ Dataset Location Error", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        print(str(err), file=sys.stderr)
        print("=" * 60 + "\n", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Fatal error during PDF ingestion: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
