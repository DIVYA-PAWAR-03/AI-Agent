"""
Repeatable CLI Command: Analyze Extracted Documents & Build Document Catalog.

Usage:
  python -m scripts.analyze_documents [--force] [--limit N] [--doc-id ID] [--verbose]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.agent.catalog_builder import CatalogBuilder, format_catalog_summary
from src.agent.llm_service import LLMConfigurationError
from src.utils.logger import setup_logger

logger = setup_logger("scripts.analyze_documents")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PDF Document Analysis & Catalog Builder (UC05)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-analysis of all documents even if previously cached",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit analysis to the first N documents (useful for testing and cost control)",
    )
    parser.add_argument(
        "--doc-id",
        type=str,
        default=None,
        help="Target a specific document ID (e.g. 'PDF_01')",
    )
    parser.add_argument(
        "--extracted-dir",
        type=str,
        default=None,
        help="Path to extracted document directory (defaults to data/extracted)",
    )
    parser.add_argument(
        "--catalog-dir",
        type=str,
        default=None,
        help="Path to catalog storage directory (defaults to data/catalog)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed debug logging",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logger("catalog_builder", log_level)
    setup_logger("llm_service", log_level)

    try:
        builder = CatalogBuilder(
            extracted_dir=Path(args.extracted_dir) if args.extracted_dir else None,
            catalog_dir=Path(args.catalog_dir) if args.catalog_dir else None,
        )
        catalog = builder.build_catalog(
            force=args.force,
            limit=args.limit,
            target_doc_id=args.doc_id,
        )
        summary = format_catalog_summary(catalog)
        print(summary)
        return 0

    except LLMConfigurationError as cfg_err:
        print("\n" + "=" * 65, file=sys.stderr)
        print("❌ LLM Configuration Required", file=sys.stderr)
        print("=" * 65, file=sys.stderr)
        print(str(cfg_err), file=sys.stderr)
        print("\nTo configure Google Gemini:", file=sys.stderr)
        print("  1. Copy .env.example to .env:  cp .env.example .env", file=sys.stderr)
        print("  2. Add your Gemini API key:   GEMINI_API_KEY=AIzaSy...", file=sys.stderr)
        print("  3. Run analysis again:        python -m scripts.analyze_documents", file=sys.stderr)
        print("=" * 65 + "\n", file=sys.stderr)
        return 1

    except FileNotFoundError as fnf_err:
        print("\n" + "=" * 65, file=sys.stderr)
        print("❌ Ingestion Prerequisite Missing", file=sys.stderr)
        print("=" * 65, file=sys.stderr)
        print(str(fnf_err), file=sys.stderr)
        print("=" * 65 + "\n", file=sys.stderr)
        return 1

    except Exception as exc:
        logger.exception("Unexpected error during document analysis: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
