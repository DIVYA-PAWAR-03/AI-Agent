"""
Repeatable CLI Command: Build ChromaDB Vector Index & Perform Semantic Retrieval.

Usage:
  python -m scripts.build_index [--force] [--chunk-size N] [--chunk-overlap N]
  python -m scripts.build_index --query "What is quantum superposition?" [--filter-docs PDF_01,PDF_15]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.utils.logger import setup_logger
from src.vectorstore.chroma import ChromaVectorStore
from src.vectorstore.indexer import IndexManager, format_index_summary

logger = setup_logger("scripts.build_index")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ChromaDB Vector Indexing & Semantic Search (UC05)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-chunking and re-indexing of all documents",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Chunk size in characters (defaults to config: 800)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=None,
        help="Chunk overlap in characters (defaults to config: 150)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Index only the first N documents",
    )
    parser.add_argument(
        "--doc-id",
        type=str,
        default=None,
        help="Target a specific document ID to index",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Execute an immediate semantic search query against the index",
    )
    parser.add_argument(
        "--filter-docs",
        type=str,
        default=None,
        help="Comma-separated document IDs to filter query (e.g. 'PDF_01,PDF_02')",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of retrieved passages to return for query",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.2,
        help="Minimum similarity score threshold for query",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed debug logging",
    )
    return parser.parse_args(argv)


def run_query_search(
    vector_store: ChromaVectorStore,
    query_text: str,
    filter_docs: list[str] | None,
    top_k: int,
    min_score: float,
) -> None:
    """Executes search query and prints formatted passage citations."""
    print("\n" + "=" * 65)
    print(f" Semantic Search Query: '{query_text}'")
    if filter_docs:
        print(f" Document Filter:      {', '.join(filter_docs)}")
    print(f" Top-K: {top_k} | Min Score: {min_score}")
    print("=" * 65)

    passages = vector_store.query(
        query_text=query_text,
        top_k=top_k,
        doc_ids=filter_docs,
        min_score=min_score,
    )

    if not passages:
        print("ℹ️ No passages met the similarity criteria or matched the filter.")
        print("=" * 65 + "\n")
        return

    for rank, p in enumerate(passages, start=1):
        print(f"\n[{rank}] Source: {p.doc_id} | Page {p.page_number} | Score: {p.score:.4f} | Chunk: {p.chunk_id}")
        print(f"    File: {p.filename}")
        print("    " + "-" * 55)
        # Indent passage text
        snippet = p.text[:300] + ("..." if len(p.text) > 300 else "")
        for line in snippet.split("\n"):
            print(f"    {line}")

    print("\n" + "=" * 65 + "\n")


def main(argv=None) -> int:
    args = parse_args(argv)
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logger("index_manager", log_level)
    setup_logger("chroma_store", log_level)
    setup_logger("embeddings", log_level)

    try:
        manager = IndexManager(
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )

        # Build / update index
        summary = manager.build_index(
            force=args.force,
            limit=args.limit,
            target_doc_id=args.doc_id,
        )
        print(format_index_summary(summary))

        # Optional query test
        if args.query:
            filter_ids = (
                [d.strip() for d in args.filter_docs.split(",") if d.strip()]
                if args.filter_docs
                else None
            )
            run_query_search(
                vector_store=manager.vector_store,
                query_text=args.query,
                filter_docs=filter_ids,
                top_k=args.top_k,
                min_score=args.min_score,
            )

        return 0

    except FileNotFoundError as fnf_err:
        print("\n" + "=" * 65, file=sys.stderr)
        print("❌ Prerequisite Missing", file=sys.stderr)
        print("=" * 65, file=sys.stderr)
        print(str(fnf_err), file=sys.stderr)
        print("=" * 65 + "\n", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Fatal error during vector indexing: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
