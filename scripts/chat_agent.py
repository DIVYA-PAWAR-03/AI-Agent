"""
Interactive and Single-Query CLI for the Grounded Document AI Agent (UC05).

Usage:
  python -m scripts.chat_agent --query "Summarize PDF_01 in two lines"
  python -m scripts.chat_agent --query "Compare PDF_01 and PDF_02"
  python -m scripts.chat_agent --interactive [--doc-id PDF_01]
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

# Ensure Windows terminal handles UTF-8 output safely
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src.agent.qa_engine import DocumentAgent
from src.utils.logger import setup_logger

logger = setup_logger("scripts.chat_agent")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evidence-Grounded PDF Document AI Agent CLI (UC05)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Single question or command to ask the agent",
    )
    parser.add_argument(
        "--doc-id",
        type=str,
        default=None,
        help="Pre-select an active focus document ID (e.g. 'PDF_01')",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Launch an interactive multi-turn conversational session in the console",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed debug logging",
    )
    return parser.parse_args(argv)


def print_agent_response(resp) -> None:
    """Formats and prints agent response with citations."""
    print("\n" + "=" * 65)
    print(f" Operation: [{resp.operation.upper()}] | Active Doc: {resp.active_doc_id or 'Corpus-wide'}")
    print("=" * 65)
    print(resp.answer)
    print("=" * 65)

    if resp.citations:
        print("\nGrounding Citations:")
        for idx, c in enumerate(resp.citations, start=1):
            page_info = f", Page {c.page_number}" if c.page_number else ""
            print(f"  [{idx}] Document: {c.doc_id}{page_info}")
            if c.snippet:
                print(f"      Excerpt: \"{c.snippet.strip()}\"")
        print("-" * 65)


def run_interactive_session(agent: DocumentAgent, initial_doc: Optional[str] = None) -> None:
    """Runs a multi-turn conversation loop in the terminal."""
    print("=" * 65)
    print(" 📄 PDF Document Intelligence Agent — Interactive Session")
    print(" Type 'exit' or 'quit' to end. Type 'clear' to reset memory.")
    if initial_doc:
        print(f" Active Focus Document: {initial_doc}")
    print("=" * 65 + "\n")

    current_doc = initial_doc

    while True:
        try:
            prompt_str = f"User [{current_doc or 'All'}] > "
            user_input = input(prompt_str).strip()

            if not user_input:
                continue

            if user_input.lower() in ("exit", "quit"):
                print("Ending session. Goodbye!")
                break

            if user_input.lower() == "clear":
                agent.session.clear()
                current_doc = None
                print("Conversation history cleared.")
                continue

            # Switch active doc command: e.g. "select PDF_05"
            if user_input.lower().startswith("select pdf_") or user_input.lower().startswith("use pdf_"):
                parts = user_input.split()
                if len(parts) >= 2:
                    current_doc = parts[1].upper()
                    agent.session.set_active_document(current_doc)
                    print(f"Switched active document to: {current_doc}\n")
                    continue

            resp = agent.process_query(user_input, selected_doc_id=current_doc)
            current_doc = resp.active_doc_id
            print_agent_response(resp)

        except (KeyboardInterrupt, EOFError):
            print("\nSession interrupted. Exiting.")
            break
        except Exception as exc:
            print(f"\n❌ Error processing query: {exc}\n")


def main(argv=None) -> int:
    args = parse_args(argv)
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logger("document_agent", log_level)

    try:
        agent = DocumentAgent()

        if args.interactive:
            run_interactive_session(agent, initial_doc=args.doc_id)
            return 0

        if args.query:
            resp = agent.process_query(args.query, selected_doc_id=args.doc_id)
            print_agent_response(resp)
            return 0

        print("Please provide a query via `--query \"...\"` or run `--interactive` mode.")
        return 1

    except Exception as exc:
        logger.exception("Fatal error in chat agent: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
