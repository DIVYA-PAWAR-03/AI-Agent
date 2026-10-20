"""
Intent classification and document reference resolver for Document AI Agent.
"""

from __future__ import annotations

from enum import Enum
import re
from typing import List, Optional, Set, Tuple


class OperationType(str, Enum):
    TWO_LINE_INSIGHT = "two_line_insight"
    DETAILED_SUMMARY = "detailed_summary"
    CATEGORIZATION_LOOKUP = "categorization_lookup"
    FACTS_AND_RECOMMENDATIONS = "facts_and_recommendations"
    COMPARE_DOCUMENTS = "compare_documents"
    DOCUMENT_QA = "document_qa"
    FOLLOW_UP_QA = "follow_up_qa"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


# Known topic keywords mapping directly to standard document IDs
TOPIC_TO_DOC_MAP = {
    "quantum computing": "PDF_01",
    "artificial intelligence": "PDF_02",
    "deep learning": "PDF_03",
    "retrieval-augmented generation": "PDF_04",
    "rag": "PDF_04",
    "blockchain": "PDF_05",
    "cryptography": "PDF_06",
    "machine learning": "PDF_07",
    "neural network": "PDF_08",
    "autonomous agent": "PDF_09",
    "edge computing": "PDF_10",
    "exoplanet": "PDF_11",
    "astrophysics": "PDF_12",
    "crispr": "PDF_13",
    "gene editing": "PDF_14",
    "quantum entanglement": "PDF_15",
    "microfluidics": "PDF_16",
    "dark matter": "PDF_17",
    "synthetic biology": "PDF_18",
    "perovskite solar cell": "PDF_19",
    "perovskite": "PDF_19",
    "superconductivity": "PDF_20",
    "central bank digital currency": "PDF_21",
    "cbdc": "PDF_21",
    "behavioral economics": "PDF_22",
    "supply chain": "PDF_23",
    "black-scholes": "PDF_24",
    "circular economy": "PDF_25",
    "microfinance": "PDF_26",
    "high-frequency trading": "PDF_27",
    "hft": "PDF_27",
    "sovereign debt": "PDF_28",
    "venture capital": "PDF_29",
    "carbon market": "PDF_30",
    "roman engineering": "PDF_31",
    "gothic architecture": "PDF_32",
    "scientific revolution": "PDF_33",
    "silk road": "PDF_34",
    "cuneiform": "PDF_35",
    "linear perspective": "PDF_36",
    "byzantine architecture": "PDF_37",
    "stoicism": "PDF_38",
    "industrial revolution": "PDF_39",
    "terrace agriculture": "PDF_40",
    "oceanic conveyor belt": "PDF_41",
    "glacial isostatic adjustment": "PDF_42",
    "atmospheric aerosol": "PDF_43",
    "soil microbiome": "PDF_44",
    "hydrothermal vent": "PDF_45",
    "paleoclimatology": "PDF_46",
    "seismic wave": "PDF_47",
    "eutrophication": "PDF_48",
    "subduction": "PDF_49",
    "forest canopy": "PDF_50",
}


def extract_document_references(
    query: str,
    active_doc_id: Optional[str] = None,
) -> List[str]:
    """
    Extracts all document IDs (e.g. PDF_01, PDF_15) mentioned in the query.
    Falls back to known topic lookup or active conversation document.
    """
    q_lower = query.lower()
    found: List[str] = []

    # 1. Regex match for PDF_XX, PDF-XX, PDF XX, or PDF_NAME (e.g. PDF_EMPTY)
    sep_matches = re.findall(r"\bpdf[-_]([a-zA-Z0-9_]+)\b", query, flags=re.IGNORECASE)
    for m in sep_matches:
        if m.isdigit():
            doc_id = f"PDF_{int(m):02d}"
        else:
            doc_id = f"PDF_{m.upper()}"
        if doc_id not in found:
            found.append(doc_id)

    num_matches = re.findall(r"\bpdf\s*([0-9]{1,2})\b", query, flags=re.IGNORECASE)
    for m in num_matches:
        doc_id = f"PDF_{int(m):02d}"
        if doc_id not in found:
            found.append(doc_id)

    # 2. Semantic topic keywords lookup if no direct ID found
    if not found:
        for topic, doc_id in TOPIC_TO_DOC_MAP.items():
            if re.search(r"\b" + re.escape(topic) + r"\b", q_lower):
                if doc_id not in found:
                    found.append(doc_id)

    # 3. Contextual fallback to active document if query refers to "this", "it", "the document"
    if not found and active_doc_id:
        pronoun_triggers = [
            "this document",
            "the document",
            "this pdf",
            "the pdf",
            " it ",
            " it?",
            " its ",
            "that finding",
            "that paper",
            "this paper",
        ]
        if any(trigger in f" {q_lower} " for trigger in pronoun_triggers):
            found.append(active_doc_id)

    return found


def classify_operation(
    query: str,
    target_docs: List[str],
    has_history: bool = False,
    active_doc_id: Optional[str] = None,
) -> OperationType:
    """
    Classifies a user query into one of the supported agent operations.
    """
    q_lower = query.lower().strip()

    # Comparison Trigger
    compare_words = ["compare", "contrast", "difference between", "versus", "vs", "similarities between"]
    if any(cw in q_lower for cw in compare_words) or len(target_docs) >= 2:
        return OperationType.COMPARE_DOCUMENTS

    # Two-line insight trigger
    if any(phrase in q_lower for phrase in ["two line", "two-line", "2 line", "2-line", "in two lines", "in 2 lines"]):
        return OperationType.TWO_LINE_INSIGHT

    # Categorization and Metadata trigger
    cat_words = [
        "what category",
        "which category",
        "what domain",
        "which domain",
        "categorize",
        "categorization",
        "main topic",
        "purpose of",
        "what is the purpose",
        "metadata",
    ]
    if any(cw in q_lower for cw in cat_words):
        return OperationType.CATEGORIZATION_LOOKUP

    # Facts, metrics, findings, recommendations trigger
    facts_words = [
        "recommendation",
        "recommendations",
        "finding",
        "findings",
        "statistic",
        "statistics",
        "metric",
        "metrics",
        "numerical",
        "dates",
        "numbers",
        "limitation",
        "limitations",
        "risk",
        "risks",
        "conclusion",
        "conclusions",
    ]
    if any(fw in q_lower for fw in facts_words):
        return OperationType.FACTS_AND_RECOMMENDATIONS

    # Detailed summary trigger
    summary_words = ["summarize", "summary", "overview", "synopsis", "executive summary", "abstract"]
    if any(sw in q_lower for sw in summary_words):
        return OperationType.DETAILED_SUMMARY

    # Follow-up triggers
    follow_up_triggers = [
        "what about",
        "how about",
        "and what",
        "why is that",
        "explain that",
        "tell me more",
        "what else",
        "how does it",
        "why does it",
        "can you elaborate",
    ]
    if has_history and (any(q_lower.startswith(ft) for ft in follow_up_triggers) or " that " in q_lower or " it " in q_lower):
        return OperationType.FOLLOW_UP_QA

    # Default to standard document QA
    return OperationType.DOCUMENT_QA
