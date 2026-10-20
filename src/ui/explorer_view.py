"""
Document Explorer view for PDF Document Summarization & Insight Agent.
Provides searchable, filterable document list with 2-line insights, detailed summaries,
and page-by-page text inspection.
"""

from __future__ import annotations

import streamlit as st
from src.ui.state import load_all_documents, load_extracted_page_text


def render_explorer_view() -> None:
    """Renders the Document Explorer interface."""
    docs = load_all_documents()

    if not docs:
        st.warning("No documents found. Please run the PDF Ingestion Pipeline from the Dashboard tab.")
        return

    # Extract all distinct categories
    all_categories = sorted(list({d.get("category", "Uncategorized") for d in docs if d.get("category")}))
    category_options = ["All Categories"] + all_categories

    # Top search and filter bar
    st.markdown("### 📑 Document Explorer")
    st.caption("Browse extracted research papers, view AI-generated two-line insights, and inspect document evidence.")

    f_col1, f_col2, f_col3 = st.columns([3, 2, 2])

    with f_col1:
        search_query = st.text_input(
            "🔍 Search Documents",
            value=st.session_state.get("search_query", ""),
            placeholder="Search by ID (e.g. PDF_01), title, or keywords...",
            key="explorer_search_input",
        )
    with f_col2:
        selected_cat = st.selectbox(
            "🏷️ Filter by Category / Domain",
            options=category_options,
            index=0,
            key="explorer_cat_filter",
        )
    with f_col3:
        status_filter = st.selectbox(
            "⚡ Filter by Status",
            options=["All Documents", "Analyzed Only", "Analysis Pending", "Needs OCR"],
            index=0,
            key="explorer_status_filter",
        )

    # Filter documents based on criteria
    filtered_docs = []
    q = search_query.strip().lower()

    for d in docs:
        # Search match
        matches_query = True
        if q:
            matches_query = (
                q in d["doc_id"].lower()
                or q in d["title"].lower()
                or q in d["filename"].lower()
                or q in d.get("purpose", "").lower()
            )

        # Category match
        matches_cat = True
        if selected_cat != "All Categories":
            matches_cat = d.get("category") == selected_cat

        # Status match
        matches_status = True
        if status_filter == "Analyzed Only":
            matches_status = d.get("is_analyzed", False) is True
        elif status_filter == "Analysis Pending":
            matches_status = d.get("is_analyzed", False) is False
        elif status_filter == "Needs OCR":
            matches_status = d.get("needs_ocr", False) is True

        if matches_query and matches_cat and matches_status:
            filtered_docs.append(d)

    st.write(f"Showing **{len(filtered_docs)}** of {len(docs)} documents")

    if not filtered_docs:
        st.info("No documents matched the specified filters. Try adjusting your search term or category.")
        return

    # Document Cards
    for doc in filtered_docs:
        doc_id = doc["doc_id"]
        is_analyzed = doc.get("is_analyzed", False)
        title = doc["title"]
        category = doc.get("category", "General")
        page_count = doc.get("page_count", 0)
        char_count = doc.get("total_char_count", 0)
        needs_ocr = doc.get("needs_ocr", False)

        with st.container():
            # Card Header
            header_col, action_col = st.columns([5, 2])
            with header_col:
                # Badge line
                ocr_badge = " `⚠️ NEEDS OCR`" if needs_ocr else ""
                status_badge = " `✅ ANALYZED`" if is_analyzed else " `⏳ ANALYSIS PENDING`"
                st.markdown(f"#### `{doc_id}` — {title}{status_badge}{ocr_badge}")
                st.markdown(
                    f"**Domain:** `{category}` | **Pages:** `{page_count}` | "
                    f"**Characters:** `{char_count:,}` | **File:** `{doc['filename']}`"
                )
            with action_col:
                if st.button("💬 Ask in AI Chat", key=f"chat_with_{doc_id}", use_container_width=True):
                    st.session_state.selected_chat_doc = doc_id
                    st.session_state.current_view = "AI Chat"
                    st.rerun()

            # Two-line Insight
            insights = doc.get("two_line_insight")
            if insights and len(insights) > 0:
                st.markdown(
                    f"""
                    <div class="insight-quote">
                        <strong>💡 Key Insight:</strong><br>
                        • {insights[0]}<br>
                        {f'• {insights[1]}' if len(insights) > 1 else ''}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            elif is_analyzed:
                st.info("No two-line insight generated for this document.")
            else:
                st.caption("ℹ️ *Two-line insight pending AI analysis with GEMINI_API_KEY.*")

            # Document Details Expander
            with st.expander(f"📖 Detailed View & Evidence for {doc_id}"):
                det_tab1, det_tab2, det_tab3 = st.tabs(
                    ["Summary & Findings", "Facts, Metrics & Recommendations", "Extracted Page Browser"]
                )

                # Tab 1: Executive Summary & Findings
                with det_tab1:
                    if doc.get("purpose"):
                        st.markdown(f"**Research Purpose & Objective:**\n{doc['purpose']}")

                    if doc.get("summary"):
                        st.markdown(f"**Executive Summary:**\n{doc['summary']}")

                    findings = doc.get("key_findings", [])
                    if findings:
                        st.markdown("**Key Empirical Findings:**")
                        for f in findings:
                            st.markdown(f"- {f}")

                    conclusions = doc.get("conclusions", [])
                    if conclusions:
                        st.markdown("**Conclusions:**")
                        for c in conclusions:
                            st.markdown(f"- {c}")

                    risks = doc.get("risks", [])
                    if risks:
                        st.markdown("**Risks & Limitations:**")
                        for r in risks:
                            st.markdown(f"- ⚠️ {r}")

                    if not is_analyzed:
                        st.info(
                            "Full executive analysis is pending. Configure GEMINI_API_KEY and run document analysis to generate structured findings."
                        )

                # Tab 2: Facts, Metrics & Recommendations
                with det_tab2:
                    facts = doc.get("facts_and_metrics", [])
                    if facts:
                        st.markdown("**Quantitative Facts & Evidence:**")
                        for f in facts:
                            if isinstance(f, dict):
                                p_str = f" `[Page {f.get('page_number')}]`" if f.get("page_number") else ""
                                fact_text = f.get("fact", "")
                                st.markdown(f"- {fact_text}{p_str}")
                            else:
                                st.markdown(f"- {f}")
                    else:
                        st.write("No specific quantitative metrics listed.")

                    st.markdown("---")
                    st.markdown("**Actionable Recommendations:**")
                    recs = doc.get("recommendations", [])
                    if recs:
                        for r in recs:
                            st.markdown(f"- 🎯 {r}")
                    else:
                        st.info("ℹ️ *This document does not explicitly state any recommendations.*")

                # Tab 3: Page Text Browser
                with det_tab3:
                    if page_count > 0:
                        selected_page = st.selectbox(
                            f"Select Page (1 to {page_count})",
                            options=list(range(1, page_count + 1)),
                            key=f"page_sel_{doc_id}",
                        )
                        page_text = load_extracted_page_text(doc_id, selected_page)
                        if page_text:
                            st.text_area(
                                f"Extracted Text (Page {selected_page})",
                                value=page_text,
                                height=260,
                                key=f"text_preview_{doc_id}_{selected_page}",
                            )
                        else:
                            st.write("Text preview not available for this page.")
                    else:
                        st.write("No extracted pages available.")

            st.markdown("---")
