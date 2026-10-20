"""
Corpus Analysis and Synthesis view for PDF Document Summarization & Insight Agent.
Presents domain categorization breakdown, quantitative metrics matrix,
recommendations audit, and PDF ingestion health.
"""

from __future__ import annotations

import streamlit as st
from src.ui.state import load_all_documents, load_system_metrics


def render_analysis_view() -> None:
    """Renders the Corpus Analysis & Synthesis view."""
    docs = load_all_documents()
    metrics = load_system_metrics()

    st.markdown("### 📈 Corpus Analysis & Multi-Domain Synthesis")
    st.caption("Cross-document insights, domain taxonomy, empirical facts database, and extraction quality audit.")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🏷️ Domain Categorization",
        "🔢 Extracted Facts & Metrics",
        "🎯 Recommendations Matrix",
        "📋 Ingestion & OCR Quality Audit",
    ])

    # -------------------------------------------------------------------------
    # Tab 1: Domain Categorization Breakdown
    # -------------------------------------------------------------------------
    with tab1:
        st.markdown("#### Domain Distribution Across Research Papers")
        cats = metrics.get("categories", {})

        if cats:
            c_cols = st.columns(min(4, len(cats)))
            for idx, (cat_name, count) in enumerate(sorted(cats.items(), key=lambda x: -x[1])):
                col = c_cols[idx % len(c_cols)]
                with col:
                    st.metric(label=cat_name, value=f"{count} Documents")

            st.markdown("---")
            st.markdown("##### Documents by Category:")
            for cat_name in sorted(cats.keys()):
                matching = [d for d in docs if d.get("category") == cat_name]
                with st.expander(f"{cat_name} ({len(matching)} documents)"):
                    for m in matching:
                        st.markdown(f"- **`{m['doc_id']}`**: {m['title']} (`{m['filename']}`)")
        else:
            st.info(
                "Document domain classification is pending. "
                "Configure GEMINI_API_KEY and run Document Analysis from the Dashboard tab."
            )

    # -------------------------------------------------------------------------
    # Tab 2: Extracted Facts & Metrics
    # -------------------------------------------------------------------------
    with tab2:
        st.markdown("#### Verifiable Empirical Metrics & Numerical Facts")
        st.caption("Quantitative facts extracted from documents, preserved with source page citations.")

        all_facts = []
        for d in docs:
            for f in d.get("facts_and_metrics", []):
                if isinstance(f, dict):
                    all_facts.append({
                        "Document ID": d["doc_id"],
                        "Title": d["title"],
                        "Page": f.get("page_number", "N/A"),
                        "Fact / Metric": f.get("fact", ""),
                    })

        if all_facts:
            # Fact filter
            fact_query = st.text_input("Filter metrics and facts:", placeholder="Search numbers, benchmarks, dates...", key="fact_search")
            if fact_query:
                filtered_facts = [f for f in all_facts if fact_query.lower() in f["Fact / Metric"].lower() or fact_query.lower() in f["Document ID"].lower()]
            else:
                filtered_facts = all_facts

            st.write(f"Displaying **{len(filtered_facts)}** empirical facts across documents:")
            st.dataframe(filtered_facts, use_container_width=True)
        else:
            st.info("No structured quantitative facts cataloged yet. Run Document Analysis once GEMINI_API_KEY is configured.")

    # -------------------------------------------------------------------------
    # Tab 3: Recommendations Matrix
    # -------------------------------------------------------------------------
    with tab3:
        st.markdown("#### Evidence-Grounded Recommendations")
        st.caption("Recommendations are only listed when explicitly stated in source documents. The agent never hallucinates advice.")

        docs_with_recs = [d for d in docs if d.get("recommendations")]
        docs_without_recs = [d for d in docs if d.get("is_analyzed") and not d.get("recommendations")]

        rec_c1, rec_c2 = st.columns(2)
        with rec_c1:
            st.metric("Documents with Explicit Recommendations", len(docs_with_recs))
        with rec_c2:
            st.metric("Documents with NO Recommendations", len(docs_without_recs))

        if docs_with_recs:
            st.markdown("##### Documents with Stated Recommendations:")
            for d in docs_with_recs:
                with st.expander(f"🎯 `{d['doc_id']}`: {d['title']}"):
                    for r in d["recommendations"]:
                        st.markdown(f"- {r}")

        if docs_without_recs:
            with st.expander("ℹ️ Documents explicitly stating NO recommendations (Anti-Hallucination Verified)"):
                for d in docs_without_recs:
                    st.markdown(f"- **`{d['doc_id']}`**: {d['title']} — *No actionable recommendations asserted in source text.*")

        if not docs_with_recs and not docs_without_recs:
            st.info("Recommendations audit will populate once documents are analyzed.")

    # -------------------------------------------------------------------------
    # Tab 4: Ingestion & OCR Quality Audit
    # -------------------------------------------------------------------------
    with tab4:
        st.markdown("#### Ingestion & Document Quality Audit Table")
        st.caption("Review extracted characters, page counts, and OCR status across all discovered PDFs.")

        audit_data = []
        for d in docs:
            audit_data.append({
                "Doc ID": d["doc_id"],
                "Filename": d["filename"],
                "Pages": d.get("page_count", 0),
                "Characters": f"{d.get('total_char_count', 0):,}",
                "Words": f"{d.get('total_word_count', 0):,}",
                "Needs OCR": "⚠️ Yes" if d.get("needs_ocr") else "✅ No",
                "Extraction Status": d.get("status", "unknown").upper(),
                "AI Analysis": "✅ Done" if d.get("is_analyzed") else "⏳ Pending",
            })

        st.dataframe(audit_data, use_container_width=True)
