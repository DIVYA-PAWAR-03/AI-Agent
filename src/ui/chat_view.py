"""
AI Chat view for PDF Document Summarization & Insight Agent.
Provides evidence-grounded multi-turn conversational interface with source citations,
document scope filtering, and multi-document comparison.
"""

from __future__ import annotations

from typing import List
import streamlit as st
from src.agent.qa_engine import INSUFFICIENT_EVIDENCE_MESSAGE
from src.models.schemas import Citation
from src.ui.state import clear_chat, get_agent, load_all_documents


def render_chat_view() -> None:
    """Renders the AI Chat interface."""
    docs = load_all_documents()
    agent = get_agent(st.session_state.conv_session)

    st.markdown("### 💬 Evidence-Grounded AI Research Assistant")
    st.caption(
        "Ask questions across all indexed research papers or target a specific PDF. "
        "Answers cite exact page numbers and evidence from the text."
    )

    # 1. Document Scope Selector & Controls Bar
    scope_col, action_col = st.columns([4, 2])

    with scope_col:
        # Build dropdown options
        doc_options = ["All Documents (Entire Corpus)"] + [f"{d['doc_id']}: {d['title']}" for d in docs]

        # Determine current selection index
        current_idx = 0
        if st.session_state.selected_chat_doc:
            for idx, opt in enumerate(doc_options[1:], start=1):
                if opt.startswith(f"{st.session_state.selected_chat_doc}:"):
                    current_idx = idx
                    break

        selected_scope = st.selectbox(
            "🎯 Document Scope",
            options=doc_options,
            index=current_idx,
            key="chat_scope_selector",
            help="Restrict evidence retrieval to a specific document or search the entire corpus.",
        )

        # Update session state with selected doc ID
        if selected_scope == "All Documents (Entire Corpus)":
            st.session_state.selected_chat_doc = None
        else:
            selected_doc_id = selected_scope.split(":")[0].strip()
            st.session_state.selected_chat_doc = selected_doc_id

    with action_col:
        st.write("")  # Vertical spacing
        st.write("")
        if st.button("🧹 Clear Chat / New Conversation", use_container_width=True, key="btn_clear_chat"):
            clear_chat()
            st.rerun()

    # Scope Info Pill
    if st.session_state.selected_chat_doc:
        st.info(f"Target Scope: **`{st.session_state.selected_chat_doc}`** (Queries will prioritize this document unless a comparison is requested).")
    else:
        st.info("Target Scope: **Corpus-Wide** (Cross-document synthesis & semantic retrieval across all 50 PDFs).")

    # 2. Suggested Prompts
    st.markdown("**💡 Suggested Queries:**")
    suggest_cols = st.columns(4)

    suggestions = [
        ("Quantum Speedup in PDF_01", "What speedup is reported in PDF_01 on page 2?"),
        ("Compare PDF_01 & PDF_02", "Compare the core technologies in PDF_01 and PDF_02"),
        ("Two-Line Insight for PDF_05", "Give me a dense two-line insight for PDF_05"),
        ("Recommendations Check", "What recommendations are provided in PDF_01?"),
    ]

    clicked_suggestion = None
    for idx, (label, query_text) in enumerate(suggestions):
        with suggest_cols[idx]:
            if st.button(label, key=f"sug_{idx}", use_container_width=True):
                clicked_suggestion = query_text

    st.markdown("---")

    # 3. Message Stream Display
    for msg in st.session_state.chat_messages:
        role = msg["role"]
        with st.chat_message(role):
            if role == "user":
                st.markdown(f"**{msg['content']}**")
            else:
                op_name = msg.get("operation", "document_qa")
                is_supported = msg.get("is_supported", True)
                citations: List[dict] = msg.get("citations", [])
                active_doc = msg.get("active_doc_id")

                # Grounding and Operation Badge
                badge_str = f"`[{op_name.upper()}]`"
                if active_doc:
                    badge_str += f" `[Target: {active_doc}]`"

                if is_supported:
                    st.markdown(f"{badge_str} `✅ Grounded in Evidence`")
                else:
                    st.markdown(f"{badge_str} `⚠️ Insufficient Evidence / Outside Scope`")

                # Render Answer
                st.markdown(msg["content"])

                # Evidence & Citations Expander
                if citations:
                    with st.expander(f"📚 View Retrieved Evidence ({len(citations)} Passages / Citations)"):
                        for c in citations:
                            doc_id = c.get("doc_id", "Unknown")
                            page_num = c.get("page_number", "Unknown")
                            snippet = c.get("snippet", "")
                            st.markdown(
                                f"""
                                <div class="citation-box">
                                    <strong>Document:</strong> <code>{doc_id}</code> | <strong>Page:</strong> <code>{page_num}</code><br>
                                    <em>"{snippet}"</em>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

    # 4. Handle Incoming Query (from chat input or suggestion button)
    user_input = st.chat_input("Ask a question about the document(s)...")
    prompt_to_process = clicked_suggestion or user_input

    if prompt_to_process:
        # Display user message
        st.session_state.chat_messages.append({"role": "user", "content": prompt_to_process})
        with st.chat_message("user"):
            st.markdown(f"**{prompt_to_process}**")

        # Process with Agent
        with st.chat_message("assistant"):
            with st.spinner("Analyzing documents & retrieving grounded evidence..."):
                resp = agent.process_query(
                    query=prompt_to_process,
                    selected_doc_id=st.session_state.selected_chat_doc,
                )

                op_name = resp.operation
                is_supported = resp.is_directly_supported
                active_doc = resp.active_doc_id
                citations_data = [
                    {"doc_id": c.doc_id, "page_number": c.page_number, "snippet": c.snippet}
                    for c in resp.citations
                ]

                # Render response
                badge_str = f"`[{op_name.upper()}]`"
                if active_doc:
                    badge_str += f" `[Target: {active_doc}]`"

                if is_supported:
                    st.markdown(f"{badge_str} `✅ Grounded in Evidence`")
                else:
                    st.markdown(f"{badge_str} `⚠️ Insufficient Evidence / Outside Scope`")

                st.markdown(resp.answer)

                if citations_data:
                    with st.expander(f"📚 View Retrieved Evidence ({len(citations_data)} Passages / Citations)"):
                        for c in citations_data:
                            st.markdown(
                                f"""
                                <div class="citation-box">
                                    <strong>Document:</strong> <code>{c['doc_id']}</code> | <strong>Page:</strong> <code>{c['page_number']}</code><br>
                                    <em>"{c['snippet']}"</em>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                # Save assistant response to state
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": resp.answer,
                    "operation": op_name,
                    "active_doc_id": active_doc,
                    "citations": citations_data,
                    "is_supported": is_supported,
                })
