"""
ResearchMind Streamlit frontend.

Talks to the FastAPI backend over HTTP. Chat history is kept in
st.session_state and sent to the backend on every turn so follow-up
questions can be resolved against prior context.
"""

import time

import streamlit as st

from researchmind.frontend.api_client import (
    APIError,
    check_health,
    delete_paper,
    get_ingest_status,
    ingest_paper,
    list_papers,
    run_query,
)

st.set_page_config(page_title="ResearchMind", page_icon="📚", layout="wide")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pending_delete" not in st.session_state:
    st.session_state.pending_delete = None



def _show_api_error(exc: APIError) -> None:
    if exc.status_code == 0:
        st.error(f"⚠️ {exc.detail}")
    else:
        st.error(f"**{exc.error}** (HTTP {exc.status_code}): {exc.detail}")


st.title("📚 ResearchMind")
st.caption("Multi-agent research paper analysis — powered by Groq, Chroma, and LangGraph.")

if not check_health():
    st.error(
        "⚠️ Cannot reach the backend API. Start it with:\n\n"
        "`uvicorn researchmind.api.main:app --reload --port 8000`\n\n"
        "then refresh this page."
    )
    st.stop()

tab_papers, tab_chat, tab_compare = st.tabs(["📄 Papers", "💬 Ask a Question", "🔍 Compare Papers"])

with tab_papers:
    st.subheader("Upload a paper")
    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])

    if uploaded_file is not None and st.button("Ingest this paper"):
        try:
            response = ingest_paper(uploaded_file.name, uploaded_file.getvalue())
            task_id = response["task_id"]

            with st.spinner(f"Ingesting {uploaded_file.name}..."):
                status = "processing"
                while status == "processing":
                    time.sleep(1)
                    status_response = get_ingest_status(task_id)
                    status = status_response["status"]

            if status == "completed":
                st.success(f"✅ Ingested {uploaded_file.name} ({status_response['chunks_created']} chunks created).")
            else:
                st.error(f"❌ Ingestion failed: {status_response['error']}")
        except APIError as exc:
            _show_api_error(exc)

    st.divider()
    st.subheader("Papers in the store")

    if st.button("🔄 Refresh list"):
        st.rerun()

    try:
        papers = list_papers()
        if papers:
            for paper in papers:
                col_name, col_btn = st.columns([6, 1])
                col_name.write(f"- {paper}")

                # Two-step delete: arm with 🗑️, confirm with a warning button.
                # This prevents accidental deletion with a single mis-click.
                if st.session_state.pending_delete == paper:
                    if col_btn.button("✅ Confirm", key=f"confirm_{paper}", type="primary"):
                        try:
                            result = delete_paper(paper)
                            st.success(f"🗑️ Deleted '{paper}' ({result['chunks_deleted']} chunks removed).")
                            st.session_state.pending_delete = None
                            st.rerun()
                        except APIError as exc:
                            _show_api_error(exc)
                            st.session_state.pending_delete = None
                else:
                    if col_btn.button("🗑️", key=f"delete_{paper}", help=f"Delete {paper}"):
                        st.session_state.pending_delete = paper
                        st.rerun()
        else:
            st.info("No papers ingested yet. Upload one above to get started.")
    except APIError as exc:
        _show_api_error(exc)

with tab_chat:
    st.subheader("Ask anything about your ingested papers")
    st.caption(
        "Routes automatically based on your question. Follow-ups like "
        "'what about the polite one?' use the conversation so far to "
        "figure out what you mean."
    )

    if st.button("🗑️ Clear conversation"):
        st.session_state.chat_history = []
        st.rerun()

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("intent"):
                st.caption(f"Intent: `{message['intent']}` | Sources: {message.get('source_files', [])}")

    user_query = st.chat_input("Ask a question...")

    if user_query:
        history_payload = [
            {"role": m["role"], "content": m["content"]} for m in st.session_state.chat_history
        ]

        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = run_query(user_query, history_payload)
                    st.markdown(result["answer"])
                    st.caption(f"Intent: `{result['intent']}` | Sources: {result['source_files']}")
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": result["answer"],
                        "intent": result["intent"],
                        "source_files": result["source_files"],
                    })
                except APIError as exc:
                    _show_api_error(exc)
                    st.session_state.chat_history.append({"role": "assistant", "content": f"⚠️ Error: {exc.detail}"})

with tab_compare:
    st.subheader("Compare papers")
    st.caption(
        "Select two or more papers to compare. Explicitly naming them here is "
        "more reliable than typing a free-form comparison request, since it "
        "removes any ambiguity in which papers should be included."
    )

    try:
        available_papers = list_papers()
    except APIError as exc:
        _show_api_error(exc)
        available_papers = []

    if len(available_papers) < 2:
        st.info("Ingest at least 2 papers (in the Papers tab) to use comparison.")
    else:
        selected = st.multiselect(
            f"Select 2 to {len(available_papers)} papers to compare",
            options=available_papers,
            max_selections=len(available_papers),
        )
        aspect = st.text_input("What aspect should be compared?", value="methodologies")

        if st.button("Compare", disabled=len(selected) < 2):
            papers_list = ", ".join(selected[:-1]) + f", and {selected[-1]}" if len(selected) > 2 else " and ".join(selected)
            comparison_query = f"Compare the {aspect} of {papers_list}"

            # Pass the full conversation history so follow-up comparisons
            # ("now compare their datasets instead") can resolve against the
            # prior comparison turn — previously this always sent empty history.
            history_payload = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.chat_history
            ]

            with st.spinner(f"Comparing {len(selected)} papers..."):
                try:
                    result = run_query(comparison_query, history_payload)
                    st.markdown(result["answer"])
                    st.caption(f"Intent: `{result['intent']}` | Sources: {result['source_files']}")

                    # Append to chat_history so the Chat tab can reference
                    # this comparison in follow-up questions.
                    st.session_state.chat_history.append({"role": "user", "content": comparison_query})
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": result["answer"],
                        "intent": result["intent"],
                        "source_files": result["source_files"],
                    })

                    if len(result["source_files"]) != len(selected):
                        st.warning(
                            f"⚠️ You selected {len(selected)} papers, but the response only used "
                            f"{len(result['source_files'])}. The Planner may not have picked up all "
                            f"of them from the phrasing — try naming fewer papers per comparison if "
                            f"this happens often."
                        )
                except APIError as exc:
                    _show_api_error(exc)