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
    get_ingest_status,
    ingest_paper,
    list_papers,
    run_query,
)

st.set_page_config(page_title="ResearchMind", page_icon="📚", layout="wide")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


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
                st.write(f"- {paper}")
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
        # History sent to the backend excludes the current message — it's
        # only the prior turns that provide context for resolving this one.
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
    st.subheader("Compare two papers")
    st.caption(
        "Explicitly naming both papers here is more reliable than typing "
        "a free-form comparison request."
    )

    try:
        available_papers = list_papers()
    except APIError as exc:
        _show_api_error(exc)
        available_papers = []

    if len(available_papers) < 2:
        st.info("Ingest at least 2 papers (in the Papers tab) to use comparison.")
    else:
        selected = st.multiselect("Select exactly 2 papers to compare", options=available_papers, max_selections=2)
        aspect = st.text_input("What aspect should be compared?", value="methodologies")

        if st.button("Compare", disabled=len(selected) != 2):
            comparison_query = f"Compare the {aspect} of {selected[0]} and {selected[1]}"
            with st.spinner("Comparing..."):
                try:
                    result = run_query(comparison_query)
                    st.markdown(result["answer"])
                    st.caption(f"Intent: `{result['intent']}` | Sources: {result['source_files']}")
                except APIError as exc:
                    _show_api_error(exc)