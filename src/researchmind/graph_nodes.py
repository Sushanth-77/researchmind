"""
Node functions for the LangGraph orchestration.

Each node function reads only the namespace(s) it needs and returns a
partial state update touching only its own namespace — LangGraph merges
these, so no node can accidentally overwrite a field owned by another agent.
"""

from researchmind.agents import (
    analysis_agent,
    extractor,
    graph_agent,
    planner,
    qa_agent,
    survey_agent,
)
from researchmind.agents.messages import AgentMessage
from researchmind.conversation import contextualize_query
from researchmind.graph_state import GraphState
from researchmind.retrieval import retrieve
from researchmind.schemas import PaperMetadata
from researchmind.title_cache import get_titles_for
from researchmind.vectorstore import list_source_files


def contextualize_node(state: GraphState) -> dict:
    """
    Rewrite the query into a standalone form using conversation history,
    if any is present. With no history, this makes no Groq call and
    returns the query unchanged — zero added cost for single-turn use.
    """
    original = state["query"]
    history = state.get("conversation_history") or []
    resolved = contextualize_query(original, history)

    msg = AgentMessage(
        sender="contextualizer",
        receiver="orchestrator",
        context={"original_query": original, "resolved_query": resolved},
        confidence=1.0,
    )

    return {"query": resolved, "trace": [msg]}


def _rank_files_by_relevance(query: str, available_files: list[str], pool_size: int = 20) -> list[str]:
    """
    Order available_files by relevance to the query using hybrid retrieval
    over the whole collection. Used as a tiebreaker signal when the query
    doesn't explicitly name a paper — see planner.py's SYSTEM_PROMPT for
    how this is weighed against explicit title matches.
    """
    if not available_files:
        return available_files

    try:
        chunks = retrieve(query, top_k=min(pool_size, len(available_files) * 3))
    except ValueError:
        return available_files

    ranked: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        if chunk.source_file in available_files and chunk.source_file not in seen:
            ranked.append(chunk.source_file)
            seen.add(chunk.source_file)

    remaining = [f for f in available_files if f not in seen]
    return ranked + remaining


def planner_node(state: GraphState) -> dict:
    """
    Run the Planner agent and populate the 'planner' namespace only.

    Provides the Planner with both a relevance-ordered file list AND each
    file's title, so it can correctly resolve explicit descriptive
    references ("the Kantian ethics paper") to the right file even when
    that file isn't top-ranked by content relevance for this query — the
    fix for a real routing bug found in testing, where "GPU" and "trained"
    in a query about the Kantian ethics paper caused content-relevance
    ranking to favor an unrelated ML paper instead.
    """
    available_files = list_source_files()
    ranked_files = _rank_files_by_relevance(state["query"], available_files)
    file_titles = get_titles_for(ranked_files)
    msg = planner.plan(state["query"], ranked_files, file_titles=file_titles)
    decision = msg.context["decision"]
    return {"planner": {"decision": decision}, "trace": [msg]}


def extractor_node(state: GraphState) -> dict:
    """Run the Extractor agent and populate the 'extractor' namespace only."""
    decision = state["planner"]["decision"]
    msg = extractor.extract(decision.source_files)
    metadata = msg.context["metadata"]
    return {
        "extractor": {"metadata": metadata},
        "trace": [msg],
        "final_answer": _format_metadata_answer(metadata),
    }


def qa_node(state: GraphState) -> dict:
    """Run the QA agent and populate the 'qa' namespace only."""
    decision = state["planner"]["decision"]
    msg = qa_agent.answer(state["query"], decision.source_files)
    answer_text = msg.context["answer"]
    return {"qa": {"answer": answer_text}, "trace": [msg], "final_answer": answer_text}


def analysis_node(state: GraphState) -> dict:
    """Run the Analysis agent and populate the 'analysis' namespace only."""
    decision = state["planner"]["decision"]
    msg = analysis_agent.analyze(decision.source_files)
    answer_text = msg.context["answer"]
    return {"analysis": {"answer": answer_text}, "trace": [msg], "final_answer": answer_text}


def survey_node(state: GraphState) -> dict:
    """Run the Survey agent and populate the 'survey' namespace only."""
    decision = state["planner"]["decision"]
    msg = survey_agent.survey(decision.source_files)
    answer_text = msg.context["answer"]
    return {"survey": {"answer": answer_text}, "trace": [msg], "final_answer": answer_text}


def kg_node(state: GraphState) -> dict:
    """Run the Knowledge Graph agent and populate the 'kg' namespace only."""
    msg = graph_agent.query_graph(state["query"])
    answer_text = msg.context["answer"]
    return {"kg": {"answer": answer_text}, "trace": [msg], "final_answer": answer_text}


def route_by_intent(state: GraphState) -> str:
    """Conditional edge: send to the node matching the Planner's intent."""
    decision = state["planner"]["decision"]
    routing_map = {
        "metadata_extraction": "extractor",
        "analysis": "analysis",
        "survey": "survey",
        "knowledge_graph": "kg",
    }
    return routing_map.get(decision.intent, "qa")


def _format_metadata_answer(metadata_by_file: dict[str, PaperMetadata]) -> str:
    """Render extracted metadata as readable text for the final answer."""
    sections = []
    for source_file, meta in metadata_by_file.items():
        authors = ", ".join(meta.authors) if meta.authors else "Not specified"
        metrics = ", ".join(meta.evaluation_metrics) if meta.evaluation_metrics else "None listed"
        sections.append(
            f"Paper: {source_file}\n"
            f"  Title: {meta.title}\n"
            f"  Authors: {authors}\n"
            f"  Methodology: {meta.methodology}\n"
            f"  Dataset: {meta.dataset}\n"
            f"  Evaluation Metrics: {metrics}"
        )
    return "\n\n".join(sections)