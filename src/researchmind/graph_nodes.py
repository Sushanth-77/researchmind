"""
Node functions for the LangGraph orchestration.

Each node function reads only the namespace(s) it needs (its own, plus
the Planner's decision when it must route work) and returns a partial
state update touching only its own namespace — never the full state.
LangGraph merges these partial updates, so no node can accidentally
overwrite a field owned by another agent.
"""

from researchmind.agents import (
    analysis_agent,
    extractor,
    graph_agent,
    planner,
    qa_agent,
    survey_agent,
)
from researchmind.graph_state import GraphState
from researchmind.schemas import PaperMetadata
from researchmind.vectorstore import list_source_files


def planner_node(state: GraphState) -> dict:
    """Run the Planner agent and populate the 'planner' namespace only."""
    available_files = list_source_files()
    msg = planner.plan(state["query"], available_files)
    decision = msg.context["decision"]

    return {
        "planner": {"decision": decision},
        "trace": [msg],
    }


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

    return {
        "qa": {"answer": answer_text},
        "trace": [msg],
        "final_answer": answer_text,
    }


def analysis_node(state: GraphState) -> dict:
    """Run the Analysis agent and populate the 'analysis' namespace only."""
    decision = state["planner"]["decision"]
    msg = analysis_agent.analyze(decision.source_files)
    answer_text = msg.context["answer"]

    return {
        "analysis": {"answer": answer_text},
        "trace": [msg],
        "final_answer": answer_text,
    }


def survey_node(state: GraphState) -> dict:
    """Run the Survey agent and populate the 'survey' namespace only."""
    decision = state["planner"]["decision"]
    msg = survey_agent.survey(decision.source_files)
    answer_text = msg.context["answer"]

    return {
        "survey": {"answer": answer_text},
        "trace": [msg],
        "final_answer": answer_text,
    }


def kg_node(state: GraphState) -> dict:
    """Run the Knowledge Graph agent and populate the 'kg' namespace only."""
    msg = graph_agent.query_graph(state["query"])
    answer_text = msg.context["answer"]

    return {
        "kg": {"answer": answer_text},
        "trace": [msg],
        "final_answer": answer_text,
    }


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
        metrics = (
            ", ".join(meta.evaluation_metrics)
            if meta.evaluation_metrics
            else "None listed"
        )
        sections.append(
            f"Paper: {source_file}\n"
            f"  Title: {meta.title}\n"
            f"  Authors: {authors}\n"
            f"  Methodology: {meta.methodology}\n"
            f"  Dataset: {meta.dataset}\n"
            f"  Evaluation Metrics: {metrics}"
        )
    return "\n\n".join(sections)