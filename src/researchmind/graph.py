"""
LangGraph state machine: Contextualizer -> Planner -> (Extractor | QA |
Analysis | Survey | KG) based on intent.
"""

from langgraph.graph import END, START, StateGraph

from researchmind.graph_nodes import (
    analysis_node,
    contextualize_node,
    extractor_node,
    kg_node,
    planner_node,
    qa_node,
    route_by_intent,
    survey_node,
)
from researchmind.graph_state import GraphState


def build_graph():
    """Construct and compile the orchestration graph."""
    graph = StateGraph(GraphState)

    graph.add_node("contextualize", contextualize_node)
    graph.add_node("planner", planner_node)
    graph.add_node("extractor", extractor_node)
    graph.add_node("qa", qa_node)
    graph.add_node("analysis", analysis_node)
    graph.add_node("survey", survey_node)
    graph.add_node("kg", kg_node)

    graph.add_edge(START, "contextualize")
    graph.add_edge("contextualize", "planner")
    graph.add_conditional_edges(
        "planner",
        route_by_intent,
        {
            "extractor": "extractor",
            "qa": "qa",
            "analysis": "analysis",
            "survey": "survey",
            "kg": "kg",
        },
    )
    graph.add_edge("extractor", END)
    graph.add_edge("qa", END)
    graph.add_edge("analysis", END)
    graph.add_edge("survey", END)
    graph.add_edge("kg", END)

    return graph.compile()


def run_query(query: str, conversation_history: list[dict] | None = None) -> GraphState:
    """
    Run a query through the compiled graph.

    Args:
        query: the user's latest message.
        conversation_history: prior turns as [{"role", "content"}, ...],
            used to resolve follow-up questions into standalone queries.
            Defaults to no history (single-turn behavior, unchanged cost).

    Raises:
        ValueError: propagated if no papers are ingested or the query is empty.
        RuntimeError: propagated if any Groq call fails after retries.
    """
    app = build_graph()
    initial_state: GraphState = {
        "query": query,
        "conversation_history": conversation_history or [],
        "planner": {"decision": None},
        "extractor": {"metadata": None},
        "qa": {"answer": None},
        "analysis": {"answer": None},
        "survey": {"answer": None},
        "kg": {"answer": None},
        "trace": [],
        "final_answer": None,
    }
    return app.invoke(initial_state)