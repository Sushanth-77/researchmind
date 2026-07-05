"""
Knowledge Graph agent.

Answers structural questions (shared authors, papers by metric, graph
summary) via deterministic Cypher queries rather than another Groq call.
The graph's value is exact, reproducible structural answers — routing
this through an LLM would add hallucination risk for no benefit over
directly querying the graph.
"""

from researchmind.agents.messages import AgentMessage
from researchmind.knowledge_graph import (
    find_papers_by_metric,
    find_shared_authors,
    get_graph_summary,
)


def query_graph(query: str) -> AgentMessage:
    """
    Answer a structural question about the knowledge graph using simple
    keyword-based routing to the relevant Cypher query.

    Raises:
        RuntimeError: propagated if Neo4j is unreachable.
        EnvironmentError: propagated if Neo4j credentials are unset.
    """
    lowered = query.lower()

    if "author" in lowered and (
        any(w in lowered for w in ("shared", "common", "both", "multiple"))
        or "more than one" in lowered
        or "more than 1" in lowered
    ):
        shared = find_shared_authors()
        answer = (
            "\n".join(f"{s['author']} appears on: {', '.join(s['papers'])}" for s in shared)
            if shared
            else "No authors appear on more than one ingested paper."
        )

    elif any(w in lowered for w in ("metric", "accuracy", "evaluat")):
        answer = None
        for word in lowered.replace("?", "").split():
            matches = find_papers_by_metric(word)
            if matches:
                answer = f"Papers evaluated with a metric matching '{word}': {', '.join(matches)}"
                break
        if answer is None:
            summary = get_graph_summary()
            answer = (
                f"The graph tracks {summary['metrics']} distinct metrics across "
                f"{summary['papers']} papers. Ask about a specific metric name "
                f"(e.g. 'which papers use accuracy') for a filtered result."
            )

    else:
        summary = get_graph_summary()
        answer = (
            f"Knowledge graph summary: {summary['papers']} papers, {summary['authors']} authors, "
            f"{summary['methodologies']} methodologies, {summary['datasets']} datasets, "
            f"{summary['metrics']} metrics."
        )

    return AgentMessage(
        sender="graph_agent",
        receiver="orchestrator",
        context={"answer": answer},
        confidence=1.0,
    )