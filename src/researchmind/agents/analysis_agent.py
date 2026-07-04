"""
Analysis agent.

Wraps analysis.analyze_trends_and_gaps, defaulting to all ingested papers
when the Planner left source_files empty (analysis is inherently
cross-corpus, so "no papers named" means "consider everything ingested").
"""

from researchmind.agents.messages import AgentMessage
from researchmind.analysis import analyze_trends_and_gaps
from researchmind.vectorstore import list_source_files


def analyze(source_files: list[str]) -> AgentMessage:
    """
    Run cross-paper trend/gap analysis.

    Raises:
        ValueError: if no papers are ingested at all.
        RuntimeError: propagated if the Groq API call fails.
    """
    files = source_files if source_files else list_source_files()
    if not files:
        raise ValueError("No papers ingested. Ingest at least one paper before analysis.")

    result = analyze_trends_and_gaps(files)

    return AgentMessage(
        sender="analysis_agent",
        receiver="orchestrator",
        context={"answer": result.answer, "result": result},
        confidence=1.0,
    )