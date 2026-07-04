"""
Survey agent.

Wraps survey.generate_survey, defaulting to all ingested papers when the
Planner left source_files empty, mirroring analysis_agent's behavior.
"""

from researchmind.agents.messages import AgentMessage
from researchmind.survey import generate_survey
from researchmind.vectorstore import list_source_files


def survey(source_files: list[str]) -> AgentMessage:
    """
    Generate a literature-review-style synthesis across papers.

    Raises:
        ValueError: if no papers are ingested at all.
        RuntimeError: propagated if the Groq API call fails.
    """
    files = source_files if source_files else list_source_files()
    if not files:
        raise ValueError("No papers ingested. Ingest at least one paper before a survey.")

    result = generate_survey(files)

    return AgentMessage(
        sender="survey_agent",
        receiver="orchestrator",
        context={"answer": result.answer, "result": result},
        confidence=1.0,
    )