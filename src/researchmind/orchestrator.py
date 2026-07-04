"""
Manual orchestration of the Planner, Extractor, and QA agents.

Plain Python coordination — no agent framework. The orchestrator asks
the Planner to classify the query, then dispatches to the appropriate
agent based on the decided intent, collecting every AgentMessage produced
along the way into a trace for transparency and debugging.
"""

from dataclasses import dataclass, field

from researchmind.agents import extractor, planner, qa_agent
from researchmind.agents.messages import AgentMessage
from researchmind.schemas import PaperMetadata
from researchmind.vectorstore import list_source_files


@dataclass
class OrchestratorResult:
    """Final response plus the full trace of agent messages produced."""

    intent: str
    answer: str
    trace: list[AgentMessage] = field(default_factory=list)


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


def handle_query(query: str) -> OrchestratorResult:
    """
    Route a user query through the Planner, then the appropriate
    Extractor or QA agent, returning the final answer and full trace.

    Raises:
        ValueError: if no papers have been ingested, or the query is empty.
        RuntimeError: if any underlying Groq call fails after retries.
    """
    available_files = list_source_files()

    planner_msg = planner.plan(query, available_files)
    decision = planner_msg.context["decision"]
    trace = [planner_msg]

    if decision.intent == "metadata_extraction":
        extractor_msg = extractor.extract(decision.source_files)
        trace.append(extractor_msg)
        answer_text = _format_metadata_answer(extractor_msg.context["metadata"])

    else:  # "single_paper_qa" or "comparison"
        qa_msg = qa_agent.answer(query, decision.source_files)
        trace.append(qa_msg)
        answer_text = qa_msg.context["answer"]

    return OrchestratorResult(intent=decision.intent, answer=answer_text, trace=trace)