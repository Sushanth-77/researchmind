"""
QA agent.

Wraps single-paper QA (qa.answer_question_for_source), whole-collection
QA (qa.answer_question), and multi-paper comparison (comparison.compare_papers)
behind one agent interface, choosing the right one based on how many
source files the Planner selected.
"""

from researchmind.agents.messages import AgentMessage
from researchmind.comparison import compare_papers
from researchmind.qa import answer_question, answer_question_for_source


def answer(query: str, source_files: list[str]) -> AgentMessage:
    """
    Answer a question, routing to single-source, whole-collection, or
    comparison logic based on the number of source files provided.

    Raises:
        ValueError: propagated from underlying retrieval/comparison calls.
        RuntimeError: propagated if the Groq API call fails.
    """
    if len(source_files) >= 2:
        result = compare_papers(query, source_files)
        answer_text = result.answer
    elif len(source_files) == 1:
        result = answer_question_for_source(query, source_files[0])
        answer_text = result.answer
    else:
        # No specific paper identified — search the whole collection.
        result = answer_question(query)
        answer_text = result.answer

    return AgentMessage(
        sender="qa_agent",
        receiver="orchestrator",
        context={"answer": answer_text, "result": result},
        confidence=1.0,
    )