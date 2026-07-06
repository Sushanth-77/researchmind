"""
Graph state definitions for the LangGraph orchestration.

Each agent owns a namespaced slice of the overall graph state and only
ever reads/writes its own slice — this isolation is what makes the graph
safe to extend: a new agent gets its own namespace and cannot silently
overwrite another agent's data.
"""

import operator
from typing import Annotated, Optional, TypedDict

from researchmind.agents.messages import AgentMessage
from researchmind.schemas import PaperMetadata, PlannerDecision


class PlannerState(TypedDict):
    decision: Optional[PlannerDecision]


class ExtractorState(TypedDict):
    metadata: Optional[dict[str, PaperMetadata]]


class QAState(TypedDict):
    answer: Optional[str]


class AnalysisState(TypedDict):
    answer: Optional[str]


class SurveyState(TypedDict):
    answer: Optional[str]


class KGState(TypedDict):
    answer: Optional[str]


class GraphState(TypedDict):
    """
    Overall graph state.

    'query' starts as the raw user input and is overwritten in place by
    the contextualizer node into a standalone, history-resolved form
    before the Planner ever sees it. Every downstream node already only
    consumed 'query' as an opaque string, so this required no signature
    changes to Planner/Extractor/QA/Analysis/Survey/KG.

    'conversation_history' is a shared input field (like 'query'): prior
    turns as [{"role": "user"|"assistant", "content": str}, ...].
    """

    query: str
    conversation_history: list[dict]
    planner: PlannerState
    extractor: ExtractorState
    qa: QAState
    analysis: AnalysisState
    survey: SurveyState
    kg: KGState
    trace: Annotated[list[AgentMessage], operator.add]
    final_answer: Optional[str]