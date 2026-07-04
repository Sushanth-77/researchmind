"""
Graph state definitions for the LangGraph orchestration.

Each agent owns a namespaced slice of the overall graph state (PlannerState,
ExtractorState, QAState) and only ever reads/writes its own slice. Nodes
do not reach into another agent's namespace directly — anything one agent
needs from another is passed explicitly through routing logic in graph.py,
mirroring the AgentMessage-based isolation from Phase 4's manual orchestrator.

This isolation is what makes the graph safe to extend later: a new agent
gets its own namespace and cannot silently overwrite another agent's data,
even if graph topology grows more complex (parallel branches, subgraphs).
"""

import operator
from typing import Annotated, Optional, TypedDict

from researchmind.agents.messages import AgentMessage
from researchmind.schemas import PaperMetadata, PlannerDecision


class PlannerState(TypedDict):
    """Planner agent's isolated state slice."""

    decision: Optional[PlannerDecision]


class ExtractorState(TypedDict):
    """Extractor agent's isolated state slice."""

    metadata: Optional[dict[str, PaperMetadata]]


class QAState(TypedDict):
    """QA agent's isolated state slice."""

    answer: Optional[str]


class GraphState(TypedDict):
    """
    Overall graph state.

    'query' and 'final_answer' are the only fields shared across the whole
    graph by design (the input and the output). 'trace' accumulates via the
    operator.add reducer so each node's message appends rather than
    overwrites. Every other field lives inside a per-agent namespace that
    only that agent's node writes to.
    """

    query: str
    planner: PlannerState
    extractor: ExtractorState
    qa: QAState
    trace: Annotated[list[AgentMessage], operator.add]
    final_answer: Optional[str]