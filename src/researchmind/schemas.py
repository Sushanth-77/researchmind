"""
Pydantic schemas for structured LLM output.

Fields use sentinel defaults ("Not specified" / empty list) rather than
Optional[None] so downstream code (Phase 3 comparison, Phase 6 analysis)
can always treat these fields as present and iterable, never None-check.
"""

from typing import Literal

from pydantic import BaseModel, Field

NOT_SPECIFIED = "Not specified"


class PaperMetadata(BaseModel):
    """Structured metadata extracted from a research paper."""

    title: str = Field(description="Full title of the paper.")
    authors: list[str] = Field(
        default_factory=list,
        description="List of author names as they appear on the paper.",
    )
    methodology: str = Field(
        default=NOT_SPECIFIED,
        description="Brief description of the research methodology or approach used.",
    )
    dataset: str = Field(
        default=NOT_SPECIFIED,
        description="Dataset(s) used in the study, if any.",
    )
    evaluation_metrics: list[str] = Field(
        default_factory=list,
        description="Metrics used to evaluate results (e.g. accuracy, F1, p-value).",
    )


class PlannerDecision(BaseModel):
    """The Planner agent's routing decision for a user query."""

    intent: Literal[
        "single_paper_qa",
        "metadata_extraction",
        "comparison",
        "analysis",
        "survey",
        "knowledge_graph",
    ] = Field(description="Which capability should handle this query.")
    source_files: list[str] = Field(
        default_factory=list,
        description="Filenames (from the available list) relevant to this query.",
    )
    reasoning: str = Field(
        default="",
        description="One-sentence justification for the intent and file selection.",
    )