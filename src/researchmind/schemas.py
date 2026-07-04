"""
Pydantic schemas for structured LLM output.

Fields use sentinel defaults ("Not specified" / empty list) rather than
Optional[None] so downstream code (Phase 3 comparison, Phase 6 analysis)
can always treat these fields as present and iterable, never None-check.
"""

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