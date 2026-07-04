"""
Request/response models for the FastAPI backend.

Kept separate from researchmind/schemas.py (LLM output schemas) since
these describe the HTTP contract, not model output structure.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    """Returned immediately after an ingestion request is accepted."""

    task_id: str
    filename: str
    status: Literal["processing"]


class IngestStatusResponse(BaseModel):
    """Returned when polling an ingestion task's status."""

    task_id: str
    filename: str
    status: Literal["processing", "completed", "failed"]
    chunks_created: Optional[int] = None
    error: Optional[str] = None


class PapersListResponse(BaseModel):
    """List of papers currently stored in the vector store."""

    papers: list[str]


class QueryRequest(BaseModel):
    """A user query to route through the orchestration graph."""

    query: str = Field(min_length=1, description="Natural-language query.")


class TraceEntry(BaseModel):
    """One agent message from the graph's trace, flattened for JSON."""

    sender: str
    receiver: str
    task_id: str
    timestamp: str
    confidence: float


class QueryResponse(BaseModel):
    """The graph's final answer plus routing metadata and trace."""

    intent: str
    source_files: list[str]
    answer: str
    trace: list[TraceEntry]


class ErrorResponse(BaseModel):
    """Uniform error shape for all non-2xx responses."""

    error: str
    detail: str