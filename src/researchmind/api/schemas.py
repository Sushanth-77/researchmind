"""
Request/response models for the FastAPI backend.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    task_id: str
    filename: str
    status: Literal["processing"]


class IngestStatusResponse(BaseModel):
    task_id: str
    filename: str
    status: Literal["processing", "completed", "failed"]
    chunks_created: Optional[int] = None
    error: Optional[str] = None


class PapersListResponse(BaseModel):
    papers: list[str]


class ChatTurn(BaseModel):
    """One prior turn in the conversation, for multi-turn query resolution."""

    role: Literal["user", "assistant"]
    content: str


class QueryRequest(BaseModel):
    """A user query to route through the orchestration graph."""

    query: str = Field(min_length=1, description="Natural-language query.")
    conversation_history: list[ChatTurn] = Field(
        default_factory=list,
        description="Prior turns, most recent last, used to resolve follow-up questions.",
    )


class TraceEntry(BaseModel):
    sender: str
    receiver: str
    task_id: str
    timestamp: str
    confidence: float


class QueryResponse(BaseModel):
    intent: str
    source_files: list[str]
    answer: str
    trace: list[TraceEntry]


class ErrorResponse(BaseModel):
    error: str
    detail: str