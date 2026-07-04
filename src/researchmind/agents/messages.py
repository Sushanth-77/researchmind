"""
Structured inter-agent message format.

Every agent communicates by returning an AgentMessage rather than a raw
value, so the orchestrator has a uniform, loggable record of who did what,
with what confidence, and when — this is the scaffolding Phase 5 will
lift directly into LangGraph state.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class AgentMessage:
    """A single structured message passed between agents."""

    sender: str
    receiver: str
    context: dict[str, Any]
    confidence: float
    task_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def summary(self) -> str:
        """One-line human-readable summary for trace logging."""
        return (
            f"[{self.timestamp}] {self.sender} -> {self.receiver} "
            f"(task {self.task_id}, confidence {self.confidence:.2f})"
        )