"""
In-memory task store for background ingestion status tracking.

Deliberately not a database: this is a local, single-process free-tier
project, and adding persistence here would be exactly the kind of
premature abstraction the project constraints warn against. The real
limitation this creates — task status is lost on server restart and
only visible within one process — is acceptable at this scale but
would need revisiting (e.g. Redis, a table) before any real deployment.

A threading.Lock guards the dict because FastAPI's BackgroundTasks run
in a threadpool, so writes here happen off the main request thread.
"""

import threading
from typing import Literal, Optional, TypedDict


class TaskRecord(TypedDict):
    filename: str
    status: Literal["processing", "completed", "failed"]
    chunks_created: Optional[int]
    error: Optional[str]


_lock = threading.Lock()
_tasks: dict[str, TaskRecord] = {}


def create_task(task_id: str, filename: str) -> None:
    """Register a new task as 'processing'."""
    with _lock:
        _tasks[task_id] = {
            "filename": filename,
            "status": "processing",
            "chunks_created": None,
            "error": None,
        }


def mark_completed(task_id: str, chunks_created: int) -> None:
    """Mark a task as successfully completed."""
    with _lock:
        if task_id in _tasks:
            _tasks[task_id]["status"] = "completed"
            _tasks[task_id]["chunks_created"] = chunks_created


def mark_failed(task_id: str, error: str) -> None:
    """Mark a task as failed with an error message."""
    with _lock:
        if task_id in _tasks:
            _tasks[task_id]["status"] = "failed"
            _tasks[task_id]["error"] = error


def get_task(task_id: str) -> Optional[TaskRecord]:
    """Fetch a task's current record, or None if task_id is unknown."""
    with _lock:
        record = _tasks.get(task_id)
        return dict(record) if record else None