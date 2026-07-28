"""
Ingestion task status tracking, backed by the shared SQLite kv_store.

Previously an in-memory dict — status was lost on every process restart
(and invisible across processes). Now persisted, so a task's status
survives backend restarts (e.g. a Docker container recreation) instead
of silently reverting to "unknown task_id".
"""

import json
import sqlite3
from typing import Literal, Optional, TypedDict

from researchmind.kv_store import DB_PATH, get_value, set_value

NAMESPACE = "ingest_tasks"


class TaskRecord(TypedDict):
    filename: str
    status: Literal["processing", "completed", "failed"]
    chunks_created: Optional[int]
    error: Optional[str]


def create_task(task_id: str, filename: str) -> None:
    """Register a new task as 'processing'."""
    set_value(NAMESPACE, task_id, {
        "filename": filename,
        "status": "processing",
        "chunks_created": None,
        "error": None,
    })


def mark_completed(task_id: str, chunks_created: int) -> None:
    """Mark a task as successfully completed."""
    record = get_value(NAMESPACE, task_id, as_json=True)
    if record:
        record["status"] = "completed"
        record["chunks_created"] = chunks_created
        set_value(NAMESPACE, task_id, record)


def mark_failed(task_id: str, error: str) -> None:
    """Mark a task as failed with an error message."""
    record = get_value(NAMESPACE, task_id, as_json=True)
    if record:
        record["status"] = "failed"
        record["error"] = error
        set_value(NAMESPACE, task_id, record)


def get_task(task_id: str) -> Optional[TaskRecord]:
    """Fetch a task's current record, or None if task_id is unknown."""
    return get_value(NAMESPACE, task_id, as_json=True)


def is_filename_processing(filename: str) -> bool:
    """
    Return True if any known task for this filename is currently 'processing'.

    Used by the upload endpoint to reject a duplicate in-flight upload before
    it overwrites the PDF on disk mid-read by the first task, which would
    corrupt that ingestion silently.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        rows = conn.execute(
            "SELECT value FROM kv_store WHERE namespace = ?", (NAMESPACE,)
        ).fetchall()
        conn.close()
    except Exception:
        return False

    for (value,) in rows:
        try:
            record = json.loads(value)
            if record.get("filename") == filename and record.get("status") == "processing":
                return True
        except Exception:
            continue
    return False