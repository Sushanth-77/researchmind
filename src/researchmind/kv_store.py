"""
Lightweight SQLite-backed key-value store.

Replaces the two separate in-memory dicts previously used by
api/task_store.py (ingestion task status) and title_cache.py (paper
title lookups) with a single, persistent, namespaced store. Two
independent in-memory dicts was flagged as technical debt: state was
lost on every process restart, and a third cache added later would have
meant a third bespoke pattern.

SQLite (Python stdlib, no new dependency) is the right fit here — single
process, low write volume, no need for a separate running service. It's
free and gives actual persistence across restarts, which the in-memory
dicts could not.

A new connection is opened per operation rather than held open for the
process lifetime. At this write/read volume that costs a negligible
fraction of a millisecond and completely sidesteps SQLite's connection
thread-affinity rules — relevant because FastAPI's BackgroundTasks run
off the main request thread.
"""

import json
import sqlite3
import threading
from typing import Optional, Union

from researchmind.config import PROJECT_ROOT

DB_PATH = PROJECT_ROOT / "data" / "app_state.db"

_write_lock = threading.Lock()


def _get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS kv_store (
            namespace TEXT NOT NULL,
            key TEXT NOT NULL,
            value TEXT NOT NULL,
            PRIMARY KEY (namespace, key)
        )
        """
    )
    return conn


def set_value(namespace: str, key: str, value: Union[dict, str]) -> None:
    """Store a value under (namespace, key). Dicts are JSON-serialized; strings stored as-is."""
    serialized = json.dumps(value) if isinstance(value, dict) else value
    with _write_lock:
        conn = _get_connection()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO kv_store (namespace, key, value) VALUES (?, ?, ?)",
                (namespace, key, serialized),
            )
            conn.commit()
        finally:
            conn.close()


def get_value(namespace: str, key: str, as_json: bool = False) -> Optional[Union[dict, str]]:
    """Fetch a value by (namespace, key), or None if not found."""
    conn = _get_connection()
    try:
        row = conn.execute(
            "SELECT value FROM kv_store WHERE namespace = ? AND key = ?",
            (namespace, key),
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        return None
    return json.loads(row[0]) if as_json else row[0]