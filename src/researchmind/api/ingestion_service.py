"""
Background ingestion logic invoked by the /papers/ingest endpoint.

Isolated from the endpoint itself so the background task function has
no dependency on FastAPI's request/response objects.
"""

from pathlib import Path

from researchmind.api import task_store
from researchmind.ingestion import ingest_pdf
from researchmind.vectorstore import store_chunks


def run_ingestion(task_id: str, pdf_path: Path) -> None:
    """
    Ingest a PDF and store its chunks, updating the task store with the
    outcome. Runs in FastAPI's background threadpool — must not raise,
    since there is no request context left to catch an exception.
    """
    try:
        chunks = ingest_pdf(pdf_path)
        store_chunks(chunks)
        task_store.mark_completed(task_id, chunks_created=len(chunks))
    except Exception as exc:
        task_store.mark_failed(task_id, error=str(exc))