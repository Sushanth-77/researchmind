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

    The PDF is deleted from disk after successful ingestion: the raw file
    is only needed while chunking. Leaving it permanently would accumulate
    all uploaded PDFs on disk indefinitely (a disk leak). On failure the
    file is intentionally kept so the error can be inspected; it will be
    overwritten and cleaned up on the next successful re-upload.
    """
    try:
        chunks = ingest_pdf(pdf_path)
        store_chunks(chunks)
        task_store.mark_completed(task_id, chunks_created=len(chunks))
        pdf_path.unlink(missing_ok=True)
    except Exception as exc:
        task_store.mark_failed(task_id, error=str(exc))