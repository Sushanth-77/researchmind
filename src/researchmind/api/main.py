"""
FastAPI application wrapping the ResearchMind LangGraph orchestration.
"""

import ntpath
import re
import secrets
import uuid
from pathlib import Path

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Security, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.requests import Request
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from starlette.exceptions import HTTPException as StarletteHTTPException

from researchmind.api import task_store
from researchmind.api.ingestion_service import run_ingestion
from researchmind.api.schemas import (
    ErrorResponse,
    IngestResponse,
    IngestStatusResponse,
    PapersListResponse,
    QueryRequest,
    QueryResponse,
    TraceEntry,
)
from researchmind.config import API_KEY, DATA_DIR
from researchmind.graph import run_query
from researchmind.vectorstore import list_source_files

app = FastAPI(
    title="ResearchMind API",
    description="Multi-agent research paper analysis platform.",
    version="0.1.0",
)

if API_KEY is None:
    print(
        "⚠️  API_KEY not set — the API is running WITHOUT authentication. "
        "Set API_KEY in .env if this backend is reachable beyond your own machine."
    )

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(provided_key: str = Security(_api_key_header)) -> None:
    """
    Auth dependency applied to every non-health endpoint.

    No-op (auth disabled) if API_KEY isn't configured — see the startup
    warning above. When API_KEY is set, the header must match exactly,
    compared in constant time so response timing can't leak information
    about how many leading characters of a guess were correct.
    """
    if API_KEY is None:
        return
    if not secrets.compare_digest(provided_key or "", API_KEY):
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid API key. Provide it via the X-API-Key header.",
        )


def _sanitize_filename(filename: str) -> str:
    """
    Reduce a client-supplied filename to a safe basename before it's used
    to build a filesystem path.

    Uses ntpath.basename() rather than pathlib.Path(...).name — pathlib's
    separator handling is platform-dependent (only '/' on Linux, both '/'
    and '\\' on Windows), which meant a Windows-style traversal attempt
    like "..\\..\\evil.pdf" was NOT stripped correctly when this ran on a
    Linux CI runner, even though it worked fine locally on Windows. ntpath
    is a pure string-splitting module that always treats both separators
    as path boundaries, regardless of the host OS actually running this
    code — so behavior is now identical on Windows, Linux, and in Docker.

    Raises:
        HTTPException: 400, if the filename is empty or becomes empty
            after sanitization (e.g. the input was pure path separators).
    """
    basename = ntpath.basename(filename)
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", basename)

    if not safe or safe in (".", ".."):
        raise HTTPException(status_code=400, detail="Invalid filename.")

    return safe


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    """Bad input (empty query, unknown paper, no papers ingested) -> 400."""
    return JSONResponse(
        status_code=400,
        content=ErrorResponse(error="invalid_request", detail=str(exc)).model_dump(),
    )


@app.exception_handler(RuntimeError)
async def runtime_error_handler(request: Request, exc: RuntimeError) -> JSONResponse:
    """Upstream Groq API failure -> 502 (this service is fine, the dependency isn't)."""
    return JSONResponse(
        status_code=502,
        content=ErrorResponse(error="upstream_failure", detail=str(exc)).model_dump(),
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Normalize Pydantic's default validation error shape to match ErrorResponse."""
    first_error = exc.errors()[0]
    field = ".".join(str(loc) for loc in first_error["loc"] if loc != "body")
    detail = f"{field}: {first_error['msg']}" if field else first_error["msg"]
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(error="validation_error", detail=detail).model_dump(),
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """
    Normalize any raised HTTPException (401 from auth, 404 unknown task_id,
    400 bad upload, etc.) to the same ErrorResponse shape as every other
    error, so callers never have to branch on which handler fired.
    """
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(error="http_error", detail=str(exc.detail)).model_dump(),
    )


@app.get("/health")
def health() -> dict:
    """Basic liveness check. Never requires authentication."""
    return {"status": "ok"}


@app.post(
    "/papers/ingest",
    response_model=IngestResponse,
    status_code=202,
    dependencies=[Depends(verify_api_key)],
)
async def ingest_paper(file: UploadFile, background_tasks: BackgroundTasks) -> IngestResponse:
    """
    Accept a PDF upload and ingest it in the background.

    Returns immediately with a task_id; poll GET /papers/ingest/{task_id}
    for completion status. Rejects non-PDF uploads and sanitizes the
    filename before it's ever used to construct a filesystem path.
    """
    safe_filename = _sanitize_filename(file.filename)

    if not safe_filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are accepted.")

    # Guard: reject if the same filename is already being ingested.
    # A second upload would overwrite the file on disk mid-read by the
    # first background task, silently corrupting that ingestion.
    if task_store.is_filename_processing(safe_filename):
        raise HTTPException(
            status_code=409,
            detail=f"{safe_filename!r} is already being ingested. Wait for it to complete before re-uploading.",
        )

    papers_dir = DATA_DIR / "papers"
    papers_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = papers_dir / safe_filename

    if papers_dir.resolve() not in pdf_path.resolve().parents:
        raise HTTPException(status_code=400, detail="Invalid filename.")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    pdf_path.write_bytes(contents)

    task_id = str(uuid.uuid4())[:8]
    task_store.create_task(task_id, filename=safe_filename)
    background_tasks.add_task(run_ingestion, task_id, pdf_path)

    return IngestResponse(task_id=task_id, filename=safe_filename, status="processing")


@app.get(
    "/papers/ingest/{task_id}",
    response_model=IngestStatusResponse,
    dependencies=[Depends(verify_api_key)],
)
def get_ingest_status(task_id: str) -> IngestStatusResponse:
    """Poll the status of a previously submitted ingestion task."""
    record = task_store.get_task(task_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Unknown task_id: {task_id}")

    return IngestStatusResponse(
        task_id=task_id,
        filename=record["filename"],
        status=record["status"],
        chunks_created=record["chunks_created"],
        error=record["error"],
    )


@app.get("/papers", response_model=PapersListResponse, dependencies=[Depends(verify_api_key)])
def get_papers() -> PapersListResponse:
    """List papers currently stored in the vector store."""
    return PapersListResponse(papers=list_source_files())


@app.post("/query", response_model=QueryResponse, dependencies=[Depends(verify_api_key)])
def query(request: QueryRequest) -> QueryResponse:
    """
    Run a query through the Contextualizer -> Planner -> (Extractor | QA |
    Analysis | Survey | KG) LangGraph orchestration.
    """
    history = [turn.model_dump() for turn in request.conversation_history]
    result = run_query(request.query, conversation_history=history)
    decision = result["planner"]["decision"]

    trace = [
        TraceEntry(
            sender=msg.sender,
            receiver=msg.receiver,
            task_id=msg.task_id,
            timestamp=msg.timestamp,
            confidence=msg.confidence,
        )
        for msg in result["trace"]
    ]

    return QueryResponse(
        intent=decision.intent,
        source_files=decision.source_files,
        answer=result["final_answer"],
        trace=trace,
    )