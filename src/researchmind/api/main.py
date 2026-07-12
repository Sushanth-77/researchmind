"""
FastAPI application wrapping the ResearchMind LangGraph orchestration.

Endpoints:
- POST /papers/ingest        upload a PDF, ingestion runs in the background
- GET  /papers/ingest/{id}   poll ingestion status
- GET  /papers               list papers currently in the vector store
- POST /query                run a query through the orchestration graph
- GET  /health                basic liveness check (never requires auth)

Every domain exception, request-validation failure, and raised
HTTPException is mapped to a uniform ErrorResponse body, so callers
(including the Streamlit frontend) can handle all errors the same way.

Authentication: if config.API_KEY is set, every endpoint except /health
requires a matching X-API-Key header. If unset, the API runs open — a
loud warning is printed at startup so this isn't silently forgotten.
"""

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
    warning above. When API_KEY is set, the header must match exactly.
    """
    if API_KEY is None:
        return
    if provided_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid API key. Provide it via the X-API-Key header.",
        )


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
    for completion status. Rejects non-PDF uploads before saving anything.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are accepted.")

    papers_dir = DATA_DIR / "papers"
    papers_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = papers_dir / file.filename

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    pdf_path.write_bytes(contents)

    task_id = str(uuid.uuid4())[:8]
    task_store.create_task(task_id, filename=file.filename)
    background_tasks.add_task(run_ingestion, task_id, pdf_path)

    return IngestResponse(task_id=task_id, filename=file.filename, status="processing")


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