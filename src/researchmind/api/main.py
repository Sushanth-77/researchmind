"""
FastAPI application wrapping the ResearchMind LangGraph orchestration.

Endpoints:
- POST /papers/ingest        upload a PDF, ingestion runs in the background
- GET  /papers/ingest/{id}   poll ingestion status
- GET  /papers               list papers currently in the vector store
- POST /query                run a query through the orchestration graph
- GET  /health                basic liveness check

Every domain exception (ValueError for bad input, RuntimeError for
upstream Groq failures) is mapped to a specific HTTP status via the
exception handlers below, rather than surfacing as an unhandled 500.
"""

import uuid
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile
from fastapi.requests import Request
from fastapi.responses import JSONResponse

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
from researchmind.config import DATA_DIR
from researchmind.graph import run_query
from researchmind.vectorstore import list_source_files

app = FastAPI(
    title="ResearchMind API",
    description="Multi-agent research paper analysis platform.",
    version="0.1.0",
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


@app.get("/health")
def health() -> dict:
    """Basic liveness check."""
    return {"status": "ok"}


@app.post("/papers/ingest", response_model=IngestResponse, status_code=202)
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


@app.get("/papers/ingest/{task_id}", response_model=IngestStatusResponse)
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


@app.get("/papers", response_model=PapersListResponse)
def get_papers() -> PapersListResponse:
    """List papers currently stored in the vector store."""
    return PapersListResponse(papers=list_source_files())


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    """
    Run a query through the Planner -> (Extractor | QA | Analysis | Survey)
    LangGraph orchestration and return the final answer with trace.

    Raises (mapped by the handlers above):
        ValueError -> 400: no papers ingested, empty query, unknown filename.
        RuntimeError -> 502: a Groq API call failed after retries.
    """
    result = run_query(request.query)
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