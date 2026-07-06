"""
FastAPI application wrapping the ResearchMind LangGraph orchestration.
"""

import uuid
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile
from fastapi.exceptions import RequestValidationError
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
    return JSONResponse(
        status_code=400,
        content=ErrorResponse(error="invalid_request", detail=str(exc)).model_dump(),
    )


@app.exception_handler(RuntimeError)
async def runtime_error_handler(request: Request, exc: RuntimeError) -> JSONResponse:
    return JSONResponse(
        status_code=502,
        content=ErrorResponse(error="upstream_failure", detail=str(exc)).model_dump(),
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    first_error = exc.errors()[0]
    field = ".".join(str(loc) for loc in first_error["loc"] if loc != "body")
    detail = f"{field}: {first_error['msg']}" if field else first_error["msg"]
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(error="validation_error", detail=detail).model_dump(),
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/papers/ingest", response_model=IngestResponse, status_code=202)
async def ingest_paper(file: UploadFile, background_tasks: BackgroundTasks) -> IngestResponse:
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
    return PapersListResponse(papers=list_source_files())


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    """
    Run a query through the Contextualizer -> Planner -> (Extractor | QA |
    Analysis | Survey | KG) LangGraph orchestration. conversation_history,
    if provided, lets follow-up questions resolve references to prior turns.
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