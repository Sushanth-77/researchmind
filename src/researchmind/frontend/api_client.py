"""
Thin HTTP client wrapping the ResearchMind FastAPI backend.

Isolates all requests-library and HTTP-status-code handling behind typed
functions so streamlit_app.py never touches raw responses directly —
same isolation principle as graph.py hiding LangGraph specifics from its
callers. Every failure mode (backend down, 400, 422, 502) is normalized
into a single APIError type so the UI has one consistent way to display
any error, regardless of cause.
"""

import requests

API_BASE_URL = "http://127.0.0.1:8000"
REQUEST_TIMEOUT_SECONDS = 120  # Groq calls under the hood can be slow; see Phase 7 latency logs


class APIError(Exception):
    """Raised for any backend failure: connection refused, 4xx, or 5xx."""

    def __init__(self, status_code: int, error: str, detail: str):
        self.status_code = status_code
        self.error = error
        self.detail = detail
        super().__init__(f"[{status_code}] {error}: {detail}")


def _request(method: str, path: str, **kwargs) -> dict:
    """
    Shared request wrapper: makes the call, normalizes connection failures
    and non-2xx responses into APIError, returns parsed JSON on success.
    """
    url = f"{API_BASE_URL}{path}"
    try:
        response = requests.request(method, url, timeout=REQUEST_TIMEOUT_SECONDS, **kwargs)
    except requests.exceptions.ConnectionError:
        raise APIError(
            status_code=0,
            error="connection_error",
            detail=f"Could not connect to the backend at {API_BASE_URL}. Is uvicorn running?",
        )
    except requests.exceptions.Timeout:
        raise APIError(
            status_code=0,
            error="timeout",
            detail=f"Request to {path} timed out after {REQUEST_TIMEOUT_SECONDS}s.",
        )

    if response.ok:
        return response.json()

    try:
        body = response.json()
        error = body.get("error", "unknown_error")
        detail = body.get("detail", response.text)
    except ValueError:
        error = "unknown_error"
        detail = response.text

    raise APIError(response.status_code, error, detail)


def check_health() -> bool:
    """Return True if the backend is reachable and healthy."""
    try:
        _request("GET", "/health")
        return True
    except APIError:
        return False


def list_papers() -> list[str]:
    """Fetch the list of papers currently stored in the vector store."""
    data = _request("GET", "/papers")
    return data["papers"]


def ingest_paper(filename: str, file_bytes: bytes) -> dict:
    """Upload a PDF for background ingestion. Returns {"task_id", "filename", "status"}."""
    files = {"file": (filename, file_bytes, "application/pdf")}
    return _request("POST", "/papers/ingest", files=files)


def get_ingest_status(task_id: str) -> dict:
    """Poll ingestion status. Returns {"status", "chunks_created", "error", ...}."""
    return _request("GET", f"/papers/ingest/{task_id}")


def run_query(query: str) -> dict:
    """Run a query through the orchestration graph. Returns {"intent", "source_files", "answer", "trace"}."""
    return _request("POST", "/query", json={"query": query})