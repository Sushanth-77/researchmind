"""
Thin HTTP client wrapping the ResearchMind FastAPI backend.
"""

import requests

API_BASE_URL = "http://127.0.0.1:8000"
REQUEST_TIMEOUT_SECONDS = 120


class APIError(Exception):
    """Raised for any backend failure: connection refused, 4xx, or 5xx."""

    def __init__(self, status_code: int, error: str, detail: str):
        self.status_code = status_code
        self.error = error
        self.detail = detail
        super().__init__(f"[{status_code}] {error}: {detail}")


def _request(method: str, path: str, **kwargs) -> dict:
    url = f"{API_BASE_URL}{path}"
    try:
        response = requests.request(method, url, timeout=REQUEST_TIMEOUT_SECONDS, **kwargs)
    except requests.exceptions.ConnectionError:
        raise APIError(0, "connection_error", f"Could not connect to the backend at {API_BASE_URL}. Is uvicorn running?")
    except requests.exceptions.Timeout:
        raise APIError(0, "timeout", f"Request to {path} timed out after {REQUEST_TIMEOUT_SECONDS}s.")

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
    try:
        _request("GET", "/health")
        return True
    except APIError:
        return False


def list_papers() -> list[str]:
    data = _request("GET", "/papers")
    return data["papers"]


def ingest_paper(filename: str, file_bytes: bytes) -> dict:
    files = {"file": (filename, file_bytes, "application/pdf")}
    return _request("POST", "/papers/ingest", files=files)


def get_ingest_status(task_id: str) -> dict:
    return _request("GET", f"/papers/ingest/{task_id}")


def run_query(query: str, conversation_history: list[dict] | None = None) -> dict:
    """
    Run a query through the orchestration graph.

    conversation_history: prior turns as [{"role": "user"|"assistant",
    "content": str}, ...], used to resolve follow-up questions.
    """
    payload = {"query": query, "conversation_history": conversation_history or []}
    return _request("POST", "/query", json=payload)