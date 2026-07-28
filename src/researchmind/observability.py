"""
Structured logging for every Groq API call, with retry-and-backoff
resilience for transient failures.

Centralizes all Groq calls behind call_groq() so every call site (qa,
comparison, metadata_extraction, planner, analysis, survey, conversation)
is instrumented and protected in exactly one place.

Retries only apply to transient, retryable failures (rate limits,
connection errors, timeouts, 5xx server errors) — a 4xx client error
(e.g. a bad API key) is not retried, since retrying a permanently broken
request just fails the same way, slower.
"""

import json
import random
import time
from dataclasses import dataclass

from groq import (
    APIConnectionError,
    APITimeoutError,
    Groq,
    InternalServerError,
    RateLimitError,
)

from researchmind.config import GROQ_API_KEY, GROQ_MODEL, PROJECT_ROOT

LOG_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOG_DIR / "groq_calls.jsonl"

MAX_RETRIES = 3
BASE_BACKOFF_SECONDS = 2.0

RETRYABLE_EXCEPTIONS = (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)

_groq_client: Groq | None = None


def _get_groq_client() -> Groq:
    """Return a process-level Groq client singleton.

    Instantiating Groq() initialises an HTTP transport and sets auth
    headers — doing this once per process (not once per call) removes
    measurable overhead at high call volumes.
    """
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=GROQ_API_KEY)
    return _groq_client


@dataclass
class GroqCallResult:
    """Content plus the metrics captured for this call."""

    content: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_seconds: float
    retry_attempts: int


def call_groq(
    messages: list[dict],
    caller: str,
    max_tokens: int = 500,
    temperature: float = 0.0,
) -> GroqCallResult:
    """
    Call Groq's chat completion endpoint with structured logging and
    automatic retry-with-exponential-backoff on transient failures.

    Raises:
        RuntimeError: if the call fails after all retries, or fails with
            a non-retryable error.
    """
    client = _get_groq_client()
    last_exception: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        start = time.perf_counter()
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except RETRYABLE_EXCEPTIONS as exc:
            latency = time.perf_counter() - start
            last_exception = exc
            _log_entry(
                caller, None, None, None, latency,
                error=f"attempt {attempt}/{MAX_RETRIES}: {exc}",
                retry_attempts=attempt,
            )
            if attempt < MAX_RETRIES:
                backoff = BASE_BACKOFF_SECONDS * (2 ** (attempt - 1)) + random.uniform(0, 1)
                time.sleep(backoff)
                continue
            raise RuntimeError(
                f"Groq API call failed after {MAX_RETRIES} attempts: {exc}"
            ) from exc
        except Exception as exc:
            # Non-retryable: fail immediately rather than retrying a request
            # that will fail the same way every time.
            latency = time.perf_counter() - start
            _log_entry(caller, None, None, None, latency, error=str(exc), retry_attempts=attempt)
            raise RuntimeError(f"Groq API call failed: {exc}") from exc
        else:
            latency = time.perf_counter() - start
            usage = response.usage
            content = response.choices[0].message.content

            _log_entry(
                caller, usage.prompt_tokens, usage.completion_tokens, usage.total_tokens,
                latency, retry_attempts=attempt,
            )

            return GroqCallResult(
                content=content,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
                latency_seconds=latency,
                retry_attempts=attempt,
            )

    # NOTE: this line is intentionally unreachable — the loop always either
    # returns on success, raises immediately on non-retryable errors, or
    # raises RuntimeError on the final retry attempt (see the branch at
    # `if attempt < MAX_RETRIES` above).  It is kept as a safety net for
    # any future refactor that changes the loop exit conditions.
    raise RuntimeError(f"Groq API call failed: {last_exception}")  # pragma: no cover


def _log_entry(
    caller: str,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    total_tokens: int | None,
    latency_seconds: float,
    retry_attempts: int = 1,
    error: str | None = None,
) -> None:
    """Append one structured JSON line per Groq call attempt to the log file."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    entry = {
        "timestamp": time.time(),
        "caller": caller,
        "model": GROQ_MODEL,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "latency_seconds": round(latency_seconds, 3),
        "retry_attempts": retry_attempts,
        "cost_usd": 0.0,
        "error": error,
    }

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")