"""
Structured logging for every Groq API call.

Centralizes all Groq calls behind call_groq() so every call site (qa,
comparison, metadata_extraction, planner, analysis, survey) is instrumented
in exactly one place, rather than duplicating timing/logging logic six times.

Not a decorator: retry loops (planner, metadata_extraction) need
per-attempt logging visibility, which a decorator wrapping the whole
outer function couldn't see.
"""

import json
import time
from dataclasses import dataclass

from groq import Groq

from researchmind.config import GROQ_API_KEY, GROQ_MODEL, PROJECT_ROOT

LOG_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOG_DIR / "groq_calls.jsonl"


@dataclass
class GroqCallResult:
    """Content plus the metrics captured for this call."""

    content: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_seconds: float


def call_groq(
    messages: list[dict],
    caller: str,
    max_tokens: int = 500,
    temperature: float = 0.0,
) -> GroqCallResult:
    """
    Call Groq's chat completion endpoint with structured logging of
    latency and token usage.

    Args:
        messages: chat messages in Groq's expected format.
        caller: identifies which module/function made this call (e.g.
            "qa.answer_question", "planner.plan[attempt=2]") for log analysis.
        max_tokens: max tokens in the response.
        temperature: sampling temperature.

    Cost is always logged as $0.0 since this is Groq's free tier, kept as
    a field for forward-compatibility if a paid tier is ever introduced.

    Raises:
        RuntimeError: if the Groq API call fails.
    """
    client = Groq(api_key=GROQ_API_KEY)
    start = time.perf_counter()

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    except Exception as exc:
        _log_entry(caller, None, None, None, time.perf_counter() - start, error=str(exc))
        raise RuntimeError(f"Groq API call failed: {exc}") from exc

    latency = time.perf_counter() - start
    usage = response.usage
    content = response.choices[0].message.content

    _log_entry(caller, usage.prompt_tokens, usage.completion_tokens, usage.total_tokens, latency)

    return GroqCallResult(
        content=content,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        total_tokens=usage.total_tokens,
        latency_seconds=latency,
    )


def _log_entry(
    caller: str,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    total_tokens: int | None,
    latency_seconds: float,
    error: str | None = None,
) -> None:
    """Append one structured JSON line per Groq call to the log file."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    entry = {
        "timestamp": time.time(),
        "caller": caller,
        "model": GROQ_MODEL,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "latency_seconds": round(latency_seconds, 3),
        "cost_usd": 0.0,
        "error": error,
    }

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")