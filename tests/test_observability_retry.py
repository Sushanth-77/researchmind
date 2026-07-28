"""
Unit tests for Groq call retry/backoff logic. Mocks the Groq client
entirely, so no real API key or network access is required, and no real
tokens are spent — these should run in well under a second.
"""

import pytest

import researchmind.observability as obs


class _FakeTransientError(Exception):
    """Stand-in for a retryable Groq error, avoiding the need to construct
    real groq SDK exception objects (which require httpx.Response internals)."""


def _make_mock_response(content: str, prompt_tokens=10, completion_tokens=5, total_tokens=15):
    class _Usage:
        pass

    class _Message:
        pass

    class _Choice:
        pass

    class _Response:
        pass

    usage = _Usage()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    usage.total_tokens = total_tokens

    message = _Message()
    message.content = content

    choice = _Choice()
    choice.message = message

    response = _Response()
    response.choices = [choice]
    response.usage = usage
    return response


def test_call_groq_retries_transient_errors_then_succeeds(monkeypatch):
    monkeypatch.setattr(obs, "RETRYABLE_EXCEPTIONS", (_FakeTransientError,))
    monkeypatch.setattr(obs, "MAX_RETRIES", 3)
    monkeypatch.setattr(obs, "BASE_BACKOFF_SECONDS", 0.01)
    monkeypatch.setattr(obs.time, "sleep", lambda _: None)
    # Reset the singleton so the patched Groq factory is called fresh.
    monkeypatch.setattr(obs, "_groq_client", None)

    call_count = {"n": 0}

    def fake_create(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] < 3:
            raise _FakeTransientError("simulated rate limit")
        return _make_mock_response("final answer")

    class _FakeClient:
        class chat:
            class completions:
                create = staticmethod(fake_create)

    monkeypatch.setattr(obs, "Groq", lambda api_key: _FakeClient())

    result = obs.call_groq(messages=[{"role": "user", "content": "hi"}], caller="test.retry")

    assert result.content == "final answer"
    assert result.retry_attempts == 3
    assert call_count["n"] == 3


def test_call_groq_raises_runtimeerror_after_exhausting_retries(monkeypatch):
    monkeypatch.setattr(obs, "RETRYABLE_EXCEPTIONS", (_FakeTransientError,))
    monkeypatch.setattr(obs, "MAX_RETRIES", 2)
    monkeypatch.setattr(obs, "BASE_BACKOFF_SECONDS", 0.01)
    monkeypatch.setattr(obs.time, "sleep", lambda _: None)
    # Reset the singleton so the patched Groq factory is called fresh.
    monkeypatch.setattr(obs, "_groq_client", None)

    def always_fails(*args, **kwargs):
        raise _FakeTransientError("always fails")

    class _FakeClient:
        class chat:
            class completions:
                create = staticmethod(always_fails)

    monkeypatch.setattr(obs, "Groq", lambda api_key: _FakeClient())

    with pytest.raises(RuntimeError):
        obs.call_groq(messages=[{"role": "user", "content": "hi"}], caller="test.retry_fail")


def test_call_groq_does_not_retry_non_retryable_errors(monkeypatch):
    """A non-retryable error (e.g. a bad request) should fail immediately,
    not retry MAX_RETRIES times."""
    monkeypatch.setattr(obs, "RETRYABLE_EXCEPTIONS", (_FakeTransientError,))
    monkeypatch.setattr(obs, "MAX_RETRIES", 3)
    # Reset the singleton so the patched Groq factory is called fresh.
    monkeypatch.setattr(obs, "_groq_client", None)

    call_count = {"n": 0}

    def raises_other_error(*args, **kwargs):
        call_count["n"] += 1
        raise ValueError("not a retryable error type")

    class _FakeClient:
        class chat:
            class completions:
                create = staticmethod(raises_other_error)

    monkeypatch.setattr(obs, "Groq", lambda api_key: _FakeClient())

    with pytest.raises(RuntimeError):
        obs.call_groq(messages=[{"role": "user", "content": "hi"}], caller="test.no_retry")

    assert call_count["n"] == 1