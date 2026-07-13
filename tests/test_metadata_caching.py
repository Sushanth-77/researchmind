"""
Unit tests for the metadata extraction cache.

Mocks call_groq and _gather_context entirely, and uses a temporary
kv_store DB path, so these tests never touch the real Groq API, Chroma,
or the running application's cached data.
"""

import json

import researchmind.kv_store as kv_store
import researchmind.metadata_extraction as me_module


def _fake_groq_response():
    class _FakeResult:
        content = json.dumps({
            "title": "Test Paper",
            "authors": ["A. Author"],
            "methodology": "test methodology",
            "dataset": "test dataset",
            "evaluation_metrics": ["accuracy"],
        })
    return _FakeResult()


def test_extract_metadata_only_calls_groq_once_across_repeated_calls(tmp_path, monkeypatch):
    monkeypatch.setattr(kv_store, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(me_module, "_gather_context", lambda source_file: "fake context")

    call_count = {"n": 0}

    def fake_call_groq(messages, caller, max_tokens=800):
        call_count["n"] += 1
        return _fake_groq_response()

    monkeypatch.setattr(me_module, "call_groq", fake_call_groq)

    first = me_module.extract_metadata("paper_a.pdf")
    second = me_module.extract_metadata("paper_a.pdf")

    assert call_count["n"] == 1
    assert first.title == "Test Paper"
    assert second.title == "Test Paper"
    assert first == second


def test_extract_metadata_cache_is_scoped_per_source_file(tmp_path, monkeypatch):
    monkeypatch.setattr(kv_store, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(me_module, "_gather_context", lambda source_file: "fake context")

    call_count = {"n": 0}

    def fake_call_groq(messages, caller, max_tokens=800):
        call_count["n"] += 1
        return _fake_groq_response()

    monkeypatch.setattr(me_module, "call_groq", fake_call_groq)

    me_module.extract_metadata("paper_a.pdf")
    me_module.extract_metadata("paper_b.pdf")

    assert call_count["n"] == 2