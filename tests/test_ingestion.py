"""
Unit tests for PDF ingestion and chunking logic. No external services
required — pure string manipulation, runnable in any environment.
"""

from pathlib import Path

import pytest

from researchmind.ingestion import Chunk, chunk_text, extract_text


def test_chunk_text_produces_overlapping_chunks():
    text = "A" * 1000
    chunks = chunk_text(text, source_file="synthetic.pdf", chunk_size=100, chunk_overlap=20)
    assert len(chunks) > 1
    assert all(isinstance(c, Chunk) for c in chunks)
    assert chunks[0].source_file == "synthetic.pdf"
    assert chunks[0].chunk_index == 0
    assert chunks[1].chunk_index == 1


def test_chunk_text_overlap_content_repeats():
    text = "0123456789" * 20
    chunks = chunk_text(text, source_file="synthetic.pdf", chunk_size=50, chunk_overlap=10)
    tail_of_first = chunks[0].text[-10:]
    head_of_second = chunks[1].text[:10]
    assert tail_of_first == head_of_second


def test_chunk_text_rejects_overlap_greater_or_equal_to_chunk_size():
    with pytest.raises(ValueError):
        chunk_text("some text", source_file="x.pdf", chunk_size=50, chunk_overlap=50)


def test_extract_text_raises_for_missing_file():
    with pytest.raises(FileNotFoundError):
        extract_text(Path("does_not_exist_12345.pdf"))