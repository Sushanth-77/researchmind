"""
Unit tests for filename sanitization, guarding against path traversal
in the /papers/ingest upload endpoint.
"""

import pytest
from fastapi import HTTPException

from researchmind.api.main import _sanitize_filename


def test_normal_filename_passes_through():
    assert _sanitize_filename("paper.pdf") == "paper.pdf"


def test_unix_path_traversal_stripped_to_basename():
    assert _sanitize_filename("../../etc/passwd.pdf") == "passwd.pdf"


def test_windows_path_traversal_stripped_to_basename():
    assert _sanitize_filename("..\\..\\Windows\\System32\\evil.pdf") == "evil.pdf"


def test_special_characters_replaced():
    result = _sanitize_filename("weird name!@#$.pdf")
    assert result == "weird_name____.pdf"


def test_empty_filename_rejected():
    with pytest.raises(HTTPException) as exc_info:
        _sanitize_filename("")
    assert exc_info.value.status_code == 400


def test_pure_traversal_rejected():
    with pytest.raises(HTTPException) as exc_info:
        _sanitize_filename("../../../")
    assert exc_info.value.status_code == 400