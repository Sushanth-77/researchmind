"""
Unit tests for Pydantic schemas — defaults and validation behavior.
"""

import pytest
from pydantic import ValidationError

from researchmind.schemas import NOT_SPECIFIED, PaperMetadata, PlannerDecision


def test_paper_metadata_defaults():
    meta = PaperMetadata(title="Some Title")
    assert meta.authors == []
    assert meta.methodology == NOT_SPECIFIED
    assert meta.dataset == NOT_SPECIFIED
    assert meta.evaluation_metrics == []


def test_paper_metadata_requires_title():
    with pytest.raises(ValidationError):
        PaperMetadata()


def test_planner_decision_rejects_invalid_intent():
    with pytest.raises(ValidationError):
        PlannerDecision(intent="not_a_real_intent", source_files=[])


def test_planner_decision_accepts_all_valid_intents():
    for intent in [
        "single_paper_qa", "metadata_extraction", "comparison",
        "analysis", "survey", "knowledge_graph",
    ]:
        decision = PlannerDecision(intent=intent, source_files=["a.pdf"])
        assert decision.intent == intent