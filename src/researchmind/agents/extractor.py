"""
Extractor agent.

Thin wrapper around metadata_extraction.extract_metadata that reports
its result as a structured AgentMessage rather than returning a raw
PaperMetadata object directly.
"""

from researchmind.agents.messages import AgentMessage
from researchmind.metadata_extraction import extract_metadata


def extract(source_files: list[str]) -> AgentMessage:
    """
    Extract structured metadata for one or more papers.

    Raises:
        ValueError: propagated if a source_file has no chunks stored.
        RuntimeError: propagated if Groq fails to produce valid metadata.
    """
    results = {sf: extract_metadata(sf) for sf in source_files}

    return AgentMessage(
        sender="extractor",
        receiver="orchestrator",
        context={"metadata": results},
        confidence=1.0,
    )