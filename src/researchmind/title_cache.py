"""
Persistent cache of paper titles, keyed by source_file, backed by the
shared SQLite kv_store.

Previously an in-memory dict, rebuilt (one Groq call per paper) on every
process restart. Now persisted, so titles survive backend restarts
instead of re-extracting metadata for every ingested paper on every
Planner call after a restart.
"""

from researchmind.kv_store import get_value, set_value
from researchmind.metadata_extraction import extract_metadata

NAMESPACE = "paper_titles"


def get_paper_title(source_file: str) -> str:
    """
    Return the cached title for source_file, extracting and caching it
    on first call. Falls back to the filename itself if extraction fails,
    so a single bad/unparseable paper can't break routing for every query.
    """
    cached = get_value(NAMESPACE, source_file)
    if cached is not None:
        return cached

    try:
        metadata = extract_metadata(source_file)
        title = metadata.title
    except Exception:
        title = source_file

    set_value(NAMESPACE, source_file, title)
    return title


def get_titles_for(source_files: list[str]) -> dict[str, str]:
    """Return {source_file: title} for each file, using the cache."""
    return {sf: get_paper_title(sf) for sf in source_files}