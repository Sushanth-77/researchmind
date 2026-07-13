"""
Paper title lookup, used for Planner routing.

Previously maintained its own separate cache. Now that extract_metadata
itself caches results persistently (see metadata_extraction.py), this is
just a thin convenience wrapper — no separate caching logic needed here,
avoiding two caches that could theoretically drift out of sync.
"""

from researchmind.metadata_extraction import extract_metadata


def get_paper_title(source_file: str) -> str:
    """
    Return the title for source_file. Falls back to the filename itself
    if extraction fails, so a single bad/unparseable paper can't break
    routing for every query.
    """
    try:
        return extract_metadata(source_file).title
    except Exception:
        return source_file


def get_titles_for(source_files: list[str]) -> dict[str, str]:
    """Return {source_file: title} for each file."""
    return {sf: get_paper_title(sf) for sf in source_files}