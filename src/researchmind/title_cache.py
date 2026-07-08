"""
In-memory cache of paper titles, keyed by source_file.

Lets the Planner see human-readable titles ("Automated Kantian Ethics: A
Faithful Implementation") alongside filenames, so it can resolve explicit
descriptive references ("the Kantian ethics paper") to the correct file —
something a bare filename list or pure content-relevance ranking cannot do.

Same in-memory, single-process limitation as api/task_store.py: cache is
lost on restart, rebuilt lazily (once per paper, ever) on first use.
Acceptable at this scale; a real multi-instance deployment would want
this persisted (e.g. as Chroma collection metadata) instead.
"""

from researchmind.metadata_extraction import extract_metadata

_title_cache: dict[str, str] = {}


def get_paper_title(source_file: str) -> str:
    """
    Return the cached title for source_file, extracting and caching it
    on first call. Falls back to the filename itself if extraction fails,
    so a single bad/unparseable paper can't break routing for every query.
    """
    if source_file in _title_cache:
        return _title_cache[source_file]

    try:
        metadata = extract_metadata(source_file)
        title = metadata.title
    except Exception:
        title = source_file

    _title_cache[source_file] = title
    return title


def get_titles_for(source_files: list[str]) -> dict[str, str]:
    """Return {source_file: title} for each file, using the cache."""
    return {sf: get_paper_title(sf) for sf in source_files}