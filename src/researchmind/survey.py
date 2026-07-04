"""
Literature review / survey synthesis across multiple papers.

Same grounding discipline as analysis.py: the model must attribute claims
to specific papers and explicitly flag when papers don't share a real
narrative thread, rather than inventing one to satisfy a "survey" format.
"""

from dataclasses import dataclass

from researchmind.metadata_extraction import extract_metadata
from researchmind.observability import call_groq
from researchmind.retrieval import retrieve_from_source

DEFAULT_TOP_K_PER_PAPER = 6

SYSTEM_PROMPT = """You are a literature review assistant. You synthesize multiple \
papers into a short survey-style narrative.

Rules you must follow:
1. Only use information present in the provided metadata and excerpts.
2. Structure the survey as: one overview paragraph, then one short paragraph per \
paper covering its contribution and approach, then a closing paragraph on how the \
papers relate.
3. Do not invent connections between papers that share no real relationship — if the \
papers span unrelated topics or fields, say that plainly in the closing paragraph \
instead of forcing a narrative thread between them.
4. Attribute every claim to a specific paper by name.
5. Be concise: aim for roughly 200-350 words total."""


@dataclass
class SurveyResult:
    """The survey text plus which papers it was grounded in."""

    answer: str
    source_files: list[str]


def _gather_paper_context(source_file: str) -> str:
    """Combine a paper's metadata with retrieved contribution-relevant excerpts."""
    metadata = extract_metadata(source_file)
    chunks = retrieve_from_source(
        "main contribution, approach, and significance of this paper",
        source_file,
        top_k=DEFAULT_TOP_K_PER_PAPER,
    )
    excerpts = "\n\n".join(f"  {c.text}" for c in chunks)

    return (
        f"=== Paper: {source_file} ===\n"
        f"Title: {metadata.title}\n"
        f"Authors: {', '.join(metadata.authors) or 'Not specified'}\n"
        f"Methodology: {metadata.methodology}\n\n"
        f"Excerpts:\n{excerpts}"
    )


def generate_survey(source_files: list[str]) -> SurveyResult:
    """
    Generate a short literature-review-style synthesis across papers.

    Raises:
        ValueError: if source_files is empty.
        RuntimeError: propagated if the Groq API call fails.
    """
    if not source_files:
        raise ValueError("At least one source file is required for a survey.")

    context_block = "\n\n".join(_gather_paper_context(sf) for sf in source_files)

    user_prompt = f"""Papers to synthesize:

{context_block}

Write the survey following all rules."""

    result = call_groq(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        caller="survey.generate_survey",
        max_tokens=800,
    )

    return SurveyResult(answer=result.content, source_files=source_files)