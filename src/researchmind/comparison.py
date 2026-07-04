"""
Multi-paper comparison.

Plain Python orchestration (no agent framework): for a comparison query,
retrieve relevant context independently from each named paper, tag each
chunk with its source paper in the prompt, and ask Groq to reason across
them. Agent frameworks (Phase 4+) will formalize this pattern; here it's
just sequential function calls.
"""

from dataclasses import dataclass

from groq import Groq

from researchmind.config import GROQ_API_KEY, GROQ_MODEL
from researchmind.retrieval import RetrievedChunk, retrieve_from_source

DEFAULT_TOP_K_PER_PAPER = 4

SYSTEM_PROMPT = """You are a research paper comparison assistant. You will be given \
context excerpts from two or more papers, each clearly labeled with its source paper. \

Rules you must follow:
1. Only use information present in the excerpts below. Do not use outside knowledge.
2. Clearly attribute each claim to its source paper by name when comparing.
3. If the excerpts don't contain enough information to compare on the requested \
dimension for one or more papers, say so explicitly rather than guessing.
4. Structure your answer so similarities and differences are easy to distinguish."""


@dataclass
class ComparisonResult:
    """The comparison answer plus the chunks retrieved from each paper."""

    answer: str
    source_chunks: dict[str, list[RetrievedChunk]]


def _build_context_block(source_chunks: dict[str, list[RetrievedChunk]]) -> str:
    """Format per-paper retrieved chunks into a labeled context block."""
    sections = []
    for source_file, chunks in source_chunks.items():
        chunk_texts = "\n\n".join(
            f"  [chunk {i}] {c.text}" for i, c in enumerate(chunks)
        )
        sections.append(f"=== Paper: {source_file} ===\n{chunk_texts}")
    return "\n\n".join(sections)


def compare_papers(
    query: str,
    source_files: list[str],
    top_k_per_paper: int = DEFAULT_TOP_K_PER_PAPER,
) -> ComparisonResult:
    """
    Answer a comparison question across multiple papers.

    Args:
        query: the comparison question (e.g. "compare the methodologies").
        source_files: filenames of the papers to compare, as stored in Chroma.
        top_k_per_paper: chunks retrieved independently per paper.

    Raises:
        ValueError: if source_files has fewer than 2 entries, or if
            retrieval fails for any of them (e.g. paper not ingested).
        RuntimeError: if the Groq API call fails.
    """
    if len(source_files) < 2:
        raise ValueError("Comparison requires at least 2 source files.")

    source_chunks: dict[str, list[RetrievedChunk]] = {}
    for source_file in source_files:
        source_chunks[source_file] = retrieve_from_source(
            query, source_file, top_k=top_k_per_paper
        )

    context_block = _build_context_block(source_chunks)

    user_prompt = f"""Context from each paper:

{context_block}

Comparison question: {query}

Answer using only the context above, following all rules."""

    client = Groq(api_key=GROQ_API_KEY)

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=800,
            temperature=0.0,
        )
    except Exception as exc:
        raise RuntimeError(f"Groq API call failed: {exc}") from exc

    answer = response.choices[0].message.content

    return ComparisonResult(answer=answer, source_chunks=source_chunks)