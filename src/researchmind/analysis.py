"""
Cross-paper trend detection and research-gap analysis.

Unlike Phases 1-5, there is no single correct answer here, so the main
engineering risk shifts from "does it hallucinate a specific fact" to
"does it manufacture a false pattern because it was asked to find one."
The system prompt explicitly permits — and expects — the model to say
two papers don't meaningfully relate, rather than force a connection.
"""

from dataclasses import dataclass

from groq import Groq

from researchmind.config import GROQ_API_KEY, GROQ_MODEL
from researchmind.metadata_extraction import extract_metadata
from researchmind.retrieval import retrieve_from_source

DEFAULT_TOP_K_PER_PAPER = 6

SYSTEM_PROMPT = """You are a research analysis assistant. You identify trends across \
multiple papers and suggest concrete research gaps.

Rules you must follow:
1. Only use information present in the provided metadata and excerpts. Do not use \
outside knowledge of these research fields.
2. Be explicit about the limits of your analysis. With only a small number of papers, \
avoid describing anything as an established "trend" unless multiple papers clearly \
converge on it — otherwise call it an "observation" specific to that paper.
3. Research gaps must be concrete and tied to a specific limitation, method, or open \
question stated in the papers — not generic statements like "more research is needed."
4. Attribute every observation to a specific paper by name.
5. If the papers are too dissimilar in topic or method to yield any meaningful \
cross-paper trend, say so explicitly and plainly rather than manufacturing a \
superficial connection between them."""


@dataclass
class AnalysisResult:
    """The analysis answer plus which papers it was grounded in."""

    answer: str
    source_files: list[str]


def _gather_paper_context(source_file: str) -> str:
    """Combine a paper's metadata with retrieved findings/limitations excerpts."""
    metadata = extract_metadata(source_file)
    chunks = retrieve_from_source(
        "key findings, contributions, results, and limitations of this paper",
        source_file,
        top_k=DEFAULT_TOP_K_PER_PAPER,
    )
    excerpts = "\n\n".join(f"  {c.text}" for c in chunks)
    metrics = ", ".join(metadata.evaluation_metrics) or "None listed"

    return (
        f"=== Paper: {source_file} ===\n"
        f"Title: {metadata.title}\n"
        f"Methodology: {metadata.methodology}\n"
        f"Dataset: {metadata.dataset}\n"
        f"Evaluation Metrics: {metrics}\n\n"
        f"Excerpts:\n{excerpts}"
    )


def analyze_trends_and_gaps(source_files: list[str]) -> AnalysisResult:
    """
    Identify cross-paper trends and suggest research gaps.

    Raises:
        ValueError: if source_files is empty.
        RuntimeError: if the Groq API call fails.
    """
    if not source_files:
        raise ValueError("At least one source file is required for analysis.")

    context_block = "\n\n".join(_gather_paper_context(sf) for sf in source_files)

    user_prompt = f"""Papers under analysis:

{context_block}

Identify any genuine cross-paper trends and suggest concrete research gaps or open \
questions that follow from these papers, following all rules."""

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
    return AnalysisResult(answer=answer, source_files=source_files)