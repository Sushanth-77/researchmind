"""
Grounded generation layer.

Builds a strict RAG prompt from retrieved chunks and calls Groq. The
prompt explicitly forbids answering from outside knowledge and requires
an explicit refusal when the retrieved context doesn't contain the
answer — this is the primary defense against hallucination in this
project, not something to be caught after the fact.
"""

from dataclasses import dataclass

from researchmind.observability import call_groq
from researchmind.retrieval import RetrievedChunk, retrieve, retrieve_from_source

DEFAULT_TOP_K = 5

REFUSAL_PHRASE = "I cannot answer this question based on the provided context."

SYSTEM_PROMPT = f"""You are a research paper analysis assistant. You answer questions \
strictly using the provided context chunks from a research paper. \

Rules you must follow:
1. Only use information present in the context below. Do not use any outside \
knowledge, even if you are confident it is correct.
2. If the context does not contain enough information to answer the question, \
respond with exactly: "{REFUSAL_PHRASE}" and nothing else.
3. When you do answer, cite the chunk number(s) you used in the format [chunk N] \
immediately after the relevant sentence.
4. Be concise and precise. Do not pad your answer with restated context."""


@dataclass
class QAResult:
    """The final answer plus the retrieved chunks it was grounded in."""

    answer: str
    source_chunks: list[RetrievedChunk]


def _build_context_block(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved chunks into a numbered context block for the prompt."""
    parts = []
    for i, chunk in enumerate(chunks):
        parts.append(f"[chunk {i}] (source: {chunk.source_file})\n{chunk.text}")
    return "\n\n".join(parts)


def _generate_answer(query: str, chunks: list[RetrievedChunk]) -> str:
    """
    Shared generation step: build the grounded prompt and call Groq.

    Isolated from retrieval so both whole-collection and single-source
    QA paths reuse identical prompting logic.

    Raises:
        RuntimeError: propagated if the Groq API call fails.
    """
    context_block = _build_context_block(chunks)

    user_prompt = f"""Context:
{context_block}

Question: {query}

Answer using only the context above, following all rules."""

    result = call_groq(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        caller="qa._generate_answer",
        max_tokens=500,
    )
    return result.content


def answer_question(query: str, top_k: int = DEFAULT_TOP_K) -> QAResult:
    """
    Answer a question using retrieval-augmented generation over the
    entire collection (all ingested papers).

    Raises:
        ValueError: propagated from retrieve() if the query is empty or
            the vector store has no data.
        RuntimeError: propagated if the Groq API call fails.
    """
    chunks = retrieve(query, top_k=top_k)
    answer = _generate_answer(query, chunks)
    return QAResult(answer=answer, source_chunks=chunks)


def answer_question_for_source(
    query: str, source_file: str, top_k: int = DEFAULT_TOP_K
) -> QAResult:
    """
    Answer a question using retrieval-augmented generation restricted
    to a single named paper.

    Raises:
        ValueError: if query is empty or source_file has no stored chunks.
        RuntimeError: propagated if the Groq API call fails.
    """
    chunks = retrieve_from_source(query, source_file, top_k=top_k)
    answer = _generate_answer(query, chunks)
    return QAResult(answer=answer, source_chunks=chunks)