"""
Structured metadata extraction with retry/repair.

The free Groq model is not guaranteed to return valid JSON on the first
try. Rather than fail outright, we send the parsing/validation error back
to the model and ask it to correct its own output, up to MAX_ATTEMPTS times.
"""

import json

from pydantic import ValidationError

from researchmind.observability import call_groq
from researchmind.retrieval import retrieve_from_source
from researchmind.schemas import PaperMetadata
from researchmind.vectorstore import get_chunks_by_source

MAX_ATTEMPTS = 3
OPENING_CHUNKS_COUNT = 3
SEMANTIC_TOP_K = 5

SYSTEM_PROMPT = """You are a metadata extraction assistant for research papers. \
You will be given excerpts from a paper and must extract structured metadata.

Respond with ONLY a JSON object, no preamble, no markdown code fences, no \
explanation. The JSON must have exactly these keys:
- "title": string, the paper's full title
- "authors": array of strings, author names
- "methodology": string, brief description of the research approach
- "dataset": string, dataset(s) used, or "Not specified" if none is mentioned
- "evaluation_metrics": array of strings, metrics used to evaluate results

If a field cannot be determined from the excerpts, use "Not specified" for \
string fields or an empty array for list fields. Do not invent information \
not present in the excerpts."""


def _gather_context(source_file: str) -> str:
    """
    Build extraction context from opening chunks + semantically retrieved
    methodology/dataset/metric-relevant chunks, deduplicated by chunk_index.

    Both retrieval steps are scoped to source_file via retrieve_from_source.
    Previously the semantic step used the unscoped retrieve(), which let
    other papers' chunks leak into this paper's extraction context whenever
    their content scored well against the semantic query — this caused a
    reproducible cross-paper attribution bug (e.g. one paper's dataset
    description bleeding into another paper's metadata). Fixed by scoping
    both retrieval calls to the same source_file.
    """
    all_chunks = get_chunks_by_source(source_file)
    opening = all_chunks[:OPENING_CHUNKS_COUNT]

    semantic_hits = retrieve_from_source(
        "research methodology, dataset, and evaluation metrics used in this study",
        source_file,
        top_k=SEMANTIC_TOP_K,
    )

    seen_indices = {idx for idx, _ in opening}
    combined = list(opening)

    for chunk in semantic_hits:
        if chunk.chunk_index not in seen_indices:
            combined.append((chunk.chunk_index, chunk.text))
            seen_indices.add(chunk.chunk_index)

    combined.sort(key=lambda p: p[0])
    return "\n\n".join(f"[chunk {idx}]\n{text}" for idx, text in combined)


def _strip_code_fences(text: str) -> str:
    """Defensively strip markdown code fences if the model adds them anyway."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:] if lines[0].startswith("```") else lines
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()


def extract_metadata(source_file: str) -> PaperMetadata:
    """
    Extract structured metadata for a paper, with retry-and-repair on
    malformed or schema-invalid JSON output.

    Raises:
        ValueError: if source_file has no chunks in the vector store.
        RuntimeError: if Groq fails to produce valid output after
            MAX_ATTEMPTS tries, or if the API itself fails.
    """
    context = _gather_context(source_file)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Paper excerpts:\n\n{context}"},
    ]

    last_error: str = ""

    for attempt in range(1, MAX_ATTEMPTS + 1):
        result = call_groq(
            messages=messages,
            caller=f"metadata_extraction.extract_metadata[attempt={attempt}]",
            max_tokens=800,
        )
        raw_output = result.content
        cleaned = _strip_code_fences(raw_output)

        try:
            data = json.loads(cleaned)
            return PaperMetadata.model_validate(data)
        except json.JSONDecodeError as exc:
            last_error = f"Your response was not valid JSON: {exc}"
        except ValidationError as exc:
            last_error = f"Your JSON did not match the required schema: {exc}"

        messages.append({"role": "assistant", "content": raw_output})
        messages.append({
            "role": "user",
            "content": f"{last_error}\n\nReturn ONLY the corrected JSON object, nothing else.",
        })

    raise RuntimeError(
        f"Failed to extract valid metadata after {MAX_ATTEMPTS} attempts. "
        f"Last error: {last_error}"
    )