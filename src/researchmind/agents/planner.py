"""
Planner agent.

Classifies a user query into one of six intents and selects which
ingested paper(s) are relevant, using Groq with retry-and-repair against
the PlannerDecision schema.
"""

import json

from pydantic import ValidationError

from researchmind.agents.messages import AgentMessage
from researchmind.observability import call_groq
from researchmind.schemas import PlannerDecision
from researchmind.utils import strip_code_fences

MAX_ATTEMPTS = 3

SYSTEM_PROMPT = """You are a routing planner for a research paper analysis system. \
Given a user query and a list of available papers (each shown as filename — title, \
ordered by relevance to the query), decide which capability should handle it and \
which paper(s) are relevant.

Two signals are given for each paper: its title, and its position in the \
relevance-ordered list. Resolve them in this priority order:
1. If the query explicitly names or describes a specific paper (by title, topic, or \
subject matter — e.g. "the Kantian ethics paper", "the RAG paper"), match it to \
whichever paper's TITLE corresponds, even if that paper is not first in the list. \
The title is the authoritative signal for an explicit reference.
2. Only when the query does NOT name or describe a specific paper, use the \
relevance-ordering as a tiebreaker and prefer papers earlier in the list.

Intents:
- "single_paper_qa": ANY specific factual question about a paper's content — \
including numbers, names, results, definitions, methods, hardware, sample sizes, \
dates, or anything else stated (or not stated) in the paper — that is not one of the \
exact five fields listed under "metadata_extraction" below. This is the default \
intent for factual questions; use it unless the query clearly matches one of the \
other intents.
- "metadata_extraction": the user explicitly wants a structured summary covering \
SPECIFICALLY these five fields together: title, authors, methodology, dataset, \
evaluation metrics — e.g. "give me the metadata for X" or "summarize the title, \
authors, and methodology of X". A question asking for ONE specific fact that happens \
to resemble one of these fields (e.g. "what dataset size did they use") is still \
"single_paper_qa", not this intent — reserve "metadata_extraction" for requests that \
clearly want the structured multi-field summary as a whole.
- "comparison": the user wants two or more specific papers compared or contrasted.
- "analysis": the user wants cross-paper trends, patterns, or research-gap \
identification across the ingested papers.
- "survey": the user wants a literature-review-style synthesis or summary across \
papers.
- "knowledge_graph": the user wants structural facts derived from the paper graph \
itself (e.g. shared authors across papers, which papers use a given evaluation \
metric, or a graph-wide summary) rather than content from within a paper.

Respond with ONLY a JSON object, no preamble, no markdown fences. The JSON must have \
exactly these keys:
- "intent": one of "single_paper_qa", "metadata_extraction", "comparison", \
"analysis", "survey", "knowledge_graph"
- "source_files": array of FILENAMES ONLY (not titles) chosen from the available \
list (exact filename matches only)
- "reasoning": one sentence explaining the choice

For "single_paper_qa", include exactly one filename unless the query is genuinely \
ambiguous across papers. For "comparison", include two or more filenames. For \
"analysis", "survey", and "knowledge_graph", include specific filenames only if the \
user names them; otherwise leave source_files as an empty array to signal "use all \
ingested papers." Never invent a filename not in the available list."""


def plan(query: str, available_files: list[str], file_titles: dict[str, str] | None = None) -> AgentMessage:
    """
    Produce a routing decision for a query.

    Raises:
        ValueError: if available_files is empty.
        RuntimeError: if Groq fails to produce a valid decision after
            MAX_ATTEMPTS tries, or the API call itself fails.
    """
    if not available_files:
        raise ValueError("No papers available to route queries to. Ingest a paper first.")

    file_titles = file_titles or {}
    files_block = "\n".join(
        f"{i+1}. {f} — {file_titles.get(f, f)}" for i, f in enumerate(available_files)
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Available papers (ordered by relevance):\n{files_block}\n\nUser query: {query}"},
    ]

    last_error = ""

    for attempt in range(1, MAX_ATTEMPTS + 1):
        result = call_groq(
            messages=messages,
            caller=f"planner.plan[attempt={attempt}]",
            max_tokens=300,
        )
        raw_output = result.content
        cleaned = strip_code_fences(raw_output)

        try:
            data = json.loads(cleaned)
            decision = PlannerDecision.model_validate(data)

            invalid = [f for f in decision.source_files if f not in available_files]
            if invalid:
                raise ValueError(f"Referenced unknown file(s): {invalid}")

            return AgentMessage(
                sender="planner",
                receiver="orchestrator",
                context={"decision": decision, "query": query},
                confidence=1.0,
            )
        except json.JSONDecodeError as exc:
            last_error = f"Your response was not valid JSON: {exc}"
        except (ValidationError, ValueError) as exc:
            last_error = str(exc)

        messages.append({"role": "assistant", "content": raw_output})
        messages.append({
            "role": "user",
            "content": f"{last_error}\n\nReturn ONLY the corrected JSON object, nothing else.",
        })

    raise RuntimeError(
        f"Planner failed to produce a valid decision after {MAX_ATTEMPTS} attempts. "
        f"Last error: {last_error}"
    )