"""
Planner agent.

Classifies a user query into one of five intents and selects which
ingested paper(s) are relevant, using Groq with retry-and-repair against
the PlannerDecision schema.
"""

import json

from pydantic import ValidationError

from researchmind.agents.messages import AgentMessage
from researchmind.observability import call_groq
from researchmind.schemas import PlannerDecision

MAX_ATTEMPTS = 3

SYSTEM_PROMPT = """You are a routing planner for a research paper analysis system. \
Given a user query and a list of available papers, decide which capability should \
handle it and which paper(s) are relevant.

Intents:
- "single_paper_qa": a factual question answerable from one specific paper, or from \
whichever paper seems most relevant if none is named.
- "metadata_extraction": the user wants structured facts about a paper (title, \
authors, methodology, dataset, evaluation metrics) rather than a specific answer.
- "comparison": the user wants two or more specific papers compared or contrasted.
- "analysis": the user wants cross-paper trends, patterns, or research-gap \
identification across the ingested papers.
- "survey": the user wants a literature-review-style synthesis or summary across \
papers.

Respond with ONLY a JSON object, no preamble, no markdown fences. The JSON must have \
exactly these keys:
- "intent": one of "single_paper_qa", "metadata_extraction", "comparison", \
"analysis", "survey"
- "source_files": array of filenames chosen from the available list (exact matches only)
- "reasoning": one sentence explaining the choice

For "single_paper_qa", include exactly one filename unless the query is genuinely \
ambiguous across papers, in which case include the most likely one. For "comparison", \
include two or more filenames. For "analysis" and "survey", include specific \
filenames only if the user names them; otherwise leave source_files as an empty array \
to signal "use all ingested papers." Never invent a filename not in the available list."""


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:] if lines[0].startswith("```") else lines
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()


def plan(query: str, available_files: list[str]) -> AgentMessage:
    """
    Produce a routing decision for a query.

    Raises:
        ValueError: if available_files is empty.
        RuntimeError: if Groq fails to produce a valid decision after
            MAX_ATTEMPTS tries, or the API call itself fails.
    """
    if not available_files:
        raise ValueError("No papers available to route queries to. Ingest a paper first.")

    files_block = "\n".join(f"- {f}" for f in available_files)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Available papers:\n{files_block}\n\nUser query: {query}"},
    ]

    last_error = ""

    for attempt in range(1, MAX_ATTEMPTS + 1):
        result = call_groq(
            messages=messages,
            caller=f"planner.plan[attempt={attempt}]",
            max_tokens=300,
        )
        raw_output = result.content
        cleaned = _strip_code_fences(raw_output)

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