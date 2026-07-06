"""
Multi-turn conversation support.

Rewrites a follow-up query into a standalone question using recent chat
history, so retrieval and the Planner always see a self-contained query
rather than an ambiguous fragment like "what about the polite one?".

Only makes a Groq call when history is non-empty — a fresh, single-turn
conversation (every test run through Phases 4-11) costs nothing extra.
"""

from researchmind.observability import call_groq

MAX_HISTORY_TURNS = 6  # last N messages to include as context

SYSTEM_PROMPT = """You rewrite a user's latest message into a fully self-contained \
question, using the conversation history for context.

Rules:
1. If the latest message is already self-contained (no pronouns or implicit \
references to prior turns), return it completely unchanged.
2. If it depends on prior context (e.g. "what about the polite one?", "and for the \
other paper?"), rewrite it into a standalone question that includes the needed \
context explicitly.
3. Do not answer the question. Only rewrite it.
4. Respond with ONLY the rewritten question text, nothing else — no quotes, no preamble."""


def contextualize_query(query: str, history: list[dict]) -> str:
    """
    Rewrite `query` into a standalone question using `history`, if any
    history is present. history is [{"role": "user"|"assistant", "content": str}, ...].

    Raises:
        RuntimeError: propagated if the Groq API call fails after retries.
    """
    if not history:
        return query

    recent = history[-MAX_HISTORY_TURNS:]
    history_block = "\n".join(f"{turn['role']}: {turn['content']}" for turn in recent)

    result = call_groq(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Conversation history:\n{history_block}\n\nLatest message: {query}"},
        ],
        caller="conversation.contextualize_query",
        max_tokens=150,
    )
    return result.content.strip()