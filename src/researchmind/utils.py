"""
Shared utility helpers used across multiple modules.

Centralising these here avoids duplication between modules that need the
same logic (e.g. _strip_code_fences was previously copy-pasted verbatim
in both metadata_extraction.py and agents/planner.py).
"""


def strip_code_fences(text: str) -> str:
    """
    Strip markdown code fences (``` ... ```) if the LLM adds them despite
    being asked not to.  Applied defensively before JSON parsing so that a
    model that wraps its JSON in a fenced block doesn't cause a hard failure.
    """
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:] if lines[0].startswith("```") else lines
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()
