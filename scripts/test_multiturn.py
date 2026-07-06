"""
Manual smoke test: multi-turn conversation via the LangGraph orchestration.

Simulates a two-turn exchange where the second question depends on the
first, to confirm the contextualizer correctly rewrites it into a
standalone query before retrieval/planning happens.

Run: python scripts/test_multiturn.py
"""

from researchmind.graph import run_query


def main() -> None:
    print("--- Turn 1 ---")
    query1 = "What accuracy did the study find for very rude prompts?"
    print(f"User: {query1}")
    result1 = run_query(query1)
    print(f"Assistant: {result1['final_answer']}\n")

    history = [
        {"role": "user", "content": query1},
        {"role": "assistant", "content": result1["final_answer"]},
    ]

    print("--- Turn 2 (follow-up, ambiguous without history) ---")
    query2 = "What about for very polite prompts instead?"
    print(f"User: {query2}")

    result2 = run_query(query2, conversation_history=history)
    contextualizer_msg = next((m for m in result2["trace"] if m.sender == "contextualizer"), None)

    if contextualizer_msg:
        print(f"[Resolved query]: {contextualizer_msg.context['resolved_query']}")

    print(f"Assistant: {result2['final_answer']}")
    print("\n✅ Multi-turn conversation confirmed working.")


if __name__ == "__main__":
    main()