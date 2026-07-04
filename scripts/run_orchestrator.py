"""
Manual smoke test: send a query through the full Planner -> Extractor/QA
agent pipeline and print the routing decision, trace, and final answer.

Requires at least one paper already ingested (test_vectorstore.py).

Run: python scripts/run_orchestrator.py "your query here"
"""

import sys

from researchmind.orchestrator import handle_query


def main() -> None:
    if len(sys.argv) != 2:
        print('Usage: python scripts/run_orchestrator.py "your query here"')
        sys.exit(1)

    query = sys.argv[1]
    print(f"Query: {query}\n")

    result = handle_query(query)

    print(f"--- Routing Decision ---")
    print(f"Intent: {result.intent}")

    print(f"\n--- Agent Trace ---")
    for msg in result.trace:
        print(msg.summary())

    print(f"\n--- Final Answer ---")
    print(result.answer)

    print("\n✅ Multi-agent orchestration confirmed working.")


if __name__ == "__main__":
    main()