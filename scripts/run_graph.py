"""
Manual smoke test: send a query through the compiled LangGraph and print
the routing decision, agent trace, and final answer.

Requires at least one paper already ingested (test_vectorstore.py).

Run: python scripts/run_graph.py "your query here"
"""

import sys

from researchmind.graph import run_query


def main() -> None:
    if len(sys.argv) != 2:
        print('Usage: python scripts/run_graph.py "your query here"')
        sys.exit(1)

    query = sys.argv[1]
    print(f"Query: {query}\n")

    result = run_query(query)
    decision = result["planner"]["decision"]

    print("--- Routing Decision ---")
    print(f"Intent: {decision.intent}")
    print(f"Source files: {decision.source_files}")

    print("\n--- Agent Trace ---")
    for msg in result["trace"]:
        print(msg.summary())

    print("\n--- Final Answer ---")
    print(result["final_answer"])

    print("\n✅ LangGraph orchestration confirmed working.")


if __name__ == "__main__":
    main()