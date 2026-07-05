"""
Query the Neo4j knowledge graph via the graph_agent.

Run: python scripts/query_graph.py "your structural question"
"""

import sys

from researchmind.agents.graph_agent import query_graph


def main() -> None:
    if len(sys.argv) != 2:
        print('Usage: python scripts/query_graph.py "your structural question"')
        sys.exit(1)

    msg = query_graph(sys.argv[1])
    print(msg.context["answer"])


if __name__ == "__main__":
    main()