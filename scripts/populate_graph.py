"""
Populate the Neo4j knowledge graph from all currently ingested papers.

Requires Neo4j Desktop running locally and NEO4J_PASSWORD set in .env.

Run: python scripts/populate_graph.py
"""

from researchmind.knowledge_graph import get_graph_summary, populate_graph_for_all_papers


def main() -> None:
    print("Populating knowledge graph from ingested papers...")
    results = populate_graph_for_all_papers()

    for r in results:
        print(f"\n  {r['source_file']}")
        print(f"    Title: {r['title']}")
        print(f"    Authors: {', '.join(r['authors']) or 'Unknown'}")

    summary = get_graph_summary()
    print(f"\n--- Graph Summary ---")
    print(f"Papers: {summary['papers']} | Authors: {summary['authors']} | "
          f"Methodologies: {summary['methodologies']} | Datasets: {summary['datasets']} | "
          f"Metrics: {summary['metrics']}")

    print("\n✅ Knowledge graph populated.")


if __name__ == "__main__":
    main()