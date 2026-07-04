"""
Manual smoke test: run a query against the already-populated Chroma
collection and print the top-k retrieved chunks with their distances.

Requires test_vectorstore.py to have been run first (collection must
already contain chunks).

Run: python scripts/test_retrieval.py "your question here"
"""

import sys

from researchmind.retrieval import retrieve


def main() -> None:
    if len(sys.argv) != 2:
        print('Usage: python scripts/test_retrieval.py "your question here"')
        sys.exit(1)

    query = sys.argv[1]
    print(f"Query: {query}\n")

    results = retrieve(query, top_k=3)

    for i, chunk in enumerate(results):
        print(f"--- Result {i+1} (distance={chunk.distance:.4f}) ---")
        print(f"Source: {chunk.source_file} | chunk_index: {chunk.chunk_index}")
        print(chunk.text[:300])
        print()

    print("✅ Retrieval confirmed working.")


if __name__ == "__main__":
    main()