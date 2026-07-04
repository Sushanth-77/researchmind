"""
Manual smoke test: compare two already-ingested papers on a given dimension.

Both papers must already be stored in Chroma (via test_vectorstore.py).

Run: python scripts/compare_papers.py "<query>" <paper1.pdf> <paper2.pdf>
"""

import sys

from researchmind.comparison import compare_papers


def main() -> None:
    if len(sys.argv) < 4:
        print('Usage: python scripts/compare_papers.py "<query>" <paper1.pdf> <paper2.pdf> [more papers...]')
        sys.exit(1)

    query = sys.argv[1]
    source_files = sys.argv[2:]

    print(f"Comparison query: {query}")
    print(f"Papers: {', '.join(source_files)}\n")

    result = compare_papers(query, source_files)

    print("--- Comparison Answer ---")
    print(result.answer)

    print("\n--- Chunks used per paper ---")
    for source_file, chunks in result.source_chunks.items():
        print(f"\n{source_file}:")
        for i, chunk in enumerate(chunks):
            print(f"  [chunk {i}] chunk_index={chunk.chunk_index} distance={chunk.distance:.4f}")

    print("\n✅ Multi-paper comparison confirmed working.")


if __name__ == "__main__":
    main()