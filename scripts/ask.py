"""
Interactive-ish CLI for asking questions against the ingested paper(s).

Run: python scripts/ask.py "your question here"
"""

import sys

from researchmind.qa import answer_question


def main() -> None:
    if len(sys.argv) != 2:
        print('Usage: python scripts/ask.py "your question here"')
        sys.exit(1)

    query = sys.argv[1]
    print(f"Question: {query}\n")

    result = answer_question(query)

    print("--- Answer ---")
    print(result.answer)

    print("\n--- Source chunks used (by [chunk N] reference above) ---")
    for i, chunk in enumerate(result.source_chunks):
        print(f"[chunk {i}] {chunk.source_file} | original chunk_index={chunk.chunk_index} | distance={chunk.distance:.4f}")


if __name__ == "__main__":
    main()