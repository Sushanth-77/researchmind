"""
Manual smoke test: extract structured metadata for an already-ingested
paper (must have been run through test_vectorstore.py first).

Run: python scripts/extract_metadata.py <source_filename>
"""

import sys

from researchmind.metadata_extraction import extract_metadata


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python scripts/extract_metadata.py <source_filename>")
        sys.exit(1)

    source_file = sys.argv[1]
    print(f"Extracting metadata for: {source_file}\n")

    metadata = extract_metadata(source_file)

    print("--- Extracted Metadata ---")
    print(f"Title: {metadata.title}")
    print(f"Authors: {', '.join(metadata.authors) if metadata.authors else 'Not specified'}")
    print(f"Methodology: {metadata.methodology}")
    print(f"Dataset: {metadata.dataset}")
    print(f"Evaluation Metrics: {', '.join(metadata.evaluation_metrics) if metadata.evaluation_metrics else 'None listed'}")

    print("\n✅ Metadata extraction confirmed working.")


if __name__ == "__main__":
    main()