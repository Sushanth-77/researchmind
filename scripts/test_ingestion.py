"""
Manual smoke test: ingest a sample PDF and print chunking stats.

Drop any research paper PDF into data/papers/ before running.
Run: python scripts/test_ingestion.py <pdf_filename>
"""

import sys
from pathlib import Path

from researchmind.config import DATA_DIR
from researchmind.ingestion import ingest_pdf


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python scripts/test_ingestion.py <pdf_filename>")
        print(f"Place the PDF in: {DATA_DIR / 'papers'}")
        sys.exit(1)

    pdf_filename = sys.argv[1]
    pdf_path = DATA_DIR / "papers" / pdf_filename

    print(f"Ingesting: {pdf_path}")
    chunks = ingest_pdf(pdf_path)

    print(f"\nTotal chunks created: {len(chunks)}")
    print(f"\n--- First chunk (index 0) ---")
    print(chunks[0].text)
    print(f"\n--- Last chunk (index {chunks[-1].chunk_index}) ---")
    print(chunks[-1].text)
    print(f"\n✅ Ingestion pipeline confirmed working on {chunks[0].source_file}.")


if __name__ == "__main__":
    main()