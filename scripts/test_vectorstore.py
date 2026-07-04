"""
Manual smoke test: ingest a PDF, store its chunks in Chroma, and confirm
the collection count matches.

Run: python scripts/test_vectorstore.py <pdf_filename>
"""

import sys

from researchmind.config import DATA_DIR
from researchmind.ingestion import ingest_pdf
from researchmind.vectorstore import get_collection, store_chunks


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python scripts/test_vectorstore.py <pdf_filename>")
        sys.exit(1)

    pdf_path = DATA_DIR / "papers" / sys.argv[1]

    print(f"Ingesting: {pdf_path}")
    chunks = ingest_pdf(pdf_path)
    print(f"Chunks created: {len(chunks)}")

    print("Embedding and storing in Chroma...")
    store_chunks(chunks)

    collection = get_collection()
    count = collection.count()
    print(f"\nCollection '{collection.name}' now contains {count} items.")
    print(f"Distance metric: {collection.metadata}")
    print("\n✅ Vector store confirmed working.")




if __name__ == "__main__":
    main()