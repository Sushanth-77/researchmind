"""
Chroma vector store wrapper.

Uses a local persistent client so data survives across process restarts.
Explicitly sets hnsw:space to cosine — Chroma's default is L2, which is
the wrong distance metric for sentence-transformers embeddings (which are
optimized for cosine similarity).
"""

import chromadb
from chromadb.api.models.Collection import Collection

from researchmind.config import CHROMA_DIR
from researchmind.embeddings import embed_texts
from researchmind.ingestion import Chunk

COLLECTION_NAME = "papers"


def get_client() -> chromadb.ClientAPI:
    """Return a persistent local Chroma client."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def get_collection() -> Collection:
    """
    Get or create the papers collection with explicit cosine distance.

    hnsw:space must be set at creation time — it cannot be changed on an
    existing collection, so this is only applied the first time the
    collection is created.
    """
    client = get_client()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def store_chunks(chunks: list[Chunk]) -> None:
    """
    Embed and store a list of chunks in the Chroma collection.

    Each chunk gets a unique ID of the form "<source_file>::<chunk_index>"
    so re-ingesting the same file overwrites (upserts) rather than
    duplicating entries.

    Raises:
        ValueError: if chunks is empty.
    """
    if not chunks:
        raise ValueError("Cannot store an empty list of chunks.")

    collection = get_collection()

    ids = [f"{c.source_file}::{c.chunk_index}" for c in chunks]
    documents = [c.text for c in chunks]
    metadatas = [
        {"source_file": c.source_file, "chunk_index": c.chunk_index}
        for c in chunks
    ]
    embeddings = embed_texts(documents)

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )