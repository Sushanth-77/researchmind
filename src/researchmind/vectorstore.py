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
from researchmind.kv_store import delete_value

COLLECTION_NAME = "papers"

_chroma_client: chromadb.ClientAPI | None = None
_chroma_collection: Collection | None = None


def get_client() -> chromadb.ClientAPI:
    """Return a persistent local Chroma client singleton.

    Creating a PersistentClient re-opens the underlying SQLite store.
    Caching it as a module-level singleton means that cost is paid once
    per process rather than once per vectorstore call.
    """
    global _chroma_client
    if _chroma_client is None:
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return _chroma_client


def get_collection() -> Collection:
    """
    Get or create the papers collection with explicit cosine distance.

    hnsw:space must be set at creation time — it cannot be changed on an
    existing collection, so this is only applied the first time the
    collection is created.  The collection object is cached for the same
    reason the client is: every call to get_or_create_collection hits
    Chroma's internal metadata store.
    """
    global _chroma_collection
    if _chroma_collection is None:
        client = get_client()
        _chroma_collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _chroma_collection


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


def get_chunks_by_source(source_file: str) -> list[tuple[int, str]]:
    """
    Fetch all chunks for a given source file, ordered by chunk_index.

    Used for metadata extraction, where we need the paper's opening
    chunks (title/authors/abstract) rather than semantically-ranked ones.

    Raises:
        ValueError: if no chunks exist for source_file.
    """
    collection = get_collection()
    results = collection.get(where={"source_file": source_file})

    if not results["documents"]:
        raise ValueError(f"No chunks found for source_file: {source_file}")

    pairs = list(zip(
        [m["chunk_index"] for m in results["metadatas"]],
        results["documents"],
    ))
    pairs.sort(key=lambda p: p[0])
    return pairs


def list_source_files() -> list[str]:
    """
    Return the distinct source_file names currently stored in the collection.

    Used by the Planner agent to know which papers it can route queries to,
    without hardcoding filenames anywhere.

    include=["metadatas"] is specified explicitly so Chroma does not also
    fetch document text and embeddings — we only need the metadata here.
    """
    collection = get_collection()
    results = collection.get(include=["metadatas"])

    if not results["metadatas"]:
        return []

    sources = {m["source_file"] for m in results["metadatas"]}
    return sorted(sources)


def delete_source(source_file: str) -> int:
    """
    Delete all chunks for source_file from the collection and evict its
    cached metadata from the kv_store.

    Returns the number of chunks deleted.

    Raises:
        ValueError: if no chunks exist for source_file.
    """
    collection = get_collection()
    existing = collection.get(where={"source_file": source_file}, include=["metadatas"])
    if not existing["ids"]:
        raise ValueError(f"No chunks found for source_file: {source_file}")

    chunk_count = len(existing["ids"])
    collection.delete(where={"source_file": source_file})

    # Evict the cached metadata so a re-ingested paper with the same name
    # always gets fresh extraction rather than serving stale cached data.
    delete_value("paper_metadata", source_file)

    return chunk_count