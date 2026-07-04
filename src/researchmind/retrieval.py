"""
Semantic retrieval over the Chroma vector store.

Embeds a query with the same local model used for storage, then runs
Chroma's nearest-neighbor search under the collection's configured
cosine distance metric.
"""

from dataclasses import dataclass

from researchmind.embeddings import embed_texts
from researchmind.vectorstore import get_collection

DEFAULT_TOP_K = 3


@dataclass
class RetrievedChunk:
    """A chunk returned from retrieval, with its similarity score."""

    text: str
    source_file: str
    chunk_index: int
    distance: float


def retrieve(query: str, top_k: int = DEFAULT_TOP_K) -> list[RetrievedChunk]:
    """
    Retrieve the top_k most semantically similar chunks to a query.

    Args:
        query: natural-language question or search string.
        top_k: number of chunks to return.

    Raises:
        ValueError: if query is empty/whitespace, or the collection
            has no stored chunks yet.
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    collection = get_collection()
    if collection.count() == 0:
        raise ValueError(
            "Vector store is empty. Ingest and store a paper before retrieving."
        )

    query_embedding = embed_texts([query])[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
    )

    retrieved = []
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for doc, meta, dist in zip(documents, metadatas, distances):
        retrieved.append(
            RetrievedChunk(
                text=doc,
                source_file=meta["source_file"],
                chunk_index=meta["chunk_index"],
                distance=dist,
            )
        )

    return retrieved


def retrieve_from_source(
    query: str, source_file: str, top_k: int = DEFAULT_TOP_K
) -> list[RetrievedChunk]:
    """
    Retrieve the top_k most semantically similar chunks to a query,
    restricted to a single paper via metadata filtering.

    This is what makes multi-paper comparison possible without cross-paper
    contamination: each paper's context is retrieved independently, then
    combined at the prompt level rather than at the vector-search level.

    Raises:
        ValueError: if query is empty, or no chunks exist for source_file.
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    collection = get_collection()
    matching_count = collection.get(where={"source_file": source_file})
    if not matching_count["ids"]:
        raise ValueError(f"No chunks found for source_file: {source_file}")

    query_embedding = embed_texts([query])[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, len(matching_count["ids"])),
        where={"source_file": source_file},
    )

    retrieved = []
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for doc, meta, dist in zip(documents, metadatas, distances):
        retrieved.append(
            RetrievedChunk(
                text=doc,
                source_file=meta["source_file"],
                chunk_index=meta["chunk_index"],
                distance=dist,
            )
        )

    return retrieved