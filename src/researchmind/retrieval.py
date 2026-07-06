"""
Semantic + keyword hybrid retrieval over the Chroma vector store.

Retrieves an oversampled candidate pool via Chroma's cosine search, then
re-ranks that pool using BM25 keyword scoring fused with the embedding
ranking via Reciprocal Rank Fusion (RRF). This catches cases pure
embedding search can miss — an exact number, acronym, or rare term that
scores well on keyword overlap but not necessarily as the top embedding
match — while keeping cost bounded: BM25 only ever runs over a small
candidate pool (top_k * OVERSAMPLE_FACTOR), never the whole collection.
"""

import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from researchmind.embeddings import embed_texts
from researchmind.vectorstore import get_collection

DEFAULT_TOP_K = 5
OVERSAMPLE_FACTOR = 4
RRF_K = 60  # standard Reciprocal Rank Fusion smoothing constant


@dataclass
class RetrievedChunk:
    """A chunk returned from retrieval, with its embedding similarity score."""

    text: str
    source_file: str
    chunk_index: int
    distance: float


def _tokenize(text: str) -> list[str]:
    """Simple lowercase word tokenizer for BM25 — no stemming/stopwords needed at this scale."""
    return re.findall(r"\w+", text.lower())


def _rerank_hybrid(query: str, candidates: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
    """
    Re-rank an embedding-retrieved candidate pool (already sorted best-first
    by cosine distance) using RRF over that embedding rank and a BM25
    keyword rank computed just for this pool.
    """
    if len(candidates) <= top_k:
        return candidates

    tokenized_corpus = [_tokenize(c.text) for c in candidates]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = bm25.get_scores(_tokenize(query))

    bm25_ranks = {
        idx: rank
        for rank, idx in enumerate(sorted(range(len(candidates)), key=lambda i: -bm25_scores[i]))
    }

    def rrf_score(embedding_rank: int) -> float:
        return 1 / (RRF_K + embedding_rank) + 1 / (RRF_K + bm25_ranks[embedding_rank])

    ranked_indices = sorted(range(len(candidates)), key=lambda i: -rrf_score(i))
    return [candidates[i] for i in ranked_indices[:top_k]]


def retrieve(query: str, top_k: int = DEFAULT_TOP_K) -> list[RetrievedChunk]:
    """
    Retrieve the top_k most relevant chunks to a query using hybrid
    (embedding + BM25) ranking over the whole collection.

    Raises:
        ValueError: if query is empty/whitespace, or the collection has no stored chunks.
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    collection = get_collection()
    if collection.count() == 0:
        raise ValueError("Vector store is empty. Ingest and store a paper before retrieving.")

    pool_size = min(top_k * OVERSAMPLE_FACTOR, collection.count())
    query_embedding = embed_texts([query])[0]

    results = collection.query(query_embeddings=[query_embedding], n_results=pool_size)

    candidates = [
        RetrievedChunk(text=doc, source_file=meta["source_file"], chunk_index=meta["chunk_index"], distance=dist)
        for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0])
    ]

    return _rerank_hybrid(query, candidates, top_k)


def retrieve_from_source(query: str, source_file: str, top_k: int = DEFAULT_TOP_K) -> list[RetrievedChunk]:
    """
    Retrieve the top_k most relevant chunks to a query, restricted to a
    single paper, using the same hybrid ranking as retrieve().

    Raises:
        ValueError: if query is empty, or no chunks exist for source_file.
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    collection = get_collection()
    matching = collection.get(where={"source_file": source_file})
    if not matching["ids"]:
        raise ValueError(f"No chunks found for source_file: {source_file}")

    pool_size = min(top_k * OVERSAMPLE_FACTOR, len(matching["ids"]))
    query_embedding = embed_texts([query])[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=pool_size,
        where={"source_file": source_file},
    )

    candidates = [
        RetrievedChunk(text=doc, source_file=meta["source_file"], chunk_index=meta["chunk_index"], distance=dist)
        for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0])
    ]

    return _rerank_hybrid(query, candidates, top_k)