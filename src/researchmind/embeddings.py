"""
Local embedding generation via sentence-transformers.

Wrapped in a module-level singleton so the model is loaded from disk
once per process, not once per call — model loading is the expensive part.
"""

from sentence_transformers import SentenceTransformer

from researchmind.config import EMBEDDING_MODEL

_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    """Lazily load and cache the embedding model for this process."""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a batch of texts into vectors.

    Raises:
        ValueError: if texts is empty.
    """
    if not texts:
        raise ValueError("Cannot embed an empty list of texts.")

    model = get_embedding_model()
    embeddings = model.encode(texts, convert_to_numpy=True)
    return embeddings.tolist()