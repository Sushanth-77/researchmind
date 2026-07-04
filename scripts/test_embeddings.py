"""
Manual smoke test: confirms the local sentence-transformers embedding
model downloads and runs correctly (no API key, no network cost after
first download).

Run directly: python scripts/test_embeddings.py
"""

from sentence_transformers import SentenceTransformer

from researchmind.config import EMBEDDING_MODEL


def main() -> None:
    print(f"Loading local embedding model: {EMBEDDING_MODEL} ...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    sentences = [
        "Transformers use self-attention to process sequences in parallel.",
        "The cat sat on the mat.",
    ]

    embeddings = model.encode(sentences)

    print(f"Generated {len(embeddings)} embeddings.")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"First 5 values of embedding 0: {embeddings[0][:5]}")
    print("\n✅ Local embedding model confirmed working.")


if __name__ == "__main__":
    main()