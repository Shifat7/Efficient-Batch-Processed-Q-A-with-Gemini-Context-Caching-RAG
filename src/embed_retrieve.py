from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

_MODEL_NAME = "all-MiniLM-L6-v2"


def build_index(chunks: list[str]):
    """Embed chunks and build a FAISS L2 index.

    Args:
        chunks: Non-empty list of text strings to index.

    Returns:
        Tuple of (SentenceTransformer model, faiss.Index, np.ndarray of embeddings).

    Raises:
        ValueError: If chunks is empty.
    """
    if not chunks:
        raise ValueError("chunks must be a non-empty list.")

    model = SentenceTransformer(_MODEL_NAME)
    embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
    embeddings = embeddings.astype(np.float32)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return model, index, embeddings


def retrieve(query: str, model, index, chunks: list[str], top_k: int = 3) -> list[str]:
    """Retrieve the top_k most semantically relevant chunks for a query.

    Args:
        query: The question or query string.
        model: A SentenceTransformer model instance.
        index: A FAISS index containing chunk embeddings.
        chunks: The original list of text chunks (same order as index).
        top_k: Number of results to return. Capped at len(chunks).

    Returns:
        List of up to top_k chunk strings, ordered by relevance (most relevant first).
    """
    top_k = min(top_k, len(chunks))
    q_emb = model.encode([query], convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
    _, indices = index.search(q_emb, top_k)
    return [chunks[i] for i in indices[0]]
