"""Transcript ingestion and sliding-window chunking."""
from __future__ import annotations


def load_and_segment_transcript(
    path: str,
    max_words: int = 800,
    overlap_words: int = 100,
) -> list[str]:
    """Load a transcript file and split it into overlapping word-window chunks.

    The function uses a sliding window over the full word list. Paragraph
    boundaries (double newlines) are NOT used as hard split points — the
    window slides uniformly so every chunk has a predictable maximum length.

    Args:
        path: Path to a UTF-8 plain-text transcript file.
        max_words: Maximum number of words per chunk.
        overlap_words: Number of words to carry over into the next chunk.
                       Must be less than max_words.

    Returns:
        List of text chunk strings. Empty list if the file is empty.

    Raises:
        FileNotFoundError: If path does not exist.
        ValueError: If overlap_words >= max_words.
    """
    if overlap_words >= max_words:
        raise ValueError(
            f"overlap_words ({overlap_words}) must be less than max_words ({max_words})."
        )

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    if not text.strip():
        return []

    words = text.split()

    # If the whole text fits in one chunk (and no overlap needed), return early
    if len(words) <= max_words:
        return [text]

    step = max(1, max_words - overlap_words)
    chunks: list[str] = []
    i = 0
    while i < len(words):
        end = min(i + max_words, len(words))
        chunks.append(" ".join(words[i:end]))
        if end == len(words):
            break
        i += step

    return chunks
