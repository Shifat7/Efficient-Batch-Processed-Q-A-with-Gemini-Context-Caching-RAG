"""Chunk summarisation using Gemini with a simple truncation fallback."""
from __future__ import annotations

DEFAULT_MAX_WORDS = 80
_FALLBACK_MARKER = "..."  # appended when falling back to truncation


def get_model():
    """Return a configured Gemini GenerativeModel instance.

    Importing and calling this lazily keeps the module importable even when
    GOOGLE_API_KEY is not set (useful in tests that mock get_model).
    """
    import os
    import google.generativeai as genai

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "Set GOOGLE_API_KEY or GEMINI_API_KEY to use Gemini summarisation."
        )
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-1.5-flash")


def summarize_chunk(text: str, max_words: int = DEFAULT_MAX_WORDS) -> str:
    """Summarise a single text chunk.

    Attempts to use Gemini. Falls back to word-truncation when the API key is
    not available or the model call fails.

    Args:
        text: The chunk text to summarise.
        max_words: Soft upper bound on summary length (words).

    Returns:
        A summary string no longer than max_words words.
    """
    if not text or not text.strip():
        return ""

    try:
        model = get_model()
        prompt = (
            f"Summarise the following transcript excerpt in at most {max_words} words. "
            f"Be concise and preserve the key facts.\n\n{text}"
        )
        response = model.generate_content(prompt)
        summary = response.text.strip()
        # Honour max_words even if the model ignores the instruction
        words = summary.split()
        if len(words) > max_words:
            summary = " ".join(words[:max_words]) + _FALLBACK_MARKER
        return summary
    except Exception:
        # Graceful fallback: truncate to max_words
        words = text.split()
        if len(words) <= max_words:
            return text
        return " ".join(words[:max_words]) + _FALLBACK_MARKER
