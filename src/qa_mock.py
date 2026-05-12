"""Kept for backwards compatibility. Use src.qa for real Gemini calls."""
from src.qa import gemini_qa


def mock_gemini_qa(question: str, context: str) -> str:
    """Deprecated shim — now delegates to the real Gemini QA implementation.

    Kept so any external code referencing mock_gemini_qa continues to work.
    """
    return gemini_qa(question, context)
