"""Gemini-backed QA with batch prediction support."""
from __future__ import annotations
import re


def get_model():
    """Return a configured Gemini GenerativeModel instance.

    Importing lazily keeps the module importable without a valid API key
    (tests mock this function).
    """
    import os
    import google.generativeai as genai

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "Set GOOGLE_API_KEY or GEMINI_API_KEY to use Gemini QA."
        )
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-1.5-flash")


def build_batch_prompt(questions: list[str], context: str) -> str:
    """Build a single prompt asking Gemini to answer all questions.

    Args:
        questions: List of question strings.
        context: Retrieved context text relevant to all questions.

    Returns:
        A formatted prompt string.
    """
    numbered = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
    return (
        "You are a helpful assistant. Use ONLY the context below to answer each question.\n"
        "Answer each question with a numbered response matching the question number.\n"
        "If the context does not contain enough information, say so briefly.\n\n"
        f"Context:\n{context}\n\n"
        f"Questions:\n{numbered}\n\n"
        "Answers:"
    )


def parse_batch_response(raw: str, questions: list[str]) -> dict[str, str]:
    """Parse a numbered Gemini response back into a question → answer dict.

    Args:
        raw: Raw text response from Gemini, expected format:
             "1. Answer one.\n2. Answer two.\n ..."
        questions: The original ordered list of questions.

    Returns:
        Dict mapping each question to its answer string.
        Questions with no corresponding answer are mapped to empty string.
    """
    # Split on lines that start with a number followed by . or )
    pattern = re.compile(r"^\s*\d+[.)]", re.MULTILINE)
    parts = pattern.split(raw)
    # parts[0] is any text before the first numbered item — discard it
    answers = [p.strip() for p in parts[1:]]

    result: dict[str, str] = {}
    for i, question in enumerate(questions):
        result[question] = answers[i] if i < len(answers) else ""
    return result


def gemini_qa(question: str, context: str) -> str:
    """Answer a single question using Gemini with the provided context.

    Args:
        question: The question to answer.
        context: Retrieved context chunks joined as a single string.

    Returns:
        Answer string from Gemini.

    Raises:
        Exception: Propagates any Gemini API errors.
    """
    model = get_model()
    prompt = build_batch_prompt([question], context)
    response = model.generate_content(prompt)
    raw = response.text.strip()
    parsed = parse_batch_response(raw, [question])
    return parsed.get(question, raw)


def batch_gemini_qa(questions: list[str], context: str) -> dict[str, str]:
    """Answer all questions in a single Gemini call (batch prediction).

    Reduces API calls from N (one per question) to 1.

    Args:
        questions: List of question strings.
        context: Retrieved context text.

    Returns:
        Dict mapping each question to its answer.

    Raises:
        Exception: Propagates any Gemini API errors.
    """
    if not questions:
        return {}

    model = get_model()
    prompt = build_batch_prompt(questions, context)
    response = model.generate_content(prompt)
    return parse_batch_response(response.text.strip(), questions)
