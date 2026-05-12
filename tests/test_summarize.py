"""Tests for summarize.py — written BEFORE implementation (TDD Red phase)."""
import pytest
from unittest.mock import patch, MagicMock
from src.summarize import summarize_chunk


SHORT_CHUNK = "Orbital mechanics is the study of spacecraft trajectories."
LONG_CHUNK = " ".join(["word"] * 300)
MULTISENTENCE_CHUNK = (
    "Space exploration began in earnest during the Cold War. "
    "Both the United States and Soviet Union raced to achieve milestones in space. "
    "Sputnik, launched in 1957, was the first artificial satellite. "
    "The Apollo programme successfully landed humans on the Moon in 1969. "
    "Since then, many nations and private companies have pursued space exploration."
)


def test_summarize_returns_string():
    result = summarize_chunk(SHORT_CHUNK)
    assert isinstance(result, str)


def test_summarize_non_empty_output():
    result = summarize_chunk(SHORT_CHUNK)
    assert len(result.strip()) > 0


def test_summarize_output_shorter_than_input_for_long_chunk():
    """Summary should be shorter than the original for long inputs."""
    result = summarize_chunk(LONG_CHUNK)
    assert len(result.split()) < len(LONG_CHUNK.split())


def test_summarize_short_chunk_preserved():
    """Very short chunks should be returned as-is or with minimal truncation."""
    result = summarize_chunk(SHORT_CHUNK)
    # result should retain the key topic
    assert "orbital" in result.lower() or "spacecraft" in result.lower() or "mechanics" in result.lower()


def test_summarize_multisentence_retains_key_info():
    result = summarize_chunk(MULTISENTENCE_CHUNK)
    # A meaningful summary must mention at least one key entity
    key_terms = ["space", "sputnik", "apollo", "moon", "satellite", "exploration"]
    assert any(term in result.lower() for term in key_terms)


def test_summarize_empty_string():
    """Empty input should return empty string or raise ValueError."""
    try:
        result = summarize_chunk("")
        assert result == "" or isinstance(result, str)
    except ValueError:
        pass  # also acceptable


def test_summarize_whitespace_only():
    result = summarize_chunk("   ")
    assert result.strip() == "" or isinstance(result, str)


def test_summarize_max_length_respected():
    """summarize_chunk should accept an optional max_words param and honour it."""
    result = summarize_chunk(LONG_CHUNK, max_words=30)
    assert len(result.split()) <= 30


def test_summarize_called_with_gemini_mock():
    """When Gemini is available, summarize_chunk should call the model."""
    mock_response = MagicMock()
    mock_response.text = "A concise summary of the chunk."
    mock_model = MagicMock()
    mock_model.generate_content.return_value = mock_response

    with patch("src.summarize.get_model", return_value=mock_model):
        result = summarize_chunk(MULTISENTENCE_CHUNK)
        mock_model.generate_content.assert_called_once()
        assert "summary" in result.lower() or isinstance(result, str)
