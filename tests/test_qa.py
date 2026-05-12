"""Tests for qa.py (replaces qa_mock.py) — written BEFORE implementation (TDD Red phase)."""
import pytest
from unittest.mock import patch, MagicMock
from src.qa import gemini_qa, build_batch_prompt, parse_batch_response


CONTEXT = (
    "Orbital mechanics is the study of spacecraft motion. "
    "The Hohmann transfer uses minimal fuel to move between orbits. "
    "Deep space missions rely on radioisotope thermoelectric generators."
)
QUESTION = "What is a Hohmann transfer?"
QUESTIONS = [
    "What is orbital mechanics?",
    "How do deep space missions get power?",
    "What is the Hohmann transfer orbit?",
]


# --- build_batch_prompt ---

def test_build_batch_prompt_returns_string():
    prompt = build_batch_prompt(QUESTIONS, CONTEXT)
    assert isinstance(prompt, str)


def test_build_batch_prompt_contains_all_questions():
    prompt = build_batch_prompt(QUESTIONS, CONTEXT)
    for q in QUESTIONS:
        assert q in prompt


def test_build_batch_prompt_contains_context():
    prompt = build_batch_prompt(QUESTIONS, CONTEXT)
    assert "Hohmann" in prompt or "orbital" in prompt.lower()


def test_build_batch_prompt_numbered_questions():
    """Questions in prompt should be numbered so the model can mirror the numbering."""
    prompt = build_batch_prompt(QUESTIONS, CONTEXT)
    assert "1." in prompt or "1)" in prompt


# --- parse_batch_response ---

def test_parse_batch_response_returns_dict():
    raw = "1. Answer one.\n2. Answer two.\n3. Answer three."
    result = parse_batch_response(raw, QUESTIONS)
    assert isinstance(result, dict)


def test_parse_batch_response_keys_match_questions():
    raw = "1. Answer one.\n2. Answer two.\n3. Answer three."
    result = parse_batch_response(raw, QUESTIONS)
    assert set(result.keys()) == set(QUESTIONS)


def test_parse_batch_response_values_non_empty():
    raw = "1. Answer one.\n2. Answer two.\n3. Answer three."
    result = parse_batch_response(raw, QUESTIONS)
    assert all(len(v.strip()) > 0 for v in result.values())


def test_parse_batch_response_single_question():
    raw = "1. A spacecraft moves under gravity."
    result = parse_batch_response(raw, [QUESTION])
    assert QUESTION in result
    assert len(result[QUESTION].strip()) > 0


def test_parse_batch_response_handles_missing_answers():
    """If the model returns fewer answers than questions, remaining should be empty string."""
    raw = "1. Only one answer."
    result = parse_batch_response(raw, QUESTIONS)
    assert isinstance(result, dict)
    assert len(result) == len(QUESTIONS)


# --- gemini_qa (mocked Gemini) ---

def test_gemini_qa_returns_string_with_mock():
    mock_response = MagicMock()
    mock_response.text = "The Hohmann transfer is a fuel-efficient orbital manoeuvre."
    mock_model = MagicMock()
    mock_model.generate_content.return_value = mock_response

    with patch("src.qa.get_model", return_value=mock_model):
        result = gemini_qa(QUESTION, CONTEXT)
        assert isinstance(result, str)
        assert len(result.strip()) > 0


def test_gemini_qa_calls_model_once():
    mock_response = MagicMock()
    mock_response.text = "Answer."
    mock_model = MagicMock()
    mock_model.generate_content.return_value = mock_response

    with patch("src.qa.get_model", return_value=mock_model):
        gemini_qa(QUESTION, CONTEXT)
        mock_model.generate_content.assert_called_once()


def test_gemini_qa_passes_context_in_prompt():
    """The prompt sent to Gemini must contain the context text."""
    mock_response = MagicMock()
    mock_response.text = "Answer."
    mock_model = MagicMock()
    mock_model.generate_content.return_value = mock_response

    with patch("src.qa.get_model", return_value=mock_model):
        gemini_qa(QUESTION, CONTEXT)
        call_args = mock_model.generate_content.call_args
        prompt_text = str(call_args)
        assert "Hohmann" in prompt_text or "orbital" in prompt_text.lower()


def test_gemini_qa_empty_context():
    """Should not crash on empty context — returns a best-effort answer."""
    mock_response = MagicMock()
    mock_response.text = "I don't have enough context."
    mock_model = MagicMock()
    mock_model.generate_content.return_value = mock_response

    with patch("src.qa.get_model", return_value=mock_model):
        result = gemini_qa(QUESTION, "")
        assert isinstance(result, str)


def test_gemini_qa_api_error_raises():
    """API errors should propagate rather than silently return empty strings."""
    mock_model = MagicMock()
    mock_model.generate_content.side_effect = Exception("API quota exceeded")

    with patch("src.qa.get_model", return_value=mock_model):
        with pytest.raises(Exception, match="API quota exceeded"):
            gemini_qa(QUESTION, CONTEXT)
