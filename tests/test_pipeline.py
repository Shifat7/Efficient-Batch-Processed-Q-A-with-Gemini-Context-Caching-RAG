"""Integration tests for pipeline.py — written BEFORE implementation (TDD Red phase)."""
import os
import tempfile
import pytest
from unittest.mock import patch, MagicMock
from src.pipeline import run_pipeline


TRANSCRIPT = (
    "Orbital mechanics is the study of spacecraft motion under gravity. "
    "The Hohmann transfer orbit is fuel-efficient. "
    "Deep space missions travel beyond the Earth-Moon system. "
    "Radioisotope generators power long-duration probes. "
    "Chemical propulsion uses fuel combustion for thrust. "
    "Gravity assists can dramatically increase spacecraft velocity without extra fuel. "
    "The Saturn V used staged chemical rockets to escape Earth gravity. "
    "Communication delays increase with distance from Earth. "
    "Autonomous systems are critical for deep space operations. "
    "Astrodynamics underpins all mission trajectory planning."
)
QUESTIONS = [
    "What is orbital mechanics?",
    "How do deep space probes get power?",
]


@pytest.fixture
def transcript_file():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(TRANSCRIPT)
        path = f.name
    yield path
    os.unlink(path)


def _make_mock_model():
    mock_response = MagicMock()
    mock_response.text = "1. Orbital mechanics is the study of motion.\n2. Probes use radioisotope generators."
    mock_model = MagicMock()
    mock_model.generate_content.return_value = mock_response
    return mock_model


# --- run_pipeline return contract ---

def test_pipeline_returns_dict(transcript_file):
    with patch("src.qa.get_model", return_value=_make_mock_model()), \
         patch("src.summarize.get_model", return_value=_make_mock_model()):
        result = run_pipeline(transcript_file, QUESTIONS)
    assert isinstance(result, dict)


def test_pipeline_keys_match_questions(transcript_file):
    with patch("src.qa.get_model", return_value=_make_mock_model()), \
         patch("src.summarize.get_model", return_value=_make_mock_model()):
        result = run_pipeline(transcript_file, QUESTIONS)
    assert set(result.keys()) == set(QUESTIONS)


def test_pipeline_values_are_non_empty_strings(transcript_file):
    with patch("src.qa.get_model", return_value=_make_mock_model()), \
         patch("src.summarize.get_model", return_value=_make_mock_model()):
        result = run_pipeline(transcript_file, QUESTIONS)
    for q, a in result.items():
        assert isinstance(a, str), f"Answer for '{q}' is not a string"
        assert len(a.strip()) > 0, f"Answer for '{q}' is empty"


def test_pipeline_single_question(transcript_file):
    single_q = ["What is orbital mechanics?"]
    with patch("src.qa.get_model", return_value=_make_mock_model()), \
         patch("src.summarize.get_model", return_value=_make_mock_model()):
        result = run_pipeline(transcript_file, single_q)
    assert len(result) == 1
    assert single_q[0] in result


def test_pipeline_missing_transcript_raises():
    with pytest.raises(FileNotFoundError):
        run_pipeline("nonexistent_transcript.txt", QUESTIONS)


def test_pipeline_empty_questions_returns_empty_dict(transcript_file):
    with patch("src.qa.get_model", return_value=_make_mock_model()), \
         patch("src.summarize.get_model", return_value=_make_mock_model()):
        result = run_pipeline(transcript_file, [])
    assert result == {}


# --- Gemini is called (batch, not per-question loop) ---

def test_pipeline_calls_gemini_once_for_all_questions(transcript_file):
    """Batch prediction: Gemini should be called ONCE regardless of question count."""
    mock_model = _make_mock_model()
    with patch("src.qa.get_model", return_value=mock_model), \
         patch("src.summarize.get_model", return_value=_make_mock_model()):
        run_pipeline(transcript_file, QUESTIONS)
    # generate_content called exactly once (batch)
    assert mock_model.generate_content.call_count == 1


# --- Metrics are logged ---

def test_pipeline_logs_metrics(transcript_file):
    """run_pipeline should return metrics alongside answers, or log them without crashing."""
    with patch("src.qa.get_model", return_value=_make_mock_model()), \
         patch("src.summarize.get_model", return_value=_make_mock_model()), \
         patch("src.utils.log_metrics") as mock_log:
        run_pipeline(transcript_file, QUESTIONS)
        mock_log.assert_called_once()
