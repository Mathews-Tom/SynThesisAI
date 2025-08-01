# Standard Library
from unittest.mock import MagicMock, patch

# Third-Party Library
import pytest

# SynThesisAI Modules
from core.orchestration.evaluate_target_model import model_attempts_answer


@pytest.fixture
def openai_response():
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "x = 2"
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 20

    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


@pytest.fixture
def gemini_model():
    mock_model = MagicMock()
    mock_model.generate_content.return_value.text = "x = 5"
    return mock_model


@patch("core.orchestration.evaluate_target_model._get_openai_client")
def test_model_attempts_answer_openai(mock_get_openai_client, openai_response):
    """Test model_attempts_answer using OpenAI provider."""
    mock_get_openai_client.return_value = openai_response

    result = model_attempts_answer(
        problem="x + 1 = 3",
        model_config={"provider": "openai", "model_name": "gpt-4.1"},
    )

    assert result["output"] == "x = 2"
    assert result["tokens_prompt"] == 10
    assert result["tokens_completion"] == 20


@patch("core.orchestration.evaluate_target_model.genai")
def test_model_attempts_answer_gemini(mock_genai, gemini_model):
    """Test model_attempts_answer using Gemini provider."""
    mock_genai.GenerativeModel.return_value = gemini_model

    result = model_attempts_answer(
        problem="2x + 1 = 11",
        model_config={"provider": "gemini", "model_name": "gemini-2.5-pro"},
    )

    assert result["output"] == "x = 5"
    assert result["tokens_prompt"] == 0
    assert result["tokens_completion"] == 0
