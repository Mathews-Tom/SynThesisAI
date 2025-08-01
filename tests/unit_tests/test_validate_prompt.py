# Standard Library
from unittest.mock import patch

# Third-Party Library
import pytest

# SynThesisAI Modules
from core.checker.validate_prompt import validate_problem


@pytest.mark.parametrize(
    "mock_response, expected_valid, expected_corrected_hints",
    [
        (
            {
                "valid": True,
                "corrected_hints": None,
                "tokens_prompt": 5,
                "tokens_completion": 10,
                "raw_output": "OK",
                "raw_prompt": "PROMPT",
            },
            True,
            None,
        ),
        (
            {
                "valid": True,
                "corrected_hints": {"1": "Subtract 1 from both sides"},
                "tokens_prompt": 5,
                "tokens_completion": 10,
                "raw_output": "OK",
                "raw_prompt": "PROMPT",
            },
            True,
            {"1": "Subtract 1 from both sides"},
        ),
    ],
)
@patch("core.checker.validate_prompt.call_openai")
def test_validate_problem(
    mock_call_openai, mock_response, expected_valid, expected_corrected_hints
):
    """
    validate_problem should return correct structure and values for various call_openai responses.
    """
    mock_call_openai.return_value = mock_response

    problem_data = {
        "problem": "x + 1 = 2",
        "answer": "x = 1",
        "hints": {"1": "Subtract 1", "2": "Solve"},
    }

    result = validate_problem(problem_data, mode="initial", provider="openai", model_name="o3")

    assert isinstance(result, dict)
    assert result["valid"] is expected_valid
    assert result.get("corrected_hints") == expected_corrected_hints
    assert "tokens_prompt" in result
    assert "raw_output" in result
