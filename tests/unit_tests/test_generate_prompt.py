# Standard Library
from unittest.mock import patch

# SynThesisAI Modules
from core.engineer.generate_prompt import generate_full_problem


@patch("core.engineer.generate_prompt.call_openai")
def test_generate_full_problem_returns_expected_structure(mock_call_openai):
    """generate_full_problem should return dict with all expected fields and types."""
    mock_call_openai.return_value = {
        "subject": "Algebra",
        "topic": "Linear Equations",
        "problem": "Solve x + 2 = 4",
        "answer": "x = 2",
        "hints": {"step1": "Subtract 2", "step2": "Simplify", "step3": "Solve"},
        "tokens_prompt": 10,
        "tokens_completion": 20,
        "raw_output": "some output",
        "raw_prompt": "some prompt",
    }

    result = generate_full_problem(
        seed=None,
        subject="Algebra",
        topic="Linear Equations",
        provider="openai",
        model_name="o3",
    )

    assert isinstance(result, dict)
    expected_keys = {
        "subject",
        "topic",
        "problem",
        "answer",
        "hints",
        "tokens_prompt",
        "tokens_completion",
        "raw_output",
        "raw_prompt",
    }
    missing = expected_keys - set(result.keys())
    assert not missing, f"Missing keys in result: {missing}"
    assert isinstance(result["hints"], dict), "hints should be a dict"
