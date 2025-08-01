# Standard Library
from unittest.mock import MagicMock, patch

# Third-Party Library
import pytest

# SynThesisAI Modules
from core.llm.llm_dispatch import call_checker, call_engineer, call_target_model


@pytest.fixture
def sample_output():
    return {
        "output": '{"subject": "Algebra", "topic": "Linear", "problem": "Test problem", "answer": "Test answer", "hints": {"hint1": "Test hint 1", "hint2": "Test hint 2", "hint3": "Test hint 3"}}',
        "tokens_prompt": 10,
        "tokens_completion": 20,
    }


@patch("core.llm.llm_dispatch.create_engineer_agent")
def test_call_engineer_returns_hints_dict(mock_create_engineer_agent, sample_output):
    mock_agent = MagicMock()
    mock_agent.generate.return_value = sample_output
    mock_create_engineer_agent.return_value = mock_agent

    cfg = {"provider": "openai", "model_name": "o3"}
    result = call_engineer("Algebra", "Linear", None, cfg)

    assert isinstance(result["hints"], dict)
    mock_agent.generate.assert_called_once_with(
        subject="Algebra", topic="Linear", seed_prompt=None, difficulty_level=None
    )


@patch("core.llm.llm_dispatch.create_checker_agent")
def test_call_checker_returns_validation(mock_create_checker_agent):
    mock_agent = MagicMock()
    expected = {"valid": True, "tokens": {"prompt": 5, "completion": 5}}
    mock_agent.validate.return_value = expected
    mock_create_checker_agent.return_value = mock_agent

    problem_data = {"problem": "Test"}
    cfg = {"provider": "openai", "model_name": "o3"}
    result = call_checker(problem_data, cfg, mode="initial")

    assert result == expected
    mock_agent.validate.assert_called_once_with(
        problem_data=problem_data, mode="initial"
    )


@patch("core.llm.llm_dispatch.create_target_agent")
def test_call_target_model_returns_answer(mock_create_target_agent):
    mock_agent = MagicMock()
    expected = {"answer": "42", "tokens": {"prompt": 3, "completion": 2}}
    mock_agent.solve.return_value = expected
    mock_create_target_agent.return_value = mock_agent

    cfg = {"provider": "openai", "model_name": "o3"}
    result = call_target_model("What is 6*7?", cfg)

    assert result == expected
    mock_agent.solve.assert_called_once_with(problem_text="What is 6*7?")
