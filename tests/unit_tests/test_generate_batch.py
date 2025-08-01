"""Unit tests for core.orchestration.generate_batch."""

# Standard Library
from unittest.mock import patch

# SynThesisAI Modules
from core.orchestration.generate_batch import _generate_and_validate_prompt
from utils.costs import CostTracker


@patch("core.orchestration.generate_batch.call_target_model")
@patch("core.orchestration.generate_batch.call_checker")
@patch("core.orchestration.generate_batch.call_engineer")
def test_prompt_accepted_on_model_failure(mock_engineer, mock_checker, mock_target, dummy_config):
    """Ensure prompt is accepted when checker validates but target model fails."""
    mock_engineer.return_value = {
        "problem": "x + 1 = 2",
        "answer": "x = 1",
        "hints": ["Subtract 1"],
        "tokens_prompt": 10,
        "tokens_completion": 5,
        "raw_prompt": "engineer",
        "raw_output": "engine-out",
    }
    mock_checker.side_effect = [
        {
            "valid": True,
            "corrected_hints": None,
            "tokens_prompt": 5,
            "tokens_completion": 5,
            "raw_prompt": "check1",
            "raw_output": "out1",
        },
        {
            "valid": False,
            "tokens_prompt": 5,
            "tokens_completion": 5,
            "raw_prompt": "check2",
            "raw_output": "out2",
        },
    ]
    mock_target.return_value = {
        "output": "x = 5",
        "tokens_prompt": 5,
        "tokens_completion": 5,
        "raw_prompt": "target",
        "raw_output": "target-out",
    }

    result, data = _generate_and_validate_prompt(dummy_config, CostTracker())
    assert result == "accepted"
