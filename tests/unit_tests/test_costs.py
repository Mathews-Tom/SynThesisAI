# Third-Party Library
import pytest

# SynThesisAI Modules
from utils.cost_estimation import safe_log_cost
from utils.costs import CostTracker


@pytest.mark.parametrize(
    "cfg, input_count, output_count, expected_input, expected_output",
    [
        (
            {"provider": "openai", "model_name": "gpt-4o"},
            1000,
            500,
            1000,
            500,
        ),
        (
            {
                "provider": "gemini",
                "model_name": "gemini-2.5-pro",
                "raw_prompt": "a" * 400,
                "raw_output": "b" * 800,
            },
            0,
            0,
            100,
            200,
        ),
    ],
)
def test_cost_logging_estimation(cfg, input_count, output_count, expected_input, expected_output):
    tracker = CostTracker()
    tracker.log(cfg, input_count, output_count)
    key = f"{cfg['provider']}:{cfg['model_name']}"
    stats = tracker.get_breakdown()[key]
    assert stats["input_tokens"] == expected_input
    assert stats["output_tokens"] == expected_output


def test_safe_log_cost_wraps_successfully():
    tracker = CostTracker()
    cfg = {"provider": "openai", "model_name": "gpt-4.1"}
    safe_log_cost(tracker, cfg, 50, 100, raw_prompt="q", raw_output="a")
    assert f"{cfg['provider']}:{cfg['model_name']}" in tracker.get_breakdown()
