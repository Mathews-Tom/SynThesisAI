from typing import Any, Mapping
from utils.logging_config import get_logger
from utils.costs import CostTracker

# Get logger instance
logger = get_logger(__name__)


def safe_log_cost(
    cost_tracker: CostTracker,
    model_config: Mapping[str, Any],
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    raw_output: str = "",
    raw_prompt: str = "",
) -> None:
    """Safely log model cost with fallback values and exception handling.

    Args:
        cost_tracker: CostTracker instance.
        model_config: Mapping containing at least "provider" and "model_name".
        prompt_tokens: Number of tokens in the prompt (estimated or actual).
        completion_tokens: Number of tokens in the model output (estimated or actual).
        raw_output: Raw output string from model.
        raw_prompt: Raw prompt string sent to model.
    """
    try:
        tracker_args = {**model_config, "raw_output": raw_output, "raw_prompt": raw_prompt}
        cost_tracker.log(tracker_args, prompt_tokens, completion_tokens)
    except Exception as exc:
        logger.error("Failed to log model cost: %s", exc, exc_info=True)
