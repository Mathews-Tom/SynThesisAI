import textwrap

from core.llm.openai_utils import call_openai_model
from utils.cost_estimation import safe_log_cost
from utils.costs import CostTracker
from utils.exceptions import ModelError
from utils.json_utils import safe_json_parse


def classify_subject_topic(
    problem_text: str, model_name: str = "gpt-4.1", cost_tracker: CostTracker = None
) -> tuple[str, str]:
    """
    Classifies a math problem into (subject, topic) using a lightweight LLM.
    """
    raw_prompt = textwrap.dedent(
        f"""
    You are a math taxonomy expert.

    Classify the following math problem into the most appropriate subject and topic.

    Problem:
    \"\"\"
    {problem_text.strip()}
    \"\"\"

    Respond in this JSON format:
    {{
      "subject": "...",
      "topic": "..."
    }}
    """
    ).strip()

    llm_response = call_openai_model(
        role="classifier", prompt=raw_prompt, model_name=model_name, effort="medium"
    )

    if cost_tracker:
        safe_log_cost(
            cost_tracker,
            {"provider": "openai", "model_name": model_name},
            llm_response.get("tokens_prompt", 0),
            llm_response.get("tokens_completion", 0),
            raw_prompt=raw_prompt,
            raw_output=llm_response.get("output", ""),
        )

    output = llm_response.get("output", "")
    if not output:
        raise ModelError(
            "Classifier LLM returned no usable output",
            model_name=model_name,
            provider="openai",
        )

    parsed = safe_json_parse(output)
    return parsed.get("subject", "Unknown"), parsed.get("topic", "Unknown")
