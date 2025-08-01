"""
Module for generating math problems using different LLM providers.

This module provides functions to call OpenAI and Gemini models to generate
math problems, with or without a seed problem.
"""

# Standard Library
import logging
from typing import Any, Dict, List

# Third-Party Library
import google.generativeai as genai

# SynThesisAI Modules
from core.llm.openai_utils import call_openai_model
from utils.config_manager import get_config_manager
from utils.exceptions import ModelError, ValidationError
from utils.json_utils import safe_json_parse
from utils.system_messages import ENGINEER_MESSAGE, ENGINEER_MESSAGE_SEED

# Set up logging
logger = logging.getLogger(__name__)


def call_openai(
    system_prompt: str, user_prompt: str, model_name: str
) -> Dict[str, Any]:
    """
    Call the OpenAI model and parse the response.

    Args:
        system_prompt: The system prompt to use.
        user_prompt: The user prompt to use.
        model_name: The name of the OpenAI model to use.

    Returns:
        A dictionary containing the parsed output and token usage.

    Raises:
        ModelError: If the model returns no usable output.
    """
    full_prompt = f"{system_prompt.strip()}\n\n{user_prompt.strip()}"
    response = call_openai_model("engineer", full_prompt, model_name, effort="high")

    if not response or "output" not in response:
        raise ModelError(
            f"OpenAI model '{model_name}' returned no usable output.",
            model_name=model_name,
            provider="openai",
        )

    parsed_output = safe_json_parse(response["output"])
    parsed_output.update(
        {
            "tokens_prompt": response.get("tokens_prompt", 0),
            "tokens_completion": response.get("tokens_completion", 0),
        }
    )
    return parsed_output


def call_gemini(messages: List[Dict[str, str]], model_name: str) -> Dict[str, Any]:
    """
    Call the Gemini model and parse the response.

    Args:
        messages: A list of messages to send to the model.
        model_name: The name of the Gemini model to use.

    Returns:
        A dictionary containing the parsed output and token usage.

    Raises:
        ModelError: If the Gemini API key is missing.
    """
    config_manager = get_config_manager()
    gemini_key = config_manager.get_api_key("gemini")

    if not gemini_key:
        raise ModelError("Missing GEMINI_KEY", provider="gemini")

    genai.configure(api_key=gemini_key)
    prompt = "\n".join([msg["content"] for msg in messages])
    model = genai.GenerativeModel(model_name=model_name)
    response = model.generate_content(prompt)
    parsed_output = safe_json_parse(response.text)
    parsed_output.update(
        {
            "tokens_prompt": 0,
            "tokens_completion": 0,
            "raw_output": response.text,
            "raw_prompt": prompt,
        }
    )
    return parsed_output


def generate_full_problem(
    seed: str, subject: str, topic: str, provider: str, model_name: str
) -> Dict[str, Any]:
    """
    Generate a complete math problem with hints and metadata.

    This function selects the appropriate system message and user prompt based
    on whether a seed problem is provided, then calls the specified LLM
    provider to generate the problem.

    Args:
        seed: The seed problem to use as a base, if any.
        subject: The subject of the math problem.
        topic: The topic of the math problem.
        provider: The LLM provider to use ('openai' or 'gemini').
        model_name: The name of the model to use.

    Returns:
        A dictionary containing the generated problem data and token usage.

    Raises:
        ModelError: If an unsupported provider is specified.
        ValidationError: If the generated hints are invalid.
    """
    # Choose correct system message and user prompt
    if seed:
        system_prompt = ENGINEER_MESSAGE_SEED
        user_prompt = (
            "Use the following math problem as a base and modify it to make it "
            "more difficult while keeping the topic and final answer format "
            f'consistent:\n\nOriginal problem:\n"""\n{seed}\n"""'
        )
    else:
        system_prompt = ENGINEER_MESSAGE
        user_prompt = (
            f"Generate a math problem in {subject} under the topic '{topic}' "
            "with hints."
        )

    if provider == "openai":
        data = call_openai(system_prompt, user_prompt, model_name)
    elif provider == "gemini":
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        data = call_gemini(messages, model_name)
    else:
        raise ModelError(
            f"Unsupported engineer provider: {provider}", provider=provider
        )

    if not isinstance(data.get("hints"), dict) or len(data["hints"]) < 3:
        raise ValidationError("Invalid or too few hints returned.", field="hints")

    logger.info("✅ Problem generated with %d hints.", len(data["hints"]))

    return {
        "subject": data.get("subject", subject),
        "topic": data.get("topic", topic),
        "problem": data["problem"],
        "answer": data["answer"],
        "hints": data["hints"],
        "tokens_prompt": data.get("tokens_prompt", 0),
        "tokens_completion": data.get("tokens_completion", 0),
        "raw_output": data.get("raw_output", ""),
        "raw_prompt": data.get("raw_prompt", ""),
    }
