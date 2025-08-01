# SynThesisAI Modules
from core.llm.openai_utils import extract_tokens_from_response


class MockUsage:
    def __init__(
        self,
        prompt_tokens: int = None,
        completion_tokens: int = None,
        input_tokens: int = None,
        output_tokens: int = None,
    ) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class MockResponse:
    def __init__(
        self,
        usage: MockUsage = None,
        response_metadata: object = None,
    ) -> None:
        self.usage = usage
        self.response_metadata = response_metadata


def test_extract_tokens_from_chat_format() -> None:
    """Chat format should return prompt and completion tokens."""
    response = MockResponse(usage=MockUsage(prompt_tokens=42, completion_tokens=100))
    assert extract_tokens_from_response(response) == (42, 100)


def test_extract_tokens_from_response_format() -> None:
    """Response format should return input and output tokens."""
    metadata = type("Meta", (), {"usage": MockUsage(input_tokens=80, output_tokens=160)})
    response = MockResponse(response_metadata=metadata)
    assert extract_tokens_from_response(response) == (80, 160)


def test_extract_tokens_missing_all() -> None:
    """Missing usage information should return zeros."""
    response = MockResponse()
    assert extract_tokens_from_response(response) == (0, 0)
