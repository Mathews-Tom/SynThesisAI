"""Unit tests for assert_valid_model_config in utils.validation."""

# Third-Party Library
import pytest

# SynThesisAI Modules
from utils.exceptions import ValidationError
from utils.validation import assert_valid_model_config


def test_invalid_model_config_raises():
    """assert_valid_model_config should raise ValidationError when model_name missing."""
    invalid_config = {"provider": "openai"}
    with pytest.raises(ValidationError) as exc_info:
        assert_valid_model_config("engineer", invalid_config)
    message = str(exc_info.value)
    assert "model_name" in message
