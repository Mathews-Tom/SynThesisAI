"""Fixtures for the test suite.

This module provides reusable fixtures for tests.
"""

# Standard Library
from typing import Any, Dict

# Third-Party Library
import pytest


@pytest.fixture
def dummy_config() -> Dict[str, Any]:
    """
    Provide a dummy configuration for tests.

    Returns:
        Dict[str, Any]: A dictionary containing dummy configuration parameters.
    """
    return {
        "taxonomy": {"Algebra": ["Linear Equations"]},
        "engineer_model": {"provider": "openai", "model_name": "o3"},
        "checker_model": {"provider": "openai", "model_name": "o3"},
        "target_model": {"provider": "openai", "model_name": "o3"},
        "use_search": False,
    }
