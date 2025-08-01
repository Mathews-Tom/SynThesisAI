"""
Integration tests for DSPy Engineer Agent.

These tests verify the integration of the DSPyEngineerAgent with the
rest of the system, including actual content generation and optimization.
"""

from __future__ import annotations

# Standard Library
from unittest.mock import MagicMock, patch

# Third-Party Library
import pytest

# SynThesisAI Modules
from core.dspy.engineer_agent import DSPyEngineerAgent, create_dspy_engineer_agent
from utils.exceptions import ValidationError


@pytest.mark.integration
class TestDSPyEngineerIntegration:
    """Integration tests for DSPyEngineerAgent."""

    @patch("core.dspy.base_module.DSPY_AVAILABLE", False)
    def test_fallback_when_dspy_unavailable(self) -> None:
        """Test fallback to legacy implementation when DSPy is unavailable."""
        # Create agent
        agent: DSPyEngineerAgent = create_dspy_engineer_agent()

        # Generate content (should use fallback)
        result = agent.generate("Mathematics", "Calculus", difficulty_level="Undergraduate")

        # Verify result structure
        assert "subject" in result
        assert "topic" in result
        assert "problem" in result
        assert "answer" in result
        assert "hints" in result
        assert isinstance(result["hints"], dict)
        assert len(result["hints"]) >= 3

    @patch("core.dspy.engineer_agent.DSPyEngineerAgent._convert_dspy_result")
    @patch("core.dspy.engineer_agent.DSPyEngineerAgent.initialize_dspy_module")
    def test_dspy_result_conversion(
        self, mock_initialize: MagicMock, mock_convert: MagicMock
    ) -> None:
        """Test conversion of DSPy result to expected format."""
        # Setup mocks
        mock_convert.side_effect = ValidationError("Test error", field="test")

        # Create agent with mock DSPy module
        agent: DSPyEngineerAgent = DSPyEngineerAgent()
        agent.dspy_module = lambda **kwargs: None

        # Generate content (should use fallback due to conversion error)
        result = agent.generate("Mathematics", "Calculus")

        # Verify fallback was used
        assert mock_convert.called
        assert "subject" in result
        assert "topic" in result
        assert "problem" in result

    @pytest.mark.skipif(not hasattr(pytest, "real_dspy"), reason="Requires real DSPy")
    def test_with_real_dspy(self) -> None:
        """Test with real DSPy if available (marked to skip if not)."""
        # This test only runs if pytest has a 'real_dspy' attribute
        # Create agent
        agent = create_dspy_engineer_agent()

        # Generate content
        result = agent.generate("Mathematics", "Basic Algebra", difficulty_level="High School")

        # Verify result
        assert result["subject"] == "Mathematics"
        assert result["topic"] == "Basic Algebra"
        assert "problem" in result
        assert "answer" in result
        assert "hints" in result
        assert "dspy_generated" in result and result["dspy_generated"] is True


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
