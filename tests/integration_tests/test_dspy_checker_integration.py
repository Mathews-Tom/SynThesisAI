"""
Integration tests for DSPy Checker Agent.

These tests verify the integration of the DSPyCheckerAgent with the
rest of the system, including actual validation and equivalence checking.
"""

# Standard Library
from unittest.mock import patch

# Third-Party Library
import pytest

# SynThesisAI Modules
from core.dspy.checker_agent import DSPyCheckerAgent, create_dspy_checker_agent
from utils.exceptions import ValidationError


@pytest.mark.integration
class TestDSPyCheckerIntegration:
    """Integration tests for DSPyCheckerAgent."""

    @patch("core.dspy.base_module.DSPY_AVAILABLE", False)
    def test_fallback_when_dspy_unavailable(self) -> None:
        """Test fallback to legacy implementation when DSPy is unavailable."""
        # Create agent
        agent = create_dspy_checker_agent()

        # Test validation (should use fallback)
        problem_data = {
            "subject": "Mathematics",
            "problem": "Solve for x: 2x + 3 = 7",
            "answer": "x = 2",
            "hints": {
                "0": "Subtract 3 from both sides",
                "1": "Divide both sides by 2",
                "2": "Simplify to get the answer",
            },
        }

        result = agent.validate(problem_data, "initial")

        # Verify result structure
        assert "valid" in result
        assert "reason" in result
        assert "corrected_hints" in result
        assert "dspy_validated" in result and result["dspy_validated"] is False

    @patch("core.dspy.checker_agent.DSPyCheckerAgent._convert_dspy_result")
    @patch("core.dspy.checker_agent.DSPyCheckerAgent.initialize_dspy_module")
    def test_dspy_result_conversion_error(self, mock_initialize, mock_convert) -> None:
        """Test handling of conversion errors."""
        # Setup mocks
        mock_convert.side_effect = ValidationError("Test error", field="test")

        # Create agent with mock DSPy module
        agent = DSPyCheckerAgent()
        agent.dspy_module = lambda **kwargs: None

        # Test validation (should use fallback due to conversion error)
        problem_data = {
            "subject": "Mathematics",
            "problem": "Test problem",
            "answer": "Test answer",
        }

        result = agent.validate(problem_data, "initial")

        # Verify fallback was used
        assert mock_convert.called
        assert "valid" in result
        assert "dspy_validated" in result and result["dspy_validated"] is False

    def test_equivalence_check_with_different_formats(self) -> None:
        """Test equivalence checking with different answer formats."""
        # Create agent
        agent = create_dspy_checker_agent()

        # Test equivalence checking with different formats
        problem_data = {
            "subject": "Mathematics",
            "problem": "Solve for x: 2x + 3 = 7",
            "answer": "x = 2",
            "target_model_answer": "2",  # Different format but equivalent
        }

        # Use fallback for this test to ensure it works
        result = agent._fallback_validate(problem_data, "equivalence_check")

        # Verify result
        assert "valid" in result
        assert "reason" in result
        assert "equivalence_confidence" in result

    @pytest.mark.skipif(not hasattr(pytest, "real_dspy"), reason="Requires real DSPy")
    def test_with_real_dspy(self) -> None:
        """Test with real DSPy if available (marked to skip if not)."""
        # This test only runs if pytest has a 'real_dspy' attribute
        # Create agent
        agent = create_dspy_checker_agent()

        # Test validation
        problem_data = {
            "subject": "Mathematics",
            "problem": "Solve for x: 2x + 3 = 7",
            "answer": "x = 2",
            "hints": {
                "0": "Subtract 3 from both sides",
                "1": "Divide both sides by 2",
                "2": "Simplify to get the answer",
            },
            "difficulty_level": "High School",
        }

        result = agent.validate(problem_data, "initial")

        # Verify result
        assert "valid" in result
        assert "reason" in result
        assert "corrected_hints" in result
        assert "dspy_validated" in result


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
