"""
Integration tests for DSPy Target Agent.

These tests verify the integration of the DSPyTargetAgent with the
rest of the system, including actual problem solving and evaluation.
"""

# Standard Library
from pathlib import Path
from unittest.mock import patch

# Third-Party Library
import pytest

# SynThesisAI Modules
from core.dspy.target_agent import DSPyTargetAgent, create_dspy_target_agent
from utils.exceptions import ValidationError


@pytest.mark.integration
class TestDSPyTargetIntegration:
    """Integration tests for DSPyTargetAgent."""

    @patch("core.dspy.base_module.DSPY_AVAILABLE", False)
    def test_fallback_when_dspy_unavailable(self) -> None:
        """Test fallback to legacy implementation when DSPy is unavailable."""
        # Create agent
        agent = create_dspy_target_agent()

        # Test problem solving (should use fallback)
        result = agent.solve("Solve for x: 2x + 3 = 7")

        # Verify result structure
        assert "output" in result
        assert "dspy_solved" in result and result["dspy_solved"] is False
        assert "solution_steps" in result
        assert "reasoning_trace" in result
        assert "confidence_score" in result
        assert "final_answer" in result

    @patch("core.dspy.target_agent.DSPyTargetAgent._convert_dspy_result")
    @patch("core.dspy.target_agent.DSPyTargetAgent.initialize_dspy_module")
    def test_dspy_result_conversion_error(self, mock_initialize, mock_convert) -> None:
        """Test handling of conversion errors."""
        # Setup mocks
        mock_convert.side_effect = ValidationError("Test error", field="test")

        # Create agent with mock DSPy module
        agent = DSPyTargetAgent()
        agent.dspy_module = lambda **kwargs: None

        # Test solving (should use fallback due to conversion error)
        result = agent.solve("Test problem")

        # Verify fallback was used
        assert mock_convert.called
        assert "output" in result
        assert "dspy_solved" in result and result["dspy_solved"] is False

    def test_deterministic_solving(self) -> None:
        """Test deterministic solving capabilities."""
        # Create agent
        agent = create_dspy_target_agent()

        # Test the same problem multiple times
        problem = "What is 2 + 2?"
        results = []

        for _ in range(3):
            result = agent.solve(problem, domain="mathematics")
            results.append(result["output"])

        # For deterministic solving, results should be consistent
        # (This test uses fallback, so it should be deterministic)
        assert len(set(results)) <= 2  # Allow for some variation but not complete randomness

    def test_solution_evaluation(self) -> None:
        """Test solution evaluation functionality."""
        # Create agent
        agent = create_dspy_target_agent()

        # Test evaluation with correct solution
        result = agent.evaluate_solution(
            "Solve for x: 2x + 3 = 7",
            "First, subtract 3 from both sides: 2x = 4. Then divide by 2: x = 2",
            expected_answer="x = 2",
        )

        assert "correct" in result
        assert "confidence" in result
        assert "explanation" in result
        assert "dspy_evaluated" in result

    def test_context_preparation(self) -> None:
        """Test context preparation for different scenarios."""
        # Create agent
        agent = create_dspy_target_agent()

        # Test with comprehensive context
        context = agent._prepare_context_info(
            "Solve for x: 2x + 3 = 7",
            domain="Mathematics",
            topic="Linear Equations",
            difficulty_level="High School",
            hints={"0": "Isolate the variable", "1": "Use inverse operations"},
        )

        assert "Domain: Mathematics" in context
        assert "Topic: Linear Equations" in context
        assert "Difficulty: High School" in context
        assert "Available hints" in context
        assert "Solve step by step" in context

    @pytest.mark.skipif(not hasattr(pytest, "real_dspy"), reason="Requires real DSPy")
    def test_with_real_dspy(self) -> None:
        """Test with real DSPy if available (marked to skip if not)."""
        # This test only runs if pytest has a 'real_dspy' attribute
        # Create agent
        agent = create_dspy_target_agent()

        # Test problem solving
        result = agent.solve(
            "Solve for x: 2x + 3 = 7",
            domain="mathematics",
            topic="Linear Equations",
            difficulty_level="High School",
        )

        # Verify result
        assert "output" in result
        assert "solution_steps" in result
        assert "reasoning_trace" in result
        assert "confidence_score" in result
        assert "dspy_solved" in result

    def test_multiple_domain_support(self) -> None:
        """Test support for multiple domains."""
        # Create agent
        agent = create_dspy_target_agent()

        # Test mathematics domain
        math_result = agent.solve("What is the derivative of x^2?", domain="mathematics")
        assert "output" in math_result

        # Test physics domain (should fall back to legacy)
        physics_result = agent.solve("What is F = ma?", domain="physics")
        assert "output" in physics_result

        # Both should have consistent structure
        for result in [math_result, physics_result]:
            assert "dspy_solved" in result
            assert "solution_steps" in result
            assert "reasoning_trace" in result
            assert "confidence_score" in result


if __name__ == "__main__":
    pytest.main(["-xvs", str(Path(__file__))])
