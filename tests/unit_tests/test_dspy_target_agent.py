"""
Unit tests for DSPy Target Agent.

These tests verify the functionality of the DSPyTargetAgent class,
including initialization, DSPy module management, and problem solving.
"""

import unittest
from unittest.mock import MagicMock, patch

import pytest

from core.dspy.exceptions import DSPyIntegrationError
from core.dspy.target_agent import DSPyTargetAgent


class TestDSPyTargetAgent:
    """Test DSPyTargetAgent functionality."""

    def test_initialization(self):
        """Test DSPyTargetAgent initialization."""
        agent = DSPyTargetAgent()
        assert agent.agent_type == "target"
        assert agent.config_key == "target_model"
        assert agent.dspy_module is None
        assert agent.optimization_engine is not None

    @patch("core.dspy.target_agent.STREAMContentGenerator")
    @patch("core.dspy.target_agent.get_dspy_config")
    def test_initialize_dspy_module(self, mock_get_dspy_config, mock_stream_generator):
        """Test initializing DSPy module."""
        # Setup mocks
        mock_config = MagicMock()
        mock_config.is_enabled.return_value = False
        mock_get_dspy_config.return_value = mock_config

        mock_module = MagicMock()
        mock_stream_generator.return_value = mock_module

        # Create agent and initialize module
        agent = DSPyTargetAgent()
        agent.initialize_dspy_module("mathematics")

        # Verify module was created
        mock_stream_generator.assert_called_once()
        assert agent.dspy_module == mock_module

        # Test with optimization enabled
        mock_config.is_enabled.return_value = True
        agent.optimization_engine = MagicMock()
        agent.optimization_engine.optimize_for_domain.return_value = mock_module

        agent.initialize_dspy_module("mathematics")
        agent.optimization_engine.optimize_for_domain.assert_called_once()

    @patch("core.dspy.target_agent.STREAMContentGenerator")
    @patch("core.dspy.target_agent.get_dspy_config")
    def test_initialize_dspy_module_failure(
        self, mock_get_dspy_config, mock_stream_generator
    ):
        """Test failure in initializing DSPy module."""
        # Setup mocks
        mock_config = MagicMock()
        mock_get_dspy_config.return_value = mock_config
        mock_stream_generator.side_effect = Exception("Test error")

        # Create agent and test initialization failure
        agent = DSPyTargetAgent()
        with pytest.raises(DSPyIntegrationError):
            agent.initialize_dspy_module("mathematics")

    def test_get_domain_signature(self):
        """Test getting domain signatures."""
        agent = DSPyTargetAgent()

        # Test mathematics domain
        math_sig = agent._get_domain_signature("mathematics")
        assert "problem_statement, context_info" in math_sig
        assert "solution_steps, final_answer, reasoning_trace" in math_sig

        # Test other domains
        assert agent._get_domain_signature("physics") is None
        assert agent._get_domain_signature("computer science") is None

    def test_prepare_context_info(self):
        """Test preparing context information."""
        agent = DSPyTargetAgent()

        # Test with minimal parameters
        context = agent._prepare_context_info("Test problem")
        assert "Solve step by step" in context
        assert "Show your reasoning" in context

        # Test with full parameters
        kwargs = {
            "domain": "Mathematics",
            "topic": "Algebra",
            "difficulty_level": "High School",
            "hints": {"0": "Hint 1", "1": "Hint 2"},
        }

        context = agent._prepare_context_info("Test problem", **kwargs)
        assert "Domain: Mathematics" in context
        assert "Topic: Algebra" in context
        assert "Difficulty: High School" in context
        assert "Available hints: Hint 1; Hint 2" in context

    @patch("core.dspy.target_agent.DSPyTargetAgent.initialize_dspy_module")
    def test_solve(self, mock_initialize):
        """Test problem solving with DSPy."""
        # Create agent with mock DSPy module
        agent = DSPyTargetAgent()
        agent.dspy_module = MagicMock()

        # Mock DSPy result
        mock_result = MagicMock()
        mock_result.solution_steps = "Step 1: Do this\nStep 2: Do that"
        mock_result.final_answer = "x = 5"
        mock_result.reasoning_trace = "Test reasoning trace"
        mock_result.confidence_score = 0.9

        agent.dspy_module.return_value = mock_result

        # Test solving
        result = agent.solve("Solve for x: 2x + 3 = 13", domain="mathematics")

        # Verify result
        assert "Step 1: Do this" in result["output"]
        assert "Final Answer: x = 5" in result["output"]
        assert result["solution_steps"] == "Step 1: Do this\nStep 2: Do that"
        assert result["final_answer"] == "x = 5"
        assert result["reasoning_trace"] == "Test reasoning trace"
        assert result["confidence_score"] == 0.9
        assert result["dspy_solved"] is True

    @patch("core.dspy.target_agent.TargetAgent")
    @patch("core.dspy.target_agent.DSPyTargetAgent.initialize_dspy_module")
    def test_fallback_solve(self, mock_initialize, mock_target_agent):
        """Test fallback to legacy implementation."""
        # Setup mocks
        mock_initialize.side_effect = DSPyIntegrationError("Test error")

        mock_legacy_agent = MagicMock()
        mock_legacy_result = {
            "output": "Legacy solution",
            "tokens_prompt": 10,
            "tokens_completion": 20,
            "latency": 1.5,
        }
        mock_legacy_agent.solve.return_value = mock_legacy_result
        mock_target_agent.return_value = mock_legacy_agent

        # Create agent
        agent = DSPyTargetAgent()

        # Test solving with fallback
        result = agent.solve("Test problem")

        # Verify fallback was used
        mock_legacy_agent.solve.assert_called_once()
        assert result["output"] == "Legacy solution"
        assert result["dspy_solved"] is False
        assert result["final_answer"] == "Legacy solution"

    def test_convert_dspy_result(self):
        """Test converting DSPy result to expected format."""
        agent = DSPyTargetAgent()

        # Create mock DSPy result
        mock_result = MagicMock()
        mock_result.solution_steps = "Step 1: Subtract 3\nStep 2: Divide by 2"
        mock_result.final_answer = "x = 5"
        mock_result.reasoning_trace = "Test reasoning"
        mock_result.confidence_score = 0.85

        # Test conversion
        result = agent._convert_dspy_result(mock_result, "Test problem")

        assert "Step 1: Subtract 3" in result["output"]
        assert "Final Answer: x = 5" in result["output"]
        assert result["solution_steps"] == "Step 1: Subtract 3\nStep 2: Divide by 2"
        assert result["final_answer"] == "x = 5"
        assert result["reasoning_trace"] == "Test reasoning"
        assert result["confidence_score"] == 0.85
        assert result["dspy_solved"] is True

        # Test with only final answer
        mock_result.solution_steps = ""
        result = agent._convert_dspy_result(mock_result, "Test problem")
        assert result["output"] == "x = 5"

        # Test with only solution steps
        mock_result.solution_steps = "Step 1: Do something"
        mock_result.final_answer = ""
        result = agent._convert_dspy_result(mock_result, "Test problem")
        assert result["output"] == "Step 1: Do something"

    def test_evaluate_solution(self):
        """Test solution evaluation."""
        agent = DSPyTargetAgent()

        # Test evaluation with expected answer
        result = agent.evaluate_solution(
            "Solve for x: 2x + 3 = 13", "x = 5", expected_answer="x = 5"
        )

        assert result["correct"] is True
        assert result["confidence"] == 0.9
        assert "matches expected answer" in result["explanation"]
        assert result["dspy_evaluated"] is True

        # Test evaluation with wrong answer
        result = agent.evaluate_solution(
            "Solve for x: 2x + 3 = 13", "x = 3", expected_answer="x = 5"
        )

        assert result["correct"] is False
        assert result["confidence"] == 0.3
        assert "does not match" in result["explanation"]

        # Test evaluation without expected answer
        result = agent.evaluate_solution("Solve for x: 2x + 3 = 13", "x = 5")

        assert result["correct"] is True
        assert result["confidence"] == 0.8
        assert "appears to be correct" in result["explanation"]


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
