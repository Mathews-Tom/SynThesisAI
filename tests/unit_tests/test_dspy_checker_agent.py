"""
Unit tests for DSPy Checker Agent.

These tests verify the functionality of the DSPyCheckerAgent class,
including initialization, DSPy module management, and validation.
"""

import unittest
from unittest.mock import MagicMock, patch

import pytest

from core.dspy.checker_agent import DSPyCheckerAgent
from core.dspy.exceptions import DSPyIntegrationError
from utils.exceptions import ValidationError


class TestDSPyCheckerAgent:
    """Test DSPyCheckerAgent functionality."""

    def test_initialization(self):
        """Test DSPyCheckerAgent initialization."""
        agent = DSPyCheckerAgent()
        assert agent.agent_type == "checker"
        assert agent.config_key == "checker_model"
        assert agent.dspy_module is None
        assert agent.optimization_engine is not None

    @patch("core.dspy.checker_agent.STREAMContentGenerator")
    @patch("core.dspy.checker_agent.get_dspy_config")
    def test_initialize_dspy_module(self, mock_get_dspy_config, mock_stream_generator):
        """Test initializing DSPy module."""
        # Setup mocks
        mock_config = MagicMock()
        mock_config.is_enabled.return_value = False
        mock_get_dspy_config.return_value = mock_config

        mock_module = MagicMock()
        mock_stream_generator.return_value = mock_module

        # Create agent and initialize module
        agent = DSPyCheckerAgent()
        agent.initialize_dspy_module("mathematics", "validation")

        # Verify module was created
        mock_stream_generator.assert_called_once()
        assert agent.dspy_module == mock_module

        # Test with optimization enabled
        mock_config.is_enabled.return_value = True
        agent.optimization_engine = MagicMock()
        agent.optimization_engine.optimize_for_domain.return_value = mock_module

        agent.initialize_dspy_module("mathematics", "equivalence_check")
        agent.optimization_engine.optimize_for_domain.assert_called_once()

    @patch("core.dspy.checker_agent.STREAMContentGenerator")
    @patch("core.dspy.checker_agent.get_dspy_config")
    def test_initialize_dspy_module_failure(
        self, mock_get_dspy_config, mock_stream_generator
    ):
        """Test failure in initializing DSPy module."""
        # Setup mocks
        mock_config = MagicMock()
        mock_get_dspy_config.return_value = mock_config
        mock_stream_generator.side_effect = Exception("Test error")

        # Create agent and test initialization failure
        agent = DSPyCheckerAgent()
        with pytest.raises(DSPyIntegrationError):
            agent.initialize_dspy_module("mathematics", "validation")

    def test_get_domain_signature(self):
        """Test getting domain signatures."""
        agent = DSPyCheckerAgent()

        # Test mathematics domain with validation mode
        math_validation_sig = agent._get_domain_signature("mathematics", "validation")
        assert "problem_statement, solution, hints" in math_validation_sig
        assert "valid, reason, corrected_hints" in math_validation_sig

        # Test mathematics domain with equivalence_check mode
        math_equiv_sig = agent._get_domain_signature("mathematics", "equivalence_check")
        assert "problem_statement, true_answer, model_answer" in math_equiv_sig
        assert "equivalent, confidence_score, explanation" in math_equiv_sig

        # Test other domains
        assert agent._get_domain_signature("physics", "validation") is None
        assert (
            agent._get_domain_signature("computer science", "equivalence_check") is None
        )

    def test_prepare_dspy_inputs(self):
        """Test preparing inputs for DSPy module."""
        agent = DSPyCheckerAgent()

        # Test initial validation mode
        problem_data = {
            "problem": "Test problem",
            "answer": "Test answer",
            "hints": {"0": "Hint 1", "1": "Hint 2"},
            "difficulty_level": "Undergraduate",
        }

        inputs = agent._prepare_dspy_inputs(problem_data, "initial")
        assert inputs["problem_statement"] == "Test problem"
        assert inputs["solution"] == "Test answer"
        assert inputs["hints"] == {"0": "Hint 1", "1": "Hint 2"}
        assert inputs["expected_difficulty"] == "Undergraduate"

        # Test equivalence_check mode
        problem_data = {
            "problem": "Test problem",
            "answer": "Test answer",
            "target_model_answer": "Model answer",
            "topic_description": "Topic description",
        }

        inputs = agent._prepare_dspy_inputs(problem_data, "equivalence_check")
        assert inputs["problem_statement"] == "Test problem"
        assert inputs["true_answer"] == "Test answer"
        assert inputs["model_answer"] == "Model answer"
        assert inputs["solution_context"] == "Topic description"

        # Test invalid mode
        with pytest.raises(ValidationError):
            agent._prepare_dspy_inputs(problem_data, "invalid_mode")

    @patch("core.dspy.checker_agent.DSPyCheckerAgent.initialize_dspy_module")
    def test_validate_initial(self, mock_initialize):
        """Test validation in initial mode."""
        # Create agent with mock DSPy module
        agent = DSPyCheckerAgent()
        agent.dspy_module = MagicMock()

        # Mock DSPy result
        mock_result = MagicMock()
        mock_result.valid = True
        mock_result.reason = "Test reason"
        mock_result.corrected_hints = {"0": "Corrected hint"}
        mock_result.quality_score = 0.9
        mock_result.mathematical_accuracy = 0.95
        mock_result.pedagogical_value = 0.85

        agent.dspy_module.return_value = mock_result

        # Test validation
        problem_data = {
            "subject": "Mathematics",
            "problem": "Test problem",
            "answer": "Test answer",
            "hints": {"0": "Hint 1", "1": "Hint 2"},
        }

        result = agent.validate(problem_data, "initial")

        # Verify result
        assert result["valid"] is True
        assert result["reason"] == "Test reason"
        assert result["corrected_hints"] == {"0": "Corrected hint"}
        assert result["quality_score"] == 0.9
        assert result["mathematical_accuracy"] == 0.95
        assert result["pedagogical_value"] == 0.85
        assert result["dspy_validated"] is True

    @patch("core.dspy.checker_agent.DSPyCheckerAgent.initialize_dspy_module")
    def test_validate_equivalence_check(self, mock_initialize):
        """Test validation in equivalence_check mode."""
        # Create agent with mock DSPy module
        agent = DSPyCheckerAgent()
        agent.dspy_module = MagicMock()

        # Mock DSPy result
        mock_result = MagicMock()
        mock_result.equivalent = True
        mock_result.explanation = "Test explanation"
        mock_result.confidence_score = 0.85
        mock_result.mathematical_justification = "Test justification"

        agent.dspy_module.return_value = mock_result

        # Test validation
        problem_data = {
            "subject": "Mathematics",
            "problem": "Test problem",
            "answer": "Test answer",
            "target_model_answer": "Model answer",
        }

        result = agent.validate(problem_data, "equivalence_check")

        # Verify result
        assert result["valid"] is True
        assert result["reason"] == "Test explanation"
        assert result["equivalence_confidence"] == 0.85
        assert result["mathematical_justification"] == "Test justification"
        assert result["dspy_validated"] is True

    @patch("core.dspy.checker_agent.CheckerAgent")
    @patch("core.dspy.checker_agent.DSPyCheckerAgent.initialize_dspy_module")
    def test_fallback_validate(self, mock_initialize, mock_checker_agent):
        """Test fallback to legacy implementation."""
        # Setup mocks
        mock_initialize.side_effect = DSPyIntegrationError("Test error")

        mock_legacy_agent = MagicMock()
        mock_legacy_result = {
            "valid": True,
            "reason": "Legacy reason",
            "corrected_hints": {"0": "Legacy hint"},
            "equivalence_confidence": 0.8,
        }
        mock_legacy_agent.validate.return_value = mock_legacy_result
        mock_checker_agent.return_value = mock_legacy_agent

        # Create agent
        agent = DSPyCheckerAgent()

        # Test validation with fallback
        problem_data = {
            "subject": "Mathematics",
            "problem": "Test problem",
            "answer": "Test answer",
        }

        result = agent.validate(problem_data, "initial")

        # Verify fallback was used
        mock_legacy_agent.validate.assert_called_once()
        assert result["valid"] is True
        assert result["reason"] == "Legacy reason"
        assert result["dspy_validated"] is False


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
