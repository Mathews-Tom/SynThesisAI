"""
Unit tests for DSPy Engineer Agent.

These tests verify the functionality of the DSPyEngineerAgent class,
including initialization, DSPy module management, and content generation.
"""

import unittest
from unittest.mock import MagicMock, patch

import pytest

from core.dspy.engineer_agent import DSPyEngineerAgent
from core.dspy.exceptions import DSPyIntegrationError


class TestDSPyEngineerAgent:
    """Test DSPyEngineerAgent functionality."""

    def test_initialization(self):
        """Test DSPyEngineerAgent initialization."""
        agent = DSPyEngineerAgent()
        assert agent.agent_type == "engineer"
        assert agent.config_key == "engineer_model"
        assert agent.dspy_module is None
        assert agent.optimization_engine is not None

    @patch("core.dspy.engineer_agent.STREAMContentGenerator")
    @patch("core.dspy.engineer_agent.get_dspy_config")
    def test_initialize_dspy_module(self, mock_get_dspy_config, mock_stream_generator):
        """Test initializing DSPy module."""
        # Setup mocks
        mock_config = MagicMock()
        mock_config.is_enabled.return_value = False
        mock_get_dspy_config.return_value = mock_config

        mock_module = MagicMock()
        mock_stream_generator.return_value = mock_module

        # Create agent and initialize module
        agent = DSPyEngineerAgent()
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

    @patch("core.dspy.engineer_agent.STREAMContentGenerator")
    @patch("core.dspy.engineer_agent.get_dspy_config")
    def test_initialize_dspy_module_failure(
        self, mock_get_dspy_config, mock_stream_generator
    ):
        """Test failure in initializing DSPy module."""
        # Setup mocks
        mock_config = MagicMock()
        mock_get_dspy_config.return_value = mock_config
        mock_stream_generator.side_effect = Exception("Test error")

        # Create agent and test initialization failure
        agent = DSPyEngineerAgent()
        with pytest.raises(DSPyIntegrationError):
            agent.initialize_dspy_module("mathematics")

    @patch("core.dspy.engineer_agent.DSPyEngineerAgent._get_topic_info")
    @patch("core.dspy.engineer_agent.DSPyEngineerAgent.initialize_dspy_module")
    def test_generate(self, mock_initialize, mock_get_topic_info):
        """Test content generation with DSPy."""
        # Setup mocks
        mock_get_topic_info.return_value = {
            "level": "Undergraduate",
            "description": "Test topic description",
        }

        # Create agent with mock DSPy module
        agent = DSPyEngineerAgent()
        agent.dspy_module = MagicMock()

        # Mock DSPy result
        mock_result = MagicMock()
        mock_result.problem_statement = "Test problem statement"
        mock_result.solution = "Test solution"
        mock_result.proof = "Test proof"
        mock_result.reasoning_trace = "Test reasoning trace"
        mock_result.hints = {"0": "Hint 1", "1": "Hint 2", "2": "Hint 3"}

        agent.dspy_module.return_value = mock_result

        # Test generation
        result = agent.generate(
            "Mathematics", "Calculus", difficulty_level="Undergraduate"
        )

        # Verify result
        assert result["subject"] == "Mathematics"
        assert result["topic"] == "Calculus"
        assert result["problem"] == "Test problem statement"
        assert result["answer"] == "Test solution"
        assert len(result["hints"]) >= 3
        assert result["difficulty_level"] == "Undergraduate"
        assert result["dspy_generated"] is True

    @patch("core.dspy.engineer_agent.EngineerAgent")
    @patch("core.dspy.engineer_agent.DSPyEngineerAgent.initialize_dspy_module")
    def test_fallback_generate(self, mock_initialize, mock_engineer_agent):
        """Test fallback to legacy implementation."""
        # Setup mocks
        mock_initialize.side_effect = DSPyIntegrationError("Test error")

        mock_legacy_agent = MagicMock()
        mock_legacy_result = {
            "subject": "Mathematics",
            "topic": "Calculus",
            "problem": "Legacy problem",
            "answer": "Legacy answer",
            "hints": {"0": "Legacy hint 1", "1": "Legacy hint 2", "2": "Legacy hint 3"},
        }
        mock_legacy_agent.generate.return_value = mock_legacy_result
        mock_engineer_agent.return_value = mock_legacy_agent

        # Create agent
        agent = DSPyEngineerAgent()

        # Test generation with fallback
        result = agent.generate("Mathematics", "Calculus")

        # Verify fallback was used
        mock_legacy_agent.generate.assert_called_once()
        assert result["problem"] == "Legacy problem"
        assert result["answer"] == "Legacy answer"

    def test_get_domain_signature(self):
        """Test getting domain signatures."""
        agent = DSPyEngineerAgent()

        # Test mathematics domain
        math_sig = agent._get_domain_signature("mathematics")
        assert "subject, topic, difficulty_level" in math_sig
        assert "problem_statement, solution, proof" in math_sig

        # Test other domains
        assert agent._get_domain_signature("physics") is None
        assert agent._get_domain_signature("computer science") is None

    def test_convert_dspy_result(self):
        """Test converting DSPy result to expected format."""
        agent = DSPyEngineerAgent()

        # Create mock DSPy result
        mock_result = MagicMock()
        mock_result.problem_statement = "Test problem statement"
        mock_result.solution = "Test solution"
        mock_result.proof = "Test proof"
        mock_result.reasoning_trace = "Step 1\n\nStep 2\n\nStep 3"

        # Test with hints in result
        mock_result.hints = {"0": "Hint 1", "1": "Hint 2", "2": "Hint 3"}
        result = agent._convert_dspy_result(
            mock_result, "Mathematics", "Calculus", "Undergraduate", "Test description"
        )
        assert result["subject"] == "Mathematics"
        assert result["topic"] == "Calculus"
        assert result["problem"] == "Test problem statement"
        assert result["answer"] == "Test solution"
        assert len(result["hints"]) >= 3
        assert result["hints"]["0"] == "Hint 1"

        # Test without hints (should create from reasoning trace)
        delattr(mock_result, "hints")
        result = agent._convert_dspy_result(
            mock_result, "Mathematics", "Calculus", "Undergraduate", "Test description"
        )
        assert len(result["hints"]) >= 3
        assert "Step 1" in result["hints"]["0"]


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
