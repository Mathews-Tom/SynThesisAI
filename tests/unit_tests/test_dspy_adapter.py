"""
Unit tests for DSPy adapter pattern.

These tests verify the adapter pattern for backward compatibility
with existing agent interfaces.
"""

# Standard Library
from unittest.mock import MagicMock, patch

# Third-Party Library
import pytest

# SynThesisAI Modules
from core.dspy.adapter import (
    AgentAdapter,
    CheckerAgentAdapter,
    EngineerAgentAdapter,
    TargetAgentAdapter,
    create_checker_agent,
    create_engineer_agent,
    create_target_agent,
)


class TestAgentAdapter:
    """Test base AgentAdapter class."""

    def test_initialization(self):
        """Test adapter initialization."""
        # Cannot instantiate abstract class directly
        with pytest.raises(TypeError):
            AgentAdapter("mathematics")


class TestEngineerAgentAdapter:
    """Test EngineerAgentAdapter class."""

    @patch("core.dspy.adapter.DSPyEngineerAgent")
    def test_initialization(self, mock_dspy_agent):
        """Test adapter initialization."""
        adapter = EngineerAgentAdapter("mathematics")
        assert adapter.domain == "mathematics"
        assert adapter.use_dspy is True

    @patch("core.dspy.adapter.DSPyEngineerAgent")
    def test_dspy_module_initialization(self, mock_dspy_agent):
        """Test DSPy module initialization."""
        mock_instance = MagicMock()
        mock_dspy_agent.return_value = mock_instance

        adapter = EngineerAgentAdapter("mathematics")
        module = adapter._get_dspy_module()

        assert module is mock_instance
        mock_dspy_agent.assert_called_once_with("mathematics")

    @patch("core.dspy.adapter.DSPyEngineerAgent")
    def test_dspy_module_initialization_error(self, mock_dspy_agent):
        """Test DSPy module initialization error handling."""
        mock_dspy_agent.side_effect = Exception("Test error")

        adapter = EngineerAgentAdapter("mathematics")
        module = adapter._get_dspy_module()

        assert module is None
        assert adapter.use_dspy is False

    def test_convert_to_dspy_format(self):
        """Test conversion to DSPy format."""
        adapter = EngineerAgentAdapter("mathematics")

        legacy_input = {
            "difficulty": "high",
            "objectives": ["understand algebra"],
            "topic": "quadratic equations",
            "extra_field": "value",
        }

        dspy_input = adapter._convert_to_dspy_format(legacy_input)

        assert dspy_input["difficulty_level"] == "high"
        assert dspy_input["learning_objectives"] == ["understand algebra"]
        assert dspy_input["topic"] == "quadratic equations"
        assert dspy_input["extra_field"] == "value"

    def test_convert_to_legacy_format(self):
        """Test conversion to legacy format."""
        adapter = EngineerAgentAdapter("mathematics")

        dspy_result = {
            "problem_statement": "Solve x^2 + 5x + 6 = 0",
            "solution": "x = -2 or x = -3",
            "reasoning_trace": "Factor the equation...",
            "difficulty_level": "high",
            "extra_field": "value",
        }

        legacy_result = adapter._convert_to_legacy_format(dspy_result)

        assert legacy_result["problem"] == "Solve x^2 + 5x + 6 = 0"
        assert legacy_result["solution"] == "x = -2 or x = -3"
        assert legacy_result["explanation"] == "Factor the equation..."
        assert legacy_result["difficulty"] == "high"
        assert legacy_result["extra_field"] == "value"

    @patch("core.dspy.adapter.DSPyEngineerAgent")
    @patch("core.dspy.adapter.get_dspy_config")
    def test_generate_problem_dspy(self, mock_config, mock_dspy_agent):
        """Test generate_problem using DSPy."""
        # Setup mocks
        mock_config.return_value.is_enabled.return_value = True
        mock_module = MagicMock()
        mock_module.generate.return_value = {
            "problem_statement": "Solve x^2 + 5x + 6 = 0",
            "solution": "x = -2 or x = -3",
        }
        mock_dspy_agent.return_value = mock_module

        # Create adapter and call method
        adapter = EngineerAgentAdapter("mathematics")
        result = adapter.generate_problem({"difficulty": "high"})

        # Verify result
        assert result["problem"] == "Solve x^2 + 5x + 6 = 0"
        assert result["solution"] == "x = -2 or x = -3"
        mock_module.generate.assert_called_once()

    @patch("core.dspy.adapter.DSPyEngineerAgent")
    @patch("core.dspy.adapter.get_dspy_config")
    def test_generate_problem_legacy_fallback(self, mock_config, mock_dspy_agent):
        """Test generate_problem with fallback to legacy."""
        # Setup mocks
        mock_config.return_value.is_enabled.return_value = True
        mock_dspy_agent.side_effect = Exception("Test error")

        # Mock legacy agent
        mock_legacy_agent = MagicMock()
        mock_legacy_agent.generate_problem.return_value = {
            "problem": "Legacy problem",
            "solution": "Legacy solution",
        }

        # Patch the _get_legacy_agent method
        with patch.object(
            EngineerAgentAdapter, "_get_legacy_agent", return_value=mock_legacy_agent
        ):
            # Create adapter and call method
            adapter = EngineerAgentAdapter("mathematics")
            result = adapter.generate_problem({"difficulty": "high"})

            # Verify result
            assert result["problem"] == "Legacy problem"
            assert result["solution"] == "Legacy solution"
            mock_legacy_agent.generate_problem.assert_called_once()


class TestCheckerAgentAdapter:
    """Test CheckerAgentAdapter class."""

    @patch("core.dspy.adapter.DSPyCheckerAgent")
    def test_initialization(self, mock_dspy_agent):
        """Test adapter initialization."""
        adapter = CheckerAgentAdapter("mathematics")
        assert adapter.domain == "mathematics"
        assert adapter.use_dspy is True

    @patch("core.dspy.adapter.DSPyCheckerAgent")
    def test_convert_to_dspy_format(self, mock_dspy_agent):
        """Test conversion to DSPy format."""
        adapter = CheckerAgentAdapter("mathematics")

        legacy_input = {
            "problem": "Solve x^2 + 5x + 6 = 0",
            "solution": "x = -2 or x = -3",
            "explanation": "Factor the equation...",
        }

        dspy_input = adapter._convert_to_dspy_format(legacy_input)

        assert dspy_input["problem_statement"] == "Solve x^2 + 5x + 6 = 0"
        assert dspy_input["solution"] == "x = -2 or x = -3"
        assert dspy_input["reasoning_trace"] == "Factor the equation..."

    @patch("core.dspy.adapter.DSPyCheckerAgent")
    def test_convert_to_legacy_format(self, mock_dspy_agent):
        """Test conversion to legacy format."""
        adapter = CheckerAgentAdapter("mathematics")

        dspy_result = {
            "is_valid": True,
            "feedback": "Good solution",
            "score": 0.95,
            "detailed_scores": {"accuracy": 0.9, "clarity": 1.0},
        }

        legacy_result = adapter._convert_to_legacy_format(dspy_result)

        assert legacy_result["is_valid"] is True
        assert legacy_result["feedback"] == "Good solution"
        assert legacy_result["score"] == 0.95
        assert legacy_result["detailed_scores"]["accuracy"] == 0.9


class TestTargetAgentAdapter:
    """Test TargetAgentAdapter class."""

    @patch("core.dspy.adapter.DSPyTargetAgent")
    def test_initialization(self, mock_dspy_agent):
        """Test adapter initialization."""
        adapter = TargetAgentAdapter("mathematics")
        assert adapter.domain == "mathematics"
        assert adapter.use_dspy is True

    @patch("core.dspy.adapter.DSPyTargetAgent")
    def test_convert_to_dspy_format_string(self, mock_dspy_agent):
        """Test conversion to DSPy format with string input."""
        adapter = TargetAgentAdapter("mathematics")

        legacy_input = "Solve x^2 + 5x + 6 = 0"
        dspy_input = adapter._convert_to_dspy_format(legacy_input)

        assert dspy_input["problem_statement"] == "Solve x^2 + 5x + 6 = 0"

    @patch("core.dspy.adapter.DSPyTargetAgent")
    def test_convert_to_dspy_format_dict(self, mock_dspy_agent):
        """Test conversion to DSPy format with dict input."""
        adapter = TargetAgentAdapter("mathematics")

        legacy_input = {"problem": "Solve x^2 + 5x + 6 = 0"}
        dspy_input = adapter._convert_to_dspy_format(legacy_input)

        assert dspy_input["problem_statement"] == "Solve x^2 + 5x + 6 = 0"

    @patch("core.dspy.adapter.DSPyTargetAgent")
    def test_convert_to_legacy_format(self, mock_dspy_agent):
        """Test conversion to legacy format."""
        adapter = TargetAgentAdapter("mathematics")

        dspy_result = {
            "solution": "x = -2 or x = -3",
            "reasoning_trace": "Factor the equation...",
        }

        legacy_result = adapter._convert_to_legacy_format(dspy_result)

        assert legacy_result["solution"] == "x = -2 or x = -3"
        assert legacy_result["explanation"] == "Factor the equation..."


class TestFactoryFunctions:
    """Test factory functions for creating adapters."""

    @patch("core.dspy.adapter.EngineerAgentAdapter")
    def test_create_engineer_agent(self, mock_adapter_class):
        """Test create_engineer_agent factory function."""
        mock_adapter = MagicMock()
        mock_adapter_class.return_value = mock_adapter

        adapter = create_engineer_agent("mathematics", use_dspy=True)

        assert adapter is mock_adapter
        mock_adapter_class.assert_called_once_with("mathematics", True)

    @patch("core.dspy.adapter.CheckerAgentAdapter")
    def test_create_checker_agent(self, mock_adapter_class):
        """Test create_checker_agent factory function."""
        mock_adapter = MagicMock()
        mock_adapter_class.return_value = mock_adapter

        adapter = create_checker_agent("mathematics", use_dspy=False)

        assert adapter is mock_adapter
        mock_adapter_class.assert_called_once_with("mathematics", False)

    @patch("core.dspy.adapter.TargetAgentAdapter")
    def test_create_target_agent(self, mock_adapter_class):
        """Test create_target_agent factory function."""
        mock_adapter = MagicMock()
        mock_adapter_class.return_value = mock_adapter

        adapter = create_target_agent("mathematics")

        assert adapter is mock_adapter
        mock_adapter_class.assert_called_once_with("mathematics", True)
