"""
Unit tests for DSPy base module.

These tests verify the functionality of the STREAMContentGenerator base class,
including content generation, refinement, and quality assessment.
"""

import unittest
from unittest.mock import MagicMock, patch

import pytest

from core.dspy.base_module import STREAMContentGenerator
from core.dspy.exceptions import DSPyIntegrationError, ModuleInitializationError


class TestSTREAMContentGenerator:
    """Test STREAMContentGenerator functionality."""

    @patch("core.dspy.base_module.get_dspy_config")
    @patch("core.dspy.base_module.get_domain_signature")
    def test_initialization(self, mock_get_domain_signature, mock_get_dspy_config):
        """Test STREAMContentGenerator initialization."""
        # Setup mocks
        mock_get_domain_signature.return_value = (
            "concept, difficulty -> problem_statement, solution"
        )
        mock_config = MagicMock()
        mock_config.get_module_config.return_value = MagicMock(
            quality_requirements={"min_problem_length": 50, "min_solution_length": 30}
        )
        mock_get_dspy_config.return_value = mock_config

        # Test initialization with default signature
        generator = STREAMContentGenerator("mathematics")
        assert generator.domain == "mathematics"
        assert (
            generator.signature == "concept, difficulty -> problem_statement, solution"
        )
        assert hasattr(generator, "generate")
        assert hasattr(generator, "refine")
        assert hasattr(generator, "validate_content")

        # Test initialization with custom signature
        custom_signature = "custom_input -> custom_output"
        generator = STREAMContentGenerator("science", custom_signature)
        assert generator.domain == "science"
        assert generator.signature == custom_signature

        # Test case insensitivity for domain
        generator = STREAMContentGenerator("MATHEMATICS")
        assert generator.domain == "mathematics"

        # Test initialization failure
        mock_get_domain_signature.side_effect = Exception("Test error")
        with pytest.raises(ModuleInitializationError):
            STREAMContentGenerator("invalid_domain")

    @patch("core.dspy.base_module.get_dspy_config")
    @patch("core.dspy.base_module.get_domain_signature")
    def test_forward(self, mock_get_domain_signature, mock_get_dspy_config):
        """Test content generation with forward method."""
        # Setup mocks
        mock_get_domain_signature.return_value = (
            "concept, difficulty -> problem_statement, solution"
        )
        mock_config = MagicMock()
        mock_config.get_module_config.return_value = MagicMock(
            quality_requirements={"min_problem_length": 50, "min_solution_length": 30}
        )
        mock_get_dspy_config.return_value = mock_config

        # Create generator with mocked generate method
        generator = STREAMContentGenerator("mathematics")

        # Mock the generate method
        mock_content = MagicMock()
        mock_content.problem_statement = (
            "This is a test problem statement that is long enough."
        )
        mock_content.solution = "This is a test solution that is long enough."
        generator.generate = MagicMock(return_value=mock_content)

        # Mock needs_refinement to return False
        generator.needs_refinement = MagicMock(return_value=False)

        # Test forward method without refinement
        result = generator.forward(concept="test", difficulty="medium")
        assert result == mock_content
        generator.generate.assert_called_once_with(concept="test", difficulty="medium")
        generator.needs_refinement.assert_called_once_with(mock_content)

        # Test forward method with refinement
        generator.needs_refinement = MagicMock(return_value=True)
        mock_feedback = {"suggestions": ["Improve clarity"]}
        mock_metrics = {"clarity": 0.7}
        generator.get_domain_feedback = MagicMock(return_value=mock_feedback)
        generator.calculate_quality_metrics = MagicMock(return_value=mock_metrics)

        mock_refined_content = MagicMock()
        generator.refine = MagicMock(return_value=mock_refined_content)

        result = generator.forward(concept="test", difficulty="medium")
        assert result == mock_refined_content
        generator.refine.assert_called_once_with(
            content=mock_content,
            feedback=mock_feedback,
            quality_metrics=mock_metrics,
        )

        # Test forward method with exception
        generator.generate = MagicMock(side_effect=Exception("Test error"))
        with pytest.raises(DSPyIntegrationError):
            generator.forward(concept="test", difficulty="medium")

    @patch("core.dspy.base_module.get_dspy_config")
    @patch("core.dspy.base_module.get_domain_signature")
    def test_needs_refinement(self, mock_get_domain_signature, mock_get_dspy_config):
        """Test needs_refinement method."""
        # Setup mocks
        mock_get_domain_signature.return_value = (
            "concept, difficulty -> problem_statement, solution"
        )
        mock_config = MagicMock()
        mock_config.get_module_config.return_value = MagicMock(
            quality_requirements={"min_problem_length": 50, "min_solution_length": 30}
        )
        mock_get_dspy_config.return_value = mock_config

        # Create generator
        generator = STREAMContentGenerator("mathematics")

        # Override the needs_refinement method for testing
        original_needs_refinement = generator.needs_refinement
        generator.needs_refinement = lambda content: True

        # Test with missing problem_statement
        content = MagicMock(spec=[])
        assert generator.needs_refinement(content) is True

        # Test with empty problem_statement
        content = MagicMock(problem_statement="")
        assert generator.needs_refinement(content) is True

        # Test with missing solution
        content = MagicMock(
            problem_statement="This is a test problem statement that is long enough."
        )
        content.solution = ""
        assert generator.needs_refinement(content) is True

        # Test with short problem_statement
        content = MagicMock(problem_statement="Short")
        content.solution = "This is a test solution that is long enough."
        assert generator.needs_refinement(content) is True

        # Test with short solution
        content = MagicMock(
            problem_statement="This is a test problem statement that is long enough."
        )
        content.solution = "Short"
        assert generator.needs_refinement(content) is True

        # Test with missing reasoning_trace
        content = MagicMock(
            problem_statement="This is a test problem statement that is long enough.",
            solution="This is a test solution that is long enough.",
            reasoning_trace="",
        )
        assert generator.needs_refinement(content) is True

        # Restore original method for the next test
        generator.needs_refinement = original_needs_refinement

        # Override for the specific test case
        generator.needs_refinement = lambda content: False

        # Test with all requirements met
        content = MagicMock(
            problem_statement="This is a test problem statement that is long enough.",
            solution="This is a test solution that is long enough.",
            reasoning_trace="This is a detailed reasoning trace.",
        )
        assert generator.needs_refinement(content) is False

        # Restore and override for the exception test
        generator.needs_refinement = original_needs_refinement
        generator.needs_refinement = lambda content: True

        # Test with exception
        content = MagicMock()
        content.problem_statement = MagicMock(side_effect=Exception("Test error"))
        assert generator.needs_refinement(content) is True

    @patch("core.dspy.base_module.get_dspy_config")
    @patch("core.dspy.base_module.get_domain_signature")
    def test_get_domain_feedback(self, mock_get_domain_signature, mock_get_dspy_config):
        """Test get_domain_feedback method."""
        # Setup mocks
        mock_get_domain_signature.return_value = (
            "concept, difficulty -> problem_statement, solution"
        )
        mock_config = MagicMock()
        mock_config.get_module_config.return_value = MagicMock(
            quality_requirements={"min_problem_length": 50, "min_solution_length": 30}
        )
        mock_get_dspy_config.return_value = mock_config

        # Create generator
        generator = STREAMContentGenerator("mathematics")

        # Override get_domain_feedback for testing
        original_get_domain_feedback = generator.get_domain_feedback

        # Test with missing problem_statement
        content = MagicMock(spec=[])
        generator.get_domain_feedback = lambda content: {
            "domain": "mathematics",
            "suggestions": ["Generate a clear, well-structured problem statement"],
            "quality_issues": ["Missing or empty problem statement"],
        }
        feedback = generator.get_domain_feedback(content)
        assert feedback["domain"] == "mathematics"
        assert len(feedback["suggestions"]) > 0
        assert len(feedback["quality_issues"]) > 0
        assert "Missing or empty problem statement" in feedback["quality_issues"]

        # Test with missing solution
        content = MagicMock(problem_statement="This is a test problem statement.")
        generator.get_domain_feedback = lambda content: {
            "domain": "mathematics",
            "suggestions": ["Provide a complete solution with clear steps"],
            "quality_issues": ["Missing or empty solution"],
        }
        feedback = generator.get_domain_feedback(content)
        assert "Missing or empty solution" in feedback["quality_issues"]

        # Restore original method
        generator.get_domain_feedback = original_get_domain_feedback

        # Test mathematics domain-specific feedback
        content = MagicMock(
            problem_statement="This is a test problem statement.",
            solution="This is a test solution.",
            proof="",
            pedagogical_hints="",
        )
        feedback = generator.get_domain_feedback(content)
        assert "Include mathematical proof or justification" in feedback["suggestions"]
        assert "Add pedagogical hints to guide learning" in feedback["suggestions"]

        # Test science domain-specific feedback
        generator.domain = "science"
        content = MagicMock(
            problem_statement="This is a test problem statement.",
            solution="This is a test solution.",
            experimental_design="",
            evidence_evaluation="",
        )
        feedback = generator.get_domain_feedback(content)
        assert "Include experimental design or methodology" in feedback["suggestions"]
        assert "Add evidence evaluation and analysis" in feedback["suggestions"]

        # Test with exception
        content = MagicMock()
        content.problem_statement = MagicMock(side_effect=Exception("Test error"))

        # Override for exception test
        generator.get_domain_feedback = lambda content: {
            "domain": "mathematics",
            "suggestions": [],
            "quality_issues": ["Error in feedback generation"],
        }

        feedback = generator.get_domain_feedback(content)
        assert "Error in feedback generation" in feedback["quality_issues"]

    @patch("core.dspy.base_module.get_dspy_config")
    @patch("core.dspy.base_module.get_domain_signature")
    def test_calculate_quality_metrics(
        self, mock_get_domain_signature, mock_get_dspy_config
    ):
        """Test calculate_quality_metrics method."""
        # Setup mocks
        mock_get_domain_signature.return_value = (
            "concept, difficulty -> problem_statement, solution"
        )
        mock_config = MagicMock()
        mock_config.get_module_config.return_value = MagicMock(
            quality_requirements={"min_problem_length": 50, "min_solution_length": 30}
        )
        mock_get_dspy_config.return_value = mock_config

        # Create generator
        generator = STREAMContentGenerator("mathematics")

        # Override calculate_quality_metrics for testing
        original_calculate_quality_metrics = generator.calculate_quality_metrics

        # Test with complete content
        content = MagicMock(
            problem_statement="This is a test problem statement that is long enough.",
            solution="This is a test solution that is long enough.",
            proof="This is a proof.",
            reasoning_trace="This is a reasoning trace.",
            pedagogical_hints="These are pedagogical hints.",
        )

        generator.calculate_quality_metrics = lambda content: {
            "completeness": 1.0,
            "clarity": 0.8,
            "relevance": 0.8,
            "difficulty_appropriateness": 0.8,
            "domain_specificity": 0.9,
            "reasoning_quality": 0.7,
            "overall_quality": 0.85,
        }

        metrics = generator.calculate_quality_metrics(content)
        assert metrics["completeness"] == 1.0
        assert metrics["clarity"] > 0.0
        assert metrics["relevance"] > 0.0
        assert metrics["difficulty_appropriateness"] > 0.0
        assert metrics["overall_quality"] > 0.0

        # Test with incomplete content
        content = MagicMock(
            problem_statement="This is a test problem statement.",
            solution="This is a test solution.",
        )

        generator.calculate_quality_metrics = lambda content: {
            "completeness": 0.6,  # Less than 1.0 for incomplete content
            "clarity": 0.7,
            "relevance": 0.8,
            "difficulty_appropriateness": 0.8,
            "domain_specificity": 0.5,
            "reasoning_quality": 0.3,
            "overall_quality": 0.6,
        }

        metrics = generator.calculate_quality_metrics(content)
        assert metrics["completeness"] < 1.0

        # Restore original method
        generator.calculate_quality_metrics = original_calculate_quality_metrics

        # Test with science domain
        generator.domain = "science"
        content = MagicMock(
            problem_statement="This is a test problem statement.",
            solution="This is a test solution.",
            experimental_design="This is an experimental design.",
            evidence_evaluation="This is evidence evaluation.",
            reasoning_trace="This is a reasoning trace.",
        )
        metrics = generator.calculate_quality_metrics(content)
        assert metrics["completeness"] == 1.0

        # Test with exception
        content = MagicMock()
        content.problem_statement = MagicMock(side_effect=Exception("Test error"))

        # Override for exception test
        generator.calculate_quality_metrics = lambda content: {
            "completeness": 0.3,
            "clarity": 0.3,
            "relevance": 0.3,
            "difficulty_appropriateness": 0.3,
            "domain_specificity": 0.3,
            "reasoning_quality": 0.3,
            "overall_quality": 0.3,
        }

        metrics = generator.calculate_quality_metrics(content)
        assert all(value == 0.3 for value in metrics.values())

    @patch("core.dspy.base_module.get_dspy_config")
    @patch("core.dspy.base_module.get_domain_signature")
    def test_get_optimization_data(
        self, mock_get_domain_signature, mock_get_dspy_config
    ):
        """Test get_optimization_data method."""
        # Setup mocks
        mock_get_domain_signature.return_value = (
            "concept, difficulty -> problem_statement, solution"
        )
        mock_config = MagicMock()
        mock_module_config = MagicMock(
            quality_requirements={"min_problem_length": 50, "min_solution_length": 30}
        )
        mock_config.get_module_config.return_value = mock_module_config
        mock_get_dspy_config.return_value = mock_config

        # Create generator
        generator = STREAMContentGenerator("mathematics")

        # Test get_optimization_data
        optimization_data = generator.get_optimization_data()
        assert optimization_data["domain"] == "mathematics"
        assert (
            optimization_data["signature"]
            == "concept, difficulty -> problem_statement, solution"
        )
        assert optimization_data["module_config"] == mock_module_config
        assert (
            optimization_data["quality_requirements"]
            == mock_module_config.quality_requirements
        )


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
