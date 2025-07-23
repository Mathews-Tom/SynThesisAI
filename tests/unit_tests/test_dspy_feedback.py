"""
Unit tests for DSPy feedback integration.

These tests verify the feedback collection, management, and integration
functionality for continuous improvement of DSPy modules.
"""

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from core.dspy.base_module import STREAMContentGenerator
from core.dspy.feedback import (
    DSPyFeedbackIntegrator,
    Feedback,
    FeedbackManager,
    FeedbackSeverity,
    FeedbackSource,
    FeedbackType,
    get_feedback_integrator,
    get_feedback_manager,
)


class TestFeedback:
    """Test Feedback class functionality."""

    def test_feedback_initialization(self):
        """Test feedback initialization."""
        feedback = Feedback(
            content="Test feedback content",
            feedback_type=FeedbackType.ACCURACY,
            source=FeedbackSource.SYSTEM,
            domain="mathematics",
            severity=FeedbackSeverity.HIGH,
        )

        assert feedback.content == "Test feedback content"
        assert feedback.feedback_type == FeedbackType.ACCURACY
        assert feedback.source == FeedbackSource.SYSTEM
        assert feedback.domain == "mathematics"
        assert feedback.severity == FeedbackSeverity.HIGH
        assert feedback.feedback_id is not None
        assert isinstance(feedback.timestamp, datetime)

    def test_feedback_with_string_enums(self):
        """Test feedback initialization with string enum values."""
        feedback = Feedback(
            content="Test feedback",
            feedback_type="accuracy",
            source="system",
            domain="mathematics",
            severity="high",
        )

        assert feedback.feedback_type == FeedbackType.ACCURACY
        assert feedback.source == FeedbackSource.SYSTEM
        assert feedback.severity == FeedbackSeverity.HIGH

    def test_feedback_to_dict(self):
        """Test feedback dictionary conversion."""
        feedback = Feedback(
            content="Test feedback",
            feedback_type=FeedbackType.ACCURACY,
            source=FeedbackSource.SYSTEM,
            domain="mathematics",
        )

        feedback_dict = feedback.to_dict()

        assert feedback_dict["content"] == "Test feedback"
        assert feedback_dict["feedback_type"] == "accuracy"
        assert feedback_dict["source"] == "system"
        assert feedback_dict["domain"] == "mathematics"
        assert feedback_dict["severity"] == "medium"
        assert "feedback_id" in feedback_dict
        assert "timestamp" in feedback_dict

    def test_feedback_from_dict(self):
        """Test feedback creation from dictionary."""
        feedback_data = {
            "content": "Test feedback",
            "feedback_type": "accuracy",
            "source": "system",
            "domain": "mathematics",
            "severity": "high",
            "timestamp": "2024-01-01T12:00:00",
            "feedback_id": "test_id_123",
        }

        feedback = Feedback.from_dict(feedback_data)

        assert feedback.content == "Test feedback"
        assert feedback.feedback_type == FeedbackType.ACCURACY
        assert feedback.source == FeedbackSource.SYSTEM
        assert feedback.domain == "mathematics"
        assert feedback.severity == FeedbackSeverity.HIGH
        assert feedback.feedback_id == "test_id_123"


class TestFeedbackManager:
    """Test FeedbackManager functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.feedback_manager = FeedbackManager(feedback_dir=self.temp_dir.name)

    def teardown_method(self):
        """Clean up test environment."""
        self.temp_dir.cleanup()

    def test_feedback_manager_initialization(self):
        """Test feedback manager initialization."""
        assert self.feedback_manager.feedback_dir.exists()
        assert isinstance(self.feedback_manager.feedback_cache, dict)

    def test_add_feedback(self):
        """Test adding feedback."""
        feedback = self.feedback_manager.add_feedback(
            content="Test accuracy feedback",
            feedback_type=FeedbackType.ACCURACY,
            source=FeedbackSource.SYSTEM,
            domain="mathematics",
            severity=FeedbackSeverity.HIGH,
        )

        assert feedback.content == "Test accuracy feedback"
        assert feedback.feedback_type == FeedbackType.ACCURACY
        assert feedback.domain == "mathematics"
        assert "mathematics" in self.feedback_manager.feedback_cache
        assert len(self.feedback_manager.feedback_cache["mathematics"]) == 1

    def test_get_feedback(self):
        """Test retrieving feedback."""
        # Add multiple feedback items
        self.feedback_manager.add_feedback(
            content="Accuracy issue",
            feedback_type=FeedbackType.ACCURACY,
            source=FeedbackSource.SYSTEM,
            domain="mathematics",
            severity=FeedbackSeverity.HIGH,
        )
        self.feedback_manager.add_feedback(
            content="Relevance issue",
            feedback_type=FeedbackType.RELEVANCE,
            source=FeedbackSource.USER,
            domain="mathematics",
            severity=FeedbackSeverity.MEDIUM,
        )

        # Get all feedback
        all_feedback = self.feedback_manager.get_feedback("mathematics")
        assert len(all_feedback) == 2

        # Filter by type
        accuracy_feedback = self.feedback_manager.get_feedback(
            "mathematics", feedback_type=FeedbackType.ACCURACY
        )
        assert len(accuracy_feedback) == 1
        assert accuracy_feedback[0].feedback_type == FeedbackType.ACCURACY

        # Filter by source
        system_feedback = self.feedback_manager.get_feedback(
            "mathematics", source=FeedbackSource.SYSTEM
        )
        assert len(system_feedback) == 1
        assert system_feedback[0].source == FeedbackSource.SYSTEM

        # Filter by severity
        high_severity_feedback = self.feedback_manager.get_feedback(
            "mathematics", min_severity=FeedbackSeverity.HIGH
        )
        assert len(high_severity_feedback) == 1
        assert high_severity_feedback[0].severity == FeedbackSeverity.HIGH

    def test_feedback_persistence(self):
        """Test feedback persistence to file."""
        # Add feedback
        self.feedback_manager.add_feedback(
            content="Persistent feedback",
            feedback_type=FeedbackType.ACCURACY,
            source=FeedbackSource.SYSTEM,
            domain="mathematics",
        )

        # Check file was created
        feedback_file = Path(self.temp_dir.name) / "mathematics_feedback.json"
        assert feedback_file.exists()

        # Verify file content
        feedback_data = json.loads(feedback_file.read_text(encoding="utf-8"))
        assert feedback_data["domain"] == "mathematics"
        assert len(feedback_data["feedback"]) == 1
        assert feedback_data["feedback"][0]["content"] == "Persistent feedback"

    def test_feedback_loading(self):
        """Test loading feedback from file."""
        # Add feedback and clear cache
        self.feedback_manager.add_feedback(
            content="Loadable feedback",
            feedback_type=FeedbackType.COHERENCE,
            source=FeedbackSource.EXPERT,
            domain="science",
        )
        self.feedback_manager.feedback_cache.clear()

        # Load feedback
        loaded_feedback = self.feedback_manager.get_feedback("science")
        assert len(loaded_feedback) == 1
        assert loaded_feedback[0].content == "Loadable feedback"
        assert loaded_feedback[0].feedback_type == FeedbackType.COHERENCE

    def test_get_feedback_summary(self):
        """Test feedback summary generation."""
        # Add various feedback items
        self.feedback_manager.add_feedback(
            content="Accuracy issue 1",
            feedback_type=FeedbackType.ACCURACY,
            source=FeedbackSource.SYSTEM,
            domain="mathematics",
            severity=FeedbackSeverity.HIGH,
        )
        self.feedback_manager.add_feedback(
            content="Accuracy issue 2",
            feedback_type=FeedbackType.ACCURACY,
            source=FeedbackSource.USER,
            domain="mathematics",
            severity=FeedbackSeverity.MEDIUM,
        )
        self.feedback_manager.add_feedback(
            content="Relevance issue",
            feedback_type=FeedbackType.RELEVANCE,
            source=FeedbackSource.SYSTEM,
            domain="mathematics",
            severity=FeedbackSeverity.LOW,
        )

        summary = self.feedback_manager.get_feedback_summary("mathematics")

        assert summary["domain"] == "mathematics"
        assert summary["total_feedback"] == 3
        assert summary["by_type"]["accuracy"] == 2
        assert summary["by_type"]["relevance"] == 1
        assert summary["by_source"]["system"] == 2
        assert summary["by_source"]["user"] == 1
        assert summary["by_severity"]["high"] == 1
        assert summary["by_severity"]["medium"] == 1
        assert summary["by_severity"]["low"] == 1
        assert len(summary["recent_feedback"]) == 3


class TestDSPyFeedbackIntegrator:
    """Test DSPyFeedbackIntegrator functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.integrator = DSPyFeedbackIntegrator()
        self.integrator.feedback_manager = FeedbackManager(
            feedback_dir=self.temp_dir.name
        )

    def teardown_method(self):
        """Clean up test environment."""
        self.temp_dir.cleanup()

    def test_collect_validation_feedback(self):
        """Test collecting feedback from validation results."""
        domain_module = STREAMContentGenerator("mathematics")
        validation_results = {
            "accuracy_score": 0.6,  # Low accuracy
            "relevance_score": 0.75,  # Moderate relevance
            "coherence_score": 0.65,  # Low coherence
            "pedagogical_score": 0.85,  # Good pedagogical value
        }

        feedback_items = self.integrator.collect_validation_feedback(
            domain_module, validation_results
        )

        # Should generate feedback for low accuracy and coherence
        assert len(feedback_items) >= 2

        # Check accuracy feedback
        accuracy_feedback = [
            f for f in feedback_items if f.feedback_type == FeedbackType.ACCURACY
        ]
        assert len(accuracy_feedback) == 1
        assert "Low accuracy score" in accuracy_feedback[0].content

        # Check coherence feedback
        coherence_feedback = [
            f for f in feedback_items if f.feedback_type == FeedbackType.COHERENCE
        ]
        assert len(coherence_feedback) == 1
        assert "Low coherence score" in coherence_feedback[0].content

    def test_integrate_feedback_for_optimization(self):
        """Test integrating feedback into optimization parameters."""
        # Add feedback items
        self.integrator.feedback_manager.add_feedback(
            content="Accuracy needs improvement",
            feedback_type=FeedbackType.ACCURACY,
            source=FeedbackSource.SYSTEM,
            domain="mathematics",
            severity=FeedbackSeverity.HIGH,
        )
        self.integrator.feedback_manager.add_feedback(
            content="Relevance could be better",
            feedback_type=FeedbackType.RELEVANCE,
            source=FeedbackSource.USER,
            domain="mathematics",
            severity=FeedbackSeverity.MEDIUM,
        )

        adjustments = self.integrator.integrate_feedback_for_optimization("mathematics")

        assert "quality_requirements" in adjustments
        assert "optimization_params" in adjustments
        assert "training_focus" in adjustments

        # Should have quality requirement adjustments
        quality_reqs = adjustments["quality_requirements"]
        assert "min_accuracy" in quality_reqs or "min_relevance" in quality_reqs

    def test_analyze_feedback_patterns(self):
        """Test feedback pattern analysis."""
        feedback_items = [
            Feedback(
                "Accuracy issue 1",
                FeedbackType.ACCURACY,
                FeedbackSource.SYSTEM,
                "math",
                FeedbackSeverity.HIGH,
            ),
            Feedback(
                "Accuracy issue 2",
                FeedbackType.ACCURACY,
                FeedbackSource.USER,
                "math",
                FeedbackSeverity.MEDIUM,
            ),
            Feedback(
                "Relevance issue",
                FeedbackType.RELEVANCE,
                FeedbackSource.SYSTEM,
                "math",
                FeedbackSeverity.LOW,
            ),
        ]

        analysis = self.integrator._analyze_feedback_patterns(feedback_items)

        assert analysis["total_feedback"] == 3
        assert analysis["type_distribution"]["accuracy"] == 2
        assert analysis["type_distribution"]["relevance"] == 1
        assert analysis["severity_distribution"]["high"] == 1
        assert analysis["severity_distribution"]["medium"] == 1
        assert analysis["severity_distribution"]["low"] == 1

        # Should identify accuracy as common issue (>30% threshold)
        common_issues = analysis["common_issues"]
        accuracy_issue = next(
            (issue for issue in common_issues if issue["type"] == "accuracy"), None
        )
        assert accuracy_issue is not None
        assert accuracy_issue["frequency"] == 2

    def test_generate_optimization_adjustments(self):
        """Test optimization adjustment generation."""
        feedback_analysis = {
            "total_feedback": 5,
            "common_issues": [
                {"type": "accuracy", "frequency": 3, "percentage": 60},
                {"type": "coherence", "frequency": 2, "percentage": 40},
            ],
            "improvement_areas": ["high_priority_issues"],
        }

        adjustments = self.integrator._generate_optimization_adjustments(
            feedback_analysis
        )

        assert "quality_requirements" in adjustments
        assert "optimization_params" in adjustments
        assert "training_focus" in adjustments

        # Should adjust quality requirements for common issues
        quality_reqs = adjustments["quality_requirements"]
        assert "min_accuracy" in quality_reqs
        assert "min_coherence" in quality_reqs

        # Should adjust optimization params for high-priority issues
        opt_params = adjustments["optimization_params"]
        assert "max_labeled_demos" in opt_params
        assert "num_candidate_programs" in opt_params

        # Should include training focus areas
        training_focus = adjustments["training_focus"]
        assert "factual_correctness" in training_focus
        assert "logical_structure" in training_focus


class TestFeedbackGlobalFunctions:
    """Test global feedback functions."""

    def test_get_feedback_integrator(self):
        """Test getting global feedback integrator."""
        integrator1 = get_feedback_integrator()
        integrator2 = get_feedback_integrator()

        assert integrator1 is integrator2  # Should be singleton
        assert isinstance(integrator1, DSPyFeedbackIntegrator)

    def test_get_feedback_manager(self):
        """Test getting global feedback manager."""
        manager1 = get_feedback_manager()
        manager2 = get_feedback_manager()

        assert manager1 is manager2  # Should be same instance
        assert isinstance(manager1, FeedbackManager)


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
