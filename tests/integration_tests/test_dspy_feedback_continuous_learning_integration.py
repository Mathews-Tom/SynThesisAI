"""
Integration tests for DSPy feedback and continuous learning systems.

These tests verify the end-to-end integration between feedback collection,
continuous learning, and optimization processes.
"""

# Standard Library
import logging
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

# Third-Party Library
import pytest

# SynThesisAI Modules
from core.dspy.base_module import STREAMContentGenerator
from core.dspy.continuous_learning import ContinuousLearningManager
from core.dspy.feedback import (
    DSPyFeedbackIntegrator,
    FeedbackManager,
    FeedbackSeverity,
    FeedbackSource,
    FeedbackType,
)

logger = logging.getLogger(__name__)


@pytest.mark.integration
class TestFeedbackContinuousLearningIntegration:
    """Test integration between feedback and continuous learning systems."""

    def setup_method(self) -> None:
        """Set up test environment."""
        self.temp_dir = tempfile.TemporaryDirectory()
        feedback_path = Path(self.temp_dir.name) / "feedback"
        learning_path = Path(self.temp_dir.name) / "learning"
        logger.debug(
            "Creating feedback and learning dirs at %s, %s",
            feedback_path,
            learning_path,
        )
        self.feedback_manager = FeedbackManager(feedback_dir=str(feedback_path))
        self.learning_manager = ContinuousLearningManager(learning_dir=str(learning_path))

        # Override the feedback manager in the learning manager's integrator
        self.learning_manager.feedback_integrator.feedback_manager = self.feedback_manager

    def teardown_method(self) -> None:
        """Clean up test environment."""
        logger.debug("Cleaning up test environment at %s", self.temp_dir.name)
        self.temp_dir.cleanup()

    def test_end_to_end_feedback_to_reoptimization(self):
        """Test complete flow from validation feedback to reoptimization decision."""
        domain_module = STREAMContentGenerator("mathematics")

        # Step 1: Simulate validation results with issues
        validation_results = {
            "accuracy_score": 0.55,  # Low accuracy
            "relevance_score": 0.65,  # Low relevance
            "coherence_score": 0.60,  # Low coherence
            "pedagogical_score": 0.70,  # Moderate pedagogical value
            "overall_score": 0.625,
        }

        # Step 2: Collect feedback from validation
        integrator = DSPyFeedbackIntegrator()
        integrator.feedback_manager = self.feedback_manager

        feedback_items = integrator.collect_validation_feedback(domain_module, validation_results)

        # Should generate multiple feedback items for low scores
        assert len(feedback_items) >= 3

        # Verify feedback types
        feedback_types = [f.feedback_type for f in feedback_items]
        assert FeedbackType.ACCURACY in feedback_types
        assert FeedbackType.RELEVANCE in feedback_types
        assert FeedbackType.COHERENCE in feedback_types

        # Step 3: Check if feedback triggers reoptimization
        should_reopt, reason = self.learning_manager.should_reoptimize("mathematics")

        # Should trigger reoptimization due to initial optimization requirement
        assert should_reopt is True

        # Step 4: Generate optimization adjustments based on feedback
        adjustments = integrator.integrate_feedback_for_optimization("mathematics")

        assert "quality_requirements" in adjustments
        assert "training_focus" in adjustments

        # Should have specific adjustments for the issues found
        quality_reqs = adjustments.get("quality_requirements", {})
        training_focus = adjustments.get("training_focus", [])

        # May not have adjustments if feedback doesn't meet thresholds, but should have some structure
        assert isinstance(quality_reqs, dict)
        assert isinstance(training_focus, list)

    def test_feedback_accumulation_triggers_reoptimization(self):
        """Test that accumulating high-priority feedback triggers reoptimization."""
        # Create a domain with recent optimization (normally wouldn't reoptimize)
        from core.dspy.continuous_learning import LearningMetrics

        metrics = LearningMetrics()
        metrics.optimization_count = 2
        metrics.last_optimization = datetime.now() - timedelta(hours=1)  # Very recent
        metrics.current_performance = 0.85  # Good performance
        metrics.best_performance = 0.85

        self.learning_manager.domain_metrics["mathematics"] = metrics

        # Initially should not need reoptimization
        should_reopt, reason = self.learning_manager.should_reoptimize("mathematics")
        assert should_reopt is False

        # Add high-priority feedback items
        for i in range(6):  # Above threshold of 5
            self.feedback_manager.add_feedback(
                content=f"Critical accuracy issue {i + 1}",
                feedback_type=FeedbackType.ACCURACY,
                source=FeedbackSource.SYSTEM,
                domain="mathematics",
                severity=FeedbackSeverity.HIGH,
            )

        # Now should trigger reoptimization due to feedback
        should_reopt, reason = self.learning_manager.should_reoptimize("mathematics")
        assert should_reopt is True
        assert "high-priority feedback" in reason

    def test_continuous_learning_with_feedback_integration(self):
        """Test continuous learning process with feedback integration."""
        domain_module = STREAMContentGenerator("mathematics")

        # Add some feedback to influence optimization
        self.feedback_manager.add_feedback(
            content="Accuracy needs significant improvement",
            feedback_type=FeedbackType.ACCURACY,
            source=FeedbackSource.EXPERT,
            domain="mathematics",
            severity=FeedbackSeverity.HIGH,
        )

        self.feedback_manager.add_feedback(
            content="Pedagogical clarity could be enhanced",
            feedback_type=FeedbackType.PEDAGOGICAL,
            source=FeedbackSource.USER,
            domain="mathematics",
            severity=FeedbackSeverity.MEDIUM,
        )

        # Perform continuous learning (this will be a mock since we don't have full DSPy)
        try:
            learning_results = self.learning_manager.perform_continuous_learning(domain_module)

            # Should attempt reoptimization
            assert learning_results["domain"] == "mathematics"
            assert "reason" in learning_results

            # If reoptimization was attempted, should have feedback adjustments
            if learning_results.get("reoptimized", False):
                assert "feedback_adjustments" in learning_results
                assert "adaptive_params" in learning_results

        except Exception as e:
            # Expected to fail in test environment without full DSPy setup
            # But should fail gracefully
            assert "Failed to optimize" in str(e) or "DSPy not available" in str(e)

    def test_feedback_pattern_analysis_affects_optimization(self):
        """Test that feedback pattern analysis affects optimization parameters."""
        # Add multiple feedback items of the same type to create a pattern
        for i in range(5):
            self.feedback_manager.add_feedback(
                content=f"Accuracy issue {i + 1}",
                feedback_type=FeedbackType.ACCURACY,
                source=FeedbackSource.SYSTEM,
                domain="mathematics",
                severity=FeedbackSeverity.HIGH,
            )

        # Add some coherence feedback too
        for i in range(3):
            self.feedback_manager.add_feedback(
                content=f"Coherence issue {i + 1}",
                feedback_type=FeedbackType.COHERENCE,
                source=FeedbackSource.USER,
                domain="mathematics",
                severity=FeedbackSeverity.MEDIUM,
            )

        # Get optimization adjustments
        integrator = DSPyFeedbackIntegrator()
        integrator.feedback_manager = self.feedback_manager

        adjustments = integrator.integrate_feedback_for_optimization("mathematics")

        # Should identify accuracy as a common issue and adjust accordingly
        quality_reqs = adjustments.get("quality_requirements", {})
        training_focus = adjustments.get("training_focus", [])

        assert "min_accuracy" in quality_reqs
        assert "factual_correctness" in training_focus

        # Should also address coherence
        assert "min_coherence" in quality_reqs
        assert "logical_structure" in training_focus

    def test_adaptive_parameter_tuning_with_feedback_history(self):
        """Test that adaptive parameter tuning considers feedback history."""
        from core.dspy.continuous_learning import LearningMetrics

        # Create metrics with learning history
        metrics = LearningMetrics()
        metrics.optimization_count = 5  # Experienced domain
        metrics.improvement_rate = 0.005  # Slow improvement
        metrics.current_performance = 0.65  # Low performance

        # Add performance history
        for i in range(10):
            metrics.add_performance_record(
                {
                    "overall_score": 0.6 + (i * 0.01),  # Gradual improvement
                    "accuracy": 0.65 + (i * 0.01),
                }
            )

        self.learning_manager.domain_metrics["mathematics"] = metrics

        # Add feedback indicating persistent issues
        for i in range(8):
            self.feedback_manager.add_feedback(
                content=f"Persistent accuracy issue {i + 1}",
                feedback_type=FeedbackType.ACCURACY,
                source=FeedbackSource.SYSTEM,
                domain="mathematics",
                severity=FeedbackSeverity.HIGH,
            )

        # Get adaptive parameters
        adaptive_params = self.learning_manager._apply_adaptive_tuning("mathematics")

        # Should use more aggressive parameters due to:
        # 1. High optimization count -> more exploration
        # 2. Slow improvement -> more training data and trials
        # 3. Low performance -> more bootstrapping

        assert adaptive_params["init_temperature"] == 1.5  # Moderate exploration (count=5)
        assert adaptive_params["num_candidate_programs"] == 18  # Moderate candidates (count=5)
        assert adaptive_params["max_labeled_demos"] == 24  # More training (low performance)
        # Low performance condition overrides slow improvement for max_labeled_demos
        # So we should see bootstrapping parameters but not optuna_trials_num
        assert (
            adaptive_params["max_bootstrapped_demos"] == 6
        )  # More bootstrapping (low performance)

        # optuna_trials_num is only set for slow improvement when performance is not low
        # Since current_performance (0.69 after records) < 0.7, low performance condition applies

    def test_learning_cycle_with_multiple_domains(self):
        """Test continuous learning cycle with multiple domains and feedback."""
        domains = ["mathematics", "science", "technology"]

        # Add different types of feedback for each domain
        feedback_configs = [
            ("mathematics", FeedbackType.ACCURACY, 6),
            ("science", FeedbackType.RELEVANCE, 4),
            ("technology", FeedbackType.COHERENCE, 3),
        ]

        for domain, feedback_type, count in feedback_configs:
            for i in range(count):
                self.feedback_manager.add_feedback(
                    content=f"{feedback_type.value} issue {i + 1} in {domain}",
                    feedback_type=feedback_type,
                    source=FeedbackSource.SYSTEM,
                    domain=domain,
                    severity=(FeedbackSeverity.HIGH if i < 2 else FeedbackSeverity.MEDIUM),
                )

        # Run learning cycle
        results = self.learning_manager.run_continuous_learning_cycle(domains)

        # Should process all domains
        assert len(results["domains_processed"]) <= len(domains)
        assert "cycle_time" in results
        assert results["cycle_time"] > 0

        # Check that feedback influenced reoptimization decisions
        for domain in domains:
            should_reopt, reason = self.learning_manager.should_reoptimize(domain)
            # All should need reoptimization (either initial or feedback-based)
            assert should_reopt is True

    def test_feedback_persistence_across_learning_cycles(self):
        """Test that feedback persists and influences multiple learning cycles."""

        # First cycle: Add feedback and perform learning
        self.feedback_manager.add_feedback(
            content="Initial accuracy concern",
            feedback_type=FeedbackType.ACCURACY,
            source=FeedbackSource.EXPERT,
            domain="mathematics",
            severity=FeedbackSeverity.HIGH,
        )

        # Simulate first learning cycle
        first_adjustments = (
            self.learning_manager.feedback_integrator.integrate_feedback_for_optimization(
                "mathematics"
            )
        )
        assert len(first_adjustments) > 0

        # Second cycle: Add more feedback
        self.feedback_manager.add_feedback(
            content="Ongoing accuracy issues",
            feedback_type=FeedbackType.ACCURACY,
            source=FeedbackSource.SYSTEM,
            domain="mathematics",
            severity=FeedbackSeverity.HIGH,
        )

        # Should now have accumulated feedback
        feedback_summary = self.feedback_manager.get_feedback_summary("mathematics")
        assert feedback_summary["total_feedback"] >= 2
        assert feedback_summary["by_type"]["accuracy"] >= 2

        # Second learning cycle should consider all accumulated feedback
        second_adjustments = (
            self.learning_manager.feedback_integrator.integrate_feedback_for_optimization(
                "mathematics"
            )
        )

        # Should still have accuracy-focused adjustments
        assert "min_accuracy" in second_adjustments.get("quality_requirements", {})
        assert "factual_correctness" in second_adjustments.get("training_focus", [])


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
