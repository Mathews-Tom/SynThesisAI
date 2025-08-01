"""
Unit tests for DSPy continuous learning system.

These tests verify the continuous learning functionality including
performance tracking, reoptimization decisions, and adaptive parameter tuning.
"""

# Standard Library
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

# Third-Party Library

# SynThesisAI Modules
from core.dspy.continuous_learning import (
    ContinuousLearningManager,
    LearningMetrics,
    get_continuous_learning_manager,
)
from core.dspy.feedback import (
    FeedbackSeverity,
    FeedbackSource,
    FeedbackType,
)


class TestLearningMetrics:
    """Test LearningMetrics functionality."""

    def test_learning_metrics_initialization(self):
        """Test learning metrics initialization."""
        metrics = LearningMetrics()

        assert metrics.optimization_count == 0
        assert metrics.performance_history == []
        assert metrics.improvement_rate == 0.0
        assert metrics.last_optimization is None
        assert metrics.best_performance == 0.0
        assert metrics.current_performance == 0.0

    def test_add_performance_record(self):
        """Test adding performance records."""
        metrics = LearningMetrics()

        # Add first record
        metrics.add_performance_record({"overall_score": 0.75, "accuracy": 0.8, "relevance": 0.7})

        assert len(metrics.performance_history) == 1
        assert metrics.current_performance == 0.75
        assert metrics.best_performance == 0.75
        assert metrics.optimization_count == 0  # Not incremented by add_performance_record

        # Add second record with better performance
        metrics.add_performance_record({"overall_score": 0.85, "accuracy": 0.9, "relevance": 0.8})

        assert len(metrics.performance_history) == 2
        assert metrics.current_performance == 0.85
        assert metrics.best_performance == 0.85

    def test_improvement_rate_calculation(self):
        """Test improvement rate calculation."""
        metrics = LearningMetrics()

        # Add records with improving performance
        for i in range(10):
            score = 0.5 + (i * 0.05)  # Gradually improving scores
            metrics.add_performance_record({"overall_score": score})

        # Should have positive improvement rate
        assert metrics.improvement_rate > 0

        # Add records with declining performance
        for i in range(5):
            score = 0.9 - (i * 0.1)  # Declining scores
            metrics.add_performance_record({"overall_score": score})

        # Should have negative improvement rate
        assert metrics.improvement_rate < 0

    def test_metrics_serialization(self):
        """Test metrics to/from dictionary conversion."""
        metrics = LearningMetrics()
        metrics.optimization_count = 5
        metrics.best_performance = 0.85
        metrics.current_performance = 0.80
        metrics.last_optimization = datetime.now()
        metrics.add_performance_record({"overall_score": 0.75})

        # Convert to dict
        metrics_dict = metrics.to_dict()

        assert metrics_dict["optimization_count"] == 5
        assert metrics_dict["best_performance"] == 0.85
        assert metrics_dict["current_performance"] == 0.75  # Updated by add_performance_record
        assert "last_optimization" in metrics_dict
        assert len(metrics_dict["performance_history"]) == 1

        # Convert back from dict
        restored_metrics = LearningMetrics.from_dict(metrics_dict)

        assert restored_metrics.optimization_count == 5
        assert restored_metrics.best_performance == 0.85
        assert restored_metrics.current_performance == 0.75  # Matches the serialized value
        assert len(restored_metrics.performance_history) == 1


class TestContinuousLearningManager:
    """Test ContinuousLearningManager functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.learning_manager = ContinuousLearningManager(learning_dir=self.temp_dir.name)

    def teardown_method(self):
        """Clean up test environment."""
        self.temp_dir.cleanup()

    def test_learning_manager_initialization(self):
        """Test learning manager initialization."""
        assert self.learning_manager.learning_dir.exists()
        assert isinstance(self.learning_manager.domain_metrics, dict)
        assert self.learning_manager.optimization_engine is not None
        assert self.learning_manager.feedback_integrator is not None
        assert self.learning_manager.quality_assessor is not None

    def test_should_reoptimize_initial(self):
        """Test reoptimization decision for never-optimized domain."""
        # Use a unique domain to avoid interference from other tests
        should_reopt, reason = self.learning_manager.should_reoptimize("test_initial_domain")

        assert should_reopt is True
        assert "Initial optimization required" in reason

    def test_should_reoptimize_time_based(self):
        """Test time-based reoptimization decision."""
        # Create metrics with old optimization
        metrics = LearningMetrics()
        metrics.optimization_count = 1
        metrics.last_optimization = datetime.now() - timedelta(days=10)  # 10 days ago
        metrics.current_performance = 0.8
        metrics.best_performance = 0.8

        self.learning_manager.domain_metrics["mathematics"] = metrics

        should_reopt, reason = self.learning_manager.should_reoptimize("mathematics")

        assert should_reopt is True
        assert "days ago" in reason

    def test_should_reoptimize_performance_degradation(self):
        """Test reoptimization decision based on performance degradation."""
        # Create metrics with performance degradation
        metrics = LearningMetrics()
        metrics.optimization_count = 3
        metrics.last_optimization = datetime.now() - timedelta(days=2)  # Recent
        metrics.current_performance = 0.7  # Current performance
        metrics.best_performance = 0.9  # Much better past performance (>10% degradation)

        self.learning_manager.domain_metrics["mathematics"] = metrics

        should_reopt, reason = self.learning_manager.should_reoptimize("mathematics")

        assert should_reopt is True
        assert "Performance degraded" in reason

    def test_should_reoptimize_feedback_based(self):
        """Test reoptimization decision based on high-priority feedback."""
        # Create metrics that wouldn't normally trigger reoptimization
        metrics = LearningMetrics()
        metrics.optimization_count = 2
        metrics.last_optimization = datetime.now() - timedelta(days=1)  # Very recent
        metrics.current_performance = 0.85
        metrics.best_performance = 0.85

        self.learning_manager.domain_metrics["mathematics"] = metrics

        # Add high-priority feedback
        feedback_manager = self.learning_manager.feedback_integrator.feedback_manager
        for i in range(6):  # Above threshold of 5
            feedback_manager.add_feedback(
                content=f"Critical issue {i}",
                feedback_type=FeedbackType.ACCURACY,
                source=FeedbackSource.SYSTEM,
                domain="mathematics",
                severity=FeedbackSeverity.HIGH,
            )

        should_reopt, reason = self.learning_manager.should_reoptimize("mathematics")

        assert should_reopt is True
        assert "high-priority feedback" in reason

    def test_should_reoptimize_stagnation(self):
        """Test reoptimization decision based on improvement stagnation."""
        # Create metrics with stagnant improvement
        metrics = LearningMetrics()
        metrics.optimization_count = 5
        metrics.last_optimization = datetime.now() - timedelta(days=2)
        metrics.current_performance = 0.8
        metrics.best_performance = 0.8

        # Add many records with minimal improvement
        for i in range(15):
            metrics.add_performance_record(
                {"overall_score": 0.8 + (i * 0.001)}
            )  # Very small improvements

        self.learning_manager.domain_metrics["mathematics"] = metrics

        should_reopt, reason = self.learning_manager.should_reoptimize("mathematics")

        assert should_reopt is True
        assert "stagnated" in reason

    def test_get_sample_inputs(self):
        """Test sample input generation for different domains."""
        # Test mathematics samples
        math_samples = self.learning_manager._get_sample_inputs("mathematics")
        assert len(math_samples) == 3
        assert all("topic" in sample for sample in math_samples)
        assert all("difficulty_level" in sample for sample in math_samples)
        assert all("learning_objectives" in sample for sample in math_samples)

        # Test science samples
        science_samples = self.learning_manager._get_sample_inputs("science")
        assert len(science_samples) == 3
        assert any("physics" in str(sample) for sample in science_samples)

        # Test unknown domain (should get default)
        unknown_samples = self.learning_manager._get_sample_inputs("unknown_domain")
        assert len(unknown_samples) == 1
        assert unknown_samples[0]["topic"] == "general"

    def test_apply_adaptive_tuning(self):
        """Test adaptive parameter tuning."""
        # Test new domain (low optimization count)
        metrics = LearningMetrics()
        metrics.optimization_count = 1
        self.learning_manager.domain_metrics["mathematics"] = metrics

        adaptive_params = self.learning_manager._apply_adaptive_tuning("mathematics")
        assert adaptive_params["init_temperature"] == 1.3  # Conservative
        assert adaptive_params["num_candidate_programs"] == 16

        # Test experienced domain (high optimization count)
        metrics.optimization_count = 10
        adaptive_params = self.learning_manager._apply_adaptive_tuning("mathematics")
        assert adaptive_params["init_temperature"] == 1.6  # More exploration
        assert adaptive_params["num_candidate_programs"] == 20

        # Test slow improvement (with good performance to avoid override)
        metrics.improvement_rate = 0.005  # Very slow
        metrics.current_performance = 0.8  # Good performance to avoid low performance override
        adaptive_params = self.learning_manager._apply_adaptive_tuning("mathematics")
        assert adaptive_params["max_labeled_demos"] == 20  # More training data
        assert adaptive_params["optuna_trials_num"] == 150  # More trials

        # Test low performance
        metrics.current_performance = 0.6  # Low performance
        adaptive_params = self.learning_manager._apply_adaptive_tuning("mathematics")
        assert adaptive_params["max_bootstrapped_demos"] == 6  # More bootstrapping
        assert adaptive_params["max_labeled_demos"] == 24  # More labeled examples

    def test_learning_metrics_persistence(self):
        """Test learning metrics persistence."""
        # Add metrics for a domain
        metrics = LearningMetrics()
        metrics.optimization_count = 3
        metrics.current_performance = 0.85
        metrics.add_performance_record({"overall_score": 0.8})

        self.learning_manager.domain_metrics["mathematics"] = metrics

        # Save metrics
        self.learning_manager._save_learning_metrics()

        # Verify file was created
        metrics_file = Path(self.temp_dir.name) / "learning_metrics.json"
        assert metrics_file.exists()

        # Clear metrics and reload
        self.learning_manager.domain_metrics.clear()
        self.learning_manager._load_learning_metrics()

        # Verify metrics were restored
        assert "mathematics" in self.learning_manager.domain_metrics
        restored_metrics = self.learning_manager.domain_metrics["mathematics"]
        assert restored_metrics.optimization_count == 3
        assert restored_metrics.current_performance == 0.8  # Updated by add_performance_record
        assert len(restored_metrics.performance_history) == 1

    def test_get_learning_summary_single_domain(self):
        """Test learning summary for single domain."""
        # Add metrics for a domain
        metrics = LearningMetrics()
        metrics.optimization_count = 2
        metrics.current_performance = 0.8
        metrics.best_performance = 0.85
        metrics.last_optimization = datetime.now()

        self.learning_manager.domain_metrics["mathematics"] = metrics

        summary = self.learning_manager.get_learning_summary("mathematics")

        assert summary["domain"] == "mathematics"
        assert "metrics" in summary
        assert "feedback_summary" in summary
        assert summary["metrics"]["optimization_count"] == 2
        assert summary["metrics"]["current_performance"] == 0.8

    def test_get_learning_summary_all_domains(self):
        """Test learning summary for all domains."""
        # Add metrics for multiple domains
        for domain in ["mathematics", "science", "technology"]:
            metrics = LearningMetrics()
            metrics.optimization_count = 1
            metrics.current_performance = 0.75
            self.learning_manager.domain_metrics[domain] = metrics

        summary = self.learning_manager.get_learning_summary()

        assert summary["total_domains"] == 3
        assert "domains" in summary
        assert "mathematics" in summary["domains"]
        assert "science" in summary["domains"]
        assert "technology" in summary["domains"]

        for domain_summary in summary["domains"].values():
            assert "optimization_count" in domain_summary
            assert "current_performance" in domain_summary
            assert "best_performance" in domain_summary

    def test_run_continuous_learning_cycle(self):
        """Test running a continuous learning cycle."""
        # Test with specific domains
        domains = ["mathematics", "science"]
        results = self.learning_manager.run_continuous_learning_cycle(domains)

        assert "cycle_start" in results
        assert "domains_processed" in results
        assert "domains_optimized" in results
        assert "total_improvements" in results
        assert "cycle_time" in results

        # Should process the specified domains
        assert len(results["domains_processed"]) <= len(domains)

        # Cycle time should be reasonable
        assert results["cycle_time"] > 0
        assert results["cycle_time"] < 60  # Should complete within a minute for test


class TestContinuousLearningGlobalFunctions:
    """Test global continuous learning functions."""

    def test_get_continuous_learning_manager(self):
        """Test getting global continuous learning manager."""
        manager1 = get_continuous_learning_manager()
        manager2 = get_continuous_learning_manager()

        assert manager1 is manager2  # Should be singleton
        assert isinstance(manager1, ContinuousLearningManager)


class TestContinuousLearningIntegration:
    """Test integration between continuous learning and other components."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.learning_manager = ContinuousLearningManager(learning_dir=self.temp_dir.name)

    def teardown_method(self):
        """Clean up test environment."""
        self.temp_dir.cleanup()

    def test_feedback_integration(self):
        """Test integration with feedback system."""
        # Add feedback that should trigger reoptimization
        feedback_manager = self.learning_manager.feedback_integrator.feedback_manager
        for i in range(7):  # Above threshold
            feedback_manager.add_feedback(
                content=f"Issue {i}",
                feedback_type=FeedbackType.ACCURACY,
                source=FeedbackSource.SYSTEM,
                domain="mathematics",
                severity=FeedbackSeverity.HIGH,
            )

        # Should trigger reoptimization
        should_reopt, reason = self.learning_manager.should_reoptimize("mathematics")
        assert should_reopt is True
        assert "high-priority feedback" in reason

        # Should generate optimization adjustments
        adjustments = self.learning_manager.feedback_integrator.integrate_feedback_for_optimization(
            "mathematics"
        )
        assert len(adjustments) > 0

    def test_quality_assessment_integration(self):
        """Test integration with quality assessment."""
        # The quality assessor should be available
        assert self.learning_manager.quality_assessor is not None

        # Should be able to get sample inputs for evaluation
        samples = self.learning_manager._get_sample_inputs("mathematics")
        assert len(samples) > 0

    def test_optimization_engine_integration(self):
        """Test integration with optimization engine."""
        # The optimization engine should be available
        assert self.learning_manager.optimization_engine is not None

        # Should be able to generate adaptive parameters
        adaptive_params = self.learning_manager._apply_adaptive_tuning("mathematics")
        assert isinstance(adaptive_params, dict)
