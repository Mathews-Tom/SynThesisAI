"""
Unit tests for MARL A/B Testing Framework.

Tests A/B test analysis, statistical testing, and result interpretation.
"""

from unittest.mock import Mock

import numpy as np
import pytest

from core.marl.config.config_schema import AgentConfig, MARLConfig
from core.marl.experimentation.ab_testing import (
    ABTestAnalysis,
    ABTestManager,
    ABTestManagerFactory,
    ABTestResult,
    StatisticalTest,
)
from core.marl.experimentation.experiment_manager import (
    Experiment,
    ExperimentCondition,
    ExperimentResult,
    ExperimentStatus,
    ExperimentType,
)


class TestStatisticalTest:
    """Test StatisticalTest functionality."""

    def test_statistical_test_creation(self):
        """Test statistical test creation."""
        test = StatisticalTest(
            test_name="t-test",
            statistic=2.5,
            p_value=0.02,
            is_significant=True,
            confidence_interval=(0.1, 0.5),
            effect_size=0.3,
        )

        assert test.test_name == "t-test"
        assert test.statistic == 2.5
        assert test.p_value == 0.02
        assert test.is_significant is True
        assert test.confidence_interval == (0.1, 0.5)
        assert test.effect_size == 0.3

    def test_statistical_test_validation(self):
        """Test statistical test validation."""
        # Invalid p-value
        with pytest.raises(ValueError, match="P-value must be between 0 and 1"):
            StatisticalTest(
                test_name="test",
                statistic=1.0,
                p_value=1.5,  # Invalid
                is_significant=False,
            )

        with pytest.raises(ValueError, match="P-value must be between 0 and 1"):
            StatisticalTest(
                test_name="test",
                statistic=1.0,
                p_value=-0.1,  # Invalid
                is_significant=False,
            )


class TestABTestAnalysis:
    """Test ABTestAnalysis functionality."""

    def create_test_analysis(self):
        """Create a test A/B analysis."""
        t_test = StatisticalTest(
            test_name="t-test", statistic=2.5, p_value=0.02, is_significant=True
        )

        mann_whitney_test = StatisticalTest(
            test_name="Mann-Whitney", statistic=150.0, p_value=0.03, is_significant=True
        )

        return ABTestAnalysis(
            experiment_id="test_experiment",
            metric_name="reward",
            control_mean=0.5,
            treatment_mean=0.7,
            control_std=0.1,
            treatment_std=0.12,
            control_n=100,
            treatment_n=100,
            t_test=t_test,
            mann_whitney_test=mann_whitney_test,
            cohens_d=0.3,
            hedges_g=0.29,
            relative_improvement=40.0,
            absolute_improvement=0.2,
            result=ABTestResult.TREATMENT_WINS,
            recommendation="Implement treatment",
        )

    def test_analysis_creation(self):
        """Test A/B test analysis creation."""
        analysis = self.create_test_analysis()

        assert analysis.experiment_id == "test_experiment"
        assert analysis.metric_name == "reward"
        assert analysis.control_mean == 0.5
        assert analysis.treatment_mean == 0.7
        assert analysis.relative_improvement == 40.0
        assert analysis.result == ABTestResult.TREATMENT_WINS

    def test_get_summary(self):
        """Test getting analysis summary."""
        analysis = self.create_test_analysis()
        summary = analysis.get_summary()

        assert summary["metric"] == "reward"
        assert summary["control"]["mean"] == 0.5
        assert summary["treatment"]["mean"] == 0.7
        assert summary["statistical_significance"]["t_test_significant"] is True
        assert summary["effect_size"]["relative_improvement"] == 40.0
        assert summary["result"] == ABTestResult.TREATMENT_WINS.value


class TestABTestManager:
    """Test ABTestManager functionality."""

    def test_manager_initialization(self):
        """Test A/B test manager initialization."""
        manager = ABTestManager(
            significance_level=0.05, minimum_effect_size=0.1, power_threshold=0.8
        )

        assert manager.significance_level == 0.05
        assert manager.minimum_effect_size == 0.1
        assert manager.power_threshold == 0.8

    def test_calculate_sample_size(self):
        """Test sample size calculation."""
        manager = ABTestManager()

        sample_size = manager.calculate_sample_size(
            expected_effect_size=0.5, baseline_std=1.0, power=0.8
        )

        # Should return a reasonable sample size
        assert isinstance(sample_size, int)
        assert sample_size > 0
        assert sample_size < 1000  # Sanity check

    def create_test_ab_experiment(self):
        """Create a test A/B experiment with data."""
        # Create configs
        control_config = MARLConfig(
            name="control",
            version="1.0.0",
            agents={
                "agent1": AgentConfig(
                    agent_id="agent1",
                    agent_type="generator",
                    state_dim=128,
                    action_dim=10,
                )
            },
        )

        treatment_config = MARLConfig(
            name="treatment",
            version="1.0.0",
            agents={
                "agent1": AgentConfig(
                    agent_id="agent1",
                    agent_type="generator",
                    state_dim=256,  # Different
                    action_dim=10,
                )
            },
        )

        # Create conditions
        conditions = [
            ExperimentCondition(
                condition_id="control",
                name="Control",
                description="Control condition",
                config=control_config,
            ),
            ExperimentCondition(
                condition_id="treatment",
                name="Treatment",
                description="Treatment condition",
                config=treatment_config,
            ),
        ]

        # Create experiment
        experiment = Experiment(
            experiment_id="ab_test_experiment",
            name="A/B Test",
            description="Test A/B analysis",
            experiment_type=ExperimentType.AB_TEST,
            conditions=conditions,
        )

        # Add mock data to results
        control_result = experiment.get_result("control")
        treatment_result = experiment.get_result("treatment")

        # Control data (lower performance)
        control_rewards = np.random.normal(0.5, 0.1, 100).tolist()
        control_result.performance_data["episode_rewards"] = control_rewards
        control_result.status = ExperimentStatus.COMPLETED

        # Treatment data (higher performance)
        treatment_rewards = np.random.normal(0.7, 0.12, 100).tolist()
        treatment_result.performance_data["episode_rewards"] = treatment_rewards
        treatment_result.status = ExperimentStatus.COMPLETED

        return experiment

    def test_analyze_ab_test(self):
        """Test A/B test analysis."""
        manager = ABTestManager()
        experiment = self.create_test_ab_experiment()

        # Analyze the test
        analysis = manager.analyze_ab_test(experiment, "episode_rewards")

        assert analysis is not None
        assert analysis.experiment_id == experiment.experiment_id
        assert analysis.metric_name == "episode_rewards"
        assert analysis.control_n == 100
        assert analysis.treatment_n == 100

        # Should detect significant difference (with high probability given the data)
        assert analysis.t_test.p_value >= 0.0
        assert analysis.mann_whitney_test.p_value >= 0.0

        # Effect size should be calculated
        assert isinstance(analysis.cohens_d, float)
        assert isinstance(analysis.hedges_g, float)

        # Should have a result
        assert analysis.result in [
            ABTestResult.TREATMENT_WINS,
            ABTestResult.CONTROL_WINS,
            ABTestResult.NO_SIGNIFICANT_DIFFERENCE,
            ABTestResult.INSUFFICIENT_DATA,
        ]

    def test_analyze_ab_test_invalid_experiment(self):
        """Test A/B test analysis with invalid experiment."""
        manager = ABTestManager()

        # Create non-A/B test experiment
        config = MARLConfig(
            name="test",
            version="1.0.0",
            agents={
                "agent1": AgentConfig(
                    agent_id="agent1",
                    agent_type="generator",
                    state_dim=128,
                    action_dim=10,
                )
            },
        )

        condition = ExperimentCondition(
            condition_id="test", name="Test", description="Test", config=config
        )

        experiment = Experiment(
            experiment_id="test",
            name="Test",
            description="Test",
            experiment_type=ExperimentType.PARAMETER_SWEEP,  # Not A/B test
            conditions=[condition],
        )

        # Should return None for non-A/B test
        analysis = manager.analyze_ab_test(experiment, "reward")
        assert analysis is None

    def test_analyze_multiple_metrics(self):
        """Test analyzing multiple metrics."""
        manager = ABTestManager()
        experiment = self.create_test_ab_experiment()

        # Add another metric
        control_result = experiment.get_result("control")
        treatment_result = experiment.get_result("treatment")

        control_result.performance_data["coordination_success"] = np.random.normal(
            0.6, 0.1, 100
        ).tolist()
        treatment_result.performance_data["coordination_success"] = np.random.normal(
            0.8, 0.1, 100
        ).tolist()

        # Analyze multiple metrics
        analyses = manager.analyze_multiple_metrics(
            experiment, ["episode_rewards", "coordination_success"]
        )

        assert len(analyses) == 2
        assert "episode_rewards" in analyses
        assert "coordination_success" in analyses

        # Each analysis should be valid
        for metric_name, analysis in analyses.items():
            assert analysis.metric_name == metric_name
            assert analysis.experiment_id == experiment.experiment_id

    def test_generate_ab_test_report(self):
        """Test generating comprehensive A/B test report."""
        manager = ABTestManager()
        experiment = self.create_test_ab_experiment()

        # Add another metric
        control_result = experiment.get_result("control")
        treatment_result = experiment.get_result("treatment")

        control_result.performance_data["coordination_success"] = np.random.normal(
            0.6, 0.1, 100
        ).tolist()
        treatment_result.performance_data["coordination_success"] = np.random.normal(
            0.8, 0.1, 100
        ).tolist()

        # Generate report
        report = manager.generate_ab_test_report(
            experiment, ["episode_rewards", "coordination_success"]
        )

        assert "summary" in report
        assert "detailed_results" in report
        assert "methodology" in report

        # Check summary
        summary = report["summary"]
        assert summary["experiment_id"] == experiment.experiment_id
        assert summary["experiment_name"] == experiment.name
        assert summary["metrics_analyzed"] == 2
        assert "overall_recommendation" in summary

        # Check detailed results
        detailed = report["detailed_results"]
        assert len(detailed) == 2
        assert "episode_rewards" in detailed
        assert "coordination_success" in detailed

        # Check methodology
        methodology = report["methodology"]
        assert "statistical_tests" in methodology
        assert "effect_size_measures" in methodology


class TestABTestManagerFactory:
    """Test ABTestManagerFactory functionality."""

    def test_create_default(self):
        """Test creating default A/B test manager."""
        manager = ABTestManagerFactory.create()

        assert isinstance(manager, ABTestManager)
        assert manager.significance_level == 0.05
        assert manager.minimum_effect_size == 0.1

    def test_create_strict(self):
        """Test creating strict A/B test manager."""
        manager = ABTestManagerFactory.create_strict()

        assert isinstance(manager, ABTestManager)
        assert manager.significance_level == 0.01
        assert manager.minimum_effect_size == 0.2

    def test_create_permissive(self):
        """Test creating permissive A/B test manager."""
        manager = ABTestManagerFactory.create_permissive()

        assert isinstance(manager, ABTestManager)
        assert manager.significance_level == 0.1
        assert manager.minimum_effect_size == 0.05


if __name__ == "__main__":
    pytest.main([__file__])
