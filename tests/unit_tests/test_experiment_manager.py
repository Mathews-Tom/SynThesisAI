"""
Unit tests for MARL Experiment Manager.

Tests experiment creation, management, and analysis functionality.
"""

# Standard Library
import tempfile
from pathlib import Path

# Third-Party Library
import pytest

# SynThesisAI Modules
from core.marl.config.config_schema import AgentConfig, MARLConfig
from core.marl.experimentation.experiment_manager import (
    Experiment,
    ExperimentCondition,
    ExperimentManager,
    ExperimentManagerFactory,
    ExperimentResult,
    ExperimentStatus,
    ExperimentType,
)


class TestExperimentCondition:
    """Test ExperimentCondition functionality."""

    def test_condition_creation(self):
        """Test experiment condition creation."""
        config = MARLConfig(
            name="test_config",
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
            condition_id="test_condition",
            name="Test Condition",
            description="A test condition",
            config=config,
            parameters={"learning_rate": 0.001},
            metadata={"version": "1.0"},
        )

        assert condition.condition_id == "test_condition"
        assert condition.name == "Test Condition"
        assert condition.description == "A test condition"
        assert condition.config == config
        assert condition.parameters["learning_rate"] == 0.001
        assert condition.metadata["version"] == "1.0"

    def test_condition_validation(self):
        """Test experiment condition validation."""
        config = MARLConfig(
            name="test_config",
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

        # Empty condition ID
        with pytest.raises(ValueError, match="Condition ID cannot be empty"):
            ExperimentCondition(condition_id="", name="Test", description="Test", config=config)

        # Empty name
        with pytest.raises(ValueError, match="Condition name cannot be empty"):
            ExperimentCondition(condition_id="test", name="", description="Test", config=config)


class TestExperimentResult:
    """Test ExperimentResult functionality."""

    def test_result_creation(self):
        """Test experiment result creation."""
        result = ExperimentResult(condition_id="test_condition", status=ExperimentStatus.CREATED)

        assert result.condition_id == "test_condition"
        assert result.status == ExperimentStatus.CREATED
        assert result.start_time is None
        assert result.end_time is None
        assert result.metrics == {}
        assert result.performance_data == {}

    def test_add_metric(self):
        """Test adding metrics to result."""
        result = ExperimentResult(condition_id="test_condition", status=ExperimentStatus.RUNNING)

        # Add metric
        result.add_metric("reward", 0.5)

        assert "reward" in result.metrics
        assert len(result.metrics["reward"]) == 1
        assert result.metrics["reward"][0]["value"] == 0.5

        # Add another value
        result.add_metric("reward", 0.7)

        assert len(result.metrics["reward"]) == 2
        assert result.metrics["reward"][1]["value"] == 0.7

    def test_add_performance_data(self):
        """Test adding performance data to result."""
        result = ExperimentResult(condition_id="test_condition", status=ExperimentStatus.RUNNING)

        # Add performance data
        result.add_performance_data("episode_rewards", [0.1, 0.2, 0.3])

        assert "episode_rewards" in result.performance_data
        assert result.performance_data["episode_rewards"] == [0.1, 0.2, 0.3]

        # Add more data
        result.add_performance_data("episode_rewards", [0.4, 0.5])

        assert result.performance_data["episode_rewards"] == [0.1, 0.2, 0.3, 0.4, 0.5]

    def test_get_final_metric(self):
        """Test getting final metric value."""
        result = ExperimentResult(condition_id="test_condition", status=ExperimentStatus.COMPLETED)

        # No metric
        assert result.get_final_metric("nonexistent") is None

        # Add metrics
        result.add_metric("reward", 0.5)
        result.add_metric("reward", 0.7)
        result.add_metric("reward", 0.9)

        # Should get most recent value
        assert result.get_final_metric("reward") == 0.9

    def test_get_average_metric(self):
        """Test getting average metric value."""
        result = ExperimentResult(condition_id="test_condition", status=ExperimentStatus.COMPLETED)

        # No metric
        assert result.get_average_metric("nonexistent") is None

        # Add numeric metrics
        result.add_metric("reward", 0.5)
        result.add_metric("reward", 0.7)
        result.add_metric("reward", 0.8)

        # Should get average
        assert result.get_average_metric("reward") == 0.6666666666666666

        # Add non-numeric metric
        result.add_metric("status", "completed")

        # Should handle mixed types
        assert result.get_average_metric("reward") == 0.6666666666666666


class TestExperiment:
    """Test Experiment functionality."""

    def create_test_experiment(self):
        """Create a test experiment."""
        config1 = MARLConfig(
            name="config1",
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

        config2 = MARLConfig(
            name="config2",
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

        conditions = [
            ExperimentCondition(
                condition_id="control",
                name="Control",
                description="Control condition",
                config=config1,
            ),
            ExperimentCondition(
                condition_id="treatment",
                name="Treatment",
                description="Treatment condition",
                config=config2,
            ),
        ]

        return Experiment(
            experiment_id="test_experiment",
            name="Test Experiment",
            description="A test experiment",
            experiment_type=ExperimentType.AB_TEST,
            conditions=conditions,
        )

    def test_experiment_creation(self):
        """Test experiment creation."""
        experiment = self.create_test_experiment()

        assert experiment.experiment_id == "test_experiment"
        assert experiment.name == "Test Experiment"
        assert experiment.experiment_type == ExperimentType.AB_TEST
        assert len(experiment.conditions) == 2
        assert experiment.status == ExperimentStatus.CREATED
        assert len(experiment.results) == 2  # Results initialized for each condition

    def test_experiment_validation(self):
        """Test experiment validation."""
        config = MARLConfig(
            name="test_config",
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

        # Empty experiment ID
        with pytest.raises(ValueError, match="Experiment ID cannot be empty"):
            Experiment(
                experiment_id="",
                name="Test",
                description="Test",
                experiment_type=ExperimentType.AB_TEST,
                conditions=[condition],
            )

        # Empty name
        with pytest.raises(ValueError, match="Experiment name cannot be empty"):
            Experiment(
                experiment_id="test",
                name="",
                description="Test",
                experiment_type=ExperimentType.AB_TEST,
                conditions=[condition],
            )

        # No conditions
        with pytest.raises(ValueError, match="Experiment must have at least one condition"):
            Experiment(
                experiment_id="test",
                name="Test",
                description="Test",
                experiment_type=ExperimentType.AB_TEST,
                conditions=[],
            )

    def test_get_condition(self):
        """Test getting condition by ID."""
        experiment = self.create_test_experiment()

        # Existing condition
        control = experiment.get_condition("control")
        assert control is not None
        assert control.condition_id == "control"
        assert control.name == "Control"

        # Non-existent condition
        assert experiment.get_condition("nonexistent") is None

    def test_get_result(self):
        """Test getting result by condition ID."""
        experiment = self.create_test_experiment()

        # Existing result
        control_result = experiment.get_result("control")
        assert control_result is not None
        assert control_result.condition_id == "control"
        assert control_result.status == ExperimentStatus.CREATED

        # Non-existent result
        assert experiment.get_result("nonexistent") is None

    def test_completion_percentage(self):
        """Test completion percentage calculation."""
        experiment = self.create_test_experiment()

        # Initially 0%
        assert experiment.get_completion_percentage() == 0.0

        # Complete one condition
        experiment.results["control"].status = ExperimentStatus.COMPLETED
        assert experiment.get_completion_percentage() == 50.0

        # Complete both conditions
        experiment.results["treatment"].status = ExperimentStatus.COMPLETED
        assert experiment.get_completion_percentage() == 100.0

        # One failed
        experiment.results["treatment"].status = ExperimentStatus.FAILED
        assert experiment.get_completion_percentage() == 100.0  # Failed counts as completed


class TestExperimentManager:
    """Test ExperimentManager functionality."""

    def test_manager_initialization(self):
        """Test experiment manager initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ExperimentManager(temp_dir)

            assert manager.experiments_dir == Path(temp_dir)
            assert manager.auto_save is True
            assert len(manager._experiments) == 0

    def test_create_experiment(self):
        """Test experiment creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ExperimentManager(temp_dir)

            config = MARLConfig(
                name="test_config",
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
                condition_id="test_condition",
                name="Test Condition",
                description="A test condition",
                config=config,
            )

            experiment = manager.create_experiment(
                name="Test Experiment",
                description="A test experiment",
                experiment_type=ExperimentType.PARAMETER_SWEEP,
                conditions=[condition],
            )

            assert experiment.name == "Test Experiment"
            assert experiment.experiment_type == ExperimentType.PARAMETER_SWEEP
            assert len(experiment.conditions) == 1
            assert experiment.experiment_id in manager._experiments

    def test_create_ab_test(self):
        """Test A/B test creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ExperimentManager(temp_dir)

            control_config = MARLConfig(
                name="control_config",
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
                name="treatment_config",
                version="1.0.0",
                agents={
                    "agent1": AgentConfig(
                        agent_id="agent1",
                        agent_type="generator",
                        state_dim=256,  # Different state dim
                        action_dim=10,
                    )
                },
            )

            experiment = manager.create_ab_test(
                name="Test A/B Test",
                description="Testing different state dimensions",
                control_config=control_config,
                treatment_config=treatment_config,
            )

            assert experiment.experiment_type == ExperimentType.AB_TEST
            assert len(experiment.conditions) == 2

            # Check conditions
            control_condition = experiment.get_condition("control")
            treatment_condition = experiment.get_condition("treatment")

            assert control_condition is not None
            assert treatment_condition is not None
            assert control_condition.name == "Control"
            assert treatment_condition.name == "Treatment"

    def test_create_parameter_sweep(self):
        """Test parameter sweep creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ExperimentManager(temp_dir)

            base_config = MARLConfig(
                name="base_config",
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

            parameter_ranges = {
                "learning_rate": [0.001, 0.01, 0.1],
                "batch_size": [32, 64],
            }

            experiment = manager.create_parameter_sweep(
                name="Parameter Sweep Test",
                description="Testing different parameter combinations",
                base_config=base_config,
                parameter_ranges=parameter_ranges,
            )

            assert experiment.experiment_type == ExperimentType.PARAMETER_SWEEP
            # Should have 3 * 2 = 6 combinations
            assert len(experiment.conditions) == 6

            # Check that parameters are set correctly
            for condition in experiment.conditions:
                assert "learning_rate" in condition.parameters
                assert "batch_size" in condition.parameters
                assert condition.parameters["learning_rate"] in [0.001, 0.01, 0.1]
                assert condition.parameters["batch_size"] in [32, 64]

    def test_create_algorithm_comparison(self):
        """Test algorithm comparison creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ExperimentManager(temp_dir)

            config1 = MARLConfig(
                name="dqn_config",
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

            config2 = MARLConfig(
                name="ppo_config",
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

            algorithm_configs = {"DQN": config1, "PPO": config2}

            experiment = manager.create_algorithm_comparison(
                name="Algorithm Comparison",
                description="Comparing DQN vs PPO",
                algorithm_configs=algorithm_configs,
            )

            assert experiment.experiment_type == ExperimentType.ALGORITHM_COMPARISON
            assert len(experiment.conditions) == 2

            # Check conditions
            dqn_condition = experiment.get_condition("dqn")
            ppo_condition = experiment.get_condition("ppo")

            assert dqn_condition is not None
            assert ppo_condition is not None
            assert dqn_condition.name == "DQN"
            assert ppo_condition.name == "PPO"

    def test_get_experiment(self):
        """Test getting experiment by ID."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ExperimentManager(temp_dir)

            config = MARLConfig(
                name="test_config",
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
                condition_id="test_condition",
                name="Test Condition",
                description="A test condition",
                config=config,
            )

            experiment = manager.create_experiment(
                name="Test Experiment",
                description="A test experiment",
                experiment_type=ExperimentType.PARAMETER_SWEEP,
                conditions=[condition],
            )

            # Get existing experiment
            retrieved = manager.get_experiment(experiment.experiment_id)
            assert retrieved is not None
            assert retrieved.experiment_id == experiment.experiment_id
            assert retrieved.name == experiment.name

            # Get non-existent experiment
            assert manager.get_experiment("nonexistent") is None

    def test_list_experiments(self):
        """Test listing experiments with filtering."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ExperimentManager(temp_dir)

            config = MARLConfig(
                name="test_config",
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
                condition_id="test_condition",
                name="Test Condition",
                description="A test condition",
                config=config,
            )

            # Create different types of experiments
            ab_test = manager.create_experiment(
                name="A/B Test",
                description="An A/B test",
                experiment_type=ExperimentType.AB_TEST,
                conditions=[condition],
            )

            param_sweep = manager.create_experiment(
                name="Parameter Sweep",
                description="A parameter sweep",
                experiment_type=ExperimentType.PARAMETER_SWEEP,
                conditions=[condition],
            )

            # List all experiments
            all_experiments = manager.list_experiments()
            assert len(all_experiments) == 2

            # Filter by type
            ab_tests = manager.list_experiments(experiment_type=ExperimentType.AB_TEST)
            assert len(ab_tests) == 1
            assert ab_tests[0].experiment_id == ab_test.experiment_id

            param_sweeps = manager.list_experiments(experiment_type=ExperimentType.PARAMETER_SWEEP)
            assert len(param_sweeps) == 1
            assert param_sweeps[0].experiment_id == param_sweep.experiment_id

            # Filter by status
            created_experiments = manager.list_experiments(status=ExperimentStatus.CREATED)
            assert len(created_experiments) == 2

            running_experiments = manager.list_experiments(status=ExperimentStatus.RUNNING)
            assert len(running_experiments) == 0

    def test_delete_experiment(self):
        """Test experiment deletion."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ExperimentManager(temp_dir)

            config = MARLConfig(
                name="test_config",
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
                condition_id="test_condition",
                name="Test Condition",
                description="A test condition",
                config=config,
            )

            experiment = manager.create_experiment(
                name="Test Experiment",
                description="A test experiment",
                experiment_type=ExperimentType.PARAMETER_SWEEP,
                conditions=[condition],
            )

            experiment_id = experiment.experiment_id

            # Verify experiment exists
            assert manager.get_experiment(experiment_id) is not None

            # Delete experiment
            success = manager.delete_experiment(experiment_id)
            assert success is True

            # Verify experiment is deleted
            assert manager.get_experiment(experiment_id) is None

            # Try to delete non-existent experiment
            success = manager.delete_experiment("nonexistent")
            assert success is False

    def test_update_experiment_result(self):
        """Test updating experiment results."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ExperimentManager(temp_dir)

            config = MARLConfig(
                name="test_config",
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
                condition_id="test_condition",
                name="Test Condition",
                description="A test condition",
                config=config,
            )

            experiment = manager.create_experiment(
                name="Test Experiment",
                description="A test experiment",
                experiment_type=ExperimentType.PARAMETER_SWEEP,
                conditions=[condition],
            )

            # Update result
            result_update = {
                "status": ExperimentStatus.COMPLETED,
                "duration_seconds": 120.5,
            }

            success = manager.update_experiment_result(
                experiment.experiment_id, "test_condition", result_update
            )

            assert success is True

            # Verify update
            result = experiment.get_result("test_condition")
            assert result.status == ExperimentStatus.COMPLETED
            assert result.duration_seconds == 120.5

    def test_get_experiment_summary(self):
        """Test getting experiment summary."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ExperimentManager(temp_dir)

            config = MARLConfig(
                name="test_config",
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
                condition_id="test_condition",
                name="Test Condition",
                description="A test condition",
                config=config,
            )

            experiment = manager.create_experiment(
                name="Test Experiment",
                description="A test experiment",
                experiment_type=ExperimentType.PARAMETER_SWEEP,
                conditions=[condition],
            )

            # Add some metrics to result
            result = experiment.get_result("test_condition")
            result.add_metric("reward", 0.5)
            result.add_metric("reward", 0.7)
            result.status = ExperimentStatus.COMPLETED

            # Get summary
            summary = manager.get_experiment_summary(experiment.experiment_id)

            assert summary is not None
            assert summary["experiment_id"] == experiment.experiment_id
            assert summary["name"] == experiment.name
            assert summary["type"] == ExperimentType.PARAMETER_SWEEP.value
            assert summary["total_conditions"] == 1
            assert summary["completed_conditions"] == 1
            assert summary["completion_percentage"] == 100.0
            assert "metric_summaries" in summary


class TestExperimentManagerFactory:
    """Test ExperimentManagerFactory functionality."""

    def test_create_default(self):
        """Test creating default experiment manager."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ExperimentManagerFactory.create(temp_dir)

            assert isinstance(manager, ExperimentManager)
            assert manager.experiments_dir == Path(temp_dir)
            assert manager.auto_save is True

    def test_create_with_auto_save(self):
        """Test creating experiment manager with auto-save configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ExperimentManagerFactory.create_with_auto_save(temp_dir, auto_save=False)

            assert isinstance(manager, ExperimentManager)
            assert manager.experiments_dir == Path(temp_dir)
            assert manager.auto_save is False


if __name__ == "__main__":
    pytest.main([__file__])
