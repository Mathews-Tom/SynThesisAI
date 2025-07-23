"""
Unit tests for MARL foundation components.

This module tests the basic MARL infrastructure including configuration,
exceptions, logging, and monitoring components.
"""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import yaml

from core.marl.config import (
    AgentConfig,
    CoordinationConfig,
    CurriculumAgentConfig,
    ExperienceConfig,
    GeneratorAgentConfig,
    MARLConfig,
    ValidatorAgentConfig,
    get_default_marl_config,
)
from core.marl.exceptions import (
    AgentFailureError,
    CommunicationError,
    ConsensusTimeoutError,
    CoordinationError,
    ExperienceBufferError,
    LearningDivergenceError,
    MARLError,
    OptimizationFailureError,
    PolicyNetworkError,
)
from core.marl.logging_config import MARLLogger, get_marl_logger, setup_marl_logging
from core.marl.monitoring import (
    AgentMetrics,
    CoordinationMetrics,
    MARLPerformanceMonitor,
    SystemMetrics,
)
from utils.exceptions import ValidationError


class TestMARLConfig:
    """Test MARL configuration management."""

    def test_agent_config_defaults(self):
        """Test default agent configuration values."""
        config = AgentConfig()

        assert config.hidden_layers == [256, 128, 64]
        assert config.activation == "relu"
        assert config.learning_rate == 0.001
        assert config.gamma == 0.99
        assert config.epsilon_initial == 1.0
        assert config.epsilon_decay == 0.995
        assert config.epsilon_min == 0.01
        assert config.buffer_size == 100000
        assert config.batch_size == 32
        assert config.target_update_freq == 1000

    def test_agent_config_validation(self):
        """Test agent configuration validation."""
        config = AgentConfig()

        # Valid configuration should not raise
        config.validate()

        # Invalid learning rate
        config.learning_rate = -0.1
        with pytest.raises(
            ValidationError, match="Learning rate must be between 0 and 1"
        ):
            config.validate()

        config.learning_rate = 1.5
        with pytest.raises(
            ValidationError, match="Learning rate must be between 0 and 1"
        ):
            config.validate()

        # Reset and test gamma
        config.learning_rate = 0.001
        config.gamma = -0.1
        with pytest.raises(ValidationError, match="Gamma must be between 0 and 1"):
            config.validate()

        # Reset and test buffer size
        config.gamma = 0.99
        config.buffer_size = -100
        with pytest.raises(ValidationError, match="Buffer size must be positive"):
            config.validate()

        # Reset and test batch size
        config.buffer_size = 1000
        config.batch_size = 2000
        with pytest.raises(
            ValidationError, match="Batch size must be positive and <= buffer size"
        ):
            config.validate()

    def test_generator_agent_config(self):
        """Test generator agent specific configuration."""
        config = GeneratorAgentConfig()

        assert config.strategy_count == 8
        assert config.novelty_weight == 0.3
        assert config.quality_weight == 0.5
        assert config.efficiency_weight == 0.2
        assert config.quality_threshold == 0.8
        assert config.novelty_threshold == 0.6
        assert config.coordination_bonus == 0.1
        assert config.validation_penalty == 0.2

    def test_validator_agent_config(self):
        """Test validator agent specific configuration."""
        config = ValidatorAgentConfig()

        assert config.threshold_count == 8
        assert config.validation_accuracy_weight == 0.7
        assert config.efficiency_weight == 0.3
        assert config.feedback_quality_weight == 0.2
        assert config.false_positive_penalty == 0.1
        assert config.false_negative_penalty == 0.15
        assert config.feedback_quality_bonus == 0.2

    def test_curriculum_agent_config(self):
        """Test curriculum agent specific configuration."""
        config = CurriculumAgentConfig()

        assert config.strategy_count == 8
        assert config.pedagogical_coherence_weight == 0.4
        assert config.learning_progression_weight == 0.4
        assert config.objective_alignment_weight == 0.2
        assert config.coherence_threshold == 0.7
        assert config.progression_threshold == 0.7
        assert config.integration_bonus == 0.15

    def test_coordination_config(self):
        """Test coordination configuration."""
        config = CoordinationConfig()

        assert config.consensus_strategy == "adaptive_consensus"
        assert config.min_consensus_quality == 0.8
        assert config.consensus_timeout == 30.0
        assert config.max_negotiation_rounds == 5
        assert config.conflict_resolution_strategy == "weighted_priority"
        assert config.message_queue_size == 1000
        assert config.communication_timeout == 10.0
        assert config.coordination_success_threshold == 0.85
        assert config.coordination_failure_threshold == 0.3

    def test_experience_config(self):
        """Test experience management configuration."""
        config = ExperienceConfig()

        assert config.shared_buffer_size == 50000
        assert config.agent_buffer_size == 25000
        assert config.high_reward_threshold == 0.8
        assert config.novelty_threshold == 0.7
        assert config.sharing_probability == 0.3
        assert config.priority_alpha == 0.6
        assert config.importance_sampling_beta == 0.4

    def test_marl_config_defaults(self):
        """Test main MARL configuration defaults."""
        config = MARLConfig()

        assert isinstance(config.generator_config, GeneratorAgentConfig)
        assert isinstance(config.validator_config, ValidatorAgentConfig)
        assert isinstance(config.curriculum_config, CurriculumAgentConfig)
        assert isinstance(config.coordination_config, CoordinationConfig)
        assert isinstance(config.experience_config, ExperienceConfig)

        assert config.monitoring_enabled is True
        assert config.metrics_collection_interval == 100
        assert config.performance_report_interval == 1000
        assert config.distributed_training is False
        assert config.num_workers == 4
        assert config.gpu_enabled is True
        assert config.log_level == "INFO"
        assert config.debug_mode is False
        assert config.checkpoint_interval == 5000

    def test_marl_config_validation(self):
        """Test MARL configuration validation."""
        config = MARLConfig()

        # Valid configuration should not raise
        config.validate()

        # Invalid coordination parameters
        config.coordination_config.min_consensus_quality = 1.5
        with pytest.raises(
            ValidationError, match="Min consensus quality must be between 0 and 1"
        ):
            config.validate()

        # Reset and test coordination success threshold
        config.coordination_config.min_consensus_quality = 0.8
        config.coordination_config.coordination_success_threshold = -0.1
        with pytest.raises(
            ValidationError,
            match="Coordination success threshold must be between 0 and 1",
        ):
            config.validate()

        # Reset and test system parameters
        config.coordination_config.coordination_success_threshold = 0.85
        config.num_workers = -1
        with pytest.raises(ValidationError, match="Number of workers must be positive"):
            config.validate()

        # Reset and test checkpoint interval
        config.num_workers = 4
        config.checkpoint_interval = -100
        with pytest.raises(
            ValidationError, match="Checkpoint interval must be positive"
        ):
            config.validate()

    def test_config_file_operations(self):
        """Test configuration file loading and saving."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_marl_config.yaml"

            # Create and save configuration
            original_config = MARLConfig()
            original_config.debug_mode = True
            original_config.log_level = "DEBUG"
            original_config.generator_config.learning_rate = 0.002

            original_config.save_to_file(config_path)

            # Load configuration
            loaded_config = MARLConfig.from_file(config_path)

            assert loaded_config.debug_mode is True
            assert loaded_config.log_level == "DEBUG"
            assert loaded_config.generator_config.learning_rate == 0.002

    def test_config_to_dict(self):
        """Test configuration serialization to dictionary."""
        config = MARLConfig()
        config_dict = config.to_dict()

        assert "generator" in config_dict
        assert "validator" in config_dict
        assert "curriculum" in config_dict
        assert "coordination" in config_dict
        assert "experience" in config_dict
        assert "system" in config_dict

        assert config_dict["system"]["monitoring_enabled"] is True
        assert config_dict["system"]["log_level"] == "INFO"

    def test_get_default_marl_config(self):
        """Test default configuration factory function."""
        config = get_default_marl_config()

        assert isinstance(config, MARLConfig)
        assert config.debug_mode is True
        assert config.log_level == "DEBUG"
        assert config.checkpoint_interval == 100
        assert config.generator_config.max_episodes == 1000


class TestMARLExceptions:
    """Test MARL exception classes."""

    def test_marl_error_base(self):
        """Test base MARL error class."""
        details = {"context": "test", "value": 42}
        error = MARLError("Test error", details)

        assert str(error) == "Test error"
        assert error.details == details

    def test_coordination_error(self):
        """Test coordination error class."""
        agent_states = {"agent1": {"state": "active"}, "agent2": {"state": "failed"}}
        error = CoordinationError(
            "Coordination failed",
            coordination_type="consensus",
            agent_states=agent_states,
        )

        assert str(error) == "Coordination failed"
        assert error.coordination_type == "consensus"
        assert error.agent_states == agent_states

    def test_agent_failure_error(self):
        """Test agent failure error class."""
        agent_state = {"policy_loss": 0.5, "epsilon": 0.1}
        error = AgentFailureError(
            "Agent failed",
            agent_id="generator",
            failure_type="policy_divergence",
            agent_state=agent_state,
        )

        assert str(error) == "Agent failed"
        assert error.agent_id == "generator"
        assert error.failure_type == "policy_divergence"
        assert error.agent_state == agent_state

    def test_optimization_failure_error(self):
        """Test optimization failure error class."""
        params = {"learning_rate": 0.001, "batch_size": 32}
        error = OptimizationFailureError(
            "Optimization failed", optimizer_type="MIPROv2", optimization_params=params
        )

        assert str(error) == "Optimization failed"
        assert error.optimizer_type == "MIPROv2"
        assert error.optimization_params == params

    def test_learning_divergence_error(self):
        """Test learning divergence error class."""
        metrics = {"loss_variance": 10.5, "reward_variance": 5.2}
        error = LearningDivergenceError(
            "Learning diverged", agent_id="validator", divergence_metrics=metrics
        )

        assert str(error) == "Learning diverged"
        assert error.agent_id == "validator"
        assert error.divergence_metrics == metrics

    def test_consensus_timeout_error(self):
        """Test consensus timeout error class."""
        partial_consensus = {"agreement_level": 0.6}
        error = ConsensusTimeoutError(
            "Consensus timeout",
            timeout_duration=30.0,
            partial_consensus=partial_consensus,
        )

        assert str(error) == "Consensus timeout"
        assert error.coordination_type == "consensus_timeout"
        assert error.timeout_duration == 30.0
        assert error.partial_consensus == partial_consensus

    def test_communication_error(self):
        """Test communication error class."""
        error = CommunicationError(
            "Message failed",
            sender_id="generator",
            receiver_id="validator",
            message_type="action_proposal",
        )

        assert str(error) == "Message failed"
        assert error.sender_id == "generator"
        assert error.receiver_id == "validator"
        assert error.message_type == "action_proposal"

    def test_experience_buffer_error(self):
        """Test experience buffer error class."""
        error = ExperienceBufferError(
            "Buffer overflow", buffer_type="shared", buffer_size=50000, operation="add"
        )

        assert str(error) == "Buffer overflow"
        assert error.buffer_type == "shared"
        assert error.buffer_size == 50000
        assert error.operation == "add"

    def test_policy_network_error(self):
        """Test policy network error class."""
        error = PolicyNetworkError(
            "Network forward pass failed",
            agent_id="curriculum",
            network_type="q_network",
            operation="forward",
        )

        assert str(error) == "Network forward pass failed"
        assert error.agent_id == "curriculum"
        assert error.network_type == "q_network"
        assert error.operation == "forward"


class TestMARLLogging:
    """Test MARL logging configuration."""

    def test_marl_logger_initialization(self):
        """Test MARL logger initialization."""
        config = get_default_marl_config()
        logger = MARLLogger("test.component", config)

        assert logger.logger.name == "test.component"
        assert logger.config == config

    def test_marl_logger_without_config(self):
        """Test MARL logger without configuration."""
        logger = MARLLogger("test.component")

        assert logger.logger.name == "test.component"
        assert logger.config is None

    @patch("core.marl.logging_config.logging.getLogger")
    def test_coordination_logging_methods(self, mock_get_logger):
        """Test coordination-specific logging methods."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        logger = MARLLogger("test.coordinator")

        # Test coordination start logging
        logger.log_coordination_start("req123", ["generator", "validator"])
        mock_logger.info.assert_called_with(
            "Starting coordination for request %s with agents: %s",
            "req123",
            "generator, validator",
        )

        # Test coordination success logging
        logger.log_coordination_success("req123", 2.5, 0.95)
        mock_logger.info.assert_called_with(
            "Coordination completed for request %s in %.2fs (success rate: %.2f)",
            "req123",
            2.5,
            0.95,
        )

        # Test coordination failure logging
        logger.log_coordination_failure("req123", "timeout", "Consensus not reached")
        mock_logger.error.assert_called_with(
            "Coordination failed for request %s - %s: %s",
            "req123",
            "timeout",
            "Consensus not reached",
        )

    @patch("core.marl.logging_config.logging.getLogger")
    def test_agent_logging_methods(self, mock_get_logger):
        """Test agent-specific logging methods."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        logger = MARLLogger("test.agent")

        # Test agent action logging
        logger.log_agent_action("generator", "strategy_1", 0.85, "high_quality_request")
        mock_logger.debug.assert_called_with(
            "Agent %s selected action '%s' (confidence: %.2f) - State: %s",
            "generator",
            "strategy_1",
            0.85,
            "high_quality_request",
        )

        # Test learning update logging
        logger.log_learning_update("validator", 100, 0.75, 0.02, 0.3)
        mock_logger.debug.assert_called_with(
            "Agent %s learning update - Episode: %d, Reward: %.3f, Loss: %.4f, Epsilon: %.3f",
            "validator",
            100,
            0.75,
            0.02,
            0.3,
        )

    def test_setup_marl_logging(self):
        """Test MARL logging setup function."""
        config = get_default_marl_config()
        loggers = setup_marl_logging(config)

        expected_components = [
            "marl.coordinator",
            "marl.generator_agent",
            "marl.validator_agent",
            "marl.curriculum_agent",
            "marl.coordination_policy",
            "marl.consensus_mechanism",
            "marl.communication_protocol",
            "marl.experience_manager",
            "marl.performance_monitor",
            "marl.error_handler",
        ]

        for component in expected_components:
            assert component in loggers
            assert isinstance(loggers[component], MARLLogger)

    def test_get_marl_logger(self):
        """Test MARL logger factory function."""
        config = get_default_marl_config()
        logger = get_marl_logger("test_component", config)

        assert isinstance(logger, MARLLogger)
        assert logger.logger.name == "marl.test_component"
        assert logger.config == config


class TestMARLMonitoring:
    """Test MARL monitoring components."""

    def test_coordination_metrics(self):
        """Test coordination metrics tracking."""
        metrics = CoordinationMetrics()

        # Initial state
        assert metrics.total_episodes == 0
        assert metrics.successful_episodes == 0
        assert metrics.failed_episodes == 0
        assert metrics.success_rate == 0.0
        assert metrics.average_time == 0.0

        # Record successful episodes
        metrics.record_success(1.5)
        metrics.record_success(2.0)

        assert metrics.total_episodes == 2
        assert metrics.successful_episodes == 2
        assert metrics.failed_episodes == 0
        assert metrics.success_rate == 1.0
        assert metrics.average_time == 1.75

        # Record failed episode
        metrics.record_failure(3.0)

        assert metrics.total_episodes == 3
        assert metrics.successful_episodes == 2
        assert metrics.failed_episodes == 1
        assert metrics.success_rate == 2 / 3
        assert metrics.average_time == (1.5 + 2.0 + 3.0) / 3

    def test_agent_metrics(self):
        """Test agent metrics tracking."""
        metrics = AgentMetrics(agent_id="test_agent")

        # Initial state
        assert metrics.agent_id == "test_agent"
        assert metrics.total_actions == 0
        assert metrics.successful_actions == 0
        assert metrics.success_rate == 0.0
        assert metrics.average_reward == 0.0
        assert metrics.learning_progress == 0.0

        # Update with successful actions
        metrics.update(action_success=True, reward=0.8, loss=0.1, epsilon=0.5)
        metrics.update(action_success=True, reward=0.9, loss=0.08, epsilon=0.45)
        metrics.update(action_success=False, reward=0.2, loss=0.15, epsilon=0.4)

        assert metrics.total_actions == 3
        assert metrics.successful_actions == 2
        assert metrics.success_rate == 2 / 3
        assert metrics.average_reward == (0.8 + 0.9 + 0.2) / 3

        # Test summary
        summary = metrics.get_summary()
        assert summary["agent_id"] == "test_agent"
        assert summary["total_actions"] == 3
        assert summary["success_rate"] == 2 / 3
        assert summary["current_epsilon"] == 0.4
        assert summary["recent_loss"] == 0.15

    def test_system_metrics(self):
        """Test system metrics tracking."""
        metrics = SystemMetrics()

        # Initial state
        assert metrics.total_requests == 0
        assert metrics.successful_requests == 0
        assert metrics.success_rate == 0.0
        assert metrics.average_processing_time == 0.0
        assert isinstance(metrics.uptime, timedelta)

        # Update with requests
        metrics.update(
            request_success=True, processing_time=1.2, memory_mb=512.0, cpu_percent=45.0
        )
        metrics.update(
            request_success=True, processing_time=0.8, memory_mb=520.0, cpu_percent=50.0
        )
        metrics.update(
            request_success=False,
            processing_time=2.0,
            memory_mb=530.0,
            cpu_percent=60.0,
        )

        assert metrics.total_requests == 3
        assert metrics.successful_requests == 2
        assert metrics.success_rate == 2 / 3
        assert metrics.average_processing_time == (1.2 + 0.8 + 2.0) / 3

        # Test summary
        summary = metrics.get_summary()
        assert summary["total_requests"] == 3
        assert summary["success_rate"] == 2 / 3
        assert summary["current_memory_mb"] == 530.0
        assert summary["current_cpu_percent"] == 60.0

    def test_marl_performance_monitor(self):
        """Test MARL performance monitor."""
        config = get_default_marl_config()
        monitor = MARLPerformanceMonitor(config)

        # Initial state
        assert monitor.episode_count == 0
        assert isinstance(monitor.coordination_metrics, CoordinationMetrics)
        assert len(monitor.agent_metrics) == 3
        assert "generator" in monitor.agent_metrics
        assert "validator" in monitor.agent_metrics
        assert "curriculum" in monitor.agent_metrics

        # Record episode
        episode_data = {
            "coordination_success": True,
            "coordination_time": 1.5,
            "agent_data": {
                "generator": {
                    "action_success": True,
                    "reward": 0.8,
                    "loss": 0.1,
                    "epsilon": 0.5,
                },
                "validator": {
                    "action_success": True,
                    "reward": 0.9,
                    "loss": 0.08,
                    "epsilon": 0.45,
                },
            },
            "system_metrics": {"memory_mb": 512.0, "cpu_percent": 45.0},
        }

        monitor.record_coordination_episode(episode_data)

        assert monitor.episode_count == 1
        assert monitor.coordination_metrics.total_episodes == 1
        assert monitor.coordination_metrics.successful_episodes == 1
        assert monitor.agent_metrics["generator"].total_actions == 1
        assert monitor.agent_metrics["validator"].total_actions == 1
        assert monitor.system_metrics.total_requests == 1

    def test_performance_report_generation(self):
        """Test performance report generation."""
        monitor = MARLPerformanceMonitor()

        # Record some episodes
        for i in range(5):
            episode_data = {
                "coordination_success": i % 2 == 0,  # Alternate success/failure
                "coordination_time": 1.0 + i * 0.1,
                "agent_data": {
                    "generator": {
                        "action_success": True,
                        "reward": 0.8,
                        "loss": 0.1,
                        "epsilon": 0.5,
                    }
                },
                "system_metrics": {
                    "memory_mb": 500.0 + i * 10,
                    "cpu_percent": 40.0 + i * 2,
                },
            }
            monitor.record_coordination_episode(episode_data)

        # Generate report
        report = monitor.get_performance_report()

        assert "timestamp" in report
        assert report["episode_count"] == 5
        assert "coordination_metrics" in report
        assert "agent_performance" in report
        assert "system_performance" in report
        assert "improvement_recommendations" in report

        # Check coordination metrics
        coord_metrics = report["coordination_metrics"]
        assert coord_metrics["total_episodes"] == 5
        assert coord_metrics["success_rate"] == 3 / 5  # 3 successful out of 5

    def test_recommendation_generation(self):
        """Test performance improvement recommendations."""
        monitor = MARLPerformanceMonitor()

        # Create scenario with poor performance
        for i in range(10):
            episode_data = {
                "coordination_success": False,  # All failures
                "coordination_time": 15.0,  # Long coordination time
                "agent_data": {
                    "generator": {
                        "action_success": False,
                        "reward": 0.1,
                        "loss": 0.5,
                        "epsilon": 0.9,
                    }
                },
                "system_metrics": {"memory_mb": 1000.0, "cpu_percent": 90.0},
            }
            monitor.record_coordination_episode(episode_data)

        recommendations = monitor.generate_recommendations()

        # Should have recommendations for poor performance
        assert len(recommendations) > 0

        # Check for specific recommendation types
        recommendation_text = " ".join(recommendations)
        assert "coordination success rate" in recommendation_text.lower()
        assert "coordination time" in recommendation_text.lower()

    def test_metrics_export(self):
        """Test metrics export functionality."""
        monitor = MARLPerformanceMonitor()

        # Record some data
        episode_data = {
            "coordination_success": True,
            "coordination_time": 1.5,
            "agent_data": {"generator": {"action_success": True, "reward": 0.8}},
        }
        monitor.record_coordination_episode(episode_data)

        # Export metrics
        with tempfile.TemporaryDirectory() as temp_dir:
            export_path = Path(temp_dir) / "metrics_export.json"
            monitor.export_metrics(export_path)

            assert export_path.exists()

            # Verify export content
            import json

            with open(export_path, "r", encoding="utf-8") as f:
                export_data = json.load(f)

            assert "export_timestamp" in export_data
            assert "performance_report" in export_data
            assert "performance_history" in export_data

    def test_real_time_status(self):
        """Test real-time status reporting."""
        monitor = MARLPerformanceMonitor()

        # Record some episodes
        episode_data = {
            "coordination_success": True,
            "coordination_time": 1.5,
            "agent_data": {
                "generator": {"action_success": True, "reward": 0.8},
                "validator": {"action_success": True, "reward": 0.9},
            },
        }
        monitor.record_coordination_episode(episode_data)

        status = monitor.get_real_time_status()

        assert "timestamp" in status
        assert "system_status" in status
        assert "coordination_success_rate" in status
        assert "active_episodes" in status
        assert "agent_status" in status

        # Check agent status
        agent_status = status["agent_status"]
        assert "generator" in agent_status
        assert "validator" in agent_status
        assert "success_rate" in agent_status["generator"]
        assert "learning_progress" in agent_status["generator"]

    def test_metrics_reset(self):
        """Test metrics reset functionality."""
        monitor = MARLPerformanceMonitor()

        # Record some data
        episode_data = {
            "coordination_success": True,
            "coordination_time": 1.5,
            "agent_data": {"generator": {"action_success": True, "reward": 0.8}},
        }
        monitor.record_coordination_episode(episode_data)

        assert monitor.episode_count == 1
        assert monitor.coordination_metrics.total_episodes == 1

        # Reset metrics
        monitor.reset_metrics()

        assert monitor.episode_count == 0
        assert monitor.coordination_metrics.total_episodes == 0
        assert len(monitor.performance_history) == 0
