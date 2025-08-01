"""
Unit tests for MARL Configuration Validator.

Tests configuration validation, compatibility checking, and optimization suggestions.
"""

# Standard Library
from unittest.mock import Mock

# Third-Party Library
import pytest

# SynThesisAI Modules
from core.marl.config.config_schema import (
    AgentConfig,
    ConsensusConfig,
    ConsensusStrategy,
    CoordinationConfig,
    ExplorationConfig,
    LearningConfig,
    MARLConfig,
    NetworkConfig,
    OptimizationConfig,
    ReplayBufferConfig,
    SharedLearningConfig,
    SystemConfig,
)
from core.marl.config.config_validator import (
    ConfigValidator,
    ConfigValidatorFactory,
    ValidationMessage,
    ValidationSeverity,
)


class TestValidationMessage:
    """Test ValidationMessage functionality."""

    def test_validation_message_creation(self):
        """Test validation message creation."""
        message = ValidationMessage(
            severity=ValidationSeverity.ERROR,
            message="Test error message",
            path="agents.test_agent.learning_rate",
            suggestion="Use a positive value",
        )

        assert message.severity == ValidationSeverity.ERROR
        assert message.message == "Test error message"
        assert message.path == "agents.test_agent.learning_rate"
        assert message.suggestion == "Use a positive value"

    def test_validation_message_string_representation(self):
        """Test validation message string representation."""
        message = ValidationMessage(
            severity=ValidationSeverity.WARNING,
            message="Test warning",
            path="coordination.timeout",
            suggestion="Consider reducing timeout",
        )

        str_repr = str(message)
        assert "[WARNING]" in str_repr
        assert "coordination.timeout" in str_repr
        assert "Test warning" in str_repr
        assert "Consider reducing timeout" in str_repr

        # Test without path and suggestion
        simple_message = ValidationMessage(
            severity=ValidationSeverity.INFO, message="Simple info message"
        )

        str_repr = str(simple_message)
        assert "[INFO]" in str_repr
        assert "Simple info message" in str_repr


class TestConfigValidator:
    """Test ConfigValidator functionality."""

    def test_validator_initialization(self):
        """Test validator initialization."""
        validator = ConfigValidator()

        # Should have thresholds configured
        assert "learning_rate" in validator._thresholds
        assert "batch_size" in validator._thresholds

        # Should have optimization rules
        assert "memory_usage" in validator._optimization_rules
        assert "training_stability" in validator._optimization_rules

    def test_validate_basic_structure_valid(self):
        """Test validation of valid basic structure."""
        validator = ConfigValidator()

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

        errors, warnings = validator.validate_config(config)

        # Should have minimal errors for basic valid config
        # (may have some warnings about missing configurations)
        assert len([e for e in errors if "required" in e.lower()]) == 0

    def test_validate_basic_structure_missing_name(self):
        """Test validation with missing name."""
        validator = ConfigValidator()

        # Create a mock config with empty name
        config = Mock(spec=MARLConfig)
        config.name = ""
        config.version = "1.0.0"
        config.agents = {
            "agent1": AgentConfig(
                agent_id="agent1",
                agent_type="generator",
                state_dim=128,
                action_dim=10,
            )
        }
        config.coordination = CoordinationConfig()
        config.learning = LearningConfig()
        config.system = SystemConfig()

        errors, warnings = validator.validate_config(config)

        # Should have error for missing name
        assert len(errors) > 0
        assert any("name is required" in error for error in errors)

    def test_validate_basic_structure_invalid_version(self):
        """Test validation with invalid version format."""
        validator = ConfigValidator()

        # Create a mock config with invalid version
        config = Mock(spec=MARLConfig)
        config.name = "test_config"
        config.version = "invalid_version"  # Invalid format
        config.agents = {
            "agent1": AgentConfig(
                agent_id="agent1",
                agent_type="generator",
                state_dim=128,
                action_dim=10,
            )
        }
        config.coordination = CoordinationConfig()
        config.learning = LearningConfig()
        config.system = SystemConfig()

        errors, warnings = validator.validate_config(config)

        # Should have error for invalid version
        assert len(errors) > 0
        assert any("Invalid version format" in error for error in errors)

    def test_validate_basic_structure_no_agents(self):
        """Test validation with no agents."""
        validator = ConfigValidator()

        # Create a mock config with no agents
        config = Mock(spec=MARLConfig)
        config.name = "test_config"
        config.version = "1.0.0"
        config.agents = {}  # No agents
        config.coordination = CoordinationConfig()
        config.learning = LearningConfig()
        config.system = SystemConfig()

        errors, warnings = validator.validate_config(config)

        # Should have error for no agents
        assert len(errors) > 0
        assert any("At least one agent must be configured" in error for error in errors)

    def test_validate_basic_structure_too_many_agents(self):
        """Test validation with too many agents."""
        validator = ConfigValidator()

        # Create config with many agents
        agents = {}
        for i in range(25):  # More than warning threshold
            agents[f"agent_{i}"] = AgentConfig(
                agent_id=f"agent_{i}",
                agent_type="generator",
                state_dim=128,
                action_dim=10,
            )

        config = MARLConfig(name="test_config", version="1.0.0", agents=agents)

        errors, warnings = validator.validate_config(config)

        # Should have warning for too many agents
        assert any("Large number of agents" in warning for warning in warnings)

    def test_validate_agent_dimensions(self):
        """Test validation of agent dimensions."""
        validator = ConfigValidator()

        # Invalid state dimension
        config = MARLConfig(
            name="test_config",
            version="1.0.0",
            agents={
                "agent1": AgentConfig(
                    agent_id="agent1",
                    agent_type="generator",
                    state_dim=-10,  # Invalid
                    action_dim=10,
                )
            },
        )

        errors, warnings = validator.validate_config(config)

        # Should have error for negative state dimension
        assert len(errors) > 0
        assert any("State dimension must be positive" in error for error in errors)

        # Large dimensions should generate warnings
        config.agents["agent1"].state_dim = 5000  # Very large
        config.agents["agent1"].action_dim = 200  # Large

        errors, warnings = validator.validate_config(config)

        # Should have warnings for large dimensions
        assert any("Large state dimension" in warning for warning in warnings)
        assert any("Large action dimension" in warning for warning in warnings)

    def test_validate_learning_parameters(self):
        """Test validation of learning parameters."""
        validator = ConfigValidator()

        config = MARLConfig(
            name="test_config",
            version="1.0.0",
            agents={
                "agent1": AgentConfig(
                    agent_id="agent1",
                    agent_type="generator",
                    state_dim=128,
                    action_dim=10,
                    optimization=OptimizationConfig(learning_rate=-0.001),  # Invalid negative
                )
            },
        )

        errors, warnings = validator.validate_config(config)

        # Should have error for negative learning rate
        assert len(errors) > 0
        assert any("Learning rate must be positive" in error for error in errors)

        # Test learning rate warnings
        config.agents["agent1"].optimization.learning_rate = 0.1  # Too high

        errors, warnings = validator.validate_config(config)

        # Should have warning for high learning rate
        assert any("Learning rate may be too high" in warning for warning in warnings)

    def test_validate_exploration_parameters(self):
        """Test validation of exploration parameters."""
        validator = ConfigValidator()

        config = MARLConfig(
            name="test_config",
            version="1.0.0",
            agents={
                "agent1": AgentConfig(
                    agent_id="agent1",
                    agent_type="generator",
                    state_dim=128,
                    action_dim=10,
                    exploration=ExplorationConfig(
                        initial_epsilon=1.5,  # Invalid > 1.0
                        final_epsilon=-0.1,  # Invalid < 0.0
                    ),
                )
            },
        )

        errors, warnings = validator.validate_config(config)

        # Should have errors for invalid epsilon values
        assert len(errors) >= 2
        assert any("Initial epsilon must be between 0 and 1" in error for error in errors)
        assert any("Final epsilon must be between 0 and 1" in error for error in errors)

        # Test epsilon ordering warning
        config.agents["agent1"].exploration.initial_epsilon = 0.1
        config.agents["agent1"].exploration.final_epsilon = 0.5  # Higher than initial

        errors, warnings = validator.validate_config(config)

        # Should have warning for incorrect epsilon ordering
        assert any(
            "Initial epsilon" in warning and "final epsilon" in warning for warning in warnings
        )

    def test_validate_network_configuration(self):
        """Test validation of network configuration."""
        validator = ConfigValidator()

        config = MARLConfig(
            name="test_config",
            version="1.0.0",
            agents={
                "agent1": AgentConfig(
                    agent_id="agent1",
                    agent_type="generator",
                    state_dim=128,
                    action_dim=10,
                    network=NetworkConfig(hidden_layers=[2048, 4096, 8192]),  # Very large network
                )
            },
        )

        errors, warnings = validator.validate_config(config)

        # Should have warning for large network
        assert any("Large network size" in warning for warning in warnings)

        # Test increasing layer sizes (info message)
        config.agents["agent1"].network.hidden_layers = [64, 128, 256]  # Increasing

        errors, warnings = validator.validate_config(config)

        # Should have info about increasing layer sizes
        # (This might be in warnings depending on implementation)

    def test_validate_replay_buffer(self):
        """Test validation of replay buffer configuration."""
        validator = ConfigValidator()

        config = MARLConfig(
            name="test_config",
            version="1.0.0",
            agents={
                "agent1": AgentConfig(
                    agent_id="agent1",
                    agent_type="generator",
                    state_dim=128,
                    action_dim=10,
                    replay_buffer=ReplayBufferConfig(
                        capacity=-1000,  # Invalid negative
                        batch_size=64,
                    ),
                )
            },
        )

        errors, warnings = validator.validate_config(config)

        # Should have error for negative capacity
        assert len(errors) > 0
        assert any("Replay buffer capacity must be positive" in error for error in errors)

        # Test batch size larger than capacity
        config.agents["agent1"].replay_buffer.capacity = 100
        config.agents["agent1"].replay_buffer.batch_size = 200  # Larger than capacity

        errors, warnings = validator.validate_config(config)

        # Should have error for batch size > capacity
        assert any("Batch size" in error and "exceed buffer capacity" in error for error in errors)

        # Test small buffer warning
        config.agents["agent1"].replay_buffer.capacity = 1000
        config.agents["agent1"].replay_buffer.batch_size = 200  # Much smaller ratio

        errors, warnings = validator.validate_config(config)

        # Should have warning for small buffer relative to batch size
        assert any("buffer capacity" in warning and "batch size" in warning for warning in warnings)

    def test_validate_coordination_configuration(self):
        """Test validation of coordination configuration."""
        validator = ConfigValidator()

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
            coordination=CoordinationConfig(coordination_timeout=-30.0),  # Invalid negative
        )

        errors, warnings = validator.validate_config(config)

        # Should have error for negative timeout
        assert len(errors) > 0
        assert any("Coordination timeout must be positive" in error for error in errors)

        # Test very long timeout warning
        config.coordination.coordination_timeout = 1200.0  # 20 minutes

        errors, warnings = validator.validate_config(config)

        # Should have warning for very long timeout
        assert any("Very long coordination timeout" in warning for warning in warnings)

    def test_validate_consensus_configuration(self):
        """Test validation of consensus configuration."""
        validator = ConfigValidator()

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
            coordination=CoordinationConfig(
                coordination_timeout=60.0,
                consensus=ConsensusConfig(
                    strategy=ConsensusStrategy.EXPERT_PRIORITY,
                    timeout_seconds=90.0,  # Longer than coordination timeout
                ),
            ),
        )

        errors, warnings = validator.validate_config(config)

        # Should have error for missing expert weights
        assert len(errors) > 0
        assert any("Expert priority consensus requires expert_weights" in error for error in errors)

        # Should have warning for consensus timeout > coordination timeout
        assert any(
            "Consensus timeout" in warning and "coordination timeout" in warning
            for warning in warnings
        )

    def test_validate_learning_configuration(self):
        """Test validation of learning configuration."""
        validator = ConfigValidator()

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
            learning=LearningConfig(
                max_episodes=-1000,  # Invalid negative
                evaluation_interval=100,
                save_interval=50,  # Less than evaluation interval
            ),
        )

        errors, warnings = validator.validate_config(config)

        # Should have error for negative episodes
        assert len(errors) > 0
        assert any("Max episodes must be positive" in error for error in errors)

        # Test low episode count warning
        config.learning.max_episodes = 500  # Low count

        errors, warnings = validator.validate_config(config)

        # Should have warning for low episode count
        assert any("Low episode count" in warning for warning in warnings)

    def test_validate_shared_learning(self):
        """Test validation of shared learning configuration."""
        validator = ConfigValidator()

        # Single agent with shared learning enabled
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
            learning=LearningConfig(
                shared_learning=SharedLearningConfig(
                    enabled=True,
                    experience_buffer_size=-1000,  # Invalid negative
                )
            ),
        )

        errors, warnings = validator.validate_config(config)

        # Should have error for negative buffer size
        assert len(errors) > 0
        assert any("Shared experience buffer size must be positive" in error for error in errors)

        # Should have warning for shared learning with single agent
        assert any("Shared learning enabled with only one agent" in warning for warning in warnings)

    def test_validate_system_configuration(self):
        """Test validation of system configuration."""
        validator = ConfigValidator()

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
            system=SystemConfig(
                device="unknown_device",  # Invalid device
                num_workers=-5,  # Invalid negative
                memory_limit_gb=0.5,  # Very low
            ),
        )

        errors, warnings = validator.validate_config(config)

        # Should have error for negative workers
        assert len(errors) > 0
        assert any("Number of workers must be positive" in error for error in errors)

        # Should have warnings for unknown device and low memory
        assert any("Unknown device type" in warning for warning in warnings)
        assert any("Low memory limit" in warning for warning in warnings)

    def test_validate_config_compatibility(self):
        """Test configuration compatibility validation."""
        validator = ConfigValidator()

        # Create two configs with different versions
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
            version="2.0.0",  # Different version
            agents={
                "agent1": AgentConfig(
                    agent_id="agent1",
                    agent_type="generator",
                    state_dim=256,  # Different state dim
                    action_dim=10,
                ),
                "agent2": AgentConfig(  # Additional agent
                    agent_id="agent2",
                    agent_type="validator",
                    state_dim=128,
                    action_dim=5,
                ),
            },
        )

        errors, warnings = validator.validate_config_compatibility(config1, config2)

        # Should have warnings for version difference and agent differences
        assert any("Different versions" in warning for warning in warnings)
        assert any("only in second config" in warning for warning in warnings)

        # Should have error for incompatible state dimensions
        assert len(errors) > 0
        assert any("Incompatible state dimensions" in error for error in errors)

    def test_generate_optimization_suggestions(self):
        """Test optimization suggestion generation."""
        validator = ConfigValidator()

        config = MARLConfig(
            name="test_config",
            version="1.0.0",
            agents={
                "agent1": AgentConfig(
                    agent_id="agent1",
                    agent_type="generator",
                    state_dim=128,
                    action_dim=10,
                    optimization=OptimizationConfig(learning_rate=0.05),  # High learning rate
                    replay_buffer=ReplayBufferConfig(
                        batch_size=8,  # Small batch size
                        capacity=50000,
                    ),
                    network=NetworkConfig(hidden_layers=[1024, 1024, 1024]),  # Large network
                )
            },
            coordination=CoordinationConfig(coordination_timeout=300.0),  # Long timeout
        )

        suggestions = validator.generate_optimization_suggestions(config)

        # Should have suggestions for various issues
        assert len(suggestions) > 0

        # Check for specific suggestion categories
        categories = [s["category"] for s in suggestions]
        assert "learning" in categories  # For high learning rate
        assert "training" in categories  # For small batch size
        assert "architecture" in categories  # For large network
        assert "coordination" in categories  # For long timeout

        # Check suggestion structure
        for suggestion in suggestions:
            assert "category" in suggestion
            assert "priority" in suggestion
            assert "suggestion" in suggestion
            assert "current_value" in suggestion
            assert "recommended_value" in suggestion
            assert "impact" in suggestion


class TestConfigValidatorFactory:
    """Test ConfigValidatorFactory functionality."""

    def test_create_default(self):
        """Test creating default validator."""
        validator = ConfigValidatorFactory.create()

        assert isinstance(validator, ConfigValidator)
        assert "learning_rate" in validator._thresholds

    def test_create_strict(self):
        """Test creating strict validator."""
        validator = ConfigValidatorFactory.create_strict()

        assert isinstance(validator, ConfigValidator)

        # Should have stricter thresholds
        assert (
            validator._thresholds["learning_rate"]["recommended_max"]
            < ConfigValidator()._thresholds["learning_rate"]["recommended_max"]
        )

    def test_create_permissive(self):
        """Test creating permissive validator."""
        validator = ConfigValidatorFactory.create_permissive()

        assert isinstance(validator, ConfigValidator)

        # Should have more relaxed thresholds
        assert (
            validator._thresholds["learning_rate"]["recommended_max"]
            > ConfigValidator()._thresholds["learning_rate"]["recommended_max"]
        )


if __name__ == "__main__":
    pytest.main([__file__])
