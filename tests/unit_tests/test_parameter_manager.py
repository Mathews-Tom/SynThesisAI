"""
Unit tests for MARL Parameter Manager.

Tests parameter validation, optimization, and management functionality.
"""

# Standard Library
import json
import tempfile
from pathlib import Path

# Third-Party Library
import pytest

# SynThesisAI Modules
from core.marl.config.config_schema import AgentConfig, MARLConfig
from core.marl.config.parameter_manager import (
    MARLParameterManager,
    MARLParameterManagerFactory,
    ParameterConstraint,
    ParameterRegistry,
    ParameterSpec,
    ParameterType,
)


class TestParameterSpec:
    """Test ParameterSpec functionality."""

    def test_parameter_spec_creation(self):
        """Test parameter specification creation."""
        spec = ParameterSpec(
            name="learning_rate",
            param_type=ParameterType.LEARNING_RATE,
            default_value=0.001,
            min_value=1e-6,
            max_value=0.1,
            constraints=[ParameterConstraint.RANGE, ParameterConstraint.POSITIVE],
            description="Learning rate for optimization",
        )

        assert spec.name == "learning_rate"
        assert spec.param_type == ParameterType.LEARNING_RATE
        assert spec.default_value == 0.001
        assert spec.min_value == 1e-6
        assert spec.max_value == 0.1
        assert ParameterConstraint.RANGE in spec.constraints
        assert ParameterConstraint.POSITIVE in spec.constraints

    def test_parameter_spec_validation_errors(self):
        """Test parameter specification validation errors."""
        # Invalid range
        with pytest.raises(ValueError, match="min_value must be less than max_value"):
            ParameterSpec(
                name="test",
                param_type=ParameterType.LEARNING_RATE,
                default_value=0.001,
                min_value=0.1,
                max_value=0.01,
            )

        # Discrete without values
        with pytest.raises(ValueError, match="discrete_values required"):
            ParameterSpec(
                name="test",
                param_type=ParameterType.BATCH_SIZE,
                default_value=32,
                constraints=[ParameterConstraint.DISCRETE],
            )

    def test_validate_value_range(self):
        """Test value validation with range constraints."""
        spec = ParameterSpec(
            name="learning_rate",
            param_type=ParameterType.LEARNING_RATE,
            default_value=0.001,
            min_value=1e-6,
            max_value=0.1,
            constraints=[ParameterConstraint.RANGE],
        )

        # Valid values
        assert spec.validate_value(0.001)[0] is True
        assert spec.validate_value(1e-6)[0] is True
        assert spec.validate_value(0.1)[0] is True

        # Invalid values
        assert spec.validate_value(1e-7)[0] is False
        assert spec.validate_value(0.2)[0] is False

    def test_validate_value_discrete(self):
        """Test value validation with discrete constraints."""
        spec = ParameterSpec(
            name="batch_size",
            param_type=ParameterType.BATCH_SIZE,
            default_value=32,
            discrete_values=[16, 32, 64, 128],
            constraints=[ParameterConstraint.DISCRETE],
        )

        # Valid values
        assert spec.validate_value(32)[0] is True
        assert spec.validate_value(64)[0] is True

        # Invalid values
        assert spec.validate_value(48)[0] is False
        assert spec.validate_value(256)[0] is False

    def test_validate_value_power_of_two(self):
        """Test value validation with power of two constraint."""
        spec = ParameterSpec(
            name="buffer_size",
            param_type=ParameterType.BUFFER_SIZE,
            default_value=1024,
            constraints=[ParameterConstraint.POWER_OF_TWO],
        )

        # Valid values
        assert spec.validate_value(1024)[0] is True
        assert spec.validate_value(512)[0] is True
        assert spec.validate_value(2048)[0] is True

        # Invalid values
        assert spec.validate_value(1000)[0] is False
        assert spec.validate_value(0)[0] is False
        assert spec.validate_value(-1)[0] is False

    def test_validate_value_probability(self):
        """Test value validation with probability constraint."""
        spec = ParameterSpec(
            name="epsilon",
            param_type=ParameterType.EPSILON,
            default_value=0.1,
            constraints=[ParameterConstraint.PROBABILITY],
        )

        # Valid values
        assert spec.validate_value(0.0)[0] is True
        assert spec.validate_value(0.5)[0] is True
        assert spec.validate_value(1.0)[0] is True

        # Invalid values
        assert spec.validate_value(-0.1)[0] is False
        assert spec.validate_value(1.1)[0] is False

    def test_validate_value_custom_function(self):
        """Test value validation with custom function."""

        def custom_validator(value):
            if value % 2 == 0:
                return True, None
            return False, "Value must be even"

        spec = ParameterSpec(
            name="even_number",
            param_type=ParameterType.BATCH_SIZE,
            default_value=32,
            validation_function=custom_validator,
        )

        # Valid values
        assert spec.validate_value(32)[0] is True
        assert spec.validate_value(64)[0] is True

        # Invalid values
        is_valid, error = spec.validate_value(33)
        assert is_valid is False
        assert "even" in error

    def test_suggest_value_learning_rate(self):
        """Test value suggestion for learning rate."""
        spec = ParameterSpec(
            name="learning_rate",
            param_type=ParameterType.LEARNING_RATE,
            default_value=0.001,
            min_value=1e-6,
            max_value=0.1,
        )

        # Need faster learning
        suggested = spec.suggest_value(current_performance=0.5, target_performance=0.8)
        assert suggested > spec.default_value

        # Learning too fast
        suggested = spec.suggest_value(current_performance=0.8, target_performance=0.7)
        assert suggested < spec.default_value

        # Performance OK
        suggested = spec.suggest_value(current_performance=0.8, target_performance=0.82)
        assert suggested == spec.default_value

    def test_suggest_value_epsilon(self):
        """Test value suggestion for epsilon."""
        spec = ParameterSpec(
            name="epsilon",
            param_type=ParameterType.EPSILON,
            default_value=0.1,
            min_value=0.0,
            max_value=1.0,
        )

        # Need more exploration
        suggested = spec.suggest_value(current_performance=0.5, target_performance=0.8)
        assert suggested > spec.default_value

        # Too much exploration
        suggested = spec.suggest_value(current_performance=0.8, target_performance=0.7)
        assert suggested < spec.default_value


class TestParameterRegistry:
    """Test ParameterRegistry functionality."""

    def test_registry_initialization(self):
        """Test parameter registry initialization."""
        registry = ParameterRegistry()

        # Should have default parameters
        assert len(registry.get_all_parameters()) > 0

        # Should have parameter groups
        groups = registry.get_parameter_groups()
        assert "learning" in groups
        assert "exploration" in groups
        assert "network" in groups

    def test_register_parameter(self):
        """Test parameter registration."""
        registry = ParameterRegistry()

        spec = ParameterSpec(
            name="custom_param",
            param_type=ParameterType.LEARNING_RATE,
            default_value=0.005,
        )

        registry.register_parameter(spec)

        retrieved = registry.get_parameter("custom_param")
        assert retrieved is not None
        assert retrieved.name == "custom_param"
        assert retrieved.default_value == 0.005

    def test_get_parameters_by_group(self):
        """Test getting parameters by group."""
        registry = ParameterRegistry()

        learning_params = registry.get_parameters_by_group("learning")
        assert len(learning_params) > 0

        # All should be learning-related
        for param in learning_params:
            assert param.name in ["learning_rate", "gamma", "tau"]

        # Non-existent group
        empty_params = registry.get_parameters_by_group("nonexistent")
        assert len(empty_params) == 0

    def test_validate_parameter_set(self):
        """Test parameter set validation."""
        registry = ParameterRegistry()

        # Valid parameters
        valid_params = {"learning_rate": 0.001, "gamma": 0.99, "batch_size": 32}

        is_valid, errors = registry.validate_parameter_set(valid_params)
        assert is_valid is True
        assert len(errors) == 0

        # Invalid parameters
        invalid_params = {
            "learning_rate": -0.001,  # Negative
            "gamma": 1.5,  # > 1.0
            "unknown_param": 123,  # Unknown
        }

        is_valid, errors = registry.validate_parameter_set(invalid_params)
        assert is_valid is False
        assert len(errors) > 0

    def test_suggest_parameter_values(self):
        """Test parameter value suggestions."""
        registry = ParameterRegistry()

        suggestions = registry.suggest_parameter_values(
            current_performance=0.5, target_performance=0.8
        )

        assert len(suggestions) > 0
        assert "learning_rate" in suggestions
        assert "initial_epsilon" in suggestions

        # Focus on specific groups
        learning_suggestions = registry.suggest_parameter_values(
            current_performance=0.5, target_performance=0.8, focus_groups=["learning"]
        )

        # Should only contain learning parameters
        for param_name in learning_suggestions:
            param_spec = registry.get_parameter(param_name)
            assert param_spec is not None


class TestMARLParameterManager:
    """Test MARLParameterManager functionality."""

    def test_manager_initialization(self):
        """Test parameter manager initialization."""
        manager = MARLParameterManager()

        # Should have default parameters loaded
        params = manager.get_all_parameters()
        assert len(params) > 0
        assert "learning_rate" in params
        assert "gamma" in params

    def test_manager_initialization_with_config(self):
        """Test parameter manager initialization with config."""
        # Create mock config
        agent_config = AgentConfig(
            agent_id="test_agent", agent_type="generator", state_dim=128, action_dim=10
        )

        config = MARLConfig(
            name="test_config", version="1.0.0", agents={"test_agent": agent_config}
        )

        manager = MARLParameterManager(config)

        # Should have parameters from config
        params = manager.get_all_parameters()
        assert len(params) > 0

    def test_get_set_parameter(self):
        """Test getting and setting parameters."""
        manager = MARLParameterManager()

        # Get parameter
        lr = manager.get_parameter("learning_rate")
        assert lr is not None

        # Set parameter
        success, error = manager.set_parameter("learning_rate", 0.005)
        assert success is True
        assert error is None

        # Verify change
        new_lr = manager.get_parameter("learning_rate")
        assert new_lr == 0.005

        # Invalid parameter
        with pytest.raises(KeyError):
            manager.get_parameter("nonexistent")

        # Invalid value
        success, error = manager.set_parameter("learning_rate", -0.001)
        assert success is False
        assert error is not None

    def test_set_parameters_batch(self):
        """Test setting multiple parameters."""
        manager = MARLParameterManager()

        params = {"learning_rate": 0.002, "gamma": 0.95, "batch_size": 64}

        success, errors = manager.set_parameters(params)
        assert success is True
        assert len(errors) == 0

        # Verify changes
        assert manager.get_parameter("learning_rate") == 0.002
        assert manager.get_parameter("gamma") == 0.95
        assert manager.get_parameter("batch_size") == 64

    def test_get_parameters_by_group(self):
        """Test getting parameters by group."""
        manager = MARLParameterManager()

        learning_params = manager.get_parameters_by_group("learning")
        assert len(learning_params) > 0
        assert "learning_rate" in learning_params
        assert "gamma" in learning_params

    def test_validate_current_parameters(self):
        """Test current parameter validation."""
        manager = MARLParameterManager()

        # Should be valid initially
        is_valid, errors = manager.validate_current_parameters()
        assert is_valid is True
        assert len(errors) == 0

        # Make invalid change
        manager.set_parameter("learning_rate", -0.001, validate=False)

        # Should now be invalid
        is_valid, errors = manager.validate_current_parameters()
        assert is_valid is False
        assert len(errors) > 0

    def test_optimize_parameters(self):
        """Test parameter optimization."""
        manager = MARLParameterManager()

        suggestions = manager.optimize_parameters(current_performance=0.6, target_performance=0.8)

        assert len(suggestions) > 0
        assert isinstance(suggestions, dict)

        # Test with auto-apply
        suggestions = manager.optimize_parameters(
            current_performance=0.6, target_performance=0.8, apply_suggestions=True
        )

        # Parameters should have changed
        # (exact values depend on suggestion logic)
        assert len(suggestions) > 0

    def test_reset_to_defaults(self):
        """Test resetting parameters to defaults."""
        manager = MARLParameterManager()

        # Change some parameters
        manager.set_parameter("learning_rate", 0.005)
        manager.set_parameter("gamma", 0.95)

        # Reset specific parameters
        manager.reset_to_defaults(["learning_rate"])

        # Should be back to default
        default_lr = manager.registry.get_parameter("learning_rate").default_value
        assert manager.get_parameter("learning_rate") == default_lr

        # Other parameter should remain changed
        assert manager.get_parameter("gamma") == 0.95

        # Reset all
        manager.reset_to_defaults()

        # All should be defaults
        default_gamma = manager.registry.get_parameter("gamma").default_value
        assert manager.get_parameter("gamma") == default_gamma

    def test_get_parameter_info(self):
        """Test getting parameter information."""
        manager = MARLParameterManager()

        info = manager.get_parameter_info("learning_rate")
        assert info is not None
        assert info["name"] == "learning_rate"
        assert "current_value" in info
        assert "default_value" in info
        assert "description" in info

        # Non-existent parameter
        info = manager.get_parameter_info("nonexistent")
        assert info is None

    def test_parameter_history(self):
        """Test parameter change history."""
        manager = MARLParameterManager()

        # Initially empty
        history = manager.get_parameter_history()
        initial_length = len(history)

        # Make changes
        manager.set_parameter("learning_rate", 0.005)
        manager.set_parameter("gamma", 0.95)

        # History should have entries
        history = manager.get_parameter_history()
        assert len(history) == initial_length + 2

        # Get specific parameter history
        lr_history = manager.get_parameter_history("learning_rate")
        assert len(lr_history) >= 1
        assert lr_history[-1]["parameter"] == "learning_rate"
        assert lr_history[-1]["new_value"] == 0.005

    def test_performance_history(self):
        """Test performance history tracking."""
        manager = MARLParameterManager()

        # Initially empty
        history = manager.get_performance_history()
        assert len(history) == 0

        # Optimize parameters (adds to history)
        manager.optimize_parameters(0.6, 0.8)
        manager.optimize_parameters(0.7, 0.8)

        # Should have entries
        history = manager.get_performance_history()
        assert len(history) == 2
        assert history[0] == 0.6
        assert history[1] == 0.7

    def test_export_import_parameters(self):
        """Test parameter export and import."""
        manager = MARLParameterManager()

        # Change some parameters
        manager.set_parameter("learning_rate", 0.005)
        manager.set_parameter("gamma", 0.95)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            # Export
            success = manager.export_parameters(temp_path, file_format="json")
            assert success is True

            # Verify file exists and has content
            path = Path(temp_path)
            assert path.exists()

            with path.open("r") as f:
                data = json.load(f)

            assert "parameters" in data
            assert data["parameters"]["learning_rate"] == 0.005
            assert data["parameters"]["gamma"] == 0.95

            # Create new manager and import
            new_manager = MARLParameterManager()
            success = new_manager.import_parameters(temp_path)
            assert success is True

            # Verify parameters were imported
            assert new_manager.get_parameter("learning_rate") == 0.005
            assert new_manager.get_parameter("gamma") == 0.95

        finally:
            # Cleanup
            Path(temp_path).unlink(missing_ok=True)

    def test_create_parameter_report(self):
        """Test parameter report creation."""
        manager = MARLParameterManager()

        # Make some changes to create history
        manager.set_parameter("learning_rate", 0.005)
        manager.optimize_parameters(0.6, 0.8)

        report = manager.create_parameter_report()

        assert "timestamp" in report
        assert "validation" in report
        assert "statistics" in report
        assert "current_parameters" in report
        assert "parameter_history_length" in report
        assert "performance_history_length" in report

        # Validation should be included
        assert "is_valid" in report["validation"]
        assert "errors" in report["validation"]

        # Statistics should be included
        stats = report["statistics"]
        assert "total_parameters" in stats
        assert "parameters_by_group" in stats


class TestMARLParameterManagerFactory:
    """Test MARLParameterManagerFactory functionality."""

    def test_create_default(self):
        """Test creating default parameter manager."""
        manager = MARLParameterManagerFactory.create()

        assert isinstance(manager, MARLParameterManager)
        assert len(manager.get_all_parameters()) > 0

    def test_create_with_config(self):
        """Test creating parameter manager with config."""
        agent_config = AgentConfig(
            agent_id="test_agent", agent_type="generator", state_dim=128, action_dim=10
        )

        config = MARLConfig(
            name="test_config", version="1.0.0", agents={"test_agent": agent_config}
        )

        manager = MARLParameterManagerFactory.create(config)

        assert isinstance(manager, MARLParameterManager)
        assert len(manager.get_all_parameters()) > 0

    def test_create_with_custom_registry(self):
        """Test creating parameter manager with custom registry."""
        custom_registry = ParameterRegistry()

        # Add custom parameter
        custom_spec = ParameterSpec(
            name="custom_param",
            param_type=ParameterType.LEARNING_RATE,
            default_value=0.123,
        )
        custom_registry.register_parameter(custom_spec)

        manager = MARLParameterManagerFactory.create_with_custom_registry(custom_registry)

        assert isinstance(manager, MARLParameterManager)
        assert manager.get_parameter("custom_param") == 0.123
