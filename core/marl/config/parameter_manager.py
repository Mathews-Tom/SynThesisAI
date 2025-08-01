"""MARL Parameter Manager.

This module provides comprehensive parameter management for MARL systems,
including parameter validation, optimization, and runtime adjustment.
"""

# Standard Library
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Third-Party Library
import yaml

# SynThesisAI Modules
from core.marl.config.config_schema import MARLConfig
from utils.logging_config import get_logger


class ParameterType(Enum):
    """Defines the types of parameters that can be managed."""

    LEARNING_RATE = "learning_rate"
    EPSILON = "epsilon"
    GAMMA = "gamma"
    TAU = "tau"
    BATCH_SIZE = "batch_size"
    BUFFER_SIZE = "buffer_size"
    NETWORK_SIZE = "network_size"
    UPDATE_FREQUENCY = "update_frequency"
    COORDINATION_TIMEOUT = "coordination_timeout"
    CONSENSUS_THRESHOLD = "consensus_threshold"
    EXPLORATION_DECAY = "exploration_decay"
    REWARD_SCALING = "reward_scaling"


class ParameterConstraint(Enum):
    """Defines the types of parameter constraints."""

    RANGE = "range"
    DISCRETE = "discrete"
    POWER_OF_TWO = "power_of_two"
    POSITIVE = "positive"
    PROBABILITY = "probability"
    INTEGER = "integer"


@dataclass
class ParameterSpec:
    """Specification for a configurable parameter.

    Defines the parameter's type, constraints, default value,
    and optimization properties.

    Attributes:
        name: The name of the parameter.
        param_type: The type of the parameter.
        default_value: The default value for the parameter.
        min_value: The minimum allowed value.
        max_value: The maximum allowed value.
        constraints: A list of constraints the parameter must adhere to.
        discrete_values: A list of allowed discrete values.
        description: A description of the parameter.
        optimization_priority: The priority for optimization (1=low, 5=high).
        affects_performance: Whether the parameter affects system performance.
        requires_restart: Whether changing the parameter requires a system restart.
        validation_function: An optional custom validation function.
    """

    name: str
    param_type: ParameterType
    default_value: Any
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    constraints: List[ParameterConstraint] = field(default_factory=list)
    discrete_values: Optional[List[Any]] = None
    description: str = ""
    optimization_priority: int = 1
    affects_performance: bool = True
    requires_restart: bool = False
    validation_function: Optional[Callable[[Any], Tuple[bool, Optional[str]]]] = None

    def __post_init__(self):
        """Validates the parameter specification after initialization."""
        if self.min_value is not None and self.max_value is not None:
            if self.min_value >= self.max_value:
                raise ValueError(f"min_value must be less than max_value for {self.name}")

        if ParameterConstraint.DISCRETE in self.constraints and not self.discrete_values:
            raise ValueError(f"discrete_values required for discrete parameter {self.name}")

    def validate_value(self, value: Any) -> Tuple[bool, Optional[str]]:
        """Validates a parameter value against its constraints.

        Args:
            value: The value to validate.

        Returns:
            A tuple containing a boolean indicating validity and an optional
            error message.
        """
        try:
            # Type-specific validation
            if self.param_type in [
                ParameterType.LEARNING_RATE,
                ParameterType.EPSILON,
                ParameterType.GAMMA,
                ParameterType.TAU,
            ]:
                if not isinstance(value, (int, float)):
                    return False, f"Parameter {self.name} must be numeric."

            # Constraint validation
            for constraint in self.constraints:
                if constraint == ParameterConstraint.RANGE:
                    if self.min_value is not None and value < self.min_value:
                        return (
                            False,
                            f"Parameter {self.name} below minimum {self.min_value}.",
                        )
                    if self.max_value is not None and value > self.max_value:
                        return (
                            False,
                            f"Parameter {self.name} above maximum {self.max_value}.",
                        )

                elif constraint == ParameterConstraint.DISCRETE:
                    if self.discrete_values and value not in self.discrete_values:
                        return (
                            False,
                            f"Parameter {self.name} must be one of {self.discrete_values}.",
                        )

                elif constraint == ParameterConstraint.POWER_OF_TWO:
                    if not isinstance(value, int) or value <= 0 or (value & (value - 1)) != 0:
                        return False, f"Parameter {self.name} must be a power of 2."

                elif constraint == ParameterConstraint.POSITIVE:
                    if value <= 0:
                        return False, f"Parameter {self.name} must be positive."

                elif constraint == ParameterConstraint.PROBABILITY:
                    if not (0.0 <= value <= 1.0):
                        return False, f"Parameter {self.name} must be between 0 and 1."

                elif constraint == ParameterConstraint.INTEGER:
                    if not isinstance(value, int):
                        return False, f"Parameter {self.name} must be an integer."

            # Custom validation function
            if self.validation_function:
                is_valid, error = self.validation_function(value)
                if not is_valid:
                    return False, error

            return True, None

        except Exception as e:
            return False, f"Validation error for {self.name}: {e}"

    def suggest_value(self, current_performance: float, target_performance: float) -> Any:
        """Suggests a parameter value based on performance metrics.

        Args:
            current_performance: The current system performance (0-1).
            target_performance: The target performance (0-1).

        Returns:
            A suggested parameter value.
        """
        if not self.affects_performance:
            return self.default_value

        performance_gap = target_performance - current_performance

        # Parameter-specific suggestions
        if self.param_type == ParameterType.LEARNING_RATE:
            if performance_gap > 0.1:  # Need faster learning
                return min(self.default_value * 1.5, self.max_value or float("inf"))
            elif performance_gap < -0.05:  # Learning too fast, unstable
                return max(self.default_value * 0.7, self.min_value or 0)

        elif self.param_type == ParameterType.EPSILON:
            if performance_gap > 0.1:  # Need more exploration
                return min(self.default_value * 1.2, self.max_value or 1.0)
            elif performance_gap < -0.05:  # Too much exploration
                return max(self.default_value * 0.8, self.min_value or 0)

        elif self.param_type == ParameterType.BATCH_SIZE:
            if performance_gap > 0.1:  # Try larger batches for stability
                current_batch = self.default_value
                if ParameterConstraint.POWER_OF_TWO in self.constraints:
                    return min(current_batch * 2, self.max_value or 1024)
                else:
                    return min(int(current_batch * 1.5), self.max_value or 1024)

        return self.default_value


class ParameterRegistry:
    """Registry of all configurable MARL parameters.

    Maintains specifications for all parameters that can be
    configured and optimized in the MARL system.
    """

    def __init__(self):
        """Initializes the parameter registry."""
        self.logger = get_logger(__name__)
        self._parameters: Dict[str, ParameterSpec] = {}
        self._parameter_groups: Dict[str, List[str]] = {}

        self._register_default_parameters()

        self.logger.info("Parameter registry initialized with %d parameters", len(self._parameters))

    def _register_default_parameters(self):
        """Registers default MARL parameters."""
        # Learning parameters
        self.register_parameter(
            ParameterSpec(
                name="learning_rate",
                param_type=ParameterType.LEARNING_RATE,
                default_value=0.001,
                min_value=1e-6,
                max_value=0.1,
                constraints=[ParameterConstraint.RANGE, ParameterConstraint.POSITIVE],
                description="Learning rate for neural network optimization.",
                optimization_priority=5,
            )
        )
        self.register_parameter(
            ParameterSpec(
                name="gamma",
                param_type=ParameterType.GAMMA,
                default_value=0.99,
                min_value=0.0,
                max_value=1.0,
                constraints=[
                    ParameterConstraint.RANGE,
                    ParameterConstraint.PROBABILITY,
                ],
                description="Discount factor for future rewards.",
                optimization_priority=4,
            )
        )
        self.register_parameter(
            ParameterSpec(
                name="tau",
                param_type=ParameterType.TAU,
                default_value=0.005,
                min_value=0.001,
                max_value=0.1,
                constraints=[ParameterConstraint.RANGE, ParameterConstraint.POSITIVE],
                description="Soft update rate for target networks.",
                optimization_priority=3,
            )
        )

        # Exploration parameters
        self.register_parameter(
            ParameterSpec(
                name="initial_epsilon",
                param_type=ParameterType.EPSILON,
                default_value=1.0,
                min_value=0.0,
                max_value=1.0,
                constraints=[
                    ParameterConstraint.RANGE,
                    ParameterConstraint.PROBABILITY,
                ],
                description="Initial exploration rate.",
                optimization_priority=4,
            )
        )
        self.register_parameter(
            ParameterSpec(
                name="final_epsilon",
                param_type=ParameterType.EPSILON,
                default_value=0.01,
                min_value=0.0,
                max_value=1.0,
                constraints=[
                    ParameterConstraint.RANGE,
                    ParameterConstraint.PROBABILITY,
                ],
                description="Final exploration rate.",
                optimization_priority=3,
            )
        )
        self.register_parameter(
            ParameterSpec(
                name="epsilon_decay",
                param_type=ParameterType.EXPLORATION_DECAY,
                default_value=0.995,
                min_value=0.9,
                max_value=0.9999,
                constraints=[ParameterConstraint.RANGE],
                description="Exploration decay rate.",
                optimization_priority=3,
            )
        )

        # Network parameters
        self.register_parameter(
            ParameterSpec(
                name="batch_size",
                param_type=ParameterType.BATCH_SIZE,
                default_value=32,
                min_value=8,
                max_value=512,
                constraints=[
                    ParameterConstraint.RANGE,
                    ParameterConstraint.INTEGER,
                    ParameterConstraint.POWER_OF_TWO,
                ],
                description="Training batch size.",
                optimization_priority=4,
            )
        )
        self.register_parameter(
            ParameterSpec(
                name="replay_buffer_capacity",
                param_type=ParameterType.BUFFER_SIZE,
                default_value=50000,
                min_value=1000,
                max_value=1000000,
                constraints=[
                    ParameterConstraint.RANGE,
                    ParameterConstraint.INTEGER,
                    ParameterConstraint.POSITIVE,
                ],
                description="Replay buffer capacity.",
                optimization_priority=2,
                requires_restart=True,
            )
        )
        self.register_parameter(
            ParameterSpec(
                name="hidden_layer_size",
                param_type=ParameterType.NETWORK_SIZE,
                default_value=256,
                discrete_values=[64, 128, 256, 512, 1024],
                constraints=[
                    ParameterConstraint.DISCRETE,
                    ParameterConstraint.POSITIVE,
                ],
                description="Hidden layer size for neural networks.",
                optimization_priority=3,
                requires_restart=True,
            )
        )
        self.register_parameter(
            ParameterSpec(
                name="update_frequency",
                param_type=ParameterType.UPDATE_FREQUENCY,
                default_value=4,
                min_value=1,
                max_value=32,
                constraints=[
                    ParameterConstraint.RANGE,
                    ParameterConstraint.INTEGER,
                    ParameterConstraint.POSITIVE,
                ],
                description="Frequency of network updates.",
                optimization_priority=3,
            )
        )

        # Coordination parameters
        self.register_parameter(
            ParameterSpec(
                name="coordination_timeout",
                param_type=ParameterType.COORDINATION_TIMEOUT,
                default_value=60.0,
                min_value=5.0,
                max_value=300.0,
                constraints=[ParameterConstraint.RANGE, ParameterConstraint.POSITIVE],
                description="Timeout for coordination attempts.",
                optimization_priority=2,
                affects_performance=False,
            )
        )
        self.register_parameter(
            ParameterSpec(
                name="consensus_threshold",
                param_type=ParameterType.CONSENSUS_THRESHOLD,
                default_value=0.6,
                min_value=0.5,
                max_value=1.0,
                constraints=[
                    ParameterConstraint.RANGE,
                    ParameterConstraint.PROBABILITY,
                ],
                description="Threshold for consensus agreement.",
                optimization_priority=3,
            )
        )

        # Reward parameters
        self.register_parameter(
            ParameterSpec(
                name="reward_scaling",
                param_type=ParameterType.REWARD_SCALING,
                default_value=1.0,
                min_value=0.1,
                max_value=10.0,
                constraints=[ParameterConstraint.RANGE, ParameterConstraint.POSITIVE],
                description="Scaling factor for rewards.",
                optimization_priority=3,
            )
        )

        # Create parameter groups
        self._parameter_groups = {
            "learning": ["learning_rate", "gamma", "tau"],
            "exploration": ["initial_epsilon", "final_epsilon", "epsilon_decay"],
            "network": [
                "batch_size",
                "replay_buffer_capacity",
                "hidden_layer_size",
                "update_frequency",
            ],
            "coordination": ["coordination_timeout", "consensus_threshold"],
            "reward": ["reward_scaling"],
            "performance_critical": [
                "learning_rate",
                "gamma",
                "initial_epsilon",
                "batch_size",
            ],
            "requires_restart": ["replay_buffer_capacity", "hidden_layer_size"],
        }

    def register_parameter(self, param_spec: ParameterSpec):
        """Registers a parameter specification.

        Args:
            param_spec: The parameter specification to register.
        """
        self._parameters[param_spec.name] = param_spec
        self.logger.debug("Registered parameter: %s", param_spec.name)

    def get_parameter(self, name: str) -> Optional[ParameterSpec]:
        """Gets a parameter specification by name.

        Args:
            name: The name of the parameter.

        Returns:
            The parameter specification or None if not found.
        """
        return self._parameters.get(name)

    def get_parameters_by_group(self, group: str) -> List[ParameterSpec]:
        """Gets parameters by group name.

        Args:
            group: The name of the group.

        Returns:
            A list of parameter specifications in the group.
        """
        if group not in self._parameter_groups:
            return []

        return [
            self._parameters[name]
            for name in self._parameter_groups[group]
            if name in self._parameters
        ]

    def get_all_parameters(self) -> Dict[str, ParameterSpec]:
        """Gets all registered parameters.

        Returns:
            A dictionary of all registered parameters.
        """
        return self._parameters.copy()

    def get_parameter_groups(self) -> Dict[str, List[str]]:
        """Gets all parameter groups.

        Returns:
            A dictionary of all parameter groups.
        """
        return self._parameter_groups.copy()

    def validate_parameter_set(self, parameters: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validates a set of parameters.

        Args:
            parameters: A dictionary of parameter values.

        Returns:
            A tuple containing a boolean indicating validity and a list of
            error messages.
        """
        errors = []
        for name, value in parameters.items():
            param_spec = self.get_parameter(name)
            if not param_spec:
                errors.append(f"Unknown parameter: {name}")
                continue

            is_valid, error = param_spec.validate_value(value)
            if not is_valid and error:
                errors.append(error)

        return not errors, errors

    def suggest_parameter_values(
        self,
        current_performance: float,
        target_performance: float,
        focus_groups: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Suggests parameter values based on performance metrics.

        Args:
            current_performance: The current system performance (0-1).
            target_performance: The target performance (0-1).
            focus_groups: A list of parameter groups to focus on.

        Returns:
            A dictionary of suggested parameter values.
        """
        suggestions = {}
        parameters_to_consider: List[ParameterSpec] = []

        if focus_groups:
            for group in focus_groups:
                parameters_to_consider.extend(self.get_parameters_by_group(group))
        else:
            parameters_to_consider = list(self._parameters.values())

        # Sort by optimization priority (higher priority first)
        parameters_to_consider.sort(key=lambda p: p.optimization_priority, reverse=True)

        for param_spec in parameters_to_consider:
            if param_spec.affects_performance:
                suggested_value = param_spec.suggest_value(current_performance, target_performance)
                suggestions[param_spec.name] = suggested_value

        return suggestions


class MARLParameterManager:
    """Comprehensive parameter manager for MARL systems.

    Provides parameter validation, optimization, runtime adjustment,
    and configuration management capabilities.
    """

    def __init__(self, config: Optional[MARLConfig] = None):
        """Initializes the parameter manager.

        Args:
            config: An optional MARL configuration to initialize with.
        """
        self.logger = get_logger(__name__)
        self.registry = ParameterRegistry()
        self._current_parameters: Dict[str, Any] = {}
        self._parameter_history: List[Dict[str, Any]] = []
        self._performance_history: List[float] = []

        if config:
            self.load_from_config(config)
        else:
            self._load_default_parameters()

        self.logger.info(
            "MARL parameter manager initialized with %d parameters.",
            len(self._current_parameters),
        )

    def _load_default_parameters(self):
        """Loads default parameter values from the registry."""
        for param_spec in self.registry.get_all_parameters().values():
            self._current_parameters[param_spec.name] = param_spec.default_value

    def load_from_config(self, config: MARLConfig):
        """Loads parameters from a MARL configuration object.

        Args:
            config: The MARL configuration to load from.
        """
        self.logger.info("Loading parameters from configuration: %s", config.name)
        extracted_params = self._extract_parameters_from_config(config)
        is_valid, errors = self.set_parameters(extracted_params, validate=True)

        if not is_valid:
            self.logger.warning("Configuration parameter validation errors: %s", "; ".join(errors))
            self._load_default_parameters()

        self.logger.info("Loaded %d parameters from configuration.", len(self._current_parameters))

    def _extract_parameters_from_config(self, config: MARLConfig) -> Dict[str, Any]:
        """Extracts parameter values from a configuration structure.

        Args:
            config: The MARL configuration object.

        Returns:
            A dictionary of extracted parameter values.
        """
        params = {}
        if not config.agents:
            return params

        first_agent = next(iter(config.agents.values()))

        if hasattr(first_agent, "optimization"):
            params["learning_rate"] = first_agent.optimization.learning_rate
        params["gamma"] = getattr(first_agent, "gamma", 0.99)
        params["tau"] = getattr(first_agent, "tau", 0.005)

        if hasattr(first_agent, "exploration"):
            params["initial_epsilon"] = first_agent.exploration.initial_epsilon
            params["final_epsilon"] = first_agent.exploration.final_epsilon
            params["epsilon_decay"] = first_agent.exploration.epsilon_decay

        if hasattr(first_agent, "replay_buffer"):
            params["batch_size"] = first_agent.replay_buffer.batch_size
            params["replay_buffer_capacity"] = first_agent.replay_buffer.capacity

        if hasattr(first_agent, "network") and first_agent.network.hidden_layers:
            params["hidden_layer_size"] = first_agent.network.hidden_layers[0]

        params["update_frequency"] = getattr(first_agent, "update_frequency", 4)
        params["reward_scaling"] = getattr(first_agent, "reward_scaling", 1.0)

        if hasattr(config, "coordination"):
            params["coordination_timeout"] = config.coordination.coordination_timeout
            if hasattr(config.coordination, "consensus"):
                params["consensus_threshold"] = getattr(
                    config.coordination.consensus, "threshold", 0.6
                )

        return params

    def get_parameter(self, name: str) -> Any:
        """Gets the current value of a parameter.

        Args:
            name: The name of the parameter.

        Returns:
            The value of the parameter.

        Raises:
            KeyError: If the parameter is not found.
        """
        if name not in self._current_parameters:
            raise KeyError(f"Parameter not found: {name}")
        return self._current_parameters[name]

    def set_parameter(
        self, name: str, value: Any, validate: bool = True
    ) -> Tuple[bool, Optional[str]]:
        """Sets the value of a parameter.

        Args:
            name: The name of the parameter.
            value: The new value for the parameter.
            validate: Whether to validate the value before setting.

        Returns:
            A tuple indicating success and an optional error message.
        """
        if validate:
            param_spec = self.registry.get_parameter(name)
            if not param_spec:
                return False, f"Unknown parameter: {name}"

            is_valid, error = param_spec.validate_value(value)
            if not is_valid:
                return False, error

        old_value = self._current_parameters.get(name)
        self._current_parameters[name] = value

        self._parameter_history.append(
            {
                "parameter": name,
                "old_value": old_value,
                "new_value": value,
                "timestamp": self._get_timestamp(),
            }
        )
        self.logger.debug("Set parameter %s: %s -> %s", name, old_value, value)
        return True, None

    def set_parameters(
        self, parameters: Dict[str, Any], validate: bool = True
    ) -> Tuple[bool, List[str]]:
        """Sets multiple parameters.

        Args:
            parameters: A dictionary of parameter values.
            validate: Whether to validate values before setting.

        Returns:
            A tuple indicating success and a list of error messages.
        """
        errors = []
        for name, value in parameters.items():
            success, error = self.set_parameter(name, value, validate)
            if not success and error:
                errors.append(error)
        return not errors, errors

    def get_all_parameters(self) -> Dict[str, Any]:
        """Gets all current parameter values.

        Returns:
            A dictionary of all current parameter values.
        """
        return self._current_parameters.copy()

    def get_parameters_by_group(self, group: str) -> Dict[str, Any]:
        """Gets parameters by group.

        Args:
            group: The name of the parameter group.

        Returns:
            A dictionary of parameter values in the group.
        """
        group_params = {}
        param_names = self.registry.get_parameter_groups().get(group, [])
        for name in param_names:
            if name in self._current_parameters:
                group_params[name] = self._current_parameters[name]
        return group_params

    def validate_current_parameters(self) -> Tuple[bool, List[str]]:
        """Validates all current parameters.

        Returns:
            A tuple indicating validity and a list of error messages.
        """
        return self.registry.validate_parameter_set(self._current_parameters)

    def optimize_parameters(
        self,
        current_performance: float,
        target_performance: float = 0.9,
        focus_groups: Optional[List[str]] = None,
        apply_suggestions: bool = False,
    ) -> Dict[str, Any]:
        """Optimizes parameters based on performance metrics.

        Args:
            current_performance: The current system performance (0-1).
            target_performance: The target performance (0-1).
            focus_groups: A list of parameter groups to focus on.
            apply_suggestions: Whether to apply suggestions automatically.

        Returns:
            A dictionary of suggested parameter values.
        """
        self.logger.info(
            "Optimizing parameters: current=%.3f, target=%.3f",
            current_performance,
            target_performance,
        )
        self._performance_history.append(current_performance)
        suggestions = self.registry.suggest_parameter_values(
            current_performance, target_performance, focus_groups
        )

        if apply_suggestions:
            success, errors = self.set_parameters(suggestions, validate=True)
            if not success:
                self.logger.warning(
                    "Failed to apply some parameter suggestions: %s", "; ".join(errors)
                )

        self.logger.info("Generated %d parameter suggestions.", len(suggestions))
        return suggestions

    def reset_to_defaults(self, parameter_names: Optional[List[str]] = None):
        """Resets parameters to their default values.

        Args:
            parameter_names: A list of specific parameters to reset. If None,
                all parameters are reset.
        """
        names_to_reset = parameter_names or list(self._current_parameters.keys())
        for name in names_to_reset:
            param_spec = self.registry.get_parameter(name)
            if param_spec:
                self.set_parameter(name, param_spec.default_value, validate=False)
        self.logger.info("Reset %d parameters to defaults.", len(names_to_reset))

    def get_parameter_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Gets comprehensive information about a parameter.

        Args:
            name: The name of the parameter.

        Returns:
            A dictionary of parameter information or None if not found.
        """
        param_spec = self.registry.get_parameter(name)
        if not param_spec:
            return None

        return {
            "name": param_spec.name,
            "type": param_spec.param_type.value,
            "current_value": self._current_parameters.get(name),
            "default_value": param_spec.default_value,
            "min_value": param_spec.min_value,
            "max_value": param_spec.max_value,
            "constraints": [c.value for c in param_spec.constraints],
            "discrete_values": param_spec.discrete_values,
            "description": param_spec.description,
            "optimization_priority": param_spec.optimization_priority,
            "affects_performance": param_spec.affects_performance,
            "requires_restart": param_spec.requires_restart,
        }

    def get_parameter_history(self, name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Gets the parameter change history.

        Args:
            name: A specific parameter name to filter by. If None, returns
                history for all parameters.

        Returns:
            A list of parameter change records.
        """
        if name is None:
            return self._parameter_history.copy()
        return [record for record in self._parameter_history if record["parameter"] == name]

    def get_performance_history(self) -> List[float]:
        """Gets the performance history.

        Returns:
            A list of recorded performance values.
        """
        return self._performance_history.copy()

    def export_parameters(self, file_path: Union[str, Path], file_format: str = "json") -> bool:
        """Exports the current parameters to a file.

        Args:
            file_path: The path to the export file.
            file_format: The export format ('json' or 'yaml').

        Returns:
            True if the export was successful, False otherwise.
        """
        try:
            path = Path(file_path)
            export_data = {
                "parameters": self._current_parameters,
                "metadata": {
                    "timestamp": self._get_timestamp(),
                    "total_parameters": len(self._current_parameters),
                    "performance_history_length": len(self._performance_history),
                },
            }

            with path.open("w", encoding="utf-8") as f:
                if file_format.lower() == "json":
                    json.dump(export_data, f, indent=2, default=str)
                elif file_format.lower() == "yaml":
                    yaml.dump(export_data, f, default_flow_style=False, indent=2)
                else:
                    self.logger.error("Unsupported export format: %s", file_format)
                    return False

            self.logger.info("Exported parameters to: %s", path)
            return True
        except (IOError, ImportError) as e:
            self.logger.error("Failed to export parameters to %s: %s", file_path, e)
            return False

    def import_parameters(self, file_path: Union[str, Path], validate: bool = True) -> bool:
        """Imports parameters from a file.

        Args:
            file_path: The path to the import file.
            validate: Whether to validate the imported parameters.

        Returns:
            True if the import was successful, False otherwise.
        """
        try:
            path = Path(file_path)
            if not path.exists():
                self.logger.error("Import file not found: %s", path)
                return False

            with path.open("r", encoding="utf-8") as f:
                if path.suffix.lower() == ".json":
                    import_data = json.load(f)
                elif path.suffix.lower() in [".yaml", ".yml"]:
                    import_data = yaml.safe_load(f)
                else:
                    self.logger.error("Unsupported import format: %s", path.suffix)
                    return False

            parameters = import_data.get("parameters", {})
            success, errors = self.set_parameters(parameters, validate)

            if not success:
                self.logger.warning("Parameter import validation errors: %s", "; ".join(errors))
                return False

            self.logger.info("Imported %d parameters from: %s", len(parameters), path)
            return True
        except (IOError, ImportError, yaml.YAMLError) as e:
            self.logger.error("Failed to import parameters from %s: %s", file_path, e)
            return False

    def create_parameter_report(self) -> Dict[str, Any]:
        """Creates a comprehensive parameter report.

        Returns:
            A dictionary containing the parameter report.
        """
        is_valid, validation_errors = self.validate_current_parameters()
        param_stats = {
            "total_parameters": len(self._current_parameters),
            "parameters_by_group": {},
            "requires_restart_count": 0,
            "performance_critical_count": 0,
        }

        for group_name, param_names in self.registry.get_parameter_groups().items():
            param_stats["parameters_by_group"][group_name] = len(
                [name for name in param_names if name in self._current_parameters]
            )

        for param_spec in self.registry.get_all_parameters().values():
            if param_spec.requires_restart:
                param_stats["requires_restart_count"] += 1
            if param_spec.affects_performance:
                param_stats["performance_critical_count"] += 1

        return {
            "timestamp": self._get_timestamp(),
            "validation": {"is_valid": is_valid, "errors": validation_errors},
            "statistics": param_stats,
            "current_parameters": self._current_parameters.copy(),
            "parameter_history_length": len(self._parameter_history),
            "performance_history_length": len(self._performance_history),
            "recent_performance": (
                self._performance_history[-5:] if self._performance_history else []
            ),
        }

    def _get_timestamp(self) -> str:
        """Gets the current timestamp as an ISO 8601 formatted string.

        Returns:
            The timestamp string.
        """
        return datetime.now().isoformat()


class MARLParameterManagerFactory:
    """Factory for creating MARLParameterManager instances."""

    @staticmethod
    def create(config: Optional[MARLConfig] = None) -> MARLParameterManager:
        """Creates a MARLParameterManager instance.

        Args:
            config: An optional MARL configuration to initialize with.

        Returns:
            A new MARLParameterManager instance.
        """
        return MARLParameterManager(config)

    @staticmethod
    def create_with_custom_registry(
        registry: ParameterRegistry,
    ) -> MARLParameterManager:
        """Creates a manager with a custom parameter registry.

        Args:
            registry: A custom ParameterRegistry instance.

        Returns:
            A new MARLParameterManager instance with the custom registry.
        """
        manager = MARLParameterManager()
        manager.registry = registry
        manager._load_default_parameters()
        return manager
