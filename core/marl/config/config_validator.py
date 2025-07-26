"""
MARL Configuration Validator.

This module provides comprehensive validation for MARL configurations,
including schema validation, compatibility checking, and performance analysis.
"""

import re
from typing import Any, Dict, List, Optional, Tuple, Union

from utils.logging_config import get_logger

from .config_schema import (
    AgentConfig,
    ConsensusStrategy,
    ExplorationStrategy,
    MARLConfig,
)


class ValidationSeverity:
    """Validation message severity levels."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class ValidationMessage:
    """Validation message with context."""

    def __init__(
        self, severity: str, message: str, path: str = "", suggestion: str = ""
    ):
        """
        Initialize validation message.

        Args:
            severity: Message severity (error, warning, info)
            message: Validation message
            path: Configuration path where issue was found
            suggestion: Suggested fix or improvement
        """
        self.severity = severity
        self.message = message
        self.path = path
        self.suggestion = suggestion

    def __str__(self) -> str:
        """String representation of validation message."""
        path_str = f" at {self.path}" if self.path else ""
        suggestion_str = f" Suggestion: {self.suggestion}" if self.suggestion else ""
        return f"[{self.severity.upper()}]{path_str}: {self.message}{suggestion_str}"


class ConfigValidator:
    """
    Comprehensive configuration validator for MARL systems.

    Validates configuration schema, parameter ranges, compatibility,
    and provides performance optimization suggestions.
    """

    def __init__(self):
        """Initialize the configuration validator."""
        self.logger = get_logger(__name__)

        # Validation thresholds and limits
        self._thresholds = {
            "learning_rate": {
                "min": 1e-6,
                "max": 0.1,
                "recommended_min": 1e-4,
                "recommended_max": 0.01,
            },
            "epsilon": {
                "min": 0.0,
                "max": 1.0,
                "recommended_initial": 1.0,
                "recommended_final": 0.01,
            },
            "gamma": {
                "min": 0.0,
                "max": 1.0,
                "recommended_min": 0.9,
                "recommended_max": 0.999,
            },
            "batch_size": {
                "min": 1,
                "max": 1024,
                "recommended_min": 16,
                "recommended_max": 256,
            },
            "replay_buffer_capacity": {
                "min": 100,
                "max": 10000000,
                "recommended_min": 10000,
                "recommended_max": 1000000,
            },
            "coordination_timeout": {
                "min": 1.0,
                "max": 3600.0,
                "recommended_min": 30.0,
                "recommended_max": 300.0,
            },
        }

        # Performance optimization rules
        self._optimization_rules = {
            "memory_usage": {
                "replay_buffer_warning_threshold": 500000,
                "network_size_warning_threshold": 1024,
            },
            "training_stability": {
                "learning_rate_stability_threshold": 0.01,
                "epsilon_decay_stability_threshold": 0.99,
            },
            "coordination_efficiency": {
                "timeout_efficiency_threshold": 120.0,
                "max_agents_efficiency_threshold": 10,
            },
        }

        self.logger.info("Configuration validator initialized")

    def validate_config(self, config: MARLConfig) -> Tuple[List[str], List[str]]:
        """
        Validate a complete MARL configuration.

        Args:
            config: MARL configuration to validate

        Returns:
            Tuple of (error_messages, warning_messages)
        """
        messages = []

        # Basic structure validation
        messages.extend(self._validate_basic_structure(config))

        # Agent validation
        messages.extend(self._validate_agents(config))

        # Coordination validation
        messages.extend(self._validate_coordination(config))

        # Learning configuration validation
        messages.extend(self._validate_learning(config))

        # System configuration validation
        messages.extend(self._validate_system(config))

        # Cross-component compatibility validation
        messages.extend(self._validate_compatibility(config))

        # Performance optimization suggestions
        messages.extend(self._validate_performance_optimization(config))

        # Separate errors and warnings
        errors = [
            msg.message for msg in messages if msg.severity == ValidationSeverity.ERROR
        ]
        warnings = [
            msg.message
            for msg in messages
            if msg.severity == ValidationSeverity.WARNING
        ]

        self.logger.info(
            "Configuration validation completed: %d errors, %d warnings",
            len(errors),
            len(warnings),
        )

        return errors, warnings

    def _validate_basic_structure(self, config: MARLConfig) -> List[ValidationMessage]:
        """Validate basic configuration structure."""
        messages = []

        # Required fields
        if not config.name:
            messages.append(
                ValidationMessage(
                    ValidationSeverity.ERROR, "Configuration name is required", "name"
                )
            )

        if not config.version:
            messages.append(
                ValidationMessage(
                    ValidationSeverity.ERROR,
                    "Configuration version is required",
                    "version",
                )
            )
        elif not self._is_valid_version_format(config.version):
            messages.append(
                ValidationMessage(
                    ValidationSeverity.ERROR,
                    f"Invalid version format: {config.version}",
                    "version",
                    "Use semantic versioning (e.g., '1.0.0')",
                )
            )

        # Agent count validation
        if not config.agents:
            messages.append(
                ValidationMessage(
                    ValidationSeverity.ERROR,
                    "At least one agent must be configured",
                    "agents",
                )
            )
        elif len(config.agents) > 20:
            messages.append(
                ValidationMessage(
                    ValidationSeverity.WARNING,
                    f"Large number of agents ({len(config.agents)}) may impact performance",
                    "agents",
                    "Consider reducing agent count or using distributed deployment",
                )
            )

        return messages

    def _validate_agents(self, config: MARLConfig) -> List[ValidationMessage]:
        """Validate agent configurations."""
        messages = []

        agent_types = set()
        agent_ids = set()

        for agent_id, agent_config in config.agents.items():
            path_prefix = f"agents.{agent_id}"

            # Validate agent ID uniqueness
            if agent_id in agent_ids:
                messages.append(
                    ValidationMessage(
                        ValidationSeverity.ERROR,
                        f"Duplicate agent ID: {agent_id}",
                        path_prefix,
                    )
                )
            agent_ids.add(agent_id)

            # Track agent types
            agent_types.add(agent_config.agent_type)

            # Validate individual agent
            messages.extend(self._validate_single_agent(agent_config, path_prefix))

        # Validate agent type distribution
        if len(agent_types) == 1 and len(config.agents) > 1:
            messages.append(
                ValidationMessage(
                    ValidationSeverity.WARNING,
                    "All agents have the same type - consider diversifying agent types",
                    "agents",
                    "Use different agent types (generator, validator, curriculum) for better coordination",
                )
            )

        return messages

    def _validate_single_agent(
        self, agent: AgentConfig, path_prefix: str
    ) -> List[ValidationMessage]:
        """Validate a single agent configuration."""
        messages = []

        # State and action dimensions
        if agent.state_dim <= 0:
            messages.append(
                ValidationMessage(
                    ValidationSeverity.ERROR,
                    f"State dimension must be positive: {agent.state_dim}",
                    f"{path_prefix}.state_dim",
                )
            )
        elif agent.state_dim > 2048:
            messages.append(
                ValidationMessage(
                    ValidationSeverity.WARNING,
                    f"Large state dimension ({agent.state_dim}) may impact performance",
                    f"{path_prefix}.state_dim",
                    "Consider dimensionality reduction or feature selection",
                )
            )

        if agent.action_dim <= 0:
            messages.append(
                ValidationMessage(
                    ValidationSeverity.ERROR,
                    f"Action dimension must be positive: {agent.action_dim}",
                    f"{path_prefix}.action_dim",
                )
            )
        elif agent.action_dim > 100:
            messages.append(
                ValidationMessage(
                    ValidationSeverity.WARNING,
                    f"Large action dimension ({agent.action_dim}) may slow learning",
                    f"{path_prefix}.action_dim",
                    "Consider action space discretization or hierarchical actions",
                )
            )

        # Learning parameters
        if hasattr(agent, "optimization") and agent.optimization:
            opt_path = f"{path_prefix}.optimization"

            lr = agent.optimization.learning_rate
            if lr <= 0:
                messages.append(
                    ValidationMessage(
                        ValidationSeverity.ERROR,
                        f"Learning rate must be positive: {lr}",
                        f"{opt_path}.learning_rate",
                    )
                )
            elif lr < self._thresholds["learning_rate"]["recommended_min"]:
                messages.append(
                    ValidationMessage(
                        ValidationSeverity.WARNING,
                        f"Learning rate may be too low: {lr}",
                        f"{opt_path}.learning_rate",
                        f"Consider values between {self._thresholds['learning_rate']['recommended_min']} and {self._thresholds['learning_rate']['recommended_max']}",
                    )
                )
            elif lr > self._thresholds["learning_rate"]["recommended_max"]:
                messages.append(
                    ValidationMessage(
                        ValidationSeverity.WARNING,
                        f"Learning rate may be too high: {lr}",
                        f"{opt_path}.learning_rate",
                        "High learning rates can cause training instability",
                    )
                )

        # Exploration parameters
        if hasattr(agent, "exploration") and agent.exploration:
            exp_path = f"{path_prefix}.exploration"

            initial_eps = agent.exploration.initial_epsilon
            final_eps = agent.exploration.final_epsilon

            if not (0.0 <= initial_eps <= 1.0):
                messages.append(
                    ValidationMessage(
                        ValidationSeverity.ERROR,
                        f"Initial epsilon must be between 0 and 1: {initial_eps}",
                        f"{exp_path}.initial_epsilon",
                    )
                )

            if not (0.0 <= final_eps <= 1.0):
                messages.append(
                    ValidationMessage(
                        ValidationSeverity.ERROR,
                        f"Final epsilon must be between 0 and 1: {final_eps}",
                        f"{exp_path}.final_epsilon",
                    )
                )

            if initial_eps < final_eps:
                messages.append(
                    ValidationMessage(
                        ValidationSeverity.WARNING,
                        f"Initial epsilon ({initial_eps}) should be >= final epsilon ({final_eps})",
                        f"{exp_path}",
                        "Exploration should decrease over time",
                    )
                )

            # Epsilon decay validation
            if hasattr(agent.exploration, "epsilon_decay"):
                decay = agent.exploration.epsilon_decay
                if not (0.9 <= decay <= 0.9999):
                    messages.append(
                        ValidationMessage(
                            ValidationSeverity.WARNING,
                            f"Epsilon decay rate may be suboptimal: {decay}",
                            f"{exp_path}.epsilon_decay",
                            "Typical values are between 0.99 and 0.999",
                        )
                    )

        # Network configuration
        if hasattr(agent, "network") and agent.network:
            net_path = f"{path_prefix}.network"

            if agent.network.hidden_layers:
                total_params = sum(agent.network.hidden_layers)
                if total_params > 10000:
                    messages.append(
                        ValidationMessage(
                            ValidationSeverity.WARNING,
                            f"Large network size may slow training: {total_params} total units",
                            f"{net_path}.hidden_layers",
                            "Consider smaller networks for faster training",
                        )
                    )

                # Check for decreasing layer sizes
                layers = agent.network.hidden_layers
                if len(layers) > 1:
                    for i in range(len(layers) - 1):
                        if layers[i] < layers[i + 1]:
                            messages.append(
                                ValidationMessage(
                                    ValidationSeverity.INFO,
                                    "Network layers increase in size - consider decreasing sizes",
                                    f"{net_path}.hidden_layers",
                                    "Typically, hidden layers decrease in size",
                                )
                            )
                            break

        # Replay buffer configuration
        if hasattr(agent, "replay_buffer") and agent.replay_buffer:
            buffer_path = f"{path_prefix}.replay_buffer"

            capacity = agent.replay_buffer.capacity
            batch_size = agent.replay_buffer.batch_size

            if capacity <= 0:
                messages.append(
                    ValidationMessage(
                        ValidationSeverity.ERROR,
                        f"Replay buffer capacity must be positive: {capacity}",
                        f"{buffer_path}.capacity",
                    )
                )
            elif capacity < batch_size * 10:
                messages.append(
                    ValidationMessage(
                        ValidationSeverity.WARNING,
                        f"Replay buffer capacity ({capacity}) should be much larger than batch size ({batch_size})",
                        f"{buffer_path}",
                        "Recommended: capacity >= 10 * batch_size",
                    )
                )

            if batch_size <= 0:
                messages.append(
                    ValidationMessage(
                        ValidationSeverity.ERROR,
                        f"Batch size must be positive: {batch_size}",
                        f"{buffer_path}.batch_size",
                    )
                )
            elif batch_size > capacity:
                messages.append(
                    ValidationMessage(
                        ValidationSeverity.ERROR,
                        f"Batch size ({batch_size}) cannot exceed buffer capacity ({capacity})",
                        f"{buffer_path}",
                    )
                )

        return messages

    def _validate_coordination(self, config: MARLConfig) -> List[ValidationMessage]:
        """Validate coordination configuration."""
        messages = []

        if not hasattr(config, "coordination") or not config.coordination:
            messages.append(
                ValidationMessage(
                    ValidationSeverity.WARNING,
                    "No coordination configuration specified",
                    "coordination",
                    "Add coordination configuration for multi-agent scenarios",
                )
            )
            return messages

        coord = config.coordination

        # Timeout validation
        if coord.coordination_timeout <= 0:
            messages.append(
                ValidationMessage(
                    ValidationSeverity.ERROR,
                    f"Coordination timeout must be positive: {coord.coordination_timeout}",
                    "coordination.coordination_timeout",
                )
            )
        elif coord.coordination_timeout > 600:  # 10 minutes
            messages.append(
                ValidationMessage(
                    ValidationSeverity.WARNING,
                    f"Very long coordination timeout: {coord.coordination_timeout}s",
                    "coordination.coordination_timeout",
                    "Long timeouts may reduce system responsiveness",
                )
            )

        # Concurrent coordinations
        if hasattr(coord, "max_concurrent_coordinations"):
            max_concurrent = coord.max_concurrent_coordinations
            if max_concurrent <= 0:
                messages.append(
                    ValidationMessage(
                        ValidationSeverity.ERROR,
                        f"Max concurrent coordinations must be positive: {max_concurrent}",
                        "coordination.max_concurrent_coordinations",
                    )
                )
            elif max_concurrent > len(config.agents) * 2:
                messages.append(
                    ValidationMessage(
                        ValidationSeverity.WARNING,
                        f"Max concurrent coordinations ({max_concurrent}) is very high for {len(config.agents)} agents",
                        "coordination.max_concurrent_coordinations",
                        "Consider reducing to avoid resource contention",
                    )
                )

        # Consensus configuration
        if hasattr(coord, "consensus") and coord.consensus:
            consensus = coord.consensus

            # Consensus strategy validation
            if hasattr(consensus, "strategy"):
                strategy = consensus.strategy
                if strategy == ConsensusStrategy.EXPERT_PRIORITY:
                    if (
                        not hasattr(consensus, "expert_weights")
                        or not consensus.expert_weights
                    ):
                        messages.append(
                            ValidationMessage(
                                ValidationSeverity.ERROR,
                                "Expert priority consensus requires expert_weights configuration",
                                "coordination.consensus.expert_weights",
                            )
                        )

            # Timeout validation
            if hasattr(consensus, "timeout_seconds"):
                consensus_timeout = consensus.timeout_seconds
                if consensus_timeout >= coord.coordination_timeout:
                    messages.append(
                        ValidationMessage(
                            ValidationSeverity.WARNING,
                            f"Consensus timeout ({consensus_timeout}s) should be less than coordination timeout ({coord.coordination_timeout}s)",
                            "coordination.consensus.timeout_seconds",
                        )
                    )

        return messages

    def _validate_learning(self, config: MARLConfig) -> List[ValidationMessage]:
        """Validate learning configuration."""
        messages = []

        if not hasattr(config, "learning") or not config.learning:
            messages.append(
                ValidationMessage(
                    ValidationSeverity.WARNING,
                    "No learning configuration specified",
                    "learning",
                    "Add learning configuration for training parameters",
                )
            )
            return messages

        learning = config.learning

        # Episode limits
        if hasattr(learning, "max_episodes"):
            max_episodes = learning.max_episodes
            if max_episodes <= 0:
                messages.append(
                    ValidationMessage(
                        ValidationSeverity.ERROR,
                        f"Max episodes must be positive: {max_episodes}",
                        "learning.max_episodes",
                    )
                )
            elif max_episodes < 1000:
                messages.append(
                    ValidationMessage(
                        ValidationSeverity.WARNING,
                        f"Low episode count may not allow sufficient learning: {max_episodes}",
                        "learning.max_episodes",
                        "Consider at least 10,000 episodes for stable learning",
                    )
                )

        # Evaluation and save intervals
        if hasattr(learning, "evaluation_interval") and hasattr(
            learning, "save_interval"
        ):
            eval_interval = learning.evaluation_interval
            save_interval = learning.save_interval

            if eval_interval <= 0:
                messages.append(
                    ValidationMessage(
                        ValidationSeverity.ERROR,
                        f"Evaluation interval must be positive: {eval_interval}",
                        "learning.evaluation_interval",
                    )
                )

            if save_interval <= 0:
                messages.append(
                    ValidationMessage(
                        ValidationSeverity.ERROR,
                        f"Save interval must be positive: {save_interval}",
                        "learning.save_interval",
                    )
                )

            if save_interval < eval_interval:
                messages.append(
                    ValidationMessage(
                        ValidationSeverity.INFO,
                        f"Save interval ({save_interval}) is less than evaluation interval ({eval_interval})",
                        "learning",
                        "Consider saving less frequently than evaluation",
                    )
                )

        # Shared learning validation
        if hasattr(learning, "shared_learning") and learning.shared_learning:
            shared = learning.shared_learning

            if shared.enabled and len(config.agents) == 1:
                messages.append(
                    ValidationMessage(
                        ValidationSeverity.WARNING,
                        "Shared learning enabled with only one agent",
                        "learning.shared_learning.enabled",
                        "Shared learning requires multiple agents",
                    )
                )

            if hasattr(shared, "experience_buffer_size"):
                buffer_size = shared.experience_buffer_size
                if buffer_size <= 0:
                    messages.append(
                        ValidationMessage(
                            ValidationSeverity.ERROR,
                            f"Shared experience buffer size must be positive: {buffer_size}",
                            "learning.shared_learning.experience_buffer_size",
                        )
                    )
                elif buffer_size > 1000000:
                    messages.append(
                        ValidationMessage(
                            ValidationSeverity.WARNING,
                            f"Large shared experience buffer may consume significant memory: {buffer_size}",
                            "learning.shared_learning.experience_buffer_size",
                            "Monitor memory usage with large buffers",
                        )
                    )

        return messages

    def _validate_system(self, config: MARLConfig) -> List[ValidationMessage]:
        """Validate system configuration."""
        messages = []

        if not hasattr(config, "system") or not config.system:
            messages.append(
                ValidationMessage(
                    ValidationSeverity.INFO,
                    "No system configuration specified, using defaults",
                    "system",
                )
            )
            return messages

        system = config.system

        # Device validation
        if hasattr(system, "device"):
            device = system.device
            if device not in ["auto", "cpu", "cuda", "mps"]:
                messages.append(
                    ValidationMessage(
                        ValidationSeverity.WARNING,
                        f"Unknown device type: {device}",
                        "system.device",
                        "Use 'auto', 'cpu', 'cuda', or 'mps'",
                    )
                )

        # Worker count validation
        if hasattr(system, "num_workers"):
            num_workers = system.num_workers
            if num_workers <= 0:
                messages.append(
                    ValidationMessage(
                        ValidationSeverity.ERROR,
                        f"Number of workers must be positive: {num_workers}",
                        "system.num_workers",
                    )
                )
            elif num_workers > 32:
                messages.append(
                    ValidationMessage(
                        ValidationSeverity.WARNING,
                        f"Large number of workers may not improve performance: {num_workers}",
                        "system.num_workers",
                        "Consider using fewer workers unless you have many CPU cores",
                    )
                )

        # Memory limits
        if hasattr(system, "memory_limit_gb") and system.memory_limit_gb is not None:
            memory_limit = system.memory_limit_gb
            if memory_limit <= 0:
                messages.append(
                    ValidationMessage(
                        ValidationSeverity.ERROR,
                        f"Memory limit must be positive: {memory_limit}GB",
                        "system.memory_limit_gb",
                    )
                )
            elif memory_limit < 2.0:
                messages.append(
                    ValidationMessage(
                        ValidationSeverity.WARNING,
                        f"Low memory limit may cause issues: {memory_limit}GB",
                        "system.memory_limit_gb",
                        "Consider at least 4GB for stable operation",
                    )
                )

        return messages

    def _validate_compatibility(self, config: MARLConfig) -> List[ValidationMessage]:
        """Validate cross-component compatibility."""
        messages = []

        # Check agent compatibility
        if len(config.agents) > 1:
            # Check for compatible learning rates
            learning_rates = []
            for agent in config.agents.values():
                if hasattr(agent, "optimization") and agent.optimization:
                    learning_rates.append(agent.optimization.learning_rate)

            if learning_rates and len(set(learning_rates)) > 1:
                lr_range = max(learning_rates) - min(learning_rates)
                if lr_range > 0.01:  # Significant difference
                    messages.append(
                        ValidationMessage(
                            ValidationSeverity.INFO,
                            f"Agents have different learning rates (range: {lr_range:.4f})",
                            "agents",
                            "Consider using similar learning rates for coordinated learning",
                        )
                    )

            # Check for compatible exploration strategies
            exploration_strategies = set()
            for agent in config.agents.values():
                if hasattr(agent, "exploration") and agent.exploration:
                    if hasattr(agent.exploration, "strategy"):
                        exploration_strategies.add(agent.exploration.strategy)

            if len(exploration_strategies) > 1:
                messages.append(
                    ValidationMessage(
                        ValidationSeverity.INFO,
                        f"Agents use different exploration strategies: {list(exploration_strategies)}",
                        "agents",
                        "Mixed exploration strategies can be beneficial but may complicate coordination",
                    )
                )

        # Check coordination vs agent count compatibility
        if hasattr(config, "coordination") and config.coordination:
            if (
                hasattr(config.coordination, "consensus")
                and config.coordination.consensus
            ):
                consensus = config.coordination.consensus
                if hasattr(consensus, "strategy"):
                    strategy = consensus.strategy
                    agent_count = len(config.agents)

                    if (
                        strategy == ConsensusStrategy.MAJORITY_VOTE
                        and agent_count % 2 == 0
                    ):
                        messages.append(
                            ValidationMessage(
                                ValidationSeverity.WARNING,
                                f"Even number of agents ({agent_count}) with majority vote may cause ties",
                                "coordination.consensus.strategy",
                                "Consider using weighted average or adding an agent",
                            )
                        )

        return messages

    def _validate_performance_optimization(
        self, config: MARLConfig
    ) -> List[ValidationMessage]:
        """Validate configuration for performance optimization."""
        messages = []

        # Memory usage estimation
        total_buffer_size = 0
        total_network_params = 0

        for agent in config.agents.values():
            if hasattr(agent, "replay_buffer") and agent.replay_buffer:
                total_buffer_size += agent.replay_buffer.capacity

            if (
                hasattr(agent, "network")
                and agent.network
                and agent.network.hidden_layers
            ):
                # Rough parameter estimation
                layers = (
                    [agent.state_dim] + agent.network.hidden_layers + [agent.action_dim]
                )
                for i in range(len(layers) - 1):
                    total_network_params += layers[i] * layers[i + 1]

        # Memory warnings
        if total_buffer_size > 1000000:
            estimated_memory_gb = total_buffer_size * 32 / (1024**3)  # Rough estimate
            messages.append(
                ValidationMessage(
                    ValidationSeverity.WARNING,
                    f"Large total replay buffer size may use ~{estimated_memory_gb:.1f}GB memory",
                    "agents.*.replay_buffer.capacity",
                    "Monitor memory usage and consider reducing buffer sizes",
                )
            )

        if total_network_params > 1000000:
            messages.append(
                ValidationMessage(
                    ValidationSeverity.WARNING,
                    f"Large total network size (~{total_network_params:,} parameters) may slow training",
                    "agents.*.network.hidden_layers",
                    "Consider smaller networks for faster training",
                )
            )

        # Training efficiency suggestions
        batch_sizes = []
        for agent in config.agents.values():
            if hasattr(agent, "replay_buffer") and agent.replay_buffer:
                batch_sizes.append(agent.replay_buffer.batch_size)

        if batch_sizes:
            avg_batch_size = sum(batch_sizes) / len(batch_sizes)
            if avg_batch_size < 16:
                messages.append(
                    ValidationMessage(
                        ValidationSeverity.INFO,
                        f"Small average batch size ({avg_batch_size:.1f}) may slow training",
                        "agents.*.replay_buffer.batch_size",
                        "Consider larger batch sizes (32-128) for better GPU utilization",
                    )
                )

        return messages

    def _is_valid_version_format(self, version: str) -> bool:
        """Check if version string follows semantic versioning."""
        pattern = r"^\d+\.\d+\.\d+(-[a-zA-Z0-9\-\.]+)?(\+[a-zA-Z0-9\-\.]+)?$"
        return bool(re.match(pattern, version))

    def validate_config_compatibility(
        self, config1: MARLConfig, config2: MARLConfig
    ) -> Tuple[List[str], List[str]]:
        """
        Validate compatibility between two configurations.

        Args:
            config1: First configuration
            config2: Second configuration

        Returns:
            Tuple of (error_messages, warning_messages)
        """
        messages = []

        # Version compatibility
        if config1.version != config2.version:
            messages.append(
                ValidationMessage(
                    ValidationSeverity.WARNING,
                    f"Different versions: {config1.version} vs {config2.version}",
                    "version",
                    "Ensure configurations are compatible across versions",
                )
            )

        # Agent compatibility
        agents1 = set(config1.agents.keys())
        agents2 = set(config2.agents.keys())

        if agents1 != agents2:
            only_in_1 = agents1 - agents2
            only_in_2 = agents2 - agents1

            if only_in_1:
                messages.append(
                    ValidationMessage(
                        ValidationSeverity.WARNING,
                        f"Agents only in first config: {list(only_in_1)}",
                        "agents",
                    )
                )

            if only_in_2:
                messages.append(
                    ValidationMessage(
                        ValidationSeverity.WARNING,
                        f"Agents only in second config: {list(only_in_2)}",
                        "agents",
                    )
                )

        # Check common agents for compatibility
        common_agents = agents1 & agents2
        for agent_id in common_agents:
            agent1 = config1.agents[agent_id]
            agent2 = config2.agents[agent_id]

            # State/action dimension compatibility
            if agent1.state_dim != agent2.state_dim:
                messages.append(
                    ValidationMessage(
                        ValidationSeverity.ERROR,
                        f"Incompatible state dimensions for {agent_id}: {agent1.state_dim} vs {agent2.state_dim}",
                        f"agents.{agent_id}.state_dim",
                    )
                )

            if agent1.action_dim != agent2.action_dim:
                messages.append(
                    ValidationMessage(
                        ValidationSeverity.ERROR,
                        f"Incompatible action dimensions for {agent_id}: {agent1.action_dim} vs {agent2.action_dim}",
                        f"agents.{agent_id}.action_dim",
                    )
                )

        # Coordination compatibility
        if hasattr(config1, "coordination") and hasattr(config2, "coordination"):
            if config1.coordination and config2.coordination:
                # Check consensus strategy compatibility
                if hasattr(config1.coordination, "consensus") and hasattr(
                    config2.coordination, "consensus"
                ):
                    if (
                        config1.coordination.consensus
                        and config2.coordination.consensus
                        and hasattr(config1.coordination.consensus, "strategy")
                        and hasattr(config2.coordination.consensus, "strategy")
                    ):
                        if (
                            config1.coordination.consensus.strategy
                            != config2.coordination.consensus.strategy
                        ):
                            messages.append(
                                ValidationMessage(
                                    ValidationSeverity.WARNING,
                                    f"Different consensus strategies: {config1.coordination.consensus.strategy} vs {config2.coordination.consensus.strategy}",
                                    "coordination.consensus.strategy",
                                )
                            )

        # Separate errors and warnings
        errors = [
            msg.message for msg in messages if msg.severity == ValidationSeverity.ERROR
        ]
        warnings = [
            msg.message
            for msg in messages
            if msg.severity == ValidationSeverity.WARNING
        ]

        return errors, warnings

    def generate_optimization_suggestions(
        self, config: MARLConfig
    ) -> List[Dict[str, Any]]:
        """
        Generate performance optimization suggestions.

        Args:
            config: MARL configuration to analyze

        Returns:
            List of optimization suggestions
        """
        suggestions = []

        # Analyze learning rates
        learning_rates = []
        for agent in config.agents.values():
            if hasattr(agent, "optimization") and agent.optimization:
                learning_rates.append(agent.optimization.learning_rate)

        if learning_rates:
            avg_lr = sum(learning_rates) / len(learning_rates)
            if avg_lr > 0.01:
                suggestions.append(
                    {
                        "category": "learning",
                        "priority": "high",
                        "suggestion": "Consider reducing learning rates for more stable training",
                        "current_value": f"Average: {avg_lr:.4f}",
                        "recommended_value": "0.001 - 0.005",
                        "impact": "Improved training stability",
                    }
                )

        # Analyze batch sizes
        batch_sizes = []
        for agent in config.agents.values():
            if hasattr(agent, "replay_buffer") and agent.replay_buffer:
                batch_sizes.append(agent.replay_buffer.batch_size)

        if batch_sizes:
            avg_batch = sum(batch_sizes) / len(batch_sizes)
            if avg_batch < 32:
                suggestions.append(
                    {
                        "category": "training",
                        "priority": "medium",
                        "suggestion": "Consider larger batch sizes for better GPU utilization",
                        "current_value": f"Average: {avg_batch:.1f}",
                        "recommended_value": "32 - 128",
                        "impact": "Faster training, better gradient estimates",
                    }
                )

        # Analyze network sizes
        for agent_id, agent in config.agents.items():
            if (
                hasattr(agent, "network")
                and agent.network
                and agent.network.hidden_layers
            ):
                total_units = sum(agent.network.hidden_layers)
                if total_units > 2048:
                    suggestions.append(
                        {
                            "category": "architecture",
                            "priority": "medium",
                            "suggestion": f"Consider smaller network for agent {agent_id}",
                            "current_value": f"Total units: {total_units}",
                            "recommended_value": "< 1024 total units",
                            "impact": "Faster training, reduced memory usage",
                        }
                    )

        # Analyze coordination settings
        if hasattr(config, "coordination") and config.coordination:
            if config.coordination.coordination_timeout > 120:
                suggestions.append(
                    {
                        "category": "coordination",
                        "priority": "low",
                        "suggestion": "Consider shorter coordination timeout for better responsiveness",
                        "current_value": f"{config.coordination.coordination_timeout}s",
                        "recommended_value": "30 - 120s",
                        "impact": "Improved system responsiveness",
                    }
                )

        return suggestions


class ConfigValidatorFactory:
    """Factory for creating configuration validators."""

    @staticmethod
    def create() -> ConfigValidator:
        """Create a standard configuration validator."""
        return ConfigValidator()

    @staticmethod
    def create_strict() -> ConfigValidator:
        """Create a strict configuration validator with tighter thresholds."""
        validator = ConfigValidator()

        # Tighten thresholds for strict validation
        validator._thresholds["learning_rate"]["recommended_max"] = 0.005
        validator._thresholds["batch_size"]["recommended_min"] = 32
        validator._optimization_rules["memory_usage"][
            "replay_buffer_warning_threshold"
        ] = 100000

        return validator

    @staticmethod
    def create_permissive() -> ConfigValidator:
        """Create a permissive configuration validator with relaxed thresholds."""
        validator = ConfigValidator()

        # Relax thresholds for permissive validation
        validator._thresholds["learning_rate"]["recommended_max"] = 0.05
        validator._thresholds["batch_size"]["recommended_min"] = 8
        validator._optimization_rules["memory_usage"][
            "replay_buffer_warning_threshold"
        ] = 2000000

        return validator
