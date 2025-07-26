"""
MARL Configuration Schema.

This module defines comprehensive configuration schemas for all MARL components
including agents, coordination, learning, and system parameters.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from utils.logging_config import get_logger


class OptimizationType(Enum):
    """Types of optimization algorithms."""

    ADAM = "adam"
    SGD = "sgd"
    RMSPROP = "rmsprop"
    ADAMW = "adamw"


class ExplorationStrategy(Enum):
    """Types of exploration strategies."""

    EPSILON_GREEDY = "epsilon_greedy"
    BOLTZMANN = "boltzmann"
    UCB = "ucb"
    THOMPSON_SAMPLING = "thompson_sampling"


class ConsensusStrategy(Enum):
    """Types of consensus strategies."""

    MAJORITY_VOTE = "majority_vote"
    WEIGHTED_AVERAGE = "weighted_average"
    EXPERT_PRIORITY = "expert_priority"
    ADAPTIVE = "adaptive"


class NetworkArchitecture(Enum):
    """Types of neural network architectures."""

    FEEDFORWARD = "feedforward"
    CONVOLUTIONAL = "convolutional"
    RECURRENT = "recurrent"
    TRANSFORMER = "transformer"


@dataclass
class NetworkConfig:
    """Neural network configuration."""

    architecture: NetworkArchitecture = NetworkArchitecture.FEEDFORWARD
    hidden_layers: List[int] = field(default_factory=lambda: [256, 128, 64])
    activation_function: str = "relu"
    dropout_rate: float = 0.1
    batch_normalization: bool = True
    weight_initialization: str = "xavier_uniform"

    def __post_init__(self):
        """Validate network configuration."""
        if self.dropout_rate < 0.0 or self.dropout_rate > 1.0:
            raise ValueError("Dropout rate must be between 0.0 and 1.0")
        if not self.hidden_layers:
            raise ValueError("At least one hidden layer must be specified")
        if any(layer <= 0 for layer in self.hidden_layers):
            raise ValueError("All hidden layer sizes must be positive")


@dataclass
class OptimizationConfig:
    """Optimization configuration."""

    optimizer_type: OptimizationType = OptimizationType.ADAM
    learning_rate: float = 0.001
    learning_rate_decay: float = 0.95
    learning_rate_schedule: str = "exponential"
    weight_decay: float = 0.0001
    gradient_clipping: float = 1.0
    momentum: float = 0.9  # For SGD and RMSprop
    beta1: float = 0.9  # For Adam
    beta2: float = 0.999  # For Adam
    epsilon: float = 1e-8  # For Adam and RMSprop

    def __post_init__(self):
        """Validate optimization configuration."""
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        if self.learning_rate_decay < 0 or self.learning_rate_decay > 1:
            raise ValueError("Learning rate decay must be between 0 and 1")
        if self.gradient_clipping <= 0:
            raise ValueError("Gradient clipping must be positive")


@dataclass
class ExplorationConfig:
    """Exploration strategy configuration."""

    strategy: ExplorationStrategy = ExplorationStrategy.EPSILON_GREEDY
    initial_epsilon: float = 1.0
    final_epsilon: float = 0.01
    epsilon_decay: float = 0.995
    epsilon_decay_steps: int = 10000
    temperature: float = 1.0  # For Boltzmann exploration
    ucb_c: float = 2.0  # For UCB exploration

    def __post_init__(self):
        """Validate exploration configuration."""
        if self.initial_epsilon < 0 or self.initial_epsilon > 1:
            raise ValueError("Initial epsilon must be between 0 and 1")
        if self.final_epsilon < 0 or self.final_epsilon > 1:
            raise ValueError("Final epsilon must be between 0 and 1")
        if self.epsilon_decay <= 0 or self.epsilon_decay > 1:
            raise ValueError("Epsilon decay must be between 0 and 1")
        if self.temperature <= 0:
            raise ValueError("Temperature must be positive")


@dataclass
class ReplayBufferConfig:
    """Replay buffer configuration."""

    capacity: int = 100000
    batch_size: int = 32
    min_size_to_sample: int = 1000
    prioritized_replay: bool = False
    alpha: float = 0.6  # Prioritization exponent
    beta: float = 0.4  # Importance sampling exponent
    beta_increment: float = 0.001

    def __post_init__(self):
        """Validate replay buffer configuration."""
        if self.capacity <= 0:
            raise ValueError("Capacity must be positive")
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if self.min_size_to_sample < 0:
            raise ValueError("Minimum size to sample must be non-negative")
        if self.batch_size > self.capacity:
            raise ValueError("Batch size cannot exceed capacity")


@dataclass
class AgentConfig:
    """Individual agent configuration."""

    agent_id: str
    agent_type: str  # "generator", "validator", "curriculum"
    network: NetworkConfig = field(default_factory=NetworkConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    exploration: ExplorationConfig = field(default_factory=ExplorationConfig)
    replay_buffer: ReplayBufferConfig = field(default_factory=ReplayBufferConfig)

    # Agent-specific parameters
    gamma: float = 0.99  # Discount factor
    tau: float = 0.005  # Target network update rate
    update_frequency: int = 4  # Training frequency
    target_update_frequency: int = 1000
    double_dqn: bool = True
    dueling_dqn: bool = True

    # Agent-specific hyperparameters
    state_dim: int = 128
    action_dim: int = 10
    reward_scaling: float = 1.0

    def __post_init__(self):
        """Validate agent configuration."""
        if not self.agent_id:
            raise ValueError("Agent ID cannot be empty")
        if self.gamma < 0 or self.gamma > 1:
            raise ValueError("Gamma must be between 0 and 1")
        if self.tau <= 0 or self.tau > 1:
            raise ValueError("Tau must be between 0 and 1")
        if self.update_frequency <= 0:
            raise ValueError("Update frequency must be positive")
        if self.target_update_frequency <= 0:
            raise ValueError("Target update frequency must be positive")
        if self.state_dim <= 0:
            raise ValueError("State dimension must be positive")
        if self.action_dim <= 0:
            raise ValueError("Action dimension must be positive")


@dataclass
class ConsensusConfig:
    """Consensus mechanism configuration."""

    strategy: ConsensusStrategy = ConsensusStrategy.WEIGHTED_AVERAGE
    voting_threshold: float = 0.5
    confidence_threshold: float = 0.7
    timeout_seconds: float = 30.0
    max_iterations: int = 10
    convergence_threshold: float = 0.01

    # Strategy-specific parameters
    expert_weights: Dict[str, float] = field(default_factory=dict)
    adaptive_learning_rate: float = 0.1

    def __post_init__(self):
        """Validate consensus configuration."""
        if self.voting_threshold < 0 or self.voting_threshold > 1:
            raise ValueError("Voting threshold must be between 0 and 1")
        if self.confidence_threshold < 0 or self.confidence_threshold > 1:
            raise ValueError("Confidence threshold must be between 0 and 1")
        if self.timeout_seconds <= 0:
            raise ValueError("Timeout must be positive")
        if self.max_iterations <= 0:
            raise ValueError("Max iterations must be positive")
        if self.convergence_threshold <= 0:
            raise ValueError("Convergence threshold must be positive")


@dataclass
class CommunicationConfig:
    """Communication protocol configuration."""

    message_queue_size: int = 1000
    message_timeout: float = 10.0
    max_retries: int = 3
    retry_delay: float = 1.0
    compression_enabled: bool = True
    encryption_enabled: bool = False
    heartbeat_interval: float = 5.0

    def __post_init__(self):
        """Validate communication configuration."""
        if self.message_queue_size <= 0:
            raise ValueError("Message queue size must be positive")
        if self.message_timeout <= 0:
            raise ValueError("Message timeout must be positive")
        if self.max_retries < 0:
            raise ValueError("Max retries must be non-negative")
        if self.retry_delay < 0:
            raise ValueError("Retry delay must be non-negative")
        if self.heartbeat_interval <= 0:
            raise ValueError("Heartbeat interval must be positive")


@dataclass
class CoordinationConfig:
    """Coordination system configuration."""

    consensus: ConsensusConfig = field(default_factory=ConsensusConfig)
    communication: CommunicationConfig = field(default_factory=CommunicationConfig)

    # Coordination-specific parameters
    coordination_timeout: float = 60.0
    max_concurrent_coordinations: int = 10
    conflict_resolution_strategy: str = "priority_based"
    deadlock_detection_enabled: bool = True
    deadlock_timeout: float = 120.0

    def __post_init__(self):
        """Validate coordination configuration."""
        if self.coordination_timeout <= 0:
            raise ValueError("Coordination timeout must be positive")
        if self.max_concurrent_coordinations <= 0:
            raise ValueError("Max concurrent coordinations must be positive")
        if self.deadlock_timeout <= 0:
            raise ValueError("Deadlock timeout must be positive")


@dataclass
class SharedLearningConfig:
    """Shared learning configuration."""

    enabled: bool = True
    experience_sharing_rate: float = 0.1
    experience_buffer_size: int = 50000
    sharing_strategy: str = "high_reward"  # "all", "high_reward", "novel", "adaptive"
    novelty_threshold: float = 0.8
    reward_threshold: float = 0.7

    # Continuous learning parameters
    continuous_learning_enabled: bool = True
    learning_update_interval: float = 10.0
    performance_window_size: int = 100
    adaptation_threshold: float = 0.05

    def __post_init__(self):
        """Validate shared learning configuration."""
        if self.experience_sharing_rate < 0 or self.experience_sharing_rate > 1:
            raise ValueError("Experience sharing rate must be between 0 and 1")
        if self.experience_buffer_size <= 0:
            raise ValueError("Experience buffer size must be positive")
        if self.novelty_threshold < 0 or self.novelty_threshold > 1:
            raise ValueError("Novelty threshold must be between 0 and 1")
        if self.reward_threshold < 0 or self.reward_threshold > 1:
            raise ValueError("Reward threshold must be between 0 and 1")
        if self.learning_update_interval <= 0:
            raise ValueError("Learning update interval must be positive")
        if self.performance_window_size <= 0:
            raise ValueError("Performance window size must be positive")


@dataclass
class LearningConfig:
    """Learning system configuration."""

    shared_learning: SharedLearningConfig = field(default_factory=SharedLearningConfig)

    # General learning parameters
    training_enabled: bool = True
    evaluation_interval: int = 1000
    save_interval: int = 10000
    max_episodes: int = 100000
    max_steps_per_episode: int = 1000

    # Performance monitoring
    performance_tracking_enabled: bool = True
    metrics_collection_interval: float = 1.0
    trend_analysis_enabled: bool = True

    def __post_init__(self):
        """Validate learning configuration."""
        if self.evaluation_interval <= 0:
            raise ValueError("Evaluation interval must be positive")
        if self.save_interval <= 0:
            raise ValueError("Save interval must be positive")
        if self.max_episodes <= 0:
            raise ValueError("Max episodes must be positive")
        if self.max_steps_per_episode <= 0:
            raise ValueError("Max steps per episode must be positive")
        if self.metrics_collection_interval <= 0:
            raise ValueError("Metrics collection interval must be positive")


@dataclass
class SystemConfig:
    """System-level configuration."""

    device: str = "auto"  # "cpu", "cuda", "auto"
    num_workers: int = 4
    seed: Optional[int] = None
    deterministic: bool = False

    # Resource management
    memory_limit_gb: Optional[float] = None
    gpu_memory_fraction: float = 0.8

    # Logging and monitoring
    log_level: str = "INFO"
    log_to_file: bool = True
    log_file_path: str = "logs/marl.log"
    metrics_enabled: bool = True

    def __post_init__(self):
        """Validate system configuration."""
        if self.num_workers <= 0:
            raise ValueError("Number of workers must be positive")
        if self.gpu_memory_fraction <= 0 or self.gpu_memory_fraction > 1:
            raise ValueError("GPU memory fraction must be between 0 and 1")
        if self.memory_limit_gb is not None and self.memory_limit_gb <= 0:
            raise ValueError("Memory limit must be positive")


@dataclass
class MARLConfig:
    """
    Comprehensive MARL configuration.

    This is the main configuration class that contains all MARL system parameters
    including agents, coordination, learning, and system settings.
    """

    # Configuration metadata
    version: str = "1.0.0"
    name: str = "default_marl_config"
    description: str = "Default MARL configuration"
    created_at: Optional[str] = None

    # Core configuration sections
    agents: Dict[str, AgentConfig] = field(default_factory=dict)
    coordination: CoordinationConfig = field(default_factory=CoordinationConfig)
    learning: LearningConfig = field(default_factory=LearningConfig)
    system: SystemConfig = field(default_factory=SystemConfig)

    # Environment-specific parameters
    environment: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate MARL configuration."""
        if not self.version:
            raise ValueError("Version cannot be empty")
        if not self.name:
            raise ValueError("Name cannot be empty")
        if not self.agents:
            raise ValueError("At least one agent must be configured")

        # Validate agent IDs are unique
        agent_ids = set()
        for agent_id, agent_config in self.agents.items():
            if agent_id in agent_ids:
                raise ValueError(f"Duplicate agent ID: {agent_id}")
            agent_ids.add(agent_id)

            # Ensure agent config ID matches key
            if agent_config.agent_id != agent_id:
                agent_config.agent_id = agent_id

    def get_agent_config(self, agent_id: str) -> Optional[AgentConfig]:
        """Get configuration for a specific agent."""
        return self.agents.get(agent_id)

    def add_agent_config(self, agent_config: AgentConfig):
        """Add or update agent configuration."""
        self.agents[agent_config.agent_id] = agent_config

    def remove_agent_config(self, agent_id: str) -> bool:
        """Remove agent configuration."""
        if agent_id in self.agents:
            del self.agents[agent_id]
            return True
        return False

    def get_agent_ids(self) -> List[str]:
        """Get list of all configured agent IDs."""
        return list(self.agents.keys())

    def validate(self) -> List[str]:
        """
        Validate the entire configuration and return list of validation errors.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        try:
            # Validate basic structure
            if not self.agents:
                errors.append("No agents configured")

            # Validate each agent
            for agent_id, agent_config in self.agents.items():
                try:
                    agent_config.__post_init__()
                except ValueError as e:
                    errors.append(f"Agent {agent_id}: {str(e)}")

            # Validate coordination
            try:
                self.coordination.__post_init__()
            except ValueError as e:
                errors.append(f"Coordination: {str(e)}")

            # Validate learning
            try:
                self.learning.__post_init__()
            except ValueError as e:
                errors.append(f"Learning: {str(e)}")

            # Validate system
            try:
                self.system.__post_init__()
            except ValueError as e:
                errors.append(f"System: {str(e)}")

        except Exception as e:
            errors.append(f"Configuration validation error: {str(e)}")

        return errors

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        from dataclasses import asdict

        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MARLConfig":
        """Create configuration from dictionary."""
        # Convert nested dictionaries to appropriate dataclass instances
        config_data = data.copy()

        # Convert agents
        if "agents" in config_data:
            agents = {}
            for agent_id, agent_data in config_data["agents"].items():
                # Convert nested configs
                if "network" in agent_data:
                    agent_data["network"] = NetworkConfig(**agent_data["network"])
                if "optimization" in agent_data:
                    agent_data["optimization"] = OptimizationConfig(
                        **agent_data["optimization"]
                    )
                if "exploration" in agent_data:
                    agent_data["exploration"] = ExplorationConfig(
                        **agent_data["exploration"]
                    )
                if "replay_buffer" in agent_data:
                    agent_data["replay_buffer"] = ReplayBufferConfig(
                        **agent_data["replay_buffer"]
                    )

                agents[agent_id] = AgentConfig(**agent_data)
            config_data["agents"] = agents

        # Convert coordination
        if "coordination" in config_data:
            coord_data = config_data["coordination"]
            if "consensus" in coord_data:
                coord_data["consensus"] = ConsensusConfig(**coord_data["consensus"])
            if "communication" in coord_data:
                coord_data["communication"] = CommunicationConfig(
                    **coord_data["communication"]
                )
            config_data["coordination"] = CoordinationConfig(**coord_data)

        # Convert learning
        if "learning" in config_data:
            learn_data = config_data["learning"]
            if "shared_learning" in learn_data:
                learn_data["shared_learning"] = SharedLearningConfig(
                    **learn_data["shared_learning"]
                )
            config_data["learning"] = LearningConfig(**learn_data)

        # Convert system
        if "system" in config_data:
            config_data["system"] = SystemConfig(**config_data["system"])

        return cls(**config_data)
