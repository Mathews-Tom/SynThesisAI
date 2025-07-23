"""
MARL Configuration Management

This module provides configuration classes for multi-agent reinforcement learning
components, following the development standards for comprehensive configuration
management with validation and compatibility checking.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from utils.exceptions import ValidationError

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Configuration for individual RL agents."""

    # Neural network architecture
    hidden_layers: List[int] = field(default_factory=lambda: [256, 128, 64])
    activation: str = "relu"
    learning_rate: float = 0.001

    # Q-learning parameters
    gamma: float = 0.99
    epsilon_initial: float = 1.0
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.01

    # Experience replay
    buffer_size: int = 100000
    batch_size: int = 32
    target_update_freq: int = 1000

    # Training parameters
    max_episodes: int = 10000
    max_steps_per_episode: int = 1000

    def validate(self) -> None:
        """Validate agent configuration parameters."""
        if self.learning_rate <= 0 or self.learning_rate > 1:
            raise ValidationError(
                "Learning rate must be between 0 and 1", field="learning_rate"
            )

        if self.gamma <= 0 or self.gamma > 1:
            raise ValidationError("Gamma must be between 0 and 1", field="gamma")

        if self.buffer_size <= 0:
            raise ValidationError("Buffer size must be positive", field="buffer_size")

        if self.batch_size <= 0 or self.batch_size > self.buffer_size:
            raise ValidationError(
                "Batch size must be positive and <= buffer size", field="batch_size"
            )


@dataclass
class GeneratorAgentConfig(AgentConfig):
    """Configuration specific to Generator RL Agent."""

    # Generation strategy parameters
    strategy_count: int = 8
    novelty_weight: float = 0.3
    quality_weight: float = 0.5
    efficiency_weight: float = 0.2

    # Reward function parameters
    quality_threshold: float = 0.8
    novelty_threshold: float = 0.6
    coordination_bonus: float = 0.1
    validation_penalty: float = 0.2


@dataclass
class ValidatorAgentConfig(AgentConfig):
    """Configuration specific to Validator RL Agent."""

    # Validation strategy parameters
    threshold_count: int = 8
    validation_accuracy_weight: float = 0.7
    efficiency_weight: float = 0.3
    feedback_quality_weight: float = 0.2

    # Reward function parameters
    false_positive_penalty: float = 0.1
    false_negative_penalty: float = 0.15
    feedback_quality_bonus: float = 0.2


@dataclass
class CurriculumAgentConfig(AgentConfig):
    """Configuration specific to Curriculum RL Agent."""

    # Curriculum strategy parameters
    strategy_count: int = 8
    pedagogical_coherence_weight: float = 0.4
    learning_progression_weight: float = 0.4
    objective_alignment_weight: float = 0.2

    # Reward function parameters
    coherence_threshold: float = 0.7
    progression_threshold: float = 0.7
    integration_bonus: float = 0.15


@dataclass
class CoordinationConfig:
    """Configuration for multi-agent coordination mechanisms."""

    # Consensus parameters
    consensus_strategy: str = "adaptive_consensus"
    min_consensus_quality: float = 0.8
    consensus_timeout: float = 30.0

    # Conflict resolution
    max_negotiation_rounds: int = 5
    conflict_resolution_strategy: str = "weighted_priority"

    # Communication
    message_queue_size: int = 1000
    communication_timeout: float = 10.0

    # Coordination thresholds
    coordination_success_threshold: float = 0.85
    coordination_failure_threshold: float = 0.3


@dataclass
class ExperienceConfig:
    """Configuration for shared experience management."""

    # Buffer sizes
    shared_buffer_size: int = 50000
    agent_buffer_size: int = 25000

    # Experience sharing
    high_reward_threshold: float = 0.8
    novelty_threshold: float = 0.7
    sharing_probability: float = 0.3

    # Experience prioritization
    priority_alpha: float = 0.6
    importance_sampling_beta: float = 0.4


@dataclass
class MARLConfig:
    """Main configuration class for MARL system."""

    # Agent configurations
    generator_config: GeneratorAgentConfig = field(default_factory=GeneratorAgentConfig)
    validator_config: ValidatorAgentConfig = field(default_factory=ValidatorAgentConfig)
    curriculum_config: CurriculumAgentConfig = field(
        default_factory=CurriculumAgentConfig
    )

    # System configurations
    coordination_config: CoordinationConfig = field(default_factory=CoordinationConfig)
    experience_config: ExperienceConfig = field(default_factory=ExperienceConfig)

    # Performance monitoring
    monitoring_enabled: bool = True
    metrics_collection_interval: int = 100
    performance_report_interval: int = 1000

    # Distributed training
    distributed_training: bool = False
    num_workers: int = 4
    gpu_enabled: bool = True

    # Logging and debugging
    log_level: str = "INFO"
    debug_mode: bool = False
    checkpoint_interval: int = 5000

    @classmethod
    def from_file(cls, config_path: Path) -> "MARLConfig":
        """Load MARL configuration from YAML file."""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)

            # Create configuration with validation
            config = cls()

            # Update with loaded data
            if "generator" in config_data:
                config.generator_config = GeneratorAgentConfig(
                    **config_data["generator"]
                )

            if "validator" in config_data:
                config.validator_config = ValidatorAgentConfig(
                    **config_data["validator"]
                )

            if "curriculum" in config_data:
                config.curriculum_config = CurriculumAgentConfig(
                    **config_data["curriculum"]
                )

            if "coordination" in config_data:
                config.coordination_config = CoordinationConfig(
                    **config_data["coordination"]
                )

            if "experience" in config_data:
                config.experience_config = ExperienceConfig(**config_data["experience"])

            # Update system settings
            for key, value in config_data.get("system", {}).items():
                if hasattr(config, key):
                    setattr(config, key, value)

            # Validate configuration
            config.validate()

            logger.info("MARL configuration loaded successfully from %s", config_path)
            return config

        except Exception as e:
            logger.error(
                "Failed to load MARL configuration from %s: %s", config_path, str(e)
            )
            raise ValidationError(f"Invalid MARL configuration: {str(e)}") from e

    def validate(self) -> None:
        """Validate the complete MARL configuration."""
        # Validate individual agent configurations
        self.generator_config.validate()
        self.validator_config.validate()
        self.curriculum_config.validate()

        # Validate coordination parameters
        if (
            self.coordination_config.min_consensus_quality <= 0
            or self.coordination_config.min_consensus_quality > 1
        ):
            raise ValidationError(
                "Min consensus quality must be between 0 and 1",
                field="min_consensus_quality",
            )

        if (
            self.coordination_config.coordination_success_threshold <= 0
            or self.coordination_config.coordination_success_threshold > 1
        ):
            raise ValidationError(
                "Coordination success threshold must be between 0 and 1",
                field="coordination_success_threshold",
            )

        # Validate system parameters
        if self.num_workers <= 0:
            raise ValidationError(
                "Number of workers must be positive", field="num_workers"
            )

        if self.checkpoint_interval <= 0:
            raise ValidationError(
                "Checkpoint interval must be positive", field="checkpoint_interval"
            )

        logger.info("MARL configuration validation completed successfully")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        return {
            "generator": self.generator_config.__dict__,
            "validator": self.validator_config.__dict__,
            "curriculum": self.curriculum_config.__dict__,
            "coordination": self.coordination_config.__dict__,
            "experience": self.experience_config.__dict__,
            "system": {
                "monitoring_enabled": self.monitoring_enabled,
                "metrics_collection_interval": self.metrics_collection_interval,
                "performance_report_interval": self.performance_report_interval,
                "distributed_training": self.distributed_training,
                "num_workers": self.num_workers,
                "gpu_enabled": self.gpu_enabled,
                "log_level": self.log_level,
                "debug_mode": self.debug_mode,
                "checkpoint_interval": self.checkpoint_interval,
            },
        }

    def save_to_file(self, config_path: Path) -> None:
        """Save configuration to YAML file."""
        try:
            config_path.parent.mkdir(parents=True, exist_ok=True)

            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)

            logger.info("MARL configuration saved to %s", config_path)

        except Exception as e:
            logger.error(
                "Failed to save MARL configuration to %s: %s", config_path, str(e)
            )
            raise ValidationError(f"Failed to save MARL configuration: {str(e)}") from e


def get_default_marl_config() -> MARLConfig:
    """Get default MARL configuration for development and testing."""
    config = MARLConfig()

    # Set development-friendly defaults
    config.generator_config.max_episodes = 1000
    config.validator_config.max_episodes = 1000
    config.curriculum_config.max_episodes = 1000

    config.debug_mode = True
    config.log_level = "DEBUG"
    config.checkpoint_interval = 100

    return config
