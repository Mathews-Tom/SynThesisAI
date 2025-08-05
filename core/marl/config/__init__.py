# -*- coding: utf-8 -*-
"""Configuration management for the MARL system.

This module centralizes all configuration-related components for the Multi-Agent
Reinforcement Learning (MARL) coordination system. It exposes key classes for
managing, validating, migrating, and templating configurations, as well as the
core data structures that define the configuration schemas.

The module also handles the temporary aliasing of legacy and new configuration
models to support a phased migration.
"""

# Standard Library
from typing import List

# SynThesisAI Modules
from .config_manager import MARLConfigManager
from .config_migration import ConfigMigrationManager
from .config_schema import (
    AgentConfig,
    CommunicationConfig,
    ConsensusConfig,
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
from .config_templates import ConfigTemplateManager
from .config_validator import ConfigValidator


# Create aliases for specialized agent configs using the new AgentConfig
# These provide backward compatibility for specialized agent types
def create_curriculum_agent_config(agent_id: str = "curriculum") -> AgentConfig:
    """Create a curriculum agent configuration."""
    return AgentConfig(agent_id=agent_id, agent_type="curriculum")


def create_generator_agent_config(agent_id: str = "generator") -> AgentConfig:
    """Create a generator agent configuration."""
    return AgentConfig(agent_id=agent_id, agent_type="generator")


def create_validator_agent_config(agent_id: str = "validator") -> AgentConfig:
    """Create a validator agent configuration."""
    return AgentConfig(agent_id=agent_id, agent_type="validator")


# Backward compatibility aliases
CurriculumAgentConfig = create_curriculum_agent_config
GeneratorAgentConfig = create_generator_agent_config
ValidatorAgentConfig = create_validator_agent_config

# Experience config is now part of SharedLearningConfig
ExperienceConfig = SharedLearningConfig

__all__: List[str] = [
    "AgentConfig",
    "ConfigMigrationManager",
    "ConfigTemplateManager",
    "ConfigValidator",
    "CoordinationConfig",
    "LearningConfig",
    "MARLConfig",
    "MARLConfigManager",
    "NetworkConfig",
    "OptimizationConfig",
    "ExplorationConfig",
    "ReplayBufferConfig",
    "ConsensusConfig",
    "CommunicationConfig",
    "SharedLearningConfig",
    "SystemConfig",
    # Legacy compatibility (will be removed)
    "CurriculumAgentConfig",
    "GeneratorAgentConfig",
    "ValidatorAgentConfig",
    "ExperienceConfig",
]
