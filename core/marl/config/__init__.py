"""
MARL Configuration Module.

This module provides comprehensive configuration management for the
Multi-Agent Reinforcement Learning coordination system, including
hyperparameter management, validation, templates, and versioning.
"""

# Import the legacy AgentConfig and specialized configs from the config file
from ..config_legacy import (AgentConfig, CurriculumAgentConfig,
                             GeneratorAgentConfig, ValidatorAgentConfig)
from .config_manager import MARLConfigManager
from .config_migration import ConfigMigrationManager
from .config_schema import AgentConfig as NewAgentConfig
from .config_schema import CoordinationConfig, LearningConfig, MARLConfig
from .config_templates import ConfigTemplateManager
from .config_validator import ConfigValidator

__all__ = [
    "MARLConfigManager",
    "MARLConfig",
    "AgentConfig",
    "CoordinationConfig",
    "LearningConfig",
    "ConfigValidator",
    "ConfigTemplateManager",
    "ConfigMigrationManager",
    "GeneratorAgentConfig",
    "ValidatorAgentConfig",
    "CurriculumAgentConfig",
]
