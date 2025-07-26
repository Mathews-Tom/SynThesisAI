"""
MARL Configuration Module.

This module provides comprehensive configuration management for the
Multi-Agent Reinforcement Learning coordination system, including
hyperparameter management, validation, templates, and versioning.
"""

from .config_manager import MARLConfigManager
from .config_migration import ConfigMigrationManager
from .config_schema import AgentConfig as NewAgentConfig
from .config_schema import CoordinationConfig, LearningConfig, MARLConfig
from .config_templates import ConfigTemplateManager
from .config_validator import ConfigValidator

# Import the old AgentConfig and specialized configs from the legacy config file
try:
    from ..config import (
        AgentConfig,
        CurriculumAgentConfig,
        GeneratorAgentConfig,
        ValidatorAgentConfig,
    )
except ImportError:
    # Fallback: use the new AgentConfig and create compatible specialized configs
    AgentConfig = NewAgentConfig

    # Fallback: create compatible config classes that accept any parameters
    class GeneratorAgentConfig:
        """Fallback Generator agent configuration."""

        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    class ValidatorAgentConfig:
        """Fallback Validator agent configuration."""

        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    class CurriculumAgentConfig:
        """Fallback Curriculum agent configuration."""

        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)


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
