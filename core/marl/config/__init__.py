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
# Import the legacy AgentConfig and specialized configs from the config file
from ..config_legacy import (
    AgentConfig,
    CurriculumAgentConfig,
    GeneratorAgentConfig,
    ValidatorAgentConfig,
)
from .config_manager import MARLConfigManager
from .config_migration import ConfigMigrationManager
from .config_schema import AgentConfig as NewAgentConfig
from .config_schema import CoordinationConfig, LearningConfig, MARLConfig
from .config_templates import ConfigTemplateManager
from .config_validator import ConfigValidator

__all__: List[str] = [
    "AgentConfig",
    "ConfigMigrationManager",
    "ConfigTemplateManager",
    "ConfigValidator",
    "CoordinationConfig",
    "CurriculumAgentConfig",
    "GeneratorAgentConfig",
    "LearningConfig",
    "MARLConfig",
    "MARLConfigManager",
    "NewAgentConfig",
    "ValidatorAgentConfig",
]
