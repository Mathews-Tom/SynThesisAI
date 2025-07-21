"""
DSPy Configuration Management

This module handles configuration for DSPy integration, including module settings,
optimization parameters, and caching configuration.
"""

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from utils.config_manager import get_config_manager

logger = logging.getLogger(__name__)


@dataclass
class DSPyModuleConfig:
    """Configuration for a DSPy module."""

    domain: str
    signature: str
    optimization_params: Dict[str, Any] = field(default_factory=dict)
    quality_requirements: Dict[str, Any] = field(default_factory=dict)
    training_data_path: Optional[str] = None
    validation_data_path: Optional[str] = None
    cache_enabled: bool = True
    cache_ttl: int = 86400  # 24 hours in seconds


@dataclass
class OptimizationResult:
    """Result of DSPy optimization."""

    optimized_module: Any  # STREAMContentGenerator
    optimization_metrics: Dict[str, float]
    training_time: float
    validation_score: float
    cache_key: str
    timestamp: datetime


@dataclass
class TrainingExample:
    """Training example for DSPy optimization."""

    inputs: Dict[str, Any]
    expected_outputs: Dict[str, Any]
    quality_score: float
    domain: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class DSPyConfig:
    """
    Central configuration manager for DSPy integration.

    Handles loading and managing DSPy-specific configuration settings,
    including optimization parameters, caching settings, and model configurations.
    """

    def __init__(self):
        """Initialize DSPy configuration."""
        self.config_manager = get_config_manager()
        self.logger = logging.getLogger(f"{__name__}.DSPyConfig")

        # Load DSPy configuration
        self.dspy_config = self.config_manager.get("dspy", {})

        # Set up default configuration
        self._setup_defaults()

        # Initialize paths
        self._setup_paths()

        self.logger.info("DSPy configuration initialized")

    def _setup_defaults(self):
        """Set up default DSPy configuration values."""
        defaults = {
            "enabled": True,
            "cache_dir": ".cache/dspy",
            "cache_ttl": 86400,  # 24 hours
            "optimization": {
                "mipro_v2": {
                    "optuna_trials_num": 100,
                    "max_bootstrapped_demos": 4,
                    "max_labeled_demos": 16,
                    "num_candidate_programs": 16,
                    "init_temperature": 1.4,
                },
                "bootstrap": {"max_bootstrapped_demos": 4, "max_labeled_demos": 16},
            },
            "training_data": {
                "min_examples": 50,
                "max_examples": 1000,
                "validation_split": 0.2,
            },
            "quality_requirements": {
                "min_accuracy": 0.8,
                "min_coherence": 0.7,
                "min_relevance": 0.8,
            },
            "fallback": {
                "enabled": True,
                "max_retries": 3,
                "timeout": 300,  # 5 minutes
            },
        }

        # Merge with existing config
        for key, value in defaults.items():
            if key not in self.dspy_config:
                self.dspy_config[key] = value
            elif isinstance(value, dict) and isinstance(self.dspy_config[key], dict):
                for subkey, subvalue in value.items():
                    if subkey not in self.dspy_config[key]:
                        self.dspy_config[key][subkey] = subvalue

    def _setup_paths(self):
        """Set up DSPy-related paths."""
        self.cache_dir = Path(self.dspy_config.get("cache_dir", ".cache/dspy"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.training_data_dir = Path("data/training")
        self.training_data_dir.mkdir(parents=True, exist_ok=True)

        self.validation_data_dir = Path("data/validation")
        self.validation_data_dir.mkdir(parents=True, exist_ok=True)

        self.logger.debug(
            "DSPy paths initialized: cache=%s, training=%s",
            self.cache_dir,
            self.training_data_dir,
        )

    def is_enabled(self) -> bool:
        """Check if DSPy integration is enabled."""
        return self.dspy_config.get("enabled", True)

    def get_cache_config(self) -> Dict[str, Any]:
        """Get caching configuration."""
        return {
            "cache_dir": str(self.cache_dir),
            "cache_ttl": self.dspy_config.get("cache_ttl", 86400),
            "enabled": self.dspy_config.get("cache_enabled", True),
        }

    def get_optimization_config(
        self, optimizer_type: str = "mipro_v2"
    ) -> Dict[str, Any]:
        """Get optimization configuration for specified optimizer."""
        optimization_config = self.dspy_config.get("optimization", {})
        return optimization_config.get(optimizer_type, {})

    def get_training_config(self) -> Dict[str, Any]:
        """Get training data configuration."""
        return self.dspy_config.get("training_data", {})

    def get_quality_requirements(self) -> Dict[str, Any]:
        """Get quality requirements configuration."""
        return self.dspy_config.get("quality_requirements", {})

    def get_fallback_config(self) -> Dict[str, Any]:
        """Get fallback configuration."""
        return self.dspy_config.get("fallback", {})

    def get_module_config(self, domain: str, signature: str) -> DSPyModuleConfig:
        """
        Get configuration for a specific DSPy module.

        Args:
            domain: The STREAM domain (e.g., 'mathematics', 'science')
            signature: The DSPy signature string

        Returns:
            DSPyModuleConfig instance
        """
        domain_config = self.dspy_config.get("domains", {}).get(domain, {})

        return DSPyModuleConfig(
            domain=domain,
            signature=signature,
            optimization_params=domain_config.get("optimization_params", {}),
            quality_requirements=domain_config.get(
                "quality_requirements", self.get_quality_requirements()
            ),
            training_data_path=str(self.training_data_dir / f"{domain}_training.json"),
            validation_data_path=str(
                self.validation_data_dir / f"{domain}_validation.json"
            ),
            cache_enabled=domain_config.get("cache_enabled", True),
            cache_ttl=domain_config.get(
                "cache_ttl", self.dspy_config.get("cache_ttl", 86400)
            ),
        )

    def update_config(self, updates: Dict[str, Any]):
        """
        Update DSPy configuration.

        Args:
            updates: Dictionary of configuration updates
        """
        self.dspy_config.update(updates)
        self.logger.info("DSPy configuration updated: %s", list(updates.keys()))

    def get_dspy_version(self) -> str:
        """Get DSPy version for cache key generation."""
        try:
            import dspy

            return getattr(dspy, "__version__", "unknown")
        except ImportError:
            return "not_installed"


# Global configuration instance
_dspy_config = None


def get_dspy_config() -> DSPyConfig:
    """Get the global DSPy configuration instance."""
    global _dspy_config
    if _dspy_config is None:
        _dspy_config = DSPyConfig()
    return _dspy_config
