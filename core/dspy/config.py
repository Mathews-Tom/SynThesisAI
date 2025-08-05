"""
Configuration management for DSPy integration.

This module provides configuration management for DSPy integration,
including optimization parameters, quality requirements, and result tracking.
"""

# Standard Library
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Result of an optimization run."""

    optimized_module: Any  # STREAMContentGenerator
    optimization_metrics: Dict[str, Any]
    training_time: float
    validation_score: float
    cache_key: str
    timestamp: datetime


@dataclass
class DSPyModuleConfig:
    """Configuration for a DSPy module."""

    domain: str
    signature: str
    optimization_params: Dict[str, Any] = None
    quality_requirements: Dict[str, Any] = None
    training_data_path: str = None
    validation_data_path: str = None
    cache_enabled: bool = True
    cache_ttl: int = 86400  # 24 hours in seconds

    def __post_init__(self):
        """Initialize default values."""
        if self.optimization_params is None:
            self.optimization_params = {}
        if self.quality_requirements is None:
            self.quality_requirements = {"min_accuracy": 0.8}


@dataclass
class TrainingExample:
    """Training example for DSPy optimization."""

    inputs: Dict[str, Any]
    expected_outputs: Dict[str, Any]
    quality_score: float
    domain: str
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        """Initialize default values."""
        if self.metadata is None:
            self.metadata = {}


@dataclass
class DSPyConfig:
    """Configuration for DSPy integration."""

    # Optimization parameters
    max_concurrent_jobs: int = 3
    default_priority: int = 5
    optimization_timeout: int = 3600  # seconds
    enabled: bool = True  # Whether DSPy integration is enabled

    # Quality requirements
    default_quality_requirements: Dict[str, Any] = None

    # Caching parameters
    cache_enabled: bool = True
    cache_ttl: int = 86400  # seconds (24 hours)

    # Monitoring parameters
    metrics_history_size: int = 100

    # Version information
    version: str = "1.0.0"

    def __post_init__(self):
        """Initialize default values."""
        if self.default_quality_requirements is None:
            self.default_quality_requirements = {
                "min_accuracy": 0.8,
                "max_training_time": 300,  # seconds
            }

    def is_enabled(self) -> bool:
        """Check if DSPy integration is enabled."""
        return self.enabled

    def get_optimization_config(
        self, optimizer_type: str = "mipro_v2"
    ) -> Dict[str, Any]:
        """
        Get optimization configuration for specified optimizer.

        Args:
            optimizer_type: Type of optimizer

        Returns:
            Optimization configuration
        """
        # Default optimization configurations
        default_configs = {
            "mipro_v2": {
                "max_bootstrapped_demos": 2,
                "max_labeled_demos": 8,
                "init_temperature": 1.0,
                "optuna_trials_num": 20,
            },
            "bootstrap": {
                "max_bootstrapped_demos": 2,
                "max_labeled_demos": 8,
            },
            "copro": {
                "max_bootstrapped_demos": 2,
                "max_labeled_demos": 8,
            },
        }

        # Return default config for specified optimizer
        return default_configs.get(optimizer_type, {})

    def get_training_config(self) -> Dict[str, Any]:
        """
        Get training configuration.

        Returns:
            Training configuration dictionary
        """
        return {
            "min_quality_score": 0.5,
            "min_examples": 1,
            "max_examples": 100,
            "validation_split": 0.2,
        }

    def get_cache_config(self) -> Dict[str, Any]:
        """
        Get cache configuration.

        Returns:
            Cache configuration dictionary
        """
        return {
            "enabled": self.cache_enabled,
            "ttl": self.cache_ttl,
            "max_memory_entries": 1000,
            "max_disk_size_mb": 100,
        }

    def get_module_config(self, domain: str, signature: str) -> DSPyModuleConfig:
        """Get module configuration for specified domain and signature."""
        return DSPyModuleConfig(domain, signature)

    def get_dspy_version(self) -> str:
        """
        Get DSPy version string.

        Returns:
            DSPy version string or "0.0.0" if not available
        """
        try:
            import dspy

            return getattr(dspy, "__version__", "0.0.0")
        except ImportError:
            return "0.0.0"


# Global configuration instance
_dspy_config = None


def get_dspy_config() -> DSPyConfig:
    """Get the global DSPy configuration."""
    global _dspy_config
    if _dspy_config is None:
        _dspy_config = _load_config()
    return _dspy_config


def _load_config() -> DSPyConfig:
    """Load configuration from file or environment."""
    config = DSPyConfig()

    # Try to load from config file
    config_path = Path(os.environ.get("DSPY_CONFIG_PATH", "config/dspy_config.json"))
    if config_path.exists():
        try:
            config_data = json.loads(config_path.read_text(encoding="utf-8"))

            # Update configuration
            for key, value in config_data.items():
                if hasattr(config, key):
                    setattr(config, key, value)

            logger.info("Loaded DSPy configuration from %s", config_path)
        except Exception as e:
            logger.error("Failed to load DSPy configuration: %s", str(e))

    # Override with environment variables
    if "DSPY_MAX_CONCURRENT_JOBS" in os.environ:
        try:
            config.max_concurrent_jobs = int(os.environ["DSPY_MAX_CONCURRENT_JOBS"])
        except ValueError:
            pass

    if "DSPY_OPTIMIZATION_TIMEOUT" in os.environ:
        try:
            config.optimization_timeout = int(os.environ["DSPY_OPTIMIZATION_TIMEOUT"])
        except ValueError:
            pass

    if "DSPY_CACHE_ENABLED" in os.environ:
        config.cache_enabled = os.environ["DSPY_CACHE_ENABLED"].lower() == "true"

    return config


def save_config(config: DSPyConfig) -> bool:
    """
    Save configuration to file.

    Args:
        config: Configuration to save

    Returns:
        True if successful
    """
    config_path = Path(os.environ.get("DSPY_CONFIG_PATH", "config/dspy_config.json"))

    # Create directory if it doesn't exist
    config_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Convert dataclass to dict
        config_dict = {
            "max_concurrent_jobs": config.max_concurrent_jobs,
            "default_priority": config.default_priority,
            "optimization_timeout": config.optimization_timeout,
            "default_quality_requirements": config.default_quality_requirements,
            "cache_enabled": config.cache_enabled,
            "cache_ttl": config.cache_ttl,
            "metrics_history_size": config.metrics_history_size,
        }

        # Save to file
        config_path.write_text(json.dumps(config_dict, indent=2), encoding="utf-8")

        logger.info("Saved DSPy configuration to %s", config_path)
        return True
    except Exception as e:
        logger.error("Failed to save DSPy configuration: %s", str(e))
        return False
