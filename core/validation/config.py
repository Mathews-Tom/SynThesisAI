"""
Configuration management for the STREAM domain validation system.

This module provides configuration classes and utilities for managing
validation thresholds, rules, and system parameters.
"""

# Standard Library
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# SynThesisAI Modules
from .exceptions import ValidationConfigError

logger = logging.getLogger(__name__)


@dataclass
class ValidationConfig:
    """Configuration for domain validation system."""

    domain: str
    quality_thresholds: Dict[str, float] = field(default_factory=dict)
    validation_rules: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: int = 30
    max_retries: int = 3
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600
    performance_monitoring: bool = True

    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_config()

    def _validate_config(self):
        """Validate configuration parameters."""
        if not self.domain:
            raise ValidationConfigError("domain", "Domain cannot be empty")

        if self.timeout_seconds <= 0:
            raise ValidationConfigError("timeout_seconds", "Timeout must be positive")

        if self.max_retries < 0:
            raise ValidationConfigError("max_retries", "Max retries cannot be negative")

        # Validate quality thresholds are between 0 and 1
        for metric, threshold in self.quality_thresholds.items():
            if not 0 <= threshold <= 1:
                raise ValidationConfigError(
                    f"quality_thresholds.{metric}",
                    f"Threshold {threshold} must be between 0 and 1",
                )


@dataclass
class UniversalValidationConfig:
    """Configuration for the universal validation orchestrator."""

    enabled_domains: List[str] = field(
        default_factory=lambda: [
            "mathematics",
            "science",
            "technology",
            "reading",
            "engineering",
            "arts",
        ]
    )
    global_timeout_seconds: int = 60
    parallel_validation: bool = True
    max_concurrent_validations: int = 5
    fallback_on_error: bool = True
    quality_aggregation_method: str = "weighted_average"

    def __post_init__(self):
        """Validate universal configuration after initialization."""
        self._validate_config()

    def _validate_config(self):
        """Validate universal configuration parameters."""
        valid_domains = {
            "mathematics",
            "science",
            "technology",
            "reading",
            "engineering",
            "arts",
        }
        invalid_domains = set(self.enabled_domains) - valid_domains
        if invalid_domains:
            raise ValidationConfigError(
                "enabled_domains",
                f"Invalid domains: {invalid_domains}. Valid domains: {valid_domains}",
            )

        if self.global_timeout_seconds <= 0:
            raise ValidationConfigError(
                "global_timeout_seconds", "Global timeout must be positive"
            )

        if self.max_concurrent_validations <= 0:
            raise ValidationConfigError(
                "max_concurrent_validations",
                "Max concurrent validations must be positive",
            )

        valid_aggregation_methods = {"weighted_average", "minimum", "maximum", "median"}
        if self.quality_aggregation_method not in valid_aggregation_methods:
            raise ValidationConfigError(
                "quality_aggregation_method",
                f"Invalid aggregation method. Valid methods: {valid_aggregation_methods}",
            )


class ValidationConfigManager:
    """Manager for loading and managing validation configurations."""

    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize configuration manager.

        Args:
            config_dir: Directory containing validation configuration files
        """
        self.config_dir = config_dir or Path("config/validation")
        self.logger = logging.getLogger(__name__ + ".ValidationConfigManager")
        self._domain_configs: Dict[str, ValidationConfig] = {}
        self._universal_config: Optional[UniversalValidationConfig] = None

    def load_domain_config(self, domain: str) -> ValidationConfig:
        """
        Load configuration for a specific domain.

        Args:
            domain: The STREAM domain to load configuration for

        Returns:
            ValidationConfig for the specified domain

        Raises:
            ValidationConfigError: If configuration cannot be loaded
        """
        if domain in self._domain_configs:
            return self._domain_configs[domain]

        config_file = self.config_dir / f"{domain}_validation.json"

        try:
            if config_file.exists():
                with open(config_file, "r", encoding="utf-8") as f:
                    config_data = json.load(f)

                config = ValidationConfig(domain=domain, **config_data)
                self._domain_configs[domain] = config

                self.logger.info("Loaded validation config for domain %s", domain)
                return config
            else:
                # Create default configuration
                config = self._create_default_domain_config(domain)
                self._domain_configs[domain] = config

                # Save default configuration
                self._save_domain_config(domain, config)

                self.logger.info(
                    "Created default validation config for domain %s", domain
                )
                return config

        except Exception as e:
            error_msg = f"Failed to load configuration for domain {domain}"
            self.logger.error("%s: %s", error_msg, str(e))
            raise ValidationConfigError(f"domain_config.{domain}", error_msg) from e

    def load_universal_config(self) -> UniversalValidationConfig:
        """
        Load universal validation configuration.

        Returns:
            UniversalValidationConfig for the validation system

        Raises:
            ValidationConfigError: If configuration cannot be loaded
        """
        if self._universal_config:
            return self._universal_config

        config_file = self.config_dir / "universal_validation.json"

        try:
            if config_file.exists():
                with open(config_file, "r", encoding="utf-8") as f:
                    config_data = json.load(f)

                config = UniversalValidationConfig(**config_data)
                self._universal_config = config

                self.logger.info("Loaded universal validation config")
                return config
            else:
                # Create default configuration
                config = UniversalValidationConfig()
                self._universal_config = config

                # Save default configuration
                self._save_universal_config(config)

                self.logger.info("Created default universal validation config")
                return config

        except Exception as e:
            error_msg = "Failed to load universal validation configuration"
            self.logger.error("%s: %s", error_msg, str(e))
            raise ValidationConfigError("universal_config", error_msg) from e

    def _create_default_domain_config(self, domain: str) -> ValidationConfig:
        """Create default configuration for a domain."""
        default_thresholds = {
            "fidelity_score": 0.8,
            "utility_score": 0.7,
            "safety_score": 0.9,
            "pedagogical_score": 0.75,
        }

        return ValidationConfig(
            domain=domain,
            quality_thresholds=default_thresholds,
            validation_rules={},
            timeout_seconds=30,
            max_retries=3,
            cache_enabled=True,
            cache_ttl_seconds=3600,
            performance_monitoring=True,
        )

    def _save_domain_config(self, domain: str, config: ValidationConfig):
        """Save domain configuration to file."""
        try:
            self.config_dir.mkdir(parents=True, exist_ok=True)
            config_file = self.config_dir / f"{domain}_validation.json"

            config_data = {
                "quality_thresholds": config.quality_thresholds,
                "validation_rules": config.validation_rules,
                "timeout_seconds": config.timeout_seconds,
                "max_retries": config.max_retries,
                "cache_enabled": config.cache_enabled,
                "cache_ttl_seconds": config.cache_ttl_seconds,
                "performance_monitoring": config.performance_monitoring,
            }

            with open(config_file, "w", encoding="utf-8") as f:
                json.dump(config_data, f, indent=2)

        except Exception as e:
            self.logger.warning(
                "Failed to save config for domain %s: %s", domain, str(e)
            )

    def _save_universal_config(self, config: UniversalValidationConfig):
        """Save universal configuration to file."""
        try:
            self.config_dir.mkdir(parents=True, exist_ok=True)
            config_file = self.config_dir / "universal_validation.json"

            config_data = {
                "enabled_domains": config.enabled_domains,
                "global_timeout_seconds": config.global_timeout_seconds,
                "parallel_validation": config.parallel_validation,
                "max_concurrent_validations": config.max_concurrent_validations,
                "fallback_on_error": config.fallback_on_error,
                "quality_aggregation_method": config.quality_aggregation_method,
            }

            with open(config_file, "w", encoding="utf-8") as f:
                json.dump(config_data, f, indent=2)

        except Exception as e:
            self.logger.warning("Failed to save universal config: %s", str(e))


# Global configuration manager instance
_config_manager = None


def get_config_manager() -> ValidationConfigManager:
    """Get the global validation configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ValidationConfigManager()
    return _config_manager
