"""
MARL Configuration Manager.

This module provides comprehensive configuration management for MARL systems,
including loading, saving, validation, and runtime configuration updates.
"""

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from utils.logging_config import get_logger

from .config_schema import MARLConfig
from .config_validator import ConfigValidator


class ConfigurationError(Exception):
    """Configuration management error."""

    pass


class MARLConfigManager:
    """
    Comprehensive configuration manager for MARL systems.

    Provides functionality for loading, saving, validating, and managing
    MARL configurations with support for multiple formats and environments.
    """

    def __init__(self, config_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the configuration manager.

        Args:
            config_dir: Directory for configuration files (default: ./configs)
        """
        self.logger = get_logger(__name__)
        self.config_dir = Path(config_dir) if config_dir else Path("configs")
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # Initialize validator
        self.validator = ConfigValidator()

        # Configuration cache
        self._config_cache: Dict[str, MARLConfig] = {}

        # Supported file formats
        self._supported_formats = {".json", ".yaml", ".yml"}

        self.logger.info(
            "MARL Configuration Manager initialized with config dir: %s",
            self.config_dir,
        )

    def load_config(
        self,
        config_path: Union[str, Path],
        validate: bool = True,
        use_cache: bool = True,
    ) -> MARLConfig:
        """
        Load configuration from file.

        Args:
            config_path: Path to configuration file
            validate: Whether to validate the configuration
            use_cache: Whether to use cached configuration if available

        Returns:
            Loaded MARL configuration

        Raises:
            ConfigurationError: If loading or validation fails
        """
        config_path = Path(config_path)

        # Check cache first
        cache_key = str(config_path.absolute())
        if use_cache and cache_key in self._config_cache:
            self.logger.debug("Loading configuration from cache: %s", config_path)
            return self._config_cache[cache_key]

        if not config_path.exists():
            raise ConfigurationError(f"Configuration file not found: {config_path}")

        if config_path.suffix not in self._supported_formats:
            raise ConfigurationError(
                f"Unsupported configuration format: {config_path.suffix}"
            )

        try:
            self.logger.info("Loading configuration from: %s", config_path)

            # Load configuration data
            with config_path.open("r", encoding="utf-8") as f:
                if config_path.suffix == ".json":
                    config_data = json.load(f)
                else:  # YAML
                    config_data = yaml.safe_load(f)

            # Create configuration object
            config = MARLConfig.from_dict(config_data)

            # Validate if requested
            if validate:
                errors, warnings = self.validator.validate_config(config)

                if errors:
                    error_msg = f"Configuration validation failed: {'; '.join(errors)}"
                    raise ConfigurationError(error_msg)

                if warnings:
                    self.logger.warning(
                        "Configuration warnings: %s", "; ".join(warnings)
                    )

            # Cache the configuration
            if use_cache:
                self._config_cache[cache_key] = config

            self.logger.info("Successfully loaded configuration: %s", config.name)
            return config

        except Exception as e:
            if isinstance(e, ConfigurationError):
                raise
            raise ConfigurationError(
                f"Failed to load configuration from {config_path}: {str(e)}"
            )

    def save_config(
        self,
        config: MARLConfig,
        config_path: Union[str, Path],
        format: str = "yaml",
        backup: bool = True,
        validate: bool = True,
    ) -> bool:
        """
        Save configuration to file.

        Args:
            config: MARL configuration to save
            config_path: Path to save configuration
            format: File format ('json' or 'yaml')
            backup: Whether to create backup if file exists
            validate: Whether to validate before saving

        Returns:
            True if saved successfully

        Raises:
            ConfigurationError: If saving fails
        """
        config_path = Path(config_path)

        if format not in ["json", "yaml"]:
            raise ConfigurationError(f"Unsupported format: {format}")

        try:
            # Validate if requested
            if validate:
                errors, warnings = self.validator.validate_config(config)

                if errors:
                    error_msg = f"Configuration validation failed: {'; '.join(errors)}"
                    raise ConfigurationError(error_msg)

                if warnings:
                    self.logger.warning(
                        "Configuration warnings: %s", "; ".join(warnings)
                    )

            # Create backup if file exists
            if backup and config_path.exists():
                backup_path = config_path.with_suffix(f"{config_path.suffix}.backup")
                shutil.copy2(config_path, backup_path)
                self.logger.debug("Created backup: %s", backup_path)

            # Ensure directory exists
            config_path.parent.mkdir(parents=True, exist_ok=True)

            # Update metadata
            config.created_at = datetime.now().isoformat()

            # Convert to dictionary
            config_data = config.to_dict()

            # Save configuration
            with config_path.open("w", encoding="utf-8") as f:
                if format == "json":
                    json.dump(config_data, f, indent=2, default=str)
                else:  # YAML
                    yaml.dump(config_data, f, default_flow_style=False, indent=2)

            # Update cache
            cache_key = str(config_path.absolute())
            self._config_cache[cache_key] = config

            self.logger.info("Successfully saved configuration: %s", config_path)
            return True

        except Exception as e:
            if isinstance(e, ConfigurationError):
                raise
            raise ConfigurationError(
                f"Failed to save configuration to {config_path}: {str(e)}"
            )

    def create_default_config(
        self,
        name: str = "default_marl_config",
        description: str = "Default MARL configuration",
    ) -> MARLConfig:
        """
        Create a default MARL configuration.

        Args:
            name: Configuration name
            description: Configuration description

        Returns:
            Default MARL configuration
        """
        from .config_schema import (
            AgentConfig,
            CoordinationConfig,
            LearningConfig,
            SystemConfig,
        )

        # Create default agents
        agents = {}

        # Generator agent
        agents["generator"] = AgentConfig(
            agent_id="generator", agent_type="generator", state_dim=256, action_dim=20
        )

        # Validator agent
        agents["validator"] = AgentConfig(
            agent_id="validator", agent_type="validator", state_dim=512, action_dim=10
        )

        # Curriculum agent
        agents["curriculum"] = AgentConfig(
            agent_id="curriculum", agent_type="curriculum", state_dim=128, action_dim=15
        )

        # Create configuration
        config = MARLConfig(
            name=name,
            description=description,
            agents=agents,
            coordination=CoordinationConfig(),
            learning=LearningConfig(),
            system=SystemConfig(),
        )

        self.logger.info("Created default configuration: %s", name)
        return config

    def list_configs(self, pattern: str = "*.yaml") -> List[Path]:
        """
        List available configuration files.

        Args:
            pattern: File pattern to match

        Returns:
            List of configuration file paths
        """
        config_files = list(self.config_dir.glob(pattern))
        config_files.extend(self.config_dir.glob("*.json"))

        # Filter by supported formats
        config_files = [f for f in config_files if f.suffix in self._supported_formats]

        return sorted(config_files)

    def validate_config_file(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Validate a configuration file without loading it into cache.

        Args:
            config_path: Path to configuration file

        Returns:
            Validation results dictionary
        """
        try:
            config = self.load_config(config_path, validate=False, use_cache=False)
            errors, warnings = self.validator.validate_config(config)

            return {
                "valid": len(errors) == 0,
                "errors": errors,
                "warnings": warnings,
                "config_name": config.name,
                "config_version": config.version,
            }

        except Exception as e:
            return {
                "valid": False,
                "errors": [str(e)],
                "warnings": [],
                "config_name": None,
                "config_version": None,
            }

    def compare_configs(
        self, config1_path: Union[str, Path], config2_path: Union[str, Path]
    ) -> Dict[str, Any]:
        """
        Compare two configuration files.

        Args:
            config1_path: Path to first configuration
            config2_path: Path to second configuration

        Returns:
            Comparison results dictionary
        """
        try:
            config1 = self.load_config(config1_path, validate=False, use_cache=False)
            config2 = self.load_config(config2_path, validate=False, use_cache=False)

            # Basic comparison
            differences = []

            # Compare versions
            if config1.version != config2.version:
                differences.append(f"Version: {config1.version} vs {config2.version}")

            # Compare agent counts
            if len(config1.agents) != len(config2.agents):
                differences.append(
                    f"Agent count: {len(config1.agents)} vs {len(config2.agents)}"
                )

            # Compare agent IDs
            agents1 = set(config1.agents.keys())
            agents2 = set(config2.agents.keys())

            if agents1 != agents2:
                only_in_1 = agents1 - agents2
                only_in_2 = agents2 - agents1

                if only_in_1:
                    differences.append(f"Agents only in config1: {list(only_in_1)}")
                if only_in_2:
                    differences.append(f"Agents only in config2: {list(only_in_2)}")

            # Check compatibility
            compat_errors, compat_warnings = (
                self.validator.validate_config_compatibility(config1, config2)
            )

            return {
                "compatible": len(compat_errors) == 0,
                "differences": differences,
                "compatibility_errors": compat_errors,
                "compatibility_warnings": compat_warnings,
                "config1_name": config1.name,
                "config2_name": config2.name,
            }

        except Exception as e:
            return {
                "compatible": False,
                "differences": [],
                "compatibility_errors": [str(e)],
                "compatibility_warnings": [],
                "config1_name": None,
                "config2_name": None,
            }

    def update_config(
        self, config: MARLConfig, updates: Dict[str, Any], validate: bool = True
    ) -> MARLConfig:
        """
        Update configuration with new values.

        Args:
            config: Original configuration
            updates: Dictionary of updates to apply
            validate: Whether to validate after updates

        Returns:
            Updated configuration

        Raises:
            ConfigurationError: If update fails
        """
        try:
            # Convert to dictionary for easier manipulation
            config_dict = config.to_dict()

            # Apply updates
            self._apply_nested_updates(config_dict, updates)

            # Create new configuration
            updated_config = MARLConfig.from_dict(config_dict)

            # Validate if requested
            if validate:
                errors, warnings = self.validator.validate_config(updated_config)

                if errors:
                    error_msg = (
                        f"Updated configuration validation failed: {'; '.join(errors)}"
                    )
                    raise ConfigurationError(error_msg)

                if warnings:
                    self.logger.warning(
                        "Updated configuration warnings: %s", "; ".join(warnings)
                    )

            self.logger.info(
                "Successfully updated configuration: %s", updated_config.name
            )
            return updated_config

        except Exception as e:
            if isinstance(e, ConfigurationError):
                raise
            raise ConfigurationError(f"Failed to update configuration: {str(e)}")

    def _apply_nested_updates(self, target: Dict[str, Any], updates: Dict[str, Any]):
        """Apply nested dictionary updates."""
        for key, value in updates.items():
            if (
                isinstance(value, dict)
                and key in target
                and isinstance(target[key], dict)
            ):
                self._apply_nested_updates(target[key], value)
            else:
                target[key] = value

    def get_config_summary(self, config: MARLConfig) -> Dict[str, Any]:
        """
        Get a summary of configuration key parameters.

        Args:
            config: MARL configuration

        Returns:
            Configuration summary dictionary
        """
        return {
            "name": config.name,
            "version": config.version,
            "description": config.description,
            "created_at": config.created_at,
            "num_agents": len(config.agents),
            "agent_types": list(
                set(agent.agent_type for agent in config.agents.values())
            ),
            "consensus_strategy": config.coordination.consensus.strategy.value,
            "shared_learning_enabled": config.learning.shared_learning.enabled,
            "device": config.system.device,
            "num_workers": config.system.num_workers,
            "total_replay_buffer_size": sum(
                agent.replay_buffer.capacity for agent in config.agents.values()
            ),
            "estimated_memory_gb": sum(
                agent.replay_buffer.capacity for agent in config.agents.values()
            )
            / (1024 * 1024),  # Rough estimate
        }

    def export_config_template(
        self,
        config: MARLConfig,
        template_path: Union[str, Path],
        include_comments: bool = True,
    ) -> bool:
        """
        Export configuration as a template with comments.

        Args:
            config: Configuration to export as template
            template_path: Path to save template
            include_comments: Whether to include explanatory comments

        Returns:
            True if exported successfully
        """
        try:
            template_path = Path(template_path)

            # Convert to dictionary
            config_dict = config.to_dict()

            # Add comments if requested
            if include_comments:
                config_dict = self._add_template_comments(config_dict)

            # Save as YAML with comments
            with template_path.open("w", encoding="utf-8") as f:
                f.write("# MARL Configuration Template\n")
                f.write(f"# Generated from: {config.name}\n")
                f.write(f"# Created at: {datetime.now().isoformat()}\n\n")

                yaml.dump(config_dict, f, default_flow_style=False, indent=2)

            self.logger.info("Exported configuration template: %s", template_path)
            return True

        except Exception as e:
            self.logger.error("Failed to export template: %s", str(e))
            return False

    def _add_template_comments(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Add explanatory comments to configuration dictionary."""
        # This would add YAML comments - simplified for now
        # In a full implementation, you'd use a YAML library that preserves comments
        return config_dict

    def clear_cache(self):
        """Clear the configuration cache."""
        self._config_cache.clear()
        self.logger.debug("Configuration cache cleared")

    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about the configuration cache."""
        return {
            "cached_configs": len(self._config_cache),
            "cache_keys": list(self._config_cache.keys()),
        }


class MARLConfigManagerFactory:
    """Factory for creating MARL configuration managers."""

    @staticmethod
    def create(config_dir: Optional[Union[str, Path]] = None) -> MARLConfigManager:
        """
        Create a MARL configuration manager.

        Args:
            config_dir: Configuration directory

        Returns:
            Configured MARL configuration manager
        """
        return MARLConfigManager(config_dir)

    @staticmethod
    def create_with_validation(
        config_dir: Optional[Union[str, Path]] = None, strict_validation: bool = True
    ) -> MARLConfigManager:
        """
        Create a MARL configuration manager with custom validation settings.

        Args:
            config_dir: Configuration directory
            strict_validation: Whether to use strict validation

        Returns:
            Configured MARL configuration manager
        """
        manager = MARLConfigManager(config_dir)

        # Configure validator for strict mode if needed
        if strict_validation:
            manager.validator._warning_thresholds.update(
                {
                    "high_learning_rate": 0.005,  # More strict
                    "low_learning_rate": 1e-5,
                    "large_replay_buffer": 500000,
                    "small_replay_buffer": 5000,
                }
            )

        return manager
