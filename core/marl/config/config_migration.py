"""MARL Configuration Migration.

This module provides configuration migration and versioning capabilities
to handle configuration schema changes across different versions.
"""
from __future__ import annotations

# Standard Library
import re
from collections import deque
from typing import Any

# Third-Party Library
from packaging import version

# SynThesisAI Modules
from utils.logging_config import get_logger


class MigrationError(Exception):
    """Custom exception for configuration migration errors."""

    pass


class ConfigMigration:
    """Base class for configuration migrations."""

    def __init__(self, from_version: str, to_version: str):
        """Initializes the migration.

        Args:
            from_version: The source version for the migration.
            to_version: The target version for the migration.
        """
        self.from_version = from_version
        self.to_version = to_version
        self.logger = get_logger(__name__)

    def can_migrate(self, config_version: str) -> bool:
        """Checks if this migration can handle the given configuration version.

        Args:
            config_version: The version of the configuration to check.

        Returns:
            True if the migration can handle the version, False otherwise.
        """
        return config_version == self.from_version

    def migrate(self, config_data: dict[str, Any]) -> dict[str, Any]:
        """Performs the configuration migration.

        This method must be implemented by subclasses.

        Args:
            config_data: The configuration data to migrate.

        Returns:
            The migrated configuration data.

        Raises:
            NotImplementedError: If the method is not overridden by a subclass.
        """
        raise NotImplementedError("Subclasses must implement the migrate method.")

    def validate_migration(self, original: dict[str, Any], migrated: dict[str, Any]) -> bool:
        """Validates that the migration was successful.

        Args:
            original: The original configuration data.
            migrated: The migrated configuration data.

        Returns:
            True if the migration is considered valid, False otherwise.
        """
        # Basic validation: Ensure essential fields are preserved.
        return (
            migrated.get("name") == original.get("name")
            and migrated.get("version") == self.to_version
            and "agents" in migrated
        )


class Migration_1_0_0_to_1_1_0(ConfigMigration):
    """Migration from version 1.0.0 to 1.1.0."""

    def __init__(self):
        """Initializes the migration from version 1.0.0 to 1.1.0."""
        super().__init__("1.0.0", "1.1.0")

    def migrate(self, config_data: dict[str, Any]) -> dict[str, Any]:
        """Migrates the configuration from version 1.0.0 to 1.1.0.

        Args:
            config_data: The configuration data to migrate.

        Returns:
            The migrated configuration data.
        """
        self.logger.info(
            "Migrating configuration from %s to %s", self.from_version, self.to_version
        )
        migrated = config_data.copy()
        migrated["version"] = self.to_version

        if "system" in migrated:
            system = migrated["system"]
            system.setdefault("deterministic", False)
            system.setdefault("seed", None)

        if "agents" in migrated:
            for agent_config in migrated["agents"].values():
                agent_config.setdefault("reward_scaling", 1.0)
                if "network" in agent_config:
                    agent_config["network"].setdefault("weight_initialization", "xavier_uniform")

        if "learning" in migrated:
            migrated["learning"].setdefault("trend_analysis_enabled", True)

        return migrated


class Migration_1_1_0_to_1_2_0(ConfigMigration):
    """Migration from version 1.1.0 to 1.2.0."""

    def __init__(self):
        """Initializes the migration from version 1.1.0 to 1.2.0."""
        super().__init__("1.1.0", "1.2.0")

    def migrate(self, config_data: dict[str, Any]) -> dict[str, Any]:
        """Migrates the configuration from version 1.1.0 to 1.2.0.

        Args:
            config_data: The configuration data to migrate.

        Returns:
            The migrated configuration data.
        """
        self.logger.info(
            "Migrating configuration from %s to %s", self.from_version, self.to_version
        )
        migrated = config_data.copy()
        migrated["version"] = self.to_version

        if "learning" in migrated:
            learning = migrated["learning"]
            learning.setdefault("performance_tracking_enabled", True)
            learning.setdefault("metrics_collection_interval", 1.0)
            if "shared_learning" in learning:
                shared_learning = learning["shared_learning"]
                shared_learning.setdefault("continuous_learning_enabled", True)
                shared_learning.setdefault("learning_update_interval", 10.0)
                shared_learning.setdefault("performance_window_size", 100)
                shared_learning.setdefault("adaptation_threshold", 0.05)

        if "coordination" in migrated:
            coordination = migrated["coordination"]
            coordination.setdefault("deadlock_detection_enabled", True)
            coordination.setdefault("deadlock_timeout", 120.0)

        return migrated


class Migration_1_2_0_to_2_0_0(ConfigMigration):
    """Migration from version 1.2.0 to 2.0.0, which includes breaking changes."""

    def __init__(self):
        """Initializes the migration from version 1.2.0 to 2.0.0."""
        super().__init__("1.2.0", "2.0.0")

    def migrate(self, config_data: dict[str, Any]) -> dict[str, Any]:
        """Migrates the configuration from version 1.2.0 to 2.0.0.

        Args:
            config_data: The configuration data to migrate.

        Returns:
            The migrated configuration data.
        """
        self.logger.info(
            "Migrating configuration from %s to %s", self.from_version, self.to_version
        )
        migrated = config_data.copy()
        migrated["version"] = self.to_version

        # Breaking change: Restructure consensus configuration
        if "coordination" in migrated and "consensus" in migrated["coordination"]:
            consensus = migrated["coordination"]["consensus"]
            strategy_mapping = {
                "majority": "majority_vote",
                "weighted": "weighted_average",
                "expert": "expert_priority",
            }
            if "strategy" in consensus and consensus["strategy"] in strategy_mapping:
                consensus["strategy"] = strategy_mapping[consensus["strategy"]]

        # Breaking change: Update exploration strategy names
        if "agents" in migrated:
            for agent_config in migrated["agents"].values():
                if "exploration" in agent_config:
                    exploration = agent_config["exploration"]
                    exploration_mapping = {
                        "epsilon": "epsilon_greedy",
                        "boltzmann": "boltzmann",
                        "ucb1": "ucb",
                    }
                    if "strategy" in exploration and exploration["strategy"] in exploration_mapping:
                        exploration["strategy"] = exploration_mapping[exploration["strategy"]]

        migrated.setdefault("environment", {})
        return migrated


class ConfigMigrationManager:
    """Manages configuration migrations and versioning.

    This class handles the automatic migration of configurations between different
    versions and provides version compatibility checking.
    """

    def __init__(self):
        """Initializes the migration manager."""
        self.logger = get_logger(__name__)
        self._migrations: list[ConfigMigration] = [
            Migration_1_0_0_to_1_1_0(),
            Migration_1_1_0_to_1_2_0(),
            Migration_1_2_0_to_2_0_0(),
        ]
        self.current_version = "2.0.0"
        self._compatibility_matrix = {
            "1.0.0": ["1.1.0", "1.2.0", "2.0.0"],
            "1.1.0": ["1.2.0", "2.0.0"],
            "1.2.0": ["2.0.0"],
            "2.0.0": [],
        }
        self.logger.info(
            "Configuration migration manager initialized with %d migrations.",
            len(self._migrations),
        )

    def get_supported_versions(self) -> list[str]:
        """Gets a list of all supported configuration versions.

        Returns:
            A sorted list of supported version strings.
        """
        versions = set()
        for migration in self._migrations:
            versions.add(migration.from_version)
            versions.add(migration.to_version)
        return sorted(list(versions), key=version.parse)

    def is_version_supported(self, config_version: str) -> bool:
        """Checks if a configuration version is supported.

        Args:
            config_version: The configuration version to check.

        Returns:
            True if the version is supported, False otherwise.
        """
        return config_version in self.get_supported_versions()

    def needs_migration(self, config_version: str, target_version: str | None = None) -> bool:
        """Checks if a configuration needs to be migrated.

        Args:
            config_version: The current version of the configuration.
            target_version: The target version. Defaults to the current version.

        Returns:
            True if migration is needed, False otherwise.
        """
        target = target_version or self.current_version
        return config_version != target and self.can_migrate(config_version, target)

    def can_migrate(self, from_version: str, to_version: str | None = None) -> bool:
        """Checks if a migration path exists between two versions.

        Args:
            from_version: The source version.
            to_version: The target version. Defaults to the current version.

        Returns:
            True if a migration path exists, False otherwise.
        """
        target = to_version or self.current_version
        if from_version == target:
            return True
        return self._find_migration_path(from_version, target) is not None

    def migrate_config(
        self,
        config_data: dict[str, Any],
        target_version: str | None = None,
        validate: bool = True,
    ) -> tuple[dict[str, Any], list[str]]:
        """Migrates a configuration to the target version.

        Args:
            config_data: The configuration data to migrate.
            target_version: The target version. Defaults to the current version.
            validate: Whether to validate each migration step.

        Returns:
            A tuple containing the migrated configuration and a log of migration steps.

        Raises:
            MigrationError: If no migration path is found or if a migration step fails.
        """
        target = target_version or self.current_version
        current_version = config_data.get("version", "1.0.0")

        if current_version == target:
            return config_data, ["No migration needed."]

        migration_path = self._find_migration_path(current_version, target)
        if not migration_path:
            raise MigrationError(f"No migration path found from {current_version} to {target}.")

        migrated_data = config_data.copy()
        migration_log = []

        for migration in migration_path:
            self.logger.info(
                "Applying migration from %s to %s.",
                migration.from_version,
                migration.to_version,
            )
            original_data = migrated_data.copy()
            try:
                migrated_data = migration.migrate(migrated_data)
                if validate and not migration.validate_migration(original_data, migrated_data):
                    raise MigrationError(
                        (
                            "Migration validation failed for "
                            f"{migration.from_version} -> {migration.to_version}."
                        )
                    )
                migration_log.append(
                    (
                        "Successfully migrated from "
                        f"{migration.from_version} to {migration.to_version}."
                    )
                )
            except (KeyError, TypeError) as e:
                self.logger.error(
                    "Migration failed from %s to %s: %s",
                    migration.from_version,
                    migration.to_version,
                    e,
                )
                raise MigrationError(
                    f"Migration failed ({migration.from_version} -> {migration.to_version}): {e}"
                ) from e

        self.logger.info(
            "Successfully migrated configuration from %s to %s.", current_version, target
        )
        return migrated_data, migration_log

    def _find_migration_path(
        self, from_version: str, to_version: str
    ) -> list[ConfigMigration] | None:
        """Finds the shortest migration path between two versions using BFS.

        Args:
            from_version: The source version.
            to_version: The target version.

        Returns:
            A list of `ConfigMigration` objects representing the path, or None if no path exists.
        """
        if from_version == to_version:
            return []

        queue = deque([(from_version, [])])
        visited = {from_version}

        while queue:
            current_v, path = queue.popleft()
            for migration in self._migrations:
                if migration.can_migrate(current_v):
                    next_v = migration.to_version
                    new_path = path + [migration]
                    if next_v == to_version:
                        return new_path
                    if next_v not in visited:
                        visited.add(next_v)
                        queue.append((next_v, new_path))
        return None

    def get_migration_info(
        self, from_version: str, to_version: str | None = None
    ) -> dict[str, Any]:
        """Gets information about the migration between two versions.

        Args:
            from_version: The source version.
            to_version: The target version. Defaults to the current version.

        Returns:
            A dictionary containing migration information.
        """
        target = to_version or self.current_version
        migration_path = self._find_migration_path(from_version, target)

        if migration_path is None:
            return {
                "possible": False,
                "reason": f"No migration path from {from_version} to {target}",
                "steps": [],
                "breaking_changes": False,
            }

        breaking_changes = any(
            self._is_breaking_change(m.from_version, m.to_version) for m in migration_path
        )
        steps = [
            {
                "from": m.from_version,
                "to": m.to_version,
                "breaking": self._is_breaking_change(m.from_version, m.to_version),
            }
            for m in migration_path
        ]

        return {
            "possible": True,
            "steps": steps,
            "breaking_changes": breaking_changes,
            "total_steps": len(migration_path),
        }

    def _is_breaking_change(self, from_version: str, to_version: str) -> bool:
        """Checks if a migration between versions is a major (breaking) change.

        Args:
            from_version: The source version.
            to_version: The target version.

        Returns:
            True if it is a breaking change, False otherwise.
        """
        from_major = int(from_version.split(".")[0])
        to_major = int(to_version.split(".")[0])
        return to_major > from_major

    def validate_version_format(self, version_string: str) -> bool:
        """Validates the format of a version string (e.g., '1.2.3').

        Args:
            version_string: The version string to validate.

        Returns:
            True if the format is valid, False otherwise.
        """
        return bool(re.match(r"^\d+\.\d+\.\d+$", version_string))

    def compare_versions(self, version1: str, version2: str) -> int:
        """Compares two version strings.

        Args:
            version1: The first version string.
            version2: The second version string.

        Returns:
            -1 if version1 < version2, 0 if equal, 1 if version1 > version2.
        """
        try:
            v1 = version.parse(version1)
            v2 = version.parse(version2)
            if v1 < v2:
                return -1
            if v1 > v2:
                return 1
            return 0
        except version.InvalidVersion:
            # Fallback to string comparison for non-standard versions
            if version1 < version2:
                return -1
            if version1 > version2:
                return 1
            return 0

    def get_latest_version(self) -> str:
        """Gets the latest supported version.

        Returns:
            The latest version string.
        """
        return self.current_version

    def create_migration_report(
        self,
        original_config: dict[str, Any],
        migrated_config: dict[str, Any],
        migration_log: list[str],
    ) -> dict[str, Any]:
        """Creates a detailed report of a migration.

        Args:
            original_config: The original configuration.
            migrated_config: The migrated configuration.
            migration_log: A log of migration steps.

        Returns:
            A dictionary containing the migration report.
        """
        return {
            "original_version": original_config.get("version", "unknown"),
            "migrated_version": migrated_config.get("version", "unknown"),
            "migration_steps": migration_log,
            "changes_summary": {
                "agents_count": {
                    "before": len(original_config.get("agents", {})),
                    "after": len(migrated_config.get("agents", {})),
                },
                "new_fields_added": self._find_new_fields(original_config, migrated_config),
                "fields_removed": self._find_removed_fields(original_config, migrated_config),
                "fields_renamed": self._find_renamed_fields(original_config, migrated_config),
            },
            "compatibility_notes": self._generate_compatibility_notes(
                original_config.get("version", "unknown"),
                migrated_config.get("version", "unknown"),
            ),
        }

    def _find_new_fields(self, original: dict[str, Any], migrated: dict[str, Any]) -> list[str]:
        """Finds fields that were added during migration.

        Args:
            original: The original configuration dictionary.
            migrated: The migrated configuration dictionary.

        Returns:
            A list of paths to the new fields.
        """
        new_fields = []

        def compare_dicts(orig, migr, path=""):
            for key, value in migr.items():
                current_path = f"{path}.{key}" if path else key
                if key not in orig:
                    new_fields.append(current_path)
                elif isinstance(value, dict) and isinstance(orig.get(key), dict):
                    compare_dicts(orig[key], value, current_path)

        compare_dicts(original, migrated)
        return new_fields

    def _find_removed_fields(self, original: dict[str, Any], migrated: dict[str, Any]) -> list[str]:
        """Finds fields that were removed during migration.

        Args:
            original: The original configuration dictionary.
            migrated: The migrated configuration dictionary.

        Returns:
            A list of paths to the removed fields.
        """
        removed_fields = []

        def compare_dicts(orig, migr, path=""):
            for key, value in orig.items():
                current_path = f"{path}.{key}" if path else key
                if key not in migr:
                    removed_fields.append(current_path)
                elif isinstance(value, dict) and isinstance(migr.get(key), dict):
                    compare_dicts(value, migr[key], current_path)

        compare_dicts(original, migrated)
        return removed_fields

    def _find_renamed_fields(
        self, original: dict[str, Any], migrated: dict[str, Any]
    ) -> list[dict[str, str]]:
        """Finds fields that were likely renamed during migration.

        Note: This is a simplified implementation and may not be fully accurate.

        Args:
            original: The original configuration dictionary.
            migrated: The migrated configuration dictionary.

        Returns:
            A list of dictionaries, each representing a renamed field.
        """
        # This is a complex task; a simple implementation is a placeholder.
        return []

    def _generate_compatibility_notes(self, from_version: str, to_version: str) -> list[str]:
        """Generates compatibility notes for a migration.

        Args:
            from_version: The source version.
            to_version: The target version.

        Returns:
            A list of compatibility notes.
        """
        notes = []
        if self._is_breaking_change(from_version, to_version):
            notes.append("This migration includes breaking changes that may require code updates.")
        if from_version.startswith("1.") and to_version.startswith("2."):
            notes.append(
                (
                    "Migration to version 2.x includes restructured consensus and "
                    "exploration configurations."
                )
            )
        return notes


class ConfigMigrationManagerFactory:
    """Factory for creating `ConfigMigrationManager` instances."""

    @staticmethod
    def create() -> ConfigMigrationManager:
        """Creates a standard `ConfigMigrationManager`.

        Returns:
            A new instance of `ConfigMigrationManager`.
        """
        return ConfigMigrationManager()

    @staticmethod
    def create_with_custom_migrations(
        custom_migrations: list[ConfigMigration],
    ) -> ConfigMigrationManager:
        """Creates a `ConfigMigrationManager` with additional custom migrations.

        Args:
            custom_migrations: A list of custom migration objects.

        Returns:
            A new instance of `ConfigMigrationManager` with custom migrations.
        """
        manager = ConfigMigrationManager()
        manager._migrations.extend(custom_migrations)
        return manager
