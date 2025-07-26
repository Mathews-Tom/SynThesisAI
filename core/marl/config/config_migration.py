"""
MARL Configuration Migration.

This module provides configuration migration and versioning capabilities
to handle configuration schema changes across different versions.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from packaging import version

from utils.logging_config import get_logger

from .config_schema import MARLConfig


class MigrationError(Exception):
    """Configuration migration error."""

    pass


class ConfigMigration:
    """Base class for configuration migrations."""

    def __init__(self, from_version: str, to_version: str):
        """
        Initialize migration.

        Args:
            from_version: Source version
            to_version: Target version
        """
        self.from_version = from_version
        self.to_version = to_version

    def can_migrate(self, config_version: str) -> bool:
        """Check if this migration can handle the given version."""
        return config_version == self.from_version

    def migrate(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Migrate configuration data.

        Args:
            config_data: Configuration data to migrate

        Returns:
            Migrated configuration data
        """
        raise NotImplementedError("Subclasses must implement migrate method")

    def validate_migration(
        self, original: Dict[str, Any], migrated: Dict[str, Any]
    ) -> bool:
        """
        Validate that migration was successful.

        Args:
            original: Original configuration data
            migrated: Migrated configuration data

        Returns:
            True if migration is valid
        """
        # Basic validation - ensure essential fields are preserved
        return (
            migrated.get("name") == original.get("name")
            and migrated.get("version") == self.to_version
            and "agents" in migrated
        )


class Migration_1_0_0_to_1_1_0(ConfigMigration):
    """Migration from version 1.0.0 to 1.1.0."""

    def __init__(self):
        super().__init__("1.0.0", "1.1.0")

    def migrate(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate from 1.0.0 to 1.1.0."""
        migrated = config_data.copy()

        # Update version
        migrated["version"] = self.to_version

        # Add new fields introduced in 1.1.0
        if "system" in migrated:
            system = migrated["system"]

            # Add new system fields with defaults
            if "deterministic" not in system:
                system["deterministic"] = False

            if "seed" not in system:
                system["seed"] = None

        # Update agent configurations
        if "agents" in migrated:
            for agent_id, agent_config in migrated["agents"].items():
                # Add new agent fields
                if "reward_scaling" not in agent_config:
                    agent_config["reward_scaling"] = 1.0

                # Update network configuration
                if "network" in agent_config:
                    network = agent_config["network"]
                    if "weight_initialization" not in network:
                        network["weight_initialization"] = "xavier_uniform"

        # Add new learning configuration fields
        if "learning" in migrated:
            learning = migrated["learning"]
            if "trend_analysis_enabled" not in learning:
                learning["trend_analysis_enabled"] = True

        return migrated


class Migration_1_1_0_to_1_2_0(ConfigMigration):
    """Migration from version 1.1.0 to 1.2.0."""

    def __init__(self):
        super().__init__("1.1.0", "1.2.0")

    def migrate(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate from 1.1.0 to 1.2.0."""
        migrated = config_data.copy()

        # Update version
        migrated["version"] = self.to_version

        # Add performance monitoring configuration
        if "learning" in migrated:
            learning = migrated["learning"]
            if "performance_tracking_enabled" not in learning:
                learning["performance_tracking_enabled"] = True

            if "metrics_collection_interval" not in learning:
                learning["metrics_collection_interval"] = 1.0

        # Update coordination configuration
        if "coordination" in migrated:
            coordination = migrated["coordination"]

            # Add deadlock detection
            if "deadlock_detection_enabled" not in coordination:
                coordination["deadlock_detection_enabled"] = True

            if "deadlock_timeout" not in coordination:
                coordination["deadlock_timeout"] = 120.0

        # Update shared learning configuration
        if "learning" in migrated and "shared_learning" in migrated["learning"]:
            shared_learning = migrated["learning"]["shared_learning"]

            # Add continuous learning parameters
            if "continuous_learning_enabled" not in shared_learning:
                shared_learning["continuous_learning_enabled"] = True

            if "learning_update_interval" not in shared_learning:
                shared_learning["learning_update_interval"] = 10.0

            if "performance_window_size" not in shared_learning:
                shared_learning["performance_window_size"] = 100

            if "adaptation_threshold" not in shared_learning:
                shared_learning["adaptation_threshold"] = 0.05

        return migrated


class Migration_1_2_0_to_2_0_0(ConfigMigration):
    """Migration from version 1.2.0 to 2.0.0 (breaking changes)."""

    def __init__(self):
        super().__init__("1.2.0", "2.0.0")

    def migrate(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate from 1.2.0 to 2.0.0."""
        migrated = config_data.copy()

        # Update version
        migrated["version"] = self.to_version

        # Breaking change: Restructure consensus configuration
        if "coordination" in migrated and "consensus" in migrated["coordination"]:
            consensus = migrated["coordination"]["consensus"]

            # Rename strategy values (breaking change)
            strategy_mapping = {
                "majority": "majority_vote",
                "weighted": "weighted_average",
                "expert": "expert_priority",
            }

            if "strategy" in consensus:
                old_strategy = consensus["strategy"]
                if old_strategy in strategy_mapping:
                    consensus["strategy"] = strategy_mapping[old_strategy]

        # Breaking change: Update exploration strategy names
        if "agents" in migrated:
            for agent_config in migrated["agents"].values():
                if "exploration" in agent_config:
                    exploration = agent_config["exploration"]

                    # Rename exploration strategies
                    exploration_mapping = {
                        "epsilon": "epsilon_greedy",
                        "boltzmann": "boltzmann",
                        "ucb1": "ucb",
                    }

                    if "strategy" in exploration:
                        old_strategy = exploration["strategy"]
                        if old_strategy in exploration_mapping:
                            exploration["strategy"] = exploration_mapping[old_strategy]

        # Add new required fields for 2.0.0
        if "environment" not in migrated:
            migrated["environment"] = {}

        return migrated


class ConfigMigrationManager:
    """
    Manager for configuration migrations and versioning.

    Handles automatic migration of configurations between versions
    and provides version compatibility checking.
    """

    def __init__(self):
        """Initialize the migration manager."""
        self.logger = get_logger(__name__)

        # Register available migrations
        self._migrations: List[ConfigMigration] = [
            Migration_1_0_0_to_1_1_0(),
            Migration_1_1_0_to_1_2_0(),
            Migration_1_2_0_to_2_0_0(),
        ]

        # Current supported version
        self.current_version = "2.0.0"

        # Version compatibility matrix
        self._compatibility_matrix = {
            "1.0.0": ["1.1.0", "1.2.0", "2.0.0"],
            "1.1.0": ["1.2.0", "2.0.0"],
            "1.2.0": ["2.0.0"],
            "2.0.0": [],
        }

        self.logger.info(
            "Configuration migration manager initialized with %d migrations",
            len(self._migrations),
        )

    def get_supported_versions(self) -> List[str]:
        """Get list of supported configuration versions."""
        versions = set()
        for migration in self._migrations:
            versions.add(migration.from_version)
            versions.add(migration.to_version)
        return sorted(list(versions), key=version.parse)

    def is_version_supported(self, config_version: str) -> bool:
        """Check if a configuration version is supported."""
        return config_version in self.get_supported_versions()

    def needs_migration(
        self, config_version: str, target_version: Optional[str] = None
    ) -> bool:
        """
        Check if configuration needs migration.

        Args:
            config_version: Current configuration version
            target_version: Target version (defaults to current version)

        Returns:
            True if migration is needed
        """
        target = target_version or self.current_version
        return config_version != target and self.can_migrate(config_version, target)

    def can_migrate(self, from_version: str, to_version: Optional[str] = None) -> bool:
        """
        Check if migration is possible between versions.

        Args:
            from_version: Source version
            to_version: Target version (defaults to current version)

        Returns:
            True if migration is possible
        """
        target = to_version or self.current_version

        if from_version == target:
            return True

        # Check if direct migration path exists
        migration_path = self._find_migration_path(from_version, target)
        return migration_path is not None

    def migrate_config(
        self,
        config_data: Dict[str, Any],
        target_version: Optional[str] = None,
        validate: bool = True,
    ) -> Tuple[Dict[str, Any], List[str]]:
        """
        Migrate configuration to target version.

        Args:
            config_data: Configuration data to migrate
            target_version: Target version (defaults to current version)
            validate: Whether to validate migration steps

        Returns:
            Tuple of (migrated_config, migration_log)

        Raises:
            MigrationError: If migration fails
        """
        target = target_version or self.current_version
        current_version = config_data.get("version", "1.0.0")

        if current_version == target:
            return config_data, ["No migration needed"]

        # Find migration path
        migration_path = self._find_migration_path(current_version, target)
        if not migration_path:
            raise MigrationError(
                f"No migration path from {current_version} to {target}"
            )

        # Apply migrations in sequence
        migrated_data = config_data.copy()
        migration_log = []

        for migration in migration_path:
            self.logger.info(
                "Applying migration: %s -> %s",
                migration.from_version,
                migration.to_version,
            )

            original_data = migrated_data.copy()

            try:
                migrated_data = migration.migrate(migrated_data)

                # Validate migration if requested
                if validate and not migration.validate_migration(
                    original_data, migrated_data
                ):
                    raise MigrationError(
                        f"Migration validation failed: {migration.from_version} -> {migration.to_version}"
                    )

                migration_log.append(
                    f"Migrated from {migration.from_version} to {migration.to_version}"
                )

            except Exception as e:
                raise MigrationError(
                    f"Migration failed ({migration.from_version} -> {migration.to_version}): {str(e)}"
                )

        self.logger.info(
            "Successfully migrated configuration from %s to %s", current_version, target
        )

        return migrated_data, migration_log

    def _find_migration_path(
        self, from_version: str, to_version: str
    ) -> Optional[List[ConfigMigration]]:
        """Find migration path between versions."""
        if from_version == to_version:
            return []

        # Use BFS to find shortest migration path
        from collections import deque

        queue = deque([(from_version, [])])
        visited = {from_version}

        while queue:
            current_version, path = queue.popleft()

            # Find migrations from current version
            for migration in self._migrations:
                if migration.can_migrate(current_version):
                    next_version = migration.to_version
                    new_path = path + [migration]

                    if next_version == to_version:
                        return new_path

                    if next_version not in visited:
                        visited.add(next_version)
                        queue.append((next_version, new_path))

        return None

    def get_migration_info(
        self, from_version: str, to_version: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get information about migration between versions.

        Args:
            from_version: Source version
            to_version: Target version (defaults to current version)

        Returns:
            Migration information dictionary
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

        # Check for breaking changes
        breaking_changes = any(
            self._is_breaking_change(migration.from_version, migration.to_version)
            for migration in migration_path
        )

        steps = [
            {
                "from": migration.from_version,
                "to": migration.to_version,
                "breaking": self._is_breaking_change(
                    migration.from_version, migration.to_version
                ),
            }
            for migration in migration_path
        ]

        return {
            "possible": True,
            "steps": steps,
            "breaking_changes": breaking_changes,
            "total_steps": len(migration_path),
        }

    def _is_breaking_change(self, from_version: str, to_version: str) -> bool:
        """Check if migration involves breaking changes."""
        from_major = int(from_version.split(".")[0])
        to_major = int(to_version.split(".")[0])
        return to_major > from_major

    def validate_version_format(self, version_string: str) -> bool:
        """
        Validate version string format.

        Args:
            version_string: Version string to validate

        Returns:
            True if format is valid
        """
        pattern = r"^\d+\.\d+\.\d+$"
        return bool(re.match(pattern, version_string))

    def compare_versions(self, version1: str, version2: str) -> int:
        """
        Compare two version strings.

        Args:
            version1: First version
            version2: Second version

        Returns:
            -1 if version1 < version2, 0 if equal, 1 if version1 > version2
        """
        try:
            v1 = version.parse(version1)
            v2 = version.parse(version2)

            if v1 < v2:
                return -1
            elif v1 > v2:
                return 1
            else:
                return 0
        except Exception:
            # Fallback to string comparison
            if version1 < version2:
                return -1
            elif version1 > version2:
                return 1
            else:
                return 0

    def get_latest_version(self) -> str:
        """Get the latest supported version."""
        return self.current_version

    def create_migration_report(
        self,
        original_config: Dict[str, Any],
        migrated_config: Dict[str, Any],
        migration_log: List[str],
    ) -> Dict[str, Any]:
        """
        Create a detailed migration report.

        Args:
            original_config: Original configuration
            migrated_config: Migrated configuration
            migration_log: Migration log messages

        Returns:
            Migration report dictionary
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
                "new_fields_added": self._find_new_fields(
                    original_config, migrated_config
                ),
                "fields_removed": self._find_removed_fields(
                    original_config, migrated_config
                ),
                "fields_renamed": self._find_renamed_fields(
                    original_config, migrated_config
                ),
            },
            "compatibility_notes": self._generate_compatibility_notes(
                original_config.get("version", "unknown"),
                migrated_config.get("version", "unknown"),
            ),
        }

    def _find_new_fields(
        self, original: Dict[str, Any], migrated: Dict[str, Any]
    ) -> List[str]:
        """Find fields that were added during migration."""
        # Simplified implementation - would need recursive comparison for full analysis
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

    def _find_removed_fields(
        self, original: Dict[str, Any], migrated: Dict[str, Any]
    ) -> List[str]:
        """Find fields that were removed during migration."""
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
        self, original: Dict[str, Any], migrated: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Find fields that were renamed during migration."""
        # Simplified implementation - would need more sophisticated analysis
        return []

    def _generate_compatibility_notes(
        self, from_version: str, to_version: str
    ) -> List[str]:
        """Generate compatibility notes for migration."""
        notes = []

        if self._is_breaking_change(from_version, to_version):
            notes.append(
                "This migration includes breaking changes that may require code updates"
            )

        # Version-specific notes
        if from_version.startswith("1.") and to_version.startswith("2."):
            notes.append(
                "Migration to version 2.x includes restructured consensus and exploration configurations"
            )

        return notes


class ConfigMigrationManagerFactory:
    """Factory for creating configuration migration managers."""

    @staticmethod
    def create() -> ConfigMigrationManager:
        """Create a configuration migration manager."""
        return ConfigMigrationManager()

    @staticmethod
    def create_with_custom_migrations(
        custom_migrations: List[ConfigMigration],
    ) -> ConfigMigrationManager:
        """
        Create a migration manager with custom migrations.

        Args:
            custom_migrations: List of custom migration objects

        Returns:
            Migration manager with custom migrations
        """
        manager = ConfigMigrationManager()
        manager._migrations.extend(custom_migrations)
        return manager
