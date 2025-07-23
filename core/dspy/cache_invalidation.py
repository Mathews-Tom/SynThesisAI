"""
Cache invalidation triggers for DSPy optimization.

This module provides functionality for invalidating cache entries
based on configuration changes and other triggers.
"""

import hashlib
import json
import logging
from typing import Any, Dict, List, Optional, Set

from .cache import get_optimization_cache
from .config import get_dspy_config

logger = logging.getLogger(__name__)


class CacheInvalidationManager:
    """
    Manages cache invalidation triggers for DSPy optimization.

    Provides functionality for invalidating cache entries based on
    configuration changes and other triggers.
    """

    def __init__(self):
        """Initialize cache invalidation manager."""
        self.cache = get_optimization_cache()
        self.config = get_dspy_config()
        self.logger = logging.getLogger(__name__ + ".CacheInvalidationManager")

        # Store configuration hash for change detection
        self.config_hash = self._compute_config_hash()

        self.logger.info("Initialized cache invalidation manager")

    def _compute_config_hash(self) -> str:
        """
        Compute hash of current configuration.

        Returns:
            Hash string of configuration
        """
        # Get relevant configuration sections
        optimization_config = self.config.get_optimization_config()

        # Create serializable configuration
        config_dict = {
            "optimization": optimization_config,
            "cache_ttl": self.config.cache_ttl,
            "version": getattr(self.config, "version", "1.0"),
        }

        # Compute hash
        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()

    def check_config_changes(self) -> bool:
        """
        Check if configuration has changed since last check.

        Returns:
            True if configuration has changed
        """
        current_hash = self._compute_config_hash()
        has_changed = current_hash != self.config_hash

        if has_changed:
            self.logger.info("Configuration has changed, updating hash")
            self.config_hash = current_hash

        return has_changed

    def invalidate_by_config_change(self) -> int:
        """
        Invalidate cache entries if configuration has changed.

        Returns:
            Number of entries invalidated
        """
        if not self.check_config_changes():
            return 0

        self.logger.info("Invalidating cache entries due to configuration changes")

        # Get all cache entries
        entries = self.cache.list_cache_entries()

        # Invalidate all entries
        invalidated_count = 0
        for entry in entries:
            cache_key = entry["cache_key"]
            if self.cache.invalidate(cache_key):
                invalidated_count += 1

        self.logger.info(
            "Invalidated %d cache entries due to configuration changes",
            invalidated_count,
        )
        return invalidated_count

    def invalidate_by_domain(self, domain: str) -> int:
        """
        Invalidate cache entries for a specific domain.

        Args:
            domain: Domain to invalidate cache for

        Returns:
            Number of entries invalidated
        """
        self.logger.info("Invalidating cache entries for domain: %s", domain)

        # Get cache entries for domain
        entries = self.cache.list_cache_entries(domain_filter=domain)

        # Invalidate entries
        invalidated_count = 0
        for entry in entries:
            cache_key = entry["cache_key"]
            if self.cache.invalidate(cache_key):
                invalidated_count += 1

        self.logger.info(
            "Invalidated %d cache entries for domain %s", invalidated_count, domain
        )
        return invalidated_count

    def invalidate_by_performance(self, min_validation_score: float = 0.7) -> int:
        """
        Invalidate cache entries with poor performance.

        Args:
            min_validation_score: Minimum validation score to keep

        Returns:
            Number of entries invalidated
        """
        self.logger.info(
            "Invalidating cache entries with validation score < %.2f",
            min_validation_score,
        )

        # Get all cache entries
        entries = self.cache.list_cache_entries()

        # Invalidate entries with poor performance
        invalidated_count = 0
        for entry in entries:
            if entry.get("validation_score", 1.0) < min_validation_score:
                cache_key = entry["cache_key"]
                if self.cache.invalidate(cache_key):
                    invalidated_count += 1

        self.logger.info(
            "Invalidated %d cache entries with poor performance", invalidated_count
        )
        return invalidated_count

    def run_maintenance(self) -> Dict[str, int]:
        """
        Run comprehensive cache maintenance.

        Returns:
            Dictionary with maintenance results
        """
        results = {
            "config_changes": 0,
            "performance": 0,
            "maintenance": 0,
        }

        # Check for configuration changes
        results["config_changes"] = self.invalidate_by_config_change()

        # Invalidate entries with poor performance
        results["performance"] = self.invalidate_by_performance()

        # Run general cache maintenance
        maintenance_results = self.cache.maintenance()
        results["maintenance"] = sum(maintenance_results.values())

        self.logger.info("Cache maintenance completed: %s", results)
        return results


# Global invalidation manager instance
_invalidation_manager = None


def get_invalidation_manager() -> CacheInvalidationManager:
    """Get the global cache invalidation manager instance."""
    global _invalidation_manager
    if _invalidation_manager is None:
        _invalidation_manager = CacheInvalidationManager()
    return _invalidation_manager


def invalidate_by_config_change() -> int:
    """
    Invalidate cache entries if configuration has changed.

    Returns:
        Number of entries invalidated
    """
    manager = get_invalidation_manager()
    return manager.invalidate_by_config_change()


def invalidate_by_domain(domain: str) -> int:
    """
    Invalidate cache entries for a specific domain.

    Args:
        domain: Domain to invalidate cache for

    Returns:
        Number of entries invalidated
    """
    manager = get_invalidation_manager()
    return manager.invalidate_by_domain(domain)


def run_cache_maintenance() -> Dict[str, int]:
    """
    Run comprehensive cache maintenance.

    Returns:
        Dictionary with maintenance results
    """
    manager = get_invalidation_manager()
    return manager.run_maintenance()
