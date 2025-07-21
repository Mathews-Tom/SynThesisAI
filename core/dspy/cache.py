"""
DSPy Optimization Cache

This module provides caching functionality for DSPy optimization results,
enabling efficient reuse of optimized modules and reducing redundant computations.
"""

import hashlib
import json
import logging
import pickle
import time
from pathlib import Path
from typing import Any, Dict, Optional

from .config import get_dspy_config
from .exceptions import CacheCorruptionError

logger = logging.getLogger(__name__)


class OptimizationCache:
    """
    Cache for DSPy optimization results.

    Provides both memory and persistent caching of optimized DSPy modules
    to avoid redundant optimization computations.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize optimization cache.

        Args:
            cache_dir: Optional custom cache directory
        """
        self.config = get_dspy_config()
        cache_config = self.config.get_cache_config()

        self.cache_dir = Path(cache_dir or cache_config["cache_dir"])
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.cache_ttl = cache_config["cache_ttl"]
        self.enabled = cache_config["enabled"]

        # In-memory cache for fast access
        self.memory_cache: Dict[str, Dict[str, Any]] = {}

        # Cache statistics
        self.stats = {"hits": 0, "misses": 0, "stores": 0, "evictions": 0, "errors": 0}

        self.logger = logging.getLogger(f"{__name__}.OptimizationCache")
        self.logger.info(
            "Initialized optimization cache: %s (TTL: %ds)",
            self.cache_dir,
            self.cache_ttl,
        )

    def generate_cache_key(
        self,
        domain: str,
        signature: str,
        quality_requirements: Dict[str, Any],
        optimization_params: Dict[str, Any] = None,
    ) -> str:
        """
        Generate unique cache key for optimization.

        Args:
            domain: STREAM domain
            signature: DSPy signature
            quality_requirements: Quality requirements dictionary
            optimization_params: Optional optimization parameters

        Returns:
            Unique cache key string
        """
        try:
            key_components = [
                domain,
                signature,
                json.dumps(quality_requirements, sort_keys=True),
                json.dumps(optimization_params or {}, sort_keys=True),
                self.config.get_dspy_version(),
            ]

            key_string = "|".join(key_components)
            cache_key = hashlib.md5(key_string.encode()).hexdigest()

            self.logger.debug("Generated cache key for %s: %s", domain, cache_key)
            return cache_key

        except Exception as e:
            self.logger.error("Error generating cache key: %s", str(e))
            # Fallback to simple key
            return hashlib.md5(
                f"{domain}_{signature}_{time.time()}".encode()
            ).hexdigest()

    def store(
        self,
        cache_key: str,
        optimized_module: Any,
        optimization_metrics: Dict[str, Any] = None,
    ) -> bool:
        """
        Store optimized module in cache.

        Args:
            cache_key: Unique cache key
            optimized_module: Optimized DSPy module
            optimization_metrics: Optional optimization metrics

        Returns:
            True if stored successfully
        """
        if not self.enabled:
            return False

        try:
            timestamp = time.time()

            cache_data = {
                "module": optimized_module,
                "timestamp": timestamp,
                "metrics": optimization_metrics or {},
                "metadata": {
                    "domain": getattr(optimized_module, "domain", "unknown"),
                    "signature": getattr(optimized_module, "signature", "unknown"),
                    "cache_version": "1.0",
                },
            }

            # Store in memory cache
            self.memory_cache[cache_key] = cache_data

            # Store in persistent cache
            cache_path = self.cache_dir / f"{cache_key}.pkl"
            with open(cache_path, "wb") as f:
                pickle.dump(cache_data, f)

            self.stats["stores"] += 1
            self.logger.debug("Stored optimization in cache: %s", cache_key)

            return True

        except Exception as e:
            self.stats["errors"] += 1
            self.logger.error("Error storing cache entry %s: %s", cache_key, str(e))
            return False

    def get(self, cache_key: str) -> Optional[Any]:
        """
        Retrieve optimized module from cache.

        Args:
            cache_key: Unique cache key

        Returns:
            Optimized module if found and valid, None otherwise
        """
        if not self.enabled:
            return None

        try:
            # Check memory cache first
            if cache_key in self.memory_cache:
                cached_item = self.memory_cache[cache_key]
                if self._is_cache_valid(cached_item):
                    self.stats["hits"] += 1
                    self.logger.debug("Cache hit (memory): %s", cache_key)
                    return cached_item["module"]
                else:
                    # Remove expired entry
                    del self.memory_cache[cache_key]
                    self.stats["evictions"] += 1

            # Check persistent cache
            cache_path = self.cache_dir / f"{cache_key}.pkl"
            if cache_path.exists():
                with open(cache_path, "rb") as f:
                    cached_data = pickle.load(f)

                if self._is_cache_valid(cached_data):
                    # Update memory cache
                    self.memory_cache[cache_key] = cached_data
                    self.stats["hits"] += 1
                    self.logger.debug("Cache hit (persistent): %s", cache_key)
                    return cached_data["module"]
                else:
                    # Remove expired file
                    cache_path.unlink()
                    self.stats["evictions"] += 1

            # Cache miss
            self.stats["misses"] += 1
            self.logger.debug("Cache miss: %s", cache_key)
            return None

        except Exception as e:
            self.stats["errors"] += 1
            self.logger.error("Error retrieving cache entry %s: %s", cache_key, str(e))
            return None

    def _is_cache_valid(self, cached_data: Dict[str, Any]) -> bool:
        """
        Check if cached data is still valid.

        Args:
            cached_data: Cached data dictionary

        Returns:
            True if cache entry is valid
        """
        try:
            timestamp = cached_data.get("timestamp", 0)
            age = time.time() - timestamp

            if age > self.cache_ttl:
                self.logger.debug("Cache entry expired (age: %.1fs)", age)
                return False

            # Check for required fields
            if "module" not in cached_data:
                self.logger.warning("Cache entry missing module")
                return False

            return True

        except Exception as e:
            self.logger.warning("Error validating cache entry: %s", str(e))
            return False

    def invalidate(self, cache_key: str) -> bool:
        """
        Invalidate a specific cache entry.

        Args:
            cache_key: Cache key to invalidate

        Returns:
            True if invalidated successfully
        """
        try:
            # Remove from memory cache
            if cache_key in self.memory_cache:
                del self.memory_cache[cache_key]

            # Remove from persistent cache
            cache_path = self.cache_dir / f"{cache_key}.pkl"
            if cache_path.exists():
                cache_path.unlink()

            self.stats["evictions"] += 1
            self.logger.debug("Invalidated cache entry: %s", cache_key)
            return True

        except Exception as e:
            self.stats["errors"] += 1
            self.logger.error(
                "Error invalidating cache entry %s: %s", cache_key, str(e)
            )
            return False

    def clear(self) -> bool:
        """
        Clear all cache entries.

        Returns:
            True if cleared successfully
        """
        try:
            # Clear memory cache
            cleared_memory = len(self.memory_cache)
            self.memory_cache.clear()

            # Clear persistent cache
            cleared_persistent = 0
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
                cleared_persistent += 1

            self.stats["evictions"] += cleared_memory + cleared_persistent
            self.logger.info(
                "Cleared cache: %d memory + %d persistent entries",
                cleared_memory,
                cleared_persistent,
            )
            return True

        except Exception as e:
            self.stats["errors"] += 1
            self.logger.error("Error clearing cache: %s", str(e))
            return False

    def cleanup_expired(self) -> int:
        """
        Clean up expired cache entries.

        Returns:
            Number of entries cleaned up
        """
        cleaned_count = 0

        try:
            # Clean memory cache
            expired_keys = []
            for key, data in self.memory_cache.items():
                if not self._is_cache_valid(data):
                    expired_keys.append(key)

            for key in expired_keys:
                del self.memory_cache[key]
                cleaned_count += 1

            # Clean persistent cache
            for cache_file in self.cache_dir.glob("*.pkl"):
                try:
                    with open(cache_file, "rb") as f:
                        cached_data = pickle.load(f)

                    if not self._is_cache_valid(cached_data):
                        cache_file.unlink()
                        cleaned_count += 1

                except Exception as e:
                    self.logger.warning(
                        "Error checking cache file %s: %s", cache_file, str(e)
                    )
                    # Remove corrupted files
                    cache_file.unlink()
                    cleaned_count += 1

            if cleaned_count > 0:
                self.stats["evictions"] += cleaned_count
                self.logger.info("Cleaned up %d expired cache entries", cleaned_count)

            return cleaned_count

        except Exception as e:
            self.stats["errors"] += 1
            self.logger.error("Error during cache cleanup: %s", str(e))
            return cleaned_count

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary containing cache statistics
        """
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total_requests if total_requests > 0 else 0.0

        return {
            **self.stats,
            "hit_rate": hit_rate,
            "memory_entries": len(self.memory_cache),
            "persistent_entries": len(list(self.cache_dir.glob("*.pkl"))),
            "cache_dir": str(self.cache_dir),
            "cache_ttl": self.cache_ttl,
            "enabled": self.enabled,
        }

    def get_cache_info(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific cache entry.

        Args:
            cache_key: Cache key to inspect

        Returns:
            Cache entry information or None if not found
        """
        try:
            # Check memory cache
            if cache_key in self.memory_cache:
                cached_data = self.memory_cache[cache_key]
                return {
                    "location": "memory",
                    "timestamp": cached_data.get("timestamp"),
                    "age": time.time() - cached_data.get("timestamp", 0),
                    "valid": self._is_cache_valid(cached_data),
                    "metadata": cached_data.get("metadata", {}),
                    "metrics": cached_data.get("metrics", {}),
                }

            # Check persistent cache
            cache_path = self.cache_dir / f"{cache_key}.pkl"
            if cache_path.exists():
                with open(cache_path, "rb") as f:
                    cached_data = pickle.load(f)

                return {
                    "location": "persistent",
                    "timestamp": cached_data.get("timestamp"),
                    "age": time.time() - cached_data.get("timestamp", 0),
                    "valid": self._is_cache_valid(cached_data),
                    "metadata": cached_data.get("metadata", {}),
                    "metrics": cached_data.get("metrics", {}),
                    "file_size": cache_path.stat().st_size,
                }

            return None

        except Exception as e:
            self.logger.error("Error getting cache info for %s: %s", cache_key, str(e))
            return None


# Global cache instance
_optimization_cache = None


def get_optimization_cache() -> OptimizationCache:
    """Get the global optimization cache instance."""
    global _optimization_cache
    if _optimization_cache is None:
        _optimization_cache = OptimizationCache()
    return _optimization_cache
