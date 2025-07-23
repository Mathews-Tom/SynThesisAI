"""
Caching system for DSPy optimization results.

This module provides caching functionality for DSPy optimization results
to avoid redundant computations and improve performance.
"""

import hashlib
import json
import logging
import os
import pickle
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import OptimizationResult, get_dspy_config
from .exceptions import CacheCorruptionError

logger = logging.getLogger(__name__)


class OptimizationCache:
    """
    Cache for DSPy optimization results.

    Provides both memory and persistent caching for optimization results
    to avoid redundant computations.
    """

    def __init__(self, cache_dir: str = ".cache/dspy", cache_ttl: int = None):
        """
        Initialize optimization cache.

        Args:
            cache_dir: Directory for persistent cache storage
            cache_ttl: Cache time-to-live in seconds (defaults to config value)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.memory_cache: Dict[str, Dict[str, Any]] = {}

        # Get cache TTL from config or use provided value
        config = get_dspy_config()
        self.cache_ttl = cache_ttl or config.cache_ttl

        self.logger = logging.getLogger(__name__ + ".OptimizationCache")
        self._lock = threading.RLock()  # Thread-safe operations

        # Cache statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "stores": 0,
            "invalidations": 0,
            "errors": 0,
        }

        # Initialize cache validation
        self._validate_cache_directory()

        self.logger.info(
            "Initialized optimization cache with TTL %d seconds", self.cache_ttl
        )

    def _validate_cache_directory(self) -> None:
        """Validate cache directory and perform cleanup if needed."""
        try:
            # Ensure cache directory exists and is writable
            if not self.cache_dir.exists():
                self.cache_dir.mkdir(parents=True, exist_ok=True)

            # Test write permissions
            test_file = self.cache_dir / ".cache_test"
            test_file.write_text("test")
            test_file.unlink()

            # Clean up expired entries on startup
            self._cleanup_expired_entries()

        except Exception as e:
            self.logger.error("Cache directory validation failed: %s", str(e))
            raise CacheCorruptionError(
                f"Cache directory validation failed: {str(e)}",
                cache_key="directory_validation",
            ) from e

    def _cleanup_expired_entries(self) -> int:
        """
        Clean up expired cache entries.

        Returns:
            Number of entries cleaned up
        """
        cleaned_count = 0
        current_time = time.time()

        try:
            for cache_file in self.cache_dir.glob("*.pkl"):
                try:
                    # Check if file is expired
                    if current_time - cache_file.stat().st_mtime > self.cache_ttl:
                        cache_file.unlink()
                        cleaned_count += 1
                except OSError:
                    # File might have been deleted by another process
                    continue

            if cleaned_count > 0:
                self.logger.info("Cleaned up %d expired cache entries", cleaned_count)

        except Exception as e:
            self.logger.error("Error during cache cleanup: %s", str(e))

        return cleaned_count

    def generate_cache_key(
        self, domain: str, signature: str, quality_requirements: Dict[str, Any]
    ) -> str:
        """
        Generate unique cache key for optimization.

        Args:
            domain: Domain name
            signature: DSPy signature
            quality_requirements: Quality requirements

        Returns:
            Unique cache key
        """
        try:
            key_components = [
                domain.lower(),  # Normalize domain case
                signature,
                json.dumps(quality_requirements, sort_keys=True),
                "v1.1",  # Cache version
            ]
            key_string = "|".join(key_components)
            return hashlib.md5(key_string.encode()).hexdigest()
        except Exception as e:
            self.logger.error("Failed to generate cache key: %s", str(e))
            raise CacheCorruptionError(
                f"Cache key generation failed: {str(e)}", cache_key="key_generation"
            ) from e

    def store(self, cache_key: str, result: OptimizationResult) -> bool:
        """
        Store optimization result in cache.

        Args:
            cache_key: Cache key
            result: Optimization result to store

        Returns:
            True if stored successfully in both memory and persistent cache
        """
        with self._lock:
            memory_success = False
            persistent_success = False

            try:
                current_time = time.time()

                # Store in memory cache
                self.memory_cache[cache_key] = {
                    "result": result,
                    "timestamp": current_time,
                }
                memory_success = True

                # Store in persistent cache
                cache_path = self.cache_dir / f"{cache_key}.pkl"
                cache_data = {
                    "result": result,
                    "timestamp": current_time,
                    "metadata": {
                        "domain": getattr(result.optimized_module, "domain", "unknown"),
                        "cache_key": cache_key,
                        "validation_score": result.validation_score,
                        "training_time": result.training_time,
                    },
                }

                # Write to temporary file first, then rename for atomic operation
                temp_path = cache_path.with_suffix(".tmp")
                temp_path.write_bytes(pickle.dumps(cache_data))
                temp_path.rename(cache_path)
                persistent_success = True

                self.stats["stores"] += 1
                self.logger.info("Stored optimization result in cache: %s", cache_key)
                return True

            except Exception as e:
                self.stats["errors"] += 1
                self.logger.error(
                    "Failed to store cache entry %s: %s", cache_key, str(e)
                )

                # Clean up temporary file if it exists
                temp_path = self.cache_dir / f"{cache_key}.tmp"
                if temp_path.exists():
                    try:
                        temp_path.unlink()
                    except OSError:
                        pass

                # If persistent storage failed, remove from memory cache too
                if memory_success and not persistent_success:
                    if cache_key in self.memory_cache:
                        del self.memory_cache[cache_key]

                return False

    def get(self, cache_key: str) -> Optional[OptimizationResult]:
        """
        Retrieve optimization result from cache.

        Args:
            cache_key: Cache key

        Returns:
            Cached optimization result or None if not found/expired
        """
        with self._lock:
            current_time = time.time()

            # Check memory cache first
            if cache_key in self.memory_cache:
                cached_item = self.memory_cache[cache_key]
                if current_time - cached_item["timestamp"] < self.cache_ttl:
                    self.stats["hits"] += 1
                    self.logger.debug("Cache hit (memory): %s", cache_key)
                    return cached_item["result"]

                # Remove expired entry
                del self.memory_cache[cache_key]

            # Check persistent cache
            cache_path = self.cache_dir / f"{cache_key}.pkl"
            if cache_path.exists():
                try:
                    cached_data = pickle.loads(cache_path.read_bytes())

                    if current_time - cached_data["timestamp"] < self.cache_ttl:
                        # Update memory cache
                        self.memory_cache[cache_key] = {
                            "result": cached_data["result"],
                            "timestamp": cached_data["timestamp"],
                        }
                        self.stats["hits"] += 1
                        self.logger.debug("Cache hit (persistent): %s", cache_key)
                        return cached_data["result"]

                    # Remove expired file
                    cache_path.unlink()

                except Exception as e:
                    self.stats["errors"] += 1
                    self.logger.error(
                        "Failed to load cache entry %s: %s", cache_key, str(e)
                    )

                    # Remove corrupted cache file
                    if cache_path.exists():
                        try:
                            cache_path.unlink()
                        except OSError:
                            pass

            self.stats["misses"] += 1
            self.logger.debug("Cache miss: %s", cache_key)
            return None

    def invalidate(self, cache_key: str) -> bool:
        """
        Invalidate a cache entry.

        Args:
            cache_key: Cache key to invalidate

        Returns:
            True if invalidated successfully
        """
        with self._lock:
            try:
                invalidated = False

                # Remove from memory cache
                if cache_key in self.memory_cache:
                    del self.memory_cache[cache_key]
                    invalidated = True

                # Remove from persistent cache
                cache_path = self.cache_dir / f"{cache_key}.pkl"
                if cache_path.exists():
                    cache_path.unlink()
                    invalidated = True

                if invalidated:
                    self.stats["invalidations"] += 1
                    self.logger.info("Invalidated cache entry: %s", cache_key)

                return invalidated

            except Exception as e:
                self.stats["errors"] += 1
                self.logger.error(
                    "Failed to invalidate cache entry %s: %s", cache_key, str(e)
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
            self.memory_cache.clear()

            # Clear persistent cache
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()

            self.logger.info("Cleared all cache entries")
            return True

        except Exception as e:
            self.logger.error("Failed to clear cache: %s", str(e))
            return False

    def is_fresh(self, cache_key: str) -> bool:
        """
        Check if a cache entry is fresh (not expired).

        Args:
            cache_key: Cache key to check

        Returns:
            True if entry exists and is fresh
        """
        with self._lock:
            current_time = time.time()

            # Check memory cache
            if cache_key in self.memory_cache:
                cached_item = self.memory_cache[cache_key]
                return current_time - cached_item["timestamp"] < self.cache_ttl

            # Check persistent cache
            cache_path = self.cache_dir / f"{cache_key}.pkl"
            if cache_path.exists():
                try:
                    file_mtime = cache_path.stat().st_mtime
                    return current_time - file_mtime < self.cache_ttl
                except OSError:
                    return False

            return False

    def get_cache_info(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a cache entry.

        Args:
            cache_key: Cache key

        Returns:
            Cache entry information or None if not found
        """
        with self._lock:
            # Check memory cache first
            if cache_key in self.memory_cache:
                cached_item = self.memory_cache[cache_key]
                return {
                    "cache_key": cache_key,
                    "location": "memory",
                    "timestamp": cached_item["timestamp"],
                    "age_seconds": time.time() - cached_item["timestamp"],
                    "is_fresh": time.time() - cached_item["timestamp"] < self.cache_ttl,
                }

            # Check persistent cache
            cache_path = self.cache_dir / f"{cache_key}.pkl"
            if cache_path.exists():
                try:
                    cached_data = pickle.loads(cache_path.read_bytes())

                    return {
                        "cache_key": cache_key,
                        "location": "persistent",
                        "timestamp": cached_data["timestamp"],
                        "age_seconds": time.time() - cached_data["timestamp"],
                        "is_fresh": time.time() - cached_data["timestamp"]
                        < self.cache_ttl,
                        "metadata": cached_data.get("metadata", {}),
                        "file_size": cache_path.stat().st_size,
                    }
                except Exception as e:
                    self.logger.error(
                        "Failed to get cache info for %s: %s", cache_key, str(e)
                    )

            return None

    def list_cache_entries(self, domain_filter: str = None) -> List[Dict[str, Any]]:
        """
        List all cache entries with optional domain filtering.

        Args:
            domain_filter: Optional domain to filter by

        Returns:
            List of cache entry information
        """
        entries = []

        with self._lock:
            # Get all cache files
            for cache_file in self.cache_dir.glob("*.pkl"):
                try:
                    cache_key = cache_file.stem

                    cached_data = pickle.loads(cache_file.read_bytes())

                    metadata = cached_data.get("metadata", {})
                    domain = metadata.get("domain", "unknown")

                    # Apply domain filter if specified
                    if domain_filter and domain != domain_filter:
                        continue

                    entry_info = {
                        "cache_key": cache_key,
                        "domain": domain,
                        "timestamp": cached_data["timestamp"],
                        "age_seconds": time.time() - cached_data["timestamp"],
                        "is_fresh": time.time() - cached_data["timestamp"]
                        < self.cache_ttl,
                        "file_size": cache_file.stat().st_size,
                        "validation_score": metadata.get("validation_score", 0.0),
                        "training_time": metadata.get("training_time", 0.0),
                    }
                    entries.append(entry_info)

                except Exception as e:
                    self.logger.error(
                        "Failed to read cache entry %s: %s", cache_file, str(e)
                    )
                    continue

        # Sort by timestamp (newest first)
        entries.sort(key=lambda x: x["timestamp"], reverse=True)
        return entries

    def get_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive cache statistics.

        Returns:
            Cache statistics
        """
        with self._lock:
            memory_entries = len(self.memory_cache)
            persistent_entries = len(list(self.cache_dir.glob("*.pkl")))

            # Calculate cache directory size
            total_size = 0
            for cache_file in self.cache_dir.glob("*.pkl"):
                try:
                    total_size += cache_file.stat().st_size
                except OSError:
                    continue

            # Calculate hit rate
            total_requests = self.stats["hits"] + self.stats["misses"]
            hit_rate = (
                self.stats["hits"] / total_requests if total_requests > 0 else 0.0
            )

            return {
                "memory_entries": memory_entries,
                "persistent_entries": persistent_entries,
                "total_entries": memory_entries + persistent_entries,
                "cache_dir": str(self.cache_dir),
                "cache_ttl": self.cache_ttl,
                "total_size_bytes": total_size,
                "total_size_mb": total_size / (1024 * 1024),
                "hit_rate": hit_rate,
                "stats": self.stats.copy(),
            }

    def maintenance(self) -> Dict[str, int]:
        """
        Perform cache maintenance operations.

        Returns:
            Dictionary with maintenance results
        """
        with self._lock:
            results = {
                "expired_cleaned": 0,
                "corrupted_removed": 0,
                "memory_cleared": 0,
            }

            # Clean up expired entries
            results["expired_cleaned"] = self._cleanup_expired_entries()

            # Remove corrupted cache files
            for cache_file in self.cache_dir.glob("*.pkl"):
                try:
                    pickle.loads(cache_file.read_bytes())
                except Exception:
                    try:
                        cache_file.unlink()
                        results["corrupted_removed"] += 1
                    except OSError:
                        pass

            # Clear expired memory cache entries
            current_time = time.time()
            expired_keys = [
                key
                for key, item in self.memory_cache.items()
                if current_time - item["timestamp"] >= self.cache_ttl
            ]

            for key in expired_keys:
                del self.memory_cache[key]
                results["memory_cleared"] += 1

            if any(results.values()):
                self.logger.info("Cache maintenance completed: %s", results)

            return results


# Global cache instance
_optimization_cache = None


def get_optimization_cache() -> OptimizationCache:
    """Get the global optimization cache instance."""
    global _optimization_cache
    if _optimization_cache is None:
        _optimization_cache = OptimizationCache()
    return _optimization_cache
