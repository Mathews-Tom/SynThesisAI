"""
Unit tests for DSPy optimization cache.

Tests the caching functionality including memory and persistent caching,
cache validation, freshness checking, and maintenance operations.
"""

import os
import tempfile
import time
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from core.dspy.base_module import STREAMContentGenerator
from core.dspy.cache import OptimizationCache, get_optimization_cache
from core.dspy.config import OptimizationResult
from core.dspy.exceptions import CacheCorruptionError


class TestOptimizationCache(unittest.TestCase):
    """Test optimization cache functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache = OptimizationCache(cache_dir=self.temp_dir, cache_ttl=60)

        # Create real optimization result (mocks don't pickle well)
        self.test_module = STREAMContentGenerator("mathematics")

        self.mock_result = OptimizationResult(
            optimized_module=self.test_module,
            optimization_metrics={"accuracy": 0.95},
            training_time=10.5,
            validation_score=0.92,
            cache_key="test_cache_key",
            timestamp=datetime.now(),
        )

    def tearDown(self):
        """Clean up test fixtures."""
        # Clean up temporary directory
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_cache_initialization(self):
        """Test cache initialization."""
        self.assertTrue(Path(self.temp_dir).exists())
        self.assertEqual(self.cache.cache_ttl, 60)
        self.assertEqual(len(self.cache.memory_cache), 0)
        self.assertIsNotNone(self.cache.stats)

    def test_generate_cache_key(self):
        """Test cache key generation."""
        domain = "mathematics"
        signature = "input -> output"
        quality_requirements = {"min_accuracy": 0.8}

        key1 = self.cache.generate_cache_key(domain, signature, quality_requirements)
        key2 = self.cache.generate_cache_key(domain, signature, quality_requirements)

        # Same inputs should generate same key
        self.assertEqual(key1, key2)

        # Different inputs should generate different keys
        key3 = self.cache.generate_cache_key("science", signature, quality_requirements)
        self.assertNotEqual(key1, key3)

        # Key should be a valid MD5 hash
        self.assertEqual(len(key1), 32)
        self.assertTrue(all(c in "0123456789abcdef" for c in key1))

    def test_store_and_retrieve(self):
        """Test storing and retrieving cache entries."""
        cache_key = "test_key_123"

        # Store result
        success = self.cache.store(cache_key, self.mock_result)
        self.assertTrue(success)

        # Retrieve result
        retrieved_result = self.cache.get(cache_key)
        self.assertIsNotNone(retrieved_result)
        self.assertEqual(retrieved_result.validation_score, 0.92)
        self.assertEqual(retrieved_result.training_time, 10.5)

        # Check statistics
        stats = self.cache.get_stats()
        self.assertEqual(stats["stats"]["stores"], 1)
        self.assertEqual(stats["stats"]["hits"], 1)

    def test_memory_cache_priority(self):
        """Test that memory cache is checked before persistent cache."""
        cache_key = "memory_test_key"

        # Store in cache
        self.cache.store(cache_key, self.mock_result)

        # First retrieval should hit memory cache
        result1 = self.cache.get(cache_key)
        self.assertIsNotNone(result1)

        # Clear memory cache but keep persistent cache
        self.cache.memory_cache.clear()

        # Second retrieval should hit persistent cache and reload memory cache
        result2 = self.cache.get(cache_key)
        self.assertIsNotNone(result2)
        self.assertEqual(result2.validation_score, 0.92)

    def test_cache_expiration(self):
        """Test cache entry expiration."""
        # Create cache with very short TTL
        short_cache = OptimizationCache(cache_dir=self.temp_dir, cache_ttl=1)
        cache_key = "expiration_test_key"

        # Store result
        short_cache.store(cache_key, self.mock_result)

        # Should be retrievable immediately
        result = short_cache.get(cache_key)
        self.assertIsNotNone(result)

        # Wait for expiration
        time.sleep(1.1)

        # Should be expired now
        expired_result = short_cache.get(cache_key)
        self.assertIsNone(expired_result)

    def test_cache_invalidation(self):
        """Test cache entry invalidation."""
        cache_key = "invalidation_test_key"

        # Store result
        self.cache.store(cache_key, self.mock_result)

        # Verify it's stored
        result = self.cache.get(cache_key)
        self.assertIsNotNone(result)

        # Invalidate entry
        success = self.cache.invalidate(cache_key)
        self.assertTrue(success)

        # Should be gone now
        invalidated_result = self.cache.get(cache_key)
        self.assertIsNone(invalidated_result)

        # Check statistics
        self.assertEqual(self.cache.stats["invalidations"], 1)

    def test_cache_clear(self):
        """Test clearing all cache entries."""
        # Store multiple entries
        for i in range(3):
            cache_key = f"clear_test_key_{i}"
            self.cache.store(cache_key, self.mock_result)

        # Verify entries exist
        stats_before = self.cache.get_stats()
        self.assertEqual(stats_before["memory_entries"], 3)
        self.assertEqual(stats_before["persistent_entries"], 3)

        # Clear cache
        success = self.cache.clear()
        self.assertTrue(success)

        # Verify cache is empty
        stats_after = self.cache.get_stats()
        self.assertEqual(stats_after["memory_entries"], 0)
        self.assertEqual(stats_after["persistent_entries"], 0)

    def test_is_fresh(self):
        """Test cache freshness checking."""
        cache_key = "freshness_test_key"

        # Non-existent key should not be fresh
        self.assertFalse(self.cache.is_fresh(cache_key))

        # Store entry
        self.cache.store(cache_key, self.mock_result)

        # Should be fresh immediately
        self.assertTrue(self.cache.is_fresh(cache_key))

        # Test with expired cache
        short_cache = OptimizationCache(cache_dir=self.temp_dir, cache_ttl=1)
        short_cache.store(cache_key, self.mock_result)
        time.sleep(1.1)
        self.assertFalse(short_cache.is_fresh(cache_key))

    def test_get_cache_info(self):
        """Test getting cache entry information."""
        cache_key = "info_test_key"

        # Non-existent key should return None
        info = self.cache.get_cache_info(cache_key)
        self.assertIsNone(info)

        # Store entry
        self.cache.store(cache_key, self.mock_result)

        # Get info from memory cache
        info = self.cache.get_cache_info(cache_key)
        self.assertIsNotNone(info)
        self.assertEqual(info["cache_key"], cache_key)
        self.assertEqual(info["location"], "memory")
        self.assertTrue(info["is_fresh"])

        # Clear memory cache and get info from persistent cache
        self.cache.memory_cache.clear()
        info = self.cache.get_cache_info(cache_key)
        self.assertIsNotNone(info)
        self.assertEqual(info["location"], "persistent")
        self.assertIn("metadata", info)
        self.assertIn("file_size", info)

    def test_list_cache_entries(self):
        """Test listing cache entries."""
        # Store entries for different domains
        domains = ["mathematics", "science", "technology"]
        for i, domain in enumerate(domains):
            test_module = STREAMContentGenerator(domain)

            result = OptimizationResult(
                optimized_module=test_module,
                optimization_metrics={"accuracy": 0.9 + i * 0.01},
                training_time=10.0 + i,
                validation_score=0.9 + i * 0.01,
                cache_key=f"test_key_{i}",
                timestamp=datetime.now(),
            )

            self.cache.store(f"test_key_{i}", result)

        # List all entries
        all_entries = self.cache.list_cache_entries()
        self.assertEqual(len(all_entries), 3)

        # List entries for specific domain
        math_entries = self.cache.list_cache_entries(domain_filter="mathematics")
        self.assertEqual(len(math_entries), 1)
        self.assertEqual(math_entries[0]["domain"], "mathematics")

    def test_cache_maintenance(self):
        """Test cache maintenance operations."""
        # Store some entries
        for i in range(3):
            cache_key = f"maintenance_test_key_{i}"
            self.cache.store(cache_key, self.mock_result)

        # Create an expired entry
        short_cache = OptimizationCache(cache_dir=self.temp_dir, cache_ttl=1)
        short_cache.store("expired_key", self.mock_result)
        time.sleep(1.1)

        # Run maintenance
        results = short_cache.maintenance()

        # Should have cleaned up expired entries
        self.assertGreaterEqual(results["expired_cleaned"], 1)

    def test_thread_safety(self):
        """Test thread safety of cache operations."""
        import threading

        cache_key = "thread_test_key"
        results = []

        def store_and_retrieve():
            self.cache.store(cache_key, self.mock_result)
            result = self.cache.get(cache_key)
            results.append(result is not None)

        # Run multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=store_and_retrieve)
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # All operations should have succeeded
        self.assertTrue(all(results))

    def test_error_handling(self):
        """Test error handling in cache operations."""
        # Test with cache that has permission issues
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = OptimizationCache(cache_dir=temp_dir)

            # Mock a file operation error for persistent storage only
            original_open = open

            def mock_open(path, mode="r", **kwargs):
                if str(path).endswith(".pkl") or str(path).endswith(".tmp"):
                    raise PermissionError("Permission denied")
                return original_open(path, mode, **kwargs)

            with patch("builtins.open", side_effect=mock_open):
                success = cache.store("test_key", self.mock_result)
                self.assertFalse(success)

            # The item should still be in memory cache even if persistent storage failed
            result = cache.get("test_key")
            self.assertIsNotNone(result)  # Should be in memory cache

            # Clear memory cache, then it should be None since persistent failed
            cache.memory_cache.clear()
            result = cache.get("test_key")
            self.assertIsNone(result)

    def test_cache_statistics(self):
        """Test cache statistics collection."""
        # Initial stats
        stats = self.cache.get_stats()
        self.assertEqual(stats["memory_entries"], 0)
        self.assertEqual(stats["persistent_entries"], 0)
        self.assertEqual(stats["stats"]["hits"], 0)
        self.assertEqual(stats["stats"]["misses"], 0)

        # Store and retrieve entries
        cache_key = "stats_test_key"
        self.cache.store(cache_key, self.mock_result)
        self.cache.get(cache_key)  # Hit
        self.cache.get("non_existent_key")  # Miss

        # Check updated stats
        stats = self.cache.get_stats()
        self.assertEqual(stats["memory_entries"], 1)
        self.assertEqual(stats["persistent_entries"], 1)
        self.assertEqual(stats["stats"]["hits"], 1)
        self.assertEqual(stats["stats"]["misses"], 1)
        self.assertEqual(stats["stats"]["stores"], 1)
        self.assertGreater(stats["hit_rate"], 0)

    def test_global_cache_instance(self):
        """Test global cache instance singleton."""
        cache1 = get_optimization_cache()
        cache2 = get_optimization_cache()

        # Should be the same instance
        self.assertIs(cache1, cache2)
        self.assertIsInstance(cache1, OptimizationCache)


class TestCacheErrorHandling(unittest.TestCase):
    """Test cache error handling and edge cases."""

    def test_cache_corruption_error(self):
        """Test cache corruption error handling."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create cache with invalid directory permissions
            cache_dir = Path(temp_dir) / "cache"
            cache_dir.mkdir()

            # Make directory read-only (on Unix systems)
            if os.name != "nt":  # Skip on Windows
                cache_dir.chmod(0o444)

                with self.assertRaises(CacheCorruptionError):
                    OptimizationCache(cache_dir=str(cache_dir))

    def test_corrupted_cache_file_handling(self):
        """Test handling of corrupted cache files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = OptimizationCache(cache_dir=temp_dir)

            # Create a corrupted cache file
            corrupted_file = Path(temp_dir) / "corrupted_key.pkl"
            corrupted_file.write_text("This is not a valid pickle file")

            # Should handle corrupted file gracefully
            result = cache.get("corrupted_key")
            self.assertIsNone(result)

            # Corrupted file should be removed
            self.assertFalse(corrupted_file.exists())

    def test_cache_key_generation_error(self):
        """Test cache key generation error handling."""
        cache = OptimizationCache()

        # Test with non-serializable quality requirements
        class NonSerializable:
            pass

        with self.assertRaises(CacheCorruptionError):
            cache.generate_cache_key(
                "mathematics",
                "input -> output",
                {"non_serializable": NonSerializable()},
            )
