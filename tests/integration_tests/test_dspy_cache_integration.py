"""
Integration tests for DSPy cache integration with optimization engine.

Tests the complete cache integration workflow including cache lookup,
storage, invalidation, and performance monitoring.
"""

# Standard Library
from datetime import datetime
from pathlib import Path
import shutil
import tempfile
from unittest.mock import Mock, patch

# SynThesisAI Modules
from core.dspy.base_module import STREAMContentGenerator
from core.dspy.cache import OptimizationCache
from core.dspy.cache_invalidation import (
    CacheInvalidationManager,
    invalidate_by_domain,
    run_cache_maintenance,
)
from core.dspy.cache_monitoring import (
    CachePerformanceMonitor,
    collect_cache_metrics,
    get_cache_performance_report,
)
from core.dspy.config import OptimizationResult
from core.dspy.optimization_engine import DSPyOptimizationEngine


class TestCacheIntegration:
    """Integration tests for cache integration with optimization engine."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        # Create temporary directory for cache
        self.temp_dir = tempfile.mkdtemp()

        # Create cache with short TTL for testing
        self.cache = OptimizationCache(cache_dir=self.temp_dir, cache_ttl=5)

        # Create test module
        self.test_module = STREAMContentGenerator("mathematics")

        # Create mock optimization result
        self.mock_result = OptimizationResult(
            optimized_module=self.test_module,
            optimization_metrics={"accuracy": 0.95},
            training_time=10.5,
            validation_score=0.92,
            cache_key="test_cache_key",
            timestamp=datetime.now(),
        )

    def teardown_method(self) -> None:
        """Clean up test fixtures."""
        # Clean up temporary directory

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("core.dspy.optimization_engine.get_optimization_cache")
    def test_cache_lookup_in_optimization_engine(self, mock_get_cache) -> None:
        """Test cache lookup in optimization engine."""
        # Set up mock cache
        mock_cache = Mock()
        mock_cache.get.return_value = self.test_module
        mock_get_cache.return_value = mock_cache

        # Create optimization engine
        engine = DSPyOptimizationEngine()

        # Mock DSPy import to avoid actual optimization
        with (
            patch("core.dspy.optimization_engine.dspy", create=True),
            patch("core.dspy.optimization_engine.MIPROv2", create=True),
        ):
            # Call optimize_for_domain
            result = engine.optimize_for_domain(self.test_module, {"min_accuracy": 0.9})

            # Verify cache lookup was performed
            mock_cache.get.assert_called_once()

            # Verify result is from cache
            assert result == self.test_module

    @patch("core.dspy.optimization_engine.get_optimization_cache")
    def test_cache_storage_after_optimization(self, mock_get_cache) -> None:
        """Test cache storage after optimization."""
        # Set up mock cache
        mock_cache = Mock()
        mock_cache.get.return_value = None  # No cache hit
        mock_cache.store.return_value = True
        mock_get_cache.return_value = mock_cache

        # Create optimization engine
        engine = DSPyOptimizationEngine()

        # Mock DSPy import and optimization
        with (
            patch("core.dspy.optimization_engine.dspy", create=True),
            patch("core.dspy.optimization_engine.MIPROv2", create=True) as mock_mipro,
            patch("core.dspy.optimization_engine.TrainingDataManager"),
        ):
            # Set up mock optimizer
            mock_optimizer = Mock()
            mock_optimizer.compile.return_value = self.test_module
            mock_mipro.return_value = mock_optimizer

            # Set up mock training data manager
            mock_tdm_instance = Mock()
            mock_tdm_instance.get_training_data.return_value = ["example1", "example2"]
            mock_tdm_instance.get_validation_data.return_value = ["val_example"]
            engine.training_manager = mock_tdm_instance

            # Call optimize_for_domain
            result = engine.optimize_for_domain(self.test_module, {"min_accuracy": 0.9})

            # Verify cache storage was performed
            mock_cache.store.assert_called_once()

            # Verify result is from optimization
            assert result == self.test_module

    def test_cache_invalidation_triggers(self) -> None:
        """Test cache invalidation triggers."""
        # Create cache invalidation manager
        invalidation_manager = CacheInvalidationManager()

        # Store test entry in cache
        cache_key = "test_invalidation_key"
        self.cache.store(cache_key, self.test_module)

        # Verify entry exists
        assert self.cache.get(cache_key) is not None

        # Mock configuration change
        with patch.object(invalidation_manager, "check_config_changes", return_value=True):
            # Invalidate by config change
            invalidated = invalidation_manager.invalidate_by_config_change()

            # Verify entry was invalidated
            assert invalidated >= 1
            assert self.cache.get(cache_key) is None

    def test_cache_performance_monitoring(self) -> None:
        """Test cache performance monitoring."""
        # Create temporary file for metrics
        metrics_file: Path = Path(self.temp_dir) / "metrics.json"

        # Create cache performance monitor
        monitor = CachePerformanceMonitor(metrics_file=metrics_file)

        # Store some entries in cache
        for i in range(3):
            self.cache.store(f"test_key_{i}", self.test_module)

        # Retrieve some entries (hits)
        for i in range(2):
            self.cache.get(f"test_key_{i}")

        # Try to retrieve non-existent entry (miss)
        self.cache.get("non_existent_key")

        # Collect metrics
        metrics = monitor.collect_metrics()

        # Verify metrics were collected
        assert "hit_rate" in metrics
        assert "hits" in metrics
        assert "misses" in metrics
        assert metrics["hits"] == 2
        assert metrics["misses"] == 1
        assert metrics["hit_rate"] == 2 / 3

        # Verify metrics were saved to file
        assert Path(metrics_file).exists()

        # Generate performance report
        report = monitor.get_performance_report()

        # Verify report structure
        assert "current_metrics" in report
        assert "hit_rate_trend" in report
        assert "recommendations" in report

    def test_end_to_end_cache_integration(self) -> None:
        """Test end-to-end cache integration workflow."""
        # Create optimization engine with real cache
        engine = DSPyOptimizationEngine()
        engine.cache = self.cache

        # Mock DSPy import and optimization
        with (
            patch("core.dspy.optimization_engine.dspy", create=True),
            patch("core.dspy.optimization_engine.MIPROv2", create=True) as mock_mipro,
            patch("core.dspy.optimization_engine.TrainingDataManager"),
        ):
            # Set up mock optimizer
            mock_optimizer = Mock()
            mock_optimizer.compile.return_value = self.test_module
            mock_mipro.return_value = mock_optimizer

            # Set up mock training data manager
            mock_tdm_instance = Mock()
            mock_tdm_instance.get_training_data.return_value = ["example1", "example2"]
            mock_tdm_instance.get_validation_data.return_value = ["val_example"]
            engine.training_manager = mock_tdm_instance

            # First optimization (should store in cache)
            result1 = engine.optimize_for_domain(self.test_module, {"min_accuracy": 0.9})

            # Second optimization (should hit cache)
            result2 = engine.optimize_for_domain(self.test_module, {"min_accuracy": 0.9})

            # Verify both results are the same
            assert result1 == result2

            # Verify cache hit
            cache_stats = self.cache.get_stats()
            assert cache_stats["stats"]["hits"] >= 1

            # Run cache maintenance
            run_cache_maintenance()

            # Collect cache metrics
            collect_cache_metrics()

            # Generate cache performance report
            report = get_cache_performance_report()

            # Verify report contains metrics
            assert "current_metrics" in report
            assert "hit_rate_trend" in report

    def test_cache_invalidation_by_domain(self) -> None:
        """Test cache invalidation by domain."""
        # Store entries for different domains
        domains = ["mathematics", "science", "technology"]
        for domain in domains:
            module = STREAMContentGenerator(domain)
            cache_key = f"test_{domain}_key"
            self.cache.store(cache_key, module)

        # Verify all entries exist
        for domain in domains:
            cache_key = f"test_{domain}_key"
            assert self.cache.get(cache_key) is not None

        # Invalidate entries for one domain
        invalidated = invalidate_by_domain("science")

        # Verify only that domain's entry was invalidated
        assert invalidated == 1
        assert self.cache.get("test_mathematics_key") is not None
        assert self.cache.get("test_science_key") is None
        assert self.cache.get("test_technology_key") is not None

    def test_cache_config_changes(self) -> None:
        """Test cache configuration changes."""
        # Create cache invalidation manager
        invalidation_manager = CacheInvalidationManager()

        # Store initial config hash
        initial_hash = invalidation_manager.config_hash

        # Mock configuration change
        with patch("core.dspy.cache_invalidation.get_dspy_config") as mock_get_config:
            # Create mock config with different values
            mock_config = Mock()
            mock_config.get_optimization_config.return_value = {"changed": True}
            mock_config.cache_ttl = 9999
            mock_get_config.return_value = mock_config

            # Check for config changes
            has_changed = invalidation_manager.check_config_changes()

            # Verify config change was detected
            assert has_changed is True
            assert invalidation_manager.config_hash != initial_hash
