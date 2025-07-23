"""
Cache performance monitoring for DSPy optimization.

This module provides functionality for monitoring cache performance
and collecting metrics for optimization caching.
"""

import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from .cache import get_optimization_cache
from .config import get_dspy_config

logger = logging.getLogger(__name__)


class CachePerformanceMonitor:
    """
    Monitors cache performance for DSPy optimization.

    Collects and analyzes cache performance metrics to optimize
    caching strategies and improve performance.
    """

    def __init__(self, metrics_file: str = None):
        """
        Initialize cache performance monitor.

        Args:
            metrics_file: Optional file path for storing metrics
        """
        self.cache = get_optimization_cache()
        self.config = get_dspy_config()
        self.logger = logging.getLogger(__name__ + ".CachePerformanceMonitor")

        # Set up metrics storage
        self.metrics_history: List[Dict[str, Any]] = []
        self.metrics_file = metrics_file or ".cache/dspy/cache_metrics.json"
        self.metrics_file_path = Path(self.metrics_file)
        self.metrics_file_path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing metrics if available
        self._load_metrics()

        self.logger.info("Initialized cache performance monitor")

    def _load_metrics(self) -> None:
        """Load metrics from file if available."""
        if self.metrics_file_path.exists():
            try:
                self.metrics_history = json.loads(
                    self.metrics_file_path.read_text(encoding="utf-8")
                )
                self.logger.info(
                    "Loaded %d metrics records from file", len(self.metrics_history)
                )
            except Exception as e:
                self.logger.error("Failed to load metrics from file: %s", str(e))

    def _save_metrics(self) -> None:
        """Save metrics to file."""
        try:
            self.metrics_file_path.write_text(
                json.dumps(self.metrics_history, indent=2), encoding="utf-8"
            )
            self.logger.debug(
                "Saved %d metrics records to file", len(self.metrics_history)
            )
        except Exception as e:
            self.logger.error("Failed to save metrics to file: %s", str(e))

    def collect_metrics(self) -> Dict[str, Any]:
        """
        Collect current cache performance metrics.

        Returns:
            Dictionary with cache metrics
        """
        # Get cache statistics
        cache_stats = self.cache.get_stats()

        # Calculate additional metrics
        total_requests = cache_stats["stats"]["hits"] + cache_stats["stats"]["misses"]
        hit_rate = (
            cache_stats["stats"]["hits"] / total_requests if total_requests > 0 else 0.0
        )

        # Create metrics record
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "memory_entries": cache_stats["memory_entries"],
            "persistent_entries": cache_stats["persistent_entries"],
            "total_size_mb": cache_stats["total_size_mb"],
            "hit_rate": hit_rate,
            "hits": cache_stats["stats"]["hits"],
            "misses": cache_stats["stats"]["misses"],
            "stores": cache_stats["stats"]["stores"],
            "invalidations": cache_stats["stats"]["invalidations"],
            "errors": cache_stats["stats"]["errors"],
        }

        # Add to history
        self.metrics_history.append(metrics)

        # Limit history size
        max_history = self.config.metrics_history_size
        if len(self.metrics_history) > max_history:
            self.metrics_history = self.metrics_history[-max_history:]

        # Save metrics
        self._save_metrics()

        return metrics

    def get_metrics_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """
        Get metrics history for a specific time period.

        Args:
            hours: Number of hours to include

        Returns:
            List of metrics records
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        cutoff_str = cutoff_time.isoformat()

        return [
            metrics
            for metrics in self.metrics_history
            if metrics["timestamp"] >= cutoff_str
        ]

    def get_hit_rate_trend(self, hours: int = 24) -> List[Dict[str, Any]]:
        """
        Get hit rate trend over time.

        Args:
            hours: Number of hours to include

        Returns:
            List of hit rate records
        """
        metrics_history = self.get_metrics_history(hours)

        return [
            {
                "timestamp": metrics["timestamp"],
                "hit_rate": metrics["hit_rate"],
            }
            for metrics in metrics_history
        ]

    def get_size_trend(self, hours: int = 24) -> List[Dict[str, Any]]:
        """
        Get cache size trend over time.

        Args:
            hours: Number of hours to include

        Returns:
            List of size records
        """
        metrics_history = self.get_metrics_history(hours)

        return [
            {
                "timestamp": metrics["timestamp"],
                "total_size_mb": metrics["total_size_mb"],
                "memory_entries": metrics["memory_entries"],
                "persistent_entries": metrics["persistent_entries"],
            }
            for metrics in metrics_history
        ]

    def get_performance_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive cache performance report.

        Returns:
            Dictionary with performance report
        """
        # Get current metrics
        current_metrics = self.collect_metrics()

        # Get metrics history
        metrics_history = self.get_metrics_history(24)

        # Calculate trends
        hit_rate_trend = self.get_hit_rate_trend(24)
        size_trend = self.get_size_trend(24)

        # Calculate average hit rate
        avg_hit_rate = (
            sum(m["hit_rate"] for m in metrics_history) / len(metrics_history)
            if metrics_history
            else 0.0
        )

        # Calculate hit rate change
        hit_rate_change = 0.0
        if len(hit_rate_trend) >= 2:
            first_hit_rate = hit_rate_trend[0]["hit_rate"]
            last_hit_rate = hit_rate_trend[-1]["hit_rate"]
            hit_rate_change = last_hit_rate - first_hit_rate

        # Generate recommendations
        recommendations = []

        if current_metrics["hit_rate"] < 0.5:
            recommendations.append("Consider increasing cache TTL to improve hit rate")

        if current_metrics["total_size_mb"] > 100:
            recommendations.append(
                "Consider reducing cache size by decreasing TTL or running maintenance"
            )

        if current_metrics["errors"] > 0:
            recommendations.append(
                "Investigate cache errors and consider running maintenance"
            )

        # Create report
        report = {
            "timestamp": datetime.now().isoformat(),
            "current_metrics": current_metrics,
            "avg_hit_rate_24h": avg_hit_rate,
            "hit_rate_change_24h": hit_rate_change,
            "hit_rate_trend": hit_rate_trend,
            "size_trend": size_trend,
            "recommendations": recommendations,
        }

        return report


# Global monitor instance
_cache_monitor = None


def get_cache_monitor() -> CachePerformanceMonitor:
    """Get the global cache performance monitor instance."""
    global _cache_monitor
    if _cache_monitor is None:
        _cache_monitor = CachePerformanceMonitor()
    return _cache_monitor


def collect_cache_metrics() -> Dict[str, Any]:
    """
    Collect current cache performance metrics.

    Returns:
        Dictionary with cache metrics
    """
    monitor = get_cache_monitor()
    return monitor.collect_metrics()


def get_cache_performance_report() -> Dict[str, Any]:
    """
    Generate comprehensive cache performance report.

    Returns:
        Dictionary with performance report
    """
    monitor = get_cache_monitor()
    return monitor.get_performance_report()
