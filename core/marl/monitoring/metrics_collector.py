"""
MARL Metrics Collector.

This module implements specialized metrics collection for different aspects
of the MARL system, including agent-specific metrics, coordination metrics,
and system resource metrics.
"""

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import psutil
import torch

from utils.logging_config import get_logger


class CollectionStrategy(Enum):
    """Strategies for metrics collection."""

    CONTINUOUS = "continuous"
    ON_DEMAND = "on_demand"
    PERIODIC = "periodic"
    EVENT_DRIVEN = "event_driven"


@dataclass
class MetricDefinition:
    """Definition of a metric to be collected."""

    name: str
    collection_strategy: CollectionStrategy
    collection_interval: float = 1.0
    aggregation_window: int = 100
    collector_function: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CollectedMetric:
    """A collected metric data point."""

    name: str
    value: Any
    timestamp: float
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class MetricsCollector:
    """
    Specialized metrics collector for MARL system components.

    Collects various types of metrics including agent performance,
    coordination effectiveness, and system resource utilization.
    """

    def __init__(self):
        """Initialize the metrics collector."""
        self.logger = get_logger(__name__)

        # Metrics storage
        self._metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))

        # Collection state
        self._collection_active = False
        self._collection_tasks: Dict[str, asyncio.Task] = {}

        # Registered metric definitions
        self._metric_definitions: Dict[str, MetricDefinition] = {}

        # Event handlers
        self._event_handlers: Dict[str, List[Callable]] = defaultdict(list)

        self.logger.info("Metrics collector initialized")

    def register_metric(self, metric_def: MetricDefinition):
        """
        Register a metric for collection.

        Args:
            metric_def: Metric definition
        """
        self._metric_definitions[metric_def.name] = metric_def
        self.logger.debug("Registered metric: %s", metric_def.name)

    def register_agent_metrics(self, agent_id: str):
        """
        Register standard agent metrics for collection.

        Args:
            agent_id: Agent identifier
        """
        # Agent performance metrics
        self.register_metric(
            MetricDefinition(
                name=f"agent_{agent_id}_reward",
                collection_strategy=CollectionStrategy.EVENT_DRIVEN,
                metadata={"agent_id": agent_id, "type": "performance"},
            )
        )

        self.register_metric(
            MetricDefinition(
                name=f"agent_{agent_id}_loss",
                collection_strategy=CollectionStrategy.EVENT_DRIVEN,
                metadata={"agent_id": agent_id, "type": "learning"},
            )
        )

        self.register_metric(
            MetricDefinition(
                name=f"agent_{agent_id}_epsilon",
                collection_strategy=CollectionStrategy.PERIODIC,
                collection_interval=5.0,
                metadata={"agent_id": agent_id, "type": "exploration"},
            )
        )

        self.register_metric(
            MetricDefinition(
                name=f"agent_{agent_id}_q_values",
                collection_strategy=CollectionStrategy.ON_DEMAND,
                metadata={"agent_id": agent_id, "type": "policy"},
            )
        )

    def register_coordination_metrics(self):
        """Register coordination-specific metrics."""
        self.register_metric(
            MetricDefinition(
                name="coordination_success_rate",
                collection_strategy=CollectionStrategy.CONTINUOUS,
                collection_interval=10.0,
                collector_function=self._collect_coordination_success_rate,
                metadata={"type": "coordination"},
            )
        )

        self.register_metric(
            MetricDefinition(
                name="consensus_time",
                collection_strategy=CollectionStrategy.EVENT_DRIVEN,
                metadata={"type": "coordination"},
            )
        )

        self.register_metric(
            MetricDefinition(
                name="communication_overhead",
                collection_strategy=CollectionStrategy.PERIODIC,
                collection_interval=30.0,
                collector_function=self._collect_communication_overhead,
                metadata={"type": "coordination"},
            )
        )

    def register_system_metrics(self):
        """Register system resource metrics."""
        self.register_metric(
            MetricDefinition(
                name="cpu_usage",
                collection_strategy=CollectionStrategy.CONTINUOUS,
                collection_interval=2.0,
                collector_function=self._collect_cpu_usage,
                metadata={"type": "system"},
            )
        )

        self.register_metric(
            MetricDefinition(
                name="memory_usage",
                collection_strategy=CollectionStrategy.CONTINUOUS,
                collection_interval=2.0,
                collector_function=self._collect_memory_usage,
                metadata={"type": "system"},
            )
        )

        self.register_metric(
            MetricDefinition(
                name="gpu_usage",
                collection_strategy=CollectionStrategy.CONTINUOUS,
                collection_interval=5.0,
                collector_function=self._collect_gpu_usage,
                metadata={"type": "system"},
            )
        )

    async def start_collection(self):
        """Start metrics collection for all registered metrics."""
        if self._collection_active:
            self.logger.warning("Metrics collection already active")
            return

        self._collection_active = True

        # Start collection tasks for different strategies
        for metric_name, metric_def in self._metric_definitions.items():
            if metric_def.collection_strategy == CollectionStrategy.CONTINUOUS:
                task = asyncio.create_task(
                    self._continuous_collection_loop(metric_name, metric_def)
                )
                self._collection_tasks[metric_name] = task

            elif metric_def.collection_strategy == CollectionStrategy.PERIODIC:
                task = asyncio.create_task(
                    self._periodic_collection_loop(metric_name, metric_def)
                )
                self._collection_tasks[metric_name] = task

        self.logger.info(
            "Metrics collection started for %d metrics", len(self._collection_tasks)
        )

    async def stop_collection(self):
        """Stop all metrics collection."""
        if not self._collection_active:
            return

        self._collection_active = False

        # Cancel all collection tasks
        for task in self._collection_tasks.values():
            task.cancel()

        # Wait for tasks to complete
        if self._collection_tasks:
            await asyncio.gather(
                *self._collection_tasks.values(), return_exceptions=True
            )

        self._collection_tasks.clear()
        self.logger.info("Metrics collection stopped")

    def collect_metric(
        self,
        metric_name: str,
        value: Any,
        source: str = "manual",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Manually collect a metric value.

        Args:
            metric_name: Name of the metric
            value: Metric value
            source: Source of the metric
            metadata: Additional metadata
        """
        metric = CollectedMetric(
            name=metric_name,
            value=value,
            timestamp=time.time(),
            source=source,
            metadata=metadata or {},
        )

        self._metrics[metric_name].append(metric)

        # Trigger event handlers
        for handler in self._event_handlers.get(metric_name, []):
            try:
                handler(metric)
            except Exception as e:
                self.logger.error("Error in metric event handler: %s", str(e))

    def get_metric_history(
        self, metric_name: str, count: Optional[int] = None
    ) -> List[CollectedMetric]:
        """
        Get historical values for a metric.

        Args:
            metric_name: Name of the metric
            count: Number of recent values to return (None for all)

        Returns:
            List of collected metrics
        """
        if metric_name not in self._metrics:
            return []

        metrics = list(self._metrics[metric_name])
        if count is not None:
            metrics = metrics[-count:]

        return metrics

    def get_metric_statistics(
        self, metric_name: str, time_window_seconds: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Get statistical summary for a metric.

        Args:
            metric_name: Name of the metric
            time_window_seconds: Time window for statistics

        Returns:
            Statistical summary
        """
        metrics = self.get_metric_history(metric_name)

        if time_window_seconds:
            cutoff_time = time.time() - time_window_seconds
            metrics = [m for m in metrics if m.timestamp >= cutoff_time]

        if not metrics:
            return {"error": "No data available"}

        # Extract numeric values
        numeric_values = []
        for metric in metrics:
            if isinstance(metric.value, (int, float)):
                numeric_values.append(metric.value)
            elif isinstance(metric.value, bool):
                numeric_values.append(float(metric.value))

        if not numeric_values:
            return {"error": "No numeric data available"}

        return {
            "count": len(numeric_values),
            "mean": np.mean(numeric_values),
            "std": np.std(numeric_values),
            "min": np.min(numeric_values),
            "max": np.max(numeric_values),
            "median": np.median(numeric_values),
            "latest": numeric_values[-1],
            "time_span_seconds": metrics[-1].timestamp - metrics[0].timestamp
            if len(metrics) > 1
            else 0,
        }

    def add_event_handler(self, metric_name: str, handler: Callable):
        """
        Add an event handler for a specific metric.

        Args:
            metric_name: Name of the metric
            handler: Handler function
        """
        self._event_handlers[metric_name].append(handler)

    async def _continuous_collection_loop(
        self, metric_name: str, metric_def: MetricDefinition
    ):
        """Continuous collection loop for a metric."""
        while self._collection_active:
            try:
                if metric_def.collector_function:
                    value = await self._safe_collect(metric_def.collector_function)
                    if value is not None:
                        self.collect_metric(
                            metric_name,
                            value,
                            source="continuous",
                            metadata=metric_def.metadata,
                        )

                await asyncio.sleep(metric_def.collection_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(
                    "Error in continuous collection for %s: %s", metric_name, str(e)
                )
                await asyncio.sleep(metric_def.collection_interval)

    async def _periodic_collection_loop(
        self, metric_name: str, metric_def: MetricDefinition
    ):
        """Periodic collection loop for a metric."""
        while self._collection_active:
            try:
                if metric_def.collector_function:
                    value = await self._safe_collect(metric_def.collector_function)
                    if value is not None:
                        self.collect_metric(
                            metric_name,
                            value,
                            source="periodic",
                            metadata=metric_def.metadata,
                        )

                await asyncio.sleep(metric_def.collection_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(
                    "Error in periodic collection for %s: %s", metric_name, str(e)
                )
                await asyncio.sleep(metric_def.collection_interval)

    async def _safe_collect(self, collector_function: Callable) -> Any:
        """Safely execute a collector function."""
        try:
            if asyncio.iscoroutinefunction(collector_function):
                return await collector_function()
            else:
                return collector_function()
        except Exception as e:
            self.logger.error("Error in collector function: %s", str(e))
            return None

    # Built-in collector functions

    def _collect_cpu_usage(self) -> float:
        """Collect CPU usage percentage."""
        return psutil.cpu_percent(interval=0.1)

    def _collect_memory_usage(self) -> float:
        """Collect memory usage percentage."""
        return psutil.virtual_memory().percent / 100.0

    def _collect_gpu_usage(self) -> Optional[float]:
        """Collect GPU usage if available."""
        try:
            if torch.cuda.is_available():
                # Get GPU memory usage
                memory_allocated = torch.cuda.memory_allocated()
                memory_cached = torch.cuda.memory_reserved()
                total_memory = torch.cuda.get_device_properties(0).total_memory

                usage = (memory_allocated + memory_cached) / total_memory
                return usage
            return None
        except Exception as e:
            self.logger.debug("Could not collect GPU usage: %s", str(e))
            return None

    def _collect_coordination_success_rate(self) -> Optional[float]:
        """Collect coordination success rate from recent events."""
        # This would typically interface with the performance monitor
        # For now, return a placeholder
        return None

    def _collect_communication_overhead(self) -> Optional[float]:
        """Collect communication overhead metrics."""
        # This would typically interface with the communication protocol
        # For now, return a placeholder
        return None

    def get_all_metrics_summary(self) -> Dict[str, Any]:
        """
        Get summary of all collected metrics.

        Returns:
            Summary of all metrics
        """
        summary = {
            "total_metrics": len(self._metrics),
            "collection_active": self._collection_active,
            "active_tasks": len(self._collection_tasks),
            "metrics": {},
        }

        for metric_name in self._metrics.keys():
            stats = self.get_metric_statistics(
                metric_name, time_window_seconds=300
            )  # Last 5 minutes
            summary["metrics"][metric_name] = stats

        return summary

    def reset_metrics(self):
        """Reset all collected metrics."""
        self._metrics.clear()
        self.logger.info("All metrics reset")

    def export_metrics_data(self) -> Dict[str, Any]:
        """
        Export all metrics data for external analysis.

        Returns:
            Dictionary containing all metrics data
        """
        export_data = {"timestamp": time.time(), "metrics": {}}

        for metric_name, metrics in self._metrics.items():
            export_data["metrics"][metric_name] = [
                {
                    "name": m.name,
                    "value": m.value,
                    "timestamp": m.timestamp,
                    "source": m.source,
                    "metadata": m.metadata,
                }
                for m in metrics
            ]

        return export_data


class MetricsCollectorFactory:
    """Factory for creating metrics collectors."""

    @staticmethod
    def create_standard_collector() -> MetricsCollector:
        """
        Create a metrics collector with standard metrics registered.

        Returns:
            Configured metrics collector
        """
        collector = MetricsCollector()

        # Register standard system metrics
        collector.register_system_metrics()
        collector.register_coordination_metrics()

        return collector

    @staticmethod
    def create_agent_collector(agent_ids: List[str]) -> MetricsCollector:
        """
        Create a metrics collector configured for specific agents.

        Args:
            agent_ids: List of agent identifiers

        Returns:
            Configured metrics collector
        """
        collector = MetricsCollector()

        # Register metrics for each agent
        for agent_id in agent_ids:
            collector.register_agent_metrics(agent_id)

        # Register standard metrics
        collector.register_system_metrics()
        collector.register_coordination_metrics()

        return collector
