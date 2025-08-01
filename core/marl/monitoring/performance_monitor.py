"""
MARL Performance Monitor.

This module implements comprehensive performance monitoring for the Multi-Agent
Reinforcement Learning coordination system, tracking coordination success rates,
agent performance, learning progress, and system efficiency metrics.
"""

# Standard Library
import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Third-Party Library
import numpy as np

# SynThesisAI Modules
from utils.logging_config import get_logger


class MetricType(Enum):
    """Types of metrics that can be tracked."""

    COORDINATION_SUCCESS = "coordination_success"
    AGENT_PERFORMANCE = "agent_performance"
    LEARNING_PROGRESS = "learning_progress"
    SYSTEM_PERFORMANCE = "system_performance"
    CONSENSUS_QUALITY = "consensus_quality"
    COMMUNICATION_EFFICIENCY = "communication_efficiency"


@dataclass
class PerformanceMetric:
    """Individual performance metric data point."""

    timestamp: float
    metric_type: MetricType
    agent_id: Optional[str]
    value: Union[float, int, bool]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CoordinationEvent:
    """Coordination event for tracking success/failure patterns."""

    timestamp: float
    event_type: str  # "coordination_request", "consensus_reached", "action_executed"
    agents_involved: List[str]
    success: bool
    duration: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MonitoringConfig:
    """Configuration for performance monitoring."""

    metrics_window_size: int = 1000
    coordination_timeout: float = 30.0
    performance_threshold: float = 0.85
    monitoring_interval: float = 1.0
    enable_detailed_logging: bool = True
    metrics_retention_hours: int = 24
    alert_thresholds: Dict[str, float] = field(
        default_factory=lambda: {
            "coordination_success_rate": 0.85,
            "average_response_time": 5.0,
            "agent_learning_rate": 0.1,
            "system_cpu_usage": 0.8,
            "memory_usage": 0.9,
        }
    )


class MARLPerformanceMonitor:
    """
    Comprehensive performance monitor for MARL coordination system.

    Tracks coordination success rates, agent performance metrics, learning progress,
    and system resource utilization to provide insights into system effectiveness.
    """

    def __init__(self, config: Optional[MonitoringConfig] = None):
        """
        Initialize the performance monitor.

        Args:
            config: Monitoring configuration
        """
        self.config = config or MonitoringConfig()
        self.logger = get_logger(__name__)

        # Metrics storage
        self._metrics: Dict[MetricType, deque] = {
            metric_type: deque(maxlen=self.config.metrics_window_size)
            for metric_type in MetricType
        }

        # Coordination tracking
        self._coordination_events: deque = deque(maxlen=self.config.metrics_window_size)
        self._active_coordinations: Dict[str, CoordinationEvent] = {}

        # Agent-specific metrics
        self._agent_metrics: Dict[str, Dict[str, deque]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=self.config.metrics_window_size))
        )

        # System metrics
        self._system_metrics: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.config.metrics_window_size)
        )

        # Monitoring state
        self._monitoring_active = False
        self._monitoring_task: Optional[asyncio.Task] = None
        self._start_time = time.time()

        self.logger.info("MARL Performance Monitor initialized")

    async def start_monitoring(self):
        """Start continuous performance monitoring."""
        if self._monitoring_active:
            self.logger.warning("Performance monitoring already active")
            return

        self._monitoring_active = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.logger.info("Performance monitoring started")

    async def stop_monitoring(self):
        """Stop continuous performance monitoring."""
        if not self._monitoring_active:
            return

        self._monitoring_active = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass

        self.logger.info("Performance monitoring stopped")

    def record_coordination_start(
        self,
        coordination_id: str,
        agents_involved: List[str],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Record the start of a coordination event.

        Args:
            coordination_id: Unique identifier for the coordination
            agents_involved: List of agent IDs involved
            metadata: Additional metadata about the coordination
        """
        event = CoordinationEvent(
            timestamp=time.time(),
            event_type="coordination_request",
            agents_involved=agents_involved,
            success=False,  # Will be updated when completed
            duration=0.0,
            metadata=metadata or {},
        )

        self._active_coordinations[coordination_id] = event
        self.logger.debug(
            "Coordination started: %s with agents %s", coordination_id, agents_involved
        )

    def record_coordination_end(
        self,
        coordination_id: str,
        success: bool,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Record the end of a coordination event.

        Args:
            coordination_id: Unique identifier for the coordination
            success: Whether the coordination was successful
            metadata: Additional metadata about the result
        """
        if coordination_id not in self._active_coordinations:
            self.logger.warning("Unknown coordination ID: %s", coordination_id)
            return

        event = self._active_coordinations.pop(coordination_id)
        event.success = success
        event.duration = time.time() - event.timestamp
        if metadata:
            event.metadata.update(metadata)

        self._coordination_events.append(event)

        # Record coordination success metric
        self.record_metric(
            MetricType.COORDINATION_SUCCESS,
            success,
            metadata={"duration": event.duration, "agents": len(event.agents_involved)},
        )

        self.logger.debug(
            "Coordination completed: %s, success: %s, duration: %.2fs",
            coordination_id,
            success,
            event.duration,
        )

    def record_metric(
        self,
        metric_type: MetricType,
        value: Union[float, int, bool],
        agent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Record a performance metric.

        Args:
            metric_type: Type of metric being recorded
            value: Metric value
            agent_id: Optional agent ID for agent-specific metrics
            metadata: Additional metadata
        """
        metric = PerformanceMetric(
            timestamp=time.time(),
            metric_type=metric_type,
            agent_id=agent_id,
            value=value,
            metadata=metadata or {},
        )

        self._metrics[metric_type].append(metric)

        # Store agent-specific metrics separately
        if agent_id:
            self._agent_metrics[agent_id][metric_type.value].append(metric)

    def record_agent_performance(
        self,
        agent_id: str,
        reward: float,
        loss: Optional[float] = None,
        action_confidence: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Record agent performance metrics.

        Args:
            agent_id: Agent identifier
            reward: Reward received by the agent
            loss: Training loss (if available)
            action_confidence: Confidence in selected action
            metadata: Additional performance metadata
        """
        base_metadata = metadata or {}

        # Record reward
        self.record_metric(
            MetricType.AGENT_PERFORMANCE,
            reward,
            agent_id=agent_id,
            metadata={**base_metadata, "metric_name": "reward"},
        )

        # Record loss if available
        if loss is not None:
            self.record_metric(
                MetricType.LEARNING_PROGRESS,
                loss,
                agent_id=agent_id,
                metadata={**base_metadata, "metric_name": "loss"},
            )

        # Record action confidence if available
        if action_confidence is not None:
            self.record_metric(
                MetricType.AGENT_PERFORMANCE,
                action_confidence,
                agent_id=agent_id,
                metadata={**base_metadata, "metric_name": "confidence"},
            )

    def record_system_metric(self, metric_name: str, value: float):
        """
        Record system-level performance metric.

        Args:
            metric_name: Name of the system metric
            value: Metric value
        """
        self._system_metrics[metric_name].append(
            {"timestamp": time.time(), "value": value}
        )

        self.record_metric(
            MetricType.SYSTEM_PERFORMANCE, value, metadata={"metric_name": metric_name}
        )

    def get_coordination_success_rate(
        self, time_window_seconds: Optional[float] = None
    ) -> float:
        """
        Get coordination success rate over a time window.

        Args:
            time_window_seconds: Time window in seconds (None for all data)

        Returns:
            Success rate as a percentage (0.0 to 1.0)
        """
        current_time = time.time()
        cutoff_time = current_time - time_window_seconds if time_window_seconds else 0

        relevant_events = [
            event
            for event in self._coordination_events
            if event.timestamp >= cutoff_time
        ]

        if not relevant_events:
            return 0.0

        successful_events = sum(1 for event in relevant_events if event.success)
        return successful_events / len(relevant_events)

    def get_agent_performance_summary(
        self, agent_id: str, time_window_seconds: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Get performance summary for a specific agent.

        Args:
            agent_id: Agent identifier
            time_window_seconds: Time window in seconds

        Returns:
            Performance summary dictionary
        """
        if agent_id not in self._agent_metrics:
            return {"error": "Agent not found"}

        current_time = time.time()
        cutoff_time = current_time - time_window_seconds if time_window_seconds else 0

        summary = {}

        for metric_name, metrics in self._agent_metrics[agent_id].items():
            relevant_metrics = [m for m in metrics if m.timestamp >= cutoff_time]

            if relevant_metrics:
                values = [
                    m.value
                    for m in relevant_metrics
                    if isinstance(m.value, (int, float))
                ]
                if values:
                    summary[metric_name] = {
                        "count": len(values),
                        "mean": np.mean(values),
                        "std": np.std(values),
                        "min": np.min(values),
                        "max": np.max(values),
                        "latest": values[-1],
                    }

        return summary

    def get_system_performance_summary(
        self, time_window_seconds: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Get system performance summary.

        Args:
            time_window_seconds: Time window in seconds

        Returns:
            System performance summary
        """
        current_time = time.time()
        cutoff_time = current_time - time_window_seconds if time_window_seconds else 0

        summary = {
            "uptime_seconds": current_time - self._start_time,
            "coordination_success_rate": self.get_coordination_success_rate(
                time_window_seconds
            ),
            "total_coordinations": len(self._coordination_events),
            "active_coordinations": len(self._active_coordinations),
            "system_metrics": {},
        }

        # Add system metrics
        for metric_name, metrics in self._system_metrics.items():
            relevant_metrics = [m for m in metrics if m["timestamp"] >= cutoff_time]

            if relevant_metrics:
                values = [m["value"] for m in relevant_metrics]
                summary["system_metrics"][metric_name] = {
                    "count": len(values),
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "latest": values[-1],
                }

        return summary

    def get_learning_progress(self, agent_id: str) -> Dict[str, Any]:
        """
        Get learning progress for a specific agent.

        Args:
            agent_id: Agent identifier

        Returns:
            Learning progress summary
        """
        if agent_id not in self._agent_metrics:
            return {"error": "Agent not found"}

        learning_metrics = self._agent_metrics[agent_id].get(
            MetricType.LEARNING_PROGRESS.value, []
        )

        if not learning_metrics:
            return {"error": "No learning metrics available"}

        # Calculate learning trends
        recent_metrics = list(learning_metrics)[-100:]  # Last 100 data points
        if len(recent_metrics) < 2:
            return {"error": "Insufficient data for trend analysis"}

        values = [m.value for m in recent_metrics if isinstance(m.value, (int, float))]
        timestamps = [m.timestamp for m in recent_metrics]

        if len(values) < 2:
            return {"error": "Insufficient numeric data"}

        # Simple trend calculation
        trend = np.polyfit(range(len(values)), values, 1)[0]

        return {
            "total_updates": len(learning_metrics),
            "recent_performance": {
                "mean": np.mean(values),
                "std": np.std(values),
                "trend": trend,
                "improving": trend < 0,  # For loss metrics, decreasing is improving
            },
            "latest_value": values[-1],
            "time_span_seconds": (
                timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0
            ),
        }

    def check_alert_conditions(self) -> List[Dict[str, Any]]:
        """
        Check for alert conditions based on configured thresholds.

        Returns:
            List of active alerts
        """
        alerts = []

        # Check coordination success rate
        success_rate = self.get_coordination_success_rate(3600)  # Last hour
        threshold = self.config.alert_thresholds.get("coordination_success_rate", 0.85)
        if success_rate < threshold:
            alerts.append(
                {
                    "type": "coordination_success_rate",
                    "severity": "warning",
                    "message": f"Coordination success rate ({success_rate:.2%}) below threshold ({threshold:.2%})",
                    "value": success_rate,
                    "threshold": threshold,
                }
            )

        # Check system metrics
        for metric_name, threshold in self.config.alert_thresholds.items():
            if metric_name in self._system_metrics:
                recent_metrics = list(self._system_metrics[metric_name])[
                    -10:
                ]  # Last 10 values
                if recent_metrics:
                    latest_value = recent_metrics[-1]["value"]
                    if latest_value > threshold:
                        alerts.append(
                            {
                                "type": metric_name,
                                "severity": "warning",
                                "message": f"{metric_name} ({latest_value:.2f}) above threshold ({threshold:.2f})",
                                "value": latest_value,
                                "threshold": threshold,
                            }
                        )

        return alerts

    async def _monitoring_loop(self):
        """Main monitoring loop for continuous system health checks."""
        while self._monitoring_active:
            try:
                # Check alert conditions
                alerts = self.check_alert_conditions()
                for alert in alerts:
                    self.logger.warning("Performance alert: %s", alert["message"])

                # Log periodic summary
                if self.config.enable_detailed_logging:
                    summary = self.get_system_performance_summary(300)  # Last 5 minutes
                    self.logger.info("System performance summary: %s", summary)

                await asyncio.sleep(self.config.monitoring_interval)

            except Exception as e:
                self.logger.error("Error in monitoring loop: %s", str(e))
                await asyncio.sleep(self.config.monitoring_interval)

    def reset_metrics(self):
        """Reset all collected metrics."""
        for metric_deque in self._metrics.values():
            metric_deque.clear()

        self._coordination_events.clear()
        self._active_coordinations.clear()
        self._agent_metrics.clear()
        self._system_metrics.clear()
        self._start_time = time.time()

        self.logger.info("Performance metrics reset")

    def export_metrics(self, filepath: Union[str, Path]) -> bool:
        """
        Export metrics to a file.

        Args:
            filepath: Path to export file

        Returns:
            True if export successful, False otherwise
        """
        try:
            import json

            export_data = {
                "timestamp": time.time(),
                "uptime_seconds": time.time() - self._start_time,
                "coordination_events": [
                    {
                        "timestamp": event.timestamp,
                        "event_type": event.event_type,
                        "agents_involved": event.agents_involved,
                        "success": event.success,
                        "duration": event.duration,
                        "metadata": event.metadata,
                    }
                    for event in self._coordination_events
                ],
                "system_summary": self.get_system_performance_summary(),
                "agent_summaries": {
                    agent_id: self.get_agent_performance_summary(agent_id)
                    for agent_id in self._agent_metrics.keys()
                },
            }

            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)

            with filepath.open("w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2, default=str)

            self.logger.info("Metrics exported to %s", filepath)
            return True

        except Exception as e:
            self.logger.error("Failed to export metrics: %s", str(e))
            return False


class MARLPerformanceMonitorFactory:
    """Factory for creating MARL performance monitors."""

    @staticmethod
    def create(config: Optional[MonitoringConfig] = None) -> MARLPerformanceMonitor:
        """
        Create a MARL performance monitor.

        Args:
            config: Optional monitoring configuration

        Returns:
            Configured performance monitor
        """
        return MARLPerformanceMonitor(config)

    @staticmethod
    def create_with_custom_config(
        metrics_window_size: int = 1000,
        coordination_timeout: float = 30.0,
        performance_threshold: float = 0.85,
        monitoring_interval: float = 1.0,
        **kwargs,
    ) -> MARLPerformanceMonitor:
        """
        Create a performance monitor with custom configuration.

        Args:
            metrics_window_size: Size of metrics window
            coordination_timeout: Timeout for coordination events
            performance_threshold: Performance threshold for alerts
            monitoring_interval: Monitoring loop interval
            **kwargs: Additional configuration parameters

        Returns:
            Configured performance monitor
        """
        config = MonitoringConfig(
            metrics_window_size=metrics_window_size,
            coordination_timeout=coordination_timeout,
            performance_threshold=performance_threshold,
            monitoring_interval=monitoring_interval,
            **kwargs,
        )

        return MARLPerformanceMonitor(config)
