"""
MARL System Monitor.

This module implements system-level monitoring for the MARL coordination system,
including resource utilization, health checks, and performance bottleneck detection.
"""

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import psutil
import torch

from utils.logging_config import get_logger


class HealthStatus(Enum):
    """System health status levels."""

    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class ResourceType(Enum):
    """Types of system resources to monitor."""

    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    DISK = "disk"
    NETWORK = "network"


@dataclass
class ResourceThresholds:
    """Thresholds for resource monitoring."""

    warning_threshold: float = 0.75
    critical_threshold: float = 0.90
    sustained_duration: float = 30.0  # seconds


@dataclass
class HealthCheck:
    """Individual health check result."""

    name: str
    status: HealthStatus
    value: Optional[float]
    threshold: Optional[float]
    message: str
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemSnapshot:
    """Snapshot of system state at a point in time."""

    timestamp: float
    cpu_percent: float
    memory_percent: float
    gpu_percent: Optional[float]
    disk_usage_percent: float
    network_io: Dict[str, int]
    process_count: int
    load_average: Tuple[float, float, float]
    uptime: float


class SystemMonitor:
    """
    System-level monitor for MARL coordination infrastructure.

    Monitors system resources, performs health checks, and detects
    performance bottlenecks that could affect MARL coordination.
    """

    def __init__(self, check_interval: float = 5.0):
        """
        Initialize the system monitor.

        Args:
            check_interval: Interval between health checks in seconds
        """
        self.check_interval = check_interval
        self.logger = get_logger(__name__)

        # Resource thresholds
        self._resource_thresholds: Dict[ResourceType, ResourceThresholds] = {
            ResourceType.CPU: ResourceThresholds(0.80, 0.95, 30.0),
            ResourceType.MEMORY: ResourceThresholds(0.85, 0.95, 30.0),
            ResourceType.GPU: ResourceThresholds(0.90, 0.98, 60.0),
            ResourceType.DISK: ResourceThresholds(0.85, 0.95, 300.0),
        }

        # Monitoring state
        self._monitoring_active = False
        self._monitoring_task: Optional[asyncio.Task] = None
        self._start_time = time.time()

        # Health check history
        self._health_history: deque = deque(maxlen=1000)
        self._system_snapshots: deque = deque(maxlen=500)

        # Alert tracking
        self._active_alerts: Dict[str, HealthCheck] = {}
        self._alert_callbacks: List[callable] = []

        # Resource usage tracking
        self._resource_usage: Dict[ResourceType, deque] = {
            resource_type: deque(maxlen=100) for resource_type in ResourceType
        }

        self.logger.info("System monitor initialized")

    async def start_monitoring(self):
        """Start continuous system monitoring."""
        if self._monitoring_active:
            self.logger.warning("System monitoring already active")
            return

        self._monitoring_active = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.logger.info("System monitoring started")

    async def stop_monitoring(self):
        """Stop system monitoring."""
        if not self._monitoring_active:
            return

        self._monitoring_active = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass

        self.logger.info("System monitoring stopped")

    def set_resource_thresholds(
        self, resource_type: ResourceType, thresholds: ResourceThresholds
    ):
        """
        Set custom thresholds for a resource type.

        Args:
            resource_type: Type of resource
            thresholds: Threshold configuration
        """
        self._resource_thresholds[resource_type] = thresholds
        self.logger.info("Updated thresholds for %s", resource_type.value)

    def add_alert_callback(self, callback: callable):
        """
        Add a callback function for alert notifications.

        Args:
            callback: Function to call when alerts are triggered
        """
        self._alert_callbacks.append(callback)

    def get_system_snapshot(self) -> SystemSnapshot:
        """
        Get current system snapshot.

        Returns:
            Current system state snapshot
        """
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)

            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent

            # GPU usage (if available)
            gpu_percent = None
            if torch.cuda.is_available():
                try:
                    memory_allocated = torch.cuda.memory_allocated()
                    memory_cached = torch.cuda.memory_reserved()
                    total_memory = torch.cuda.get_device_properties(0).total_memory
                    gpu_percent = (
                        (memory_allocated + memory_cached) / total_memory
                    ) * 100
                except Exception as e:
                    self.logger.debug("Could not get GPU usage: %s", str(e))

            # Disk usage
            disk = psutil.disk_usage("/")
            disk_usage_percent = (disk.used / disk.total) * 100

            # Network I/O
            network_io = psutil.net_io_counters()._asdict()

            # Process count
            process_count = len(psutil.pids())

            # Load average
            load_average = psutil.getloadavg()

            # Uptime
            uptime = time.time() - self._start_time

            return SystemSnapshot(
                timestamp=time.time(),
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                gpu_percent=gpu_percent,
                disk_usage_percent=disk_usage_percent,
                network_io=network_io,
                process_count=process_count,
                load_average=load_average,
                uptime=uptime,
            )

        except Exception as e:
            self.logger.error("Error creating system snapshot: %s", str(e))
            return SystemSnapshot(
                timestamp=time.time(),
                cpu_percent=0.0,
                memory_percent=0.0,
                gpu_percent=None,
                disk_usage_percent=0.0,
                network_io={},
                process_count=0,
                load_average=(0.0, 0.0, 0.0),
                uptime=0.0,
            )

    def perform_health_checks(self) -> List[HealthCheck]:
        """
        Perform comprehensive health checks.

        Returns:
            List of health check results
        """
        health_checks = []
        snapshot = self.get_system_snapshot()

        # CPU health check
        cpu_check = self._check_resource_health(
            "cpu_usage",
            snapshot.cpu_percent / 100.0,
            self._resource_thresholds[ResourceType.CPU],
            "CPU usage",
        )
        health_checks.append(cpu_check)

        # Memory health check
        memory_check = self._check_resource_health(
            "memory_usage",
            snapshot.memory_percent / 100.0,
            self._resource_thresholds[ResourceType.MEMORY],
            "Memory usage",
        )
        health_checks.append(memory_check)

        # GPU health check (if available)
        if snapshot.gpu_percent is not None:
            gpu_check = self._check_resource_health(
                "gpu_usage",
                snapshot.gpu_percent / 100.0,
                self._resource_thresholds[ResourceType.GPU],
                "GPU usage",
            )
            health_checks.append(gpu_check)

        # Disk health check
        disk_check = self._check_resource_health(
            "disk_usage",
            snapshot.disk_usage_percent / 100.0,
            self._resource_thresholds[ResourceType.DISK],
            "Disk usage",
        )
        health_checks.append(disk_check)

        # Load average check
        load_check = self._check_load_average(snapshot.load_average)
        health_checks.append(load_check)

        # Process count check
        process_check = self._check_process_count(snapshot.process_count)
        health_checks.append(process_check)

        # Store health checks
        for check in health_checks:
            self._health_history.append(check)

        # Store system snapshot
        self._system_snapshots.append(snapshot)

        return health_checks

    def _check_resource_health(
        self, name: str, value: float, thresholds: ResourceThresholds, description: str
    ) -> HealthCheck:
        """Check health of a resource based on thresholds."""
        if value >= thresholds.critical_threshold:
            status = HealthStatus.CRITICAL
            message = f"{description} is critical: {value:.1%}"
        elif value >= thresholds.warning_threshold:
            status = HealthStatus.WARNING
            message = f"{description} is high: {value:.1%}"
        else:
            status = HealthStatus.HEALTHY
            message = f"{description} is normal: {value:.1%}"

        return HealthCheck(
            name=name,
            status=status,
            value=value,
            threshold=thresholds.warning_threshold,
            message=message,
            timestamp=time.time(),
            metadata={
                "warning_threshold": thresholds.warning_threshold,
                "critical_threshold": thresholds.critical_threshold,
            },
        )

    def _check_load_average(
        self, load_average: Tuple[float, float, float]
    ) -> HealthCheck:
        """Check system load average health."""
        cpu_count = psutil.cpu_count()
        load_1min = load_average[0]

        # Normalize load by CPU count
        normalized_load = load_1min / cpu_count if cpu_count > 0 else load_1min

        if normalized_load > 2.0:
            status = HealthStatus.CRITICAL
            message = f"System load is very high: {load_1min:.2f} (normalized: {normalized_load:.2f})"
        elif normalized_load > 1.0:
            status = HealthStatus.WARNING
            message = f"System load is high: {load_1min:.2f} (normalized: {normalized_load:.2f})"
        else:
            status = HealthStatus.HEALTHY
            message = f"System load is normal: {load_1min:.2f} (normalized: {normalized_load:.2f})"

        return HealthCheck(
            name="load_average",
            status=status,
            value=normalized_load,
            threshold=1.0,
            message=message,
            timestamp=time.time(),
            metadata={
                "load_1min": load_1min,
                "load_5min": load_average[1],
                "load_15min": load_average[2],
                "cpu_count": cpu_count,
            },
        )

    def _check_process_count(self, process_count: int) -> HealthCheck:
        """Check process count health."""
        # These thresholds are somewhat arbitrary and system-dependent
        if process_count > 2000:
            status = HealthStatus.CRITICAL
            message = f"Very high process count: {process_count}"
        elif process_count > 1000:
            status = HealthStatus.WARNING
            message = f"High process count: {process_count}"
        else:
            status = HealthStatus.HEALTHY
            message = f"Normal process count: {process_count}"

        return HealthCheck(
            name="process_count",
            status=status,
            value=float(process_count),
            threshold=1000.0,
            message=message,
            timestamp=time.time(),
        )

    def get_overall_health_status(self) -> HealthStatus:
        """
        Get overall system health status.

        Returns:
            Overall health status
        """
        if not self._health_history:
            return HealthStatus.UNKNOWN

        # Get recent health checks (last 5 minutes)
        recent_time = time.time() - 300
        recent_checks = [
            check for check in self._health_history if check.timestamp >= recent_time
        ]

        if not recent_checks:
            return HealthStatus.UNKNOWN

        # Determine overall status based on worst individual status
        status_priority = {
            HealthStatus.HEALTHY: 0,
            HealthStatus.WARNING: 1,
            HealthStatus.CRITICAL: 2,
            HealthStatus.UNKNOWN: 3,
        }

        worst_status = HealthStatus.HEALTHY
        for check in recent_checks:
            if status_priority[check.status] > status_priority[worst_status]:
                worst_status = check.status

        return worst_status

    def get_resource_trends(
        self, resource_type: ResourceType, time_window_seconds: float = 300
    ) -> Dict[str, Any]:
        """
        Get resource usage trends.

        Args:
            resource_type: Type of resource
            time_window_seconds: Time window for trend analysis

        Returns:
            Resource trend analysis
        """
        if resource_type not in self._resource_usage:
            return {"error": "Resource type not tracked"}

        current_time = time.time()
        cutoff_time = current_time - time_window_seconds

        # Get relevant snapshots
        relevant_snapshots = [
            snapshot
            for snapshot in self._system_snapshots
            if snapshot.timestamp >= cutoff_time
        ]

        if len(relevant_snapshots) < 2:
            return {"error": "Insufficient data for trend analysis"}

        # Extract values based on resource type
        values = []
        timestamps = []

        for snapshot in relevant_snapshots:
            timestamps.append(snapshot.timestamp)

            if resource_type == ResourceType.CPU:
                values.append(snapshot.cpu_percent)
            elif resource_type == ResourceType.MEMORY:
                values.append(snapshot.memory_percent)
            elif resource_type == ResourceType.GPU and snapshot.gpu_percent is not None:
                values.append(snapshot.gpu_percent)
            elif resource_type == ResourceType.DISK:
                values.append(snapshot.disk_usage_percent)

        if not values:
            return {"error": "No data available for resource type"}

        # Calculate trend
        import numpy as np

        trend_slope = np.polyfit(range(len(values)), values, 1)[0]

        return {
            "resource_type": resource_type.value,
            "data_points": len(values),
            "time_span_seconds": timestamps[-1] - timestamps[0],
            "current_value": values[-1],
            "min_value": min(values),
            "max_value": max(values),
            "mean_value": np.mean(values),
            "trend_slope": trend_slope,
            "trend_direction": "increasing"
            if trend_slope > 0
            else "decreasing"
            if trend_slope < 0
            else "stable",
        }

    def get_performance_bottlenecks(self) -> List[Dict[str, Any]]:
        """
        Identify potential performance bottlenecks.

        Returns:
            List of identified bottlenecks
        """
        bottlenecks = []

        # Check recent health status
        recent_checks = list(self._health_history)[-20:]  # Last 20 checks

        # Group checks by name
        checks_by_name = defaultdict(list)
        for check in recent_checks:
            checks_by_name[check.name].append(check)

        # Identify sustained issues
        for check_name, checks in checks_by_name.items():
            if len(checks) >= 3:  # At least 3 data points
                warning_count = sum(
                    1 for c in checks if c.status == HealthStatus.WARNING
                )
                critical_count = sum(
                    1 for c in checks if c.status == HealthStatus.CRITICAL
                )

                if critical_count >= 2:
                    bottlenecks.append(
                        {
                            "type": "critical_resource",
                            "resource": check_name,
                            "severity": "high",
                            "description": f"{check_name} has been critical in {critical_count}/{len(checks)} recent checks",
                            "recommendation": f"Investigate {check_name} usage and consider scaling resources",
                        }
                    )
                elif warning_count >= 3:
                    bottlenecks.append(
                        {
                            "type": "resource_pressure",
                            "resource": check_name,
                            "severity": "medium",
                            "description": f"{check_name} has been elevated in {warning_count}/{len(checks)} recent checks",
                            "recommendation": f"Monitor {check_name} usage and prepare for scaling",
                        }
                    )

        # Check for resource trends
        for resource_type in [ResourceType.CPU, ResourceType.MEMORY, ResourceType.GPU]:
            trend_data = self.get_resource_trends(resource_type)
            if (
                "trend_slope" in trend_data and trend_data["trend_slope"] > 0.1
            ):  # Increasing trend
                if trend_data["current_value"] > 70:  # High current usage
                    bottlenecks.append(
                        {
                            "type": "resource_trend",
                            "resource": resource_type.value,
                            "severity": "medium",
                            "description": f"{resource_type.value} usage is trending upward (slope: {trend_data['trend_slope']:.3f})",
                            "recommendation": f"Monitor {resource_type.value} usage closely and consider proactive scaling",
                        }
                    )

        return bottlenecks

    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self._monitoring_active:
            try:
                # Perform health checks
                health_checks = self.perform_health_checks()

                # Check for new alerts
                self._process_alerts(health_checks)

                # Log periodic summary
                overall_status = self.get_overall_health_status()
                if overall_status != HealthStatus.HEALTHY:
                    self.logger.warning(
                        "System health status: %s", overall_status.value
                    )

                await asyncio.sleep(self.check_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Error in monitoring loop: %s", str(e))
                await asyncio.sleep(self.check_interval)

    def _process_alerts(self, health_checks: List[HealthCheck]):
        """Process health checks for alert conditions."""
        for check in health_checks:
            alert_key = f"{check.name}_{check.status.value}"

            # Check if this is a new alert
            if check.status in [HealthStatus.WARNING, HealthStatus.CRITICAL]:
                if alert_key not in self._active_alerts:
                    self._active_alerts[alert_key] = check
                    self._trigger_alert(check)
            else:
                # Clear resolved alerts for this resource (any status)
                keys_to_remove = [
                    key
                    for key in self._active_alerts.keys()
                    if key.startswith(f"{check.name}_")
                ]
                for key in keys_to_remove:
                    del self._active_alerts[key]

    def _trigger_alert(self, health_check: HealthCheck):
        """Trigger alert callbacks for a health check."""
        self.logger.warning("System alert: %s", health_check.message)

        for callback in self._alert_callbacks:
            try:
                callback(health_check)
            except Exception as e:
                self.logger.error("Error in alert callback: %s", str(e))

    def get_monitoring_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive monitoring summary.

        Returns:
            Monitoring summary
        """
        current_snapshot = self.get_system_snapshot()
        overall_health = self.get_overall_health_status()
        bottlenecks = self.get_performance_bottlenecks()

        return {
            "timestamp": time.time(),
            "monitoring_active": self._monitoring_active,
            "uptime_seconds": current_snapshot.uptime,
            "overall_health": overall_health.value,
            "current_snapshot": {
                "cpu_percent": current_snapshot.cpu_percent,
                "memory_percent": current_snapshot.memory_percent,
                "gpu_percent": current_snapshot.gpu_percent,
                "disk_usage_percent": current_snapshot.disk_usage_percent,
                "process_count": current_snapshot.process_count,
                "load_average": current_snapshot.load_average,
            },
            "active_alerts": len(self._active_alerts),
            "bottlenecks": len(bottlenecks),
            "health_checks_performed": len(self._health_history),
            "snapshots_collected": len(self._system_snapshots),
        }

    def reset_monitoring_data(self):
        """Reset all monitoring data."""
        self._health_history.clear()
        self._system_snapshots.clear()
        self._active_alerts.clear()
        for resource_deque in self._resource_usage.values():
            resource_deque.clear()

        self.logger.info("Monitoring data reset")


class SystemMonitorFactory:
    """Factory for creating system monitors."""

    @staticmethod
    def create(check_interval: float = 5.0) -> SystemMonitor:
        """
        Create a system monitor.

        Args:
            check_interval: Health check interval in seconds

        Returns:
            Configured system monitor
        """
        return SystemMonitor(check_interval)

    @staticmethod
    def create_with_custom_thresholds(
        cpu_thresholds: Optional[ResourceThresholds] = None,
        memory_thresholds: Optional[ResourceThresholds] = None,
        gpu_thresholds: Optional[ResourceThresholds] = None,
        disk_thresholds: Optional[ResourceThresholds] = None,
        check_interval: float = 5.0,
    ) -> SystemMonitor:
        """
        Create a system monitor with custom resource thresholds.

        Args:
            cpu_thresholds: CPU resource thresholds
            memory_thresholds: Memory resource thresholds
            gpu_thresholds: GPU resource thresholds
            disk_thresholds: Disk resource thresholds
            check_interval: Health check interval

        Returns:
            Configured system monitor
        """
        monitor = SystemMonitor(check_interval)

        if cpu_thresholds:
            monitor.set_resource_thresholds(ResourceType.CPU, cpu_thresholds)
        if memory_thresholds:
            monitor.set_resource_thresholds(ResourceType.MEMORY, memory_thresholds)
        if gpu_thresholds:
            monitor.set_resource_thresholds(ResourceType.GPU, gpu_thresholds)
        if disk_thresholds:
            monitor.set_resource_thresholds(ResourceType.DISK, disk_thresholds)

        return monitor
