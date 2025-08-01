"""Distributed Resource Management System.

This module provides resource management capabilities for distributed MARL
deployment, including resource allocation, monitoring, and optimization.
"""

import asyncio
import json
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import psutil
import torch

from utils.logging_config import get_logger


class ResourceType(Enum):
    """Resource type enumeration."""

    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    STORAGE = "storage"
    NETWORK = "network"


class AllocationStrategy(Enum):
    """Resource allocation strategy."""

    ROUND_ROBIN = "round_robin"
    LOAD_BALANCED = "load_balanced"
    PRIORITY_BASED = "priority_based"
    PERFORMANCE_OPTIMIZED = "performance_optimized"


@dataclass
class ResourceConfig:
    """Configuration for resource management."""

    # Resource limits
    max_cpu_percent: float = 80.0
    max_memory_percent: float = 80.0
    max_gpu_memory_percent: float = 90.0
    max_storage_percent: float = 85.0

    # Allocation settings
    allocation_strategy: AllocationStrategy = AllocationStrategy.LOAD_BALANCED
    enable_auto_scaling: bool = True
    scaling_threshold: float = 0.7
    scaling_cooldown: float = 300.0  # 5 minutes

    # Monitoring settings
    monitoring_interval: float = 10.0
    resource_history_size: int = 1000
    alert_thresholds: Dict[str, float] = field(
        default_factory=lambda: {
            "cpu": 85.0,
            "memory": 85.0,
            "gpu": 95.0,
            "storage": 90.0,
        }
    )

    # Optimization settings
    enable_resource_optimization: bool = True
    optimization_interval: float = 60.0
    rebalancing_threshold: float = 0.3

    def __post_init__(self):
        """Validate configuration."""
        for percent in [
            self.max_cpu_percent,
            self.max_memory_percent,
            self.max_gpu_memory_percent,
            self.max_storage_percent,
        ]:
            if percent <= 0 or percent > 100:
                raise ValueError("Resource percentages must be between 0 and 100")


@dataclass
class ResourceAllocation:
    """Resource allocation information."""

    resource_id: str
    resource_type: ResourceType
    allocated_amount: float
    total_amount: float
    allocation_time: float
    owner: str
    priority: int = 1

    @property
    def utilization_percent(self) -> float:
        """Get utilization percentage."""
        if self.total_amount == 0:
            return 0.0
        return (self.allocated_amount / self.total_amount) * 100


@dataclass
class ResourceMetrics:
    """Resource usage metrics."""

    timestamp: float
    cpu_percent: float
    memory_percent: float
    gpu_percent: float
    storage_percent: float
    network_io: Dict[str, float]
    process_count: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "cpu_percent": self.cpu_percent,
            "memory_percent": self.memory_percent,
            "gpu_percent": self.gpu_percent,
            "storage_percent": self.storage_percent,
            "network_io": self.network_io,
            "process_count": self.process_count,
        }


class ResourceManager:
    """Distributed resource management system.

    Manages resource allocation, monitoring, and optimization across
    distributed MARL deployment nodes.
    """

    def __init__(self, config: ResourceConfig):
        """
        Initialize resource manager.

        Args:
            config: Resource management configuration
        """
        self.config = config
        self.logger = get_logger(__name__)

        # Resource state
        self.is_active = False
        self.resource_allocations: Dict[str, ResourceAllocation] = {}
        self.resource_metrics_history: List[ResourceMetrics] = []

        # System information
        self.system_info = self._get_system_info()
        self.available_resources = self._get_available_resources()

        # Monitoring
        self.monitoring_thread = None
        self.optimization_thread = None
        self.resource_lock = threading.RLock()

        # Callbacks
        self.resource_alert_callbacks: List[callable] = []
        self.allocation_callbacks: List[callable] = []
        self.optimization_callbacks: List[callable] = []

        # Metrics
        self.management_metrics = {
            "total_allocations": 0,
            "successful_allocations": 0,
            "failed_allocations": 0,
            "optimization_runs": 0,
            "resource_alerts": 0,
            "average_utilization": 0.0,
        }

        self.logger.info("Resource manager initialized")

    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        try:
            system_info = {
                "cpu_count": psutil.cpu_count(),
                "cpu_freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
                "memory_total": psutil.virtual_memory().total,
                "disk_usage": {
                    partition.mountpoint: psutil.disk_usage(
                        partition.mountpoint
                    )._asdict()
                    for partition in psutil.disk_partitions()
                },
                "network_interfaces": list(psutil.net_if_addrs().keys()),
                "gpu_count": 0,
                "gpu_info": [],
            }

            # Add GPU information if available
            if torch.cuda.is_available():
                system_info["gpu_count"] = torch.cuda.device_count()
                system_info["gpu_info"] = [
                    {
                        "device_id": i,
                        "name": torch.cuda.get_device_name(i),
                        "memory_total": torch.cuda.get_device_properties(
                            i
                        ).total_memory,
                        "compute_capability": torch.cuda.get_device_properties(i).major,
                    }
                    for i in range(torch.cuda.device_count())
                ]

            return system_info

        except Exception as e:
            self.logger.error("Failed to get system info: %s", str(e))
            return {}

    def _get_available_resources(self) -> Dict[str, float]:
        """Get available resources."""
        try:
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")

            available_resources = {
                "cpu_cores": self.system_info.get("cpu_count", 1),
                "memory_gb": memory.total / (1024**3),
                "storage_gb": disk.total / (1024**3),
                "gpu_count": self.system_info.get("gpu_count", 0),
            }

            # Add GPU memory
            if torch.cuda.is_available():
                total_gpu_memory = sum(
                    gpu["memory_total"] for gpu in self.system_info.get("gpu_info", [])
                )
                available_resources["gpu_memory_gb"] = total_gpu_memory / (1024**3)
            else:
                available_resources["gpu_memory_gb"] = 0.0

            return available_resources

        except Exception as e:
            self.logger.error("Failed to get available resources: %s", str(e))
            return {}

    async def start_resource_management(self) -> None:
        """Start resource management services."""
        if self.is_active:
            self.logger.warning("Resource management already active")
            return

        try:
            self.logger.info("Starting resource management services")

            # Start monitoring
            await self._start_resource_monitoring()

            # Start optimization
            if self.config.enable_resource_optimization:
                await self._start_resource_optimization()

            self.is_active = True
            self.logger.info("Resource management services started")

        except Exception as e:
            self.logger.error("Failed to start resource management: %s", str(e))
            raise

    async def _start_resource_monitoring(self) -> None:
        """Start resource monitoring thread."""

        def monitoring_worker():
            while self.is_active:
                try:
                    # Collect resource metrics
                    metrics = self._collect_resource_metrics()

                    # Store metrics
                    with self.resource_lock:
                        self.resource_metrics_history.append(metrics)

                        # Limit history size
                        if (
                            len(self.resource_metrics_history)
                            > self.config.resource_history_size
                        ):
                            self.resource_metrics_history.pop(0)

                    # Check for alerts
                    self._check_resource_alerts(metrics)

                    time.sleep(self.config.monitoring_interval)

                except Exception as e:
                    self.logger.error("Resource monitoring error: %s", str(e))

        self.monitoring_thread = threading.Thread(target=monitoring_worker, daemon=True)
        self.monitoring_thread.start()

        self.logger.debug("Resource monitoring started")

    async def _start_resource_optimization(self) -> None:
        """Start resource optimization thread."""

        def optimization_worker():
            last_optimization = 0

            while self.is_active:
                try:
                    current_time = time.time()

                    if (
                        current_time - last_optimization
                        >= self.config.optimization_interval
                    ):
                        # Run resource optimization
                        self._optimize_resource_allocation()
                        last_optimization = current_time

                        self.management_metrics["optimization_runs"] += 1

                    time.sleep(10.0)  # Check every 10 seconds

                except Exception as e:
                    self.logger.error("Resource optimization error: %s", str(e))

        self.optimization_thread = threading.Thread(
            target=optimization_worker, daemon=True
        )
        self.optimization_thread.start()

        self.logger.debug("Resource optimization started")

    def _collect_resource_metrics(self) -> ResourceMetrics:
        """Collect current resource metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)

            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent

            # Storage metrics
            disk = psutil.disk_usage("/")
            storage_percent = (disk.used / disk.total) * 100

            # Network metrics
            network_io = psutil.net_io_counters()._asdict()

            # Process count
            process_count = len(psutil.pids())

            # GPU metrics
            gpu_percent = 0.0
            if torch.cuda.is_available():
                try:
                    # Get GPU utilization (simplified)
                    gpu_memory_used = sum(
                        torch.cuda.memory_allocated(i)
                        for i in range(torch.cuda.device_count())
                    )
                    gpu_memory_total = sum(
                        torch.cuda.get_device_properties(i).total_memory
                        for i in range(torch.cuda.device_count())
                    )

                    if gpu_memory_total > 0:
                        gpu_percent = (gpu_memory_used / gpu_memory_total) * 100

                except Exception:
                    gpu_percent = 0.0

            return ResourceMetrics(
                timestamp=time.time(),
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                gpu_percent=gpu_percent,
                storage_percent=storage_percent,
                network_io=network_io,
                process_count=process_count,
            )

        except Exception as e:
            self.logger.error("Failed to collect resource metrics: %s", str(e))
            # Return default metrics
            return ResourceMetrics(
                timestamp=time.time(),
                cpu_percent=0.0,
                memory_percent=0.0,
                gpu_percent=0.0,
                storage_percent=0.0,
                network_io={},
                process_count=0,
            )

    def _check_resource_alerts(self, metrics: ResourceMetrics) -> None:
        """Check for resource alerts."""
        alerts = []

        # Check CPU
        if metrics.cpu_percent > self.config.alert_thresholds.get("cpu", 85.0):
            alerts.append(
                {
                    "type": "cpu",
                    "current": metrics.cpu_percent,
                    "threshold": self.config.alert_thresholds["cpu"],
                    "severity": "high" if metrics.cpu_percent > 95 else "medium",
                }
            )

        # Check memory
        if metrics.memory_percent > self.config.alert_thresholds.get("memory", 85.0):
            alerts.append(
                {
                    "type": "memory",
                    "current": metrics.memory_percent,
                    "threshold": self.config.alert_thresholds["memory"],
                    "severity": "high" if metrics.memory_percent > 95 else "medium",
                }
            )

        # Check GPU
        if metrics.gpu_percent > self.config.alert_thresholds.get("gpu", 95.0):
            alerts.append(
                {
                    "type": "gpu",
                    "current": metrics.gpu_percent,
                    "threshold": self.config.alert_thresholds["gpu"],
                    "severity": "high" if metrics.gpu_percent > 98 else "medium",
                }
            )

        # Check storage
        if metrics.storage_percent > self.config.alert_thresholds.get("storage", 90.0):
            alerts.append(
                {
                    "type": "storage",
                    "current": metrics.storage_percent,
                    "threshold": self.config.alert_thresholds["storage"],
                    "severity": "high" if metrics.storage_percent > 95 else "medium",
                }
            )

        # Process alerts
        for alert in alerts:
            self._handle_resource_alert(alert)

    def _handle_resource_alert(self, alert: Dict[str, Any]) -> None:
        """Handle resource alert."""
        self.logger.warning(
            "Resource alert: %s usage %.1f%% exceeds threshold %.1f%%",
            alert["type"],
            alert["current"],
            alert["threshold"],
        )

        self.management_metrics["resource_alerts"] += 1

        # Notify callbacks
        for callback in self.resource_alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error("Resource alert callback error: %s", str(e))

    def _optimize_resource_allocation(self) -> None:
        """Optimize resource allocation."""
        try:
            self.logger.debug("Running resource optimization")

            with self.resource_lock:
                # Get current metrics
                if not self.resource_metrics_history:
                    return

                current_metrics = self.resource_metrics_history[-1]

                # Check if optimization is needed
                if not self._should_optimize(current_metrics):
                    return

                # Perform optimization based on strategy
                if self.config.allocation_strategy == AllocationStrategy.LOAD_BALANCED:
                    self._optimize_load_balanced()
                elif (
                    self.config.allocation_strategy
                    == AllocationStrategy.PERFORMANCE_OPTIMIZED
                ):
                    self._optimize_performance_based()
                elif (
                    self.config.allocation_strategy == AllocationStrategy.PRIORITY_BASED
                ):
                    self._optimize_priority_based()
                else:
                    self._optimize_round_robin()

                # Notify callbacks
                for callback in self.optimization_callbacks:
                    try:
                        callback()
                    except Exception as e:
                        self.logger.error("Optimization callback error: %s", str(e))

        except Exception as e:
            self.logger.error("Resource optimization failed: %s", str(e))

    def _should_optimize(self, metrics: ResourceMetrics) -> bool:
        """Check if resource optimization is needed."""
        # Check if any resource is above rebalancing threshold
        thresholds = {
            "cpu": self.config.rebalancing_threshold * 100,
            "memory": self.config.rebalancing_threshold * 100,
            "gpu": self.config.rebalancing_threshold * 100,
            "storage": self.config.rebalancing_threshold * 100,
        }

        return (
            metrics.cpu_percent > thresholds["cpu"]
            or metrics.memory_percent > thresholds["memory"]
            or metrics.gpu_percent > thresholds["gpu"]
            or metrics.storage_percent > thresholds["storage"]
        )

    def _optimize_load_balanced(self) -> None:
        """Optimize allocation for load balancing."""
        # Redistribute resources to balance load
        self.logger.debug("Performing load-balanced optimization")

        # This would typically involve:
        # 1. Analyzing current load distribution
        # 2. Identifying overloaded resources
        # 3. Redistributing allocations
        # 4. Updating allocation records

    def _optimize_performance_based(self) -> None:
        """Optimize allocation for performance."""
        self.logger.debug("Performing performance-based optimization")

        # This would typically involve:
        # 1. Analyzing performance metrics
        # 2. Identifying performance bottlenecks
        # 3. Reallocating resources to optimize performance

    def _optimize_priority_based(self) -> None:
        """Optimize allocation based on priorities."""
        self.logger.debug("Performing priority-based optimization")

        # Sort allocations by priority and reallocate
        sorted_allocations = sorted(
            self.resource_allocations.values(), key=lambda x: x.priority, reverse=True
        )

        # Reallocate based on priority
        for allocation in sorted_allocations:
            # Ensure high-priority allocations get resources first
            pass

    def _optimize_round_robin(self) -> None:
        """Optimize allocation using round-robin strategy."""
        self.logger.debug("Performing round-robin optimization")

        # Redistribute resources in round-robin fashion
        # This is typically used for fair resource sharing

    async def allocate_resource(
        self, resource_type: ResourceType, amount: float, owner: str, priority: int = 1
    ) -> Optional[str]:
        """Allocate resource to an owner.

        Args:
            resource_type: Type of resource to allocate
            amount: Amount of resource to allocate
            owner: Owner of the allocation
            priority: Priority of the allocation

        Returns:
            Allocation ID if successful, None otherwise
        """
        try:
            allocation_id = f"{resource_type.value}_{owner}_{int(time.time())}"

            # Check if resource is available
            if not self._is_resource_available(resource_type, amount):
                self.logger.warning(
                    "Resource allocation failed: insufficient %s (requested: %.2f)",
                    resource_type.value,
                    amount,
                )
                self.management_metrics["failed_allocations"] += 1
                return None

            # Get total resource amount
            total_amount = self._get_total_resource_amount(resource_type)

            # Create allocation
            allocation = ResourceAllocation(
                resource_id=allocation_id,
                resource_type=resource_type,
                allocated_amount=amount,
                total_amount=total_amount,
                allocation_time=time.time(),
                owner=owner,
                priority=priority,
            )

            with self.resource_lock:
                self.resource_allocations[allocation_id] = allocation

            self.management_metrics["total_allocations"] += 1
            self.management_metrics["successful_allocations"] += 1

            # Notify callbacks
            for callback in self.allocation_callbacks:
                try:
                    callback(allocation)
                except Exception as e:
                    self.logger.error("Allocation callback error: %s", str(e))

            self.logger.info(
                "Resource allocated: %s (%.2f %s to %s)",
                allocation_id,
                amount,
                resource_type.value,
                owner,
            )

            return allocation_id

        except Exception as e:
            self.logger.error("Resource allocation failed: %s", str(e))
            self.management_metrics["failed_allocations"] += 1
            return None

    def _is_resource_available(
        self, resource_type: ResourceType, amount: float
    ) -> bool:
        """Check if resource is available for allocation."""
        try:
            total_amount = self._get_total_resource_amount(resource_type)
            allocated_amount = self._get_allocated_resource_amount(resource_type)
            available_amount = total_amount - allocated_amount

            # Check against limits
            max_percent = self._get_max_resource_percent(resource_type)
            max_amount = total_amount * (max_percent / 100.0)

            return (
                allocated_amount + amount
            ) <= max_amount and amount <= available_amount

        except Exception as e:
            self.logger.error("Failed to check resource availability: %s", str(e))
            return False

    def _get_total_resource_amount(self, resource_type: ResourceType) -> float:
        """Get total amount of resource."""
        if resource_type == ResourceType.CPU:
            return float(self.available_resources.get("cpu_cores", 1))
        elif resource_type == ResourceType.MEMORY:
            return self.available_resources.get("memory_gb", 1.0)
        elif resource_type == ResourceType.GPU:
            return self.available_resources.get("gpu_memory_gb", 0.0)
        elif resource_type == ResourceType.STORAGE:
            return self.available_resources.get("storage_gb", 1.0)
        else:
            return 1.0

    def _get_allocated_resource_amount(self, resource_type: ResourceType) -> float:
        """Get currently allocated amount of resource."""
        with self.resource_lock:
            return sum(
                allocation.allocated_amount
                for allocation in self.resource_allocations.values()
                if allocation.resource_type == resource_type
            )

    def _get_max_resource_percent(self, resource_type: ResourceType) -> float:
        """Get maximum resource percentage."""
        if resource_type == ResourceType.CPU:
            return self.config.max_cpu_percent
        elif resource_type == ResourceType.MEMORY:
            return self.config.max_memory_percent
        elif resource_type == ResourceType.GPU:
            return self.config.max_gpu_memory_percent
        elif resource_type == ResourceType.STORAGE:
            return self.config.max_storage_percent
        else:
            return 80.0

    async def deallocate_resource(self, allocation_id: str) -> bool:
        """Deallocate resource.

        Args:
            allocation_id: ID of allocation to deallocate

        Returns:
            True if successful, False otherwise
        """
        try:
            with self.resource_lock:
                if allocation_id not in self.resource_allocations:
                    self.logger.warning("Allocation not found: %s", allocation_id)
                    return False

                allocation = self.resource_allocations[allocation_id]
                del self.resource_allocations[allocation_id]

            self.logger.info(
                "Resource deallocated: %s (%.2f %s from %s)",
                allocation_id,
                allocation.allocated_amount,
                allocation.resource_type.value,
                allocation.owner,
            )

            return True

        except Exception as e:
            self.logger.error("Resource deallocation failed: %s", str(e))
            return False

    def get_resource_usage(self) -> Dict[str, Any]:
        """Get current resource usage information."""
        try:
            with self.resource_lock:
                usage_by_type = {}

                for resource_type in ResourceType:
                    total_amount = self._get_total_resource_amount(resource_type)
                    allocated_amount = self._get_allocated_resource_amount(
                        resource_type
                    )

                    usage_by_type[resource_type.value] = {
                        "total": total_amount,
                        "allocated": allocated_amount,
                        "available": total_amount - allocated_amount,
                        "utilization_percent": (allocated_amount / total_amount) * 100
                        if total_amount > 0
                        else 0,
                    }

                return {
                    "usage_by_type": usage_by_type,
                    "total_allocations": len(self.resource_allocations),
                    "allocation_details": [
                        {
                            "id": alloc.resource_id,
                            "type": alloc.resource_type.value,
                            "amount": alloc.allocated_amount,
                            "owner": alloc.owner,
                            "priority": alloc.priority,
                            "utilization_percent": alloc.utilization_percent,
                        }
                        for alloc in self.resource_allocations.values()
                    ],
                }

        except Exception as e:
            self.logger.error("Failed to get resource usage: %s", str(e))
            return {}

    def get_resource_metrics(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get resource metrics history.

        Args:
            limit: Maximum number of metrics to return

        Returns:
            List of resource metrics
        """
        with self.resource_lock:
            metrics = self.resource_metrics_history.copy()

            if limit:
                metrics = metrics[-limit:]

            return [metric.to_dict() for metric in metrics]

    def get_management_metrics(self) -> Dict[str, Any]:
        """Get resource management metrics."""
        # Calculate average utilization
        if self.resource_metrics_history:
            recent_metrics = self.resource_metrics_history[-10:]  # Last 10 measurements
            avg_cpu = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
            avg_memory = sum(m.memory_percent for m in recent_metrics) / len(
                recent_metrics
            )
            avg_gpu = sum(m.gpu_percent for m in recent_metrics) / len(recent_metrics)

            self.management_metrics["average_utilization"] = (
                avg_cpu + avg_memory + avg_gpu
            ) / 3

        return self.management_metrics.copy()

    def add_resource_alert_callback(self, callback: callable) -> None:
        """Add resource alert callback."""
        self.resource_alert_callbacks.append(callback)

    def add_allocation_callback(self, callback: callable) -> None:
        """Add allocation callback."""
        self.allocation_callbacks.append(callback)

    def add_optimization_callback(self, callback: callable) -> None:
        """Add optimization callback."""
        self.optimization_callbacks.append(callback)

    async def shutdown(self) -> None:
        """Shutdown resource manager."""
        self.logger.info("Shutting down resource manager")

        self.is_active = False

        # Stop threads
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5.0)

        if self.optimization_thread and self.optimization_thread.is_alive():
            self.optimization_thread.join(timeout=5.0)

        # Clear allocations
        with self.resource_lock:
            self.resource_allocations.clear()
            self.resource_metrics_history.clear()

        # Clear callbacks
        self.resource_alert_callbacks.clear()
        self.allocation_callbacks.clear()
        self.optimization_callbacks.clear()

        self.logger.info("Resource manager shutdown complete")
