"""
Memory Management and Overflow Prevention.

This module provides memory management capabilities for preventing
memory overflow and managing resource usage in the MARL system.
"""

import asyncio
import gc
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import psutil

from utils.logging_config import get_logger


class MemoryStatus(Enum):
    """Memory status enumeration."""

    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    OVERFLOW = "overflow"


class MemoryThreshold:
    """Memory threshold configuration."""

    def __init__(
        self,
        warning_threshold: float = 0.7,
        critical_threshold: float = 0.85,
        overflow_threshold: float = 0.95,
    ):
        """
        Initialize memory thresholds.

        Args:
            warning_threshold: Warning threshold (0.0-1.0)
            critical_threshold: Critical threshold (0.0-1.0)
            overflow_threshold: Overflow threshold (0.0-1.0)
        """
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.overflow_threshold = overflow_threshold

    def get_status(self, usage_ratio: float) -> MemoryStatus:
        """Get memory status based on usage ratio."""
        if usage_ratio >= self.overflow_threshold:
            return MemoryStatus.OVERFLOW
        elif usage_ratio >= self.critical_threshold:
            return MemoryStatus.CRITICAL
        elif usage_ratio >= self.warning_threshold:
            return MemoryStatus.WARNING
        else:
            return MemoryStatus.NORMAL


@dataclass
class MemoryMetrics:
    """Memory usage metrics."""

    timestamp: datetime = field(default_factory=datetime.now)
    total_memory: int = 0  # Total system memory in bytes
    available_memory: int = 0  # Available memory in bytes
    used_memory: int = 0  # Used memory in bytes
    memory_percent: float = 0.0  # Memory usage percentage

    # Process-specific metrics
    process_memory: int = 0  # Current process memory usage
    process_percent: float = 0.0  # Process memory percentage

    # Component-specific metrics
    component_memory: Dict[str, int] = field(default_factory=dict)

    # Buffer sizes
    buffer_sizes: Dict[str, int] = field(default_factory=dict)
    total_buffer_memory: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "total_memory": self.total_memory,
            "available_memory": self.available_memory,
            "used_memory": self.used_memory,
            "memory_percent": self.memory_percent,
            "process_memory": self.process_memory,
            "process_percent": self.process_percent,
            "component_memory": self.component_memory,
            "buffer_sizes": self.buffer_sizes,
            "total_buffer_memory": self.total_buffer_memory,
        }


@dataclass
class MemoryEvent:
    """Memory-related event."""

    event_id: str
    event_type: str  # "threshold_exceeded", "overflow_prevented", "cleanup_performed"
    memory_status: MemoryStatus
    memory_usage: float
    timestamp: datetime = field(default_factory=datetime.now)

    # Event details
    affected_components: List[str] = field(default_factory=list)
    action_taken: Optional[str] = None
    memory_freed: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "memory_status": self.memory_status.value,
            "memory_usage": self.memory_usage,
            "timestamp": self.timestamp.isoformat(),
            "affected_components": self.affected_components,
            "action_taken": self.action_taken,
            "memory_freed": self.memory_freed,
            "metadata": self.metadata,
        }


class MemoryManager:
    """
    Manages memory usage and prevents overflow.

    Monitors system and process memory usage, manages buffer sizes,
    and performs cleanup operations to prevent memory overflow.
    """

    def __init__(
        self,
        monitoring_interval: float = 5.0,
        thresholds: Optional[MemoryThreshold] = None,
        enable_auto_cleanup: bool = True,
        max_buffer_memory: int = 1024 * 1024 * 1024,  # 1GB
        gc_threshold_ratio: float = 0.8,
    ):
        """
        Initialize memory manager.

        Args:
            monitoring_interval: Interval between memory checks (seconds)
            thresholds: Memory threshold configuration
            enable_auto_cleanup: Enable automatic cleanup
            max_buffer_memory: Maximum memory for buffers (bytes)
            gc_threshold_ratio: Ratio to trigger garbage collection
        """
        self.logger = get_logger(__name__)

        # Configuration
        self.monitoring_interval = monitoring_interval
        self.thresholds = thresholds or MemoryThreshold()
        self.enable_auto_cleanup = enable_auto_cleanup
        self.max_buffer_memory = max_buffer_memory
        self.gc_threshold_ratio = gc_threshold_ratio

        # State tracking
        self.current_metrics = MemoryMetrics()
        self.metrics_history: List[MemoryMetrics] = []
        self.max_history_size = 1000

        # Component tracking
        self.managed_components: Dict[str, Any] = {}
        self.buffer_managers: Dict[str, Callable] = {}

        # Event tracking
        self.memory_events: List[MemoryEvent] = []
        self.event_counter = 0

        # Monitoring state
        self.is_monitoring = False
        self.monitoring_task: Optional[asyncio.Task] = None
        self.monitoring_lock = threading.Lock()

        # Callbacks
        self.threshold_callbacks: List[Callable] = []
        self.cleanup_callbacks: List[Callable] = []
        self.overflow_callbacks: List[Callable] = []

        # Cleanup strategies
        self.cleanup_strategies = {
            MemoryStatus.WARNING: self._warning_cleanup,
            MemoryStatus.CRITICAL: self._critical_cleanup,
            MemoryStatus.OVERFLOW: self._overflow_cleanup,
        }

        self.logger.info("Memory manager initialized")

    async def start_monitoring(self) -> None:
        """Start memory monitoring."""
        if self.is_monitoring:
            self.logger.warning("Memory monitoring already running")
            return

        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())

        self.logger.info("Memory monitoring started")

    async def stop_monitoring(self) -> None:
        """Stop memory monitoring."""
        if not self.is_monitoring:
            return

        self.is_monitoring = False

        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass

        self.logger.info("Memory monitoring stopped")

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                await self._check_memory_usage()
                await asyncio.sleep(self.monitoring_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(
                    "Error in memory monitoring loop: %s", str(e), exc_info=True
                )
                await asyncio.sleep(5.0)  # Wait before retrying

    async def _check_memory_usage(self) -> None:
        """Check current memory usage."""
        with self.monitoring_lock:
            # Get system memory info
            memory_info = psutil.virtual_memory()
            process = psutil.Process()
            process_memory_info = process.memory_info()

            # Update current metrics
            self.current_metrics = MemoryMetrics(
                total_memory=memory_info.total,
                available_memory=memory_info.available,
                used_memory=memory_info.used,
                memory_percent=memory_info.percent,
                process_memory=process_memory_info.rss,
                process_percent=process.memory_percent(),
            )

            # Update component memory usage
            self._update_component_memory()

            # Update buffer memory usage
            self._update_buffer_memory()

            # Add to history
            self.metrics_history.append(self.current_metrics)
            if len(self.metrics_history) > self.max_history_size:
                self.metrics_history.pop(0)

        # Check thresholds and take action
        await self._check_thresholds()

    def _update_component_memory(self) -> None:
        """Update memory usage for managed components."""
        for component_name, component in self.managed_components.items():
            try:
                if hasattr(component, "get_memory_usage"):
                    memory_usage = component.get_memory_usage()
                    self.current_metrics.component_memory[component_name] = memory_usage
                else:
                    # Estimate memory usage (this is a rough approximation)
                    import sys

                    memory_usage = sys.getsizeof(component)
                    self.current_metrics.component_memory[component_name] = memory_usage

            except Exception as e:
                self.logger.warning(
                    "Failed to get memory usage for component %s: %s",
                    component_name,
                    str(e),
                )

    def _update_buffer_memory(self) -> None:
        """Update buffer memory usage."""
        total_buffer_memory = 0

        for buffer_name, buffer_manager in self.buffer_managers.items():
            try:
                if callable(buffer_manager):
                    buffer_size = buffer_manager()
                    self.current_metrics.buffer_sizes[buffer_name] = buffer_size
                    total_buffer_memory += buffer_size

            except Exception as e:
                self.logger.warning(
                    "Failed to get buffer size for %s: %s", buffer_name, str(e)
                )

        self.current_metrics.total_buffer_memory = total_buffer_memory

    async def _check_thresholds(self) -> None:
        """Check memory thresholds and take appropriate action."""
        memory_usage = self.current_metrics.memory_percent / 100.0
        memory_status = self.thresholds.get_status(memory_usage)

        if memory_status != MemoryStatus.NORMAL:
            await self._handle_memory_threshold(memory_status, memory_usage)

    async def _handle_memory_threshold(
        self, memory_status: MemoryStatus, memory_usage: float
    ) -> None:
        """Handle memory threshold exceeded."""
        event_id = f"memory_event_{self.event_counter}"
        self.event_counter += 1

        self.logger.warning(
            "Memory threshold exceeded: %s (usage: %.1f%%)",
            memory_status.value,
            memory_usage * 100,
        )

        # Create memory event
        event = MemoryEvent(
            event_id=event_id,
            event_type="threshold_exceeded",
            memory_status=memory_status,
            memory_usage=memory_usage,
        )

        # Notify threshold callbacks
        await self._notify_threshold_callbacks(event)

        # Perform cleanup if auto-cleanup is enabled
        if self.enable_auto_cleanup:
            cleanup_result = await self._perform_cleanup(memory_status)

            if cleanup_result:
                event.action_taken = cleanup_result["strategy"]
                event.memory_freed = cleanup_result["memory_freed"]
                event.affected_components = cleanup_result["affected_components"]

        # Record event
        self.memory_events.append(event)

        # Keep only recent events
        if len(self.memory_events) > 1000:
            self.memory_events = self.memory_events[-1000:]

    async def _perform_cleanup(
        self, memory_status: MemoryStatus
    ) -> Optional[Dict[str, Any]]:
        """Perform memory cleanup based on status."""
        try:
            cleanup_func = self.cleanup_strategies.get(memory_status)

            if cleanup_func:
                result = await cleanup_func()

                self.logger.info(
                    "Memory cleanup completed: %s (freed: %d bytes)",
                    result.get("strategy", "unknown"),
                    result.get("memory_freed", 0),
                )

                return result
            else:
                self.logger.warning(
                    "No cleanup strategy for memory status: %s", memory_status.value
                )

        except Exception as e:
            self.logger.error(
                "Error performing memory cleanup: %s", str(e), exc_info=True
            )

        return None

    async def _warning_cleanup(self) -> Dict[str, Any]:
        """Perform warning-level cleanup."""
        memory_freed = 0
        affected_components = []

        # Trigger garbage collection
        collected = gc.collect()
        self.logger.debug("Garbage collection freed %d objects", collected)

        # Reduce buffer sizes slightly
        for buffer_name, buffer_manager in self.buffer_managers.items():
            try:
                if hasattr(buffer_manager, "__self__") and hasattr(
                    buffer_manager.__self__, "reduce_size"
                ):
                    freed = buffer_manager.__self__.reduce_size(0.1)  # Reduce by 10%
                    memory_freed += freed
                    affected_components.append(buffer_name)

            except Exception as e:
                self.logger.warning(
                    "Failed to reduce buffer size for %s: %s", buffer_name, str(e)
                )

        return {
            "strategy": "warning_cleanup",
            "memory_freed": memory_freed,
            "affected_components": affected_components,
        }

    async def _critical_cleanup(self) -> Dict[str, Any]:
        """Perform critical-level cleanup."""
        memory_freed = 0
        affected_components = []

        # Aggressive garbage collection
        for _ in range(3):
            collected = gc.collect()
            self.logger.debug("Aggressive GC freed %d objects", collected)

        # Significantly reduce buffer sizes
        for buffer_name, buffer_manager in self.buffer_managers.items():
            try:
                if hasattr(buffer_manager, "__self__") and hasattr(
                    buffer_manager.__self__, "reduce_size"
                ):
                    freed = buffer_manager.__self__.reduce_size(0.3)  # Reduce by 30%
                    memory_freed += freed
                    affected_components.append(buffer_name)

            except Exception as e:
                self.logger.warning(
                    "Failed to reduce buffer size for %s: %s", buffer_name, str(e)
                )

        # Clear component caches
        for component_name, component in self.managed_components.items():
            try:
                if hasattr(component, "clear_cache"):
                    freed = component.clear_cache()
                    memory_freed += freed
                    affected_components.append(component_name)

            except Exception as e:
                self.logger.warning(
                    "Failed to clear cache for component %s: %s", component_name, str(e)
                )

        return {
            "strategy": "critical_cleanup",
            "memory_freed": memory_freed,
            "affected_components": affected_components,
        }

    async def _overflow_cleanup(self) -> Dict[str, Any]:
        """Perform overflow-level cleanup."""
        memory_freed = 0
        affected_components = []

        # Emergency cleanup
        self.logger.critical("Memory overflow detected - performing emergency cleanup")

        # Force garbage collection multiple times
        for _ in range(5):
            collected = gc.collect()
            self.logger.debug("Emergency GC freed %d objects", collected)

        # Drastically reduce or clear buffers
        for buffer_name, buffer_manager in self.buffer_managers.items():
            try:
                if hasattr(buffer_manager, "__self__"):
                    buffer_obj = buffer_manager.__self__

                    if hasattr(buffer_obj, "clear"):
                        freed = buffer_obj.clear()
                        memory_freed += freed
                        affected_components.append(buffer_name)
                        self.logger.warning("Cleared buffer: %s", buffer_name)
                    elif hasattr(buffer_obj, "reduce_size"):
                        freed = buffer_obj.reduce_size(0.8)  # Reduce by 80%
                        memory_freed += freed
                        affected_components.append(buffer_name)

            except Exception as e:
                self.logger.error("Failed to clear buffer %s: %s", buffer_name, str(e))

        # Clear all component caches and temporary data
        for component_name, component in self.managed_components.items():
            try:
                if hasattr(component, "emergency_cleanup"):
                    freed = component.emergency_cleanup()
                    memory_freed += freed
                    affected_components.append(component_name)
                elif hasattr(component, "clear_cache"):
                    freed = component.clear_cache()
                    memory_freed += freed
                    affected_components.append(component_name)

            except Exception as e:
                self.logger.error(
                    "Failed to perform emergency cleanup for component %s: %s",
                    component_name,
                    str(e),
                )

        # Notify overflow callbacks
        for callback in self.overflow_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(self.current_metrics)
                else:
                    callback(self.current_metrics)
            except Exception as e:
                self.logger.error("Error in overflow callback: %s", str(e))

        return {
            "strategy": "overflow_cleanup",
            "memory_freed": memory_freed,
            "affected_components": affected_components,
        }

    async def _notify_threshold_callbacks(self, event: MemoryEvent) -> None:
        """Notify threshold callbacks."""
        for callback in self.threshold_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
            except Exception as e:
                self.logger.error("Error in threshold callback: %s", str(e))

    # Public interface methods

    def register_component(self, name: str, component: Any) -> None:
        """Register a component for memory monitoring."""
        self.managed_components[name] = component
        self.logger.debug("Registered component for memory monitoring: %s", name)

    def unregister_component(self, name: str) -> None:
        """Unregister a component from memory monitoring."""
        if name in self.managed_components:
            del self.managed_components[name]
            self.logger.debug("Unregistered component from memory monitoring: %s", name)

    def register_buffer_manager(self, name: str, size_getter: Callable) -> None:
        """Register a buffer manager for memory monitoring."""
        self.buffer_managers[name] = size_getter
        self.logger.debug("Registered buffer manager: %s", name)

    def unregister_buffer_manager(self, name: str) -> None:
        """Unregister a buffer manager."""
        if name in self.buffer_managers:
            del self.buffer_managers[name]
            self.logger.debug("Unregistered buffer manager: %s", name)

    def get_current_metrics(self) -> MemoryMetrics:
        """Get current memory metrics."""
        return self.current_metrics

    def get_memory_status(self) -> MemoryStatus:
        """Get current memory status."""
        memory_usage = self.current_metrics.memory_percent / 100.0
        return self.thresholds.get_status(memory_usage)

    def get_memory_history(self, hours: int = 1) -> List[MemoryMetrics]:
        """Get memory metrics history."""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        return [
            metrics
            for metrics in self.metrics_history
            if metrics.timestamp >= cutoff_time
        ]

    def get_memory_events(self, hours: int = 24) -> List[MemoryEvent]:
        """Get memory events."""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        return [event for event in self.memory_events if event.timestamp >= cutoff_time]

    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        if not self.metrics_history:
            return {
                "current_usage": 0.0,
                "average_usage": 0.0,
                "peak_usage": 0.0,
                "memory_events": 0,
                "cleanup_events": 0,
            }

        current_usage = self.current_metrics.memory_percent
        usage_values = [m.memory_percent for m in self.metrics_history]

        # Count events
        total_events = len(self.memory_events)
        cleanup_events = len([e for e in self.memory_events if e.action_taken])

        return {
            "current_usage": current_usage,
            "average_usage": np.mean(usage_values),
            "peak_usage": max(usage_values),
            "memory_events": total_events,
            "cleanup_events": cleanup_events,
            "buffer_memory_usage": self.current_metrics.total_buffer_memory,
            "component_count": len(self.managed_components),
            "buffer_count": len(self.buffer_managers),
        }

    async def force_cleanup(self, level: str = "warning") -> Dict[str, Any]:
        """Force memory cleanup."""
        status_map = {
            "warning": MemoryStatus.WARNING,
            "critical": MemoryStatus.CRITICAL,
            "overflow": MemoryStatus.OVERFLOW,
        }

        memory_status = status_map.get(level, MemoryStatus.WARNING)

        self.logger.info("Forcing memory cleanup at level: %s", level)

        result = await self._perform_cleanup(memory_status)

        if result:
            # Create event
            event = MemoryEvent(
                event_id=f"forced_cleanup_{self.event_counter}",
                event_type="forced_cleanup",
                memory_status=memory_status,
                memory_usage=self.current_metrics.memory_percent / 100.0,
                action_taken=result["strategy"],
                memory_freed=result["memory_freed"],
                affected_components=result["affected_components"],
            )

            self.event_counter += 1
            self.memory_events.append(event)

            return result

        return {"strategy": "none", "memory_freed": 0, "affected_components": []}

    def add_threshold_callback(self, callback: Callable) -> None:
        """Add threshold callback."""
        self.threshold_callbacks.append(callback)

    def add_cleanup_callback(self, callback: Callable) -> None:
        """Add cleanup callback."""
        self.cleanup_callbacks.append(callback)

    def add_overflow_callback(self, callback: Callable) -> None:
        """Add overflow callback."""
        self.overflow_callbacks.append(callback)

    def remove_threshold_callback(self, callback: Callable) -> None:
        """Remove threshold callback."""
        if callback in self.threshold_callbacks:
            self.threshold_callbacks.remove(callback)

    def remove_cleanup_callback(self, callback: Callable) -> None:
        """Remove cleanup callback."""
        if callback in self.cleanup_callbacks:
            self.cleanup_callbacks.remove(callback)

    def remove_overflow_callback(self, callback: Callable) -> None:
        """Remove overflow callback."""
        if callback in self.overflow_callbacks:
            self.overflow_callbacks.remove(callback)

    async def shutdown(self) -> None:
        """Shutdown memory manager."""
        await self.stop_monitoring()

        # Clear all data
        self.managed_components.clear()
        self.buffer_managers.clear()
        self.memory_events.clear()
        self.metrics_history.clear()
        self.threshold_callbacks.clear()
        self.cleanup_callbacks.clear()
        self.overflow_callbacks.clear()

        self.logger.info("Memory manager shutdown complete")
