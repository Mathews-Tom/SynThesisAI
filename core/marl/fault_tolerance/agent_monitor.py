"""Agent Monitoring and Failure Detection.

This module provides monitoring capabilities for detecting agent failures
and health issues in the multi-agent reinforcement learning system.
"""

# Standard Library
import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

# SynThesisAI Modules
from utils.logging_config import get_logger


class AgentHealthStatus(Enum):
    """Agent health status enumeration."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    FAILED = "failed"
    UNKNOWN = "unknown"


@dataclass
class AgentHealthMetrics:
    """Agent health metrics.

    Attributes:
        agent_id: The unique identifier for the agent.
        status: The current health status of the agent.
        last_heartbeat: The timestamp of the last received heartbeat.
        response_time: The average response time of the agent.
        error_rate: The rate of errors encountered by the agent.
        memory_usage: The memory usage of the agent.
        cpu_usage: The CPU usage of the agent.
        action_success_rate: The success rate of the agent's actions.
        learning_progress: The learning progress of the agent.
        total_actions: The total number of actions performed by the agent.
        successful_actions: The number of successful actions.
        failed_actions: The number of failed actions.
        consecutive_failures: The number of consecutive failures.
        last_action_time: The timestamp of the last action.
        last_error_time: The timestamp of the last error.
        status_change_time: The timestamp of the last status change.
    """

    agent_id: str
    status: AgentHealthStatus = AgentHealthStatus.UNKNOWN
    last_heartbeat: Optional[datetime] = None
    response_time: float = 0.0
    error_rate: float = 0.0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    action_success_rate: float = 0.0
    learning_progress: float = 0.0

    # Counters
    total_actions: int = 0
    successful_actions: int = 0
    failed_actions: int = 0
    consecutive_failures: int = 0

    # Timestamps
    last_action_time: Optional[datetime] = None
    last_error_time: Optional[datetime] = None
    status_change_time: Optional[datetime] = field(default_factory=datetime.now)

    def update_action_result(self, success: bool, response_time: float = 0.0) -> None:
        """Update action result metrics.

        Args:
            success: Whether the action was successful.
            response_time: The response time for the action.
        """
        self.total_actions += 1
        self.last_action_time = datetime.now()
        self.response_time = response_time

        if success:
            self.successful_actions += 1
            self.consecutive_failures = 0
        else:
            self.failed_actions += 1
            self.consecutive_failures += 1
            self.last_error_time = datetime.now()

        # Update success rate
        if self.total_actions > 0:
            self.action_success_rate = self.successful_actions / self.total_actions

        # Update error rate (recent errors)
        if self.total_actions > 0:
            self.error_rate = self.failed_actions / self.total_actions

    def update_resource_usage(self, memory_usage: float, cpu_usage: float) -> None:
        """Update resource usage metrics.

        Args:
            memory_usage: The current memory usage.
            cpu_usage: The current CPU usage.
        """
        self.memory_usage = memory_usage
        self.cpu_usage = cpu_usage

    def update_learning_progress(self, progress: float) -> None:
        """Update learning progress.

        Args:
            progress: The learning progress value.
        """
        self.learning_progress = progress

    def record_heartbeat(self) -> None:
        """Record agent heartbeat."""
        self.last_heartbeat = datetime.now()

    def get_health_score(self) -> float:
        """Calculate overall health score (0.0 to 1.0).

        Returns:
            The calculated health score.
        """
        score = 0.0
        factors = 0

        # Action success rate (40% weight)
        if self.total_actions > 0:
            score += 0.4 * self.action_success_rate
            factors += 0.4

        # Response time (20% weight) - lower is better
        if self.response_time > 0:
            response_score = max(0, 1.0 - (self.response_time / 10.0))  # 10s max
            score += 0.2 * response_score
            factors += 0.2

        # Resource usage (20% weight) - lower is better
        if self.memory_usage > 0 or self.cpu_usage > 0:
            resource_score = max(0, 1.0 - max(self.memory_usage, self.cpu_usage))
            score += 0.2 * resource_score
            factors += 0.2

        # Learning progress (20% weight)
        if self.learning_progress > 0:
            score += 0.2 * min(1.0, self.learning_progress)
            factors += 0.2

        # Normalize by actual factors
        if factors > 0:
            return score / factors
        return 0.5  # Default neutral score

    def is_responsive(self, timeout_seconds: float = 30.0) -> bool:
        """Check if agent is responsive based on heartbeat.

        Args:
            timeout_seconds: The timeout in seconds.

        Returns:
            True if the agent is responsive, False otherwise.
        """
        if not self.last_heartbeat:
            return False

        time_since_heartbeat = datetime.now() - self.last_heartbeat
        return time_since_heartbeat.total_seconds() <= timeout_seconds

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary.

        Returns:
            A dictionary representation of the metrics.
        """
        return {
            "agent_id": self.agent_id,
            "status": self.status.value,
            "last_heartbeat": (self.last_heartbeat.isoformat() if self.last_heartbeat else None),
            "response_time": self.response_time,
            "error_rate": self.error_rate,
            "memory_usage": self.memory_usage,
            "cpu_usage": self.cpu_usage,
            "action_success_rate": self.action_success_rate,
            "learning_progress": self.learning_progress,
            "total_actions": self.total_actions,
            "successful_actions": self.successful_actions,
            "failed_actions": self.failed_actions,
            "consecutive_failures": self.consecutive_failures,
            "health_score": self.get_health_score(),
            "is_responsive": self.is_responsive(),
        }


class AgentMonitor:
    """Monitors agent health and detects failures.

    Provides comprehensive monitoring of agent status, performance,
    and health metrics with configurable thresholds and alerts.
    """

    def __init__(
        self,
        heartbeat_interval: float = 10.0,
        response_timeout: float = 30.0,
        failure_threshold: int = 5,
        degraded_threshold: float = 0.7,
        unhealthy_threshold: float = 0.4,
        enable_auto_recovery: bool = True,
    ):
        """Initialize agent monitor.

        Args:
            heartbeat_interval: Interval between heartbeat checks (seconds).
            response_timeout: Timeout for agent responsiveness (seconds).
            failure_threshold: Consecutive failures before marking as failed.
            degraded_threshold: Health score threshold for degraded status.
            unhealthy_threshold: Health score threshold for unhealthy status.
            enable_auto_recovery: Enable automatic recovery attempts.
        """
        self.logger = get_logger(__name__)

        # Configuration
        self.heartbeat_interval = heartbeat_interval
        self.response_timeout = response_timeout
        self.failure_threshold = failure_threshold
        self.degraded_threshold = degraded_threshold
        self.unhealthy_threshold = unhealthy_threshold
        self.enable_auto_recovery = enable_auto_recovery

        # Agent tracking
        self.agent_metrics: Dict[str, AgentHealthMetrics] = {}
        self.monitored_agents: List[str] = []

        # Monitoring state
        self.is_monitoring = False
        self.monitoring_task: Optional[asyncio.Task] = None

        # Callbacks
        self.failure_callbacks: List[Callable] = []
        self.recovery_callbacks: List[Callable] = []
        self.status_change_callbacks: List[Callable] = []

        self.logger.info("Agent monitor initialized")

    def register_agent(self, agent_id: str) -> None:
        """Register an agent for monitoring.

        Args:
            agent_id: The ID of the agent to register.
        """
        if agent_id not in self.agent_metrics:
            self.agent_metrics[agent_id] = AgentHealthMetrics(agent_id=agent_id)
            self.monitored_agents.append(agent_id)
            self.logger.info("Registered agent for monitoring: %s", agent_id)

    def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent from monitoring.

        Args:
            agent_id: The ID of the agent to unregister.
        """
        if agent_id in self.agent_metrics:
            del self.agent_metrics[agent_id]
        if agent_id in self.monitored_agents:
            self.monitored_agents.remove(agent_id)
        self.logger.info("Unregistered agent from monitoring: %s", agent_id)

    async def start_monitoring(self) -> None:
        """Start agent monitoring."""
        if self.is_monitoring:
            self.logger.warning("Agent monitoring already running")
            return

        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.logger.info("Agent monitoring started")

    async def stop_monitoring(self) -> None:
        """Stop agent monitoring."""
        if not self.is_monitoring:
            return

        self.is_monitoring = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Agent monitoring stopped")

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                await self._check_all_agents()
                await asyncio.sleep(self.heartbeat_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Error in monitoring loop: %s", e, exc_info=True)
                await asyncio.sleep(5.0)  # Wait before retrying

    async def _check_all_agents(self) -> None:
        """Check health of all monitored agents."""
        for agent_id in self.monitored_agents:
            try:
                await self._check_agent_health(agent_id)
            except Exception as e:
                self.logger.error("Error checking agent %s: %s", agent_id, e)

    async def _check_agent_health(self, agent_id: str) -> None:
        """Check health of a specific agent."""
        if agent_id not in self.agent_metrics:
            return

        metrics = self.agent_metrics[agent_id]
        previous_status = metrics.status

        # Calculate current health score
        health_score = metrics.get_health_score()

        # Determine new status
        new_status = self._determine_health_status(metrics, health_score)

        # Update status if changed
        if new_status != previous_status:
            metrics.status = new_status
            metrics.status_change_time = datetime.now()

            self.logger.info(
                "Agent %s status changed: %s -> %s (health score: %.2f)",
                agent_id,
                previous_status.value,
                new_status.value,
                health_score,
            )

            # Notify callbacks
            await self._notify_status_change(agent_id, previous_status, new_status)

            # Handle failures
            if new_status == AgentHealthStatus.FAILED:
                await self._handle_agent_failure(agent_id)
            elif (
                previous_status == AgentHealthStatus.FAILED
                and new_status != AgentHealthStatus.FAILED
            ):
                await self._handle_agent_recovery(agent_id)

    def _determine_health_status(
        self, metrics: AgentHealthMetrics, health_score: float
    ) -> AgentHealthStatus:
        """Determine agent health status based on metrics."""
        # Check if agent is responsive
        if not metrics.is_responsive(self.response_timeout):
            return AgentHealthStatus.FAILED

        # Check consecutive failures
        if metrics.consecutive_failures >= self.failure_threshold:
            return AgentHealthStatus.FAILED

        # Check health score thresholds
        if health_score >= self.degraded_threshold:
            return AgentHealthStatus.HEALTHY
        elif health_score >= self.unhealthy_threshold:
            return AgentHealthStatus.DEGRADED
        else:
            return AgentHealthStatus.UNHEALTHY

    async def _handle_agent_failure(self, agent_id: str) -> None:
        """Handle agent failure."""
        self.logger.warning("Agent failure detected: %s", agent_id)

        # Notify failure callbacks
        for callback in self.failure_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(agent_id, self.agent_metrics[agent_id])
                else:
                    callback(agent_id, self.agent_metrics[agent_id])
            except Exception as e:
                self.logger.error("Error in failure callback: %s", e)

        # Attempt auto-recovery if enabled
        if self.enable_auto_recovery:
            await self._attempt_agent_recovery(agent_id)

    async def _handle_agent_recovery(self, agent_id: str) -> None:
        """Handle agent recovery."""
        self.logger.info("Agent recovery detected: %s", agent_id)

        # Notify recovery callbacks
        for callback in self.recovery_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(agent_id, self.agent_metrics[agent_id])
                else:
                    callback(agent_id, self.agent_metrics[agent_id])
            except Exception as e:
                self.logger.error("Error in recovery callback: %s", e)

    async def _notify_status_change(
        self,
        agent_id: str,
        old_status: AgentHealthStatus,
        new_status: AgentHealthStatus,
    ) -> None:
        """Notify status change callbacks."""
        for callback in self.status_change_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(agent_id, old_status, new_status)
                else:
                    callback(agent_id, old_status, new_status)
            except Exception as e:
                self.logger.error("Error in status change callback: %s", e)

    async def _attempt_agent_recovery(self, agent_id: str) -> None:
        """Attempt to recover a failed agent."""
        self.logger.info("Attempting recovery for agent: %s", agent_id)

        try:
            # This would typically involve:
            # 1. Restarting the agent
            # 2. Clearing its state
            # 3. Reinitializing connections
            # 4. Verifying recovery

            # Simulate recovery attempt
            await asyncio.sleep(1.0)

            # Reset failure counters
            if agent_id in self.agent_metrics:
                metrics = self.agent_metrics[agent_id]
                metrics.consecutive_failures = 0
                metrics.record_heartbeat()

            self.logger.info("Recovery attempt completed for agent: %s", agent_id)

        except Exception as e:
            self.logger.error("Recovery attempt failed for agent %s: %s", agent_id, e)

    def record_agent_action(self, agent_id: str, success: bool, response_time: float = 0.0) -> None:
        """Record an agent action result.

        Args:
            agent_id: The ID of the agent.
            success: Whether the action was successful.
            response_time: The response time for the action.
        """
        if agent_id not in self.agent_metrics:
            self.register_agent(agent_id)
        self.agent_metrics[agent_id].update_action_result(success, response_time)

    def record_agent_heartbeat(self, agent_id: str) -> None:
        """Record agent heartbeat.

        Args:
            agent_id: The ID of the agent.
        """
        if agent_id not in self.agent_metrics:
            self.register_agent(agent_id)
        self.agent_metrics[agent_id].record_heartbeat()

    def update_agent_resources(self, agent_id: str, memory_usage: float, cpu_usage: float) -> None:
        """Update agent resource usage.

        Args:
            agent_id: The ID of the agent.
            memory_usage: The current memory usage.
            cpu_usage: The current CPU usage.
        """
        if agent_id not in self.agent_metrics:
            self.register_agent(agent_id)
        self.agent_metrics[agent_id].update_resource_usage(memory_usage, cpu_usage)

    def update_agent_learning_progress(self, agent_id: str, progress: float) -> None:
        """Update agent learning progress.

        Args:
            agent_id: The ID of the agent.
            progress: The learning progress value.
        """
        if agent_id not in self.agent_metrics:
            self.register_agent(agent_id)
        self.agent_metrics[agent_id].update_learning_progress(progress)

    def get_agent_status(self, agent_id: str) -> Optional[AgentHealthStatus]:
        """Get current agent status.

        Args:
            agent_id: The ID of the agent.

        Returns:
            The current status of the agent, or None if not found.
        """
        if agent_id in self.agent_metrics:
            return self.agent_metrics[agent_id].status
        return None

    def get_agent_metrics(self, agent_id: str) -> Optional[AgentHealthMetrics]:
        """Get agent health metrics.

        Args:
            agent_id: The ID of the agent.

        Returns:
            The health metrics of the agent, or None if not found.
        """
        return self.agent_metrics.get(agent_id)

    def get_all_agent_metrics(self) -> Dict[str, AgentHealthMetrics]:
        """Get all agent health metrics.

        Returns:
            A dictionary of all agent health metrics.
        """
        return self.agent_metrics.copy()

    def get_failed_agents(self) -> List[str]:
        """Get list of failed agents.

        Returns:
            A list of failed agent IDs.
        """
        return [
            agent_id
            for agent_id, metrics in self.agent_metrics.items()
            if metrics.status == AgentHealthStatus.FAILED
        ]

    def get_healthy_agents(self) -> List[str]:
        """Get list of healthy agents.

        Returns:
            A list of healthy agent IDs.
        """
        return [
            agent_id
            for agent_id, metrics in self.agent_metrics.items()
            if metrics.status == AgentHealthStatus.HEALTHY
        ]

    def get_system_health_summary(self) -> Dict[str, Any]:
        """Get system-wide health summary.

        Returns:
            A dictionary containing the system health summary.
        """
        if not self.agent_metrics:
            return {
                "total_agents": 0,
                "healthy_agents": 0,
                "degraded_agents": 0,
                "unhealthy_agents": 0,
                "failed_agents": 0,
                "overall_health_score": 0.0,
                "system_status": "unknown",
            }

        status_counts = {status: 0 for status in AgentHealthStatus}
        total_health_score = 0.0

        for metrics in self.agent_metrics.values():
            status_counts[metrics.status] += 1
            total_health_score += metrics.get_health_score()

        overall_health_score = total_health_score / len(self.agent_metrics)

        # Determine system status
        if status_counts[AgentHealthStatus.FAILED] > 0:
            system_status = "critical"
        elif status_counts[AgentHealthStatus.UNHEALTHY] > 0:
            system_status = "unhealthy"
        elif status_counts[AgentHealthStatus.DEGRADED] > 0:
            system_status = "degraded"
        else:
            system_status = "healthy"

        return {
            "total_agents": len(self.agent_metrics),
            "healthy_agents": status_counts[AgentHealthStatus.HEALTHY],
            "degraded_agents": status_counts[AgentHealthStatus.DEGRADED],
            "unhealthy_agents": status_counts[AgentHealthStatus.UNHEALTHY],
            "failed_agents": status_counts[AgentHealthStatus.FAILED],
            "overall_health_score": overall_health_score,
            "system_status": system_status,
        }

    def add_failure_callback(self, callback: Callable) -> None:
        """Add callback for agent failures.

        Args:
            callback: The callback function to add.
        """
        self.failure_callbacks.append(callback)

    def add_recovery_callback(self, callback: Callable) -> None:
        """Add callback for agent recoveries.

        Args:
            callback: The callback function to add.
        """
        self.recovery_callbacks.append(callback)

    def add_status_change_callback(self, callback: Callable) -> None:
        """Add callback for status changes.

        Args:
            callback: The callback function to add.
        """
        self.status_change_callbacks.append(callback)

    def remove_failure_callback(self, callback: Callable) -> None:
        """Remove failure callback.

        Args:
            callback: The callback function to remove.
        """
        if callback in self.failure_callbacks:
            self.failure_callbacks.remove(callback)

    def remove_recovery_callback(self, callback: Callable) -> None:
        """Remove recovery callback.

        Args:
            callback: The callback function to remove.
        """
        if callback in self.recovery_callbacks:
            self.recovery_callbacks.remove(callback)

    def remove_status_change_callback(self, callback: Callable) -> None:
        """Remove status change callback.

        Args:
            callback: The callback function to remove.
        """
        if callback in self.status_change_callbacks:
            self.status_change_callbacks.remove(callback)

    async def shutdown(self) -> None:
        """Shutdown agent monitor."""
        await self.stop_monitoring()

        # Clear all data
        self.agent_metrics.clear()
        self.monitored_agents.clear()
        self.failure_callbacks.clear()
        self.recovery_callbacks.clear()
        self.status_change_callbacks.clear()

        self.logger.info("Agent monitor shutdown complete")
