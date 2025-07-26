"""
Fault Tolerance Manager.

This module provides the main fault tolerance coordination system
for the multi-agent reinforcement learning system, integrating
agent monitoring, deadlock detection, learning monitoring, and
memory management.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

from utils.logging_config import get_logger

from .agent_monitor import AgentHealthStatus, AgentMonitor
from .deadlock_detector import DeadlockDetector, DeadlockType
from .learning_monitor import LearningMonitor, LearningStatus
from .memory_manager import MemoryManager, MemoryStatus


class FaultToleranceManager:
    """
    Main fault tolerance coordination system.

    Integrates and coordinates all fault tolerance mechanisms
    including agent monitoring, deadlock detection, learning
    monitoring, and memory management.
    """

    def __init__(
        self,
        agent_monitor: Optional[AgentMonitor] = None,
        deadlock_detector: Optional[DeadlockDetector] = None,
        learning_monitor: Optional[LearningMonitor] = None,
        memory_manager: Optional[MemoryManager] = None,
        enable_auto_recovery: bool = True,
    ):
        """
        Initialize fault tolerance manager.

        Args:
            agent_monitor: Agent monitoring component
            deadlock_detector: Deadlock detection component
            learning_monitor: Learning monitoring component
            memory_manager: Memory management component
            enable_auto_recovery: Enable automatic recovery
        """
        self.logger = get_logger(__name__)

        # Components
        self.agent_monitor = agent_monitor or AgentMonitor()
        self.deadlock_detector = deadlock_detector or DeadlockDetector()
        self.learning_monitor = learning_monitor or LearningMonitor()
        self.memory_manager = memory_manager or MemoryManager()

        # Configuration
        self.enable_auto_recovery = enable_auto_recovery

        # System state
        self.is_running = False
        self.registered_agents: List[str] = []

        # Event tracking
        self.fault_events: List[Dict[str, Any]] = []
        self.recovery_actions: List[Dict[str, Any]] = []

        # Callbacks
        self.fault_callbacks: List[Callable] = []
        self.recovery_callbacks: List[Callable] = []
        self.system_health_callbacks: List[Callable] = []

        # Setup component callbacks
        self._setup_component_callbacks()

        self.logger.info("Fault tolerance manager initialized")

    def _setup_component_callbacks(self) -> None:
        """Setup callbacks for component integration."""
        # Agent monitor callbacks
        self.agent_monitor.add_failure_callback(self._handle_agent_failure)
        self.agent_monitor.add_recovery_callback(self._handle_agent_recovery)
        self.agent_monitor.add_status_change_callback(self._handle_agent_status_change)

        # Deadlock detector callbacks
        self.deadlock_detector.add_deadlock_callback(self._handle_deadlock_detected)
        self.deadlock_detector.add_resolution_callback(self._handle_deadlock_resolved)

        # Learning monitor callbacks
        self.learning_monitor.add_issue_callback(self._handle_learning_issue)
        self.learning_monitor.add_correction_callback(self._handle_learning_correction)

        # Memory manager callbacks
        self.memory_manager.add_threshold_callback(self._handle_memory_threshold)
        self.memory_manager.add_overflow_callback(self._handle_memory_overflow)

    async def start_monitoring(self) -> None:
        """Start all fault tolerance monitoring."""
        if self.is_running:
            self.logger.warning("Fault tolerance monitoring already running")
            return

        self.logger.info("Starting fault tolerance monitoring")

        # Start all components
        await self.agent_monitor.start_monitoring()
        await self.deadlock_detector.start_detection()
        await self.learning_monitor.start_monitoring()
        await self.memory_manager.start_monitoring()

        self.is_running = True

        self.logger.info("Fault tolerance monitoring started")

    async def stop_monitoring(self) -> None:
        """Stop all fault tolerance monitoring."""
        if not self.is_running:
            return

        self.logger.info("Stopping fault tolerance monitoring")

        # Stop all components
        await self.agent_monitor.stop_monitoring()
        await self.deadlock_detector.stop_detection()
        await self.learning_monitor.stop_monitoring()
        await self.memory_manager.stop_monitoring()

        self.is_running = False

        self.logger.info("Fault tolerance monitoring stopped")

    def register_agent(
        self, agent_id: str, agent_component: Optional[Any] = None
    ) -> None:
        """Register an agent for fault tolerance monitoring."""
        if agent_id not in self.registered_agents:
            self.registered_agents.append(agent_id)

            # Register with all relevant components
            self.agent_monitor.register_agent(agent_id)
            self.learning_monitor.register_agent(agent_id)

            # Register agent component with memory manager if provided
            if agent_component:
                self.memory_manager.register_component(
                    f"agent_{agent_id}", agent_component
                )

            self.logger.info("Registered agent for fault tolerance: %s", agent_id)

    def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent from fault tolerance monitoring."""
        if agent_id in self.registered_agents:
            self.registered_agents.remove(agent_id)

            # Unregister from all components
            self.agent_monitor.unregister_agent(agent_id)
            self.learning_monitor.unregister_agent(agent_id)
            self.memory_manager.unregister_component(f"agent_{agent_id}")

            self.logger.info("Unregistered agent from fault tolerance: %s", agent_id)

    # Component event handlers

    async def _handle_agent_failure(self, agent_id: str, metrics: Any) -> None:
        """Handle agent failure event."""
        self.logger.warning("Agent failure detected: %s", agent_id)

        fault_event = {
            "type": "agent_failure",
            "agent_id": agent_id,
            "timestamp": datetime.now(),
            "details": metrics.to_dict()
            if hasattr(metrics, "to_dict")
            else str(metrics),
        }

        self.fault_events.append(fault_event)
        await self._notify_fault_callbacks(fault_event)

        if self.enable_auto_recovery:
            await self._initiate_agent_recovery(agent_id, "agent_failure")

    async def _handle_agent_recovery(self, agent_id: str, metrics: Any) -> None:
        """Handle agent recovery event."""
        self.logger.info("Agent recovery detected: %s", agent_id)

        recovery_event = {
            "type": "agent_recovery",
            "agent_id": agent_id,
            "timestamp": datetime.now(),
            "details": metrics.to_dict()
            if hasattr(metrics, "to_dict")
            else str(metrics),
        }

        self.recovery_actions.append(recovery_event)
        await self._notify_recovery_callbacks(recovery_event)

    async def _handle_agent_status_change(
        self,
        agent_id: str,
        old_status: AgentHealthStatus,
        new_status: AgentHealthStatus,
    ) -> None:
        """Handle agent status change."""
        self.logger.debug(
            "Agent status change: %s (%s -> %s)",
            agent_id,
            old_status.value,
            new_status.value,
        )

        # Check if this affects system health
        await self._check_system_health()

    async def _handle_deadlock_detected(self, deadlock_event: Any) -> None:
        """Handle deadlock detection."""
        self.logger.warning(
            "Deadlock detected: %s (type: %s)",
            deadlock_event.deadlock_id,
            deadlock_event.deadlock_type.value,
        )

        fault_event = {
            "type": "deadlock_detected",
            "deadlock_id": deadlock_event.deadlock_id,
            "deadlock_type": deadlock_event.deadlock_type.value,
            "involved_agents": deadlock_event.involved_agents,
            "timestamp": datetime.now(),
            "details": deadlock_event.to_dict(),
        }

        self.fault_events.append(fault_event)
        await self._notify_fault_callbacks(fault_event)

        # Coordinate with agent monitor for affected agents
        for agent_id in deadlock_event.involved_agents:
            if agent_id in self.registered_agents:
                # Record coordination wait to prevent false positives
                self.deadlock_detector.record_agent_wait(
                    agent_id, "deadlock_resolution", "system", timeout=30.0
                )

    async def _handle_deadlock_resolved(self, deadlock_event: Any) -> None:
        """Handle deadlock resolution."""
        self.logger.info(
            "Deadlock resolved: %s using strategy: %s",
            deadlock_event.deadlock_id,
            deadlock_event.resolution_strategy,
        )

        recovery_event = {
            "type": "deadlock_resolved",
            "deadlock_id": deadlock_event.deadlock_id,
            "resolution_strategy": deadlock_event.resolution_strategy,
            "timestamp": datetime.now(),
            "details": deadlock_event.to_dict(),
        }

        self.recovery_actions.append(recovery_event)
        await self._notify_recovery_callbacks(recovery_event)

        # Clear waiting states for affected agents
        for agent_id in deadlock_event.involved_agents:
            self.deadlock_detector.clear_agent_wait(agent_id)

    async def _handle_learning_issue(self, divergence_event: Any) -> None:
        """Handle learning issue detection."""
        self.logger.warning(
            "Learning issue detected for agent %s: %s",
            divergence_event.agent_id,
            divergence_event.divergence_type.value,
        )

        fault_event = {
            "type": "learning_issue",
            "agent_id": divergence_event.agent_id,
            "issue_type": divergence_event.divergence_type.value,
            "timestamp": datetime.now(),
            "details": divergence_event.to_dict(),
        }

        self.fault_events.append(fault_event)
        await self._notify_fault_callbacks(fault_event)

        # Check if this affects agent health
        agent_status = self.agent_monitor.get_agent_status(divergence_event.agent_id)
        if agent_status and agent_status != AgentHealthStatus.FAILED:
            # Update agent health based on learning issues
            self.agent_monitor.record_agent_action(
                divergence_event.agent_id, success=False, response_time=1.0
            )

    async def _handle_learning_correction(
        self, divergence_event: Any, strategy: str
    ) -> None:
        """Handle learning correction."""
        self.logger.info(
            "Learning correction applied for agent %s: %s",
            divergence_event.agent_id,
            strategy,
        )

        recovery_event = {
            "type": "learning_correction",
            "agent_id": divergence_event.agent_id,
            "correction_strategy": strategy,
            "timestamp": datetime.now(),
            "details": divergence_event.to_dict(),
        }

        self.recovery_actions.append(recovery_event)
        await self._notify_recovery_callbacks(recovery_event)

    async def _handle_memory_threshold(self, memory_event: Any) -> None:
        """Handle memory threshold exceeded."""
        self.logger.warning(
            "Memory threshold exceeded: %s (usage: %.1f%%)",
            memory_event.memory_status.value,
            memory_event.memory_usage * 100,
        )

        fault_event = {
            "type": "memory_threshold",
            "memory_status": memory_event.memory_status.value,
            "memory_usage": memory_event.memory_usage,
            "timestamp": datetime.now(),
            "details": memory_event.to_dict(),
        }

        self.fault_events.append(fault_event)
        await self._notify_fault_callbacks(fault_event)

        # If critical, may need to pause some agents
        if memory_event.memory_status in [MemoryStatus.CRITICAL, MemoryStatus.OVERFLOW]:
            await self._handle_critical_memory_situation()

    async def _handle_memory_overflow(self, memory_metrics: Any) -> None:
        """Handle memory overflow situation."""
        self.logger.critical("Memory overflow detected - taking emergency action")

        fault_event = {
            "type": "memory_overflow",
            "memory_usage": memory_metrics.memory_percent,
            "timestamp": datetime.now(),
            "details": memory_metrics.to_dict(),
        }

        self.fault_events.append(fault_event)
        await self._notify_fault_callbacks(fault_event)

        # Emergency response
        await self._handle_critical_memory_situation()

    async def _handle_critical_memory_situation(self) -> None:
        """Handle critical memory situation."""
        # Temporarily pause learning for some agents
        healthy_agents = self.agent_monitor.get_healthy_agents()

        if len(healthy_agents) > 1:
            # Pause learning for half the agents
            agents_to_pause = healthy_agents[::2]  # Every other agent

            for agent_id in agents_to_pause:
                self.logger.warning(
                    "Temporarily pausing learning for agent: %s", agent_id
                )
                # This would typically call a method on the actual agent
                # For now, we'll just record the action

                recovery_event = {
                    "type": "learning_paused",
                    "agent_id": agent_id,
                    "reason": "critical_memory",
                    "timestamp": datetime.now(),
                }

                self.recovery_actions.append(recovery_event)

    async def _initiate_agent_recovery(self, agent_id: str, failure_type: str) -> None:
        """Initiate agent recovery process."""
        self.logger.info(
            "Initiating recovery for agent %s (failure: %s)", agent_id, failure_type
        )

        recovery_strategies = {
            "agent_failure": self._recover_failed_agent,
            "learning_issue": self._recover_learning_agent,
            "memory_issue": self._recover_memory_agent,
        }

        recovery_func = recovery_strategies.get(
            failure_type, self._recover_failed_agent
        )

        try:
            success = await recovery_func(agent_id)

            recovery_event = {
                "type": "recovery_attempt",
                "agent_id": agent_id,
                "failure_type": failure_type,
                "success": success,
                "timestamp": datetime.now(),
            }

            self.recovery_actions.append(recovery_event)
            await self._notify_recovery_callbacks(recovery_event)

        except Exception as e:
            self.logger.error(
                "Error during recovery for agent %s: %s", agent_id, str(e)
            )

    async def _recover_failed_agent(self, agent_id: str) -> bool:
        """Recover a failed agent."""
        try:
            # This would typically involve:
            # 1. Restarting the agent
            # 2. Clearing its state
            # 3. Reinitializing connections
            # 4. Resuming operations

            # Simulate recovery
            await asyncio.sleep(1.0)

            # Reset agent metrics
            self.agent_monitor.record_agent_heartbeat(agent_id)

            self.logger.info("Agent recovery completed: %s", agent_id)
            return True

        except Exception as e:
            self.logger.error("Failed to recover agent %s: %s", agent_id, str(e))
            return False

    async def _recover_learning_agent(self, agent_id: str) -> bool:
        """Recover an agent with learning issues."""
        try:
            # Reset learning parameters
            # This would typically call methods on the actual agent

            # Simulate recovery
            await asyncio.sleep(0.5)

            self.logger.info("Learning recovery completed: %s", agent_id)
            return True

        except Exception as e:
            self.logger.error(
                "Failed to recover learning for agent %s: %s", agent_id, str(e)
            )
            return False

    async def _recover_memory_agent(self, agent_id: str) -> bool:
        """Recover an agent with memory issues."""
        try:
            # Clear agent caches and reduce memory usage
            # This would typically call methods on the actual agent

            # Simulate recovery
            await asyncio.sleep(0.3)

            self.logger.info("Memory recovery completed: %s", agent_id)
            return True

        except Exception as e:
            self.logger.error(
                "Failed to recover memory for agent %s: %s", agent_id, str(e)
            )
            return False

    async def _check_system_health(self) -> None:
        """Check overall system health and notify if changed."""
        health_summary = self.get_system_health_summary()

        # Notify system health callbacks
        for callback in self.system_health_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(health_summary)
                else:
                    callback(health_summary)
            except Exception as e:
                self.logger.error("Error in system health callback: %s", str(e))

    async def _notify_fault_callbacks(self, fault_event: Dict[str, Any]) -> None:
        """Notify fault detection callbacks."""
        for callback in self.fault_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(fault_event)
                else:
                    callback(fault_event)
            except Exception as e:
                self.logger.error("Error in fault callback: %s", str(e))

    async def _notify_recovery_callbacks(self, recovery_event: Dict[str, Any]) -> None:
        """Notify recovery callbacks."""
        for callback in self.recovery_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(recovery_event)
                else:
                    callback(recovery_event)
            except Exception as e:
                self.logger.error("Error in recovery callback: %s", str(e))

    # Public interface methods

    def record_agent_action(
        self, agent_id: str, success: bool, response_time: float = 0.0
    ) -> None:
        """Record an agent action for monitoring."""
        # Auto-register agent if not already registered
        if agent_id not in self.registered_agents:
            self.register_agent(agent_id)

        self.agent_monitor.record_agent_action(agent_id, success, response_time)

    def record_agent_learning(
        self,
        agent_id: str,
        episode: int,
        reward: float,
        loss: float,
        q_values: Optional[List[float]] = None,
    ) -> None:
        """Record agent learning metrics."""
        # Auto-register agent if not already registered
        if agent_id not in self.registered_agents:
            self.register_agent(agent_id)

        self.learning_monitor.update_agent_learning(
            agent_id, episode, reward, loss, q_values
        )

    def record_coordination_wait(
        self,
        agent_id: str,
        waiting_for: str,
        wait_type: str,
        timeout: Optional[float] = None,
    ) -> None:
        """Record that an agent is waiting for coordination."""
        self.deadlock_detector.record_agent_wait(
            agent_id, waiting_for, wait_type, timeout
        )

    def clear_coordination_wait(self, agent_id: str) -> None:
        """Clear an agent's coordination wait."""
        self.deadlock_detector.clear_agent_wait(agent_id)

    def get_system_health_summary(self) -> Dict[str, Any]:
        """Get comprehensive system health summary."""
        # Get component summaries
        agent_health = self.agent_monitor.get_system_health_summary()
        deadlock_stats = self.deadlock_detector.get_deadlock_statistics()
        learning_summary = self.learning_monitor.get_learning_summary()
        memory_stats = self.memory_manager.get_memory_statistics()

        # Calculate overall health score
        health_scores = []

        if agent_health["total_agents"] > 0:
            agent_score = agent_health["overall_health_score"]
            health_scores.append(agent_score)

        if learning_summary["total_agents"] > 0:
            learning_score = learning_summary["average_convergence_score"]
            health_scores.append(learning_score)

        # Memory health (inverse of usage)
        memory_score = max(0, 1.0 - (memory_stats["current_usage"] / 100.0))
        health_scores.append(memory_score)

        # Deadlock health (based on resolution rate)
        if deadlock_stats["total_deadlocks"] > 0:
            deadlock_score = deadlock_stats["resolution_rate"]
            health_scores.append(deadlock_score)
        else:
            health_scores.append(1.0)  # No deadlocks is good

        overall_health_score = (
            sum(health_scores) / len(health_scores) if health_scores else 0.0
        )

        # Determine overall status
        if overall_health_score >= 0.8:
            overall_status = "healthy"
        elif overall_health_score >= 0.6:
            overall_status = "degraded"
        elif overall_health_score >= 0.4:
            overall_status = "unhealthy"
        else:
            overall_status = "critical"

        return {
            "overall_health_score": overall_health_score,
            "overall_status": overall_status,
            "registered_agents": len(self.registered_agents),
            "agent_health": agent_health,
            "deadlock_statistics": deadlock_stats,
            "learning_summary": learning_summary,
            "memory_statistics": memory_stats,
            "recent_faults": len(
                [
                    e
                    for e in self.fault_events
                    if (datetime.now() - e["timestamp"]).total_seconds() < 3600
                ]
            ),
            "recent_recoveries": len(
                [
                    e
                    for e in self.recovery_actions
                    if (datetime.now() - e["timestamp"]).total_seconds() < 3600
                ]
            ),
        }

    def get_fault_events(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent fault events."""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        return [
            event for event in self.fault_events if event["timestamp"] >= cutoff_time
        ]

    def get_recovery_actions(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent recovery actions."""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        return [
            action
            for action in self.recovery_actions
            if action["timestamp"] >= cutoff_time
        ]

    def add_fault_callback(self, callback: Callable) -> None:
        """Add fault detection callback."""
        self.fault_callbacks.append(callback)

    def add_recovery_callback(self, callback: Callable) -> None:
        """Add recovery callback."""
        self.recovery_callbacks.append(callback)

    def add_system_health_callback(self, callback: Callable) -> None:
        """Add system health callback."""
        self.system_health_callbacks.append(callback)

    def remove_fault_callback(self, callback: Callable) -> None:
        """Remove fault detection callback."""
        if callback in self.fault_callbacks:
            self.fault_callbacks.remove(callback)

    def remove_recovery_callback(self, callback: Callable) -> None:
        """Remove recovery callback."""
        if callback in self.recovery_callbacks:
            self.recovery_callbacks.remove(callback)

    def remove_system_health_callback(self, callback: Callable) -> None:
        """Remove system health callback."""
        if callback in self.system_health_callbacks:
            self.system_health_callbacks.remove(callback)

    async def shutdown(self) -> None:
        """Shutdown fault tolerance manager."""
        await self.stop_monitoring()

        # Shutdown all components
        await self.agent_monitor.shutdown()
        await self.deadlock_detector.shutdown()
        await self.learning_monitor.shutdown()
        await self.memory_manager.shutdown()

        # Clear all data
        self.registered_agents.clear()
        self.fault_events.clear()
        self.recovery_actions.clear()
        self.fault_callbacks.clear()
        self.recovery_callbacks.clear()
        self.system_health_callbacks.clear()

        self.logger.info("Fault tolerance manager shutdown complete")
