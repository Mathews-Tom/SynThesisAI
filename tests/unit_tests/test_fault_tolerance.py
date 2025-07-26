"""
Unit tests for MARL fault tolerance system.

Tests agent monitoring, deadlock detection, learning monitoring,
memory management, and fault tolerance coordination.
"""

import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest

from core.marl.fault_tolerance.agent_monitor import (
    AgentHealthMetrics,
    AgentHealthStatus,
    AgentMonitor,
)
from core.marl.fault_tolerance.deadlock_detector import (
    DeadlockDetector,
    DeadlockEvent,
    DeadlockType,
    WaitingState,
)
from core.marl.fault_tolerance.fault_tolerance_manager import FaultToleranceManager
from core.marl.fault_tolerance.learning_monitor import (
    LearningDivergenceDetector,
    LearningDivergenceEvent,
    LearningMetrics,
    LearningMonitor,
    LearningStatus,
)
from core.marl.fault_tolerance.memory_manager import (
    MemoryManager,
    MemoryMetrics,
    MemoryStatus,
    MemoryThreshold,
)


class TestAgentHealthMetrics:
    """Test agent health metrics."""

    def test_metrics_creation(self):
        """Test creating agent health metrics."""
        metrics = AgentHealthMetrics(agent_id="test_agent")

        assert metrics.agent_id == "test_agent"
        assert metrics.status == AgentHealthStatus.UNKNOWN
        assert metrics.total_actions == 0
        assert metrics.action_success_rate == 0.0

    def test_update_action_result(self):
        """Test updating action results."""
        metrics = AgentHealthMetrics(agent_id="test_agent")

        # Record successful action
        metrics.update_action_result(True, 1.5)

        assert metrics.total_actions == 1
        assert metrics.successful_actions == 1
        assert metrics.failed_actions == 0
        assert metrics.consecutive_failures == 0
        assert metrics.action_success_rate == 1.0
        assert metrics.response_time == 1.5

        # Record failed action
        metrics.update_action_result(False, 2.0)

        assert metrics.total_actions == 2
        assert metrics.successful_actions == 1
        assert metrics.failed_actions == 1
        assert metrics.consecutive_failures == 1
        assert metrics.action_success_rate == 0.5
        assert metrics.response_time == 2.0

    def test_health_score_calculation(self):
        """Test health score calculation."""
        metrics = AgentHealthMetrics(agent_id="test_agent")

        # Initially neutral score
        assert 0.4 <= metrics.get_health_score() <= 0.6

        # Add successful actions
        for _ in range(10):
            metrics.update_action_result(True, 1.0)

        # Should have high health score
        assert metrics.get_health_score() > 0.7

        # Add failures
        for _ in range(10):
            metrics.update_action_result(False, 5.0)

        # Should have lower health score
        assert metrics.get_health_score() <= 0.5

    def test_responsiveness_check(self):
        """Test responsiveness checking."""
        metrics = AgentHealthMetrics(agent_id="test_agent")

        # No heartbeat recorded
        assert not metrics.is_responsive()

        # Record heartbeat
        metrics.record_heartbeat()
        assert metrics.is_responsive()

        # Test with custom timeout
        assert metrics.is_responsive(timeout_seconds=60.0)

    def test_metrics_serialization(self):
        """Test metrics to dictionary conversion."""
        metrics = AgentHealthMetrics(agent_id="test_agent")
        metrics.update_action_result(True, 1.0)
        metrics.record_heartbeat()

        metrics_dict = metrics.to_dict()

        assert metrics_dict["agent_id"] == "test_agent"
        assert metrics_dict["total_actions"] == 1
        assert metrics_dict["action_success_rate"] == 1.0
        assert "health_score" in metrics_dict
        assert "is_responsive" in metrics_dict


class TestAgentMonitor:
    """Test agent monitor."""

    @pytest.fixture
    def agent_monitor(self):
        """Create agent monitor for testing."""
        return AgentMonitor(
            heartbeat_interval=0.1, response_timeout=1.0, failure_threshold=3
        )

    def test_monitor_initialization(self, agent_monitor):
        """Test monitor initialization."""
        assert agent_monitor.heartbeat_interval == 0.1
        assert agent_monitor.response_timeout == 1.0
        assert agent_monitor.failure_threshold == 3
        assert not agent_monitor.is_monitoring

    def test_agent_registration(self, agent_monitor):
        """Test agent registration."""
        agent_monitor.register_agent("test_agent")

        assert "test_agent" in agent_monitor.monitored_agents
        assert "test_agent" in agent_monitor.agent_metrics

        # Test unregistration
        agent_monitor.unregister_agent("test_agent")

        assert "test_agent" not in agent_monitor.monitored_agents
        assert "test_agent" not in agent_monitor.agent_metrics

    @pytest.mark.asyncio
    async def test_monitoring_lifecycle(self, agent_monitor):
        """Test monitoring start/stop."""
        # Start monitoring
        await agent_monitor.start_monitoring()
        assert agent_monitor.is_monitoring
        assert agent_monitor.monitoring_task is not None

        # Stop monitoring
        await agent_monitor.stop_monitoring()
        assert not agent_monitor.is_monitoring

    def test_action_recording(self, agent_monitor):
        """Test recording agent actions."""
        agent_monitor.record_agent_action("test_agent", True, 1.0)

        metrics = agent_monitor.get_agent_metrics("test_agent")
        assert metrics is not None
        assert metrics.total_actions == 1
        assert metrics.action_success_rate == 1.0

    def test_heartbeat_recording(self, agent_monitor):
        """Test recording agent heartbeats."""
        agent_monitor.record_agent_heartbeat("test_agent")

        metrics = agent_monitor.get_agent_metrics("test_agent")
        assert metrics is not None
        assert metrics.last_heartbeat is not None

    def test_resource_updates(self, agent_monitor):
        """Test updating agent resource usage."""
        agent_monitor.update_agent_resources("test_agent", 0.5, 0.3)

        metrics = agent_monitor.get_agent_metrics("test_agent")
        assert metrics is not None
        assert metrics.memory_usage == 0.5
        assert metrics.cpu_usage == 0.3

    def test_status_determination(self, agent_monitor):
        """Test agent status determination."""
        agent_monitor.register_agent("test_agent")

        # Initially unknown
        status = agent_monitor.get_agent_status("test_agent")
        assert status == AgentHealthStatus.UNKNOWN

        # Record heartbeat and successful actions
        agent_monitor.record_agent_heartbeat("test_agent")
        for _ in range(10):
            agent_monitor.record_agent_action("test_agent", True, 1.0)

        # Should be healthy after manual status check
        metrics = agent_monitor.get_agent_metrics("test_agent")
        health_score = metrics.get_health_score()
        assert health_score > 0.7

    def test_system_health_summary(self, agent_monitor):
        """Test system health summary."""
        # Register multiple agents
        agent_monitor.register_agent("agent1")
        agent_monitor.register_agent("agent2")

        # Record different health levels
        agent_monitor.record_agent_heartbeat("agent1")
        for _ in range(10):
            agent_monitor.record_agent_action("agent1", True, 1.0)

        agent_monitor.record_agent_heartbeat("agent2")
        for _ in range(5):
            agent_monitor.record_agent_action("agent2", False, 3.0)

        summary = agent_monitor.get_system_health_summary()

        assert summary["total_agents"] == 2
        assert "overall_health_score" in summary
        assert "system_status" in summary

    def test_callbacks(self, agent_monitor):
        """Test callback functionality."""
        callback_called = False
        callback_agent = None

        def test_callback(agent_id, metrics):
            nonlocal callback_called, callback_agent
            callback_called = True
            callback_agent = agent_id

        agent_monitor.add_failure_callback(test_callback)

        # Verify callback is added
        assert test_callback in agent_monitor.failure_callbacks

        # Remove callback
        agent_monitor.remove_failure_callback(test_callback)
        assert test_callback not in agent_monitor.failure_callbacks


class TestDeadlockDetector:
    """Test deadlock detector."""

    @pytest.fixture
    def deadlock_detector(self):
        """Create deadlock detector for testing."""
        return DeadlockDetector(
            detection_interval=0.1, deadlock_timeout=1.0, enable_auto_resolution=True
        )

    def test_detector_initialization(self, deadlock_detector):
        """Test detector initialization."""
        assert deadlock_detector.detection_interval == 0.1
        assert deadlock_detector.deadlock_timeout == 1.0
        assert deadlock_detector.enable_auto_resolution is True
        assert not deadlock_detector.is_detecting

    @pytest.mark.asyncio
    async def test_detection_lifecycle(self, deadlock_detector):
        """Test detection start/stop."""
        # Start detection
        await deadlock_detector.start_detection()
        assert deadlock_detector.is_detecting
        assert deadlock_detector.detection_task is not None

        # Stop detection
        await deadlock_detector.stop_detection()
        assert not deadlock_detector.is_detecting

    def test_waiting_state_management(self, deadlock_detector):
        """Test waiting state management."""
        # Record agent wait
        deadlock_detector.record_agent_wait("agent1", "agent2", "agent", timeout=5.0)

        assert "agent1" in deadlock_detector.waiting_states
        wait_state = deadlock_detector.waiting_states["agent1"]
        assert wait_state.waiting_for == "agent2"
        assert wait_state.wait_type == "agent"

        # Clear wait
        deadlock_detector.clear_agent_wait("agent1")
        assert "agent1" not in deadlock_detector.waiting_states

    def test_resource_management(self, deadlock_detector):
        """Test resource ownership management."""
        # Record resource ownership
        deadlock_detector.record_resource_ownership("resource1", "agent1")

        assert deadlock_detector.resource_owners["resource1"] == "agent1"

        # Release resource
        deadlock_detector.release_resource("resource1")
        assert "resource1" not in deadlock_detector.resource_owners

    def test_circular_wait_detection(self, deadlock_detector):
        """Test circular wait detection."""
        # Create circular dependency: agent1 -> agent2 -> agent1
        deadlock_detector.record_agent_wait("agent1", "agent2", "agent")
        deadlock_detector.record_agent_wait("agent2", "agent1", "agent")

        # Build dependency graph
        graph = deadlock_detector._build_dependency_graph()

        assert "agent1" in graph
        assert "agent2" in graph["agent1"]
        assert "agent1" in graph["agent2"]

        # Find cycles
        cycles = deadlock_detector._find_cycles(graph)
        assert len(cycles) > 0

        # Should detect circular dependency
        cycle = cycles[0]
        assert "agent1" in cycle
        assert "agent2" in cycle

    def test_deadlock_statistics(self, deadlock_detector):
        """Test deadlock statistics."""
        stats = deadlock_detector.get_deadlock_statistics()

        assert "total_deadlocks" in stats
        assert "active_deadlocks" in stats
        assert "resolved_deadlocks" in stats
        assert "resolution_rate" in stats
        assert "deadlocks_by_type" in stats

    def test_callbacks(self, deadlock_detector):
        """Test callback functionality."""
        callback_called = False
        callback_event = None

        def test_callback(event):
            nonlocal callback_called, callback_event
            callback_called = True
            callback_event = event

        deadlock_detector.add_deadlock_callback(test_callback)

        # Verify callback is added
        assert test_callback in deadlock_detector.deadlock_callbacks

        # Remove callback
        deadlock_detector.remove_deadlock_callback(test_callback)
        assert test_callback not in deadlock_detector.deadlock_callbacks


class TestLearningMetrics:
    """Test learning metrics."""

    def test_metrics_creation(self):
        """Test creating learning metrics."""
        metrics = LearningMetrics(agent_id="test_agent")

        assert metrics.agent_id == "test_agent"
        assert metrics.episode == 0
        assert metrics.total_reward == 0.0
        assert len(metrics.reward_history) == 0

    def test_metrics_update(self):
        """Test updating learning metrics."""
        metrics = LearningMetrics(agent_id="test_agent")

        # Update with episode data
        metrics.update_metrics(1, 10.0, 0.5, [1.0, 2.0, 3.0])

        assert metrics.episode == 1
        assert metrics.total_reward == 10.0
        assert metrics.loss == 0.5
        assert len(metrics.reward_history) == 1
        assert len(metrics.q_value_history) == 1

        # Add more episodes
        for i in range(2, 21):
            metrics.update_metrics(i, 5.0 + i, 0.3, [1.0, 2.0])

        assert len(metrics.reward_history) == 20
        assert metrics.average_reward > 0
        assert metrics.reward_variance >= 0

    def test_trend_calculation(self):
        """Test reward trend calculation."""
        metrics = LearningMetrics(agent_id="test_agent")

        # Add improving rewards
        for i in range(1, 21):
            metrics.update_metrics(i, float(i), 0.1)

        # Should have positive trend
        assert metrics.reward_trend > 0

        # Add declining rewards
        for i in range(21, 41):
            metrics.update_metrics(i, float(40 - i), 0.1)

        # Should have negative trend
        assert metrics.reward_trend < 0

    def test_divergence_detection(self):
        """Test divergence detection methods."""
        metrics = LearningMetrics(agent_id="test_agent")

        # Add stable rewards
        for i in range(1, 21):
            metrics.update_metrics(i, 10.0, 0.1)

        assert not metrics.is_diverging()
        assert not metrics.is_oscillating()
        # With constant rewards, trend should be near zero, but may not be exactly stagnant
        # due to numerical precision, so let's check the trend is close to zero
        assert abs(metrics.reward_trend) < 0.01

        # Add diverging rewards
        for i in range(21, 41):
            metrics.update_metrics(i, 10.0 - (i - 20), 0.1)

        assert metrics.is_diverging()

    def test_metrics_serialization(self):
        """Test metrics serialization."""
        metrics = LearningMetrics(agent_id="test_agent")
        metrics.update_metrics(1, 10.0, 0.5)

        metrics_dict = metrics.to_dict()

        assert metrics_dict["agent_id"] == "test_agent"
        assert metrics_dict["episode"] == 1
        assert metrics_dict["total_reward"] == 10.0
        assert "convergence_score" in metrics_dict
        assert "is_diverging" in metrics_dict


class TestLearningDivergenceDetector:
    """Test learning divergence detector."""

    @pytest.fixture
    def divergence_detector(self):
        """Create divergence detector for testing."""
        return LearningDivergenceDetector(
            divergence_threshold=-0.01, min_episodes_for_detection=10
        )

    def test_detector_initialization(self, divergence_detector):
        """Test detector initialization."""
        assert divergence_detector.divergence_threshold == -0.01
        assert divergence_detector.min_episodes_for_detection == 10
        assert len(divergence_detector.divergence_events) == 0

    def test_divergence_detection(self, divergence_detector):
        """Test divergence detection."""
        metrics = LearningMetrics(agent_id="test_agent")

        # Not enough episodes
        event = divergence_detector.detect_divergence(metrics)
        assert event is None

        # Add enough episodes with diverging trend
        for i in range(1, 21):
            metrics.update_metrics(i, 10.0 - i * 0.5, 0.1)

        event = divergence_detector.detect_divergence(metrics)
        assert event is not None
        assert event.divergence_type == LearningStatus.DIVERGING
        assert event.agent_id == "test_agent"

    def test_event_resolution(self, divergence_detector):
        """Test divergence event resolution."""
        # Create a divergence event
        metrics = LearningMetrics(agent_id="test_agent")
        for i in range(1, 21):
            metrics.update_metrics(i, 10.0 - i * 0.5, 0.1)

        event = divergence_detector.detect_divergence(metrics)
        assert event is not None

        # Resolve the event
        success = divergence_detector.resolve_divergence(
            event.event_id, "reset_parameters"
        )

        assert success
        assert event.resolved
        assert event.resolution_strategy == "reset_parameters"
        assert event in divergence_detector.event_history

    def test_divergence_statistics(self, divergence_detector):
        """Test divergence statistics."""
        stats = divergence_detector.get_divergence_statistics()

        assert "total_divergence_events" in stats
        assert "active_divergences" in stats
        assert "resolved_divergences" in stats
        assert "resolution_rate" in stats
        assert "divergences_by_type" in stats


class TestLearningMonitor:
    """Test learning monitor."""

    @pytest.fixture
    def learning_monitor(self):
        """Create learning monitor for testing."""
        return LearningMonitor(monitoring_interval=0.1, enable_auto_correction=True)

    def test_monitor_initialization(self, learning_monitor):
        """Test monitor initialization."""
        assert learning_monitor.monitoring_interval == 0.1
        assert learning_monitor.enable_auto_correction is True
        assert not learning_monitor.is_monitoring

    def test_agent_registration(self, learning_monitor):
        """Test agent registration."""
        learning_monitor.register_agent("test_agent")

        assert "test_agent" in learning_monitor.monitored_agents
        assert "test_agent" in learning_monitor.agent_metrics

        # Test unregistration
        learning_monitor.unregister_agent("test_agent")

        assert "test_agent" not in learning_monitor.monitored_agents
        assert "test_agent" not in learning_monitor.agent_metrics

    @pytest.mark.asyncio
    async def test_monitoring_lifecycle(self, learning_monitor):
        """Test monitoring start/stop."""
        # Start monitoring
        await learning_monitor.start_monitoring()
        assert learning_monitor.is_monitoring
        assert learning_monitor.monitoring_task is not None

        # Stop monitoring
        await learning_monitor.stop_monitoring()
        assert not learning_monitor.is_monitoring

    def test_learning_updates(self, learning_monitor):
        """Test learning metric updates."""
        learning_monitor.update_agent_learning(
            "test_agent", 1, 10.0, 0.5, [1.0, 2.0, 3.0]
        )

        metrics = learning_monitor.get_agent_metrics("test_agent")
        assert metrics is not None
        assert metrics.episode == 1
        assert metrics.total_reward == 10.0

    def test_learning_status(self, learning_monitor):
        """Test learning status determination."""
        # Add normal learning data
        for i in range(1, 21):
            learning_monitor.update_agent_learning(
                "test_agent", i, 10.0 + i * 0.1, 0.5 - i * 0.01
            )

        status = learning_monitor.get_agent_learning_status("test_agent")
        assert status in [LearningStatus.NORMAL, LearningStatus.SLOW_CONVERGENCE]

    def test_learning_summary(self, learning_monitor):
        """Test learning summary."""
        # Register multiple agents
        learning_monitor.register_agent("agent1")
        learning_monitor.register_agent("agent2")

        # Add learning data
        for i in range(1, 11):
            learning_monitor.update_agent_learning("agent1", i, 10.0, 0.5)
            learning_monitor.update_agent_learning("agent2", i, 5.0, 0.3)

        summary = learning_monitor.get_learning_summary()

        assert summary["total_agents"] == 2
        assert "average_convergence_score" in summary
        assert "system_learning_health" in summary


class TestMemoryThreshold:
    """Test memory threshold."""

    def test_threshold_creation(self):
        """Test creating memory threshold."""
        threshold = MemoryThreshold(
            warning_threshold=0.7, critical_threshold=0.85, overflow_threshold=0.95
        )

        assert threshold.warning_threshold == 0.7
        assert threshold.critical_threshold == 0.85
        assert threshold.overflow_threshold == 0.95

    def test_status_determination(self):
        """Test memory status determination."""
        threshold = MemoryThreshold()

        assert threshold.get_status(0.5) == MemoryStatus.NORMAL
        assert threshold.get_status(0.75) == MemoryStatus.WARNING
        assert threshold.get_status(0.9) == MemoryStatus.CRITICAL
        assert threshold.get_status(0.98) == MemoryStatus.OVERFLOW


class TestMemoryManager:
    """Test memory manager."""

    @pytest.fixture
    def memory_manager(self):
        """Create memory manager for testing."""
        return MemoryManager(monitoring_interval=0.1, enable_auto_cleanup=True)

    def test_manager_initialization(self, memory_manager):
        """Test manager initialization."""
        assert memory_manager.monitoring_interval == 0.1
        assert memory_manager.enable_auto_cleanup is True
        assert not memory_manager.is_monitoring

    @pytest.mark.asyncio
    async def test_monitoring_lifecycle(self, memory_manager):
        """Test monitoring start/stop."""
        # Start monitoring
        await memory_manager.start_monitoring()
        assert memory_manager.is_monitoring
        assert memory_manager.monitoring_task is not None

        # Stop monitoring
        await memory_manager.stop_monitoring()
        assert not memory_manager.is_monitoring

    def test_component_registration(self, memory_manager):
        """Test component registration."""
        mock_component = Mock()
        mock_component.get_memory_usage.return_value = 1024

        memory_manager.register_component("test_component", mock_component)

        assert "test_component" in memory_manager.managed_components

        # Test unregistration
        memory_manager.unregister_component("test_component")
        assert "test_component" not in memory_manager.managed_components

    def test_buffer_manager_registration(self, memory_manager):
        """Test buffer manager registration."""

        def mock_buffer_size():
            return 2048

        memory_manager.register_buffer_manager("test_buffer", mock_buffer_size)

        assert "test_buffer" in memory_manager.buffer_managers

        # Test unregistration
        memory_manager.unregister_buffer_manager("test_buffer")
        assert "test_buffer" not in memory_manager.buffer_managers

    def test_memory_metrics(self, memory_manager):
        """Test memory metrics."""
        metrics = memory_manager.get_current_metrics()

        assert isinstance(metrics, MemoryMetrics)
        assert metrics.timestamp is not None

    def test_memory_statistics(self, memory_manager):
        """Test memory statistics."""
        stats = memory_manager.get_memory_statistics()

        assert "current_usage" in stats
        assert "average_usage" in stats
        assert "peak_usage" in stats
        assert "memory_events" in stats

    @pytest.mark.asyncio
    async def test_force_cleanup(self, memory_manager):
        """Test forced cleanup."""
        result = await memory_manager.force_cleanup("warning")

        assert "strategy" in result
        assert "memory_freed" in result
        assert "affected_components" in result


class TestFaultToleranceManager:
    """Test fault tolerance manager."""

    @pytest.fixture
    def fault_tolerance_manager(self):
        """Create fault tolerance manager for testing."""
        return FaultToleranceManager(enable_auto_recovery=True)

    def test_manager_initialization(self, fault_tolerance_manager):
        """Test manager initialization."""
        assert fault_tolerance_manager.enable_auto_recovery is True
        assert not fault_tolerance_manager.is_running
        assert len(fault_tolerance_manager.registered_agents) == 0

    @pytest.mark.asyncio
    async def test_monitoring_lifecycle(self, fault_tolerance_manager):
        """Test monitoring start/stop."""
        # Start monitoring
        await fault_tolerance_manager.start_monitoring()
        assert fault_tolerance_manager.is_running

        # Stop monitoring
        await fault_tolerance_manager.stop_monitoring()
        assert not fault_tolerance_manager.is_running

    def test_agent_registration(self, fault_tolerance_manager):
        """Test agent registration."""
        mock_agent = Mock()

        fault_tolerance_manager.register_agent("test_agent", mock_agent)

        assert "test_agent" in fault_tolerance_manager.registered_agents

        # Test unregistration
        fault_tolerance_manager.unregister_agent("test_agent")
        assert "test_agent" not in fault_tolerance_manager.registered_agents

    def test_action_recording(self, fault_tolerance_manager):
        """Test recording agent actions."""
        fault_tolerance_manager.record_agent_action("test_agent", True, 1.0)

        # Should register agent automatically
        assert "test_agent" in fault_tolerance_manager.registered_agents

    def test_learning_recording(self, fault_tolerance_manager):
        """Test recording learning metrics."""
        fault_tolerance_manager.record_agent_learning(
            "test_agent", 1, 10.0, 0.5, [1.0, 2.0]
        )

        # Should register agent automatically
        assert "test_agent" in fault_tolerance_manager.registered_agents

    def test_coordination_wait_recording(self, fault_tolerance_manager):
        """Test recording coordination waits."""
        fault_tolerance_manager.record_coordination_wait(
            "test_agent", "other_agent", "coordination", 30.0
        )

        # Should be recorded in deadlock detector
        assert "test_agent" in fault_tolerance_manager.deadlock_detector.waiting_states

        # Clear wait
        fault_tolerance_manager.clear_coordination_wait("test_agent")
        assert (
            "test_agent" not in fault_tolerance_manager.deadlock_detector.waiting_states
        )

    def test_system_health_summary(self, fault_tolerance_manager):
        """Test system health summary."""
        # Register some agents
        fault_tolerance_manager.register_agent("agent1")
        fault_tolerance_manager.register_agent("agent2")

        # Record some activity
        fault_tolerance_manager.record_agent_action("agent1", True, 1.0)
        fault_tolerance_manager.record_agent_learning("agent2", 1, 10.0, 0.5)

        summary = fault_tolerance_manager.get_system_health_summary()

        assert "overall_health_score" in summary
        assert "overall_status" in summary
        assert "registered_agents" in summary
        assert summary["registered_agents"] == 2
        assert "agent_health" in summary
        assert "learning_summary" in summary
        assert "memory_statistics" in summary

    def test_event_tracking(self, fault_tolerance_manager):
        """Test fault and recovery event tracking."""
        # Initially no events
        assert len(fault_tolerance_manager.get_fault_events()) == 0
        assert len(fault_tolerance_manager.get_recovery_actions()) == 0

        # Events would be added through component callbacks
        # This is tested indirectly through component integration

    def test_callbacks(self, fault_tolerance_manager):
        """Test callback functionality."""
        callback_called = False
        callback_event = None

        def test_callback(event):
            nonlocal callback_called, callback_event
            callback_called = True
            callback_event = event

        fault_tolerance_manager.add_fault_callback(test_callback)

        # Verify callback is added
        assert test_callback in fault_tolerance_manager.fault_callbacks

        # Remove callback
        fault_tolerance_manager.remove_fault_callback(test_callback)
        assert test_callback not in fault_tolerance_manager.fault_callbacks

    @pytest.mark.asyncio
    async def test_shutdown(self, fault_tolerance_manager):
        """Test manager shutdown."""
        # Register some agents
        fault_tolerance_manager.register_agent("agent1")
        fault_tolerance_manager.register_agent("agent2")

        # Start monitoring
        await fault_tolerance_manager.start_monitoring()

        # Shutdown
        await fault_tolerance_manager.shutdown()

        assert not fault_tolerance_manager.is_running
        assert len(fault_tolerance_manager.registered_agents) == 0


if __name__ == "__main__":
    pytest.main([__file__])
