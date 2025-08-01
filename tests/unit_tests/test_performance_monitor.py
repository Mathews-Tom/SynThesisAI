"""
Unit tests for MARL Performance Monitor.

Tests the comprehensive performance monitoring capabilities including
coordination success tracking, agent performance metrics, and system monitoring.
"""

import asyncio
import time
from unittest.mock import Mock, patch

import pytest

from core.marl.monitoring.performance_monitor import (
    CoordinationEvent,
    MARLPerformanceMonitor,
    MARLPerformanceMonitorFactory,
    MetricType,
    MonitoringConfig,
    PerformanceMetric,
)


class TestMonitoringConfig:
    """Test monitoring configuration."""

    def test_default_initialization(self):
        """Test default configuration initialization."""
        config = MonitoringConfig()

        assert config.metrics_window_size == 1000
        assert config.coordination_timeout == 30.0
        assert config.performance_threshold == 0.85
        assert config.monitoring_interval == 1.0
        assert config.enable_detailed_logging is True
        assert config.metrics_retention_hours == 24
        assert "coordination_success_rate" in config.alert_thresholds
        assert config.alert_thresholds["coordination_success_rate"] == 0.85

    def test_custom_initialization(self):
        """Test custom configuration initialization."""
        custom_thresholds = {"coordination_success_rate": 0.90, "custom_metric": 0.75}

        config = MonitoringConfig(
            metrics_window_size=500,
            coordination_timeout=60.0,
            performance_threshold=0.90,
            monitoring_interval=2.0,
            enable_detailed_logging=False,
            metrics_retention_hours=12,
            alert_thresholds=custom_thresholds,
        )

        assert config.metrics_window_size == 500
        assert config.coordination_timeout == 60.0
        assert config.performance_threshold == 0.90
        assert config.monitoring_interval == 2.0
        assert config.enable_detailed_logging is False
        assert config.metrics_retention_hours == 12
        assert config.alert_thresholds == custom_thresholds


class TestPerformanceMetric:
    """Test performance metric data structure."""

    def test_metric_creation(self):
        """Test performance metric creation."""
        timestamp = time.time()
        metadata = {"test": "value"}

        metric = PerformanceMetric(
            timestamp=timestamp,
            metric_type=MetricType.AGENT_PERFORMANCE,
            agent_id="test_agent",
            value=0.85,
            metadata=metadata,
        )

        assert metric.timestamp == timestamp
        assert metric.metric_type == MetricType.AGENT_PERFORMANCE
        assert metric.agent_id == "test_agent"
        assert metric.value == 0.85
        assert metric.metadata == metadata


class TestCoordinationEvent:
    """Test coordination event data structure."""

    def test_event_creation(self):
        """Test coordination event creation."""
        timestamp = time.time()
        agents = ["agent1", "agent2"]
        metadata = {"request_type": "generation"}

        event = CoordinationEvent(
            timestamp=timestamp,
            event_type="coordination_request",
            agents_involved=agents,
            success=True,
            duration=2.5,
            metadata=metadata,
        )

        assert event.timestamp == timestamp
        assert event.event_type == "coordination_request"
        assert event.agents_involved == agents
        assert event.success is True
        assert event.duration == 2.5
        assert event.metadata == metadata


class TestMARLPerformanceMonitor:
    """Test MARL performance monitor."""

    @pytest.fixture
    def monitor_config(self):
        """Create test monitoring configuration."""
        return MonitoringConfig(
            metrics_window_size=100,
            monitoring_interval=0.1,
            enable_detailed_logging=False,
        )

    @pytest.fixture
    def performance_monitor(self, monitor_config):
        """Create performance monitor for testing."""
        return MARLPerformanceMonitor(monitor_config)

    def test_initialization(self, performance_monitor):
        """Test performance monitor initialization."""
        assert performance_monitor.config.metrics_window_size == 100
        assert performance_monitor._monitoring_active is False
        assert performance_monitor._monitoring_task is None
        assert len(performance_monitor._metrics) == len(MetricType)
        assert len(performance_monitor._coordination_events) == 0
        assert len(performance_monitor._active_coordinations) == 0

    @pytest.mark.asyncio
    async def test_start_stop_monitoring(self, performance_monitor):
        """Test starting and stopping monitoring."""
        # Start monitoring
        await performance_monitor.start_monitoring()
        assert performance_monitor._monitoring_active is True
        assert performance_monitor._monitoring_task is not None

        # Stop monitoring
        await performance_monitor.stop_monitoring()
        assert performance_monitor._monitoring_active is False
        # Task may still exist but should be cancelled
        if performance_monitor._monitoring_task:
            assert performance_monitor._monitoring_task.cancelled()

    @pytest.mark.asyncio
    async def test_start_monitoring_already_active(self, performance_monitor):
        """Test starting monitoring when already active."""
        await performance_monitor.start_monitoring()

        # Try to start again
        with patch.object(performance_monitor.logger, "warning") as mock_warning:
            await performance_monitor.start_monitoring()
            mock_warning.assert_called_once()

        await performance_monitor.stop_monitoring()

    def test_record_coordination_start(self, performance_monitor):
        """Test recording coordination start."""
        coordination_id = "test_coord_1"
        agents = ["agent1", "agent2"]
        metadata = {"type": "generation"}

        performance_monitor.record_coordination_start(coordination_id, agents, metadata)

        assert coordination_id in performance_monitor._active_coordinations
        event = performance_monitor._active_coordinations[coordination_id]
        assert event.agents_involved == agents
        assert event.metadata == metadata
        assert event.success is False  # Not completed yet

    def test_record_coordination_end_success(self, performance_monitor):
        """Test recording successful coordination end."""
        coordination_id = "test_coord_1"
        agents = ["agent1", "agent2"]

        # Start coordination
        performance_monitor.record_coordination_start(coordination_id, agents)

        # End coordination
        metadata = {"result": "success"}
        performance_monitor.record_coordination_end(coordination_id, True, metadata)

        # Check that coordination is moved to completed events
        assert coordination_id not in performance_monitor._active_coordinations
        assert len(performance_monitor._coordination_events) == 1

        event = performance_monitor._coordination_events[0]
        assert event.success is True
        assert event.duration > 0
        assert "result" in event.metadata

        # Check that success metric was recorded
        success_metrics = performance_monitor._metrics[MetricType.COORDINATION_SUCCESS]
        assert len(success_metrics) == 1
        assert success_metrics[0].value is True

    def test_record_coordination_end_failure(self, performance_monitor):
        """Test recording failed coordination end."""
        coordination_id = "test_coord_1"
        agents = ["agent1", "agent2"]

        # Start coordination
        performance_monitor.record_coordination_start(coordination_id, agents)

        # End coordination with failure
        performance_monitor.record_coordination_end(coordination_id, False)

        event = performance_monitor._coordination_events[0]
        assert event.success is False

        # Check that failure metric was recorded
        success_metrics = performance_monitor._metrics[MetricType.COORDINATION_SUCCESS]
        assert len(success_metrics) == 1
        assert success_metrics[0].value is False

    def test_record_coordination_end_unknown_id(self, performance_monitor):
        """Test recording coordination end with unknown ID."""
        with patch.object(performance_monitor.logger, "warning") as mock_warning:
            performance_monitor.record_coordination_end("unknown_id", True)
            mock_warning.assert_called_once()

    def test_record_metric(self, performance_monitor):
        """Test recording individual metrics."""
        performance_monitor.record_metric(
            MetricType.AGENT_PERFORMANCE,
            0.85,
            agent_id="test_agent",
            metadata={"metric_name": "reward"},
        )

        # Check metric was stored
        metrics = performance_monitor._metrics[MetricType.AGENT_PERFORMANCE]
        assert len(metrics) == 1

        metric = metrics[0]
        assert metric.metric_type == MetricType.AGENT_PERFORMANCE
        assert metric.value == 0.85
        assert metric.agent_id == "test_agent"
        assert metric.metadata["metric_name"] == "reward"

        # Check agent-specific storage
        agent_metrics = performance_monitor._agent_metrics["test_agent"]
        assert len(agent_metrics[MetricType.AGENT_PERFORMANCE.value]) == 1

    def test_record_agent_performance(self, performance_monitor):
        """Test recording agent performance metrics."""
        agent_id = "test_agent"
        reward = 0.75
        loss = 0.25
        confidence = 0.90
        metadata = {"episode": 100}

        performance_monitor.record_agent_performance(agent_id, reward, loss, confidence, metadata)

        # Check that all metrics were recorded
        perf_metrics = performance_monitor._metrics[MetricType.AGENT_PERFORMANCE]
        learning_metrics = performance_monitor._metrics[MetricType.LEARNING_PROGRESS]

        # Should have 2 performance metrics (reward + confidence)
        assert len(perf_metrics) == 2
        # Should have 1 learning metric (loss)
        assert len(learning_metrics) == 1

        # Check agent-specific metrics
        agent_metrics = performance_monitor._agent_metrics[agent_id]
        assert len(agent_metrics[MetricType.AGENT_PERFORMANCE.value]) == 2
        assert len(agent_metrics[MetricType.LEARNING_PROGRESS.value]) == 1

    def test_record_system_metric(self, performance_monitor):
        """Test recording system metrics."""
        metric_name = "cpu_usage"
        value = 0.75

        performance_monitor.record_system_metric(metric_name, value)

        # Check system metrics storage
        assert metric_name in performance_monitor._system_metrics
        system_metrics = performance_monitor._system_metrics[metric_name]
        assert len(system_metrics) == 1
        assert system_metrics[0]["value"] == value

        # Check general metrics storage
        sys_metrics = performance_monitor._metrics[MetricType.SYSTEM_PERFORMANCE]
        assert len(sys_metrics) == 1
        assert sys_metrics[0].value == value

    def test_get_coordination_success_rate_empty(self, performance_monitor):
        """Test coordination success rate with no data."""
        success_rate = performance_monitor.get_coordination_success_rate()
        assert success_rate == 0.0

    def test_get_coordination_success_rate_with_data(self, performance_monitor):
        """Test coordination success rate calculation."""
        # Record some coordination events
        for i in range(5):
            coord_id = f"coord_{i}"
            performance_monitor.record_coordination_start(coord_id, ["agent1"])
            success = i < 3  # First 3 succeed, last 2 fail
            performance_monitor.record_coordination_end(coord_id, success)

        success_rate = performance_monitor.get_coordination_success_rate()
        assert success_rate == 0.6  # 3/5 = 0.6

    def test_get_coordination_success_rate_time_window(self, performance_monitor):
        """Test coordination success rate with time window."""
        # Record old event
        old_coord = "old_coord"
        performance_monitor.record_coordination_start(old_coord, ["agent1"])
        performance_monitor.record_coordination_end(old_coord, False)

        # Manually set old timestamp
        performance_monitor._coordination_events[0].timestamp = time.time() - 3600

        # Record recent event
        recent_coord = "recent_coord"
        performance_monitor.record_coordination_start(recent_coord, ["agent1"])
        performance_monitor.record_coordination_end(recent_coord, True)

        # Get success rate for last 30 minutes
        success_rate = performance_monitor.get_coordination_success_rate(1800)
        assert success_rate == 1.0  # Only recent successful event

    def test_get_agent_performance_summary_unknown_agent(self, performance_monitor):
        """Test agent performance summary for unknown agent."""
        summary = performance_monitor.get_agent_performance_summary("unknown_agent")
        assert "error" in summary
        assert summary["error"] == "Agent not found"

    def test_get_agent_performance_summary_with_data(self, performance_monitor):
        """Test agent performance summary with data."""
        agent_id = "test_agent"

        # Record some performance data
        for i in range(5):
            performance_monitor.record_agent_performance(
                agent_id, reward=0.5 + i * 0.1, loss=0.5 - i * 0.05
            )

        summary = performance_monitor.get_agent_performance_summary(agent_id)

        # Check that summary contains expected metrics
        assert MetricType.AGENT_PERFORMANCE.value in summary
        assert MetricType.LEARNING_PROGRESS.value in summary

        # Check reward statistics (only rewards are recorded, not confidences since they're None)
        reward_stats = summary[MetricType.AGENT_PERFORMANCE.value]
        assert reward_stats["count"] == 5  # 5 rewards only
        assert reward_stats["min"] >= 0.5
        assert reward_stats["max"] <= 0.9

    def test_get_system_performance_summary(self, performance_monitor):
        """Test system performance summary."""
        # Record some system metrics
        performance_monitor.record_system_metric("cpu_usage", 0.75)
        performance_monitor.record_system_metric("memory_usage", 0.60)

        summary = performance_monitor.get_system_performance_summary()

        assert "uptime_seconds" in summary
        assert "coordination_success_rate" in summary
        assert "total_coordinations" in summary
        assert "active_coordinations" in summary
        assert "system_metrics" in summary

        # Check system metrics
        sys_metrics = summary["system_metrics"]
        assert "cpu_usage" in sys_metrics
        assert "memory_usage" in sys_metrics
        assert sys_metrics["cpu_usage"]["latest"] == 0.75
        assert sys_metrics["memory_usage"]["latest"] == 0.60

    def test_get_learning_progress_unknown_agent(self, performance_monitor):
        """Test learning progress for unknown agent."""
        progress = performance_monitor.get_learning_progress("unknown_agent")
        assert "error" in progress
        assert progress["error"] == "Agent not found"

    def test_get_learning_progress_no_data(self, performance_monitor):
        """Test learning progress with no learning data."""
        agent_id = "test_agent"
        # Record non-learning metric
        performance_monitor.record_metric(MetricType.AGENT_PERFORMANCE, 0.5, agent_id=agent_id)

        progress = performance_monitor.get_learning_progress(agent_id)
        assert "error" in progress
        assert progress["error"] == "No learning metrics available"

    def test_get_learning_progress_with_data(self, performance_monitor):
        """Test learning progress with learning data."""
        agent_id = "test_agent"

        # Record learning metrics with decreasing loss (improving)
        for i in range(10):
            loss = 1.0 - i * 0.05  # Decreasing loss
            performance_monitor.record_metric(MetricType.LEARNING_PROGRESS, loss, agent_id=agent_id)

        progress = performance_monitor.get_learning_progress(agent_id)

        assert "total_updates" in progress
        assert "recent_performance" in progress
        assert "latest_value" in progress
        assert "time_span_seconds" in progress

        assert progress["total_updates"] == 10
        assert progress["recent_performance"]["improving"]  # Decreasing trend
        assert progress["latest_value"] == 0.55  # Last loss value

    def test_check_alert_conditions_no_alerts(self, performance_monitor):
        """Test alert checking with no alert conditions."""
        alerts = performance_monitor.check_alert_conditions()
        assert isinstance(alerts, list)
        # May have alerts based on system metrics, so just check it's a list

    def test_check_alert_conditions_coordination_alert(self, performance_monitor):
        """Test alert checking with coordination success rate alert."""
        # Record failed coordinations to trigger alert
        for i in range(5):
            coord_id = f"coord_{i}"
            performance_monitor.record_coordination_start(coord_id, ["agent1"])
            performance_monitor.record_coordination_end(coord_id, False)  # All fail

        alerts = performance_monitor.check_alert_conditions()

        # Should have coordination success rate alert
        coord_alerts = [a for a in alerts if a["type"] == "coordination_success_rate"]
        assert len(coord_alerts) > 0

        alert = coord_alerts[0]
        assert alert["severity"] == "warning"
        assert alert["value"] == 0.0  # 0% success rate
        assert alert["threshold"] == 0.85

    def test_reset_metrics(self, performance_monitor):
        """Test resetting all metrics."""
        # Add some data
        performance_monitor.record_metric(MetricType.AGENT_PERFORMANCE, 0.5)
        performance_monitor.record_coordination_start("test", ["agent1"])
        performance_monitor.record_system_metric("cpu", 0.5)

        # Reset
        performance_monitor.reset_metrics()

        # Check everything is cleared
        assert len(performance_monitor._coordination_events) == 0
        assert len(performance_monitor._active_coordinations) == 0
        assert len(performance_monitor._agent_metrics) == 0
        assert len(performance_monitor._system_metrics) == 0

        for metric_deque in performance_monitor._metrics.values():
            assert len(metric_deque) == 0

    @patch("pathlib.Path.open")
    @patch("json.dump")
    @patch("pathlib.Path.mkdir")
    def test_export_metrics_success(
        self, mock_mkdir, mock_json_dump, mock_open, performance_monitor
    ):
        """Test successful metrics export."""
        # Add some test data
        performance_monitor.record_metric(MetricType.AGENT_PERFORMANCE, 0.5, agent_id="test_agent")

        # Mock the file context manager
        mock_file = Mock()
        mock_open.return_value.__enter__.return_value = mock_file

        result = performance_monitor.export_metrics("test_export.json")

        assert result is True
        mock_open.assert_called_once()
        mock_json_dump.assert_called_once()

    @patch("pathlib.Path.open", side_effect=Exception("File error"))
    def test_export_metrics_failure(self, mock_open, performance_monitor):
        """Test failed metrics export."""
        result = performance_monitor.export_metrics("test_export.json")

        assert result is False


class TestMARLPerformanceMonitorFactory:
    """Test performance monitor factory."""

    def test_create_default(self):
        """Test creating monitor with default configuration."""
        monitor = MARLPerformanceMonitorFactory.create()

        assert isinstance(monitor, MARLPerformanceMonitor)
        assert monitor.config.metrics_window_size == 1000
        assert monitor.config.monitoring_interval == 1.0

    def test_create_with_config(self):
        """Test creating monitor with custom configuration."""
        config = MonitoringConfig(metrics_window_size=500)
        monitor = MARLPerformanceMonitorFactory.create(config)

        assert isinstance(monitor, MARLPerformanceMonitor)
        assert monitor.config.metrics_window_size == 500

    def test_create_with_custom_config(self):
        """Test creating monitor with custom configuration parameters."""
        monitor = MARLPerformanceMonitorFactory.create_with_custom_config(
            metrics_window_size=200,
            coordination_timeout=45.0,
            performance_threshold=0.90,
            monitoring_interval=2.0,
        )

        assert isinstance(monitor, MARLPerformanceMonitor)
        assert monitor.config.metrics_window_size == 200
        assert monitor.config.coordination_timeout == 45.0
        assert monitor.config.performance_threshold == 0.90
        assert monitor.config.monitoring_interval == 2.0


@pytest.mark.asyncio
async def test_monitoring_loop_integration():
    """Test the monitoring loop integration."""
    config = MonitoringConfig(monitoring_interval=0.05, enable_detailed_logging=False)
    monitor = MARLPerformanceMonitor(config)

    # Start monitoring
    await monitor.start_monitoring()

    # Let it run for a short time
    await asyncio.sleep(0.2)

    # Add some data during monitoring
    monitor.record_coordination_start("test_coord", ["agent1"])
    monitor.record_coordination_end("test_coord", True)

    # Stop monitoring
    await monitor.stop_monitoring()

    # Verify monitoring worked
    assert len(monitor._coordination_events) == 1
    assert monitor._coordination_events[0].success is True
