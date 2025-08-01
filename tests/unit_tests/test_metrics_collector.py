"""
Unit tests for MARL Metrics Collector.

Tests the specialized metrics collection capabilities including
agent metrics, coordination metrics, and system resource metrics.
"""

# Standard Library
import asyncio
import time
from unittest.mock import Mock, patch

# Third-Party Library
import pytest

# SynThesisAI Modules
from core.marl.monitoring.metrics_collector import (
    CollectedMetric,
    CollectionStrategy,
    MetricDefinition,
    MetricsCollector,
    MetricsCollectorFactory,
)


class TestMetricDefinition:
    """Test metric definition data structure."""

    def test_basic_definition(self):
        """Test basic metric definition creation."""
        definition = MetricDefinition(
            name="test_metric", collection_strategy=CollectionStrategy.CONTINUOUS
        )

        assert definition.name == "test_metric"
        assert definition.collection_strategy == CollectionStrategy.CONTINUOUS
        assert definition.collection_interval == 1.0
        assert definition.aggregation_window == 100
        assert definition.collector_function is None
        assert definition.metadata == {}

    def test_custom_definition(self):
        """Test custom metric definition creation."""
        collector_func = Mock()
        metadata = {"type": "performance"}

        definition = MetricDefinition(
            name="custom_metric",
            collection_strategy=CollectionStrategy.PERIODIC,
            collection_interval=5.0,
            aggregation_window=200,
            collector_function=collector_func,
            metadata=metadata,
        )

        assert definition.name == "custom_metric"
        assert definition.collection_strategy == CollectionStrategy.PERIODIC
        assert definition.collection_interval == 5.0
        assert definition.aggregation_window == 200
        assert definition.collector_function == collector_func
        assert definition.metadata == metadata


class TestCollectedMetric:
    """Test collected metric data structure."""

    def test_metric_creation(self):
        """Test collected metric creation."""
        timestamp = time.time()
        metadata = {"source": "test"}

        metric = CollectedMetric(
            name="test_metric",
            value=0.75,
            timestamp=timestamp,
            source="test_source",
            metadata=metadata,
        )

        assert metric.name == "test_metric"
        assert metric.value == 0.75
        assert metric.timestamp == timestamp
        assert metric.source == "test_source"
        assert metric.metadata == metadata


class TestMetricsCollector:
    """Test metrics collector."""

    @pytest.fixture
    def metrics_collector(self):
        """Create metrics collector for testing."""
        return MetricsCollector()

    def test_initialization(self, metrics_collector):
        """Test metrics collector initialization."""
        assert len(metrics_collector._metrics) == 0
        assert metrics_collector._collection_active is False
        assert len(metrics_collector._collection_tasks) == 0
        assert len(metrics_collector._metric_definitions) == 0
        assert len(metrics_collector._event_handlers) == 0

    def test_register_metric(self, metrics_collector):
        """Test registering a metric definition."""
        definition = MetricDefinition(
            name="test_metric", collection_strategy=CollectionStrategy.CONTINUOUS
        )

        metrics_collector.register_metric(definition)

        assert "test_metric" in metrics_collector._metric_definitions
        assert metrics_collector._metric_definitions["test_metric"] == definition

    def test_register_agent_metrics(self, metrics_collector):
        """Test registering agent-specific metrics."""
        agent_id = "test_agent"

        metrics_collector.register_agent_metrics(agent_id)

        # Check that agent metrics were registered
        expected_metrics = [
            f"agent_{agent_id}_reward",
            f"agent_{agent_id}_loss",
            f"agent_{agent_id}_epsilon",
            f"agent_{agent_id}_q_values",
        ]

        for metric_name in expected_metrics:
            assert metric_name in metrics_collector._metric_definitions

        # Check different collection strategies
        reward_def = metrics_collector._metric_definitions[f"agent_{agent_id}_reward"]
        assert reward_def.collection_strategy == CollectionStrategy.EVENT_DRIVEN

        epsilon_def = metrics_collector._metric_definitions[f"agent_{agent_id}_epsilon"]
        assert epsilon_def.collection_strategy == CollectionStrategy.PERIODIC
        assert epsilon_def.collection_interval == 5.0

    def test_register_coordination_metrics(self, metrics_collector):
        """Test registering coordination metrics."""
        metrics_collector.register_coordination_metrics()

        expected_metrics = [
            "coordination_success_rate",
            "consensus_time",
            "communication_overhead",
        ]

        for metric_name in expected_metrics:
            assert metric_name in metrics_collector._metric_definitions

        # Check collection strategies
        success_rate_def = metrics_collector._metric_definitions["coordination_success_rate"]
        assert success_rate_def.collection_strategy == CollectionStrategy.CONTINUOUS
        assert success_rate_def.collection_interval == 10.0

        consensus_def = metrics_collector._metric_definitions["consensus_time"]
        assert consensus_def.collection_strategy == CollectionStrategy.EVENT_DRIVEN

    def test_register_system_metrics(self, metrics_collector):
        """Test registering system metrics."""
        metrics_collector.register_system_metrics()

        expected_metrics = ["cpu_usage", "memory_usage", "gpu_usage"]

        for metric_name in expected_metrics:
            assert metric_name in metrics_collector._metric_definitions

        # Check collection strategies and intervals
        cpu_def = metrics_collector._metric_definitions["cpu_usage"]
        assert cpu_def.collection_strategy == CollectionStrategy.CONTINUOUS
        assert cpu_def.collection_interval == 2.0

        gpu_def = metrics_collector._metric_definitions["gpu_usage"]
        assert gpu_def.collection_interval == 5.0

    @pytest.mark.asyncio
    async def test_start_stop_collection(self, metrics_collector):
        """Test starting and stopping metrics collection."""
        # Register a continuous metric
        definition = MetricDefinition(
            name="test_continuous",
            collection_strategy=CollectionStrategy.CONTINUOUS,
            collection_interval=0.1,
            collector_function=lambda: 0.5,
        )
        metrics_collector.register_metric(definition)

        # Start collection
        await metrics_collector.start_collection()
        assert metrics_collector._collection_active is True
        assert len(metrics_collector._collection_tasks) == 1

        # Let it collect some data
        await asyncio.sleep(0.25)

        # Stop collection
        await metrics_collector.stop_collection()
        assert metrics_collector._collection_active is False
        assert len(metrics_collector._collection_tasks) == 0

    @pytest.mark.asyncio
    async def test_start_collection_already_active(self, metrics_collector):
        """Test starting collection when already active."""
        # Register a metric
        definition = MetricDefinition(
            name="test_metric",
            collection_strategy=CollectionStrategy.CONTINUOUS,
            collector_function=lambda: 0.5,
        )
        metrics_collector.register_metric(definition)

        await metrics_collector.start_collection()

        # Try to start again
        with patch.object(metrics_collector.logger, "warning") as mock_warning:
            await metrics_collector.start_collection()
            mock_warning.assert_called_once()

        await metrics_collector.stop_collection()

    def test_collect_metric_manual(self, metrics_collector):
        """Test manually collecting a metric."""
        metric_name = "test_metric"
        value = 0.75
        source = "manual_test"
        metadata = {"test": "data"}

        metrics_collector.collect_metric(metric_name, value, source, metadata)

        # Check metric was stored
        assert metric_name in metrics_collector._metrics
        metrics = metrics_collector._metrics[metric_name]
        assert len(metrics) == 1

        metric = metrics[0]
        assert metric.name == metric_name
        assert metric.value == value
        assert metric.source == source
        assert metric.metadata == metadata

    def test_collect_metric_with_event_handler(self, metrics_collector):
        """Test collecting metric with event handler."""
        metric_name = "test_metric"
        handler_mock = Mock()

        # Add event handler
        metrics_collector.add_event_handler(metric_name, handler_mock)

        # Collect metric
        metrics_collector.collect_metric(metric_name, 0.5)

        # Check handler was called
        handler_mock.assert_called_once()
        call_args = handler_mock.call_args[0]
        assert call_args[0].name == metric_name
        assert call_args[0].value == 0.5

    def test_collect_metric_handler_exception(self, metrics_collector):
        """Test collecting metric with handler that raises exception."""
        metric_name = "test_metric"
        handler_mock = Mock(side_effect=Exception("Handler error"))

        # Add event handler
        metrics_collector.add_event_handler(metric_name, handler_mock)

        # Collect metric (should not raise exception)
        with patch.object(metrics_collector.logger, "error") as mock_error:
            metrics_collector.collect_metric(metric_name, 0.5)
            mock_error.assert_called_once()

    def test_get_metric_history_empty(self, metrics_collector):
        """Test getting metric history for non-existent metric."""
        history = metrics_collector.get_metric_history("non_existent")
        assert history == []

    def test_get_metric_history_with_data(self, metrics_collector):
        """Test getting metric history with data."""
        metric_name = "test_metric"

        # Collect some metrics
        for i in range(5):
            metrics_collector.collect_metric(metric_name, i * 0.1)

        # Get all history
        history = metrics_collector.get_metric_history(metric_name)
        assert len(history) == 5
        assert history[0].value == 0.0
        assert history[4].value == 0.4

        # Get limited history
        limited_history = metrics_collector.get_metric_history(metric_name, count=3)
        assert len(limited_history) == 3
        assert limited_history[0].value == 0.2  # Last 3 values
        assert limited_history[2].value == 0.4

    def test_get_metric_statistics_no_data(self, metrics_collector):
        """Test getting statistics for metric with no data."""
        stats = metrics_collector.get_metric_statistics("non_existent")
        assert "error" in stats
        assert stats["error"] == "No data available"

    def test_get_metric_statistics_no_numeric_data(self, metrics_collector):
        """Test getting statistics for metric with no numeric data."""
        metric_name = "test_metric"

        # Collect non-numeric data
        metrics_collector.collect_metric(metric_name, "string_value")
        metrics_collector.collect_metric(metric_name, {"dict": "value"})

        stats = metrics_collector.get_metric_statistics(metric_name)
        assert "error" in stats
        assert stats["error"] == "No numeric data available"

    def test_get_metric_statistics_with_numeric_data(self, metrics_collector):
        """Test getting statistics for metric with numeric data."""
        metric_name = "test_metric"
        values = [0.1, 0.2, 0.3, 0.4, 0.5]

        # Collect numeric data
        for value in values:
            metrics_collector.collect_metric(metric_name, value)

        stats = metrics_collector.get_metric_statistics(metric_name)

        assert stats["count"] == 5
        assert stats["mean"] == 0.3
        assert stats["min"] == 0.1
        assert stats["max"] == 0.5
        assert stats["median"] == 0.3
        assert stats["latest"] == 0.5
        assert stats["std"] > 0
        assert stats["time_span_seconds"] >= 0

    def test_get_metric_statistics_with_boolean_data(self, metrics_collector):
        """Test getting statistics for metric with boolean data."""
        metric_name = "test_metric"

        # Collect boolean data
        metrics_collector.collect_metric(metric_name, True)
        metrics_collector.collect_metric(metric_name, False)
        metrics_collector.collect_metric(metric_name, True)

        stats = metrics_collector.get_metric_statistics(metric_name)

        assert stats["count"] == 3
        assert stats["mean"] == 2.0 / 3.0  # True=1.0, False=0.0
        assert stats["min"] == 0.0
        assert stats["max"] == 1.0

    def test_get_metric_statistics_time_window(self, metrics_collector):
        """Test getting statistics with time window filter."""
        metric_name = "test_metric"

        # Collect old data
        metrics_collector.collect_metric(metric_name, 0.1)
        # Manually set old timestamp
        metrics_collector._metrics[metric_name][0].timestamp = time.time() - 3600

        # Collect recent data
        metrics_collector.collect_metric(metric_name, 0.9)

        # Get statistics for last 30 minutes
        stats = metrics_collector.get_metric_statistics(metric_name, time_window_seconds=1800)

        assert stats["count"] == 1
        assert stats["latest"] == 0.9

    @pytest.mark.asyncio
    async def test_continuous_collection_loop(self, metrics_collector):
        """Test continuous collection loop."""
        call_count = 0

        def mock_collector():
            nonlocal call_count
            call_count += 1
            return call_count * 0.1

        definition = MetricDefinition(
            name="test_continuous",
            collection_strategy=CollectionStrategy.CONTINUOUS,
            collection_interval=0.05,
            collector_function=mock_collector,
        )

        metrics_collector.register_metric(definition)

        # Start collection
        await metrics_collector.start_collection()

        # Let it run for a short time
        await asyncio.sleep(0.2)

        # Stop collection
        await metrics_collector.stop_collection()

        # Check that metrics were collected
        assert "test_continuous" in metrics_collector._metrics
        metrics = metrics_collector._metrics["test_continuous"]
        assert len(metrics) > 0
        assert call_count > 0

    @pytest.mark.asyncio
    async def test_periodic_collection_loop(self, metrics_collector):
        """Test periodic collection loop."""
        call_count = 0

        def mock_collector():
            nonlocal call_count
            call_count += 1
            return call_count * 0.2

        definition = MetricDefinition(
            name="test_periodic",
            collection_strategy=CollectionStrategy.PERIODIC,
            collection_interval=0.05,
            collector_function=mock_collector,
        )

        metrics_collector.register_metric(definition)

        # Start collection
        await metrics_collector.start_collection()

        # Let it run for a short time
        await asyncio.sleep(0.2)

        # Stop collection
        await metrics_collector.stop_collection()

        # Check that metrics were collected
        assert "test_periodic" in metrics_collector._metrics
        metrics = metrics_collector._metrics["test_periodic"]
        assert len(metrics) > 0
        assert call_count > 0

    @pytest.mark.asyncio
    async def test_collection_with_exception(self, metrics_collector):
        """Test collection loop with collector function exception."""

        def failing_collector():
            raise Exception("Collector error")

        definition = MetricDefinition(
            name="test_failing",
            collection_strategy=CollectionStrategy.CONTINUOUS,
            collection_interval=0.05,
            collector_function=failing_collector,
        )

        metrics_collector.register_metric(definition)

        # Start collection
        await metrics_collector.start_collection()

        # Let it run for a short time
        await asyncio.sleep(0.1)

        # Stop collection
        await metrics_collector.stop_collection()

        # Should not have collected any metrics due to exceptions
        assert len(metrics_collector._metrics.get("test_failing", [])) == 0

    @patch("psutil.cpu_percent")
    def test_collect_cpu_usage(self, mock_cpu_percent, metrics_collector):
        """Test CPU usage collection."""
        mock_cpu_percent.return_value = 75.5

        cpu_usage = metrics_collector._collect_cpu_usage()

        assert cpu_usage == 75.5
        mock_cpu_percent.assert_called_once_with(interval=0.1)

    @patch("psutil.virtual_memory")
    def test_collect_memory_usage(self, mock_virtual_memory, metrics_collector):
        """Test memory usage collection."""
        mock_memory = Mock()
        mock_memory.percent = 60.0
        mock_virtual_memory.return_value = mock_memory

        memory_usage = metrics_collector._collect_memory_usage()

        assert memory_usage == 0.6  # Converted to fraction

    @patch("torch.cuda.is_available")
    @patch("torch.cuda.memory_allocated")
    @patch("torch.cuda.memory_reserved")
    @patch("torch.cuda.get_device_properties")
    def test_collect_gpu_usage_available(
        self,
        mock_get_device_props,
        mock_memory_reserved,
        mock_memory_allocated,
        mock_cuda_available,
        metrics_collector,
    ):
        """Test GPU usage collection when CUDA is available."""
        mock_cuda_available.return_value = True
        mock_memory_allocated.return_value = 1024 * 1024 * 1024  # 1GB
        mock_memory_reserved.return_value = 512 * 1024 * 1024  # 512MB

        mock_device_props = Mock()
        mock_device_props.total_memory = 4 * 1024 * 1024 * 1024  # 4GB
        mock_get_device_props.return_value = mock_device_props

        gpu_usage = metrics_collector._collect_gpu_usage()

        expected_usage = (1024 + 512) / (4 * 1024)  # (1GB + 512MB) / 4GB
        assert gpu_usage == expected_usage

    @patch("torch.cuda.is_available")
    def test_collect_gpu_usage_not_available(self, mock_cuda_available, metrics_collector):
        """Test GPU usage collection when CUDA is not available."""
        mock_cuda_available.return_value = False

        gpu_usage = metrics_collector._collect_gpu_usage()

        assert gpu_usage is None

    def test_get_all_metrics_summary(self, metrics_collector):
        """Test getting summary of all metrics."""
        # Collect some test data
        metrics_collector.collect_metric("metric1", 0.5)
        metrics_collector.collect_metric("metric2", 0.7)

        summary = metrics_collector.get_all_metrics_summary()

        assert "total_metrics" in summary
        assert "collection_active" in summary
        assert "active_tasks" in summary
        assert "metrics" in summary

        assert summary["total_metrics"] == 2
        assert summary["collection_active"] is False
        assert summary["active_tasks"] == 0
        assert "metric1" in summary["metrics"]
        assert "metric2" in summary["metrics"]

    def test_reset_metrics(self, metrics_collector):
        """Test resetting all metrics."""
        # Collect some data
        metrics_collector.collect_metric("test_metric", 0.5)

        assert len(metrics_collector._metrics) == 1

        # Reset
        metrics_collector.reset_metrics()

        # Check metrics are cleared
        assert len(metrics_collector._metrics) == 0

    def test_export_metrics_data(self, metrics_collector):
        """Test exporting metrics data."""
        # Collect some test data
        metrics_collector.collect_metric("metric1", 0.5, "test_source", {"key": "value"})
        metrics_collector.collect_metric("metric2", 0.7)

        export_data = metrics_collector.export_metrics_data()

        assert "timestamp" in export_data
        assert "metrics" in export_data
        assert "metric1" in export_data["metrics"]
        assert "metric2" in export_data["metrics"]

        # Check metric1 data
        metric1_data = export_data["metrics"]["metric1"]
        assert len(metric1_data) == 1
        assert metric1_data[0]["name"] == "metric1"
        assert metric1_data[0]["value"] == 0.5
        assert metric1_data[0]["source"] == "test_source"
        assert metric1_data[0]["metadata"] == {"key": "value"}


class TestMetricsCollectorFactory:
    """Test metrics collector factory."""

    def test_create_standard_collector(self):
        """Test creating standard collector."""
        collector = MetricsCollectorFactory.create_standard_collector()

        assert isinstance(collector, MetricsCollector)

        # Check that standard metrics are registered
        expected_metrics = [
            "cpu_usage",
            "memory_usage",
            "gpu_usage",
            "coordination_success_rate",
            "consensus_time",
            "communication_overhead",
        ]

        for metric_name in expected_metrics:
            assert metric_name in collector._metric_definitions

    def test_create_agent_collector(self):
        """Test creating agent-specific collector."""
        agent_ids = ["agent1", "agent2", "agent3"]
        collector = MetricsCollectorFactory.create_agent_collector(agent_ids)

        assert isinstance(collector, MetricsCollector)

        # Check that agent metrics are registered for each agent
        for agent_id in agent_ids:
            expected_agent_metrics = [
                f"agent_{agent_id}_reward",
                f"agent_{agent_id}_loss",
                f"agent_{agent_id}_epsilon",
                f"agent_{agent_id}_q_values",
            ]

            for metric_name in expected_agent_metrics:
                assert metric_name in collector._metric_definitions

        # Check that standard metrics are also registered
        assert "cpu_usage" in collector._metric_definitions
        assert "coordination_success_rate" in collector._metric_definitions


@pytest.mark.asyncio
async def test_async_collector_function():
    """Test metrics collection with async collector function."""
    collector = MetricsCollector()

    async def async_collector():
        await asyncio.sleep(0.01)
        return 0.42

    definition = MetricDefinition(
        name="async_metric",
        collection_strategy=CollectionStrategy.CONTINUOUS,
        collection_interval=0.05,
        collector_function=async_collector,
    )

    collector.register_metric(definition)

    # Start collection
    await collector.start_collection()

    # Let it run for a short time
    await asyncio.sleep(0.15)

    # Stop collection
    await collector.stop_collection()

    # Check that metrics were collected
    assert "async_metric" in collector._metrics
    metrics = collector._metrics["async_metric"]
    assert len(metrics) > 0
    assert all(m.value == 0.42 for m in metrics)
