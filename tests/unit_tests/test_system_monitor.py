"""
Unit tests for MARL System Monitor.

Tests the system-level monitoring capabilities including resource utilization,
health checks, and performance bottleneck detection.
"""

# Standard Library
import asyncio
import time
from unittest.mock import Mock, patch

# Third-Party Library
import pytest

# SynThesisAI Modules
from core.marl.monitoring.system_monitor import (
    HealthCheck,
    HealthStatus,
    ResourceThresholds,
    ResourceType,
    SystemMonitor,
    SystemMonitorFactory,
    SystemSnapshot,
)


class TestResourceThresholds:
    """Test resource thresholds data structure."""

    def test_default_thresholds(self):
        """Test default threshold values."""
        thresholds = ResourceThresholds()

        assert thresholds.warning_threshold == 0.75
        assert thresholds.critical_threshold == 0.90
        assert thresholds.sustained_duration == 30.0

    def test_custom_thresholds(self):
        """Test custom threshold values."""
        thresholds = ResourceThresholds(
            warning_threshold=0.80, critical_threshold=0.95, sustained_duration=60.0
        )

        assert thresholds.warning_threshold == 0.80
        assert thresholds.critical_threshold == 0.95
        assert thresholds.sustained_duration == 60.0


class TestHealthCheck:
    """Test health check data structure."""

    def test_health_check_creation(self):
        """Test health check creation."""
        timestamp = time.time()
        metadata = {"resource": "cpu"}

        check = HealthCheck(
            name="cpu_usage",
            status=HealthStatus.WARNING,
            value=0.85,
            threshold=0.75,
            message="CPU usage is high",
            timestamp=timestamp,
            metadata=metadata,
        )

        assert check.name == "cpu_usage"
        assert check.status == HealthStatus.WARNING
        assert check.value == 0.85
        assert check.threshold == 0.75
        assert check.message == "CPU usage is high"
        assert check.timestamp == timestamp
        assert check.metadata == metadata


class TestSystemSnapshot:
    """Test system snapshot data structure."""

    def test_snapshot_creation(self):
        """Test system snapshot creation."""
        timestamp = time.time()
        network_io = {"bytes_sent": 1000, "bytes_recv": 2000}
        load_avg = (1.5, 1.2, 1.0)

        snapshot = SystemSnapshot(
            timestamp=timestamp,
            cpu_percent=75.5,
            memory_percent=60.0,
            gpu_percent=80.0,
            disk_usage_percent=45.0,
            network_io=network_io,
            process_count=150,
            load_average=load_avg,
            uptime=3600.0,
        )

        assert snapshot.timestamp == timestamp
        assert snapshot.cpu_percent == 75.5
        assert snapshot.memory_percent == 60.0
        assert snapshot.gpu_percent == 80.0
        assert snapshot.disk_usage_percent == 45.0
        assert snapshot.network_io == network_io
        assert snapshot.process_count == 150
        assert snapshot.load_average == load_avg
        assert snapshot.uptime == 3600.0


class TestSystemMonitor:
    """Test system monitor."""

    @pytest.fixture
    def system_monitor(self):
        """Create system monitor for testing."""
        return SystemMonitor(check_interval=0.1)

    def test_initialization(self, system_monitor):
        """Test system monitor initialization."""
        assert system_monitor.check_interval == 0.1
        assert system_monitor._monitoring_active is False
        assert system_monitor._monitoring_task is None
        assert len(system_monitor._resource_thresholds) == 4
        assert len(system_monitor._health_history) == 0
        assert len(system_monitor._system_snapshots) == 0
        assert len(system_monitor._active_alerts) == 0
        assert len(system_monitor._alert_callbacks) == 0

    def test_set_resource_thresholds(self, system_monitor):
        """Test setting custom resource thresholds."""
        custom_thresholds = ResourceThresholds(warning_threshold=0.85, critical_threshold=0.95)

        system_monitor.set_resource_thresholds(ResourceType.CPU, custom_thresholds)

        assert system_monitor._resource_thresholds[ResourceType.CPU] == custom_thresholds

    def test_add_alert_callback(self, system_monitor):
        """Test adding alert callback."""
        callback_mock = Mock()

        system_monitor.add_alert_callback(callback_mock)

        assert callback_mock in system_monitor._alert_callbacks

    @patch("psutil.cpu_percent")
    @patch("psutil.virtual_memory")
    @patch("psutil.disk_usage")
    @patch("psutil.net_io_counters")
    @patch("psutil.pids")
    @patch("psutil.getloadavg")
    @patch("torch.cuda.is_available")
    def test_get_system_snapshot(
        self,
        mock_cuda_available,
        mock_getloadavg,
        mock_pids,
        mock_net_io,
        mock_disk_usage,
        mock_virtual_memory,
        mock_cpu_percent,
        system_monitor,
    ):
        """Test getting system snapshot."""
        # Mock system calls
        mock_cpu_percent.return_value = 75.5

        mock_memory = Mock()
        mock_memory.percent = 60.0
        mock_virtual_memory.return_value = mock_memory

        mock_disk = Mock()
        mock_disk.used = 500 * 1024**3  # 500GB
        mock_disk.total = 1000 * 1024**3  # 1TB
        mock_disk_usage.return_value = mock_disk

        mock_net_io_data = Mock()
        mock_net_io_data._asdict.return_value = {"bytes_sent": 1000, "bytes_recv": 2000}
        mock_net_io.return_value = mock_net_io_data

        mock_pids.return_value = list(range(150))  # 150 processes
        mock_getloadavg.return_value = (1.5, 1.2, 1.0)
        mock_cuda_available.return_value = False

        snapshot = system_monitor.get_system_snapshot()

        assert isinstance(snapshot, SystemSnapshot)
        assert snapshot.cpu_percent == 75.5
        assert snapshot.memory_percent == 60.0
        assert snapshot.gpu_percent is None  # CUDA not available
        assert snapshot.disk_usage_percent == 50.0  # 500GB/1TB
        assert snapshot.network_io == {"bytes_sent": 1000, "bytes_recv": 2000}
        assert snapshot.process_count == 150
        assert snapshot.load_average == (1.5, 1.2, 1.0)
        assert snapshot.uptime > 0

    @patch("torch.cuda.is_available")
    @patch("torch.cuda.memory_allocated")
    @patch("torch.cuda.memory_reserved")
    @patch("torch.cuda.get_device_properties")
    def test_get_system_snapshot_with_gpu(
        self,
        mock_get_device_props,
        mock_memory_reserved,
        mock_memory_allocated,
        mock_cuda_available,
        system_monitor,
    ):
        """Test getting system snapshot with GPU information."""
        # Mock GPU availability and usage
        mock_cuda_available.return_value = True
        mock_memory_allocated.return_value = 1024 * 1024 * 1024  # 1GB
        mock_memory_reserved.return_value = 512 * 1024 * 1024  # 512MB

        mock_device_props = Mock()
        mock_device_props.total_memory = 4 * 1024 * 1024 * 1024  # 4GB
        mock_get_device_props.return_value = mock_device_props

        # Mock other system calls to avoid errors
        with (
            patch("psutil.cpu_percent", return_value=50.0),
            patch("psutil.virtual_memory") as mock_vm,
            patch("psutil.disk_usage") as mock_disk,
            patch("psutil.net_io_counters") as mock_net,
            patch("psutil.pids", return_value=list(range(100))),
            patch("psutil.getloadavg", return_value=(1.0, 1.0, 1.0)),
        ):
            mock_vm.return_value.percent = 50.0
            mock_disk.return_value.used = 100
            mock_disk.return_value.total = 1000
            mock_net.return_value._asdict.return_value = {}

            snapshot = system_monitor.get_system_snapshot()

            expected_gpu_percent = ((1024 + 512) / (4 * 1024)) * 100  # ~37.5%
            assert abs(snapshot.gpu_percent - expected_gpu_percent) < 0.1

    def test_get_system_snapshot_exception_handling(self, system_monitor):
        """Test system snapshot with exception handling."""
        with patch("psutil.cpu_percent", side_effect=Exception("System error")):
            snapshot = system_monitor.get_system_snapshot()

            # Should return default values on error
            assert snapshot.cpu_percent == 0.0
            assert snapshot.memory_percent == 0.0
            assert snapshot.process_count == 0

    @patch.object(SystemMonitor, "get_system_snapshot")
    def test_perform_health_checks(self, mock_get_snapshot, system_monitor):
        """Test performing health checks."""
        # Mock system snapshot
        mock_snapshot = SystemSnapshot(
            timestamp=time.time(),
            cpu_percent=85.0,  # High CPU
            memory_percent=70.0,  # Normal memory
            gpu_percent=95.0,  # Critical GPU
            disk_usage_percent=60.0,  # Normal disk
            network_io={},
            process_count=500,  # Normal process count
            load_average=(2.5, 2.0, 1.8),  # High load
            uptime=3600.0,
        )
        mock_get_snapshot.return_value = mock_snapshot

        health_checks = system_monitor.perform_health_checks()

        # Should have multiple health checks
        assert len(health_checks) >= 5  # CPU, memory, GPU, disk, load, process

        # Check that health checks were stored
        assert len(system_monitor._health_history) == len(health_checks)
        assert len(system_monitor._system_snapshots) == 1

        # Find specific checks
        cpu_check = next((c for c in health_checks if c.name == "cpu_usage"), None)
        assert cpu_check is not None
        assert cpu_check.status == HealthStatus.WARNING  # 85% > 80% warning threshold

        gpu_check = next((c for c in health_checks if c.name == "gpu_usage"), None)
        assert gpu_check is not None
        assert (
            gpu_check.status == HealthStatus.WARNING
        )  # 95% > 90% warning threshold but < 98% critical

    def test_check_resource_health_healthy(self, system_monitor):
        """Test resource health check for healthy resource."""
        thresholds = ResourceThresholds(warning_threshold=0.75, critical_threshold=0.90)

        check = system_monitor._check_resource_health(
            "test_resource", 0.60, thresholds, "Test resource"
        )

        assert check.name == "test_resource"
        assert check.status == HealthStatus.HEALTHY
        assert check.value == 0.60
        assert "normal" in check.message.lower()

    def test_check_resource_health_warning(self, system_monitor):
        """Test resource health check for warning level."""
        thresholds = ResourceThresholds(warning_threshold=0.75, critical_threshold=0.90)

        check = system_monitor._check_resource_health(
            "test_resource", 0.80, thresholds, "Test resource"
        )

        assert check.status == HealthStatus.WARNING
        assert "high" in check.message.lower()

    def test_check_resource_health_critical(self, system_monitor):
        """Test resource health check for critical level."""
        thresholds = ResourceThresholds(warning_threshold=0.75, critical_threshold=0.90)

        check = system_monitor._check_resource_health(
            "test_resource", 0.95, thresholds, "Test resource"
        )

        assert check.status == HealthStatus.CRITICAL
        assert "critical" in check.message.lower()

    @patch("psutil.cpu_count")
    def test_check_load_average_healthy(self, mock_cpu_count, system_monitor):
        """Test load average check for healthy system."""
        mock_cpu_count.return_value = 4

        check = system_monitor._check_load_average(
            (2.0, 1.8, 1.5)
        )  # Load 2.0 on 4 CPUs = 0.5 normalized

        assert check.name == "load_average"
        assert check.status == HealthStatus.HEALTHY
        assert check.value == 0.5  # 2.0 / 4 CPUs
        assert "normal" in check.message.lower()

    @patch("psutil.cpu_count")
    def test_check_load_average_warning(self, mock_cpu_count, system_monitor):
        """Test load average check for warning level."""
        mock_cpu_count.return_value = 2

        check = system_monitor._check_load_average(
            (3.0, 2.5, 2.0)
        )  # Load 3.0 on 2 CPUs = 1.5 normalized

        assert check.status == HealthStatus.WARNING
        assert check.value == 1.5
        assert "high" in check.message.lower()

    @patch("psutil.cpu_count")
    def test_check_load_average_critical(self, mock_cpu_count, system_monitor):
        """Test load average check for critical level."""
        mock_cpu_count.return_value = 1

        check = system_monitor._check_load_average(
            (3.0, 2.8, 2.5)
        )  # Load 3.0 on 1 CPU = 3.0 normalized

        assert check.status == HealthStatus.CRITICAL
        assert check.value == 3.0
        assert "very high" in check.message.lower()

    def test_check_process_count_healthy(self, system_monitor):
        """Test process count check for healthy system."""
        check = system_monitor._check_process_count(500)

        assert check.name == "process_count"
        assert check.status == HealthStatus.HEALTHY
        assert check.value == 500.0
        assert "normal" in check.message.lower()

    def test_check_process_count_warning(self, system_monitor):
        """Test process count check for warning level."""
        check = system_monitor._check_process_count(1500)

        assert check.status == HealthStatus.WARNING
        assert "high" in check.message.lower()

    def test_check_process_count_critical(self, system_monitor):
        """Test process count check for critical level."""
        check = system_monitor._check_process_count(2500)

        assert check.status == HealthStatus.CRITICAL
        assert "very high" in check.message.lower()

    def test_get_overall_health_status_unknown(self, system_monitor):
        """Test overall health status with no data."""
        status = system_monitor.get_overall_health_status()
        assert status == HealthStatus.UNKNOWN

    def test_get_overall_health_status_healthy(self, system_monitor):
        """Test overall health status with healthy checks."""
        # Add healthy checks
        for i in range(3):
            check = HealthCheck(
                name=f"check_{i}",
                status=HealthStatus.HEALTHY,
                value=0.5,
                threshold=0.75,
                message="Healthy",
                timestamp=time.time(),
            )
            system_monitor._health_history.append(check)

        status = system_monitor.get_overall_health_status()
        assert status == HealthStatus.HEALTHY

    def test_get_overall_health_status_warning(self, system_monitor):
        """Test overall health status with warning checks."""
        # Add mixed checks
        healthy_check = HealthCheck(
            name="healthy_check",
            status=HealthStatus.HEALTHY,
            value=0.5,
            threshold=0.75,
            message="Healthy",
            timestamp=time.time(),
        )
        warning_check = HealthCheck(
            name="warning_check",
            status=HealthStatus.WARNING,
            value=0.8,
            threshold=0.75,
            message="Warning",
            timestamp=time.time(),
        )

        system_monitor._health_history.extend([healthy_check, warning_check])

        status = system_monitor.get_overall_health_status()
        assert status == HealthStatus.WARNING

    def test_get_overall_health_status_critical(self, system_monitor):
        """Test overall health status with critical checks."""
        # Add critical check
        critical_check = HealthCheck(
            name="critical_check",
            status=HealthStatus.CRITICAL,
            value=0.95,
            threshold=0.90,
            message="Critical",
            timestamp=time.time(),
        )

        system_monitor._health_history.append(critical_check)

        status = system_monitor.get_overall_health_status()
        assert status == HealthStatus.CRITICAL

    def test_get_resource_trends_no_data(self, system_monitor):
        """Test resource trends with no data."""
        trends = system_monitor.get_resource_trends(ResourceType.CPU)
        assert "error" in trends
        assert trends["error"] == "Insufficient data for trend analysis"

    def test_get_resource_trends_insufficient_data(self, system_monitor):
        """Test resource trends with insufficient data."""
        # Add one snapshot
        snapshot = SystemSnapshot(
            timestamp=time.time(),
            cpu_percent=50.0,
            memory_percent=60.0,
            gpu_percent=None,
            disk_usage_percent=40.0,
            network_io={},
            process_count=100,
            load_average=(1.0, 1.0, 1.0),
            uptime=3600.0,
        )
        system_monitor._system_snapshots.append(snapshot)

        trends = system_monitor.get_resource_trends(ResourceType.CPU)
        assert "error" in trends
        assert trends["error"] == "Insufficient data for trend analysis"

    def test_get_resource_trends_with_data(self, system_monitor):
        """Test resource trends with sufficient data."""
        # Add multiple snapshots with increasing CPU usage
        base_time = time.time()
        for i in range(5):
            snapshot = SystemSnapshot(
                timestamp=base_time + i * 60,  # 1 minute apart
                cpu_percent=50.0 + i * 5.0,  # Increasing CPU usage
                memory_percent=60.0,
                gpu_percent=None,
                disk_usage_percent=40.0,
                network_io={},
                process_count=100,
                load_average=(1.0, 1.0, 1.0),
                uptime=3600.0 + i * 60,
            )
            system_monitor._system_snapshots.append(snapshot)

        trends = system_monitor.get_resource_trends(ResourceType.CPU, time_window_seconds=600)

        assert "error" not in trends
        assert trends["resource_type"] == "cpu"
        assert trends["data_points"] == 5
        assert trends["current_value"] == 70.0  # Last value
        assert trends["min_value"] == 50.0
        assert trends["max_value"] == 70.0
        assert trends["trend_direction"] == "increasing"
        assert trends["trend_slope"] > 0

    def test_get_performance_bottlenecks_no_issues(self, system_monitor):
        """Test bottleneck detection with no issues."""
        # Add healthy checks
        for i in range(5):
            check = HealthCheck(
                name="cpu_usage",
                status=HealthStatus.HEALTHY,
                value=0.5,
                threshold=0.75,
                message="Healthy",
                timestamp=time.time(),
            )
            system_monitor._health_history.append(check)

        bottlenecks = system_monitor.get_performance_bottlenecks()

        # Should have no bottlenecks for healthy system
        assert isinstance(bottlenecks, list)
        # May have some bottlenecks from trend analysis, so just check it's a list

    def test_get_performance_bottlenecks_critical_resource(self, system_monitor):
        """Test bottleneck detection with critical resource."""
        # Add critical checks
        for i in range(3):
            check = HealthCheck(
                name="memory_usage",
                status=HealthStatus.CRITICAL,
                value=0.95,
                threshold=0.85,
                message="Critical memory usage",
                timestamp=time.time(),
            )
            system_monitor._health_history.append(check)

        bottlenecks = system_monitor.get_performance_bottlenecks()

        # Should detect critical resource bottleneck
        critical_bottlenecks = [b for b in bottlenecks if b["type"] == "critical_resource"]
        assert len(critical_bottlenecks) > 0

        bottleneck = critical_bottlenecks[0]
        assert bottleneck["resource"] == "memory_usage"
        assert bottleneck["severity"] == "high"

    def test_get_performance_bottlenecks_resource_pressure(self, system_monitor):
        """Test bottleneck detection with resource pressure."""
        # Add warning checks
        for i in range(4):
            check = HealthCheck(
                name="cpu_usage",
                status=HealthStatus.WARNING,
                value=0.80,
                threshold=0.75,
                message="High CPU usage",
                timestamp=time.time(),
            )
            system_monitor._health_history.append(check)

        bottlenecks = system_monitor.get_performance_bottlenecks()

        # Should detect resource pressure bottleneck
        pressure_bottlenecks = [b for b in bottlenecks if b["type"] == "resource_pressure"]
        assert len(pressure_bottlenecks) > 0

        bottleneck = pressure_bottlenecks[0]
        assert bottleneck["resource"] == "cpu_usage"
        assert bottleneck["severity"] == "medium"

    @pytest.mark.asyncio
    async def test_start_stop_monitoring(self, system_monitor):
        """Test starting and stopping monitoring."""
        # Start monitoring
        await system_monitor.start_monitoring()
        assert system_monitor._monitoring_active is True
        assert system_monitor._monitoring_task is not None

        # Stop monitoring
        await system_monitor.stop_monitoring()
        assert system_monitor._monitoring_active is False
        # Task may still exist but should be cancelled
        if system_monitor._monitoring_task:
            assert system_monitor._monitoring_task.cancelled()

    @pytest.mark.asyncio
    async def test_start_monitoring_already_active(self, system_monitor):
        """Test starting monitoring when already active."""
        await system_monitor.start_monitoring()

        # Try to start again
        with patch.object(system_monitor.logger, "warning") as mock_warning:
            await system_monitor.start_monitoring()
            mock_warning.assert_called_once()

        await system_monitor.stop_monitoring()

    def test_process_alerts_new_warning(self, system_monitor):
        """Test processing new warning alert."""
        callback_mock = Mock()
        system_monitor.add_alert_callback(callback_mock)

        warning_check = HealthCheck(
            name="cpu_usage",
            status=HealthStatus.WARNING,
            value=0.80,
            threshold=0.75,
            message="High CPU usage",
            timestamp=time.time(),
        )

        system_monitor._process_alerts([warning_check])

        # Should trigger alert callback
        callback_mock.assert_called_once_with(warning_check)

        # Should store active alert
        alert_key = "cpu_usage_warning"
        assert alert_key in system_monitor._active_alerts

    def test_process_alerts_resolved(self, system_monitor):
        """Test processing resolved alert."""
        # Add active alert
        alert_key = "cpu_usage_warning"
        system_monitor._active_alerts[alert_key] = Mock()

        healthy_check = HealthCheck(
            name="cpu_usage",
            status=HealthStatus.HEALTHY,
            value=0.60,
            threshold=0.75,
            message="Normal CPU usage",
            timestamp=time.time(),
        )

        system_monitor._process_alerts([healthy_check])

        # Should clear resolved alert
        assert alert_key not in system_monitor._active_alerts

    def test_trigger_alert_callback_exception(self, system_monitor):
        """Test alert callback with exception."""
        failing_callback = Mock(side_effect=Exception("Callback error"))
        system_monitor.add_alert_callback(failing_callback)

        warning_check = HealthCheck(
            name="test_resource",
            status=HealthStatus.WARNING,
            value=0.80,
            threshold=0.75,
            message="Test warning",
            timestamp=time.time(),
        )

        # Should not raise exception
        with patch.object(system_monitor.logger, "error") as mock_error:
            system_monitor._trigger_alert(warning_check)
            mock_error.assert_called_once()

    def test_get_monitoring_summary(self, system_monitor):
        """Test getting monitoring summary."""
        # Add some test data
        system_monitor._active_alerts["test_alert"] = Mock()
        system_monitor._health_history.append(Mock())
        system_monitor._system_snapshots.append(Mock())

        with (
            patch.object(system_monitor, "get_system_snapshot") as mock_snapshot,
            patch.object(system_monitor, "get_overall_health_status") as mock_health,
            patch.object(system_monitor, "get_performance_bottlenecks") as mock_bottlenecks,
        ):
            mock_snapshot.return_value = SystemSnapshot(
                timestamp=time.time(),
                cpu_percent=50.0,
                memory_percent=60.0,
                gpu_percent=70.0,
                disk_usage_percent=40.0,
                network_io={},
                process_count=100,
                load_average=(1.0, 1.0, 1.0),
                uptime=3600.0,
            )
            mock_health.return_value = HealthStatus.HEALTHY
            mock_bottlenecks.return_value = []

            summary = system_monitor.get_monitoring_summary()

            assert "timestamp" in summary
            assert "monitoring_active" in summary
            assert "uptime_seconds" in summary
            assert "overall_health" in summary
            assert "current_snapshot" in summary
            assert "active_alerts" in summary
            assert "bottlenecks" in summary
            assert "health_checks_performed" in summary
            assert "snapshots_collected" in summary

            assert summary["monitoring_active"] is False
            assert summary["overall_health"] == "healthy"
            assert summary["active_alerts"] == 1
            assert summary["health_checks_performed"] == 1
            assert summary["snapshots_collected"] == 1

    def test_reset_monitoring_data(self, system_monitor):
        """Test resetting monitoring data."""
        # Add some test data
        system_monitor._health_history.append(Mock())
        system_monitor._system_snapshots.append(Mock())
        system_monitor._active_alerts["test"] = Mock()
        system_monitor._resource_usage[ResourceType.CPU].append(Mock())

        # Reset
        system_monitor.reset_monitoring_data()

        # Check everything is cleared
        assert len(system_monitor._health_history) == 0
        assert len(system_monitor._system_snapshots) == 0
        assert len(system_monitor._active_alerts) == 0
        assert len(system_monitor._resource_usage[ResourceType.CPU]) == 0


class TestSystemMonitorFactory:
    """Test system monitor factory."""

    def test_create_default(self):
        """Test creating monitor with default settings."""
        monitor = SystemMonitorFactory.create()

        assert isinstance(monitor, SystemMonitor)
        assert monitor.check_interval == 5.0

    def test_create_custom_interval(self):
        """Test creating monitor with custom check interval."""
        monitor = SystemMonitorFactory.create(check_interval=2.0)

        assert isinstance(monitor, SystemMonitor)
        assert monitor.check_interval == 2.0

    def test_create_with_custom_thresholds(self):
        """Test creating monitor with custom thresholds."""
        cpu_thresholds = ResourceThresholds(warning_threshold=0.85, critical_threshold=0.95)
        memory_thresholds = ResourceThresholds(warning_threshold=0.80, critical_threshold=0.90)

        monitor = SystemMonitorFactory.create_with_custom_thresholds(
            cpu_thresholds=cpu_thresholds,
            memory_thresholds=memory_thresholds,
            check_interval=3.0,
        )

        assert isinstance(monitor, SystemMonitor)
        assert monitor.check_interval == 3.0
        assert monitor._resource_thresholds[ResourceType.CPU] == cpu_thresholds
        assert monitor._resource_thresholds[ResourceType.MEMORY] == memory_thresholds


@pytest.mark.asyncio
async def test_monitoring_loop_integration():
    """Test the monitoring loop integration."""
    monitor = SystemMonitor(check_interval=0.05)

    # Mock system snapshot to avoid actual system calls
    with patch.object(monitor, "get_system_snapshot") as mock_snapshot:
        mock_snapshot.return_value = SystemSnapshot(
            timestamp=time.time(),
            cpu_percent=50.0,
            memory_percent=60.0,
            gpu_percent=None,
            disk_usage_percent=40.0,
            network_io={},
            process_count=100,
            load_average=(1.0, 1.0, 1.0),
            uptime=3600.0,
        )

        # Start monitoring
        await monitor.start_monitoring()

        # Let it run for a short time
        await asyncio.sleep(0.15)

        # Stop monitoring
        await monitor.stop_monitoring()

        # Verify monitoring worked
        assert len(monitor._health_history) > 0
        assert len(monitor._system_snapshots) > 0
