"""
Unit tests for MARL Performance Analyzer.

Tests the comprehensive performance analysis capabilities including
trend analysis, performance insights, and report generation.
"""

import time
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from core.marl.monitoring.performance_analyzer import (
    AnalysisConfig,
    PerformanceAnalyzer,
    PerformanceAnalyzerFactory,
    PerformanceInsight,
    PerformanceLevel,
    PerformanceReport,
    TrendAnalysis,
    TrendDirection,
)
from core.marl.monitoring.performance_monitor import (
    MARLPerformanceMonitor,
    MetricType,
    PerformanceMetric,
)


class TestAnalysisConfig:
    """Test analysis configuration."""

    def test_default_initialization(self):
        """Test default configuration initialization."""
        config = AnalysisConfig()

        assert config.min_data_points == 10
        assert config.trend_confidence_threshold == 0.7
        assert config.volatility_threshold == 0.3
        assert config.enable_predictions is True
        assert config.prediction_horizon_seconds == 3600.0
        assert "coordination_success_rate" in config.performance_thresholds
        assert "system_health" in config.performance_thresholds
        assert "agent_performance" in config.performance_thresholds

    def test_custom_initialization(self):
        """Test custom configuration initialization."""
        custom_thresholds = {
            "coordination_success_rate": {
                "excellent": 0.98,
                "good": 0.90,
                "average": 0.75,
                "poor": 0.60,
            }
        }

        config = AnalysisConfig(
            min_data_points=20,
            trend_confidence_threshold=0.8,
            volatility_threshold=0.2,
            performance_thresholds=custom_thresholds,
            enable_predictions=False,
            prediction_horizon_seconds=7200.0,
        )

        assert config.min_data_points == 20
        assert config.trend_confidence_threshold == 0.8
        assert config.volatility_threshold == 0.2
        assert config.performance_thresholds == custom_thresholds
        assert config.enable_predictions is False
        assert config.prediction_horizon_seconds == 7200.0


class TestTrendAnalysis:
    """Test trend analysis data structure."""

    def test_trend_analysis_creation(self):
        """Test trend analysis creation."""
        trend = TrendAnalysis(
            metric_name="test_metric",
            direction=TrendDirection.IMPROVING,
            slope=0.05,
            confidence=0.85,
            r_squared=0.72,
            data_points=50,
            time_span_seconds=3600.0,
            current_value=0.75,
            predicted_value=0.80,
        )

        assert trend.metric_name == "test_metric"
        assert trend.direction == TrendDirection.IMPROVING
        assert trend.slope == 0.05
        assert trend.confidence == 0.85
        assert trend.r_squared == 0.72
        assert trend.data_points == 50
        assert trend.time_span_seconds == 3600.0
        assert trend.current_value == 0.75
        assert trend.predicted_value == 0.80


class TestPerformanceInsight:
    """Test performance insight data structure."""

    def test_insight_creation(self):
        """Test performance insight creation."""
        insight = PerformanceInsight(
            category="coordination",
            severity="warning",
            title="Low Success Rate",
            description="Coordination success rate is below optimal",
            recommendation="Review coordination policies",
            impact="Medium - System performance degraded",
            confidence=0.8,
            supporting_data={"success_rate": 0.65},
        )

        assert insight.category == "coordination"
        assert insight.severity == "warning"
        assert insight.title == "Low Success Rate"
        assert insight.description == "Coordination success rate is below optimal"
        assert insight.recommendation == "Review coordination policies"
        assert insight.impact == "Medium - System performance degraded"
        assert insight.confidence == 0.8
        assert insight.supporting_data == {"success_rate": 0.65}


class TestPerformanceAnalyzer:
    """Test performance analyzer."""

    @pytest.fixture
    def mock_performance_monitor(self):
        """Create mock performance monitor."""
        monitor = Mock(spec=MARLPerformanceMonitor)
        monitor._agent_metrics = {
            "agent1": {
                MetricType.AGENT_PERFORMANCE.value: [],
                MetricType.LEARNING_PROGRESS.value: [],
            },
            "agent2": {
                MetricType.AGENT_PERFORMANCE.value: [],
                MetricType.LEARNING_PROGRESS.value: [],
            },
        }
        monitor._metrics = {
            MetricType.COORDINATION_SUCCESS: [],
            MetricType.SYSTEM_PERFORMANCE: [],
        }
        return monitor

    @pytest.fixture
    def analysis_config(self):
        """Create test analysis configuration."""
        return AnalysisConfig(
            min_data_points=5, trend_confidence_threshold=0.6, enable_predictions=True
        )

    @pytest.fixture
    def performance_analyzer(self, mock_performance_monitor, analysis_config):
        """Create performance analyzer for testing."""
        return PerformanceAnalyzer(mock_performance_monitor, analysis_config)

    def test_initialization(
        self, performance_analyzer, mock_performance_monitor, analysis_config
    ):
        """Test performance analyzer initialization."""
        assert performance_analyzer.performance_monitor == mock_performance_monitor
        assert performance_analyzer.config == analysis_config
        assert len(performance_analyzer._trend_cache) == 0
        assert performance_analyzer._cache_timestamp == 0.0
        assert performance_analyzer._cache_ttl == 60.0

    def test_analyze_trend_insufficient_data(self, performance_analyzer):
        """Test trend analysis with insufficient data."""
        # Mock empty metric data
        performance_analyzer._get_system_metric_data = Mock(return_value=[])

        trend = performance_analyzer.analyze_trend(
            "test_metric", MetricType.COORDINATION_SUCCESS
        )

        assert trend.metric_name == "test_metric"
        assert trend.direction == TrendDirection.INSUFFICIENT_DATA
        assert trend.slope == 0.0
        assert trend.confidence == 0.0
        assert trend.data_points == 0

    def test_analyze_trend_with_data(self, performance_analyzer):
        """Test trend analysis with sufficient data."""
        # Create mock metric data with improving trend
        base_time = time.time()
        mock_metrics = []
        for i in range(10):
            metric = PerformanceMetric(
                timestamp=base_time + i * 60,
                metric_type=MetricType.COORDINATION_SUCCESS,
                agent_id=None,
                value=0.5 + i * 0.05,  # Improving trend
            )
            mock_metrics.append(metric)

        performance_analyzer._get_system_metric_data = Mock(return_value=mock_metrics)

        trend = performance_analyzer.analyze_trend(
            "coordination_success_rate", MetricType.COORDINATION_SUCCESS
        )

        assert trend.metric_name == "coordination_success_rate"
        assert trend.direction == TrendDirection.IMPROVING
        assert trend.slope > 0
        assert trend.data_points == 10
        assert trend.current_value == 0.95  # Last value

    def test_analyze_trend_declining(self, performance_analyzer):
        """Test trend analysis with declining trend."""
        # Create mock metric data with declining trend
        base_time = time.time()
        mock_metrics = []
        for i in range(10):
            metric = PerformanceMetric(
                timestamp=base_time + i * 60,
                metric_type=MetricType.COORDINATION_SUCCESS,
                agent_id=None,
                value=0.9 - i * 0.05,  # Declining trend
            )
            mock_metrics.append(metric)

        performance_analyzer._get_system_metric_data = Mock(return_value=mock_metrics)

        trend = performance_analyzer.analyze_trend(
            "coordination_success_rate", MetricType.COORDINATION_SUCCESS
        )

        assert trend.direction == TrendDirection.DECLINING
        assert trend.slope < 0

    def test_analyze_trend_stable(self, performance_analyzer):
        """Test trend analysis with stable trend."""
        # Create mock metric data with stable values
        base_time = time.time()
        mock_metrics = []
        for i in range(10):
            metric = PerformanceMetric(
                timestamp=base_time + i * 60,
                metric_type=MetricType.COORDINATION_SUCCESS,
                agent_id=None,
                value=0.75 + (i % 2) * 0.01,  # Stable with minor fluctuation
            )
            mock_metrics.append(metric)

        performance_analyzer._get_system_metric_data = Mock(return_value=mock_metrics)

        trend = performance_analyzer.analyze_trend(
            "coordination_success_rate", MetricType.COORDINATION_SUCCESS
        )

        assert trend.direction in [TrendDirection.STABLE, TrendDirection.VOLATILE]
        assert abs(trend.slope) < 0.01

    def test_analyze_trend_with_cache(self, performance_analyzer):
        """Test trend analysis caching."""
        # Mock metric data
        mock_metrics = [
            PerformanceMetric(
                timestamp=time.time() + i * 60,
                metric_type=MetricType.COORDINATION_SUCCESS,
                agent_id=None,
                value=0.5 + i * 0.05,
            )
            for i in range(10)
        ]

        performance_analyzer._get_system_metric_data = Mock(return_value=mock_metrics)

        # First call should compute trend
        trend1 = performance_analyzer.analyze_trend(
            "test_metric", MetricType.COORDINATION_SUCCESS
        )

        # Second call should use cache
        trend2 = performance_analyzer.analyze_trend(
            "test_metric", MetricType.COORDINATION_SUCCESS
        )

        assert trend1 == trend2
        # Should only call _get_system_metric_data once due to caching
        assert performance_analyzer._get_system_metric_data.call_count == 1

    def test_analyze_trend_agent_specific(self, performance_analyzer):
        """Test trend analysis for agent-specific metrics."""
        # Mock agent metric data
        mock_metrics = [
            PerformanceMetric(
                timestamp=time.time() + i * 60,
                metric_type=MetricType.AGENT_PERFORMANCE,
                agent_id="agent1",
                value=0.6 + i * 0.03,
            )
            for i in range(10)
        ]

        performance_analyzer._get_agent_metric_data = Mock(return_value=mock_metrics)

        trend = performance_analyzer.analyze_trend(
            "agent1_performance", MetricType.AGENT_PERFORMANCE, agent_id="agent1"
        )

        assert trend.direction == TrendDirection.IMPROVING
        performance_analyzer._get_agent_metric_data.assert_called_once_with(
            "agent1", MetricType.AGENT_PERFORMANCE, None
        )

    def test_generate_insights_coordination_critical(self, performance_analyzer):
        """Test insight generation for critical coordination performance."""
        # Mock critical coordination success rate
        performance_analyzer.performance_monitor.get_coordination_success_rate.return_value = 0.3

        # Mock trend analysis
        mock_trend = TrendAnalysis(
            metric_name="coordination_success_rate",
            direction=TrendDirection.DECLINING,
            slope=-0.1,
            confidence=0.8,
            r_squared=0.7,
            data_points=20,
            time_span_seconds=3600.0,
            current_value=0.3,
        )
        performance_analyzer.analyze_trend = Mock(return_value=mock_trend)

        # Mock other methods to return empty results
        performance_analyzer._analyze_agent_performance = Mock(return_value=[])
        performance_analyzer._analyze_system_performance = Mock(return_value=[])
        performance_analyzer._analyze_learning_progress = Mock(return_value=[])

        insights = performance_analyzer.generate_insights()

        # Should generate critical insight for low success rate
        critical_insights = [i for i in insights if i.severity == "critical"]
        assert len(critical_insights) > 0

        critical_insight = critical_insights[0]
        assert critical_insight.category == "coordination"
        assert "critical" in critical_insight.title.lower()
        assert critical_insight.confidence == 0.9

    def test_generate_insights_agent_disparity(self, performance_analyzer):
        """Test insight generation for agent performance disparity."""
        # Mock agent performance summaries with high disparity
        mock_summaries = {
            "agent1": {MetricType.AGENT_PERFORMANCE.value: {"mean": 0.9, "count": 10}},
            "agent2": {MetricType.AGENT_PERFORMANCE.value: {"mean": 0.4, "count": 10}},
        }

        # Set up the mock agent metrics properly
        performance_analyzer.performance_monitor._agent_metrics = {
            "agent1": {MetricType.AGENT_PERFORMANCE.value: []},
            "agent2": {MetricType.AGENT_PERFORMANCE.value: []},
        }

        performance_analyzer.performance_monitor.get_agent_performance_summary.side_effect = (
            lambda agent_id, _: mock_summaries.get(agent_id, {"error": "Not found"})
        )
        performance_analyzer.performance_monitor.get_learning_progress.return_value = {
            "error": "No data"
        }

        # Mock other analysis methods
        performance_analyzer._analyze_coordination_performance = Mock(return_value=[])
        performance_analyzer._analyze_system_performance = Mock(return_value=[])
        performance_analyzer._analyze_learning_progress = Mock(return_value=[])

        # Directly test the agent performance analysis method
        agent_insights = performance_analyzer._analyze_agent_performance(None)

        # Should generate insight about performance disparity
        disparity_insights = [
            i for i in agent_insights if "disparity" in i.title.lower()
        ]
        assert len(disparity_insights) > 0

        disparity_insight = disparity_insights[0]
        assert disparity_insight.category == "agent_performance"
        assert disparity_insight.severity == "warning"

    def test_generate_performance_report(self, performance_analyzer):
        """Test comprehensive performance report generation."""
        # Mock all required methods
        performance_analyzer.performance_monitor.get_coordination_success_rate.return_value = 0.85
        performance_analyzer.performance_monitor.get_system_performance_summary.return_value = {
            "system_metrics": {
                "cpu_usage": {"latest": 0.6},
                "memory_usage": {"latest": 0.7},
            },
            "total_coordinations": 100,
            "active_coordinations": 5,
            "uptime_seconds": 7200,
        }
        performance_analyzer.performance_monitor.get_agent_performance_summary.return_value = {
            MetricType.AGENT_PERFORMANCE.value: {"mean": 0.75}
        }

        # Mock trend analyses
        mock_trends = [
            TrendAnalysis(
                metric_name="coordination_success_rate",
                direction=TrendDirection.STABLE,
                slope=0.01,
                confidence=0.8,
                r_squared=0.7,
                data_points=20,
                time_span_seconds=3600.0,
                current_value=0.85,
            )
        ]
        performance_analyzer._generate_trend_analyses = Mock(return_value=mock_trends)

        # Mock insights
        mock_insights = [
            PerformanceInsight(
                category="system",
                severity="info",
                title="System Running Normally",
                description="All systems operational",
                recommendation="Continue monitoring",
                impact="Low",
                confidence=0.9,
            )
        ]
        performance_analyzer.generate_insights = Mock(return_value=mock_insights)

        report = performance_analyzer.generate_performance_report(3600.0)

        assert isinstance(report, PerformanceReport)
        assert report.coordination_success_rate == 0.85
        assert report.performance_level in [
            PerformanceLevel.GOOD,
            PerformanceLevel.AVERAGE,
        ]
        assert len(report.trend_analyses) == 1
        assert len(report.insights) == 1
        assert report.metadata["total_coordinations"] == 100

    def test_generate_performance_report_error_handling(self, performance_analyzer):
        """Test performance report generation with errors."""
        # Mock methods to raise exceptions
        performance_analyzer.performance_monitor.get_coordination_success_rate.side_effect = Exception(
            "Test error"
        )

        report = performance_analyzer.generate_performance_report()

        assert isinstance(report, PerformanceReport)
        assert report.performance_level == PerformanceLevel.CRITICAL
        assert report.overall_score == 0.0
        assert "System analysis failed" in report.recommendations[0]

    def test_calculate_system_health_score(self, performance_analyzer):
        """Test system health score calculation."""
        system_summary = {
            "system_metrics": {
                "cpu_usage": {"latest": 0.3},  # Good CPU usage
                "memory_usage": {"latest": 0.5},  # Good memory usage
            }
        }

        health_score = performance_analyzer._calculate_system_health_score(
            system_summary
        )

        # Should be high score for low resource usage
        assert health_score > 0.5
        assert health_score <= 1.0

    def test_calculate_system_health_score_no_data(self, performance_analyzer):
        """Test system health score with no data."""
        system_summary = {"system_metrics": {}}

        health_score = performance_analyzer._calculate_system_health_score(
            system_summary
        )

        assert health_score == 0.5  # Neutral score

    def test_calculate_overall_score(self, performance_analyzer):
        """Test overall performance score calculation."""
        coordination_success_rate = 0.9
        system_health_score = 0.8
        agent_performance_scores = {"agent1": 0.85, "agent2": 0.75}

        overall_score = performance_analyzer._calculate_overall_score(
            coordination_success_rate, system_health_score, agent_performance_scores
        )

        # Should be weighted average
        expected_score = 0.5 * 0.9 + 0.2 * 0.8 + 0.3 * 0.8  # 0.86
        assert abs(overall_score - expected_score) < 0.01
        assert 0.0 <= overall_score <= 1.0

    def test_classify_performance_level(self, performance_analyzer):
        """Test performance level classification."""
        assert (
            performance_analyzer._classify_performance_level(0.98)
            == PerformanceLevel.EXCELLENT
        )
        assert (
            performance_analyzer._classify_performance_level(0.88)
            == PerformanceLevel.GOOD
        )
        assert (
            performance_analyzer._classify_performance_level(0.72)
            == PerformanceLevel.AVERAGE
        )
        assert (
            performance_analyzer._classify_performance_level(0.55)
            == PerformanceLevel.POOR
        )
        assert (
            performance_analyzer._classify_performance_level(0.30)
            == PerformanceLevel.CRITICAL
        )

    def test_clear_cache(self, performance_analyzer):
        """Test cache clearing."""
        # Add something to cache
        performance_analyzer._trend_cache["test"] = Mock()
        performance_analyzer._cache_timestamp = time.time()

        performance_analyzer.clear_cache()

        assert len(performance_analyzer._trend_cache) == 0
        assert performance_analyzer._cache_timestamp == 0.0

    def test_get_agent_metric_data(self, performance_analyzer):
        """Test getting agent metric data."""
        # Mock agent metrics
        mock_metrics = [Mock(timestamp=time.time() - 100)]
        performance_analyzer.performance_monitor._agent_metrics = {
            "agent1": {MetricType.AGENT_PERFORMANCE.value: mock_metrics}
        }

        result = performance_analyzer._get_agent_metric_data(
            "agent1", MetricType.AGENT_PERFORMANCE, time_window_seconds=200
        )

        assert len(result) == 1
        assert result[0] == mock_metrics[0]

    def test_get_agent_metric_data_unknown_agent(self, performance_analyzer):
        """Test getting metric data for unknown agent."""
        result = performance_analyzer._get_agent_metric_data(
            "unknown_agent", MetricType.AGENT_PERFORMANCE, None
        )

        assert result == []

    def test_get_system_metric_data(self, performance_analyzer):
        """Test getting system metric data."""
        # Mock system metrics
        mock_metrics = [Mock(timestamp=time.time() - 100)]
        performance_analyzer.performance_monitor._metrics = {
            MetricType.COORDINATION_SUCCESS: mock_metrics
        }

        result = performance_analyzer._get_system_metric_data(
            MetricType.COORDINATION_SUCCESS, time_window_seconds=200
        )

        assert len(result) == 1
        assert result[0] == mock_metrics[0]

    def test_get_system_metric_data_time_filter(self, performance_analyzer):
        """Test getting system metric data with time filtering."""
        current_time = time.time()
        mock_metrics = [
            Mock(timestamp=current_time - 3600),  # 1 hour ago
            Mock(timestamp=current_time - 1800),  # 30 minutes ago
            Mock(timestamp=current_time - 600),  # 10 minutes ago
        ]
        performance_analyzer.performance_monitor._metrics = {
            MetricType.COORDINATION_SUCCESS: mock_metrics
        }

        # Get data from last 45 minutes
        result = performance_analyzer._get_system_metric_data(
            MetricType.COORDINATION_SUCCESS, time_window_seconds=2700
        )

        # Should only get the last 2 metrics
        assert len(result) == 2


class TestPerformanceAnalyzerFactory:
    """Test performance analyzer factory."""

    def test_create_default(self):
        """Test creating analyzer with default configuration."""
        mock_monitor = Mock(spec=MARLPerformanceMonitor)

        analyzer = PerformanceAnalyzerFactory.create(mock_monitor)

        assert isinstance(analyzer, PerformanceAnalyzer)
        assert analyzer.performance_monitor == mock_monitor
        assert isinstance(analyzer.config, AnalysisConfig)

    def test_create_with_config(self):
        """Test creating analyzer with custom configuration."""
        mock_monitor = Mock(spec=MARLPerformanceMonitor)
        config = AnalysisConfig(min_data_points=20)

        analyzer = PerformanceAnalyzerFactory.create(mock_monitor, config)

        assert isinstance(analyzer, PerformanceAnalyzer)
        assert analyzer.config == config
        assert analyzer.config.min_data_points == 20

    def test_create_with_custom_config(self):
        """Test creating analyzer with custom configuration parameters."""
        mock_monitor = Mock(spec=MARLPerformanceMonitor)

        analyzer = PerformanceAnalyzerFactory.create_with_custom_config(
            mock_monitor,
            min_data_points=15,
            trend_confidence_threshold=0.8,
            enable_predictions=False,
        )

        assert isinstance(analyzer, PerformanceAnalyzer)
        assert analyzer.config.min_data_points == 15
        assert analyzer.config.trend_confidence_threshold == 0.8
        assert analyzer.config.enable_predictions is False


@pytest.mark.integration
def test_trend_analysis_integration():
    """Integration test for trend analysis with real data."""
    # Create mock monitor with realistic data
    monitor = Mock(spec=MARLPerformanceMonitor)
    monitor._agent_metrics = {}
    monitor._metrics = {MetricType.COORDINATION_SUCCESS: []}

    analyzer = PerformanceAnalyzer(monitor)

    # Create realistic metric data with trend
    base_time = time.time()
    metrics = []
    for i in range(20):
        # Simulate improving coordination success rate with some noise
        base_value = 0.6 + i * 0.015  # Improving trend
        noise = np.random.normal(0, 0.02)  # Small random noise
        value = max(0.0, min(1.0, base_value + noise))

        metric = PerformanceMetric(
            timestamp=base_time + i * 300,  # Every 5 minutes
            metric_type=MetricType.COORDINATION_SUCCESS,
            agent_id=None,
            value=value,
        )
        metrics.append(metric)

    # Mock the data retrieval
    analyzer._get_system_metric_data = Mock(return_value=metrics)

    # Analyze trend
    trend = analyzer.analyze_trend(
        "coordination_success_rate",
        MetricType.COORDINATION_SUCCESS,
        time_window_seconds=6000,
    )

    # Should detect improving trend
    assert trend.direction == TrendDirection.IMPROVING
    assert trend.slope > 0
    assert trend.confidence > 0.3  # Lower threshold due to random noise
    assert trend.data_points == 20
    assert trend.current_value > 0.6
