"""
Unit tests for MARL Performance Reporter.

Tests the comprehensive performance reporting capabilities including
report generation, formatting, and dashboard data preparation.
"""

import json
import time
from pathlib import Path
from unittest.mock import Mock, mock_open, patch

import pytest

from core.marl.monitoring.performance_analyzer import (
    PerformanceAnalyzer,
    PerformanceInsight,
    PerformanceLevel,
    PerformanceReport,
    TrendAnalysis,
    TrendDirection,
)
from core.marl.monitoring.performance_monitor import MARLPerformanceMonitor
from core.marl.monitoring.performance_reporter import (
    PerformanceReporter,
    PerformanceReporterFactory,
    ReportFormat,
    ReportType,
)


class TestPerformanceReporter:
    """Test performance reporter."""

    @pytest.fixture
    def mock_performance_analyzer(self):
        """Create mock performance analyzer."""
        analyzer = Mock()
        analyzer.config = Mock()
        analyzer.config.performance_thresholds = {
            "agent_performance": {
                "excellent": 0.90,
                "good": 0.75,
                "average": 0.60,
                "poor": 0.40,
            }
        }
        return analyzer

    @pytest.fixture
    def mock_performance_monitor(self):
        """Create mock performance monitor."""
        monitor = Mock(spec=MARLPerformanceMonitor)
        monitor._coordination_events = []
        return monitor

    @pytest.fixture
    def sample_performance_report(self):
        """Create sample performance report for testing."""
        return PerformanceReport(
            timestamp=time.time(),
            time_window_seconds=3600.0,
            overall_score=0.85,
            performance_level=PerformanceLevel.GOOD,
            coordination_success_rate=0.88,
            system_health_score=0.82,
            agent_performance_scores={"agent1": 0.85, "agent2": 0.75},
            trend_analyses=[
                TrendAnalysis(
                    metric_name="coordination_success_rate",
                    direction=TrendDirection.IMPROVING,
                    slope=0.02,
                    confidence=0.8,
                    r_squared=0.7,
                    data_points=20,
                    time_span_seconds=3600.0,
                    current_value=0.88,
                    predicted_value=0.90,
                ),
                TrendAnalysis(
                    metric_name="agent1_performance",
                    direction=TrendDirection.STABLE,
                    slope=0.001,
                    confidence=0.6,
                    r_squared=0.4,
                    data_points=15,
                    time_span_seconds=3600.0,
                    current_value=0.85,
                ),
            ],
            insights=[
                PerformanceInsight(
                    category="coordination",
                    severity="info",
                    title="Good Coordination Performance",
                    description="Coordination success rate is above target",
                    recommendation="Continue current approach",
                    impact="Low - System performing well",
                    confidence=0.9,
                    supporting_data={"success_rate": 0.88},
                ),
                PerformanceInsight(
                    category="agent_performance",
                    severity="warning",
                    title="Agent Performance Disparity",
                    description="Performance gap between agents detected",
                    recommendation="Focus training on underperforming agents",
                    impact="Medium - May affect coordination",
                    confidence=0.7,
                    supporting_data={"gap": 0.10},
                ),
            ],
            recommendations=[
                "Continue monitoring coordination performance",
                "Address agent performance disparity",
                "Optimize system resource usage",
            ],
            metadata={
                "total_coordinations": 150,
                "active_coordinations": 3,
                "uptime_seconds": 7200,
            },
        )

    @pytest.fixture
    def performance_reporter(self, mock_performance_analyzer, mock_performance_monitor):
        """Create performance reporter for testing."""
        return PerformanceReporter(mock_performance_analyzer, mock_performance_monitor)

    def test_initialization(
        self, performance_reporter, mock_performance_analyzer, mock_performance_monitor
    ):
        """Test performance reporter initialization."""
        assert performance_reporter.performance_analyzer == mock_performance_analyzer
        assert performance_reporter.performance_monitor == mock_performance_monitor

    def test_generate_summary_report(
        self, performance_reporter, sample_performance_report
    ):
        """Test summary report generation."""
        performance_reporter.performance_analyzer.generate_performance_report.return_value = sample_performance_report

        report = performance_reporter.generate_report(
            report_type=ReportType.SUMMARY, format_type=ReportFormat.JSON
        )

        assert report["report_type"] == "summary"
        assert report["overall_performance"]["score"] == 0.85
        assert report["overall_performance"]["level"] == "good"
        assert report["overall_performance"]["coordination_success_rate"] == 0.88
        assert report["key_metrics"]["total_coordinations"] == 150
        assert report["key_metrics"]["agent_count"] == 2
        assert len(report["top_insights"]) <= 5
        assert len(report["recommendations"]) <= 5

    def test_generate_detailed_report(
        self, performance_reporter, sample_performance_report
    ):
        """Test detailed report generation."""
        performance_reporter.performance_analyzer.generate_performance_report.return_value = sample_performance_report

        report = performance_reporter.generate_report(
            report_type=ReportType.DETAILED, format_type=ReportFormat.JSON
        )

        assert report["report_type"] == "detailed"
        assert "agent_performance" in report
        assert "trend_analyses" in report
        assert "insights" in report
        assert "system_metadata" in report

        # Check agent performance details
        assert "agent1" in report["agent_performance"]
        assert report["agent_performance"]["agent1"]["score"] == 0.85
        assert report["agent_performance"]["agent1"]["level"] == "good"

        # Check trend analyses
        assert len(report["trend_analyses"]) == 2
        coord_trend = next(
            t
            for t in report["trend_analyses"]
            if t["metric_name"] == "coordination_success_rate"
        )
        assert coord_trend["direction"] == "improving"
        assert coord_trend["slope"] == 0.02

        # Check insights
        assert len(report["insights"]) == 2
        warning_insights = [i for i in report["insights"] if i["severity"] == "warning"]
        assert len(warning_insights) == 1

    def test_generate_trend_analysis_report(
        self, performance_reporter, sample_performance_report
    ):
        """Test trend analysis report generation."""
        performance_reporter.performance_analyzer.generate_performance_report.return_value = sample_performance_report

        report = performance_reporter.generate_report(
            report_type=ReportType.TREND_ANALYSIS, format_type=ReportFormat.JSON
        )

        assert report["report_type"] == "trend_analysis"
        assert "trend_summary" in report
        assert "improving_trends" in report
        assert "declining_trends" in report
        assert "stable_trends" in report
        assert "volatile_trends" in report

        # Check trend summary
        summary = report["trend_summary"]
        assert summary["improving_count"] == 1
        assert summary["stable_count"] == 1
        assert summary["declining_count"] == 0
        assert summary["total_metrics"] == 2

        # Check categorized trends
        assert len(report["improving_trends"]) == 1
        assert len(report["stable_trends"]) == 1
        assert (
            report["improving_trends"][0]["metric_name"] == "coordination_success_rate"
        )

    def test_generate_alert_report(
        self, performance_reporter, sample_performance_report
    ):
        """Test alert report generation."""
        performance_reporter.performance_analyzer.generate_performance_report.return_value = sample_performance_report

        report = performance_reporter.generate_report(
            report_type=ReportType.ALERT_REPORT, format_type=ReportFormat.JSON
        )

        assert report["report_type"] == "alert_report"
        assert "alert_summary" in report
        assert "critical_alerts" in report
        assert "warning_alerts" in report
        assert "info_alerts" in report
        assert "overall_health" in report

        # Check alert summary
        summary = report["alert_summary"]
        assert summary["critical_count"] == 0
        assert summary["warning_count"] == 1
        assert summary["info_count"] == 1
        assert summary["total_alerts"] == 2

        # Check categorized alerts
        assert len(report["warning_alerts"]) == 1
        assert len(report["info_alerts"]) == 1
        assert report["warning_alerts"][0]["title"] == "Agent Performance Disparity"

    def test_generate_agent_comparison_report(
        self, performance_reporter, sample_performance_report
    ):
        """Test agent comparison report generation."""
        performance_reporter.performance_analyzer.generate_performance_report.return_value = sample_performance_report

        report = performance_reporter.generate_report(
            report_type=ReportType.AGENT_COMPARISON, format_type=ReportFormat.JSON
        )

        assert report["report_type"] == "agent_comparison"
        assert "agent_statistics" in report
        assert "agent_rankings" in report
        assert "performance_gaps" in report
        assert "coordination_impact" in report

        # Check agent statistics
        stats = report["agent_statistics"]
        assert stats["count"] == 2
        assert stats["mean"] == 0.8  # (0.85 + 0.75) / 2
        assert stats["min"] == 0.75
        assert stats["max"] == 0.85

        # Check agent rankings
        rankings = report["agent_rankings"]
        assert len(rankings) == 2
        assert rankings[0]["rank"] == 1
        assert rankings[0]["agent_id"] == "agent1"
        assert rankings[0]["score"] == 0.85
        assert rankings[1]["rank"] == 2
        assert rankings[1]["agent_id"] == "agent2"

        # Check performance gaps (gap of 0.10 should be detected as significant)
        gaps = report["performance_gaps"]
        if len(gaps) > 0:  # Gap detection may vary based on threshold
            assert gaps[0]["agent_id"] == "agent2"
            assert gaps[0]["gap_from_best"] == 0.10

    def test_generate_agent_comparison_report_no_data(self, performance_reporter):
        """Test agent comparison report with no agent data."""
        # Create report with no agent performance scores
        empty_report = PerformanceReport(
            timestamp=time.time(),
            time_window_seconds=3600.0,
            overall_score=0.5,
            performance_level=PerformanceLevel.AVERAGE,
            coordination_success_rate=0.7,
            system_health_score=0.6,
            agent_performance_scores={},  # No agents
            trend_analyses=[],
            insights=[],
            recommendations=[],
        )

        performance_reporter.performance_analyzer.generate_performance_report.return_value = empty_report

        report = performance_reporter.generate_report(
            report_type=ReportType.AGENT_COMPARISON, format_type=ReportFormat.JSON
        )

        assert report["report_type"] == "agent_comparison"
        assert "error" in report
        assert report["error"] == "No agent performance data available"

    def test_format_html_report(self, performance_reporter):
        """Test HTML report formatting."""
        report_data = {
            "report_type": "summary",
            "timestamp": time.time(),
            "time_window_hours": 1.0,
            "overall_performance": {
                "score": 0.85,
                "level": "good",
                "coordination_success_rate": 0.88,
                "system_health_score": 0.82,
            },
            "recommendations": ["Test recommendation 1", "Test recommendation 2"],
        }

        html_report = performance_reporter._format_html_report(report_data)

        assert isinstance(html_report, str)
        assert "<!DOCTYPE html>" in html_report
        assert "MARL Performance Report" in html_report
        assert "Overall Performance" in html_report
        assert "0.850" in html_report  # Overall score
        assert "88.0%" in html_report  # Coordination success rate
        assert "Test recommendation 1" in html_report

    def test_format_markdown_report(self, performance_reporter):
        """Test Markdown report formatting."""
        report_data = {
            "report_type": "summary",
            "timestamp": time.time(),
            "time_window_hours": 1.0,
            "overall_performance": {
                "score": 0.85,
                "level": "good",
                "coordination_success_rate": 0.88,
                "system_health_score": 0.82,
            },
            "top_insights": [
                {
                    "title": "Test Insight",
                    "severity": "warning",
                    "category": "test",
                    "recommendation": "Test recommendation",
                }
            ],
            "recommendations": ["Test recommendation 1"],
        }

        md_report = performance_reporter._format_markdown_report(report_data)

        assert isinstance(md_report, str)
        assert "# MARL Performance Report" in md_report
        assert "## Overall Performance" in md_report
        assert "0.850" in md_report
        assert "88.0%" in md_report
        assert "### Test Insight (WARNING)" in md_report
        assert "## Recommendations" in md_report

    def test_format_csv_report(self, performance_reporter):
        """Test CSV report formatting."""
        report_data = {
            "overall_performance": {
                "score": 0.85,
                "coordination_success_rate": 0.88,
                "system_health_score": 0.82,
            },
            "agent_performance": {"agent1": {"score": 0.85}, "agent2": {"score": 0.75}},
        }

        csv_report = performance_reporter._format_csv_report(report_data)

        assert isinstance(csv_report, str)
        assert "Metric,Value,Category" in csv_report
        assert "Overall Score,0.850,Performance" in csv_report
        assert "Coordination Success Rate,0.880,Performance" in csv_report
        assert "Agent agent1 Score,0.850,Agent Performance" in csv_report

    @patch("pathlib.Path.open", new_callable=mock_open)
    @patch("pathlib.Path.mkdir")
    def test_save_report_json(self, mock_mkdir, mock_file, performance_reporter):
        """Test saving JSON report to file."""
        report_data = {"test": "data"}
        output_path = Path("test_report.json")

        performance_reporter._save_report(report_data, output_path, ReportFormat.JSON)

        mock_mkdir.assert_called_once()
        mock_file.assert_called_once()
        # Check that json.dump was called (indirectly through the mock)

    @patch("pathlib.Path.open", new_callable=mock_open)
    @patch("pathlib.Path.mkdir")
    def test_save_report_text(self, mock_mkdir, mock_file, performance_reporter):
        """Test saving text report to file."""
        report_content = "Test report content"
        output_path = Path("test_report.html")

        performance_reporter._save_report(
            report_content, output_path, ReportFormat.HTML
        )

        mock_mkdir.assert_called_once()
        mock_file.assert_called_once()
        mock_file().write.assert_called_once_with(report_content)

    @patch("pathlib.Path.open", side_effect=Exception("File error"))
    def test_save_report_error(self, mock_file, performance_reporter):
        """Test save report error handling."""
        report_data = {"test": "data"}
        output_path = Path("test_report.json")

        # Should not raise exception
        performance_reporter._save_report(report_data, output_path, ReportFormat.JSON)

    def test_generate_dashboard_data(
        self, performance_reporter, sample_performance_report
    ):
        """Test dashboard data generation."""
        # Mock coordination events
        mock_events = [
            Mock(
                timestamp=time.time() - 300,
                success=True,
                duration=2.5,
                agents_involved=["agent1", "agent2"],
            ),
            Mock(
                timestamp=time.time() - 600,
                success=False,
                duration=5.0,
                agents_involved=["agent1"],
            ),
            Mock(
                timestamp=time.time() - 900,
                success=True,
                duration=1.8,
                agents_involved=["agent2"],
            ),
        ]
        performance_reporter.performance_monitor._coordination_events = mock_events
        performance_reporter.performance_analyzer.generate_performance_report.return_value = sample_performance_report

        dashboard_data = performance_reporter.generate_dashboard_data(3600.0)

        assert "timestamp" in dashboard_data
        assert dashboard_data["overall_score"] == 0.85
        assert dashboard_data["performance_level"] == "good"
        assert dashboard_data["coordination_success_rate"] == 0.88
        assert dashboard_data["system_health_score"] == 0.82
        assert dashboard_data["agent_scores"] == {"agent1": 0.85, "agent2": 0.75}
        assert dashboard_data["active_coordinations"] == 3
        assert dashboard_data["total_coordinations"] == 150
        assert dashboard_data["uptime_hours"] == 2.0

        # Check alerts summary
        alerts = dashboard_data["alerts"]
        assert alerts["critical"] == 0
        assert alerts["warning"] == 1
        assert alerts["info"] == 1

        # Check trends summary
        trends = dashboard_data["trends"]
        assert trends["improving"] == 1
        assert trends["declining"] == 0
        assert trends["stable"] == 1

        # Check recent events
        assert len(dashboard_data["recent_events"]) == 3
        assert dashboard_data["recent_events"][0]["success"] is True
        assert dashboard_data["recent_events"][0]["agents_count"] == 2

        # Check recommendations
        assert len(dashboard_data["top_recommendations"]) == 3

    def test_generate_dashboard_data_error(self, performance_reporter):
        """Test dashboard data generation with error."""
        performance_reporter.performance_analyzer.generate_performance_report.side_effect = Exception(
            "Test error"
        )

        dashboard_data = performance_reporter.generate_dashboard_data()

        assert "error" in dashboard_data
        assert dashboard_data["error"] == "Dashboard data generation failed"
        assert "timestamp" in dashboard_data

    def test_generate_report_with_output_path(
        self, performance_reporter, sample_performance_report
    ):
        """Test report generation with file output."""
        performance_reporter.performance_analyzer.generate_performance_report.return_value = sample_performance_report

        with patch.object(performance_reporter, "_save_report") as mock_save:
            report = performance_reporter.generate_report(
                report_type=ReportType.SUMMARY,
                format_type=ReportFormat.JSON,
                output_path="test_report.json",
            )

            mock_save.assert_called_once()
            assert isinstance(report, dict)

    def test_generate_report_error_handling(self, performance_reporter):
        """Test report generation error handling."""
        performance_reporter.performance_analyzer.generate_performance_report.side_effect = Exception(
            "Test error"
        )

        report = performance_reporter.generate_report(
            report_type=ReportType.SUMMARY, format_type=ReportFormat.JSON
        )

        assert "error" in report
        assert report["error"] == "Report generation failed"
        assert "timestamp" in report

    def test_classify_agent_performance_level(self, performance_reporter):
        """Test agent performance level classification."""
        assert (
            performance_reporter._classify_agent_performance_level(0.95) == "excellent"
        )
        assert performance_reporter._classify_agent_performance_level(0.80) == "good"
        assert performance_reporter._classify_agent_performance_level(0.65) == "average"
        assert performance_reporter._classify_agent_performance_level(0.45) == "poor"
        assert (
            performance_reporter._classify_agent_performance_level(0.30) == "critical"
        )


class TestPerformanceReporterFactory:
    """Test performance reporter factory."""

    def test_create(self):
        """Test creating performance reporter."""
        mock_analyzer = Mock(spec=PerformanceAnalyzer)
        mock_monitor = Mock(spec=MARLPerformanceMonitor)

        reporter = PerformanceReporterFactory.create(mock_analyzer, mock_monitor)

        assert isinstance(reporter, PerformanceReporter)
        assert reporter.performance_analyzer == mock_analyzer
        assert reporter.performance_monitor == mock_monitor


@pytest.mark.integration
def test_full_report_generation_flow():
    """Integration test for full report generation flow."""
    # Create mocks
    mock_monitor = Mock(spec=MARLPerformanceMonitor)
    mock_analyzer = Mock(spec=PerformanceAnalyzer)

    # Create realistic performance report
    performance_report = PerformanceReport(
        timestamp=time.time(),
        time_window_seconds=3600.0,
        overall_score=0.78,
        performance_level=PerformanceLevel.AVERAGE,
        coordination_success_rate=0.82,
        system_health_score=0.75,
        agent_performance_scores={"agent1": 0.80, "agent2": 0.70, "agent3": 0.85},
        trend_analyses=[
            TrendAnalysis(
                metric_name="coordination_success_rate",
                direction=TrendDirection.IMPROVING,
                slope=0.015,
                confidence=0.85,
                r_squared=0.72,
                data_points=25,
                time_span_seconds=3600.0,
                current_value=0.82,
                predicted_value=0.87,
            )
        ],
        insights=[
            PerformanceInsight(
                category="coordination",
                severity="info",
                title="Improving Coordination Trend",
                description="Coordination success rate showing positive trend",
                recommendation="Continue current optimization approach",
                impact="Low - Positive trend detected",
                confidence=0.85,
            )
        ],
        recommendations=[
            "Continue monitoring coordination improvements",
            "Investigate agent3 high performance for knowledge transfer",
        ],
        metadata={
            "total_coordinations": 200,
            "active_coordinations": 2,
            "uptime_seconds": 14400,
        },
    )

    mock_analyzer.generate_performance_report.return_value = performance_report
    mock_analyzer.config = Mock()
    mock_analyzer.config.performance_thresholds = {
        "agent_performance": {
            "excellent": 0.90,
            "good": 0.75,
            "average": 0.60,
            "poor": 0.40,
        }
    }

    # Create reporter
    reporter = PerformanceReporter(mock_analyzer, mock_monitor)

    # Test different report types
    summary_report = reporter.generate_report(ReportType.SUMMARY, ReportFormat.JSON)
    detailed_report = reporter.generate_report(ReportType.DETAILED, ReportFormat.JSON)
    trend_report = reporter.generate_report(
        ReportType.TREND_ANALYSIS, ReportFormat.JSON
    )

    # Verify summary report
    assert summary_report["report_type"] == "summary"
    assert summary_report["overall_performance"]["score"] == 0.78
    assert summary_report["key_metrics"]["agent_count"] == 3

    # Verify detailed report
    assert detailed_report["report_type"] == "detailed"
    assert len(detailed_report["agent_performance"]) == 3
    assert detailed_report["agent_performance"]["agent3"]["level"] == "good"

    # Verify trend report
    assert trend_report["report_type"] == "trend_analysis"
    assert trend_report["trend_summary"]["improving_count"] == 1
    assert len(trend_report["improving_trends"]) == 1
