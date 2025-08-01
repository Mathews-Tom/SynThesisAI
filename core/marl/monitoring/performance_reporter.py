"""
MARL Performance Reporter.

This module implements comprehensive performance reporting capabilities for the
Multi-Agent Reinforcement Learning coordination system, including report generation,
visualization support, and dashboard integration.
"""

# Standard Library
import json
import time
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Union

# SynThesisAI Modules
from core.marl.monitoring.performance_analyzer import (
    PerformanceAnalyzer,
    PerformanceReport,
    TrendDirection,
)
from core.marl.monitoring.performance_monitor import MARLPerformanceMonitor
from utils.logging_config import get_logger


class ReportFormat(Enum):
    """Supported report formats."""

    JSON = "json"
    HTML = "html"
    MARKDOWN = "markdown"
    CSV = "csv"


class ReportType(Enum):
    """Types of performance reports."""

    SUMMARY = "summary"
    DETAILED = "detailed"
    TREND_ANALYSIS = "trend_analysis"
    ALERT_REPORT = "alert_report"
    AGENT_COMPARISON = "agent_comparison"


class PerformanceReporter:
    """
    Comprehensive performance reporter for MARL coordination system.

    Generates various types of performance reports in multiple formats,
    with support for visualization and dashboard integration.
    """

    def __init__(
        self,
        performance_analyzer: PerformanceAnalyzer,
        performance_monitor: MARLPerformanceMonitor,
    ):
        """
        Initialize the performance reporter.

        Args:
            performance_analyzer: Performance analyzer instance
            performance_monitor: Performance monitor instance
        """
        self.performance_analyzer = performance_analyzer
        self.performance_monitor = performance_monitor
        self.logger = get_logger(__name__)

        self.logger.info("Performance reporter initialized")

    def generate_report(
        self,
        report_type: ReportType = ReportType.SUMMARY,
        format_type: ReportFormat = ReportFormat.JSON,
        time_window_seconds: Optional[float] = None,
        output_path: Optional[Union[str, Path]] = None,
    ) -> Union[str, Dict[str, Any]]:
        """
        Generate a performance report.

        Args:
            report_type: Type of report to generate
            format_type: Output format for the report
            time_window_seconds: Time window for analysis
            output_path: Optional path to save the report

        Returns:
            Report content as string or dictionary
        """
        try:
            # Generate base performance report
            base_report = self.performance_analyzer.generate_performance_report(
                time_window_seconds
            )

            # Generate specific report type
            if report_type == ReportType.SUMMARY:
                report_data = self._generate_summary_report(base_report)
            elif report_type == ReportType.DETAILED:
                report_data = self._generate_detailed_report(base_report)
            elif report_type == ReportType.TREND_ANALYSIS:
                report_data = self._generate_trend_analysis_report(base_report)
            elif report_type == ReportType.ALERT_REPORT:
                report_data = self._generate_alert_report(base_report)
            elif report_type == ReportType.AGENT_COMPARISON:
                report_data = self._generate_agent_comparison_report(base_report)
            else:
                report_data = self._generate_summary_report(base_report)

            # Format the report
            formatted_report = self._format_report(report_data, format_type)

            # Save to file if path provided
            if output_path:
                self._save_report(formatted_report, output_path, format_type)

            return formatted_report

        except Exception as e:
            self.logger.error("Error generating report: %s", str(e))
            error_report = {
                "error": "Report generation failed",
                "message": str(e),
                "timestamp": time.time(),
            }
            return self._format_report(error_report, format_type)

    def _generate_summary_report(
        self, base_report: PerformanceReport
    ) -> Dict[str, Any]:
        """Generate a summary performance report."""
        return {
            "report_type": "summary",
            "timestamp": base_report.timestamp,
            "time_window_hours": base_report.time_window_seconds / 3600.0,
            "overall_performance": {
                "score": base_report.overall_score,
                "level": base_report.performance_level.value,
                "coordination_success_rate": base_report.coordination_success_rate,
                "system_health_score": base_report.system_health_score,
            },
            "key_metrics": {
                "total_coordinations": base_report.metadata.get(
                    "total_coordinations", 0
                ),
                "active_coordinations": base_report.metadata.get(
                    "active_coordinations", 0
                ),
                "uptime_hours": base_report.metadata.get("uptime_seconds", 0) / 3600.0,
                "agent_count": len(base_report.agent_performance_scores),
            },
            "top_insights": [
                {
                    "category": insight.category,
                    "severity": insight.severity,
                    "title": insight.title,
                    "recommendation": insight.recommendation,
                }
                for insight in sorted(
                    base_report.insights,
                    key=lambda x: {"critical": 3, "warning": 2, "info": 1}.get(
                        x.severity, 0
                    ),
                    reverse=True,
                )[:5]
            ],
            "recommendations": base_report.recommendations[:5],
        }

    def _generate_detailed_report(
        self, base_report: PerformanceReport
    ) -> Dict[str, Any]:
        """Generate a detailed performance report."""
        return {
            "report_type": "detailed",
            "timestamp": base_report.timestamp,
            "time_window_hours": base_report.time_window_seconds / 3600.0,
            "overall_performance": {
                "score": base_report.overall_score,
                "level": base_report.performance_level.value,
                "coordination_success_rate": base_report.coordination_success_rate,
                "system_health_score": base_report.system_health_score,
            },
            "agent_performance": {
                agent_id: {
                    "score": score,
                    "level": self._classify_agent_performance_level(score),
                }
                for agent_id, score in base_report.agent_performance_scores.items()
            },
            "trend_analyses": [
                {
                    "metric_name": trend.metric_name,
                    "direction": trend.direction.value,
                    "slope": trend.slope,
                    "confidence": trend.confidence,
                    "r_squared": trend.r_squared,
                    "data_points": trend.data_points,
                    "current_value": trend.current_value,
                    "predicted_value": trend.predicted_value,
                }
                for trend in base_report.trend_analyses
            ],
            "insights": [
                {
                    "category": insight.category,
                    "severity": insight.severity,
                    "title": insight.title,
                    "description": insight.description,
                    "recommendation": insight.recommendation,
                    "impact": insight.impact,
                    "confidence": insight.confidence,
                    "supporting_data": insight.supporting_data,
                }
                for insight in base_report.insights
            ],
            "recommendations": base_report.recommendations,
            "system_metadata": base_report.metadata,
        }

    def _generate_trend_analysis_report(
        self, base_report: PerformanceReport
    ) -> Dict[str, Any]:
        """Generate a trend analysis focused report."""
        # Categorize trends
        improving_trends = []
        declining_trends = []
        stable_trends = []
        volatile_trends = []

        for trend in base_report.trend_analyses:
            trend_data = {
                "metric_name": trend.metric_name,
                "slope": trend.slope,
                "confidence": trend.confidence,
                "r_squared": trend.r_squared,
                "data_points": trend.data_points,
                "current_value": trend.current_value,
                "predicted_value": trend.predicted_value,
            }

            if trend.direction == TrendDirection.IMPROVING:
                improving_trends.append(trend_data)
            elif trend.direction == TrendDirection.DECLINING:
                declining_trends.append(trend_data)
            elif trend.direction == TrendDirection.STABLE:
                stable_trends.append(trend_data)
            elif trend.direction == TrendDirection.VOLATILE:
                volatile_trends.append(trend_data)

        return {
            "report_type": "trend_analysis",
            "timestamp": base_report.timestamp,
            "time_window_hours": base_report.time_window_seconds / 3600.0,
            "trend_summary": {
                "improving_count": len(improving_trends),
                "declining_count": len(declining_trends),
                "stable_count": len(stable_trends),
                "volatile_count": len(volatile_trends),
                "total_metrics": len(base_report.trend_analyses),
            },
            "improving_trends": improving_trends,
            "declining_trends": declining_trends,
            "stable_trends": stable_trends,
            "volatile_trends": volatile_trends,
            "trend_insights": [
                insight
                for insight in base_report.insights
                if "trend" in insight.title.lower()
                or "declining" in insight.title.lower()
            ],
        }

    def _generate_alert_report(self, base_report: PerformanceReport) -> Dict[str, Any]:
        """Generate an alert-focused report."""
        # Categorize insights by severity
        critical_alerts = []
        warning_alerts = []
        info_alerts = []

        for insight in base_report.insights:
            alert_data = {
                "category": insight.category,
                "title": insight.title,
                "description": insight.description,
                "recommendation": insight.recommendation,
                "impact": insight.impact,
                "confidence": insight.confidence,
                "supporting_data": insight.supporting_data,
            }

            if insight.severity == "critical":
                critical_alerts.append(alert_data)
            elif insight.severity == "warning":
                warning_alerts.append(alert_data)
            else:
                info_alerts.append(alert_data)

        return {
            "report_type": "alert_report",
            "timestamp": base_report.timestamp,
            "time_window_hours": base_report.time_window_seconds / 3600.0,
            "alert_summary": {
                "critical_count": len(critical_alerts),
                "warning_count": len(warning_alerts),
                "info_count": len(info_alerts),
                "total_alerts": len(base_report.insights),
            },
            "critical_alerts": critical_alerts,
            "warning_alerts": warning_alerts,
            "info_alerts": info_alerts,
            "overall_health": {
                "score": base_report.overall_score,
                "level": base_report.performance_level.value,
                "coordination_success_rate": base_report.coordination_success_rate,
            },
            "immediate_actions": [
                rec
                for rec in base_report.recommendations
                if any(
                    word in rec.lower()
                    for word in ["critical", "urgent", "immediate", "investigate"]
                )
            ],
        }

    def _generate_agent_comparison_report(
        self, base_report: PerformanceReport
    ) -> Dict[str, Any]:
        """Generate an agent comparison report."""
        if not base_report.agent_performance_scores:
            return {
                "report_type": "agent_comparison",
                "timestamp": base_report.timestamp,
                "error": "No agent performance data available",
            }

        # Calculate statistics
        scores = list(base_report.agent_performance_scores.values())
        agent_stats = {
            "count": len(scores),
            "mean": sum(scores) / len(scores),
            "min": min(scores),
            "max": max(scores),
            "std": (
                sum((x - sum(scores) / len(scores)) ** 2 for x in scores) / len(scores)
            )
            ** 0.5,
        }

        # Rank agents
        ranked_agents = sorted(
            base_report.agent_performance_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        # Identify performance gaps
        performance_gaps = []
        if len(ranked_agents) > 1:
            best_score = ranked_agents[0][1]
            for agent_id, score in ranked_agents[1:]:
                gap = best_score - score
                if gap > 0.1:  # Significant gap
                    performance_gaps.append(
                        {
                            "agent_id": agent_id,
                            "score": score,
                            "gap_from_best": gap,
                            "performance_level": self._classify_agent_performance_level(
                                score
                            ),
                        }
                    )

        return {
            "report_type": "agent_comparison",
            "timestamp": base_report.timestamp,
            "time_window_hours": base_report.time_window_seconds / 3600.0,
            "agent_statistics": agent_stats,
            "agent_rankings": [
                {
                    "rank": i + 1,
                    "agent_id": agent_id,
                    "score": score,
                    "performance_level": self._classify_agent_performance_level(score),
                }
                for i, (agent_id, score) in enumerate(ranked_agents)
            ],
            "performance_gaps": performance_gaps,
            "agent_insights": [
                insight
                for insight in base_report.insights
                if insight.category == "agent_performance"
            ],
            "coordination_impact": {
                "success_rate": base_report.coordination_success_rate,
                "performance_disparity_impact": len(performance_gaps) > 0,
            },
        }

    def _classify_agent_performance_level(self, score: float) -> str:
        """Classify agent performance level."""
        thresholds = self.performance_analyzer.config.performance_thresholds.get(
            "agent_performance", {}
        )

        if score >= thresholds.get("excellent", 0.90):
            return "excellent"
        elif score >= thresholds.get("good", 0.75):
            return "good"
        elif score >= thresholds.get("average", 0.60):
            return "average"
        elif score >= thresholds.get("poor", 0.40):
            return "poor"
        else:
            return "critical"

    def _format_report(
        self, report_data: Dict[str, Any], format_type: ReportFormat
    ) -> Union[str, Dict[str, Any]]:
        """Format report according to specified format."""
        if format_type == ReportFormat.JSON:
            return report_data
        elif format_type == ReportFormat.HTML:
            return self._format_html_report(report_data)
        elif format_type == ReportFormat.MARKDOWN:
            return self._format_markdown_report(report_data)
        elif format_type == ReportFormat.CSV:
            return self._format_csv_report(report_data)
        else:
            return report_data

    def _format_html_report(self, report_data: Dict[str, Any]) -> str:
        """Format report as HTML."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>MARL Performance Report - {report_data.get("report_type", "Unknown").title()}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .metric {{ background-color: #f9f9f9; padding: 10px; margin: 5px 0; border-left: 4px solid #007acc; }}
                .alert-critical {{ border-left-color: #dc3545; }}
                .alert-warning {{ border-left-color: #ffc107; }}
                .alert-info {{ border-left-color: #17a2b8; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>MARL Performance Report</h1>
                <p><strong>Report Type:</strong> {report_data.get("report_type", "Unknown").title()}</p>
                <p><strong>Generated:</strong> {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(report_data.get("timestamp", time.time())))}</p>
                <p><strong>Time Window:</strong> {report_data.get("time_window_hours", 0):.1f} hours</p>
            </div>
        """

        # Add overall performance section
        if "overall_performance" in report_data:
            perf = report_data["overall_performance"]
            html_content += f"""
            <div class="section">
                <h2>Overall Performance</h2>
                <div class="metric">
                    <strong>Overall Score:</strong> {perf.get("score", 0):.3f} ({perf.get("level", "Unknown").title()})
                </div>
                <div class="metric">
                    <strong>Coordination Success Rate:</strong> {perf.get("coordination_success_rate", 0):.1%}
                </div>
                <div class="metric">
                    <strong>System Health Score:</strong> {perf.get("system_health_score", 0):.3f}
                </div>
            </div>
            """

        # Add insights/alerts section
        if "critical_alerts" in report_data:
            html_content += "<div class='section'><h2>Critical Alerts</h2>"
            for alert in report_data["critical_alerts"]:
                html_content += f"""
                <div class="metric alert-critical">
                    <strong>{alert["title"]}</strong><br>
                    {alert["description"]}<br>
                    <em>Recommendation: {alert["recommendation"]}</em>
                </div>
                """
            html_content += "</div>"

        # Add recommendations
        if "recommendations" in report_data and report_data["recommendations"]:
            html_content += "<div class='section'><h2>Recommendations</h2><ul>"
            for rec in report_data["recommendations"]:
                html_content += f"<li>{rec}</li>"
            html_content += "</ul></div>"

        html_content += "</body></html>"
        return html_content

    def _format_markdown_report(self, report_data: Dict[str, Any]) -> str:
        """Format report as Markdown."""
        md_content = f"""# MARL Performance Report

**Report Type:** {report_data.get("report_type", "Unknown").title()}
**Generated:** {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(report_data.get("timestamp", time.time())))}
**Time Window:** {report_data.get("time_window_hours", 0):.1f} hours

"""

        # Add overall performance
        if "overall_performance" in report_data:
            perf = report_data["overall_performance"]
            md_content += f"""## Overall Performance

- **Overall Score:** {perf.get("score", 0):.3f} ({perf.get("level", "Unknown").title()})
- **Coordination Success Rate:** {perf.get("coordination_success_rate", 0):.1%}
- **System Health Score:** {perf.get("system_health_score", 0):.3f}

"""

        # Add key insights
        if "top_insights" in report_data:
            md_content += "## Key Insights\n\n"
            for insight in report_data["top_insights"]:
                md_content += f"""### {insight["title"]} ({insight["severity"].upper()})

**Category:** {insight["category"]}
**Recommendation:** {insight["recommendation"]}

"""

        # Add recommendations
        if "recommendations" in report_data and report_data["recommendations"]:
            md_content += "## Recommendations\n\n"
            for i, rec in enumerate(report_data["recommendations"], 1):
                md_content += f"{i}. {rec}\n"

        return md_content

    def _format_csv_report(self, report_data: Dict[str, Any]) -> str:
        """Format report as CSV (simplified)."""
        csv_lines = []
        csv_lines.append("Metric,Value,Category")

        # Add basic metrics
        if "overall_performance" in report_data:
            perf = report_data["overall_performance"]
            csv_lines.append(f"Overall Score,{perf.get('score', 0):.3f},Performance")
            csv_lines.append(
                f"Coordination Success Rate,{perf.get('coordination_success_rate', 0):.3f},Performance"
            )
            csv_lines.append(
                f"System Health Score,{perf.get('system_health_score', 0):.3f},Performance"
            )

        # Add agent scores if available
        if "agent_performance" in report_data:
            for agent_id, agent_data in report_data["agent_performance"].items():
                csv_lines.append(
                    f"Agent {agent_id} Score,{agent_data.get('score', 0):.3f},Agent Performance"
                )

        return "\n".join(csv_lines)

    def _save_report(
        self,
        report_content: Union[str, Dict[str, Any]],
        output_path: Union[str, Path],
        format_type: ReportFormat,
    ):
        """Save report to file."""
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if format_type == ReportFormat.JSON:
                with output_path.open("w", encoding="utf-8") as f:
                    json.dump(report_content, f, indent=2, default=str)
            else:
                with output_path.open("w", encoding="utf-8") as f:
                    f.write(str(report_content))

            self.logger.info("Report saved to %s", output_path)

        except Exception as e:
            self.logger.error("Error saving report to %s: %s", output_path, str(e))

    def generate_dashboard_data(
        self, time_window_seconds: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Generate data optimized for dashboard consumption.

        Args:
            time_window_seconds: Time window for analysis

        Returns:
            Dashboard-optimized data structure
        """
        try:
            base_report = self.performance_analyzer.generate_performance_report(
                time_window_seconds
            )

            # Get recent coordination events for timeline
            recent_events = []
            for event in list(self.performance_monitor._coordination_events)[-20:]:
                recent_events.append(
                    {
                        "timestamp": event.timestamp,
                        "success": event.success,
                        "duration": event.duration,
                        "agents_count": len(event.agents_involved),
                    }
                )

            return {
                "timestamp": base_report.timestamp,
                "overall_score": base_report.overall_score,
                "performance_level": base_report.performance_level.value,
                "coordination_success_rate": base_report.coordination_success_rate,
                "system_health_score": base_report.system_health_score,
                "agent_scores": base_report.agent_performance_scores,
                "active_coordinations": base_report.metadata.get(
                    "active_coordinations", 0
                ),
                "total_coordinations": base_report.metadata.get(
                    "total_coordinations", 0
                ),
                "uptime_hours": base_report.metadata.get("uptime_seconds", 0) / 3600.0,
                "alerts": {
                    "critical": len(
                        [i for i in base_report.insights if i.severity == "critical"]
                    ),
                    "warning": len(
                        [i for i in base_report.insights if i.severity == "warning"]
                    ),
                    "info": len(
                        [i for i in base_report.insights if i.severity == "info"]
                    ),
                },
                "trends": {
                    "improving": len(
                        [
                            t
                            for t in base_report.trend_analyses
                            if t.direction == TrendDirection.IMPROVING
                        ]
                    ),
                    "declining": len(
                        [
                            t
                            for t in base_report.trend_analyses
                            if t.direction == TrendDirection.DECLINING
                        ]
                    ),
                    "stable": len(
                        [
                            t
                            for t in base_report.trend_analyses
                            if t.direction == TrendDirection.STABLE
                        ]
                    ),
                },
                "recent_events": recent_events,
                "top_recommendations": base_report.recommendations[:3],
            }

        except Exception as e:
            self.logger.error("Error generating dashboard data: %s", str(e))
            return {
                "timestamp": time.time(),
                "error": "Dashboard data generation failed",
                "message": str(e),
            }


class PerformanceReporterFactory:
    """Factory for creating performance reporters."""

    @staticmethod
    def create(
        performance_analyzer: PerformanceAnalyzer,
        performance_monitor: MARLPerformanceMonitor,
    ) -> PerformanceReporter:
        """
        Create a performance reporter.

        Args:
            performance_analyzer: Performance analyzer instance
            performance_monitor: Performance monitor instance

        Returns:
            Configured performance reporter
        """
        return PerformanceReporter(performance_analyzer, performance_monitor)
