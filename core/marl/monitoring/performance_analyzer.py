"""
MARL Performance Analyzer.

This module implements comprehensive performance analysis capabilities for the
Multi-Agent Reinforcement Learning coordination system, including trend analysis,
performance improvement recommendations, and detailed metrics analysis.
"""

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from core.marl.monitoring.performance_monitor import (
    MARLPerformanceMonitor,
    MetricType,
    PerformanceMetric,
)
from utils.logging_config import get_logger


class TrendDirection(Enum):
    """Trend direction classifications."""

    IMPROVING = "improving"
    DECLINING = "declining"
    STABLE = "stable"
    VOLATILE = "volatile"
    INSUFFICIENT_DATA = "insufficient_data"


class PerformanceLevel(Enum):
    """Performance level classifications."""

    EXCELLENT = "excellent"
    GOOD = "good"
    AVERAGE = "average"
    POOR = "poor"
    CRITICAL = "critical"


@dataclass
class TrendAnalysis:
    """Trend analysis results."""

    metric_name: str
    direction: TrendDirection
    slope: float
    confidence: float
    r_squared: float
    data_points: int
    time_span_seconds: float
    current_value: float
    predicted_value: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceInsight:
    """Performance insight with recommendations."""

    category: str
    severity: str  # "info", "warning", "critical"
    title: str
    description: str
    recommendation: str
    impact: str
    confidence: float
    supporting_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceReport:
    """Comprehensive performance report."""

    timestamp: float
    time_window_seconds: float
    overall_score: float
    performance_level: PerformanceLevel
    coordination_success_rate: float
    system_health_score: float
    agent_performance_scores: Dict[str, float]
    trend_analyses: List[TrendAnalysis]
    insights: List[PerformanceInsight]
    recommendations: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisConfig:
    """Configuration for performance analysis."""

    min_data_points: int = 10
    trend_confidence_threshold: float = 0.7
    volatility_threshold: float = 0.3
    performance_thresholds: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {
            "coordination_success_rate": {
                "excellent": 0.95,
                "good": 0.85,
                "average": 0.70,
                "poor": 0.50,
            },
            "system_health": {
                "excellent": 0.95,
                "good": 0.85,
                "average": 0.70,
                "poor": 0.50,
            },
            "agent_performance": {
                "excellent": 0.90,
                "good": 0.75,
                "average": 0.60,
                "poor": 0.40,
            },
        }
    )
    enable_predictions: bool = True
    prediction_horizon_seconds: float = 3600.0  # 1 hour


class PerformanceAnalyzer:
    """
    Advanced performance analyzer for MARL coordination system.

    Provides trend analysis, performance insights, recommendations,
    and comprehensive reporting capabilities.
    """

    def __init__(
        self,
        performance_monitor: MARLPerformanceMonitor,
        config: Optional[AnalysisConfig] = None,
    ):
        """
        Initialize the performance analyzer.

        Args:
            performance_monitor: Performance monitor instance
            config: Analysis configuration
        """
        self.performance_monitor = performance_monitor
        self.config = config or AnalysisConfig()
        self.logger = get_logger(__name__)

        # Analysis cache
        self._trend_cache: Dict[str, TrendAnalysis] = {}
        self._cache_timestamp = 0.0
        self._cache_ttl = 60.0  # Cache for 1 minute

        self.logger.info("Performance analyzer initialized")

    def analyze_trend(
        self,
        metric_name: str,
        metric_type: MetricType,
        agent_id: Optional[str] = None,
        time_window_seconds: Optional[float] = None,
    ) -> TrendAnalysis:
        """
        Analyze trend for a specific metric.

        Args:
            metric_name: Name of the metric
            metric_type: Type of metric
            agent_id: Optional agent ID for agent-specific metrics
            time_window_seconds: Time window for analysis

        Returns:
            Trend analysis results
        """
        cache_key = (
            f"{metric_name}_{metric_type.value}_{agent_id}_{time_window_seconds}"
        )

        # Check cache
        if (
            cache_key in self._trend_cache
            and time.time() - self._cache_timestamp < self._cache_ttl
        ):
            return self._trend_cache[cache_key]

        try:
            # Get metric data
            if agent_id:
                metrics = self._get_agent_metric_data(
                    agent_id, metric_type, time_window_seconds
                )
            else:
                metrics = self._get_system_metric_data(metric_type, time_window_seconds)

            if len(metrics) < self.config.min_data_points:
                return TrendAnalysis(
                    metric_name=metric_name,
                    direction=TrendDirection.INSUFFICIENT_DATA,
                    slope=0.0,
                    confidence=0.0,
                    r_squared=0.0,
                    data_points=len(metrics),
                    time_span_seconds=0.0,
                    current_value=0.0,
                )

            # Extract values and timestamps
            values = [m.value for m in metrics if isinstance(m.value, (int, float))]
            timestamps = [
                m.timestamp for m in metrics if isinstance(m.value, (int, float))
            ]

            if len(values) < self.config.min_data_points:
                return TrendAnalysis(
                    metric_name=metric_name,
                    direction=TrendDirection.INSUFFICIENT_DATA,
                    slope=0.0,
                    confidence=0.0,
                    r_squared=0.0,
                    data_points=len(values),
                    time_span_seconds=0.0,
                    current_value=0.0,
                )

            # Perform trend analysis
            trend_analysis = self._calculate_trend(metric_name, values, timestamps)

            # Cache result
            self._trend_cache[cache_key] = trend_analysis
            self._cache_timestamp = time.time()

            return trend_analysis

        except Exception as e:
            self.logger.error("Error analyzing trend for %s: %s", metric_name, str(e))
            return TrendAnalysis(
                metric_name=metric_name,
                direction=TrendDirection.INSUFFICIENT_DATA,
                slope=0.0,
                confidence=0.0,
                r_squared=0.0,
                data_points=0,
                time_span_seconds=0.0,
                current_value=0.0,
            )

    def _calculate_trend(
        self, metric_name: str, values: List[float], timestamps: List[float]
    ) -> TrendAnalysis:
        """Calculate trend analysis for metric values."""
        if len(values) < 2:
            return TrendAnalysis(
                metric_name=metric_name,
                direction=TrendDirection.INSUFFICIENT_DATA,
                slope=0.0,
                confidence=0.0,
                r_squared=0.0,
                data_points=len(values),
                time_span_seconds=0.0,
                current_value=values[0] if values else 0.0,
            )

        # Normalize timestamps to start from 0
        time_span = timestamps[-1] - timestamps[0]
        normalized_times = [(t - timestamps[0]) for t in timestamps]

        # Simple linear regression using numpy
        X = np.array(normalized_times)
        y = np.array(values)

        # Calculate slope using least squares
        if len(X) > 1:
            slope = np.polyfit(X, y, 1)[0]

            # Calculate R-squared
            y_pred = slope * X + np.mean(y) - slope * np.mean(X)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
        else:
            slope = 0.0
            r_squared = 0.0

        # Calculate confidence based on R-squared and data points
        confidence = min(r_squared * (len(values) / 50.0), 1.0)

        # Determine trend direction
        direction = self._classify_trend_direction(slope, r_squared, values)

        # Prediction if enabled
        predicted_value = None
        if (
            self.config.enable_predictions
            and confidence > self.config.trend_confidence_threshold
        ):
            prediction_time = time_span + self.config.prediction_horizon_seconds
            predicted_value = slope * prediction_time + np.mean(y) - slope * np.mean(X)

        return TrendAnalysis(
            metric_name=metric_name,
            direction=direction,
            slope=slope,
            confidence=confidence,
            r_squared=r_squared,
            data_points=len(values),
            time_span_seconds=time_span,
            current_value=values[-1],
            predicted_value=predicted_value,
        )

    def _classify_trend_direction(
        self, slope: float, r_squared: float, values: List[float]
    ) -> TrendDirection:
        """Classify trend direction based on slope and volatility."""
        # Check for insufficient confidence
        if r_squared < self.config.trend_confidence_threshold:
            # Check volatility
            if len(values) > 3:
                volatility = (
                    np.std(values) / np.mean(values) if np.mean(values) != 0 else 0
                )
                if volatility > self.config.volatility_threshold:
                    return TrendDirection.VOLATILE
            return TrendDirection.STABLE

        # Classify based on slope
        if abs(slope) < 1e-6:  # Very small slope
            return TrendDirection.STABLE
        elif slope > 0:
            return TrendDirection.IMPROVING
        else:
            return TrendDirection.DECLINING

    def generate_insights(
        self, time_window_seconds: Optional[float] = None
    ) -> List[PerformanceInsight]:
        """
        Generate performance insights and recommendations.

        Args:
            time_window_seconds: Time window for analysis

        Returns:
            List of performance insights
        """
        insights = []

        try:
            # Coordination success rate insights
            coord_insights = self._analyze_coordination_performance(time_window_seconds)
            insights.extend(coord_insights)

            # Agent performance insights
            agent_insights = self._analyze_agent_performance(time_window_seconds)
            insights.extend(agent_insights)

            # System performance insights
            system_insights = self._analyze_system_performance(time_window_seconds)
            insights.extend(system_insights)

            # Learning progress insights
            learning_insights = self._analyze_learning_progress(time_window_seconds)
            insights.extend(learning_insights)

        except Exception as e:
            self.logger.error("Error generating insights: %s", str(e))

        return insights

    def _analyze_coordination_performance(
        self, time_window_seconds: Optional[float]
    ) -> List[PerformanceInsight]:
        """Analyze coordination performance and generate insights."""
        insights = []

        # Get coordination success rate
        success_rate = self.performance_monitor.get_coordination_success_rate(
            time_window_seconds
        )

        # Analyze trend
        trend = self.analyze_trend(
            "coordination_success_rate",
            MetricType.COORDINATION_SUCCESS,
            time_window_seconds=time_window_seconds,
        )

        # Generate insights based on success rate
        if success_rate < 0.5:
            insights.append(
                PerformanceInsight(
                    category="coordination",
                    severity="critical",
                    title="Critical Coordination Success Rate",
                    description=f"Coordination success rate is critically low at {success_rate:.1%}",
                    recommendation="Investigate coordination conflicts, review consensus mechanisms, and check agent communication protocols",
                    impact="High - System effectiveness severely compromised",
                    confidence=0.9,
                    supporting_data={
                        "success_rate": success_rate,
                        "trend": trend.direction.value,
                    },
                )
            )
        elif success_rate < 0.7:
            insights.append(
                PerformanceInsight(
                    category="coordination",
                    severity="warning",
                    title="Low Coordination Success Rate",
                    description=f"Coordination success rate is below optimal at {success_rate:.1%}",
                    recommendation="Review coordination policies, optimize consensus thresholds, and monitor agent performance",
                    impact="Medium - System performance degraded",
                    confidence=0.8,
                    supporting_data={
                        "success_rate": success_rate,
                        "trend": trend.direction.value,
                    },
                )
            )

        # Generate trend-based insights
        if trend.direction == TrendDirection.DECLINING and trend.confidence > 0.7:
            insights.append(
                PerformanceInsight(
                    category="coordination",
                    severity="warning",
                    title="Declining Coordination Performance",
                    description=f"Coordination success rate is declining (slope: {trend.slope:.4f})",
                    recommendation="Investigate recent changes, monitor agent learning progress, and consider retraining",
                    impact="Medium - Performance degradation trend detected",
                    confidence=trend.confidence,
                    supporting_data={
                        "trend_slope": trend.slope,
                        "r_squared": trend.r_squared,
                    },
                )
            )

        return insights

    def _analyze_agent_performance(
        self, time_window_seconds: Optional[float]
    ) -> List[PerformanceInsight]:
        """Analyze individual agent performance."""
        insights = []

        # Get agent performance summaries
        agent_summaries = {}
        for agent_id in self.performance_monitor._agent_metrics.keys():
            summary = self.performance_monitor.get_agent_performance_summary(
                agent_id, time_window_seconds
            )
            if "error" not in summary:
                agent_summaries[agent_id] = summary

        if not agent_summaries:
            return insights

        # Analyze performance disparities
        if len(agent_summaries) > 1:
            performance_scores = []
            for agent_id, summary in agent_summaries.items():
                if MetricType.AGENT_PERFORMANCE.value in summary:
                    perf_data = summary[MetricType.AGENT_PERFORMANCE.value]
                    if "mean" in perf_data:
                        performance_scores.append((agent_id, perf_data["mean"]))

            if len(performance_scores) > 1:
                scores = [score for _, score in performance_scores]
                score_std = np.std(scores)
                score_mean = np.mean(scores)

                if score_std / score_mean > 0.3:  # High variance
                    worst_agent = min(performance_scores, key=lambda x: x[1])
                    best_agent = max(performance_scores, key=lambda x: x[1])

                    insights.append(
                        PerformanceInsight(
                            category="agent_performance",
                            severity="warning",
                            title="High Agent Performance Disparity",
                            description=f"Significant performance gap between agents (std: {score_std:.3f})",
                            recommendation=f"Focus training on underperforming agent {worst_agent[0]}, consider knowledge transfer from {best_agent[0]}",
                            impact="Medium - Uneven agent capabilities affecting coordination",
                            confidence=0.8,
                            supporting_data={
                                "performance_std": score_std,
                                "worst_agent": worst_agent[0],
                                "best_agent": best_agent[0],
                            },
                        )
                    )

        # Analyze individual agent trends
        for agent_id in agent_summaries.keys():
            learning_progress = self.performance_monitor.get_learning_progress(agent_id)
            if "error" not in learning_progress:
                if (
                    learning_progress.get("recent_performance", {}).get("improving")
                    is False
                ):
                    insights.append(
                        PerformanceInsight(
                            category="agent_performance",
                            severity="warning",
                            title=f"Agent {agent_id} Learning Stagnation",
                            description=f"Agent {agent_id} shows declining learning progress",
                            recommendation="Review learning rate, check for overfitting, consider curriculum adjustment",
                            impact="Medium - Individual agent performance degradation",
                            confidence=0.7,
                            supporting_data={
                                "agent_id": agent_id,
                                "learning_data": learning_progress,
                            },
                        )
                    )

        return insights

    def _analyze_system_performance(
        self, time_window_seconds: Optional[float]
    ) -> List[PerformanceInsight]:
        """Analyze system-level performance."""
        insights = []

        # Get system performance summary
        system_summary = self.performance_monitor.get_system_performance_summary(
            time_window_seconds
        )

        # Check coordination load
        if system_summary.get("active_coordinations", 0) > 10:
            insights.append(
                PerformanceInsight(
                    category="system_performance",
                    severity="warning",
                    title="High Coordination Load",
                    description=f"High number of active coordinations: {system_summary['active_coordinations']}",
                    recommendation="Monitor system resources, consider load balancing, review coordination timeout settings",
                    impact="Medium - Potential system overload",
                    confidence=0.8,
                    supporting_data={
                        "active_coordinations": system_summary["active_coordinations"]
                    },
                )
            )

        # Check system metrics
        system_metrics = system_summary.get("system_metrics", {})
        for metric_name, metric_data in system_metrics.items():
            if isinstance(metric_data, dict) and "latest" in metric_data:
                latest_value = metric_data["latest"]

                if metric_name in ["cpu_usage", "memory_usage"] and latest_value > 0.8:
                    insights.append(
                        PerformanceInsight(
                            category="system_performance",
                            severity="warning",
                            title=f"High {metric_name.replace('_', ' ').title()}",
                            description=f"{metric_name.replace('_', ' ').title()} is high at {latest_value:.1%}",
                            recommendation="Monitor resource usage, consider scaling resources, optimize algorithms",
                            impact="Medium - Resource constraints may affect performance",
                            confidence=0.8,
                            supporting_data={metric_name: latest_value},
                        )
                    )

        return insights

    def _analyze_learning_progress(
        self, time_window_seconds: Optional[float]
    ) -> List[PerformanceInsight]:
        """Analyze learning progress across agents."""
        insights = []

        # Check overall learning trends
        learning_trends = []
        for agent_id in self.performance_monitor._agent_metrics.keys():
            trend = self.analyze_trend(
                f"agent_{agent_id}_learning",
                MetricType.LEARNING_PROGRESS,
                agent_id=agent_id,
                time_window_seconds=time_window_seconds,
            )
            if trend.direction != TrendDirection.INSUFFICIENT_DATA:
                learning_trends.append((agent_id, trend))

        # Check for stagnant learning
        stagnant_agents = [
            agent_id
            for agent_id, trend in learning_trends
            if trend.direction == TrendDirection.STABLE and trend.confidence > 0.7
        ]

        if len(stagnant_agents) > len(learning_trends) * 0.5:  # More than half stagnant
            insights.append(
                PerformanceInsight(
                    category="learning_progress",
                    severity="warning",
                    title="Widespread Learning Stagnation",
                    description=f"{len(stagnant_agents)} out of {len(learning_trends)} agents show stagnant learning",
                    recommendation="Review learning rates, consider curriculum changes, check for convergence",
                    impact="Medium - Learning efficiency compromised",
                    confidence=0.8,
                    supporting_data={"stagnant_agents": stagnant_agents},
                )
            )

        return insights

    def generate_performance_report(
        self, time_window_seconds: Optional[float] = None
    ) -> PerformanceReport:
        """
        Generate comprehensive performance report.

        Args:
            time_window_seconds: Time window for analysis

        Returns:
            Comprehensive performance report
        """
        try:
            # Get basic metrics
            coordination_success_rate = (
                self.performance_monitor.get_coordination_success_rate(
                    time_window_seconds
                )
            )
            system_summary = self.performance_monitor.get_system_performance_summary(
                time_window_seconds
            )

            # Calculate agent performance scores
            agent_performance_scores = {}
            for agent_id in self.performance_monitor._agent_metrics.keys():
                summary = self.performance_monitor.get_agent_performance_summary(
                    agent_id, time_window_seconds
                )
                if (
                    "error" not in summary
                    and MetricType.AGENT_PERFORMANCE.value in summary
                ):
                    perf_data = summary[MetricType.AGENT_PERFORMANCE.value]
                    if "mean" in perf_data:
                        agent_performance_scores[agent_id] = perf_data["mean"]

            # Calculate system health score
            system_health_score = self._calculate_system_health_score(system_summary)

            # Calculate overall score
            overall_score = self._calculate_overall_score(
                coordination_success_rate, system_health_score, agent_performance_scores
            )

            # Determine performance level
            performance_level = self._classify_performance_level(overall_score)

            # Generate trend analyses
            trend_analyses = self._generate_trend_analyses(time_window_seconds)

            # Generate insights
            insights = self.generate_insights(time_window_seconds)

            # Generate recommendations
            recommendations = self._generate_recommendations(insights, trend_analyses)

            return PerformanceReport(
                timestamp=time.time(),
                time_window_seconds=time_window_seconds or 3600.0,
                overall_score=overall_score,
                performance_level=performance_level,
                coordination_success_rate=coordination_success_rate,
                system_health_score=system_health_score,
                agent_performance_scores=agent_performance_scores,
                trend_analyses=trend_analyses,
                insights=insights,
                recommendations=recommendations,
                metadata={
                    "total_coordinations": system_summary.get("total_coordinations", 0),
                    "active_coordinations": system_summary.get(
                        "active_coordinations", 0
                    ),
                    "uptime_seconds": system_summary.get("uptime_seconds", 0),
                },
            )

        except Exception as e:
            self.logger.error("Error generating performance report: %s", str(e))
            return PerformanceReport(
                timestamp=time.time(),
                time_window_seconds=time_window_seconds or 3600.0,
                overall_score=0.0,
                performance_level=PerformanceLevel.CRITICAL,
                coordination_success_rate=0.0,
                system_health_score=0.0,
                agent_performance_scores={},
                trend_analyses=[],
                insights=[],
                recommendations=[
                    "System analysis failed - investigate monitoring infrastructure"
                ],
            )

    def _calculate_system_health_score(self, system_summary: Dict[str, Any]) -> float:
        """Calculate system health score from system metrics."""
        try:
            system_metrics = system_summary.get("system_metrics", {})
            if not system_metrics:
                return 0.5  # Neutral score if no data

            health_scores = []

            # CPU usage (lower is better)
            if "cpu_usage" in system_metrics:
                cpu_data = system_metrics["cpu_usage"]
                if isinstance(cpu_data, dict) and "latest" in cpu_data:
                    cpu_usage = cpu_data["latest"]
                    cpu_score = max(0.0, 1.0 - cpu_usage)
                    health_scores.append(cpu_score)

            # Memory usage (lower is better)
            if "memory_usage" in system_metrics:
                memory_data = system_metrics["memory_usage"]
                if isinstance(memory_data, dict) and "latest" in memory_data:
                    memory_usage = memory_data["latest"]
                    memory_score = max(0.0, 1.0 - memory_usage)
                    health_scores.append(memory_score)

            return np.mean(health_scores) if health_scores else 0.5

        except Exception as e:
            self.logger.error("Error calculating system health score: %s", str(e))
            return 0.0

    def _calculate_overall_score(
        self,
        coordination_success_rate: float,
        system_health_score: float,
        agent_performance_scores: Dict[str, float],
    ) -> float:
        """Calculate overall performance score."""
        try:
            # Weighted scoring
            coord_weight = 0.5
            system_weight = 0.2
            agent_weight = 0.3

            # Coordination score
            coord_score = coordination_success_rate

            # System score
            system_score = system_health_score

            # Agent score (average of all agents)
            agent_score = (
                np.mean(list(agent_performance_scores.values()))
                if agent_performance_scores
                else 0.5
            )

            overall_score = (
                coord_weight * coord_score
                + system_weight * system_score
                + agent_weight * agent_score
            )

            return max(0.0, min(1.0, overall_score))

        except Exception as e:
            self.logger.error("Error calculating overall score: %s", str(e))
            return 0.0

    def _classify_performance_level(self, overall_score: float) -> PerformanceLevel:
        """Classify performance level based on overall score."""
        thresholds = self.config.performance_thresholds.get(
            "coordination_success_rate", {}
        )

        if overall_score >= thresholds.get("excellent", 0.95):
            return PerformanceLevel.EXCELLENT
        elif overall_score >= thresholds.get("good", 0.85):
            return PerformanceLevel.GOOD
        elif overall_score >= thresholds.get("average", 0.70):
            return PerformanceLevel.AVERAGE
        elif overall_score >= thresholds.get("poor", 0.50):
            return PerformanceLevel.POOR
        else:
            return PerformanceLevel.CRITICAL

    def _generate_trend_analyses(
        self, time_window_seconds: Optional[float]
    ) -> List[TrendAnalysis]:
        """Generate trend analyses for key metrics."""
        trend_analyses = []

        # Coordination success rate trend
        coord_trend = self.analyze_trend(
            "coordination_success_rate",
            MetricType.COORDINATION_SUCCESS,
            time_window_seconds=time_window_seconds,
        )
        trend_analyses.append(coord_trend)

        # Agent performance trends
        for agent_id in self.performance_monitor._agent_metrics.keys():
            agent_trend = self.analyze_trend(
                f"agent_{agent_id}_performance",
                MetricType.AGENT_PERFORMANCE,
                agent_id=agent_id,
                time_window_seconds=time_window_seconds,
            )
            trend_analyses.append(agent_trend)

        return trend_analyses

    def _generate_recommendations(
        self, insights: List[PerformanceInsight], trend_analyses: List[TrendAnalysis]
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        # Extract recommendations from insights
        for insight in insights:
            if insight.severity in ["warning", "critical"]:
                recommendations.append(insight.recommendation)

        # Add trend-based recommendations
        declining_trends = [
            trend
            for trend in trend_analyses
            if trend.direction == TrendDirection.DECLINING and trend.confidence > 0.7
        ]

        if len(declining_trends) > len(trend_analyses) * 0.3:  # More than 30% declining
            recommendations.append(
                "Multiple metrics showing declining trends - conduct comprehensive system review"
            )

        # Remove duplicates and limit
        unique_recommendations = list(dict.fromkeys(recommendations))
        return unique_recommendations[:10]  # Limit to top 10

    def _get_agent_metric_data(
        self,
        agent_id: str,
        metric_type: MetricType,
        time_window_seconds: Optional[float],
    ) -> List[PerformanceMetric]:
        """Get metric data for a specific agent."""
        if agent_id not in self.performance_monitor._agent_metrics:
            return []

        agent_metrics = self.performance_monitor._agent_metrics[agent_id]
        metric_data = agent_metrics.get(metric_type.value, [])

        if time_window_seconds:
            cutoff_time = time.time() - time_window_seconds
            metric_data = [m for m in metric_data if m.timestamp >= cutoff_time]

        return list(metric_data)

    def _get_system_metric_data(
        self, metric_type: MetricType, time_window_seconds: Optional[float]
    ) -> List[PerformanceMetric]:
        """Get system-level metric data."""
        metric_data = self.performance_monitor._metrics.get(metric_type, [])

        if time_window_seconds:
            cutoff_time = time.time() - time_window_seconds
            metric_data = [m for m in metric_data if m.timestamp >= cutoff_time]

        return list(metric_data)

    def clear_cache(self):
        """Clear analysis cache."""
        self._trend_cache.clear()
        self._cache_timestamp = 0.0
        self.logger.info("Analysis cache cleared")


class PerformanceAnalyzerFactory:
    """Factory for creating performance analyzers."""

    @staticmethod
    def create(
        performance_monitor: MARLPerformanceMonitor,
        config: Optional[AnalysisConfig] = None,
    ) -> PerformanceAnalyzer:
        """
        Create a performance analyzer.

        Args:
            performance_monitor: Performance monitor instance
            config: Optional analysis configuration

        Returns:
            Configured performance analyzer
        """
        return PerformanceAnalyzer(performance_monitor, config)

    @staticmethod
    def create_with_custom_config(
        performance_monitor: MARLPerformanceMonitor,
        min_data_points: int = 10,
        trend_confidence_threshold: float = 0.7,
        enable_predictions: bool = True,
        **kwargs,
    ) -> PerformanceAnalyzer:
        """
        Create a performance analyzer with custom configuration.

        Args:
            performance_monitor: Performance monitor instance
            min_data_points: Minimum data points for analysis
            trend_confidence_threshold: Confidence threshold for trends
            enable_predictions: Whether to enable predictions
            **kwargs: Additional configuration parameters

        Returns:
            Configured performance analyzer
        """
        config = AnalysisConfig(
            min_data_points=min_data_points,
            trend_confidence_threshold=trend_confidence_threshold,
            enable_predictions=enable_predictions,
            **kwargs,
        )

        return PerformanceAnalyzer(performance_monitor, config)
