"""Performance Validation Framework for MARL.

This module provides performance validation capabilities to ensure MARL
coordination meets the >30% improvement requirement and other performance targets.
"""

import asyncio
import statistics
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from utils.logging_config import get_logger

from .mock_environment import MockEnvironmentConfig, MockMARLEnvironment


class PerformanceMetric(Enum):
    """Performance metric enumeration."""

    COORDINATION_SUCCESS_RATE = "coordination_success_rate"
    AVERAGE_REWARD = "average_reward"
    LEARNING_SPEED = "learning_speed"
    RESPONSE_TIME = "response_time"
    THROUGHPUT = "throughput"
    RESOURCE_EFFICIENCY = "resource_efficiency"
    QUALITY_SCORE = "quality_score"


class ComparisonType(Enum):
    """Comparison type enumeration."""

    BASELINE_COMPARISON = "baseline_comparison"
    HISTORICAL_COMPARISON = "historical_comparison"
    TARGET_COMPARISON = "target_comparison"
    PEER_COMPARISON = "peer_comparison"


@dataclass
class PerformanceConfig:
    """Configuration for performance validation."""

    # Validation settings
    improvement_threshold: float = 0.30  # 30% improvement requirement
    confidence_level: float = 0.95
    min_sample_size: int = 50
    max_sample_size: int = 1000

    # Metrics to validate
    primary_metrics: List[PerformanceMetric] = field(
        default_factory=lambda: [
            PerformanceMetric.COORDINATION_SUCCESS_RATE,
            PerformanceMetric.AVERAGE_REWARD,
            PerformanceMetric.LEARNING_SPEED,
        ]
    )

    secondary_metrics: List[PerformanceMetric] = field(
        default_factory=lambda: [
            PerformanceMetric.RESPONSE_TIME,
            PerformanceMetric.THROUGHPUT,
            PerformanceMetric.RESOURCE_EFFICIENCY,
        ]
    )

    # Performance targets
    target_coordination_success_rate: float = 0.85
    target_average_reward: float = 0.70
    target_response_time_ms: float = 100.0
    target_throughput_ops_per_sec: float = 100.0

    # Baseline comparison
    enable_baseline_comparison: bool = True
    baseline_data_path: Optional[str] = None

    # Statistical settings
    statistical_test: str = "t_test"  # t_test, mann_whitney, wilcoxon
    multiple_comparison_correction: str = "bonferroni"

    def __post_init__(self):
        """Validate configuration."""
        if self.improvement_threshold <= 0:
            raise ValueError("Improvement threshold must be positive")

        if not (0 < self.confidence_level < 1):
            raise ValueError("Confidence level must be between 0 and 1")


@dataclass
class PerformanceResult:
    """Performance validation result."""

    metric: PerformanceMetric
    current_value: float
    baseline_value: Optional[float]
    improvement_percent: Optional[float]
    meets_threshold: bool
    statistical_significance: Optional[float]
    confidence_interval: Optional[Tuple[float, float]]
    sample_size: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metric": self.metric.value,
            "current_value": self.current_value,
            "baseline_value": self.baseline_value,
            "improvement_percent": self.improvement_percent,
            "meets_threshold": self.meets_threshold,
            "statistical_significance": self.statistical_significance,
            "confidence_interval": list(self.confidence_interval)
            if self.confidence_interval
            else None,
            "sample_size": self.sample_size,
        }


class PerformanceValidator:
    """Performance validation system for MARL coordination.

    Validates that MARL coordination achieves the required >30% improvement
    over baseline performance and meets other performance targets.
    """

    def __init__(self, config: PerformanceConfig):
        """
        Initialize performance validator.

        Args:
            config: Performance validation configuration
        """
        self.config = config
        self.logger = get_logger(__name__)

        # Performance data
        self.current_performance_data: Dict[PerformanceMetric, List[float]] = {
            metric: [] for metric in PerformanceMetric
        }
        self.baseline_performance_data: Dict[PerformanceMetric, List[float]] = {
            metric: [] for metric in PerformanceMetric
        }

        # Validation results
        self.validation_results: Dict[PerformanceMetric, PerformanceResult] = {}
        self.overall_validation_result: Optional[Dict[str, Any]] = None

        # Load baseline data if available
        if self.config.baseline_data_path:
            self._load_baseline_data()

        self.logger.info("Performance validator initialized")

    def _load_baseline_data(self) -> None:
        """Load baseline performance data."""
        try:
            import json
            from pathlib import Path

            baseline_path = Path(self.config.baseline_data_path)
            if baseline_path.exists():
                with open(baseline_path, "r") as f:
                    baseline_data = json.load(f)

                for metric_name, values in baseline_data.items():
                    try:
                        metric = PerformanceMetric(metric_name)
                        self.baseline_performance_data[metric] = values
                    except ValueError:
                        self.logger.warning(
                            "Unknown metric in baseline data: %s", metric_name
                        )

                self.logger.info("Loaded baseline data from %s", baseline_path)
            else:
                self.logger.warning("Baseline data file not found: %s", baseline_path)

        except Exception as e:
            self.logger.error("Failed to load baseline data: %s", str(e))

    def add_performance_sample(self, metric: PerformanceMetric, value: float) -> None:
        """Add a performance sample.

        Args:
            metric: Performance metric
            value: Metric value
        """
        self.current_performance_data[metric].append(value)

        # Limit sample size
        if len(self.current_performance_data[metric]) > self.config.max_sample_size:
            self.current_performance_data[metric] = self.current_performance_data[
                metric
            ][-self.config.max_sample_size :]

    def add_performance_batch(
        self, performance_data: Dict[PerformanceMetric, List[float]]
    ) -> None:
        """Add a batch of performance samples.

        Args:
            performance_data: Dictionary of metric to values
        """
        for metric, values in performance_data.items():
            for value in values:
                self.add_performance_sample(metric, value)

    def set_baseline_data(
        self, baseline_data: Dict[PerformanceMetric, List[float]]
    ) -> None:
        """Set baseline performance data.

        Args:
            baseline_data: Dictionary of metric to baseline values
        """
        self.baseline_performance_data = baseline_data.copy()
        self.logger.info("Baseline data set for %d metrics", len(baseline_data))

    async def collect_performance_data(
        self,
        environment: MockMARLEnvironment,
        agent_policies: Dict[str, Callable],
        num_episodes: int = 100,
    ) -> Dict[PerformanceMetric, List[float]]:
        """Collect performance data from environment runs.

        Args:
            environment: MARL environment
            agent_policies: Agent policies
            num_episodes: Number of episodes to run

        Returns:
            Collected performance data
        """
        self.logger.info("Collecting performance data over %d episodes", num_episodes)

        collected_data: Dict[PerformanceMetric, List[float]] = {
            metric: [] for metric in PerformanceMetric
        }

        for episode in range(num_episodes):
            episode_start_time = time.time()

            # Run episode
            episode_result = await environment.run_episode(agent_policies)

            episode_duration = time.time() - episode_start_time

            # Extract performance metrics
            coordination_success_rate = episode_result.get(
                "coordination_success_rate", 0.0
            )
            average_reward = episode_result.get("average_reward", 0.0)
            final_quality = episode_result.get("final_state_quality", 0.0)

            # Calculate derived metrics
            learning_speed = self._calculate_learning_speed(episode_result)
            response_time = episode_duration * 1000  # Convert to ms
            throughput = episode_result.get("total_steps", 0) / max(
                episode_duration, 0.001
            )
            resource_efficiency = self._calculate_resource_efficiency(episode_result)

            # Store metrics
            collected_data[PerformanceMetric.COORDINATION_SUCCESS_RATE].append(
                coordination_success_rate
            )
            collected_data[PerformanceMetric.AVERAGE_REWARD].append(average_reward)
            collected_data[PerformanceMetric.QUALITY_SCORE].append(final_quality)
            collected_data[PerformanceMetric.LEARNING_SPEED].append(learning_speed)
            collected_data[PerformanceMetric.RESPONSE_TIME].append(response_time)
            collected_data[PerformanceMetric.THROUGHPUT].append(throughput)
            collected_data[PerformanceMetric.RESOURCE_EFFICIENCY].append(
                resource_efficiency
            )

            if episode % 20 == 0:
                self.logger.debug("Completed episode %d/%d", episode + 1, num_episodes)

        # Add to current performance data
        self.add_performance_batch(collected_data)

        return collected_data

    def _calculate_learning_speed(self, episode_result: Dict[str, Any]) -> float:
        """Calculate learning speed metric."""
        agent_summaries = episode_result.get("agent_summaries", {})

        if not agent_summaries:
            return 0.0

        # Calculate average reward improvement rate
        learning_speeds = []
        for agent_id, summary in agent_summaries.items():
            rewards = summary.get("rewards", [])
            if len(rewards) >= 10:
                # Calculate trend in last half vs first half
                mid_point = len(rewards) // 2
                first_half_avg = statistics.mean(rewards[:mid_point])
                second_half_avg = statistics.mean(rewards[mid_point:])

                if first_half_avg > 0:
                    improvement_rate = (
                        second_half_avg - first_half_avg
                    ) / first_half_avg
                    learning_speeds.append(max(0.0, improvement_rate))

        return statistics.mean(learning_speeds) if learning_speeds else 0.0

    def _calculate_resource_efficiency(self, episode_result: Dict[str, Any]) -> float:
        """Calculate resource efficiency metric."""
        total_reward = episode_result.get("total_reward", 0.0)
        total_steps = episode_result.get("total_steps", 1)

        # Simple efficiency: reward per step
        efficiency = total_reward / total_steps
        return max(0.0, min(1.0, efficiency))  # Normalize to [0, 1]

    async def validate_performance(self) -> Dict[str, Any]:
        """Validate current performance against requirements.

        Returns:
            Validation results
        """
        self.logger.info("Starting performance validation")

        validation_results = {}
        all_metrics_pass = True

        # Validate each metric
        for metric in self.config.primary_metrics + self.config.secondary_metrics:
            result = await self._validate_metric(metric)
            validation_results[metric.value] = result.to_dict()
            self.validation_results[metric] = result

            # Check if primary metrics pass
            if metric in self.config.primary_metrics and not result.meets_threshold:
                all_metrics_pass = False

        # Overall validation result
        self.overall_validation_result = {
            "overall_pass": all_metrics_pass,
            "validation_timestamp": time.time(),
            "metrics": validation_results,
            "summary": self._generate_validation_summary(),
        }

        self.logger.info(
            "Performance validation complete: %s",
            "PASS" if all_metrics_pass else "FAIL",
        )

        return self.overall_validation_result

    async def _validate_metric(self, metric: PerformanceMetric) -> PerformanceResult:
        """Validate a specific performance metric."""
        current_data = self.current_performance_data[metric]
        baseline_data = self.baseline_performance_data.get(metric, [])

        if len(current_data) < self.config.min_sample_size:
            self.logger.warning(
                "Insufficient data for metric %s: %d samples",
                metric.value,
                len(current_data),
            )
            return PerformanceResult(
                metric=metric,
                current_value=0.0,
                baseline_value=None,
                improvement_percent=None,
                meets_threshold=False,
                statistical_significance=None,
                confidence_interval=None,
                sample_size=len(current_data),
            )

        # Calculate current performance
        current_value = statistics.mean(current_data)

        # Calculate baseline performance
        baseline_value = None
        improvement_percent = None
        statistical_significance = None

        if baseline_data and len(baseline_data) >= self.config.min_sample_size:
            baseline_value = statistics.mean(baseline_data)

            # Calculate improvement
            if baseline_value > 0:
                improvement_percent = (current_value - baseline_value) / baseline_value

            # Statistical significance test
            statistical_significance = self._calculate_statistical_significance(
                current_data, baseline_data
            )

        # Calculate confidence interval
        confidence_interval = self._calculate_confidence_interval(current_data)

        # Check if meets threshold
        meets_threshold = self._check_threshold(
            metric, current_value, improvement_percent
        )

        return PerformanceResult(
            metric=metric,
            current_value=current_value,
            baseline_value=baseline_value,
            improvement_percent=improvement_percent,
            meets_threshold=meets_threshold,
            statistical_significance=statistical_significance,
            confidence_interval=confidence_interval,
            sample_size=len(current_data),
        )

    def _calculate_statistical_significance(
        self, current_data: List[float], baseline_data: List[float]
    ) -> float:
        """Calculate statistical significance of difference."""
        try:
            from scipy import stats

            if self.config.statistical_test == "t_test":
                statistic, p_value = stats.ttest_ind(current_data, baseline_data)
            elif self.config.statistical_test == "mann_whitney":
                statistic, p_value = stats.mannwhitneyu(
                    current_data, baseline_data, alternative="two-sided"
                )
            elif self.config.statistical_test == "wilcoxon":
                # For paired samples (if same length)
                if len(current_data) == len(baseline_data):
                    statistic, p_value = stats.wilcoxon(current_data, baseline_data)
                else:
                    statistic, p_value = stats.mannwhitneyu(
                        current_data, baseline_data, alternative="two-sided"
                    )
            else:
                # Default to t-test
                statistic, p_value = stats.ttest_ind(current_data, baseline_data)

            return p_value

        except ImportError:
            self.logger.warning("scipy not available, using simple statistical test")
            return self._simple_statistical_test(current_data, baseline_data)
        except Exception as e:
            self.logger.error("Statistical test failed: %s", str(e))
            return 1.0  # No significance

    def _simple_statistical_test(
        self, current_data: List[float], baseline_data: List[float]
    ) -> float:
        """Simple statistical test without scipy."""
        # Simple t-test approximation
        current_mean = statistics.mean(current_data)
        baseline_mean = statistics.mean(baseline_data)

        current_std = statistics.stdev(current_data) if len(current_data) > 1 else 0.0
        baseline_std = (
            statistics.stdev(baseline_data) if len(baseline_data) > 1 else 0.0
        )

        # Pooled standard error
        n1, n2 = len(current_data), len(baseline_data)
        pooled_se = ((current_std**2 / n1) + (baseline_std**2 / n2)) ** 0.5

        if pooled_se == 0:
            return 0.0 if current_mean != baseline_mean else 1.0

        # t-statistic
        t_stat = abs(current_mean - baseline_mean) / pooled_se

        # Approximate p-value (very rough)
        if t_stat > 2.0:
            return 0.05
        elif t_stat > 1.5:
            return 0.1
        else:
            return 0.2

    def _calculate_confidence_interval(self, data: List[float]) -> Tuple[float, float]:
        """Calculate confidence interval for data."""
        if len(data) < 2:
            mean_val = data[0] if data else 0.0
            return (mean_val, mean_val)

        mean_val = statistics.mean(data)
        std_val = statistics.stdev(data)
        n = len(data)

        # t-distribution critical value (approximation)
        alpha = 1 - self.config.confidence_level
        if n > 30:
            t_critical = 1.96  # Normal approximation
        else:
            # Rough t-distribution values
            t_values = {10: 2.23, 20: 2.09, 30: 2.04}
            t_critical = 2.23  # Conservative estimate
            for df, t_val in t_values.items():
                if n >= df:
                    t_critical = t_val
                    break

        margin_error = t_critical * (std_val / (n**0.5))

        return (mean_val - margin_error, mean_val + margin_error)

    def _check_threshold(
        self,
        metric: PerformanceMetric,
        current_value: float,
        improvement_percent: Optional[float],
    ) -> bool:
        """Check if metric meets threshold requirements."""
        # Check improvement threshold
        if improvement_percent is not None:
            if improvement_percent < self.config.improvement_threshold:
                return False

        # Check absolute thresholds
        if metric == PerformanceMetric.COORDINATION_SUCCESS_RATE:
            return current_value >= self.config.target_coordination_success_rate
        elif metric == PerformanceMetric.AVERAGE_REWARD:
            return current_value >= self.config.target_average_reward
        elif metric == PerformanceMetric.RESPONSE_TIME:
            return current_value <= self.config.target_response_time_ms
        elif metric == PerformanceMetric.THROUGHPUT:
            return current_value >= self.config.target_throughput_ops_per_sec
        else:
            # For other metrics, just check improvement
            return improvement_percent is None or improvement_percent >= 0

    def _generate_validation_summary(self) -> Dict[str, Any]:
        """Generate validation summary."""
        primary_results = [
            self.validation_results[metric]
            for metric in self.config.primary_metrics
            if metric in self.validation_results
        ]

        secondary_results = [
            self.validation_results[metric]
            for metric in self.config.secondary_metrics
            if metric in self.validation_results
        ]

        primary_pass_rate = sum(1 for r in primary_results if r.meets_threshold) / max(
            1, len(primary_results)
        )
        secondary_pass_rate = sum(
            1 for r in secondary_results if r.meets_threshold
        ) / max(1, len(secondary_results))

        # Calculate average improvement
        improvements = [
            r.improvement_percent
            for r in primary_results + secondary_results
            if r.improvement_percent is not None
        ]
        avg_improvement = statistics.mean(improvements) if improvements else 0.0

        return {
            "primary_metrics_pass_rate": primary_pass_rate,
            "secondary_metrics_pass_rate": secondary_pass_rate,
            "average_improvement_percent": avg_improvement,
            "meets_30_percent_threshold": avg_improvement
            >= self.config.improvement_threshold,
            "total_samples": sum(
                len(data) for data in self.current_performance_data.values()
            ),
            "validation_confidence": self.config.confidence_level,
        }

    async def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report.

        Returns:
            Performance report
        """
        if not self.overall_validation_result:
            await self.validate_performance()

        report = {
            "report_timestamp": time.time(),
            "validation_result": self.overall_validation_result,
            "detailed_metrics": {},
            "statistical_analysis": {},
            "recommendations": [],
        }

        # Detailed metrics analysis
        for metric, result in self.validation_results.items():
            current_data = self.current_performance_data[metric]

            if len(current_data) > 0:
                report["detailed_metrics"][metric.value] = {
                    "current_performance": {
                        "mean": statistics.mean(current_data),
                        "median": statistics.median(current_data),
                        "std_dev": statistics.stdev(current_data)
                        if len(current_data) > 1
                        else 0.0,
                        "min": min(current_data),
                        "max": max(current_data),
                        "sample_size": len(current_data),
                    },
                    "validation_result": result.to_dict(),
                }

        # Statistical analysis
        report["statistical_analysis"] = {
            "confidence_level": self.config.confidence_level,
            "statistical_test": self.config.statistical_test,
            "improvement_threshold": self.config.improvement_threshold,
            "sample_sizes_adequate": all(
                len(data) >= self.config.min_sample_size
                for data in self.current_performance_data.values()
            ),
        }

        # Generate recommendations
        report["recommendations"] = self._generate_recommendations()

        return report

    def _generate_recommendations(self) -> List[str]:
        """Generate performance improvement recommendations."""
        recommendations = []

        for metric, result in self.validation_results.items():
            if not result.meets_threshold:
                if metric == PerformanceMetric.COORDINATION_SUCCESS_RATE:
                    recommendations.append(
                        "Improve coordination success rate by optimizing consensus mechanisms "
                        "and reducing coordination timeouts"
                    )
                elif metric == PerformanceMetric.AVERAGE_REWARD:
                    recommendations.append(
                        "Enhance reward optimization by improving agent learning algorithms "
                        "and reward function design"
                    )
                elif metric == PerformanceMetric.LEARNING_SPEED:
                    recommendations.append(
                        "Accelerate learning by implementing experience sharing and "
                        "curriculum learning strategies"
                    )
                elif metric == PerformanceMetric.RESPONSE_TIME:
                    recommendations.append(
                        "Reduce response time by optimizing coordination algorithms "
                        "and implementing parallel processing"
                    )
                elif metric == PerformanceMetric.THROUGHPUT:
                    recommendations.append(
                        "Increase throughput by implementing batch processing and "
                        "reducing coordination overhead"
                    )

        # General recommendations
        if len(recommendations) == 0:
            recommendations.append(
                "Performance meets all requirements. Continue monitoring."
            )
        else:
            recommendations.append(
                "Consider implementing distributed coordination to improve scalability"
            )
            recommendations.append(
                "Monitor performance continuously and adjust thresholds as needed"
            )

        return recommendations

    def save_baseline_data(self, filepath: str) -> None:
        """Save current performance data as baseline.

        Args:
            filepath: Path to save baseline data
        """
        try:
            import json
            from pathlib import Path

            baseline_data = {
                metric.value: data
                for metric, data in self.current_performance_data.items()
                if len(data) > 0
            }

            Path(filepath).parent.mkdir(parents=True, exist_ok=True)

            with open(filepath, "w") as f:
                json.dump(baseline_data, f, indent=2)

            self.logger.info("Saved baseline data to %s", filepath)

        except Exception as e:
            self.logger.error("Failed to save baseline data: %s", str(e))

    def clear_performance_data(self) -> None:
        """Clear all performance data."""
        for metric in PerformanceMetric:
            self.current_performance_data[metric].clear()

        self.validation_results.clear()
        self.overall_validation_result = None

        self.logger.info("Performance data cleared")

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance data summary.

        Returns:
            Performance summary
        """
        summary = {
            "data_points": {
                metric.value: len(data)
                for metric, data in self.current_performance_data.items()
            },
            "current_averages": {
                metric.value: statistics.mean(data) if data else 0.0
                for metric, data in self.current_performance_data.items()
            },
            "baseline_averages": {
                metric.value: statistics.mean(data) if data else 0.0
                for metric, data in self.baseline_performance_data.items()
            },
            "validation_status": {
                "completed": self.overall_validation_result is not None,
                "overall_pass": self.overall_validation_result.get(
                    "overall_pass", False
                )
                if self.overall_validation_result
                else False,
            },
        }

        return summary
