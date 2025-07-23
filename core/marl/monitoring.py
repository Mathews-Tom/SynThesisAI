"""
MARL Performance Monitoring Infrastructure

This module provides comprehensive monitoring and metrics collection for
multi-agent reinforcement learning components, following the development
standards for performance tracking and analysis.
"""

import json
import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

from .config import MARLConfig
from .exceptions import MARLError

logger = logging.getLogger(__name__)


@dataclass
class CoordinationMetrics:
    """Metrics for coordination performance tracking."""

    total_episodes: int = 0
    successful_episodes: int = 0
    failed_episodes: int = 0
    total_coordination_time: float = 0.0
    coordination_times: Deque[float] = field(default_factory=lambda: deque(maxlen=1000))

    @property
    def success_rate(self) -> float:
        """Calculate coordination success rate."""
        if self.total_episodes == 0:
            return 0.0
        return self.successful_episodes / self.total_episodes

    @property
    def average_time(self) -> float:
        """Calculate average coordination time."""
        if not self.coordination_times:
            return 0.0
        return sum(self.coordination_times) / len(self.coordination_times)

    def record_success(self, coordination_time: float) -> None:
        """Record a successful coordination episode."""
        self.total_episodes += 1
        self.successful_episodes += 1
        self.total_coordination_time += coordination_time
        self.coordination_times.append(coordination_time)

    def record_failure(self, coordination_time: float) -> None:
        """Record a failed coordination episode."""
        self.total_episodes += 1
        self.failed_episodes += 1
        self.total_coordination_time += coordination_time
        self.coordination_times.append(coordination_time)


@dataclass
class AgentMetrics:
    """Metrics for individual agent performance tracking."""

    agent_id: str = ""
    total_actions: int = 0
    successful_actions: int = 0
    total_reward: float = 0.0
    episode_rewards: Deque[float] = field(default_factory=lambda: deque(maxlen=1000))
    learning_losses: Deque[float] = field(default_factory=lambda: deque(maxlen=1000))
    epsilon_values: Deque[float] = field(default_factory=lambda: deque(maxlen=1000))

    @property
    def success_rate(self) -> float:
        """Calculate agent action success rate."""
        if self.total_actions == 0:
            return 0.0
        return self.successful_actions / self.total_actions

    @property
    def average_reward(self) -> float:
        """Calculate average episode reward."""
        if not self.episode_rewards:
            return 0.0
        return sum(self.episode_rewards) / len(self.episode_rewards)

    @property
    def learning_progress(self) -> float:
        """Calculate learning progress based on reward trend."""
        if len(self.episode_rewards) < 10:
            return 0.0

        # Calculate trend over last 100 episodes
        recent_rewards = list(self.episode_rewards)[-100:]
        if len(recent_rewards) < 10:
            return 0.0

        # Simple linear trend calculation
        n = len(recent_rewards)
        x_sum = sum(range(n))
        y_sum = sum(recent_rewards)
        xy_sum = sum(i * reward for i, reward in enumerate(recent_rewards))
        x2_sum = sum(i * i for i in range(n))

        # Calculate slope (learning progress)
        denominator = n * x2_sum - x_sum * x_sum
        if denominator == 0:
            return 0.0

        slope = (n * xy_sum - x_sum * y_sum) / denominator
        return max(0.0, min(1.0, slope + 0.5))  # Normalize to [0, 1]

    def update(
        self,
        action_success: bool,
        reward: float,
        loss: Optional[float] = None,
        epsilon: Optional[float] = None,
    ) -> None:
        """Update agent metrics with new data."""
        self.total_actions += 1
        if action_success:
            self.successful_actions += 1

        self.total_reward += reward
        self.episode_rewards.append(reward)

        if loss is not None:
            self.learning_losses.append(loss)

        if epsilon is not None:
            self.epsilon_values.append(epsilon)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of agent metrics."""
        return {
            "agent_id": self.agent_id,
            "total_actions": self.total_actions,
            "success_rate": self.success_rate,
            "average_reward": self.average_reward,
            "learning_progress": self.learning_progress,
            "current_epsilon": self.epsilon_values[-1] if self.epsilon_values else 0.0,
            "recent_loss": self.learning_losses[-1] if self.learning_losses else 0.0,
        }


@dataclass
class SystemMetrics:
    """System-wide performance metrics."""

    start_time: datetime = field(default_factory=datetime.now)
    total_requests: int = 0
    successful_requests: int = 0
    total_processing_time: float = 0.0
    memory_usage: Deque[float] = field(default_factory=lambda: deque(maxlen=100))
    cpu_usage: Deque[float] = field(default_factory=lambda: deque(maxlen=100))

    @property
    def uptime(self) -> timedelta:
        """Calculate system uptime."""
        return datetime.now() - self.start_time

    @property
    def success_rate(self) -> float:
        """Calculate overall system success rate."""
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests

    @property
    def average_processing_time(self) -> float:
        """Calculate average request processing time."""
        if self.total_requests == 0:
            return 0.0
        return self.total_processing_time / self.total_requests

    def update(
        self,
        request_success: bool,
        processing_time: float,
        memory_mb: Optional[float] = None,
        cpu_percent: Optional[float] = None,
    ) -> None:
        """Update system metrics."""
        self.total_requests += 1
        if request_success:
            self.successful_requests += 1

        self.total_processing_time += processing_time

        if memory_mb is not None:
            self.memory_usage.append(memory_mb)

        if cpu_percent is not None:
            self.cpu_usage.append(cpu_percent)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of system metrics."""
        return {
            "uptime_hours": self.uptime.total_seconds() / 3600,
            "total_requests": self.total_requests,
            "success_rate": self.success_rate,
            "average_processing_time": self.average_processing_time,
            "current_memory_mb": self.memory_usage[-1] if self.memory_usage else 0.0,
            "current_cpu_percent": self.cpu_usage[-1] if self.cpu_usage else 0.0,
        }


class MARLPerformanceMonitor:
    """Comprehensive performance monitoring for MARL system."""

    def __init__(self, config: Optional[MARLConfig] = None):
        """
        Initialize MARL performance monitor.

        Args:
            config: MARL configuration for monitoring settings
        """
        self.config = config
        self.coordination_metrics = CoordinationMetrics()
        self.agent_metrics = {
            "generator": AgentMetrics(agent_id="generator"),
            "validator": AgentMetrics(agent_id="validator"),
            "curriculum": AgentMetrics(agent_id="curriculum"),
        }
        self.system_metrics = SystemMetrics()

        # Performance tracking
        self.episode_count = 0
        self.last_report_time = time.time()
        self.performance_history = []

        # Thread safety
        self._lock = threading.Lock()

        logger.info("MARL performance monitor initialized")

    def record_coordination_episode(self, episode_data: Dict[str, Any]) -> None:
        """Record coordination episode for analysis."""
        with self._lock:
            coordination_success = episode_data.get("coordination_success", False)
            coordination_time = episode_data.get("coordination_time", 0.0)

            # Update coordination metrics
            if coordination_success:
                self.coordination_metrics.record_success(coordination_time)
            else:
                self.coordination_metrics.record_failure(coordination_time)

            # Update agent metrics
            for agent_id, agent_data in episode_data.get("agent_data", {}).items():
                if agent_id in self.agent_metrics:
                    self.agent_metrics[agent_id].update(
                        action_success=agent_data.get("action_success", False),
                        reward=agent_data.get("reward", 0.0),
                        loss=agent_data.get("loss"),
                        epsilon=agent_data.get("epsilon"),
                    )

            # Update system metrics
            system_data = episode_data.get("system_metrics", {})
            self.system_metrics.update(
                request_success=coordination_success,
                processing_time=coordination_time,
                memory_mb=system_data.get("memory_mb"),
                cpu_percent=system_data.get("cpu_percent"),
            )

            self.episode_count += 1

            # Generate periodic reports
            if (
                self.config
                and self.config.monitoring_enabled
                and self.episode_count % self.config.performance_report_interval == 0
            ):
                self._generate_performance_report()

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        with self._lock:
            report = {
                "timestamp": datetime.now().isoformat(),
                "episode_count": self.episode_count,
                "coordination_metrics": {
                    "success_rate": self.coordination_metrics.success_rate,
                    "average_time": self.coordination_metrics.average_time,
                    "total_episodes": self.coordination_metrics.total_episodes,
                },
                "agent_performance": {
                    agent_id: metrics.get_summary()
                    for agent_id, metrics in self.agent_metrics.items()
                },
                "system_performance": self.system_metrics.get_summary(),
                "improvement_recommendations": self.generate_recommendations(),
            }

            return report

    def generate_recommendations(self) -> List[str]:
        """Generate performance improvement recommendations."""
        recommendations = []

        # Check coordination success rate
        if self.coordination_metrics.success_rate < 0.85:
            recommendations.append(
                f"Coordination success rate ({self.coordination_metrics.success_rate:.2f}) "
                f"below target (0.85). Consider adjusting consensus mechanisms or "
                f"agent communication protocols."
            )

        # Check agent learning progress
        for agent_id, metrics in self.agent_metrics.items():
            if metrics.learning_progress < 0.1:
                recommendations.append(
                    f"{agent_id} agent showing slow learning progress "
                    f"({metrics.learning_progress:.2f}). Consider adjusting learning "
                    f"rate or reward function."
                )

            if metrics.success_rate < 0.7:
                recommendations.append(
                    f"{agent_id} agent success rate ({metrics.success_rate:.2f}) "
                    f"below expected threshold. Review action selection strategy."
                )

        # Check system performance
        if self.system_metrics.success_rate < 0.9:
            recommendations.append(
                f"System success rate ({self.system_metrics.success_rate:.2f}) "
                f"below target. Investigate system-level issues."
            )

        # Check coordination time
        if self.coordination_metrics.average_time > 10.0:
            recommendations.append(
                f"Average coordination time ({self.coordination_metrics.average_time:.2f}s) "
                f"exceeds target. Consider optimizing coordination algorithms."
            )

        return recommendations

    def _generate_performance_report(self) -> None:
        """Generate and log periodic performance report."""
        try:
            report = self.get_performance_report()

            # Log key metrics
            logger.info(
                "Performance Report - Episode %d: Coordination Success: %.2f, "
                "Avg Time: %.2fs, System Success: %.2f",
                self.episode_count,
                report["coordination_metrics"]["success_rate"],
                report["coordination_metrics"]["average_time"],
                report["system_performance"]["success_rate"],
            )

            # Log recommendations if any
            recommendations = report["improvement_recommendations"]
            if recommendations:
                logger.warning(
                    "Performance recommendations: %s", "; ".join(recommendations)
                )

            # Store in history
            self.performance_history.append(report)

            # Keep only recent history
            if len(self.performance_history) > 100:
                self.performance_history = self.performance_history[-100:]

        except Exception as e:
            logger.error("Failed to generate performance report: %s", str(e))

    def export_metrics(self, export_path: Path) -> None:
        """Export metrics to file for external analysis."""
        try:
            export_path.parent.mkdir(parents=True, exist_ok=True)

            export_data = {
                "export_timestamp": datetime.now().isoformat(),
                "performance_report": self.get_performance_report(),
                "performance_history": self.performance_history[
                    -50:
                ],  # Last 50 reports
            }

            with open(export_path, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2, default=str)

            logger.info("Metrics exported to %s", export_path)

        except Exception as e:
            logger.error("Failed to export metrics to %s: %s", export_path, str(e))
            raise MARLError(f"Metrics export failed: {str(e)}") from e

    def reset_metrics(self) -> None:
        """Reset all metrics (useful for testing or retraining)."""
        with self._lock:
            self.coordination_metrics = CoordinationMetrics()
            self.agent_metrics = {
                "generator": AgentMetrics(agent_id="generator"),
                "validator": AgentMetrics(agent_id="validator"),
                "curriculum": AgentMetrics(agent_id="curriculum"),
            }
            self.system_metrics = SystemMetrics()
            self.episode_count = 0
            self.performance_history.clear()

            logger.info("All MARL metrics reset")

    def get_real_time_status(self) -> Dict[str, Any]:
        """Get real-time system status for dashboards."""
        with self._lock:
            return {
                "timestamp": datetime.now().isoformat(),
                "system_status": "healthy"
                if self.system_metrics.success_rate > 0.9
                else "degraded",
                "coordination_success_rate": self.coordination_metrics.success_rate,
                "active_episodes": self.episode_count,
                "average_coordination_time": self.coordination_metrics.average_time,
                "agent_status": {
                    agent_id: {
                        "success_rate": metrics.success_rate,
                        "learning_progress": metrics.learning_progress,
                        "recent_reward": metrics.episode_rewards[-1]
                        if metrics.episode_rewards
                        else 0.0,
                    }
                    for agent_id, metrics in self.agent_metrics.items()
                },
            }
