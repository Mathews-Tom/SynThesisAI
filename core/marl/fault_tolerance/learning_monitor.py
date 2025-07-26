"""
Learning Monitoring and Divergence Detection.

This module provides monitoring capabilities for detecting learning
divergence and other learning-related issues in the MARL system.
"""

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from utils.logging_config import get_logger


class LearningStatus(Enum):
    """Learning status enumeration."""

    NORMAL = "normal"
    SLOW_CONVERGENCE = "slow_convergence"
    DIVERGING = "diverging"
    OSCILLATING = "oscillating"
    STAGNANT = "stagnant"
    UNSTABLE = "unstable"
    FAILED = "failed"


@dataclass
class LearningMetrics:
    """Learning metrics for an agent."""

    agent_id: str
    episode: int = 0
    total_reward: float = 0.0
    average_reward: float = 0.0
    loss: float = 0.0
    learning_rate: float = 0.001
    epsilon: float = 0.1

    # Performance tracking
    reward_history: deque = field(default_factory=lambda: deque(maxlen=100))
    loss_history: deque = field(default_factory=lambda: deque(maxlen=100))
    q_value_history: deque = field(default_factory=lambda: deque(maxlen=100))

    # Convergence metrics
    reward_variance: float = 0.0
    loss_variance: float = 0.0
    reward_trend: float = 0.0  # Positive = improving, negative = degrading
    convergence_score: float = 0.0

    # Timestamps
    last_update: Optional[datetime] = None
    learning_start_time: Optional[datetime] = None

    def update_metrics(
        self,
        episode: int,
        reward: float,
        loss: float,
        q_values: Optional[List[float]] = None,
    ) -> None:
        """Update learning metrics."""
        self.episode = episode
        self.total_reward += reward
        self.loss = loss
        self.last_update = datetime.now()

        if not self.learning_start_time:
            self.learning_start_time = datetime.now()

        # Update histories
        self.reward_history.append(reward)
        self.loss_history.append(loss)

        if q_values:
            avg_q_value = np.mean(q_values)
            self.q_value_history.append(avg_q_value)

        # Calculate running averages
        if len(self.reward_history) > 0:
            self.average_reward = np.mean(self.reward_history)
            self.reward_variance = np.var(self.reward_history)

        if len(self.loss_history) > 0:
            self.loss_variance = np.var(self.loss_history)

        # Calculate trend
        self._calculate_reward_trend()

        # Calculate convergence score
        self._calculate_convergence_score()

    def _calculate_reward_trend(self) -> None:
        """Calculate reward trend using linear regression."""
        if len(self.reward_history) < 10:
            self.reward_trend = 0.0
            return

        rewards = np.array(list(self.reward_history))
        x = np.arange(len(rewards))

        # Simple linear regression
        n = len(rewards)
        sum_x = np.sum(x)
        sum_y = np.sum(rewards)
        sum_xy = np.sum(x * rewards)
        sum_x2 = np.sum(x * x)

        if n * sum_x2 - sum_x * sum_x != 0:
            self.reward_trend = (n * sum_xy - sum_x * sum_y) / (
                n * sum_x2 - sum_x * sum_x
            )
        else:
            self.reward_trend = 0.0

    def _calculate_convergence_score(self) -> None:
        """Calculate convergence score (0.0 = not converged, 1.0 = converged)."""
        if len(self.reward_history) < 20:
            self.convergence_score = 0.0
            return

        # Check stability (low variance)
        stability_score = max(
            0, 1.0 - self.reward_variance / max(abs(self.average_reward), 1.0)
        )

        # Check positive trend
        trend_score = max(0, min(1.0, self.reward_trend * 10))  # Scale trend

        # Check recent performance
        recent_rewards = list(self.reward_history)[-10:]
        recent_avg = np.mean(recent_rewards)
        overall_avg = self.average_reward

        performance_score = max(0, min(1.0, recent_avg / max(overall_avg, 0.1)))

        # Combine scores
        self.convergence_score = (
            stability_score + trend_score + performance_score
        ) / 3.0

    def get_learning_duration(self) -> float:
        """Get learning duration in seconds."""
        if not self.learning_start_time:
            return 0.0

        return (datetime.now() - self.learning_start_time).total_seconds()

    def is_diverging(self, divergence_threshold: float = -0.01) -> bool:
        """Check if learning is diverging."""
        return (
            self.reward_trend < divergence_threshold and len(self.reward_history) >= 20
        )

    def is_oscillating(self, oscillation_threshold: float = 2.0) -> bool:
        """Check if learning is oscillating."""
        if len(self.reward_history) < 20:
            return False

        # Check if variance is high relative to mean
        if abs(self.average_reward) < 0.1:
            return self.reward_variance > oscillation_threshold
        else:
            return (
                self.reward_variance / abs(self.average_reward) > oscillation_threshold
            )

    def is_stagnant(self, stagnation_threshold: float = 0.001) -> bool:
        """Check if learning is stagnant."""
        return (
            abs(self.reward_trend) < stagnation_threshold
            and len(self.reward_history) >= 30
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "agent_id": self.agent_id,
            "episode": self.episode,
            "total_reward": self.total_reward,
            "average_reward": self.average_reward,
            "loss": self.loss,
            "learning_rate": self.learning_rate,
            "epsilon": self.epsilon,
            "reward_variance": self.reward_variance,
            "loss_variance": self.loss_variance,
            "reward_trend": self.reward_trend,
            "convergence_score": self.convergence_score,
            "learning_duration": self.get_learning_duration(),
            "is_diverging": self.is_diverging(),
            "is_oscillating": self.is_oscillating(),
            "is_stagnant": self.is_stagnant(),
        }


@dataclass
class LearningDivergenceEvent:
    """Represents a learning divergence event."""

    event_id: str
    agent_id: str
    divergence_type: LearningStatus
    detection_time: datetime = field(default_factory=datetime.now)
    resolution_time: Optional[datetime] = None
    resolved: bool = False

    # Metrics at detection
    reward_trend: float = 0.0
    reward_variance: float = 0.0
    convergence_score: float = 0.0
    episode_at_detection: int = 0

    # Resolution info
    resolution_strategy: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_id": self.event_id,
            "agent_id": self.agent_id,
            "divergence_type": self.divergence_type.value,
            "detection_time": self.detection_time.isoformat(),
            "resolution_time": self.resolution_time.isoformat()
            if self.resolution_time
            else None,
            "resolved": self.resolved,
            "reward_trend": self.reward_trend,
            "reward_variance": self.reward_variance,
            "convergence_score": self.convergence_score,
            "episode_at_detection": self.episode_at_detection,
            "resolution_strategy": self.resolution_strategy,
            "duration_seconds": (
                (self.resolution_time or datetime.now()) - self.detection_time
            ).total_seconds(),
            "metadata": self.metadata,
        }


class LearningDivergenceDetector:
    """
    Detects learning divergence and other learning issues.

    Monitors learning metrics and identifies patterns that indicate
    learning problems such as divergence, oscillation, or stagnation.
    """

    def __init__(
        self,
        divergence_threshold: float = -0.01,
        oscillation_threshold: float = 2.0,
        stagnation_threshold: float = 0.001,
        min_episodes_for_detection: int = 20,
    ):
        """
        Initialize divergence detector.

        Args:
            divergence_threshold: Threshold for detecting divergence
            oscillation_threshold: Threshold for detecting oscillation
            stagnation_threshold: Threshold for detecting stagnation
            min_episodes_for_detection: Minimum episodes before detection
        """
        self.logger = get_logger(__name__)

        # Configuration
        self.divergence_threshold = divergence_threshold
        self.oscillation_threshold = oscillation_threshold
        self.stagnation_threshold = stagnation_threshold
        self.min_episodes_for_detection = min_episodes_for_detection

        # Event tracking
        self.divergence_events: Dict[str, LearningDivergenceEvent] = {}
        self.event_history: List[LearningDivergenceEvent] = []
        self.event_counter = 0

        # Callbacks
        self.divergence_callbacks: List[Callable] = []
        self.resolution_callbacks: List[Callable] = []

        self.logger.info("Learning divergence detector initialized")

    def detect_divergence(
        self, metrics: LearningMetrics
    ) -> Optional[LearningDivergenceEvent]:
        """Detect learning divergence from metrics."""
        if metrics.episode < self.min_episodes_for_detection:
            return None

        # Check for different types of divergence
        divergence_type = None

        if metrics.is_diverging(self.divergence_threshold):
            divergence_type = LearningStatus.DIVERGING
        elif metrics.is_oscillating(self.oscillation_threshold):
            divergence_type = LearningStatus.OSCILLATING
        elif metrics.is_stagnant(self.stagnation_threshold):
            divergence_type = LearningStatus.STAGNANT
        elif metrics.convergence_score < 0.2:
            divergence_type = LearningStatus.UNSTABLE
        elif metrics.reward_trend < 0 and metrics.convergence_score < 0.5:
            divergence_type = LearningStatus.SLOW_CONVERGENCE

        if divergence_type:
            # Check if we already have an active event for this agent
            existing_event = None
            for event in self.divergence_events.values():
                if event.agent_id == metrics.agent_id and not event.resolved:
                    existing_event = event
                    break

            if not existing_event:
                # Create new divergence event
                event_id = f"divergence_{self.event_counter}"
                self.event_counter += 1

                event = LearningDivergenceEvent(
                    event_id=event_id,
                    agent_id=metrics.agent_id,
                    divergence_type=divergence_type,
                    reward_trend=metrics.reward_trend,
                    reward_variance=metrics.reward_variance,
                    convergence_score=metrics.convergence_score,
                    episode_at_detection=metrics.episode,
                )

                self.divergence_events[event_id] = event

                self.logger.warning(
                    "Learning divergence detected: %s for agent %s (type: %s)",
                    event_id,
                    metrics.agent_id,
                    divergence_type.value,
                )

                return event

        return None

    async def notify_divergence(self, event: LearningDivergenceEvent) -> None:
        """Notify divergence callbacks."""
        for callback in self.divergence_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
            except Exception as e:
                self.logger.error("Error in divergence callback: %s", str(e))

    def resolve_divergence(
        self,
        event_id: str,
        resolution_strategy: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Mark a divergence event as resolved."""
        if event_id in self.divergence_events:
            event = self.divergence_events[event_id]
            event.resolved = True
            event.resolution_time = datetime.now()
            event.resolution_strategy = resolution_strategy
            event.metadata.update(metadata or {})

            # Move to history
            self.event_history.append(event)
            del self.divergence_events[event_id]

            self.logger.info(
                "Learning divergence resolved: %s using strategy: %s",
                event_id,
                resolution_strategy,
            )

            return True

        return False

    async def notify_resolution(self, event: LearningDivergenceEvent) -> None:
        """Notify resolution callbacks."""
        for callback in self.resolution_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
            except Exception as e:
                self.logger.error("Error in resolution callback: %s", str(e))

    def get_active_divergences(self) -> List[LearningDivergenceEvent]:
        """Get list of active divergence events."""
        return list(self.divergence_events.values())

    def get_divergence_history(self) -> List[LearningDivergenceEvent]:
        """Get divergence event history."""
        return self.event_history.copy()

    def get_divergence_statistics(self) -> Dict[str, Any]:
        """Get divergence statistics."""
        total_events = len(self.event_history) + len(self.divergence_events)
        resolved_events = len(self.event_history)

        # Count by type
        type_counts = {}
        for event in self.event_history + list(self.divergence_events.values()):
            event_type = event.divergence_type.value
            type_counts[event_type] = type_counts.get(event_type, 0) + 1

        # Calculate average resolution time
        resolution_times = [
            (event.resolution_time - event.detection_time).total_seconds()
            for event in self.event_history
            if event.resolution_time
        ]

        avg_resolution_time = (
            sum(resolution_times) / len(resolution_times) if resolution_times else 0.0
        )

        return {
            "total_divergence_events": total_events,
            "active_divergences": len(self.divergence_events),
            "resolved_divergences": resolved_events,
            "resolution_rate": resolved_events / total_events
            if total_events > 0
            else 0.0,
            "average_resolution_time": avg_resolution_time,
            "divergences_by_type": type_counts,
        }

    def add_divergence_callback(self, callback: Callable) -> None:
        """Add divergence detection callback."""
        self.divergence_callbacks.append(callback)

    def add_resolution_callback(self, callback: Callable) -> None:
        """Add divergence resolution callback."""
        self.resolution_callbacks.append(callback)

    def remove_divergence_callback(self, callback: Callable) -> None:
        """Remove divergence detection callback."""
        if callback in self.divergence_callbacks:
            self.divergence_callbacks.remove(callback)

    def remove_resolution_callback(self, callback: Callable) -> None:
        """Remove divergence resolution callback."""
        if callback in self.resolution_callbacks:
            self.resolution_callbacks.remove(callback)


class LearningMonitor:
    """
    Comprehensive learning monitoring system.

    Monitors learning progress, detects issues, and provides
    insights for improving learning performance.
    """

    def __init__(
        self, monitoring_interval: float = 10.0, enable_auto_correction: bool = True
    ):
        """
        Initialize learning monitor.

        Args:
            monitoring_interval: Interval between monitoring checks (seconds)
            enable_auto_correction: Enable automatic correction of learning issues
        """
        self.logger = get_logger(__name__)

        # Configuration
        self.monitoring_interval = monitoring_interval
        self.enable_auto_correction = enable_auto_correction

        # Components
        self.divergence_detector = LearningDivergenceDetector()

        # Agent tracking
        self.agent_metrics: Dict[str, LearningMetrics] = {}
        self.monitored_agents: List[str] = []

        # Monitoring state
        self.is_monitoring = False
        self.monitoring_task: Optional[asyncio.Task] = None

        # Callbacks
        self.issue_callbacks: List[Callable] = []
        self.correction_callbacks: List[Callable] = []

        # Setup divergence detector callbacks
        self.divergence_detector.add_divergence_callback(self._handle_divergence)

        self.logger.info("Learning monitor initialized")

    def register_agent(self, agent_id: str) -> None:
        """Register an agent for learning monitoring."""
        if agent_id not in self.agent_metrics:
            self.agent_metrics[agent_id] = LearningMetrics(agent_id=agent_id)
            self.monitored_agents.append(agent_id)

            self.logger.info("Registered agent for learning monitoring: %s", agent_id)

    def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent from learning monitoring."""
        if agent_id in self.agent_metrics:
            del self.agent_metrics[agent_id]

        if agent_id in self.monitored_agents:
            self.monitored_agents.remove(agent_id)

        self.logger.info("Unregistered agent from learning monitoring: %s", agent_id)

    async def start_monitoring(self) -> None:
        """Start learning monitoring."""
        if self.is_monitoring:
            self.logger.warning("Learning monitoring already running")
            return

        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())

        self.logger.info("Learning monitoring started")

    async def stop_monitoring(self) -> None:
        """Stop learning monitoring."""
        if not self.is_monitoring:
            return

        self.is_monitoring = False

        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass

        self.logger.info("Learning monitoring stopped")

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                await self._check_all_agents()
                await asyncio.sleep(self.monitoring_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(
                    "Error in learning monitoring loop: %s", str(e), exc_info=True
                )
                await asyncio.sleep(5.0)  # Wait before retrying

    async def _check_all_agents(self) -> None:
        """Check learning status of all monitored agents."""
        for agent_id in self.monitored_agents:
            try:
                await self._check_agent_learning(agent_id)
            except Exception as e:
                self.logger.error(
                    "Error checking learning for agent %s: %s", agent_id, str(e)
                )

    async def _check_agent_learning(self, agent_id: str) -> None:
        """Check learning status of a specific agent."""
        if agent_id not in self.agent_metrics:
            return

        metrics = self.agent_metrics[agent_id]

        # Check for divergence
        divergence_event = self.divergence_detector.detect_divergence(metrics)

        if divergence_event:
            await self.divergence_detector.notify_divergence(divergence_event)

    async def _handle_divergence(self, event: LearningDivergenceEvent) -> None:
        """Handle detected learning divergence."""
        self.logger.warning(
            "Learning divergence detected for agent %s: %s",
            event.agent_id,
            event.divergence_type.value,
        )

        # Notify issue callbacks
        for callback in self.issue_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
            except Exception as e:
                self.logger.error("Error in issue callback: %s", str(e))

        # Attempt auto-correction if enabled
        if self.enable_auto_correction:
            await self._attempt_learning_correction(event)

    async def _attempt_learning_correction(
        self, event: LearningDivergenceEvent
    ) -> None:
        """Attempt to correct learning issues."""
        try:
            correction_strategy = None

            if event.divergence_type == LearningStatus.DIVERGING:
                correction_strategy = "reset_learning_parameters"
            elif event.divergence_type == LearningStatus.OSCILLATING:
                correction_strategy = "reduce_learning_rate"
            elif event.divergence_type == LearningStatus.STAGNANT:
                correction_strategy = "increase_exploration"
            elif event.divergence_type == LearningStatus.UNSTABLE:
                correction_strategy = "stabilize_training"

            if correction_strategy:
                self.logger.info(
                    "Attempting learning correction for agent %s using strategy: %s",
                    event.agent_id,
                    correction_strategy,
                )

                # Apply correction (this would typically involve calling agent methods)
                success = await self._apply_correction_strategy(
                    event.agent_id, correction_strategy
                )

                if success:
                    # Mark as resolved
                    self.divergence_detector.resolve_divergence(
                        event.event_id, correction_strategy, {"auto_corrected": True}
                    )

                    # Notify correction callbacks
                    for callback in self.correction_callbacks:
                        try:
                            if asyncio.iscoroutinefunction(callback):
                                await callback(event, correction_strategy)
                            else:
                                callback(event, correction_strategy)
                        except Exception as e:
                            self.logger.error(
                                "Error in correction callback: %s", str(e)
                            )

        except Exception as e:
            self.logger.error(
                "Error attempting learning correction for agent %s: %s",
                event.agent_id,
                str(e),
            )

    async def _apply_correction_strategy(self, agent_id: str, strategy: str) -> bool:
        """Apply correction strategy to agent."""
        try:
            # This would typically involve calling methods on the actual agent
            # For now, we'll simulate the correction

            if strategy == "reset_learning_parameters":
                # Reset learning parameters
                if agent_id in self.agent_metrics:
                    metrics = self.agent_metrics[agent_id]
                    metrics.learning_rate = 0.001  # Reset to default
                    metrics.epsilon = 0.1  # Reset exploration

            elif strategy == "reduce_learning_rate":
                # Reduce learning rate
                if agent_id in self.agent_metrics:
                    metrics = self.agent_metrics[agent_id]
                    metrics.learning_rate *= 0.5  # Halve learning rate

            elif strategy == "increase_exploration":
                # Increase exploration
                if agent_id in self.agent_metrics:
                    metrics = self.agent_metrics[agent_id]
                    metrics.epsilon = min(0.5, metrics.epsilon * 1.5)

            elif strategy == "stabilize_training":
                # Stabilize training
                if agent_id in self.agent_metrics:
                    metrics = self.agent_metrics[agent_id]
                    metrics.learning_rate *= 0.8  # Slightly reduce learning rate

            # Simulate correction delay
            await asyncio.sleep(0.5)

            self.logger.info(
                "Applied correction strategy %s to agent %s", strategy, agent_id
            )

            return True

        except Exception as e:
            self.logger.error(
                "Failed to apply correction strategy %s to agent %s: %s",
                strategy,
                agent_id,
                str(e),
            )
            return False

    def update_agent_learning(
        self,
        agent_id: str,
        episode: int,
        reward: float,
        loss: float,
        q_values: Optional[List[float]] = None,
        learning_rate: Optional[float] = None,
        epsilon: Optional[float] = None,
    ) -> None:
        """Update agent learning metrics."""
        if agent_id not in self.agent_metrics:
            self.register_agent(agent_id)

        metrics = self.agent_metrics[agent_id]
        metrics.update_metrics(episode, reward, loss, q_values)

        if learning_rate is not None:
            metrics.learning_rate = learning_rate

        if epsilon is not None:
            metrics.epsilon = epsilon

    def get_agent_learning_status(self, agent_id: str) -> Optional[LearningStatus]:
        """Get current learning status for an agent."""
        if agent_id not in self.agent_metrics:
            return None

        metrics = self.agent_metrics[agent_id]

        # Check for active divergence events
        for event in self.divergence_detector.get_active_divergences():
            if event.agent_id == agent_id:
                return event.divergence_type

        # Determine status from metrics
        if metrics.convergence_score > 0.8:
            return LearningStatus.NORMAL
        elif metrics.is_diverging():
            return LearningStatus.DIVERGING
        elif metrics.is_oscillating():
            return LearningStatus.OSCILLATING
        elif metrics.is_stagnant():
            return LearningStatus.STAGNANT
        else:
            return LearningStatus.SLOW_CONVERGENCE

    def get_agent_metrics(self, agent_id: str) -> Optional[LearningMetrics]:
        """Get learning metrics for an agent."""
        return self.agent_metrics.get(agent_id)

    def get_all_agent_metrics(self) -> Dict[str, LearningMetrics]:
        """Get all agent learning metrics."""
        return self.agent_metrics.copy()

    def get_learning_summary(self) -> Dict[str, Any]:
        """Get learning system summary."""
        if not self.agent_metrics:
            return {
                "total_agents": 0,
                "learning_normally": 0,
                "learning_issues": 0,
                "average_convergence_score": 0.0,
                "system_learning_health": "unknown",
            }

        status_counts = {}
        total_convergence = 0.0

        for agent_id in self.monitored_agents:
            status = self.get_agent_learning_status(agent_id)
            if status:
                status_counts[status.value] = status_counts.get(status.value, 0) + 1

            metrics = self.agent_metrics.get(agent_id)
            if metrics:
                total_convergence += metrics.convergence_score

        avg_convergence = (
            total_convergence / len(self.agent_metrics) if self.agent_metrics else 0.0
        )

        # Determine system health
        normal_count = status_counts.get(LearningStatus.NORMAL.value, 0)
        total_agents = len(self.agent_metrics)

        if normal_count / total_agents > 0.8:
            system_health = "healthy"
        elif normal_count / total_agents > 0.5:
            system_health = "degraded"
        else:
            system_health = "unhealthy"

        return {
            "total_agents": total_agents,
            "learning_normally": normal_count,
            "learning_issues": total_agents - normal_count,
            "average_convergence_score": avg_convergence,
            "system_learning_health": system_health,
            "status_breakdown": status_counts,
            "divergence_statistics": self.divergence_detector.get_divergence_statistics(),
        }

    def add_issue_callback(self, callback: Callable) -> None:
        """Add learning issue callback."""
        self.issue_callbacks.append(callback)

    def add_correction_callback(self, callback: Callable) -> None:
        """Add learning correction callback."""
        self.correction_callbacks.append(callback)

    def remove_issue_callback(self, callback: Callable) -> None:
        """Remove learning issue callback."""
        if callback in self.issue_callbacks:
            self.issue_callbacks.remove(callback)

    def remove_correction_callback(self, callback: Callable) -> None:
        """Remove learning correction callback."""
        if callback in self.correction_callbacks:
            self.correction_callbacks.remove(callback)

    async def shutdown(self) -> None:
        """Shutdown learning monitor."""
        await self.stop_monitoring()

        # Clear all data
        self.agent_metrics.clear()
        self.monitored_agents.clear()
        self.issue_callbacks.clear()
        self.correction_callbacks.clear()

        self.logger.info("Learning monitor shutdown complete")
