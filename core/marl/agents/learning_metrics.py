"""
Learning Metrics Tracking.

This module provides comprehensive metrics tracking for RL agent learning
progress, including loss tracking, reward analysis, and performance monitoring
following the development standards.
"""

# Standard Library
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

# Third-Party Library
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LearningMetrics:
    """
    Comprehensive learning metrics for RL agents.

    Tracks various aspects of agent learning including loss, rewards,
    exploration parameters, and performance indicators.
    """

    # Training progress
    training_steps: int = 0
    episodes_completed: int = 0
    total_training_time: float = 0.0

    # Loss tracking
    losses: Deque[float] = field(default_factory=lambda: deque(maxlen=1000))
    loss_history: List[float] = field(default_factory=list)

    # Reward tracking
    episode_rewards: Deque[float] = field(default_factory=lambda: deque(maxlen=1000))
    reward_history: List[float] = field(default_factory=list)
    cumulative_reward: float = 0.0

    # Q-value tracking
    q_values: Deque[float] = field(default_factory=lambda: deque(maxlen=1000))
    q_value_history: List[float] = field(default_factory=list)

    # Exploration tracking
    epsilon_values: Deque[float] = field(default_factory=lambda: deque(maxlen=1000))
    epsilon_history: List[float] = field(default_factory=list)

    # Performance indicators
    best_episode_reward: float = float("-inf")
    worst_episode_reward: float = float("inf")
    recent_performance_window: int = 100

    # Timing
    last_update_time: float = field(default_factory=time.time)

    def record_training_step(
        self, loss: float, reward: float, epsilon: float, q_values_mean: float
    ) -> None:
        """
        Record metrics for a training step.

        Args:
            loss: Training loss value
            reward: Reward received
            epsilon: Current exploration rate
            q_values_mean: Mean Q-values for the state
        """
        current_time = time.time()

        # Update counters
        self.training_steps += 1
        self.total_training_time += current_time - self.last_update_time
        self.last_update_time = current_time

        # Record metrics
        self.losses.append(loss)
        self.episode_rewards.append(reward)
        self.q_values.append(q_values_mean)
        self.epsilon_values.append(epsilon)

        # Update cumulative reward
        self.cumulative_reward += reward

        # Update history (less frequent for memory efficiency)
        if self.training_steps % 100 == 0:
            self.loss_history.append(loss)
            self.reward_history.append(reward)
            self.q_value_history.append(q_values_mean)
            self.epsilon_history.append(epsilon)

        # Log progress periodically
        if self.training_steps % 1000 == 0:
            logger.debug(
                "Training step %d: loss=%.4f, reward=%.3f, epsilon=%.3f, q_mean=%.3f",
                self.training_steps,
                loss,
                reward,
                epsilon,
                q_values_mean,
            )

    def record_episode_completion(self, episode_reward: float) -> None:
        """
        Record completion of an episode.

        Args:
            episode_reward: Total reward for the completed episode
        """
        self.episodes_completed += 1

        # Update best/worst performance
        if episode_reward > self.best_episode_reward:
            self.best_episode_reward = episode_reward

        if episode_reward < self.worst_episode_reward:
            self.worst_episode_reward = episode_reward

        # Log episode completion
        if self.episodes_completed % 100 == 0:
            avg_reward = self.get_average_episode_reward()
            logger.info(
                "Episode %d completed: reward=%.3f, avg_reward=%.3f, best=%.3f",
                self.episodes_completed,
                episode_reward,
                avg_reward,
                self.best_episode_reward,
            )

    def get_average_loss(self, window: Optional[int] = None) -> float:
        """
        Get average loss over a window.

        Args:
            window: Number of recent steps to average (None for all)

        Returns:
            Average loss value
        """
        if not self.losses:
            return 0.0

        if window is None:
            return float(np.mean(self.losses))

        recent_losses = list(self.losses)[-window:]
        return float(np.mean(recent_losses)) if recent_losses else 0.0

    def get_average_episode_reward(self, window: Optional[int] = None) -> float:
        """
        Get average episode reward over a window.

        Args:
            window: Number of recent episodes to average (None for all)

        Returns:
            Average episode reward
        """
        if not self.episode_rewards:
            return 0.0

        if window is None:
            return float(np.mean(self.episode_rewards))

        recent_rewards = list(self.episode_rewards)[-window:]
        return float(np.mean(recent_rewards)) if recent_rewards else 0.0

    def get_reward_trend(self, window: int = 100) -> float:
        """
        Calculate reward trend over recent episodes.

        Args:
            window: Number of recent episodes to analyze

        Returns:
            Trend coefficient (positive = improving, negative = declining)
        """
        if len(self.episode_rewards) < 10:
            return 0.0

        recent_rewards = list(self.episode_rewards)[-window:]
        if len(recent_rewards) < 10:
            return 0.0

        # Calculate linear trend
        x = np.arange(len(recent_rewards))
        y = np.array(recent_rewards)

        # Simple linear regression
        x_mean = np.mean(x)
        y_mean = np.mean(y)

        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)

        if denominator == 0:
            return 0.0

        slope = numerator / denominator
        return float(slope)

    def get_learning_stability(self) -> float:
        """
        Calculate learning stability based on loss variance.

        Returns:
            Stability score (higher = more stable)
        """
        if len(self.losses) < 10:
            return 0.0

        recent_losses = list(self.losses)[-100:]
        loss_std = np.std(recent_losses)
        loss_mean = np.mean(recent_losses)

        # Coefficient of variation (inverted for stability score)
        if loss_mean == 0:
            return 1.0

        cv = loss_std / loss_mean
        stability = 1.0 / (1.0 + cv)  # Higher stability for lower CV

        return float(stability)

    def get_exploration_progress(self) -> float:
        """
        Calculate exploration progress based on epsilon decay.

        Returns:
            Progress score between 0 and 1
        """
        if not self.epsilon_values:
            return 0.0

        current_epsilon = self.epsilon_values[-1]
        initial_epsilon = self.epsilon_values[0] if len(self.epsilon_values) > 1 else 1.0

        if initial_epsilon == 0:
            return 1.0

        progress = 1.0 - (current_epsilon / initial_epsilon)
        return max(0.0, min(1.0, progress))

    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive performance summary.

        Returns:
            Dictionary with performance metrics
        """
        return {
            "training_steps": self.training_steps,
            "episodes_completed": self.episodes_completed,
            "total_training_time_hours": self.total_training_time / 3600,
            "average_loss": self.get_average_loss(),
            "recent_average_loss": self.get_average_loss(100),
            "average_episode_reward": self.get_average_episode_reward(),
            "recent_average_reward": self.get_average_episode_reward(100),
            "best_episode_reward": (
                self.best_episode_reward if self.best_episode_reward != float("-inf") else 0.0
            ),
            "worst_episode_reward": (
                self.worst_episode_reward if self.worst_episode_reward != float("inf") else 0.0
            ),
            "cumulative_reward": self.cumulative_reward,
            "reward_trend": self.get_reward_trend(),
            "learning_stability": self.get_learning_stability(),
            "exploration_progress": self.get_exploration_progress(),
            "current_epsilon": self.epsilon_values[-1] if self.epsilon_values else 0.0,
            "current_q_mean": self.q_values[-1] if self.q_values else 0.0,
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get concise summary for monitoring."""
        return {
            "steps": self.training_steps,
            "episodes": self.episodes_completed,
            "avg_reward": self.get_average_episode_reward(100),
            "best_reward": (
                self.best_episode_reward if self.best_episode_reward != float("-inf") else 0.0
            ),
            "trend": self.get_reward_trend(),
            "stability": self.get_learning_stability(),
            "epsilon": self.epsilon_values[-1] if self.epsilon_values else 0.0,
        }

    def reset_metrics(self) -> None:
        """Reset all metrics (useful for retraining)."""
        self.training_steps = 0
        self.episodes_completed = 0
        self.total_training_time = 0.0
        self.cumulative_reward = 0.0
        self.best_episode_reward = float("-inf")
        self.worst_episode_reward = float("inf")

        # Clear collections
        self.losses.clear()
        self.episode_rewards.clear()
        self.q_values.clear()
        self.epsilon_values.clear()

        # Clear history
        self.loss_history.clear()
        self.reward_history.clear()
        self.q_value_history.clear()
        self.epsilon_history.clear()

        self.last_update_time = time.time()

        logger.info("Learning metrics reset")

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for serialization."""
        return {
            "training_steps": self.training_steps,
            "episodes_completed": self.episodes_completed,
            "total_training_time": self.total_training_time,
            "cumulative_reward": self.cumulative_reward,
            "best_episode_reward": self.best_episode_reward,
            "worst_episode_reward": self.worst_episode_reward,
            "loss_history": self.loss_history,
            "reward_history": self.reward_history,
            "q_value_history": self.q_value_history,
            "epsilon_history": self.epsilon_history,
            "performance_summary": self.get_performance_summary(),
        }

    def from_dict(self, data: Dict[str, Any]) -> None:
        """Load metrics from dictionary."""
        self.training_steps = data.get("training_steps", 0)
        self.episodes_completed = data.get("episodes_completed", 0)
        self.total_training_time = data.get("total_training_time", 0.0)
        self.cumulative_reward = data.get("cumulative_reward", 0.0)
        self.best_episode_reward = data.get("best_episode_reward", float("-inf"))
        self.worst_episode_reward = data.get("worst_episode_reward", float("inf"))

        # Load history
        self.loss_history = data.get("loss_history", [])
        self.reward_history = data.get("reward_history", [])
        self.q_value_history = data.get("q_value_history", [])
        self.epsilon_history = data.get("epsilon_history", [])

        # Populate recent deques from history
        if self.loss_history:
            self.losses.extend(self.loss_history[-1000:])
        if self.reward_history:
            self.episode_rewards.extend(self.reward_history[-1000:])
        if self.q_value_history:
            self.q_values.extend(self.q_value_history[-1000:])
        if self.epsilon_history:
            self.epsilon_values.extend(self.epsilon_history[-1000:])

        self.last_update_time = time.time()

        logger.info("Learning metrics loaded from dictionary")

    def export_for_analysis(self) -> Dict[str, Any]:
        """Export detailed data for external analysis."""
        return {
            "metadata": {
                "training_steps": self.training_steps,
                "episodes_completed": self.episodes_completed,
                "total_training_time": self.total_training_time,
                "export_timestamp": time.time(),
            },
            "performance_summary": self.get_performance_summary(),
            "detailed_history": {
                "losses": self.loss_history,
                "rewards": self.reward_history,
                "q_values": self.q_value_history,
                "epsilon_values": self.epsilon_history,
            },
            "recent_data": {
                "recent_losses": list(self.losses),
                "recent_rewards": list(self.episode_rewards),
                "recent_q_values": list(self.q_values),
                "recent_epsilon": list(self.epsilon_values),
            },
        }
