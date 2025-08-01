"""
Continuous Learning System

This module implements continuous learning workflows for real-time adaptation
of multi-agent reinforcement learning systems.
"""

# Standard Library
import asyncio
import time
from typing import Any, Dict, List, Optional

# Third-Party Library
import numpy as np

# SynThesisAI Modules
from core.marl.agents.base_agent import BaseRLAgent
from core.marl.learning.shared_experience import SharedExperienceManager
from utils.logging_config import get_logger


class LearningConfig:
    """Configuration for continuous learning system."""

    def __init__(
        self,
        learning_interval: float = 60.0,  # seconds
        batch_size: int = 32,
        min_experiences_for_learning: int = 100,
        learning_rate_decay: float = 0.995,
        min_learning_rate: float = 1e-5,
        performance_window_size: int = 100,
        adaptation_threshold: float = 0.1,
        max_learning_iterations: int = 10,
        enable_shared_learning: bool = True,
        shared_experience_ratio: float = 0.3,
    ):
        """
        Initialize continuous learning configuration.

        Args:
            learning_interval: Time between learning updates in seconds
            batch_size: Batch size for learning updates
            min_experiences_for_learning: Minimum experiences before learning
            learning_rate_decay: Learning rate decay factor
            min_learning_rate: Minimum learning rate
            performance_window_size: Window size for performance tracking
            adaptation_threshold: Threshold for triggering adaptation
            max_learning_iterations: Maximum learning iterations per update
            enable_shared_learning: Whether to use shared experiences
            shared_experience_ratio: Ratio of shared to own experiences
        """
        self.learning_interval = learning_interval
        self.batch_size = batch_size
        self.min_experiences_for_learning = min_experiences_for_learning
        self.learning_rate_decay = learning_rate_decay
        self.min_learning_rate = min_learning_rate
        self.performance_window_size = performance_window_size
        self.adaptation_threshold = adaptation_threshold
        self.max_learning_iterations = max_learning_iterations
        self.enable_shared_learning = enable_shared_learning
        self.shared_experience_ratio = shared_experience_ratio


class PerformanceTracker:
    """Tracks agent performance for adaptive learning."""

    def __init__(self, window_size: int = 100):
        """
        Initialize performance tracker.

        Args:
            window_size: Size of the performance tracking window
        """
        self.window_size = window_size
        self.performance_history = []
        self.reward_history = []
        self.loss_history = []
        self.logger = get_logger(__name__ + ".PerformanceTracker")

    def record_performance(
        self,
        reward: float,
        loss: Optional[float] = None,
        additional_metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Record performance metrics.

        Args:
            reward: Reward value
            loss: Training loss (optional)
            additional_metrics: Additional performance metrics
        """
        timestamp = time.time()

        performance_entry = {
            "timestamp": timestamp,
            "reward": reward,
            "loss": loss,
            "metrics": additional_metrics or {},
        }

        self.performance_history.append(performance_entry)
        self.reward_history.append(reward)

        if loss is not None:
            self.loss_history.append(loss)

        # Maintain window size
        if len(self.performance_history) > self.window_size:
            self.performance_history.pop(0)
            self.reward_history.pop(0)

        if len(self.loss_history) > self.window_size:
            self.loss_history.pop(0)

    def get_recent_performance(self, window: Optional[int] = None) -> Dict[str, float]:
        """
        Get recent performance statistics.

        Args:
            window: Window size for statistics (uses default if None)

        Returns:
            Dictionary of performance statistics
        """
        if not self.performance_history:
            return {
                "avg_reward": 0.0,
                "reward_trend": 0.0,
                "avg_loss": 0.0,
                "loss_trend": 0.0,
                "sample_count": 0,
            }

        window = window or len(self.performance_history)
        recent_rewards = self.reward_history[-window:]
        recent_losses = self.loss_history[-window:] if self.loss_history else []

        # Calculate statistics
        avg_reward = np.mean(recent_rewards) if recent_rewards else 0.0
        reward_trend = self._calculate_trend(recent_rewards)
        avg_loss = np.mean(recent_losses) if recent_losses else 0.0
        loss_trend = self._calculate_trend(recent_losses)

        return {
            "avg_reward": float(avg_reward),
            "reward_trend": float(reward_trend),
            "avg_loss": float(avg_loss),
            "loss_trend": float(loss_trend),
            "sample_count": len(recent_rewards),
        }

    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend (slope) of values."""
        if len(values) < 2:
            return 0.0

        x = np.arange(len(values))
        y = np.array(values)

        # Calculate linear regression slope
        n = len(values)
        slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (
            n * np.sum(x**2) - np.sum(x) ** 2
        )

        return float(slope)

    def should_adapt(self, threshold: float) -> bool:
        """
        Determine if adaptation is needed based on performance.

        Args:
            threshold: Adaptation threshold

        Returns:
            True if adaptation is recommended
        """
        if len(self.reward_history) < 10:
            return False

        recent_performance = self.get_recent_performance(window=20)

        # Adapt if performance is declining
        if recent_performance["reward_trend"] < -threshold:
            return True

        # Adapt if loss is increasing significantly
        if recent_performance["loss_trend"] > threshold:
            return True

        return False

    def reset(self) -> None:
        """Reset performance tracking."""
        self.performance_history.clear()
        self.reward_history.clear()
        self.loss_history.clear()
        self.logger.debug("Performance tracker reset")


class AdaptiveLearningRate:
    """Manages adaptive learning rate adjustment."""

    def __init__(
        self,
        initial_rate: float = 0.001,
        decay_factor: float = 0.995,
        min_rate: float = 1e-5,
        adaptation_factor: float = 1.1,
    ):
        """
        Initialize adaptive learning rate manager.

        Args:
            initial_rate: Initial learning rate
            decay_factor: Decay factor for learning rate
            min_rate: Minimum learning rate
            adaptation_factor: Factor for adaptive adjustments
        """
        self.initial_rate = initial_rate
        self.current_rate = initial_rate
        self.decay_factor = decay_factor
        self.min_rate = min_rate
        self.adaptation_factor = adaptation_factor
        self.logger = get_logger(__name__ + ".AdaptiveLearningRate")

    def update_rate(self, performance_improving: bool = True) -> None:
        """
        Update learning rate based on performance.

        Args:
            performance_improving: Whether performance is improving
        """
        if performance_improving:
            # Gradual decay when performance is good
            self.current_rate *= self.decay_factor
        else:
            # Increase rate when performance is poor
            self.current_rate *= self.adaptation_factor

        # Ensure rate stays within bounds
        self.current_rate = max(self.min_rate, self.current_rate)

        self.logger.debug(
            "Learning rate updated to %.6f (improving=%s)",
            self.current_rate,
            performance_improving,
        )

    def get_rate(self) -> float:
        """Get current learning rate."""
        return self.current_rate

    def reset(self) -> None:
        """Reset to initial learning rate."""
        self.current_rate = self.initial_rate
        self.logger.debug("Learning rate reset to %.6f", self.current_rate)


class ContinuousLearningManager:
    """
    Main manager for continuous learning workflows.

    This class coordinates continuous learning across multiple agents,
    manages shared experiences, and adapts learning parameters in real-time.
    """

    def __init__(
        self,
        config: LearningConfig,
        shared_experience_manager: Optional[SharedExperienceManager] = None,
    ):
        """
        Initialize continuous learning manager.

        Args:
            config: Learning configuration
            shared_experience_manager: Optional shared experience manager
        """
        self.config = config
        self.shared_experience_manager = shared_experience_manager
        self.logger = get_logger(__name__ + ".ContinuousLearningManager")

        # Registered agents
        self.agents: Dict[str, BaseRLAgent] = {}

        # Performance tracking
        self.performance_trackers: Dict[str, PerformanceTracker] = {}
        self.learning_rate_managers: Dict[str, AdaptiveLearningRate] = {}

        # Learning state
        self.learning_active = False
        self.learning_task = None
        self.last_learning_time = 0.0

        # Statistics
        self.learning_stats = {
            "total_learning_updates": 0,
            "successful_updates": 0,
            "failed_updates": 0,
            "adaptations_triggered": 0,
            "shared_experiences_used": 0,
        }

        self.logger.info("Initialized ContinuousLearningManager")

    def register_agent(self, agent_id: str, agent: BaseRLAgent) -> None:
        """
        Register an agent for continuous learning.

        Args:
            agent_id: Unique identifier for the agent
            agent: The RL agent instance
        """
        self.agents[agent_id] = agent
        self.performance_trackers[agent_id] = PerformanceTracker(
            self.config.performance_window_size
        )
        self.learning_rate_managers[agent_id] = AdaptiveLearningRate(
            initial_rate=agent.config.learning_rate,
            decay_factor=self.config.learning_rate_decay,
            min_rate=self.config.min_learning_rate,
        )

        # Register with shared experience manager if available
        if self.shared_experience_manager:
            self.shared_experience_manager.register_agent(agent_id)

        self.logger.info("Registered agent for continuous learning: %s", agent_id)

    def start_continuous_learning(self) -> None:
        """Start the continuous learning process."""
        if self.learning_active:
            self.logger.warning("Continuous learning is already active")
            return

        self.learning_active = True
        self.learning_task = asyncio.create_task(self._learning_loop())
        self.logger.info("Started continuous learning process")

    async def stop_continuous_learning(self) -> None:
        """Stop the continuous learning process."""
        if not self.learning_active:
            return

        self.learning_active = False

        if self.learning_task:
            self.learning_task.cancel()
            try:
                await self.learning_task
            except asyncio.CancelledError:
                pass

        self.logger.info("Stopped continuous learning process")

    async def _learning_loop(self) -> None:
        """Main continuous learning loop."""
        self.logger.info("Starting continuous learning loop")

        try:
            while self.learning_active:
                current_time = time.time()

                # Check if it's time for a learning update
                if (
                    current_time - self.last_learning_time
                ) >= self.config.learning_interval:
                    await self._perform_learning_update()
                    self.last_learning_time = current_time

                # Sleep for a short interval
                await asyncio.sleep(1.0)

        except asyncio.CancelledError:
            self.logger.info("Continuous learning loop cancelled")
        except Exception as e:
            self.logger.error("Error in continuous learning loop: %s", str(e))

    async def _perform_learning_update(self) -> None:
        """Perform learning updates for all registered agents."""
        self.logger.debug("Performing continuous learning update")
        self.learning_stats["total_learning_updates"] += 1

        successful_updates = 0

        for agent_id, agent in self.agents.items():
            try:
                # Check if agent has enough experiences
                if not self._has_sufficient_experiences(agent):
                    continue

                # Perform learning update
                success = await self._update_agent_policy(agent_id, agent)
                if success:
                    successful_updates += 1

            except Exception as e:
                self.logger.error("Error updating agent %s: %s", agent_id, str(e))
                self.learning_stats["failed_updates"] += 1

        if successful_updates > 0:
            self.learning_stats["successful_updates"] += successful_updates
            self.logger.debug(
                "Learning update completed: %d/%d agents updated successfully",
                successful_updates,
                len(self.agents),
            )

    def _has_sufficient_experiences(self, agent: BaseRLAgent) -> bool:
        """Check if agent has sufficient experiences for learning."""
        return len(agent.replay_buffer) >= self.config.min_experiences_for_learning

    async def _update_agent_policy(self, agent_id: str, agent: BaseRLAgent) -> bool:
        """
        Update an agent's policy through continuous learning.

        Args:
            agent_id: Agent identifier
            agent: Agent instance

        Returns:
            True if update was successful
        """
        try:
            # Get performance tracker and learning rate manager
            performance_tracker = self.performance_trackers[agent_id]
            lr_manager = self.learning_rate_managers[agent_id]

            # Check if adaptation is needed
            if performance_tracker.should_adapt(self.config.adaptation_threshold):
                self.learning_stats["adaptations_triggered"] += 1
                self.logger.info("Triggering adaptation for agent %s", agent_id)

            # Prepare training batch
            experiences = self._prepare_training_batch(agent_id, agent)

            if not experiences:
                return False

            # Update agent's learning rate
            recent_performance = performance_tracker.get_recent_performance()
            performance_improving = recent_performance["reward_trend"] >= 0
            lr_manager.update_rate(performance_improving)

            # Update agent's learning rate
            agent.config.learning_rate = lr_manager.get_rate()

            # Perform multiple learning iterations
            total_loss = 0.0
            iterations = min(
                self.config.max_learning_iterations,
                len(experiences) // self.config.batch_size,
            )

            for _ in range(iterations):
                # Sample batch from experiences
                batch_indices = np.random.choice(
                    len(experiences),
                    size=min(self.config.batch_size, len(experiences)),
                    replace=False,
                )
                batch = [experiences[i] for i in batch_indices]

                # Update policy
                loss = agent._update_policy_from_batch(batch)
                total_loss += loss

            avg_loss = total_loss / iterations if iterations > 0 else 0.0

            # Record performance
            avg_reward = np.mean([exp.reward for exp in experiences])
            performance_tracker.record_performance(avg_reward, avg_loss)

            self.logger.debug(
                "Updated agent %s: avg_reward=%.3f, avg_loss=%.6f, lr=%.6f",
                agent_id,
                avg_reward,
                avg_loss,
                lr_manager.get_rate(),
            )

            return True

        except Exception as e:
            self.logger.error("Failed to update agent %s policy: %s", agent_id, str(e))
            return False

    def _prepare_training_batch(self, agent_id: str, agent: BaseRLAgent) -> List[Any]:
        """
        Prepare training batch combining own and shared experiences.

        Args:
            agent_id: Agent identifier
            agent: Agent instance

        Returns:
            List of experiences for training
        """
        # Get agent's own experiences
        own_experiences = agent.replay_buffer.sample(
            min(self.config.batch_size, len(agent.replay_buffer))
        )

        experiences = list(own_experiences)

        # Add shared experiences if enabled
        if self.config.enable_shared_learning and self.shared_experience_manager:
            shared_count = int(len(experiences) * self.config.shared_experience_ratio)

            if shared_count > 0:
                shared_experiences = self.shared_experience_manager.sample_experiences(
                    agent_id, shared_count, exclude_own=True
                )

                experiences.extend(shared_experiences)
                self.learning_stats["shared_experiences_used"] += len(
                    shared_experiences
                )

        return experiences

    def record_agent_performance(
        self,
        agent_id: str,
        reward: float,
        additional_metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Record performance for an agent.

        Args:
            agent_id: Agent identifier
            reward: Reward value
            additional_metrics: Additional performance metrics
        """
        if agent_id in self.performance_trackers:
            self.performance_trackers[agent_id].record_performance(
                reward, additional_metrics=additional_metrics
            )

    def get_learning_progress(self) -> Dict[str, Any]:
        """Get comprehensive learning progress information."""
        agent_progress = {}

        for agent_id in self.agents:
            performance_tracker = self.performance_trackers[agent_id]
            lr_manager = self.learning_rate_managers[agent_id]

            recent_performance = performance_tracker.get_recent_performance()

            agent_progress[agent_id] = {
                "performance": recent_performance,
                "learning_rate": lr_manager.get_rate(),
                "should_adapt": performance_tracker.should_adapt(
                    self.config.adaptation_threshold
                ),
            }

        return {
            "learning_active": self.learning_active,
            "learning_stats": self.learning_stats.copy(),
            "agent_progress": agent_progress,
            "configuration": {
                "learning_interval": self.config.learning_interval,
                "batch_size": self.config.batch_size,
                "enable_shared_learning": self.config.enable_shared_learning,
            },
        }

    def reset_learning_progress(self) -> None:
        """Reset learning progress for all agents."""
        for tracker in self.performance_trackers.values():
            tracker.reset()

        for lr_manager in self.learning_rate_managers.values():
            lr_manager.reset()

        # Reset statistics
        self.learning_stats = {
            "total_learning_updates": 0,
            "successful_updates": 0,
            "failed_updates": 0,
            "adaptations_triggered": 0,
            "shared_experiences_used": 0,
        }

        self.logger.info("Reset learning progress for all agents")

    async def shutdown(self) -> None:
        """Gracefully shutdown the continuous learning manager."""
        self.logger.info("Shutting down continuous learning manager")
        await self.stop_continuous_learning()


def create_continuous_learning_manager(
    config: Optional[LearningConfig] = None,
    shared_experience_manager: Optional[SharedExperienceManager] = None,
) -> ContinuousLearningManager:
    """
    Factory function to create a continuous learning manager.

    Args:
        config: Optional learning configuration
        shared_experience_manager: Optional shared experience manager

    Returns:
        Configured ContinuousLearningManager instance
    """
    if config is None:
        config = LearningConfig()

    return ContinuousLearningManager(config, shared_experience_manager)
