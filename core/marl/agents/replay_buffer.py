"""
Experience Replay Buffer Implementation

This module provides experience replay buffer classes for storing and sampling
experiences in reinforcement learning, including both standard and prioritized
replay buffers for improved learning stability and sample efficiency.
"""

import logging
import random
from collections import deque
from typing import List, Optional, Tuple

import numpy as np

from ..exceptions import ExperienceBufferError
from .experience import Experience

logger = logging.getLogger(__name__)


class ReplayBuffer:
    """
    Standard experience replay buffer for RL agents.

    Stores experiences in a circular buffer and provides uniform random sampling
    for training. This implementation follows the development standards with
    proper error handling and logging.
    """

    def __init__(self, capacity: int):
        """
        Initialize replay buffer.

        Args:
            capacity: Maximum number of experiences to store
        """
        if capacity <= 0:
            raise ExperienceBufferError(
                "Buffer capacity must be positive",
                buffer_type="standard",
                buffer_size=capacity,
                operation="initialize",
            )

        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.position = 0

        logger.info("Initialized replay buffer with capacity %d", capacity)

    def add(self, experience: Experience) -> None:
        """
        Add experience to the buffer.

        Args:
            experience: Experience tuple to add
        """
        try:
            self.buffer.append(experience)
            self.position = (self.position + 1) % self.capacity

            if len(self.buffer) % 10000 == 0:
                logger.debug(
                    "Replay buffer size: %d/%d", len(self.buffer), self.capacity
                )

        except Exception as e:
            error_msg = "Failed to add experience to replay buffer"
            logger.error("%s: %s", error_msg, str(e))
            raise ExperienceBufferError(
                error_msg,
                buffer_type="standard",
                buffer_size=len(self.buffer),
                operation="add",
            ) from e

    def sample(self, batch_size: int) -> List[Experience]:
        """
        Sample a batch of experiences uniformly at random.

        Args:
            batch_size: Number of experiences to sample

        Returns:
            List of sampled experiences

        Raises:
            ExperienceBufferError: If not enough experiences or sampling fails
        """
        if len(self.buffer) < batch_size:
            raise ExperienceBufferError(
                f"Not enough experiences in buffer: {len(self.buffer)} < {batch_size}",
                buffer_type="standard",
                buffer_size=len(self.buffer),
                operation="sample",
            )

        try:
            return random.sample(list(self.buffer), batch_size)

        except Exception as e:
            error_msg = f"Failed to sample {batch_size} experiences from buffer"
            logger.error("%s: %s", error_msg, str(e))
            raise ExperienceBufferError(
                error_msg,
                buffer_type="standard",
                buffer_size=len(self.buffer),
                operation="sample",
            ) from e

    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)

    def is_full(self) -> bool:
        """Check if buffer is at capacity."""
        return len(self.buffer) == self.capacity

    def clear(self) -> None:
        """Clear all experiences from buffer."""
        self.buffer.clear()
        self.position = 0
        logger.info("Replay buffer cleared")

    def get_statistics(self) -> dict:
        """Get buffer statistics for monitoring."""
        if not self.buffer:
            return {
                "size": 0,
                "capacity": self.capacity,
                "utilization": 0.0,
                "avg_reward": 0.0,
                "reward_std": 0.0,
            }

        rewards = [exp.reward for exp in self.buffer]

        return {
            "size": len(self.buffer),
            "capacity": self.capacity,
            "utilization": len(self.buffer) / self.capacity,
            "avg_reward": np.mean(rewards),
            "reward_std": np.std(rewards),
            "min_reward": np.min(rewards),
            "max_reward": np.max(rewards),
        }


class PrioritizedReplayBuffer:
    """
    Prioritized experience replay buffer.

    Samples experiences based on their temporal difference (TD) error,
    giving higher priority to experiences that are more surprising or
    informative for learning.
    """

    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4):
        """
        Initialize prioritized replay buffer.

        Args:
            capacity: Maximum number of experiences to store
            alpha: Prioritization exponent (0 = uniform, 1 = full prioritization)
            beta: Importance sampling exponent (0 = no correction, 1 = full correction)
        """
        if capacity <= 0:
            raise ExperienceBufferError(
                "Buffer capacity must be positive",
                buffer_type="prioritized",
                buffer_size=capacity,
                operation="initialize",
            )

        if not 0 <= alpha <= 1:
            raise ExperienceBufferError(
                f"Alpha must be between 0 and 1, got {alpha}",
                buffer_type="prioritized",
                operation="initialize",
            )

        if not 0 <= beta <= 1:
            raise ExperienceBufferError(
                f"Beta must be between 0 and 1, got {beta}",
                buffer_type="prioritized",
                operation="initialize",
            )

        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0

        logger.info(
            "Initialized prioritized replay buffer with capacity %d, alpha %.2f, beta %.2f",
            capacity,
            alpha,
            beta,
        )

    def add(self, experience: Experience, td_error: Optional[float] = None) -> None:
        """
        Add experience to the buffer with priority.

        Args:
            experience: Experience tuple to add
            td_error: Temporal difference error for prioritization (optional)
        """
        try:
            # Calculate priority from TD error or use max priority for new experiences
            if td_error is not None:
                priority = (abs(td_error) + 1e-6) ** self.alpha
            else:
                priority = self.max_priority

            # Add experience
            if len(self.buffer) < self.capacity:
                self.buffer.append(experience)
            else:
                self.buffer[self.position] = experience

            # Set priority
            self.priorities[self.position] = priority
            self.max_priority = max(self.max_priority, priority)

            self.position = (self.position + 1) % self.capacity

            if len(self.buffer) % 10000 == 0:
                logger.debug(
                    "Prioritized replay buffer size: %d/%d, max priority: %.4f",
                    len(self.buffer),
                    self.capacity,
                    self.max_priority,
                )

        except Exception as e:
            error_msg = "Failed to add experience to prioritized replay buffer"
            logger.error("%s: %s", error_msg, str(e))
            raise ExperienceBufferError(
                error_msg,
                buffer_type="prioritized",
                buffer_size=len(self.buffer),
                operation="add",
            ) from e

    def sample(
        self, batch_size: int
    ) -> Tuple[List[Experience], np.ndarray, np.ndarray]:
        """
        Sample a batch of experiences based on priorities.

        Args:
            batch_size: Number of experiences to sample

        Returns:
            Tuple of (experiences, indices, importance_weights)

        Raises:
            ExperienceBufferError: If not enough experiences or sampling fails
        """
        if len(self.buffer) < batch_size:
            raise ExperienceBufferError(
                f"Not enough experiences in buffer: {len(self.buffer)} < {batch_size}",
                buffer_type="prioritized",
                buffer_size=len(self.buffer),
                operation="sample",
            )

        try:
            # Get valid priorities
            valid_priorities = self.priorities[: len(self.buffer)]

            # Calculate sampling probabilities
            probabilities = valid_priorities / valid_priorities.sum()

            # Sample indices based on probabilities
            indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)

            # Get experiences
            experiences = [self.buffer[idx] for idx in indices]

            # Calculate importance sampling weights
            weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
            weights = weights / weights.max()  # Normalize weights

            return experiences, indices, weights

        except Exception as e:
            error_msg = (
                f"Failed to sample {batch_size} experiences from prioritized buffer"
            )
            logger.error("%s: %s", error_msg, str(e))
            raise ExperienceBufferError(
                error_msg,
                buffer_type="prioritized",
                buffer_size=len(self.buffer),
                operation="sample",
            ) from e

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """
        Update priorities for sampled experiences.

        Args:
            indices: Indices of experiences to update
            td_errors: New TD errors for priority calculation
        """
        try:
            priorities = (np.abs(td_errors) + 1e-6) ** self.alpha
            self.priorities[indices] = priorities
            self.max_priority = max(self.max_priority, priorities.max())

        except Exception as e:
            error_msg = "Failed to update priorities in prioritized buffer"
            logger.error("%s: %s", error_msg, str(e))
            raise ExperienceBufferError(
                error_msg,
                buffer_type="prioritized",
                buffer_size=len(self.buffer),
                operation="update_priorities",
            ) from e

    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)

    def is_full(self) -> bool:
        """Check if buffer is at capacity."""
        return len(self.buffer) == self.capacity

    def clear(self) -> None:
        """Clear all experiences from buffer."""
        self.buffer.clear()
        self.priorities.fill(0)
        self.position = 0
        self.max_priority = 1.0
        logger.info("Prioritized replay buffer cleared")

    def get_statistics(self) -> dict:
        """Get buffer statistics for monitoring."""
        if not self.buffer:
            return {
                "size": 0,
                "capacity": self.capacity,
                "utilization": 0.0,
                "avg_reward": 0.0,
                "reward_std": 0.0,
                "avg_priority": 0.0,
                "max_priority": 0.0,
            }

        rewards = [exp.reward for exp in self.buffer]
        valid_priorities = self.priorities[: len(self.buffer)]

        return {
            "size": len(self.buffer),
            "capacity": self.capacity,
            "utilization": len(self.buffer) / self.capacity,
            "avg_reward": np.mean(rewards),
            "reward_std": np.std(rewards),
            "min_reward": np.min(rewards),
            "max_reward": np.max(rewards),
            "avg_priority": np.mean(valid_priorities),
            "max_priority": self.max_priority,
            "priority_std": np.std(valid_priorities),
        }
