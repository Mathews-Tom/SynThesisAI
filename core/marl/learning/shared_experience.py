"""
Shared Experience Management System

This module implements the shared experience management system that enables
cross-agent learning by sharing valuable experiences between agents.
"""

import logging
import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.marl.agents.base_agent import Experience
from utils.logging_config import get_logger


class ExperienceConfig:
    """Configuration for experience sharing system."""

    def __init__(
        self,
        agent_buffer_size: int = 1000,
        shared_buffer_size: int = 5000,
        high_reward_threshold: float = 0.8,
        novelty_threshold: float = 0.7,
        sharing_probability: float = 0.1,
        max_age_hours: float = 24.0,
        min_experiences_for_sharing: int = 10,
    ):
        """
        Initialize experience sharing configuration.

        Args:
            agent_buffer_size: Maximum experiences per agent buffer
            shared_buffer_size: Maximum experiences in shared buffer
            high_reward_threshold: Threshold for high-reward experience sharing
            novelty_threshold: Threshold for novelty-based sharing
            sharing_probability: Base probability for random sharing
            max_age_hours: Maximum age of experiences in hours
            min_experiences_for_sharing: Minimum experiences before sharing
        """
        self.agent_buffer_size = agent_buffer_size
        self.shared_buffer_size = shared_buffer_size
        self.high_reward_threshold = high_reward_threshold
        self.novelty_threshold = novelty_threshold
        self.sharing_probability = sharing_probability
        self.max_age_hours = max_age_hours
        self.min_experiences_for_sharing = min_experiences_for_sharing


class ExperienceMetadata:
    """Metadata for experience entries."""

    def __init__(
        self,
        agent_id: str,
        timestamp: float,
        reward: float,
        novelty_score: float = 0.0,
        sharing_reason: str = "unknown",
    ):
        """
        Initialize experience metadata.

        Args:
            agent_id: ID of the agent that generated the experience
            timestamp: Unix timestamp when experience was created
            reward: Reward value associated with the experience
            novelty_score: Computed novelty score (0.0 to 1.0)
            sharing_reason: Reason why this experience was shared
        """
        self.agent_id = agent_id
        self.timestamp = timestamp
        self.reward = reward
        self.novelty_score = novelty_score
        self.sharing_reason = sharing_reason
        self.access_count = 0
        self.last_accessed = timestamp

    def is_expired(self, max_age_hours: float) -> bool:
        """Check if experience has expired based on age."""
        age_hours = (time.time() - self.timestamp) / 3600
        return age_hours > max_age_hours

    def update_access(self):
        """Update access statistics."""
        self.access_count += 1
        self.last_accessed = time.time()


class ExperienceFilter:
    """Filters experiences based on sharing criteria."""

    def __init__(self, config: ExperienceConfig):
        """
        Initialize experience filter.

        Args:
            config: Experience sharing configuration
        """
        self.config = config
        self.logger = get_logger(__name__ + ".ExperienceFilter")

    def should_share_experience(
        self,
        experience: Experience,
        metadata: ExperienceMetadata,
        agent_history: List[Experience],
    ) -> Tuple[bool, str]:
        """
        Determine if an experience should be shared.

        Args:
            experience: The experience to evaluate
            metadata: Experience metadata
            agent_history: Recent experiences from the same agent

        Returns:
            Tuple of (should_share, reason)
        """
        # High reward experiences are always shared
        if metadata.reward >= self.config.high_reward_threshold:
            return True, "high_reward"

        # Novel experiences are shared based on novelty threshold
        if metadata.novelty_score >= self.config.novelty_threshold:
            return True, "novelty"

        # Random sharing for exploration
        if np.random.random() < self.config.sharing_probability:
            return True, "random_exploration"

        # Don't share low-value experiences
        return False, "low_value"

    def calculate_novelty_score(
        self, experience: Experience, agent_history: List[Experience]
    ) -> float:
        """
        Calculate novelty score for an experience.

        Args:
            experience: Experience to evaluate
            agent_history: Historical experiences for comparison

        Returns:
            Novelty score between 0.0 and 1.0
        """
        if not agent_history:
            return 1.0  # First experience is always novel

        # Compare state similarity with recent experiences
        similarities = []
        for hist_exp in agent_history[-50:]:  # Compare with last 50 experiences
            similarity = self._calculate_state_similarity(
                experience.state, hist_exp.state
            )
            similarities.append(similarity)

        # Novelty is inverse of maximum similarity
        max_similarity = max(similarities) if similarities else 0.0
        novelty_score = 1.0 - max_similarity

        return max(0.0, min(1.0, novelty_score))

    def _calculate_state_similarity(
        self, state1: np.ndarray, state2: np.ndarray
    ) -> float:
        """Calculate similarity between two states."""
        try:
            # Ensure states are numpy arrays
            s1 = np.array(state1, dtype=np.float32)
            s2 = np.array(state2, dtype=np.float32)

            # Handle different shapes
            if s1.shape != s2.shape:
                return 0.0

            # Calculate cosine similarity
            dot_product = np.dot(s1, s2)
            norm1 = np.linalg.norm(s1)
            norm2 = np.linalg.norm(s2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            similarity = dot_product / (norm1 * norm2)
            return max(0.0, min(1.0, similarity))

        except Exception as e:
            self.logger.warning("Error calculating state similarity: %s", str(e))
            return 0.0

    def reset(self):
        """Reset filter state."""
        self.logger.debug("Experience filter reset")


class StateNoveltyTracker:
    """Tracks state novelty for experience sharing decisions."""

    def __init__(self, window_size: int = 1000):
        """
        Initialize novelty tracker.

        Args:
            window_size: Size of the sliding window for novelty tracking
        """
        self.window_size = window_size
        self.state_history = deque(maxlen=window_size)
        self.logger = get_logger(__name__ + ".StateNoveltyTracker")

    def add_state(self, state: np.ndarray):
        """Add a state to the novelty tracking history."""
        self.state_history.append(np.array(state, dtype=np.float32))

    def calculate_novelty(self, state: np.ndarray) -> float:
        """
        Calculate novelty score for a given state.

        Args:
            state: State to evaluate for novelty

        Returns:
            Novelty score between 0.0 and 1.0
        """
        if not self.state_history:
            return 1.0

        state_array = np.array(state, dtype=np.float32)
        similarities = []

        for hist_state in self.state_history:
            try:
                if hist_state.shape == state_array.shape:
                    # Calculate Euclidean distance normalized by state dimension
                    distance = np.linalg.norm(state_array - hist_state)
                    max_distance = np.sqrt(
                        len(state_array)
                    )  # Maximum possible distance
                    similarity = 1.0 - (distance / max_distance)
                    similarities.append(max(0.0, similarity))
            except Exception as e:
                self.logger.debug("Error calculating novelty: %s", str(e))
                continue

        if not similarities:
            return 1.0

        # Novelty is inverse of maximum similarity
        max_similarity = max(similarities)
        novelty = 1.0 - max_similarity

        return max(0.0, min(1.0, novelty))


class ExperienceSharing:
    """Manages experience sharing strategies and policies."""

    def __init__(self, config: ExperienceConfig):
        """
        Initialize experience sharing manager.

        Args:
            config: Experience sharing configuration
        """
        self.config = config
        self.logger = get_logger(__name__ + ".ExperienceSharing")

        # Sharing strategies
        self.sharing_strategies = {
            "share_all": self._share_all_strategy,
            "share_high_reward": self._share_high_reward_strategy,
            "share_novel": self._share_novel_strategy,
            "adaptive": self._adaptive_strategy,
        }

        self.current_strategy = "adaptive"

    def select_experiences_to_share(
        self,
        agent_id: str,
        experiences: List[Experience],
        metadata_list: List[ExperienceMetadata],
    ) -> List[Tuple[Experience, ExperienceMetadata]]:
        """
        Select experiences to share based on current strategy.

        Args:
            agent_id: ID of the agent sharing experiences
            experiences: List of experiences to consider
            metadata_list: Corresponding metadata for experiences

        Returns:
            List of (experience, metadata) tuples to share
        """
        if len(experiences) != len(metadata_list):
            self.logger.error(
                "Experience and metadata lists have different lengths: %d vs %d",
                len(experiences),
                len(metadata_list),
            )
            return []

        strategy_func = self.sharing_strategies.get(
            self.current_strategy, self._adaptive_strategy
        )

        return strategy_func(agent_id, experiences, metadata_list)

    def _share_all_strategy(
        self,
        agent_id: str,
        experiences: List[Experience],
        metadata_list: List[ExperienceMetadata],
    ) -> List[Tuple[Experience, ExperienceMetadata]]:
        """Share all experiences."""
        return list(zip(experiences, metadata_list))

    def _share_high_reward_strategy(
        self,
        agent_id: str,
        experiences: List[Experience],
        metadata_list: List[ExperienceMetadata],
    ) -> List[Tuple[Experience, ExperienceMetadata]]:
        """Share only high-reward experiences."""
        shared = []
        for exp, meta in zip(experiences, metadata_list):
            if meta.reward >= self.config.high_reward_threshold:
                shared.append((exp, meta))
        return shared

    def _share_novel_strategy(
        self,
        agent_id: str,
        experiences: List[Experience],
        metadata_list: List[ExperienceMetadata],
    ) -> List[Tuple[Experience, ExperienceMetadata]]:
        """Share only novel experiences."""
        shared = []
        for exp, meta in zip(experiences, metadata_list):
            if meta.novelty_score >= self.config.novelty_threshold:
                shared.append((exp, meta))
        return shared

    def _adaptive_strategy(
        self,
        agent_id: str,
        experiences: List[Experience],
        metadata_list: List[ExperienceMetadata],
    ) -> List[Tuple[Experience, ExperienceMetadata]]:
        """Adaptive sharing strategy combining multiple criteria."""
        shared = []

        for exp, meta in zip(experiences, metadata_list):
            # Always share high-reward experiences
            if meta.reward >= self.config.high_reward_threshold:
                shared.append((exp, meta))
                continue

            # Share novel experiences
            if meta.novelty_score >= self.config.novelty_threshold:
                shared.append((exp, meta))
                continue

            # Random sharing for exploration
            if np.random.random() < self.config.sharing_probability:
                shared.append((exp, meta))

        return shared

    def set_strategy(self, strategy: str):
        """Set the current sharing strategy."""
        if strategy in self.sharing_strategies:
            self.current_strategy = strategy
            self.logger.info("Sharing strategy set to: %s", strategy)
        else:
            self.logger.warning(
                "Unknown sharing strategy: %s. Available: %s",
                strategy,
                list(self.sharing_strategies.keys()),
            )


class SharedExperienceManager:
    """
    Main manager for shared experience system.

    This class coordinates experience sharing between agents, manages
    shared buffers, and provides sampling mechanisms for cross-agent learning.
    """

    def __init__(self, config: ExperienceConfig):
        """
        Initialize shared experience manager.

        Args:
            config: Experience sharing configuration
        """
        self.config = config
        self.logger = get_logger(__name__ + ".SharedExperienceManager")

        # Agent-specific buffers
        self.agent_buffers: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=config.agent_buffer_size)
        )
        self.agent_metadata: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=config.agent_buffer_size)
        )

        # Shared buffer for cross-agent experiences
        self.shared_buffer = deque(maxlen=config.shared_buffer_size)
        self.shared_metadata = deque(maxlen=config.shared_buffer_size)

        # Components
        self.experience_filter = ExperienceFilter(config)
        self.novelty_tracker = StateNoveltyTracker()
        self.experience_sharing = ExperienceSharing(config)

        # Statistics
        self.stats = {
            "total_experiences_stored": 0,
            "total_experiences_shared": 0,
            "experiences_by_agent": defaultdict(int),
            "sharing_reasons": defaultdict(int),
            "sampling_requests": 0,
        }

        # Registered agents
        self.registered_agents = set()

        self.logger.info("Initialized SharedExperienceManager")

    def register_agent(self, agent_id: str):
        """Register an agent for experience sharing."""
        self.registered_agents.add(agent_id)
        self.logger.info("Registered agent for experience sharing: %s", agent_id)

    def store_experience(self, agent_id: str, experience: Experience) -> bool:
        """
        Store an experience from an agent.

        Args:
            agent_id: ID of the agent storing the experience
            experience: Experience to store

        Returns:
            True if experience was stored successfully
        """
        try:
            # Create metadata
            novelty_score = self.novelty_tracker.calculate_novelty(experience.state)
            metadata = ExperienceMetadata(
                agent_id=agent_id,
                timestamp=time.time(),
                reward=experience.reward,
                novelty_score=novelty_score,
                sharing_reason="stored",
            )

            # Store in agent buffer
            self.agent_buffers[agent_id].append(experience)
            self.agent_metadata[agent_id].append(metadata)

            # Update novelty tracker
            self.novelty_tracker.add_state(experience.state)

            # Update statistics
            self.stats["total_experiences_stored"] += 1
            self.stats["experiences_by_agent"][agent_id] += 1

            # Check if experience should be shared
            agent_history = list(self.agent_buffers[agent_id])
            should_share, reason = self.experience_filter.should_share_experience(
                experience, metadata, agent_history
            )

            if should_share:
                self._share_experience(experience, metadata, reason)

            self.logger.debug(
                "Stored experience for agent %s (novelty=%.3f, reward=%.3f, shared=%s)",
                agent_id,
                novelty_score,
                experience.reward,
                should_share,
            )

            return True

        except Exception as e:
            self.logger.error(
                "Failed to store experience for agent %s: %s", agent_id, str(e)
            )
            return False

    def _share_experience(
        self, experience: Experience, metadata: ExperienceMetadata, reason: str
    ):
        """Share an experience to the shared buffer."""
        # Update metadata with sharing reason
        metadata.sharing_reason = reason

        # Add to shared buffer
        self.shared_buffer.append(experience)
        self.shared_metadata.append(metadata)

        # Update statistics
        self.stats["total_experiences_shared"] += 1
        self.stats["sharing_reasons"][reason] += 1

        self.logger.debug(
            "Shared experience from agent %s (reason: %s, reward=%.3f)",
            metadata.agent_id,
            reason,
            metadata.reward,
        )

    def sample_experiences(
        self, requesting_agent_id: str, batch_size: int, exclude_own: bool = True
    ) -> List[Experience]:
        """
        Sample experiences for an agent from shared buffer.

        Args:
            requesting_agent_id: ID of the agent requesting experiences
            batch_size: Number of experiences to sample
            exclude_own: Whether to exclude agent's own experiences

        Returns:
            List of sampled experiences
        """
        self.stats["sampling_requests"] += 1

        # Get available experiences
        available_experiences = []
        available_metadata = []

        for exp, meta in zip(self.shared_buffer, self.shared_metadata):
            # Skip expired experiences
            if meta.is_expired(self.config.max_age_hours):
                continue

            # Skip own experiences if requested
            if exclude_own and meta.agent_id == requesting_agent_id:
                continue

            available_experiences.append(exp)
            available_metadata.append(meta)

        # Sample experiences
        if len(available_experiences) <= batch_size:
            sampled_experiences = available_experiences
            sampled_metadata = available_metadata
        else:
            # Weighted sampling based on reward and novelty
            weights = []
            for meta in available_metadata:
                weight = meta.reward + meta.novelty_score
                weights.append(weight)

            # Normalize weights
            if sum(weights) > 0:
                weights = np.array(weights) / sum(weights)
                indices = np.random.choice(
                    len(available_experiences),
                    size=batch_size,
                    replace=False,
                    p=weights,
                )
            else:
                indices = np.random.choice(
                    len(available_experiences), size=batch_size, replace=False
                )

            sampled_experiences = [available_experiences[i] for i in indices]
            sampled_metadata = [available_metadata[i] for i in indices]

        # Update access statistics
        for meta in sampled_metadata:
            meta.update_access()

        self.logger.debug(
            "Sampled %d experiences for agent %s (requested: %d, available: %d)",
            len(sampled_experiences),
            requesting_agent_id,
            batch_size,
            len(available_experiences),
        )

        return sampled_experiences

    def get_agent_experiences(
        self, agent_id: str, count: Optional[int] = None
    ) -> List[Experience]:
        """
        Get experiences for a specific agent.

        Args:
            agent_id: ID of the agent
            count: Number of recent experiences to return (None for all)

        Returns:
            List of experiences
        """
        experiences = list(self.agent_buffers[agent_id])

        if count is not None:
            experiences = experiences[-count:]

        return experiences

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about experience sharing."""
        # Calculate buffer utilization
        agent_utilization = {}
        for agent_id in self.registered_agents:
            buffer_size = len(self.agent_buffers[agent_id])
            utilization = buffer_size / self.config.agent_buffer_size
            agent_utilization[agent_id] = {
                "buffer_size": buffer_size,
                "utilization": utilization,
            }

        shared_utilization = len(self.shared_buffer) / self.config.shared_buffer_size

        return {
            "configuration": {
                "agent_buffer_size": self.config.agent_buffer_size,
                "shared_buffer_size": self.config.shared_buffer_size,
                "high_reward_threshold": self.config.high_reward_threshold,
                "novelty_threshold": self.config.novelty_threshold,
            },
            "statistics": self.stats.copy(),
            "buffer_utilization": {
                "agents": agent_utilization,
                "shared": {
                    "buffer_size": len(self.shared_buffer),
                    "utilization": shared_utilization,
                },
            },
            "registered_agents": list(self.registered_agents),
            "sharing_strategy": self.experience_sharing.current_strategy,
        }

    def clear(self):
        """Clear all buffers and reset statistics."""
        self.agent_buffers.clear()
        self.agent_metadata.clear()
        self.shared_buffer.clear()
        self.shared_metadata.clear()

        # Reset components
        self.experience_filter.reset()
        self.novelty_tracker = StateNoveltyTracker()

        # Reset statistics
        self.stats = {
            "total_experiences_stored": 0,
            "total_experiences_shared": 0,
            "experiences_by_agent": defaultdict(int),
            "sharing_reasons": defaultdict(int),
            "sampling_requests": 0,
        }

        self.logger.info("Cleared all experience buffers and reset statistics")

    def cleanup_expired_experiences(self):
        """Remove expired experiences from buffers."""
        current_time = time.time()
        max_age_seconds = self.config.max_age_hours * 3600

        # Clean shared buffer
        valid_shared = []
        valid_shared_meta = []

        for exp, meta in zip(self.shared_buffer, self.shared_metadata):
            if (current_time - meta.timestamp) <= max_age_seconds:
                valid_shared.append(exp)
                valid_shared_meta.append(meta)

        self.shared_buffer.clear()
        self.shared_buffer.extend(valid_shared)
        self.shared_metadata.clear()
        self.shared_metadata.extend(valid_shared_meta)

        # Clean agent buffers
        for agent_id in self.registered_agents:
            valid_agent = []
            valid_agent_meta = []

            agent_buffer = self.agent_buffers[agent_id]
            agent_meta = self.agent_metadata[agent_id]

            for exp, meta in zip(agent_buffer, agent_meta):
                if (current_time - meta.timestamp) <= max_age_seconds:
                    valid_agent.append(exp)
                    valid_agent_meta.append(meta)

            agent_buffer.clear()
            agent_buffer.extend(valid_agent)
            agent_meta.clear()
            agent_meta.extend(valid_agent_meta)

        self.logger.debug("Cleaned up expired experiences")


def create_shared_experience_manager(
    config: Optional[ExperienceConfig] = None,
) -> SharedExperienceManager:
    """
    Factory function to create a shared experience manager.

    Args:
        config: Optional configuration, uses defaults if None

    Returns:
        Configured SharedExperienceManager instance
    """
    if config is None:
        config = ExperienceConfig()

    return SharedExperienceManager(config)
