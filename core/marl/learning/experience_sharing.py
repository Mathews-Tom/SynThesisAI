"""
Experience Sharing Infrastructure

This module provides sophisticated experience sharing mechanisms for multi-agent
reinforcement learning, enabling agents to learn from each other's experiences
and improve coordination through shared knowledge.
"""

# Standard Library
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Tuple

# Third-Party Library
import numpy as np

# SynThesisAI Modules
from ..agents.experience import Experience
from ..config import ExperienceConfig
from ..exceptions import ExperienceBufferError
from ..logging_config import get_marl_logger

logger = logging.getLogger(__name__)


class ExperienceValue(Enum):
    """Enumeration of experience value types for sharing decisions."""

    HIGH_REWARD = "high_reward"
    NOVEL_STATE = "novel_state"
    COORDINATION_SUCCESS = "coordination_success"
    RARE_ACTION = "rare_action"
    LEARNING_MILESTONE = "learning_milestone"
    ERROR_CORRECTION = "error_correction"


@dataclass
class ExperienceMetadata:
    """Metadata for experience sharing decisions."""

    agent_id: str
    experience_id: str
    value_type: ExperienceValue
    value_score: float
    sharing_priority: int
    creation_time: float = field(default_factory=time.time)
    share_count: int = 0
    success_feedback: List[bool] = field(default_factory=list)

    @property
    def sharing_effectiveness(self) -> float:
        """Calculate sharing effectiveness based on feedback."""
        if not self.success_feedback:
            return 0.5  # Neutral when no feedback
        return sum(self.success_feedback) / len(self.success_feedback)


class SharedExperienceManager:
    """
    Advanced experience sharing manager for multi-agent learning.

    Manages the sharing of valuable experiences between agents to accelerate
    learning and improve coordination through knowledge transfer.
    """

    def __init__(self, config: ExperienceConfig):
        """
        Initialize shared experience manager.

        Args:
            config: Experience sharing configuration
        """
        self.config = config
        self.logger = get_marl_logger("experience_sharing")

        # Experience storage
        self.shared_experiences: Dict[str, Experience] = {}
        self.experience_metadata: Dict[str, ExperienceMetadata] = {}
        self.agent_contributions: Dict[str, int] = defaultdict(int)

        # Value assessment
        self.state_novelty_tracker = StateNoveltyTracker()
        self.action_frequency_tracker = ActionFrequencyTracker()
        self.coordination_success_tracker = CoordinationSuccessTracker()

        # Sharing statistics
        self.sharing_stats = {
            "total_shared": 0,
            "total_consumed": 0,
            "successful_shares": 0,
            "failed_shares": 0,
            "agent_stats": defaultdict(lambda: {"shared": 0, "consumed": 0}),
        }

        self.logger.log_experience_sharing("system", "manager_initialized", 0.0, True)

    def evaluate_experience_value(
        self, experience: Experience, agent_id: str, context: Dict[str, Any]
    ) -> Tuple[bool, ExperienceValue, float]:
        """
        Evaluate if an experience is valuable enough to share.

        Args:
            experience: Experience to evaluate
            agent_id: ID of the agent that generated the experience
            context: Additional context for evaluation

        Returns:
            Tuple of (should_share, value_type, value_score)
        """
        try:
            value_assessments = []

            # High reward assessment
            if experience.reward > self.config.high_reward_threshold:
                reward_score = min(
                    experience.reward / self.config.high_reward_threshold, 2.0
                )
                value_assessments.append((ExperienceValue.HIGH_REWARD, reward_score))

            # State novelty assessment
            novelty_score = self.state_novelty_tracker.assess_novelty(
                experience.state, agent_id
            )
            if novelty_score > self.config.novelty_threshold:
                value_assessments.append((ExperienceValue.NOVEL_STATE, novelty_score))

            # Coordination success assessment
            if context.get("coordination_success", False):
                coord_score = context.get("coordination_quality", 0.8)
                value_assessments.append(
                    (ExperienceValue.COORDINATION_SUCCESS, coord_score)
                )

            # Rare action assessment
            action_rarity = self.action_frequency_tracker.assess_rarity(
                experience.action, agent_id
            )
            if action_rarity > 0.7:  # Rare action threshold
                value_assessments.append((ExperienceValue.RARE_ACTION, action_rarity))

            # Learning milestone assessment
            if context.get("learning_milestone", False):
                milestone_score = context.get("milestone_importance", 0.9)
                value_assessments.append(
                    (ExperienceValue.LEARNING_MILESTONE, milestone_score)
                )

            # Error correction assessment
            if context.get("error_correction", False):
                correction_score = context.get("correction_importance", 0.8)
                value_assessments.append(
                    (ExperienceValue.ERROR_CORRECTION, correction_score)
                )

            # Determine if should share
            if not value_assessments:
                return False, ExperienceValue.HIGH_REWARD, 0.0

            # Select highest value assessment
            best_value_type, best_score = max(value_assessments, key=lambda x: x[1])
            should_share = (
                best_score > 0.6
                and np.random.random() < self.config.sharing_probability
            )

            return should_share, best_value_type, best_score

        except Exception as e:
            self.logger.log_error_with_context(
                e,
                {
                    "agent_id": agent_id,
                    "experience_reward": experience.reward,
                    "context_keys": list(context.keys()),
                },
            )
            return False, ExperienceValue.HIGH_REWARD, 0.0

    def share_experience(
        self,
        experience: Experience,
        agent_id: str,
        value_type: ExperienceValue,
        value_score: float,
    ) -> str:
        """
        Share an experience with other agents.

        Args:
            experience: Experience to share
            agent_id: ID of the sharing agent
            value_type: Type of value this experience provides
            value_score: Numerical value score

        Returns:
            Experience ID for tracking
        """
        try:
            # Generate unique experience ID
            experience_id = (
                f"{agent_id}_{int(time.time() * 1000)}_{len(self.shared_experiences)}"
            )

            # Create metadata
            metadata = ExperienceMetadata(
                agent_id=agent_id,
                experience_id=experience_id,
                value_type=value_type,
                value_score=value_score,
                sharing_priority=self._calculate_sharing_priority(
                    value_type, value_score
                ),
            )

            # Store experience and metadata
            self.shared_experiences[experience_id] = experience
            self.experience_metadata[experience_id] = metadata

            # Update statistics
            self.agent_contributions[agent_id] += 1
            self.sharing_stats["total_shared"] += 1
            self.sharing_stats["agent_stats"][agent_id]["shared"] += 1

            # Log sharing
            self.logger.log_experience_sharing(
                agent_id, value_type.value, value_score, True
            )

            return experience_id

        except Exception as e:
            error_msg = f"Failed to share experience from agent {agent_id}"
            self.logger.log_error_with_context(
                e,
                {
                    "agent_id": agent_id,
                    "value_type": value_type.value,
                    "value_score": value_score,
                },
            )
            raise ExperienceBufferError(
                error_msg, buffer_type="shared", operation="share"
            ) from e

    def get_experiences_for_agent(
        self, requesting_agent_id: str, max_experiences: int = 10
    ) -> List[Tuple[str, Experience]]:
        """
        Get valuable experiences for a specific agent.

        Args:
            requesting_agent_id: ID of the agent requesting experiences
            max_experiences: Maximum number of experiences to return

        Returns:
            List of (experience_id, experience) tuples
        """
        try:
            # Filter experiences (don't share agent's own experiences back to them)
            available_experiences = [
                (exp_id, exp)
                for exp_id, exp in self.shared_experiences.items()
                if self.experience_metadata[exp_id].agent_id != requesting_agent_id
            ]

            if not available_experiences:
                return []

            # Sort by sharing priority and value score
            def sort_key(item):
                exp_id, _ = item
                metadata = self.experience_metadata[exp_id]
                return (metadata.sharing_priority, metadata.value_score)

            sorted_experiences = sorted(
                available_experiences, key=sort_key, reverse=True
            )

            # Select top experiences
            selected = sorted_experiences[:max_experiences]

            # Update consumption statistics
            for exp_id, _ in selected:
                self.experience_metadata[exp_id].share_count += 1

            self.sharing_stats["total_consumed"] += len(selected)
            self.sharing_stats["agent_stats"][requesting_agent_id]["consumed"] += len(
                selected
            )

            self.logger.log_experience_sharing(
                requesting_agent_id, "experiences_retrieved", len(selected), True
            )

            return selected

        except Exception as e:
            error_msg = f"Failed to get experiences for agent {requesting_agent_id}"
            self.logger.log_error_with_context(
                e,
                {
                    "requesting_agent_id": requesting_agent_id,
                    "max_experiences": max_experiences,
                    "available_count": len(self.shared_experiences),
                },
            )
            raise ExperienceBufferError(
                error_msg, buffer_type="shared", operation="retrieve"
            ) from e

    def provide_feedback(
        self, experience_id: str, success: bool, feedback_agent_id: str
    ) -> None:
        """
        Provide feedback on the usefulness of a shared experience.

        Args:
            experience_id: ID of the experience
            success: Whether the experience was useful
            feedback_agent_id: ID of the agent providing feedback
        """
        if experience_id in self.experience_metadata:
            metadata = self.experience_metadata[experience_id]
            metadata.success_feedback.append(success)

            # Update statistics
            if success:
                self.sharing_stats["successful_shares"] += 1
            else:
                self.sharing_stats["failed_shares"] += 1

            self.logger.log_experience_sharing(
                feedback_agent_id, "feedback_provided", 1.0 if success else 0.0, success
            )

    def _calculate_sharing_priority(
        self, value_type: ExperienceValue, value_score: float
    ) -> int:
        """Calculate sharing priority based on value type and score."""
        base_priorities = {
            ExperienceValue.COORDINATION_SUCCESS: 100,
            ExperienceValue.LEARNING_MILESTONE: 90,
            ExperienceValue.ERROR_CORRECTION: 80,
            ExperienceValue.HIGH_REWARD: 70,
            ExperienceValue.NOVEL_STATE: 60,
            ExperienceValue.RARE_ACTION: 50,
        }

        base_priority = base_priorities.get(value_type, 50)
        score_bonus = int(value_score * 20)  # Up to 20 bonus points

        return base_priority + score_bonus

    def cleanup_old_experiences(self, max_age_hours: float = 24.0) -> int:
        """
        Clean up old experiences to manage memory usage.

        Args:
            max_age_hours: Maximum age of experiences to keep

        Returns:
            Number of experiences removed
        """
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600

        old_experience_ids = [
            exp_id
            for exp_id, metadata in self.experience_metadata.items()
            if current_time - metadata.creation_time > max_age_seconds
        ]

        # Remove old experiences
        for exp_id in old_experience_ids:
            del self.shared_experiences[exp_id]
            del self.experience_metadata[exp_id]

        if old_experience_ids:
            self.logger.log_experience_sharing(
                "system", "cleanup_completed", len(old_experience_ids), True
            )

        return len(old_experience_ids)

    def get_sharing_statistics(self) -> Dict[str, Any]:
        """Get comprehensive sharing statistics."""
        total_experiences = len(self.shared_experiences)

        # Calculate effectiveness metrics
        total_feedback = (
            self.sharing_stats["successful_shares"]
            + self.sharing_stats["failed_shares"]
        )
        effectiveness = (
            self.sharing_stats["successful_shares"] / total_feedback
            if total_feedback > 0
            else 0.0
        )

        # Value type distribution
        value_type_counts = defaultdict(int)
        for metadata in self.experience_metadata.values():
            value_type_counts[metadata.value_type.value] += 1

        return {
            "total_shared_experiences": total_experiences,
            "total_shared": self.sharing_stats["total_shared"],
            "total_consumed": self.sharing_stats["total_consumed"],
            "sharing_effectiveness": effectiveness,
            "successful_shares": self.sharing_stats["successful_shares"],
            "failed_shares": self.sharing_stats["failed_shares"],
            "agent_contributions": dict(self.agent_contributions),
            "agent_stats": dict(self.sharing_stats["agent_stats"]),
            "value_type_distribution": dict(value_type_counts),
        }


class StateNoveltyTracker:
    """Tracks state novelty for experience value assessment."""

    def __init__(self, novelty_threshold: float = 0.1):
        """
        Initialize state novelty tracker.

        Args:
            novelty_threshold: Threshold for considering states novel
        """
        self.novelty_threshold = novelty_threshold
        self.agent_state_histories: Dict[str, List[np.ndarray]] = defaultdict(list)
        self.max_history_size = 1000

    def assess_novelty(self, state: np.ndarray, agent_id: str) -> float:
        """
        Assess novelty of a state for a specific agent.

        Args:
            state: State to assess
            agent_id: ID of the agent

        Returns:
            Novelty score between 0 and 1
        """
        agent_history = self.agent_state_histories[agent_id]

        if not agent_history:
            # First state is always novel
            self.agent_state_histories[agent_id].append(state.copy())
            return 1.0

        # Calculate minimum distance to existing states
        distances = [np.linalg.norm(state - hist_state) for hist_state in agent_history]
        min_distance = min(distances)

        # Normalize distance to novelty score
        novelty_score = min(min_distance / (np.linalg.norm(state) + 1e-8), 1.0)

        # Add to history if novel enough
        if novelty_score > self.novelty_threshold:
            self.agent_state_histories[agent_id].append(state.copy())

            # Maintain history size
            if len(self.agent_state_histories[agent_id]) > self.max_history_size:
                self.agent_state_histories[agent_id] = self.agent_state_histories[
                    agent_id
                ][-self.max_history_size :]

        return novelty_score


class ActionFrequencyTracker:
    """Tracks action frequency for rare action detection."""

    def __init__(self):
        """Initialize action frequency tracker."""
        self.agent_action_counts: Dict[str, Dict[int, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self.agent_total_actions: Dict[str, int] = defaultdict(int)

    def assess_rarity(self, action: int, agent_id: str) -> float:
        """
        Assess rarity of an action for a specific agent.

        Args:
            action: Action to assess
            agent_id: ID of the agent

        Returns:
            Rarity score between 0 and 1
        """
        # Update counts
        self.agent_action_counts[agent_id][action] += 1
        self.agent_total_actions[agent_id] += 1

        # Calculate rarity (inverse of frequency)
        action_count = self.agent_action_counts[agent_id][action]
        total_actions = self.agent_total_actions[agent_id]

        frequency = action_count / total_actions
        rarity = 1.0 - frequency

        return rarity


class CoordinationSuccessTracker:
    """Tracks coordination success patterns for experience value assessment."""

    def __init__(self):
        """Initialize coordination success tracker."""
        self.success_history: deque = deque(maxlen=1000)
        self.success_patterns: Dict[str, int] = defaultdict(int)

    def record_coordination(self, success: bool, context: Dict[str, Any]) -> None:
        """
        Record a coordination attempt.

        Args:
            success: Whether coordination was successful
            context: Context information about the coordination
        """
        self.success_history.append(success)

        # Track patterns that lead to success
        if success:
            pattern_key = self._extract_pattern_key(context)
            self.success_patterns[pattern_key] += 1

    def _extract_pattern_key(self, context: Dict[str, Any]) -> str:
        """Extract a pattern key from coordination context."""
        # Simple pattern extraction - can be enhanced
        agents = context.get("participating_agents", [])
        strategy = context.get("coordination_strategy", "unknown")

        return f"{len(agents)}_{strategy}"

    def get_success_rate(self) -> float:
        """Get overall coordination success rate."""
        if not self.success_history:
            return 0.0

        return sum(self.success_history) / len(self.success_history)
