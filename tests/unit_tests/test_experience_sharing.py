"""
Unit tests for experience sharing infrastructure.

This module tests the experience sharing components of the MARL learning
infrastructure, including filters, sharing policies, and the shared
experience manager.
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from core.marl.agents.experience import Experience
from core.marl.config import ExperienceConfig
from core.marl.exceptions import ExperienceBufferError
from core.marl.learning.experience_sharing import (
    ActionFrequencyTracker,
    CoordinationSuccessTracker,
    ExperienceMetadata,
    ExperienceValue,
    SharedExperienceManager,
    StateNoveltyTracker,
)


class TestExperienceMetadata:
    """Test ExperienceMetadata class."""

    def test_filter_initialization(self):
        """Test filter initialization."""
        config = ExperienceConfig()
        filter = ExperienceFilter(config)

        assert filter.high_reward_threshold == config.high_reward_threshold
        assert filter.novelty_threshold == config.novelty_threshold
        assert filter.sharing_probability == config.sharing_probability
        assert isinstance(filter.seen_state_actions, set)

    def test_high_reward_experience(self):
        """Test high reward experience filtering."""
        config = ExperienceConfig()
        config.high_reward_threshold = 0.8
        filter = ExperienceFilter(config)

        # Create high reward experience
        exp = Experience(
            state=np.array([1.0, 2.0, 3.0]),
            action=0,
            reward=1.0,  # > threshold
            next_state=np.array([1.1, 2.1, 3.1]),
            done=False,
        )

        assert filter.is_valuable_experience(exp) is True

    def test_low_reward_experience(self):
        """Test low reward experience filtering."""
        config = ExperienceConfig()
        config.high_reward_threshold = 0.8
        config.novelty_threshold = 0.0  # No novelty sharing
        config.sharing_probability = 0.0  # No random sharing
        filter = ExperienceFilter(config)

        # Create low reward experience
        exp = Experience(
            state=np.array([1.0, 2.0, 3.0]),
            action=0,
            reward=0.5,  # < threshold
            next_state=np.array([1.1, 2.1, 3.1]),
            done=False,
        )

        assert filter.is_valuable_experience(exp) is False

    def test_novel_experience(self):
        """Test novel experience filtering."""
        config = ExperienceConfig()
        config.high_reward_threshold = 0.8
        config.novelty_threshold = 1.0  # Always share novel experiences
        config.sharing_probability = 0.0  # No random sharing
        filter = ExperienceFilter(config)

        # Create experience
        exp1 = Experience(
            state=np.array([1.0, 2.0, 3.0]),
            action=0,
            reward=0.5,  # < threshold
            next_state=np.array([1.1, 2.1, 3.1]),
            done=False,
        )

        # First time should be novel
        assert filter.is_valuable_experience(exp1) is True

        # Create similar experience
        exp2 = Experience(
            state=np.array([1.0, 2.0, 3.0]),  # Same state
            action=0,  # Same action
            reward=0.5,
            next_state=np.array([1.2, 2.2, 3.2]),
            done=False,
        )

        # Second time should not be novel
        assert filter.is_valuable_experience(exp2) is False

    def test_filter_reset(self):
        """Test filter reset."""
        config = ExperienceConfig()
        config.high_reward_threshold = 0.8
        config.novelty_threshold = 1.0  # Always share novel experiences
        filter = ExperienceFilter(config)

        # Add some state-action pairs
        exp = Experience(
            state=np.array([1.0, 2.0, 3.0]),
            action=0,
            reward=0.5,
            next_state=np.array([1.1, 2.1, 3.1]),
            done=False,
        )

        filter.is_valuable_experience(exp)
        assert len(filter.seen_state_actions) > 0

        # Reset filter
        filter.reset()
        assert len(filter.seen_state_actions) == 0


class TestStateNoveltyTracker:
    """Test StateNoveltyTracker class."""

    def test_sharing_initialization(self):
        """Test sharing initialization."""
        config = ExperienceConfig()
        sharing = ExperienceSharing(config)

        assert sharing.default_strategy == "selective"
        assert len(sharing.sharing_strategies) == 4

    def test_share_all_strategy(self):
        """Test share all strategy."""
        config = ExperienceConfig()
        sharing = ExperienceSharing(config)

        exp = Experience(
            state=np.array([1.0, 2.0, 3.0]),
            action=0,
            reward=0.5,
            next_state=np.array([1.1, 2.1, 3.1]),
            done=False,
        )

        result = sharing.share_experience(
            exp, "generator", ["validator", "curriculum"], strategy="all"
        )

        assert result == {"validator": True, "curriculum": True}

    def test_share_high_reward_strategy(self):
        """Test high reward sharing strategy."""
        config = ExperienceConfig()
        config.high_reward_threshold = 0.8
        sharing = ExperienceSharing(config)

        # High reward experience
        exp_high = Experience(
            state=np.array([1.0, 2.0, 3.0]),
            action=0,
            reward=1.0,  # > threshold
            next_state=np.array([1.1, 2.1, 3.1]),
            done=False,
        )

        result_high = sharing.share_experience(
            exp_high, "generator", ["validator", "curriculum"], strategy="high_reward"
        )

        assert result_high == {"validator": True, "curriculum": True}

        # Low reward experience
        exp_low = Experience(
            state=np.array([1.0, 2.0, 3.0]),
            action=0,
            reward=0.5,  # < threshold
            next_state=np.array([1.1, 2.1, 3.1]),
            done=False,
        )

        result_low = sharing.share_experience(
            exp_low, "generator", ["validator", "curriculum"], strategy="high_reward"
        )

        assert result_low == {"validator": False, "curriculum": False}

    def test_unknown_strategy_fallback(self):
        """Test fallback to default strategy for unknown strategy."""
        config = ExperienceConfig()
        sharing = ExperienceSharing(config)

        exp = Experience(
            state=np.array([1.0, 2.0, 3.0]),
            action=0,
            reward=0.5,
            next_state=np.array([1.1, 2.1, 3.1]),
            done=False,
        )

        # Mock the default strategy to always return a known result
        sharing.sharing_strategies["selective"] = lambda *args: {
            "validator": True,
            "curriculum": False,
        }

        result = sharing.share_experience(
            exp, "generator", ["validator", "curriculum"], strategy="unknown_strategy"
        )

        assert result == {"validator": True, "curriculum": False}


class TestSharedExperienceManager:
    """Test SharedExperienceManager class."""

    def test_manager_initialization(self):
        """Test manager initialization."""
        config = ExperienceConfig()
        manager = SharedExperienceManager(config)

        assert manager.shared_buffer is not None
        assert isinstance(manager.agent_buffers, dict)
        assert manager.experience_filter is not None
        assert manager.experience_sharing is not None

    def test_register_agent(self):
        """Test agent registration."""
        config = ExperienceConfig()
        manager = SharedExperienceManager(config)

        # Register agent
        manager.register_agent("generator")

        assert "generator" in manager.agent_buffers
        assert len(manager.agent_buffers["generator"]) == 0

    def test_store_experience(self):
        """Test storing experience."""
        config = ExperienceConfig()
        manager = SharedExperienceManager(config)

        # Create experience
        exp = Experience(
            state=np.array([1.0, 2.0, 3.0]),
            action=0,
            reward=0.5,
            next_state=np.array([1.1, 2.1, 3.1]),
            done=False,
        )

        # Store experience
        result = manager.store_experience("generator", exp)

        # Agent should be registered automatically
        assert "generator" in manager.agent_buffers
        assert len(manager.agent_buffers["generator"]) == 1

    def test_sample_experiences(self):
        """Test sampling experiences."""
        config = ExperienceConfig()
        config.agent_buffer_size = 10
        manager = SharedExperienceManager(config)

        # Add experiences
        for i in range(5):
            exp = Experience(
                state=np.array([float(i), float(i + 1), float(i + 2)]),
                action=i % 2,
                reward=float(i) / 5.0,
                next_state=np.array([float(i + 1), float(i + 2), float(i + 3)]),
                done=i == 4,
            )
            manager.store_experience("generator", exp)

        # Sample experiences
        samples = manager.sample_experiences("generator", 3)

        assert len(samples) == 3
        assert all(isinstance(exp, Experience) for exp in samples)

    def test_sample_insufficient_experiences(self):
        """Test sampling with insufficient experiences."""
        config = ExperienceConfig()
        manager = SharedExperienceManager(config)

        # Register agent but don't add experiences
        manager.register_agent("generator")

        # Try to sample more than available
        with pytest.raises(ExperienceBufferError, match="Not enough experiences"):
            manager.sample_experiences("generator", 5)

    def test_get_statistics(self):
        """Test getting statistics."""
        config = ExperienceConfig()
        manager = SharedExperienceManager(config)

        # Add some experiences
        for i in range(3):
            exp = Experience(
                state=np.array([float(i), float(i + 1), float(i + 2)]),
                action=i % 2,
                reward=float(i) / 3.0,
                next_state=np.array([float(i + 1), float(i + 2), float(i + 3)]),
                done=i == 2,
            )
            manager.store_experience("generator", exp)

        # Get statistics
        stats = manager.get_statistics()

        assert "shared_buffer" in stats
        assert "agent_buffers" in stats
        assert "generator" in stats["agent_buffers"]
        assert stats["agent_buffers"]["generator"]["size"] == 3
        assert "registered_agents" in stats
        assert "generator" in stats["registered_agents"]

    def test_clear(self):
        """Test clearing all buffers."""
        config = ExperienceConfig()
        manager = SharedExperienceManager(config)

        # Add some experiences
        exp = Experience(
            state=np.array([1.0, 2.0, 3.0]),
            action=0,
            reward=0.5,
            next_state=np.array([1.1, 2.1, 3.1]),
            done=False,
        )
        manager.store_experience("generator", exp)

        # Clear buffers
        manager.clear()

        # Check buffers are empty
        assert len(manager.shared_buffer) == 0
        assert len(manager.agent_buffers["generator"]) == 0
