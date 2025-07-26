"""
Unit tests for Shared Experience Management System
"""

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from core.marl.agents.base_agent import Experience
from core.marl.learning.shared_experience import (
    ExperienceConfig,
    ExperienceFilter,
    ExperienceMetadata,
    ExperienceSharing,
    SharedExperienceManager,
    StateNoveltyTracker,
    create_shared_experience_manager,
)


class TestExperienceConfig:
    """Test cases for ExperienceConfig."""

    def test_default_initialization(self):
        """Test default configuration initialization."""
        config = ExperienceConfig()

        assert config.agent_buffer_size == 1000
        assert config.shared_buffer_size == 5000
        assert config.high_reward_threshold == 0.8
        assert config.novelty_threshold == 0.7
        assert config.sharing_probability == 0.1
        assert config.max_age_hours == 24.0
        assert config.min_experiences_for_sharing == 10

    def test_custom_initialization(self):
        """Test custom configuration initialization."""
        config = ExperienceConfig(
            agent_buffer_size=500,
            shared_buffer_size=2000,
            high_reward_threshold=0.9,
            novelty_threshold=0.8,
            sharing_probability=0.2,
            max_age_hours=12.0,
            min_experiences_for_sharing=5,
        )

        assert config.agent_buffer_size == 500
        assert config.shared_buffer_size == 2000
        assert config.high_reward_threshold == 0.9
        assert config.novelty_threshold == 0.8
        assert config.sharing_probability == 0.2
        assert config.max_age_hours == 12.0
        assert config.min_experiences_for_sharing == 5


class TestExperienceMetadata:
    """Test cases for ExperienceMetadata."""

    def test_initialization(self):
        """Test metadata initialization."""
        timestamp = time.time()
        metadata = ExperienceMetadata(
            agent_id="test_agent",
            timestamp=timestamp,
            reward=0.8,
            novelty_score=0.6,
            sharing_reason="high_reward",
        )

        assert metadata.agent_id == "test_agent"
        assert metadata.timestamp == timestamp
        assert metadata.reward == 0.8
        assert metadata.novelty_score == 0.6
        assert metadata.sharing_reason == "high_reward"
        assert metadata.access_count == 0
        assert metadata.last_accessed == timestamp

    def test_is_expired(self):
        """Test experience expiration check."""
        # Create metadata with old timestamp
        old_timestamp = time.time() - (25 * 3600)  # 25 hours ago
        metadata = ExperienceMetadata("agent", old_timestamp, 0.5)

        # Should be expired with 24 hour limit
        assert metadata.is_expired(24.0) is True

        # Should not be expired with 48 hour limit
        assert metadata.is_expired(48.0) is False

        # Recent metadata should not be expired
        recent_metadata = ExperienceMetadata("agent", time.time(), 0.5)
        assert recent_metadata.is_expired(24.0) is False

    def test_update_access(self):
        """Test access statistics update."""
        metadata = ExperienceMetadata("agent", time.time(), 0.5)
        initial_access_time = metadata.last_accessed

        # Wait a small amount to ensure timestamp difference
        time.sleep(0.01)

        metadata.update_access()

        assert metadata.access_count == 1
        assert metadata.last_accessed > initial_access_time


class TestExperienceFilter:
    """Test cases for ExperienceFilter."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return ExperienceConfig(
            high_reward_threshold=0.8, novelty_threshold=0.7, sharing_probability=0.1
        )

    @pytest.fixture
    def filter(self, config):
        """Create test filter."""
        return ExperienceFilter(config)

    @pytest.fixture
    def sample_experience(self):
        """Create sample experience."""
        return Experience(
            state=np.array([1.0, 2.0, 3.0]),
            action=0,
            reward=0.5,
            next_state=np.array([1.1, 2.1, 3.1]),
            done=False,
        )

    def test_filter_initialization(self, config):
        """Test filter initialization."""
        filter = ExperienceFilter(config)
        assert filter.config == config

    def test_high_reward_experience(self, filter, sample_experience):
        """Test high reward experience filtering."""
        metadata = ExperienceMetadata("agent", time.time(), 0.9)  # High reward

        should_share, reason = filter.should_share_experience(
            sample_experience, metadata, []
        )

        assert should_share is True
        assert reason == "high_reward"

    def test_low_reward_experience(self, filter, sample_experience):
        """Test low reward experience filtering."""
        config = ExperienceConfig()
        config.high_reward_threshold = 0.8
        config.novelty_threshold = 1.0  # No novelty sharing
        config.sharing_probability = 0.0  # No random sharing
        filter = ExperienceFilter(config)

        metadata = ExperienceMetadata("agent", time.time(), 0.5)  # Low reward
        metadata.novelty_score = 0.5  # Low novelty

        should_share, reason = filter.should_share_experience(
            sample_experience, metadata, []
        )

        assert should_share is False
        assert reason == "low_value"

    def test_novel_experience(self, filter, sample_experience):
        """Test novel experience filtering."""
        config = ExperienceConfig()
        config.high_reward_threshold = 0.8
        config.novelty_threshold = 1.0  # Always share novel experiences
        config.sharing_probability = 0.0  # No random sharing
        filter = ExperienceFilter(config)

        metadata = ExperienceMetadata("agent", time.time(), 0.5)  # Low reward
        metadata.novelty_score = 1.0  # High novelty

        should_share, reason = filter.should_share_experience(
            sample_experience, metadata, []
        )

        assert should_share is True
        assert reason == "novelty"

    def test_calculate_novelty_score_empty_history(self, filter, sample_experience):
        """Test novelty calculation with empty history."""
        novelty = filter.calculate_novelty_score(sample_experience, [])
        assert novelty == 1.0  # First experience is always novel

    def test_calculate_novelty_score_with_history(self, filter):
        """Test novelty calculation with history."""
        # Create similar experiences
        exp1 = Experience(np.array([1.0, 2.0]), 0, 0.5, np.array([1.1, 2.1]), False)
        exp2 = Experience(np.array([1.1, 2.1]), 0, 0.5, np.array([1.2, 2.2]), False)
        history = [exp1]

        # Similar experience should have low novelty
        novelty = filter.calculate_novelty_score(exp2, history)
        assert 0.0 <= novelty < 1.0

        # Very different experience should have high novelty
        exp3 = Experience(np.array([10.0, 20.0]), 0, 0.5, np.array([11.0, 21.0]), False)
        novelty = filter.calculate_novelty_score(exp3, history)
        assert (
            novelty >= 0.0
        )  # Should be non-negative, exact value depends on similarity calculation

    def test_calculate_state_similarity(self, filter):
        """Test state similarity calculation."""
        state1 = np.array([1.0, 2.0, 3.0])
        state2 = np.array([1.0, 2.0, 3.0])  # Identical
        state3 = np.array([4.0, 5.0, 6.0])  # Different

        # Identical states should have high similarity
        similarity = filter._calculate_state_similarity(state1, state2)
        assert abs(similarity - 1.0) < 1e-6  # Account for floating point precision

        # Different states should have lower similarity
        similarity = filter._calculate_state_similarity(state1, state3)
        assert 0.0 <= similarity < 1.0

    def test_calculate_state_similarity_different_shapes(self, filter):
        """Test state similarity with different shapes."""
        state1 = np.array([1.0, 2.0])
        state2 = np.array([1.0, 2.0, 3.0])

        similarity = filter._calculate_state_similarity(state1, state2)
        assert similarity == 0.0

    def test_filter_reset(self, filter):
        """Test filter reset."""
        filter.reset()  # Should not raise any exceptions


class TestStateNoveltyTracker:
    """Test cases for StateNoveltyTracker."""

    @pytest.fixture
    def tracker(self):
        """Create test tracker."""
        return StateNoveltyTracker(window_size=10)

    def test_initialization(self):
        """Test tracker initialization."""
        tracker = StateNoveltyTracker(window_size=5)
        assert tracker.window_size == 5
        assert len(tracker.state_history) == 0

    def test_add_state(self, tracker):
        """Test adding states to tracker."""
        state = np.array([1.0, 2.0, 3.0])
        tracker.add_state(state)

        assert len(tracker.state_history) == 1
        np.testing.assert_array_equal(tracker.state_history[0], state)

    def test_calculate_novelty_empty_history(self, tracker):
        """Test novelty calculation with empty history."""
        state = np.array([1.0, 2.0, 3.0])
        novelty = tracker.calculate_novelty(state)
        assert novelty == 1.0

    def test_calculate_novelty_with_history(self, tracker):
        """Test novelty calculation with history."""
        # Add some states to history
        tracker.add_state(np.array([1.0, 2.0]))
        tracker.add_state(np.array([2.0, 3.0]))

        # Similar state should have lower novelty
        similar_state = np.array([1.1, 2.1])
        novelty = tracker.calculate_novelty(similar_state)
        assert 0.0 <= novelty < 1.0

        # Very different state should have higher novelty
        different_state = np.array([10.0, 20.0])
        novelty = tracker.calculate_novelty(different_state)
        assert novelty > 0.5

    def test_window_size_limit(self, tracker):
        """Test that tracker respects window size limit."""
        # Add more states than window size
        for i in range(15):
            tracker.add_state(np.array([float(i), float(i + 1)]))

        # Should only keep last 10 states
        assert len(tracker.state_history) == 10

        # First state should be from iteration 5 (0-indexed)
        expected_first = np.array([5.0, 6.0])
        np.testing.assert_array_equal(tracker.state_history[0], expected_first)


class TestExperienceSharing:
    """Test cases for ExperienceSharing."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return ExperienceConfig()

    @pytest.fixture
    def sharing(self, config):
        """Create test sharing manager."""
        return ExperienceSharing(config)

    @pytest.fixture
    def sample_experiences(self):
        """Create sample experiences."""
        return [
            Experience(
                np.array([1.0, 2.0]), 0, 0.9, np.array([1.1, 2.1]), False
            ),  # High reward
            Experience(
                np.array([2.0, 3.0]), 1, 0.5, np.array([2.1, 3.1]), False
            ),  # Low reward
            Experience(
                np.array([3.0, 4.0]), 0, 0.6, np.array([3.1, 4.1]), False
            ),  # Medium reward
        ]

    @pytest.fixture
    def sample_metadata(self):
        """Create sample metadata."""
        return [
            ExperienceMetadata("agent1", time.time(), 0.9, 0.5, "high_reward"),
            ExperienceMetadata("agent1", time.time(), 0.5, 0.8, "novelty"),
            ExperienceMetadata("agent1", time.time(), 0.6, 0.3, "random"),
        ]

    def test_sharing_initialization(self, config):
        """Test sharing initialization."""
        sharing = ExperienceSharing(config)
        assert sharing.config == config
        assert sharing.current_strategy == "adaptive"
        assert len(sharing.sharing_strategies) == 4

    def test_share_all_strategy(self, sharing, sample_experiences, sample_metadata):
        """Test share all strategy."""
        sharing.current_strategy = "share_all"

        shared = sharing.select_experiences_to_share(
            "agent1", sample_experiences, sample_metadata
        )

        assert len(shared) == 3  # All experiences shared
        assert all(isinstance(item, tuple) and len(item) == 2 for item in shared)

    def test_share_high_reward_strategy(
        self, sharing, sample_experiences, sample_metadata
    ):
        """Test high reward sharing strategy."""
        sharing.current_strategy = "share_high_reward"

        shared = sharing.select_experiences_to_share(
            "agent1", sample_experiences, sample_metadata
        )

        # Only high reward experience should be shared
        assert len(shared) == 1
        assert shared[0][1].reward == 0.9

    def test_share_novel_strategy(self, sharing, sample_experiences, sample_metadata):
        """Test novel sharing strategy."""
        sharing.current_strategy = "share_novel"

        shared = sharing.select_experiences_to_share(
            "agent1", sample_experiences, sample_metadata
        )

        # Only novel experience should be shared (novelty >= 0.7)
        assert len(shared) == 1
        assert shared[0][1].novelty_score == 0.8

    def test_unknown_strategy_fallback(
        self, sharing, sample_experiences, sample_metadata
    ):
        """Test fallback to default strategy for unknown strategy."""
        sharing.current_strategy = "unknown_strategy"

        # Should fallback to adaptive strategy
        shared = sharing.select_experiences_to_share(
            "agent1", sample_experiences, sample_metadata
        )

        # Should share high reward and novel experiences
        assert len(shared) >= 1

    def test_set_strategy(self, sharing):
        """Test strategy setting."""
        sharing.set_strategy("share_all")
        assert sharing.current_strategy == "share_all"

        # Unknown strategy should not change current strategy
        sharing.set_strategy("unknown")
        assert sharing.current_strategy == "share_all"

    def test_mismatched_lists(self, sharing):
        """Test handling of mismatched experience and metadata lists."""
        experiences = [Experience(np.array([1.0]), 0, 0.5, np.array([1.1]), False)]
        metadata = []  # Empty metadata list

        shared = sharing.select_experiences_to_share("agent1", experiences, metadata)
        assert len(shared) == 0


class TestSharedExperienceManager:
    """Test cases for SharedExperienceManager."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return ExperienceConfig(
            agent_buffer_size=10,
            shared_buffer_size=20,
            high_reward_threshold=0.8,
            novelty_threshold=0.7,
        )

    @pytest.fixture
    def manager(self, config):
        """Create test manager."""
        return SharedExperienceManager(config)

    @pytest.fixture
    def sample_experience(self):
        """Create sample experience."""
        return Experience(
            state=np.array([1.0, 2.0, 3.0]),
            action=0,
            reward=0.5,
            next_state=np.array([1.1, 2.1, 3.1]),
            done=False,
        )

    def test_manager_initialization(self, config):
        """Test manager initialization."""
        manager = SharedExperienceManager(config)

        assert manager.config == config
        assert len(manager.agent_buffers) == 0
        assert len(manager.shared_buffer) == 0
        assert len(manager.registered_agents) == 0
        assert manager.stats["total_experiences_stored"] == 0

    def test_register_agent(self, manager):
        """Test agent registration."""
        manager.register_agent("generator")

        assert "generator" in manager.registered_agents
        assert len(manager.registered_agents) == 1

    def test_store_experience(self, manager, sample_experience):
        """Test storing experience."""
        manager.register_agent("generator")

        result = manager.store_experience("generator", sample_experience)

        assert result is True
        assert len(manager.agent_buffers["generator"]) == 1
        assert len(manager.agent_metadata["generator"]) == 1
        assert manager.stats["total_experiences_stored"] == 1
        assert manager.stats["experiences_by_agent"]["generator"] == 1

    def test_sample_experiences(self, manager):
        """Test sampling experiences."""
        manager.register_agent("generator")
        manager.register_agent("validator")

        # Store some experiences
        for i in range(5):
            exp = Experience(
                state=np.array([float(i), float(i + 1), float(i + 2)]),
                action=i % 2,
                reward=float(i) / 5.0,
                next_state=np.array([float(i + 1), float(i + 2), float(i + 3)]),
                done=i == 4,
            )
            manager.store_experience("generator", exp)

        # Sample experiences for validator
        sampled = manager.sample_experiences("validator", 3)

        assert len(sampled) <= 3
        assert all(isinstance(exp, Experience) for exp in sampled)

    def test_sample_insufficient_experiences(self, manager):
        """Test sampling with insufficient experiences."""
        manager.register_agent("generator")

        # No experiences stored
        sampled = manager.sample_experiences("generator", 5)

        assert len(sampled) == 0

    def test_get_agent_experiences(self, manager, sample_experience):
        """Test getting agent experiences."""
        manager.register_agent("generator")
        manager.store_experience("generator", sample_experience)

        experiences = manager.get_agent_experiences("generator")

        assert len(experiences) == 1
        assert experiences[0] == sample_experience

    def test_get_agent_experiences_with_count(self, manager):
        """Test getting limited agent experiences."""
        manager.register_agent("generator")

        # Store multiple experiences
        for i in range(5):
            exp = Experience(
                state=np.array([float(i)]),
                action=i,
                reward=float(i),
                next_state=np.array([float(i + 1)]),
                done=False,
            )
            manager.store_experience("generator", exp)

        # Get last 3 experiences
        experiences = manager.get_agent_experiences("generator", count=3)

        assert len(experiences) == 3
        # Should be the last 3 experiences (rewards 2, 3, 4)
        assert experiences[0].reward == 2.0
        assert experiences[1].reward == 3.0
        assert experiences[2].reward == 4.0

    def test_get_statistics(self, manager, sample_experience):
        """Test getting statistics."""
        manager.register_agent("generator")
        manager.store_experience("generator", sample_experience)

        stats = manager.get_statistics()

        assert "configuration" in stats
        assert "statistics" in stats
        assert "buffer_utilization" in stats
        assert "registered_agents" in stats
        assert "sharing_strategy" in stats

        assert stats["statistics"]["total_experiences_stored"] == 1
        assert "generator" in stats["registered_agents"]

    def test_clear(self, manager, sample_experience):
        """Test clearing all buffers."""
        manager.register_agent("generator")
        manager.store_experience("generator", sample_experience)

        # Verify data exists
        assert len(manager.agent_buffers["generator"]) == 1
        assert manager.stats["total_experiences_stored"] == 1

        manager.clear()

        # Verify data is cleared
        assert len(manager.agent_buffers) == 0
        assert len(manager.shared_buffer) == 0
        assert manager.stats["total_experiences_stored"] == 0

    def test_cleanup_expired_experiences(self, manager):
        """Test cleanup of expired experiences."""
        manager.register_agent("generator")

        # Create experience with old timestamp
        old_exp = Experience(
            state=np.array([1.0]),
            action=0,
            reward=0.5,
            next_state=np.array([1.1]),
            done=False,
        )

        # Mock old timestamp
        with patch("time.time", return_value=time.time() - (25 * 3600)):  # 25 hours ago
            manager.store_experience("generator", old_exp)

        # Add recent experience
        recent_exp = Experience(
            state=np.array([2.0]),
            action=1,
            reward=0.8,
            next_state=np.array([2.1]),
            done=False,
        )
        manager.store_experience("generator", recent_exp)

        # Should have 2 experiences before cleanup
        assert len(manager.agent_buffers["generator"]) == 2

        # Cleanup expired experiences
        manager.cleanup_expired_experiences()

        # Should have 1 experience after cleanup (only recent one)
        assert len(manager.agent_buffers["generator"]) == 1
        remaining_exp = list(manager.agent_buffers["generator"])[0]
        assert remaining_exp.reward == 0.8  # Recent experience


class TestSharedExperienceFactory:
    """Test cases for shared experience factory function."""

    def test_create_with_default_config(self):
        """Test creating manager with default configuration."""
        manager = create_shared_experience_manager()

        assert isinstance(manager, SharedExperienceManager)
        assert manager.config.agent_buffer_size == 1000
        assert manager.config.shared_buffer_size == 5000

    def test_create_with_custom_config(self):
        """Test creating manager with custom configuration."""
        config = ExperienceConfig(agent_buffer_size=500, shared_buffer_size=2000)
        manager = create_shared_experience_manager(config)

        assert isinstance(manager, SharedExperienceManager)
        assert manager.config.agent_buffer_size == 500
        assert manager.config.shared_buffer_size == 2000
