"""
Unit tests for Continuous Learning System
"""

# Standard Library
import asyncio
from unittest.mock import AsyncMock, MagicMock

# Third-Party Library
import numpy as np
import pytest

# SynThesisAI Modules
from core.marl.agents.base_agent import BaseRLAgent, Experience
from core.marl.learning.continuous_learning import (
    AdaptiveLearningRate,
    ContinuousLearningManager,
    LearningConfig,
    PerformanceTracker,
    create_continuous_learning_manager,
)
from core.marl.learning.shared_experience import SharedExperienceManager


class TestLearningConfig:
    """Test cases for LearningConfig."""

    def test_default_initialization(self):
        """Test default configuration initialization."""
        config = LearningConfig()

        assert config.learning_interval == 60.0
        assert config.batch_size == 32
        assert config.min_experiences_for_learning == 100
        assert config.learning_rate_decay == 0.995
        assert config.min_learning_rate == 1e-5
        assert config.performance_window_size == 100
        assert config.adaptation_threshold == 0.1
        assert config.max_learning_iterations == 10
        assert config.enable_shared_learning is True
        assert config.shared_experience_ratio == 0.3

    def test_custom_initialization(self):
        """Test custom configuration initialization."""
        config = LearningConfig(
            learning_interval=30.0,
            batch_size=64,
            min_experiences_for_learning=50,
            learning_rate_decay=0.99,
            min_learning_rate=1e-6,
            performance_window_size=50,
            adaptation_threshold=0.2,
            max_learning_iterations=5,
            enable_shared_learning=False,
            shared_experience_ratio=0.5,
        )

        assert config.learning_interval == 30.0
        assert config.batch_size == 64
        assert config.min_experiences_for_learning == 50
        assert config.learning_rate_decay == 0.99
        assert config.min_learning_rate == 1e-6
        assert config.performance_window_size == 50
        assert config.adaptation_threshold == 0.2
        assert config.max_learning_iterations == 5
        assert config.enable_shared_learning is False
        assert config.shared_experience_ratio == 0.5


class TestPerformanceTracker:
    """Test cases for PerformanceTracker."""

    @pytest.fixture
    def tracker(self):
        """Create test tracker."""
        return PerformanceTracker(window_size=10)

    def test_initialization(self):
        """Test tracker initialization."""
        tracker = PerformanceTracker(window_size=5)
        assert tracker.window_size == 5
        assert len(tracker.performance_history) == 0
        assert len(tracker.reward_history) == 0
        assert len(tracker.loss_history) == 0

    def test_record_performance(self, tracker):
        """Test recording performance metrics."""
        tracker.record_performance(0.8, 0.1, {"accuracy": 0.9})

        assert len(tracker.performance_history) == 1
        assert len(tracker.reward_history) == 1
        assert len(tracker.loss_history) == 1

        entry = tracker.performance_history[0]
        assert entry["reward"] == 0.8
        assert entry["loss"] == 0.1
        assert entry["metrics"]["accuracy"] == 0.9

    def test_record_performance_without_loss(self, tracker):
        """Test recording performance without loss."""
        tracker.record_performance(0.7)

        assert len(tracker.performance_history) == 1
        assert len(tracker.reward_history) == 1
        assert len(tracker.loss_history) == 0

    def test_window_size_limit(self, tracker):
        """Test that tracker respects window size limit."""
        # Add more entries than window size
        for i in range(15):
            tracker.record_performance(float(i), float(i) * 0.1)

        # Should only keep last 10 entries
        assert len(tracker.performance_history) == 10
        assert len(tracker.reward_history) == 10
        assert len(tracker.loss_history) == 10

        # First entry should be from iteration 5
        assert tracker.reward_history[0] == 5.0

    def test_get_recent_performance_empty(self, tracker):
        """Test getting performance with empty history."""
        stats = tracker.get_recent_performance()

        assert stats["avg_reward"] == 0.0
        assert stats["reward_trend"] == 0.0
        assert stats["avg_loss"] == 0.0
        assert stats["loss_trend"] == 0.0
        assert stats["sample_count"] == 0

    def test_get_recent_performance_with_data(self, tracker):
        """Test getting performance with data."""
        # Add some performance data
        rewards = [0.5, 0.6, 0.7, 0.8, 0.9]
        losses = [0.5, 0.4, 0.3, 0.2, 0.1]

        for r, loss in zip(rewards, losses):
            tracker.record_performance(r, loss)

        stats = tracker.get_recent_performance()

        assert stats["avg_reward"] == 0.7  # Average of rewards
        assert stats["reward_trend"] > 0  # Positive trend
        assert stats["avg_loss"] == 0.3  # Average of losses
        assert stats["loss_trend"] < 0  # Negative trend (improving)
        assert stats["sample_count"] == 5

    def test_should_adapt_insufficient_data(self, tracker):
        """Test adaptation decision with insufficient data."""
        # Add only a few data points
        for i in range(5):
            tracker.record_performance(0.5)

        assert tracker.should_adapt(0.1) is False

    def test_should_adapt_declining_performance(self, tracker):
        """Test adaptation decision with declining performance."""
        # Add declining rewards
        rewards = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
        for r in rewards:
            tracker.record_performance(r)

        assert tracker.should_adapt(0.05) is True

    def test_should_adapt_increasing_loss(self, tracker):
        """Test adaptation decision with increasing loss."""
        # Add increasing losses
        losses = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        for loss in losses:
            tracker.record_performance(0.5, loss)

        assert tracker.should_adapt(0.05) is True

    def test_should_not_adapt_stable_performance(self, tracker):
        """Test adaptation decision with stable performance."""
        # Add stable performance
        for _ in range(15):
            tracker.record_performance(0.7, 0.3)

        assert tracker.should_adapt(0.1) is False

    def test_reset(self, tracker):
        """Test tracker reset."""
        # Add some data
        tracker.record_performance(0.8, 0.2)
        assert len(tracker.performance_history) == 1

        tracker.reset()

        assert len(tracker.performance_history) == 0
        assert len(tracker.reward_history) == 0
        assert len(tracker.loss_history) == 0


class TestAdaptiveLearningRate:
    """Test cases for AdaptiveLearningRate."""

    @pytest.fixture
    def lr_manager(self):
        """Create test learning rate manager."""
        return AdaptiveLearningRate(
            initial_rate=0.001, decay_factor=0.9, min_rate=1e-5, adaptation_factor=1.1
        )

    def test_initialization(self):
        """Test learning rate manager initialization."""
        lr_manager = AdaptiveLearningRate(
            initial_rate=0.002, decay_factor=0.95, min_rate=1e-6, adaptation_factor=1.2
        )

        assert lr_manager.initial_rate == 0.002
        assert lr_manager.current_rate == 0.002
        assert lr_manager.decay_factor == 0.95
        assert lr_manager.min_rate == 1e-6
        assert lr_manager.adaptation_factor == 1.2

    def test_update_rate_improving(self, lr_manager):
        """Test learning rate update when performance is improving."""
        initial_rate = lr_manager.current_rate

        lr_manager.update_rate(performance_improving=True)

        # Rate should decay
        assert lr_manager.current_rate == initial_rate * 0.9

    def test_update_rate_not_improving(self, lr_manager):
        """Test learning rate update when performance is not improving."""
        initial_rate = lr_manager.current_rate

        lr_manager.update_rate(performance_improving=False)

        # Rate should increase
        assert lr_manager.current_rate == initial_rate * 1.1

    def test_min_rate_constraint(self, lr_manager):
        """Test that learning rate respects minimum constraint."""
        # Decay rate many times to reach minimum
        for _ in range(100):
            lr_manager.update_rate(performance_improving=True)

        assert lr_manager.current_rate == lr_manager.min_rate

    def test_get_rate(self, lr_manager):
        """Test getting current learning rate."""
        assert lr_manager.get_rate() == 0.001

        lr_manager.update_rate(performance_improving=False)
        assert lr_manager.get_rate() == 0.0011

    def test_reset(self, lr_manager):
        """Test learning rate reset."""
        # Change the rate
        lr_manager.update_rate(performance_improving=False)
        assert lr_manager.current_rate != lr_manager.initial_rate

        lr_manager.reset()
        assert lr_manager.current_rate == lr_manager.initial_rate


class TestContinuousLearningManager:
    """Test cases for ContinuousLearningManager."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return LearningConfig(
            learning_interval=1.0,  # Short interval for testing
            batch_size=16,
            min_experiences_for_learning=10,
            max_learning_iterations=2,
        )

    @pytest.fixture
    def shared_manager(self):
        """Create mock shared experience manager."""
        return MagicMock(spec=SharedExperienceManager)

    @pytest.fixture
    def learning_manager(self, config, shared_manager):
        """Create test learning manager."""
        return ContinuousLearningManager(config, shared_manager)

    @pytest.fixture
    def mock_agent(self):
        """Create mock RL agent."""
        agent = MagicMock(spec=BaseRLAgent)
        agent.config = MagicMock()
        agent.config.learning_rate = 0.001
        agent.replay_buffer = MagicMock()
        agent.replay_buffer.__len__ = MagicMock(return_value=50)
        # Return enough experiences to ensure at least one iteration (batch_size=32)
        agent.replay_buffer.sample = MagicMock(
            return_value=[
                Experience(np.array([1.0]), 0, 0.5, np.array([1.1]), False)
                for _ in range(35)  # Increased from 10 to 35 to ensure iterations > 0
            ]
        )
        agent._update_policy_from_batch = MagicMock(return_value=0.1)
        return agent

    def test_initialization(self, config, shared_manager):
        """Test learning manager initialization."""
        manager = ContinuousLearningManager(config, shared_manager)

        assert manager.config == config
        assert manager.shared_experience_manager == shared_manager
        assert len(manager.agents) == 0
        assert len(manager.performance_trackers) == 0
        assert len(manager.learning_rate_managers) == 0
        assert manager.learning_active is False
        assert manager.learning_stats["total_learning_updates"] == 0

    def test_register_agent(self, learning_manager, mock_agent):
        """Test agent registration."""
        learning_manager.register_agent("test_agent", mock_agent)

        assert "test_agent" in learning_manager.agents
        assert "test_agent" in learning_manager.performance_trackers
        assert "test_agent" in learning_manager.learning_rate_managers
        assert learning_manager.agents["test_agent"] == mock_agent

        # Verify shared experience manager registration
        learning_manager.shared_experience_manager.register_agent.assert_called_once_with(
            "test_agent"
        )

    @pytest.mark.asyncio
    async def test_start_stop_continuous_learning(self, learning_manager):
        """Test starting and stopping continuous learning."""
        # Start learning
        learning_manager.start_continuous_learning()
        assert learning_manager.learning_active is True
        assert learning_manager.learning_task is not None

        # Wait a short time
        await asyncio.sleep(0.1)

        # Stop learning
        await learning_manager.stop_continuous_learning()
        assert learning_manager.learning_active is False

    @pytest.mark.asyncio
    async def test_start_already_active(self, learning_manager):
        """Test starting learning when already active."""
        learning_manager.learning_active = True

        # Should not create new task
        learning_manager.start_continuous_learning()
        assert learning_manager.learning_task is None

    def test_has_sufficient_experiences(self, learning_manager, mock_agent):
        """Test checking if agent has sufficient experiences."""
        # Mock agent with enough experiences
        mock_agent.replay_buffer.__len__.return_value = 50
        assert learning_manager._has_sufficient_experiences(mock_agent) is True

        # Mock agent with insufficient experiences
        mock_agent.replay_buffer.__len__.return_value = 5
        assert learning_manager._has_sufficient_experiences(mock_agent) is False

    @pytest.mark.asyncio
    async def test_update_agent_policy_success(self, learning_manager, mock_agent):
        """Test successful agent policy update."""
        learning_manager.register_agent("test_agent", mock_agent)

        # Mock shared experience manager
        learning_manager.shared_experience_manager.sample_experiences.return_value = [
            Experience(np.array([2.0]), 1, 0.7, np.array([2.1]), False)
        ]

        result = await learning_manager._update_agent_policy("test_agent", mock_agent)

        assert result is True
        assert mock_agent._update_policy_from_batch.called

        # Verify performance was recorded
        tracker = learning_manager.performance_trackers["test_agent"]
        assert len(tracker.performance_history) > 0

    @pytest.mark.asyncio
    async def test_update_agent_policy_failure(self, learning_manager, mock_agent):
        """Test agent policy update failure."""
        learning_manager.register_agent("test_agent", mock_agent)

        # Mock failure
        mock_agent._update_policy_from_batch.side_effect = Exception("Update failed")

        result = await learning_manager._update_agent_policy("test_agent", mock_agent)

        assert result is False

    def test_prepare_training_batch_own_only(self, learning_manager, mock_agent):
        """Test preparing training batch with own experiences only."""
        learning_manager.config.enable_shared_learning = False

        experiences = learning_manager._prepare_training_batch("test_agent", mock_agent)

        assert (
            len(experiences) == 35
        )  # From mock agent (updated to ensure iterations > 0)
        mock_agent.replay_buffer.sample.assert_called_once()

    def test_prepare_training_batch_with_shared(self, learning_manager, mock_agent):
        """Test preparing training batch with shared experiences."""
        learning_manager.config.enable_shared_learning = True
        learning_manager.config.shared_experience_ratio = 0.5

        # Mock shared experiences
        shared_exp = [Experience(np.array([3.0]), 2, 0.8, np.array([3.1]), False)]
        learning_manager.shared_experience_manager.sample_experiences.return_value = (
            shared_exp
        )

        experiences = learning_manager._prepare_training_batch("test_agent", mock_agent)

        assert len(experiences) == 36  # 35 own + 1 shared
        learning_manager.shared_experience_manager.sample_experiences.assert_called_once()

    def test_record_agent_performance(self, learning_manager, mock_agent):
        """Test recording agent performance."""
        learning_manager.register_agent("test_agent", mock_agent)

        learning_manager.record_agent_performance("test_agent", 0.8, {"accuracy": 0.9})

        tracker = learning_manager.performance_trackers["test_agent"]
        assert len(tracker.performance_history) == 1
        assert tracker.performance_history[0]["reward"] == 0.8

    def test_record_agent_performance_unknown_agent(self, learning_manager):
        """Test recording performance for unknown agent."""
        # Should not raise exception
        learning_manager.record_agent_performance("unknown_agent", 0.5)

    def test_get_learning_progress(self, learning_manager, mock_agent):
        """Test getting learning progress."""
        learning_manager.register_agent("test_agent", mock_agent)
        learning_manager.record_agent_performance("test_agent", 0.8)

        progress = learning_manager.get_learning_progress()

        assert "learning_active" in progress
        assert "learning_stats" in progress
        assert "agent_progress" in progress
        assert "configuration" in progress

        assert "test_agent" in progress["agent_progress"]
        agent_progress = progress["agent_progress"]["test_agent"]
        assert "performance" in agent_progress
        assert "learning_rate" in agent_progress
        assert "should_adapt" in agent_progress

    def test_reset_learning_progress(self, learning_manager, mock_agent):
        """Test resetting learning progress."""
        learning_manager.register_agent("test_agent", mock_agent)
        learning_manager.record_agent_performance("test_agent", 0.8)
        learning_manager.learning_stats["total_learning_updates"] = 5

        # Verify data exists
        tracker = learning_manager.performance_trackers["test_agent"]
        assert len(tracker.performance_history) == 1
        assert learning_manager.learning_stats["total_learning_updates"] == 5

        learning_manager.reset_learning_progress()

        # Verify data is reset
        assert len(tracker.performance_history) == 0
        assert learning_manager.learning_stats["total_learning_updates"] == 0

    @pytest.mark.asyncio
    async def test_shutdown(self, learning_manager):
        """Test learning manager shutdown."""
        learning_manager.start_continuous_learning()
        assert learning_manager.learning_active is True

        await learning_manager.shutdown()
        assert learning_manager.learning_active is False

    @pytest.mark.asyncio
    async def test_perform_learning_update(self, learning_manager, mock_agent):
        """Test performing learning update."""
        learning_manager.register_agent("test_agent", mock_agent)

        # Mock successful update
        learning_manager._update_agent_policy = AsyncMock(return_value=True)

        await learning_manager._perform_learning_update()

        assert learning_manager.learning_stats["total_learning_updates"] == 1
        assert learning_manager.learning_stats["successful_updates"] == 1

    @pytest.mark.asyncio
    async def test_perform_learning_update_insufficient_experiences(
        self, learning_manager, mock_agent
    ):
        """Test learning update with insufficient experiences."""
        learning_manager.register_agent("test_agent", mock_agent)

        # Mock insufficient experiences
        mock_agent.replay_buffer.__len__.return_value = 5

        await learning_manager._perform_learning_update()

        # Should not attempt update
        assert learning_manager.learning_stats["total_learning_updates"] == 1
        assert learning_manager.learning_stats["successful_updates"] == 0

    @pytest.mark.asyncio
    async def test_learning_loop_integration(self, learning_manager, mock_agent):
        """Test integration of learning loop."""
        learning_manager.register_agent("test_agent", mock_agent)
        learning_manager.config.learning_interval = 0.1  # Very short for testing

        # Mock successful updates
        learning_manager._update_agent_policy = AsyncMock(return_value=True)

        # Start learning and let it run briefly
        learning_manager.start_continuous_learning()
        await asyncio.sleep(0.2)  # Let it run for a bit
        await learning_manager.stop_continuous_learning()

        # Should have performed at least one update
        assert learning_manager.learning_stats["total_learning_updates"] >= 1


class TestContinuousLearningFactory:
    """Test cases for continuous learning factory function."""

    def test_create_with_default_config(self):
        """Test creating manager with default configuration."""
        manager = create_continuous_learning_manager()

        assert isinstance(manager, ContinuousLearningManager)
        assert manager.config.learning_interval == 60.0
        assert manager.shared_experience_manager is None

    def test_create_with_custom_config(self):
        """Test creating manager with custom configuration."""
        config = LearningConfig(learning_interval=30.0, batch_size=64)
        shared_manager = MagicMock(spec=SharedExperienceManager)

        manager = create_continuous_learning_manager(config, shared_manager)

        assert isinstance(manager, ContinuousLearningManager)
        assert manager.config.learning_interval == 30.0
        assert manager.config.batch_size == 64
        assert manager.shared_experience_manager == shared_manager


class TestContinuousLearningIntegration:
    """Integration tests for continuous learning system."""

    @pytest.fixture
    def integration_config(self):
        """Create integration test configuration."""
        return LearningConfig(
            learning_interval=0.5,  # Short for testing
            batch_size=8,
            min_experiences_for_learning=5,
            max_learning_iterations=1,
            enable_shared_learning=True,
            shared_experience_ratio=0.2,
        )

    @pytest.mark.asyncio
    async def test_end_to_end_continuous_learning(self, integration_config):
        """Test complete end-to-end continuous learning workflow."""
        # Create shared experience manager
        shared_manager = MagicMock(spec=SharedExperienceManager)
        shared_manager.sample_experiences.return_value = [
            Experience(np.array([5.0]), 2, 0.9, np.array([5.1]), False)
        ]

        # Create learning manager
        learning_manager = ContinuousLearningManager(integration_config, shared_manager)

        # Create mock agents
        agents = {}
        for agent_id in ["generator", "validator", "curriculum"]:
            agent = MagicMock(spec=BaseRLAgent)
            agent.config = MagicMock()
            agent.config.learning_rate = 0.001
            agent.replay_buffer = MagicMock()
            agent.replay_buffer.__len__ = MagicMock(return_value=20)
            agent.replay_buffer.sample = MagicMock(
                return_value=[
                    Experience(
                        np.array([float(i)]),
                        i % 2,
                        0.5 + i * 0.1,
                        np.array([float(i + 1)]),
                        False,
                    )
                    for i in range(8)
                ]
            )
            agent._update_policy_from_batch = MagicMock(return_value=0.05)
            agents[agent_id] = agent

            # Register agent
            learning_manager.register_agent(agent_id, agent)

        # Record some initial performance
        for agent_id in agents:
            learning_manager.record_agent_performance(agent_id, 0.7)

        # Start continuous learning
        learning_manager.start_continuous_learning()

        # Let it run for a short time
        await asyncio.sleep(1.2)  # Should trigger at least 2 learning updates

        # Stop learning
        await learning_manager.stop_continuous_learning()

        # Verify learning occurred
        progress = learning_manager.get_learning_progress()
        assert progress["learning_stats"]["total_learning_updates"] >= 2
        assert progress["learning_stats"]["successful_updates"] > 0

        # Verify all agents were updated
        for agent in agents.values():
            assert agent._update_policy_from_batch.called

        # Verify shared experiences were used
        assert progress["learning_stats"]["shared_experiences_used"] > 0

        # Verify performance tracking
        for agent_id in agents:
            agent_progress = progress["agent_progress"][agent_id]
            assert agent_progress["performance"]["sample_count"] > 0
