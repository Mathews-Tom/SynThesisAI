"""
Unit tests for base RL agent framework.

This module tests the base RL agent architecture including neural networks,
replay buffers, learning metrics, and the abstract base agent class.
"""

import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from core.marl.agents.base_agent import ActionSpace, BaseRLAgent
from core.marl.agents.experience import Experience
from core.marl.agents.learning_metrics import LearningMetrics
from core.marl.agents.neural_networks import (
    DuelingQNetwork,
    QNetwork,
    build_q_network,
    build_target_network,
    get_network_summary,
)
from core.marl.agents.replay_buffer import PrioritizedReplayBuffer, ReplayBuffer
from core.marl.config import AgentConfig
from core.marl.exceptions import (
    AgentFailureError,
    ExperienceBufferError,
    PolicyNetworkError,
)


# Test implementation of BaseRLAgent for testing
class MockRLAgent(BaseRLAgent):
    """Test implementation of BaseRLAgent."""

    def get_state_representation(self, environment_state: Dict[str, Any]) -> np.ndarray:
        """Simple state representation for testing."""
        return np.array(
            [
                environment_state.get("feature1", 0.0),
                environment_state.get("feature2", 0.0),
                environment_state.get("feature3", 0.0),
            ],
            dtype=np.float32,
        )

    def get_action_space(self) -> ActionSpace:
        """Simple action space for testing."""
        return ActionSpace(["action1", "action2", "action3", "action4"])

    def calculate_reward(
        self, state: np.ndarray, action: int, result: Dict[str, Any]
    ) -> float:
        """Simple reward calculation for testing."""
        base_reward = result.get("base_reward", 0.0)
        quality_bonus = result.get("quality_score", 0.0) * 0.5
        return base_reward + quality_bonus


class TestActionSpace:
    """Test ActionSpace class."""

    def test_action_space_creation(self):
        """Test action space creation and basic operations."""
        actions = ["move_left", "move_right", "jump", "shoot"]
        action_space = ActionSpace(actions)

        assert len(action_space) == 4
        assert action_space[0] == "move_left"
        assert action_space[3] == "shoot"

        # Test index lookup
        assert action_space.get_action_index("jump") == 2

        # Test sampling
        sampled_action = action_space.sample()
        assert 0 <= sampled_action < len(action_space)

    def test_action_space_errors(self):
        """Test action space error handling."""
        action_space = ActionSpace(["a", "b", "c"])

        # Test index out of range
        with pytest.raises(IndexError):
            _ = action_space[5]

        # Test action not found
        with pytest.raises(ValueError, match="Action 'invalid' not found"):
            action_space.get_action_index("invalid")

    def test_empty_action_space(self):
        """Test empty action space."""
        action_space = ActionSpace([])

        assert len(action_space) == 0

        # Sampling from empty space should still work (returns random int)
        with pytest.raises(ValueError):  # numpy raises ValueError for empty range
            action_space.sample()


class TestExperience:
    """Test Experience class."""

    def test_experience_creation(self):
        """Test experience creation and serialization."""
        state = np.array([1.0, 2.0, 3.0])
        next_state = np.array([1.1, 2.1, 3.1])
        metadata = {"episode": 5, "step": 10}

        exp = Experience(
            state=state,
            action=1,
            reward=0.5,
            next_state=next_state,
            done=False,
            metadata=metadata,
        )

        assert np.array_equal(exp.state, state)
        assert exp.action == 1
        assert exp.reward == 0.5
        assert np.array_equal(exp.next_state, next_state)
        assert exp.done is False
        assert exp.metadata == metadata
        assert isinstance(exp.timestamp, float)

    def test_experience_serialization(self):
        """Test experience to/from dictionary conversion."""
        state = np.array([1.0, 2.0])
        next_state = np.array([1.5, 2.5])

        exp = Experience(
            state=state, action=2, reward=1.0, next_state=next_state, done=True
        )

        # Convert to dict
        exp_dict = exp.to_dict()

        assert exp_dict["state"] == [1.0, 2.0]
        assert exp_dict["action"] == 2
        assert exp_dict["reward"] == 1.0
        assert exp_dict["next_state"] == [1.5, 2.5]
        assert exp_dict["done"] is True

        # Convert back from dict
        exp_restored = Experience.from_dict(exp_dict)

        assert np.array_equal(exp_restored.state, state)
        assert exp_restored.action == 2
        assert exp_restored.reward == 1.0
        assert np.array_equal(exp_restored.next_state, next_state)
        assert exp_restored.done is True


class TestReplayBuffer:
    """Test ReplayBuffer class."""

    def test_replay_buffer_creation(self):
        """Test replay buffer creation."""
        buffer = ReplayBuffer(capacity=100)

        assert len(buffer) == 0
        assert buffer.capacity == 100
        assert not buffer.is_full()

    def test_replay_buffer_invalid_capacity(self):
        """Test replay buffer with invalid capacity."""
        with pytest.raises(
            ExperienceBufferError, match="Buffer capacity must be positive"
        ):
            ReplayBuffer(capacity=0)

        with pytest.raises(
            ExperienceBufferError, match="Buffer capacity must be positive"
        ):
            ReplayBuffer(capacity=-10)

    def test_replay_buffer_add_and_sample(self):
        """Test adding experiences and sampling."""
        buffer = ReplayBuffer(capacity=10)

        # Add experiences
        for i in range(5):
            exp = Experience(
                state=np.array([i]),
                action=i % 2,
                reward=float(i),
                next_state=np.array([i + 1]),
                done=i == 4,
            )
            buffer.add(exp)

        assert len(buffer) == 5

        # Sample experiences
        sampled = buffer.sample(3)
        assert len(sampled) == 3
        assert all(isinstance(exp, Experience) for exp in sampled)

    def test_replay_buffer_overflow(self):
        """Test replay buffer overflow behavior."""
        buffer = ReplayBuffer(capacity=3)

        # Add more experiences than capacity
        for i in range(5):
            exp = Experience(
                state=np.array([i]),
                action=0,
                reward=float(i),
                next_state=np.array([i + 1]),
                done=False,
            )
            buffer.add(exp)

        # Should only keep the last 3 experiences
        assert len(buffer) == 3
        assert buffer.is_full()

    def test_replay_buffer_sample_insufficient(self):
        """Test sampling when insufficient experiences."""
        buffer = ReplayBuffer(capacity=10)

        # Add only 2 experiences
        for i in range(2):
            exp = Experience(
                state=np.array([i]),
                action=0,
                reward=0.0,
                next_state=np.array([i + 1]),
                done=False,
            )
            buffer.add(exp)

        # Try to sample more than available
        with pytest.raises(ExperienceBufferError, match="Not enough experiences"):
            buffer.sample(5)

    def test_replay_buffer_statistics(self):
        """Test replay buffer statistics."""
        buffer = ReplayBuffer(capacity=10)

        # Empty buffer statistics
        stats = buffer.get_statistics()
        assert stats["size"] == 0
        assert stats["utilization"] == 0.0

        # Add experiences with different rewards
        rewards = [1.0, 2.0, 3.0, 4.0, 5.0]
        for i, reward in enumerate(rewards):
            exp = Experience(
                state=np.array([i]),
                action=0,
                reward=reward,
                next_state=np.array([i + 1]),
                done=False,
            )
            buffer.add(exp)

        stats = buffer.get_statistics()
        assert stats["size"] == 5
        assert stats["utilization"] == 0.5
        assert stats["avg_reward"] == 3.0
        assert stats["min_reward"] == 1.0
        assert stats["max_reward"] == 5.0

    def test_replay_buffer_clear(self):
        """Test replay buffer clearing."""
        buffer = ReplayBuffer(capacity=10)

        # Add some experiences
        for i in range(3):
            exp = Experience(
                state=np.array([i]),
                action=0,
                reward=0.0,
                next_state=np.array([i + 1]),
                done=False,
            )
            buffer.add(exp)

        assert len(buffer) == 3

        # Clear buffer
        buffer.clear()

        assert len(buffer) == 0
        assert not buffer.is_full()


class TestPrioritizedReplayBuffer:
    """Test PrioritizedReplayBuffer class."""

    def test_prioritized_buffer_creation(self):
        """Test prioritized replay buffer creation."""
        buffer = PrioritizedReplayBuffer(capacity=100, alpha=0.6, beta=0.4)

        assert len(buffer) == 0
        assert buffer.capacity == 100
        assert buffer.alpha == 0.6
        assert buffer.beta == 0.4
        assert not buffer.is_full()

    def test_prioritized_buffer_invalid_params(self):
        """Test prioritized buffer with invalid parameters."""
        with pytest.raises(
            ExperienceBufferError, match="Buffer capacity must be positive"
        ):
            PrioritizedReplayBuffer(capacity=0)

        with pytest.raises(
            ExperienceBufferError, match="Alpha must be between 0 and 1"
        ):
            PrioritizedReplayBuffer(capacity=100, alpha=1.5)

        with pytest.raises(ExperienceBufferError, match="Beta must be between 0 and 1"):
            PrioritizedReplayBuffer(capacity=100, beta=-0.1)

    def test_prioritized_buffer_add_and_sample(self):
        """Test adding experiences with priorities and sampling."""
        buffer = PrioritizedReplayBuffer(capacity=10, alpha=0.6, beta=0.4)

        # Add experiences with different TD errors
        td_errors = [0.1, 0.5, 0.2, 0.8, 0.3]
        for i, td_error in enumerate(td_errors):
            exp = Experience(
                state=np.array([i]),
                action=i % 2,
                reward=float(i),
                next_state=np.array([i + 1]),
                done=False,
            )
            buffer.add(exp, td_error=td_error)

        assert len(buffer) == 5

        # Sample experiences
        experiences, indices, weights = buffer.sample(3)

        assert len(experiences) == 3
        assert len(indices) == 3
        assert len(weights) == 3
        assert all(isinstance(exp, Experience) for exp in experiences)
        assert all(0 <= idx < len(buffer) for idx in indices)
        assert all(0 <= w <= 1 for w in weights)

    def test_prioritized_buffer_update_priorities(self):
        """Test updating priorities in prioritized buffer."""
        buffer = PrioritizedReplayBuffer(capacity=10)

        # Add experiences
        for i in range(5):
            exp = Experience(
                state=np.array([i]),
                action=0,
                reward=0.0,
                next_state=np.array([i + 1]),
                done=False,
            )
            buffer.add(exp)

        # Sample and update priorities
        experiences, indices, weights = buffer.sample(3)
        new_td_errors = np.array([0.1, 0.5, 0.2])

        buffer.update_priorities(indices, new_td_errors)

        # Should not raise any errors
        assert len(buffer) == 5

    def test_prioritized_buffer_statistics(self):
        """Test prioritized buffer statistics."""
        buffer = PrioritizedReplayBuffer(capacity=10)

        # Add experiences
        for i in range(3):
            exp = Experience(
                state=np.array([i]),
                action=0,
                reward=float(i),
                next_state=np.array([i + 1]),
                done=False,
            )
            buffer.add(exp, td_error=0.1 * (i + 1))

        stats = buffer.get_statistics()

        assert stats["size"] == 3
        assert stats["utilization"] == 0.3
        assert "avg_priority" in stats
        assert "max_priority" in stats
        assert "priority_std" in stats


class TestQNetwork:
    """Test QNetwork class."""

    def test_q_network_creation(self):
        """Test Q-network creation."""
        network = QNetwork(
            state_size=4, action_size=3, hidden_layers=[64, 32], activation="relu"
        )

        assert network.state_size == 4
        assert network.action_size == 3
        assert network.hidden_layers == [64, 32]
        assert network.activation_name == "relu"

        # Test network info
        info = network.get_network_info()
        assert info["state_size"] == 4
        assert info["action_size"] == 3
        assert info["total_parameters"] > 0

    def test_q_network_invalid_params(self):
        """Test Q-network with invalid parameters."""
        with pytest.raises(PolicyNetworkError, match="State size must be positive"):
            QNetwork(state_size=0, action_size=3, hidden_layers=[64])

        with pytest.raises(PolicyNetworkError, match="Action size must be positive"):
            QNetwork(state_size=4, action_size=0, hidden_layers=[64])

        with pytest.raises(
            PolicyNetworkError, match="Hidden layers list cannot be empty"
        ):
            QNetwork(state_size=4, action_size=3, hidden_layers=[])

        with pytest.raises(PolicyNetworkError, match="Unknown activation function"):
            QNetwork(
                state_size=4, action_size=3, hidden_layers=[64], activation="invalid"
            )

    def test_q_network_forward_pass(self):
        """Test Q-network forward pass."""
        network = QNetwork(state_size=3, action_size=2, hidden_layers=[16, 8])

        # Test forward pass
        state = torch.FloatTensor([[1.0, 2.0, 3.0]])
        q_values = network(state)

        assert q_values.shape == (1, 2)
        assert not torch.isnan(q_values).any()

    def test_dueling_q_network(self):
        """Test DuelingQNetwork."""
        network = DuelingQNetwork(
            state_size=4, action_size=3, hidden_layers=[32, 16, 8], activation="relu"
        )

        # Test forward pass
        state = torch.FloatTensor([[1.0, 2.0, 3.0, 4.0]])
        q_values = network(state)

        assert q_values.shape == (1, 3)
        assert not torch.isnan(q_values).any()

    def test_dueling_network_insufficient_layers(self):
        """Test dueling network with insufficient layers."""
        with pytest.raises(
            PolicyNetworkError,
            match="Dueling network requires at least 2 hidden layers",
        ):
            DuelingQNetwork(
                state_size=4,
                action_size=3,
                hidden_layers=[32],  # Only 1 layer
                activation="relu",
            )

    def test_build_q_network_factory(self):
        """Test Q-network factory function."""
        # Standard network
        network = build_q_network(
            state_size=4,
            action_size=3,
            hidden_layers=[32, 16],
            activation="relu",
            network_type="standard",
        )

        assert isinstance(network, QNetwork)

        # Dueling network
        dueling_network = build_q_network(
            state_size=4,
            action_size=3,
            hidden_layers=[32, 16, 8],
            activation="relu",
            network_type="dueling",
        )

        assert isinstance(dueling_network, DuelingQNetwork)

    def test_build_target_network(self):
        """Test target network building."""
        q_network = QNetwork(state_size=3, action_size=2, hidden_layers=[16, 8])

        target_network = build_target_network(q_network)

        assert isinstance(target_network, QNetwork)
        assert target_network.state_size == q_network.state_size
        assert target_network.action_size == q_network.action_size

        # Check that weights are copied
        for target_param, q_param in zip(
            target_network.parameters(), q_network.parameters()
        ):
            assert torch.equal(target_param, q_param)

        # Check that target network parameters don't require gradients
        for param in target_network.parameters():
            assert not param.requires_grad

    def test_network_summary(self):
        """Test network summary generation."""
        network = QNetwork(state_size=4, action_size=3, hidden_layers=[32, 16])

        summary = get_network_summary(network)

        assert "Network Summary:" in summary
        assert "QNetwork" in summary
        assert "Total Parameters:" in summary
        assert "State Size: 4" in summary
        assert "Action Size: 3" in summary


class TestLearningMetrics:
    """Test LearningMetrics class."""

    def test_learning_metrics_creation(self):
        """Test learning metrics creation."""
        metrics = LearningMetrics()

        assert metrics.training_steps == 0
        assert metrics.episodes_completed == 0
        assert metrics.cumulative_reward == 0.0
        assert len(metrics.losses) == 0
        assert len(metrics.episode_rewards) == 0

    def test_record_training_step(self):
        """Test recording training steps."""
        metrics = LearningMetrics()

        # Record several training steps
        for i in range(5):
            metrics.record_training_step(
                loss=0.1 * i,
                reward=1.0 + i,
                epsilon=0.9 - 0.1 * i,
                q_values_mean=2.0 + i,
            )

        assert metrics.training_steps == 5
        assert len(metrics.losses) == 5
        assert len(metrics.episode_rewards) == 5
        assert metrics.cumulative_reward == sum(range(1, 6))

    def test_record_episode_completion(self):
        """Test recording episode completion."""
        metrics = LearningMetrics()

        # Record episodes with different rewards
        episode_rewards = [10.0, 15.0, 8.0, 20.0, 12.0]
        for reward in episode_rewards:
            metrics.record_episode_completion(reward)

        assert metrics.episodes_completed == 5
        assert metrics.best_episode_reward == 20.0
        assert metrics.worst_episode_reward == 8.0

    def test_average_calculations(self):
        """Test average calculations."""
        metrics = LearningMetrics()

        # Add some data
        losses = [0.5, 0.4, 0.3, 0.2, 0.1]
        rewards = [1.0, 2.0, 3.0, 4.0, 5.0]

        for loss, reward in zip(losses, rewards):
            metrics.record_training_step(loss, reward, 0.5, 1.0)

        assert metrics.get_average_loss() == 0.3
        assert metrics.get_average_episode_reward() == 3.0

        # Test windowed averages
        assert (
            abs(metrics.get_average_loss(window=3) - 0.2) < 1e-10
        )  # Last 3: [0.3, 0.2, 0.1]
        assert metrics.get_average_episode_reward(window=2) == 4.5  # Last 2: [4.0, 5.0]

    def test_reward_trend_calculation(self):
        """Test reward trend calculation."""
        metrics = LearningMetrics()

        # Add increasing rewards (positive trend)
        for i in range(20):
            metrics.record_training_step(0.1, float(i), 0.5, 1.0)

        trend = metrics.get_reward_trend()
        assert trend > 0  # Should be positive trend

        # Clear and add decreasing rewards (negative trend)
        metrics.reset_metrics()
        for i in range(20, 0, -1):
            metrics.record_training_step(0.1, float(i), 0.5, 1.0)

        trend = metrics.get_reward_trend()
        assert trend < 0  # Should be negative trend

    def test_learning_stability(self):
        """Test learning stability calculation."""
        metrics = LearningMetrics()

        # Add stable losses (low variance)
        stable_losses = [0.1] * 20
        for loss in stable_losses:
            metrics.record_training_step(loss, 1.0, 0.5, 1.0)

        stability = metrics.get_learning_stability()
        assert stability > 0.8  # Should be high stability

        # Add unstable losses (high variance)
        import random

        random.seed(42)
        unstable_losses = [random.uniform(0.0, 1.0) for _ in range(20)]
        for loss in unstable_losses:
            metrics.record_training_step(loss, 1.0, 0.5, 1.0)

        stability = metrics.get_learning_stability()
        assert stability < 0.8  # Should be lower stability

    def test_exploration_progress(self):
        """Test exploration progress calculation."""
        metrics = LearningMetrics()

        # Simulate epsilon decay
        epsilon_values = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1]
        for epsilon in epsilon_values:
            metrics.record_training_step(0.1, 1.0, epsilon, 1.0)

        progress = metrics.get_exploration_progress()
        assert 0.8 < progress <= 1.0  # Should show significant progress

    def test_performance_summary(self):
        """Test performance summary generation."""
        metrics = LearningMetrics()

        # Add some training data
        for i in range(10):
            metrics.record_training_step(0.1, float(i), 0.9 - 0.05 * i, 1.0)
            if i % 2 == 0:
                metrics.record_episode_completion(float(i * 2))

        summary = metrics.get_performance_summary()

        assert "training_steps" in summary
        assert "episodes_completed" in summary
        assert "average_loss" in summary
        assert "average_episode_reward" in summary
        assert "reward_trend" in summary
        assert "learning_stability" in summary
        assert "exploration_progress" in summary

        # Test concise summary
        concise = metrics.get_summary()
        assert len(concise) < len(summary)
        assert "steps" in concise
        assert "avg_reward" in concise

    def test_metrics_serialization(self):
        """Test metrics serialization and deserialization."""
        metrics = LearningMetrics()

        # Add some data
        for i in range(5):
            metrics.record_training_step(0.1 * i, float(i), 0.9 - 0.1 * i, 1.0)
            metrics.record_episode_completion(float(i * 2))

        # Convert to dict
        metrics_dict = metrics.to_dict()

        assert "training_steps" in metrics_dict
        assert "episodes_completed" in metrics_dict
        assert "performance_summary" in metrics_dict

        # Create new metrics and load from dict
        new_metrics = LearningMetrics()
        new_metrics.from_dict(metrics_dict)

        assert new_metrics.training_steps == metrics.training_steps
        assert new_metrics.episodes_completed == metrics.episodes_completed
        assert new_metrics.cumulative_reward == metrics.cumulative_reward

    def test_metrics_reset(self):
        """Test metrics reset functionality."""
        metrics = LearningMetrics()

        # Add some data
        for i in range(5):
            metrics.record_training_step(0.1, 1.0, 0.5, 1.0)

        assert metrics.training_steps == 5
        assert len(metrics.losses) == 5

        # Reset metrics
        metrics.reset_metrics()

        assert metrics.training_steps == 0
        assert len(metrics.losses) == 0
        assert metrics.cumulative_reward == 0.0

    def test_export_for_analysis(self):
        """Test export for external analysis."""
        metrics = LearningMetrics()

        # Add some data
        for i in range(3):
            metrics.record_training_step(0.1, 1.0, 0.5, 1.0)

        export_data = metrics.export_for_analysis()

        assert "metadata" in export_data
        assert "performance_summary" in export_data
        assert "detailed_history" in export_data
        assert "recent_data" in export_data


class TestBaseRLAgent:
    """Test BaseRLAgent class."""

    def test_agent_initialization(self):
        """Test agent initialization."""
        config = AgentConfig()
        agent = MockRLAgent("test_agent", config)

        assert agent.agent_id == "test_agent"
        assert agent.config == config
        assert agent.action_size == 4  # From MockRLAgent action space
        assert agent.epsilon == config.epsilon_initial
        assert agent.training_step == 0
        assert agent.episode_count == 0

    def test_agent_state_representation(self):
        """Test agent state representation."""
        config = AgentConfig()
        agent = MockRLAgent("test_agent", config)

        env_state = {"feature1": 1.0, "feature2": 2.0, "feature3": 3.0}
        state = agent.get_state_representation(env_state)

        expected = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert np.array_equal(state, expected)

    def test_agent_action_selection(self):
        """Test agent action selection."""
        config = AgentConfig()
        config.epsilon_initial = 0.0  # No exploration for deterministic testing
        agent = MockRLAgent("test_agent", config)

        state = np.array([1.0, 2.0, 3.0])
        action = agent.select_action(state, training=False)

        assert 0 <= action < agent.action_size
        assert isinstance(action, int)

    def test_agent_reward_calculation(self):
        """Test agent reward calculation."""
        config = AgentConfig()
        agent = MockRLAgent("test_agent", config)

        state = np.array([1.0, 2.0, 3.0])
        result = {"base_reward": 1.0, "quality_score": 0.8}

        reward = agent.calculate_reward(state, 0, result)
        expected = 1.0 + 0.8 * 0.5  # base_reward + quality_score * 0.5

        assert reward == expected

    @patch("torch.cuda.is_available")
    def test_agent_device_selection(self, mock_cuda):
        """Test agent device selection."""
        config = AgentConfig()

        # Test CUDA available
        mock_cuda.return_value = True
        config.gpu_enabled = True
        agent = MockRLAgent("test_agent", config)
        # Note: In testing environment, CUDA might not actually be available
        # so we just check that the agent initializes without error

        # Test CPU only
        mock_cuda.return_value = False
        config.gpu_enabled = False
        agent_cpu = MockRLAgent("test_agent_cpu", config)
        assert str(agent_cpu.device) == "cpu"

    def test_agent_policy_update(self):
        """Test agent policy update."""
        config = AgentConfig()
        config.batch_size = 2  # Small batch for testing
        config.target_update_freq = 10  # Avoid target network update during test
        agent = MockRLAgent("test_agent", config)

        # Initialize networks by selecting an action first
        initial_state = np.array([0.0, 1.0, 2.0])
        agent.select_action(initial_state)

        # Add enough experiences for training
        for i in range(5):
            state = np.array([float(i), float(i + 1), float(i + 2)])
            next_state = np.array([float(i + 1), float(i + 2), float(i + 3)])

            agent.update_policy(
                state=state,
                action=i % 4,
                reward=1.0,
                next_state=next_state,
                done=i == 4,
            )

        assert len(agent.replay_buffer) == 5
        assert agent.training_step > 0
        assert agent.epsilon < config.epsilon_initial  # Should have decayed

    def test_agent_checkpoint_operations(self):
        """Test agent checkpoint save/load."""
        config = AgentConfig()
        agent = MockRLAgent("test_agent", config)

        # Initialize networks by selecting an action
        state = np.array([1.0, 2.0, 3.0])
        agent.select_action(state)

        # Add some training data
        agent.training_step = 100
        agent.episode_count = 10
        agent.epsilon = 0.5

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "test_checkpoint.pt"

            # Save checkpoint
            agent.save_checkpoint(str(checkpoint_path))
            assert checkpoint_path.exists()

            # Create new agent and load checkpoint
            new_agent = MockRLAgent("test_agent", config)
            new_agent.load_checkpoint(str(checkpoint_path))

            assert new_agent.training_step == 100
            assert new_agent.episode_count == 10
            assert new_agent.epsilon == 0.5

    def test_agent_summary(self):
        """Test agent summary generation."""
        config = AgentConfig()
        agent = MockRLAgent("test_agent", config)

        # Initialize networks
        state = np.array([1.0, 2.0, 3.0])
        agent.select_action(state)

        summary = agent.get_agent_summary()

        assert summary["agent_id"] == "test_agent"
        assert summary["episode_count"] == 0
        assert summary["training_step"] == 0
        assert summary["action_space_size"] == 4
        assert summary["state_size"] == 3
        assert summary["networks_initialized"] is True
        assert "learning_metrics" in summary

    def test_agent_episode_reset(self):
        """Test agent episode reset."""
        config = AgentConfig()
        agent = MockRLAgent("test_agent", config)

        # Set some state
        agent.current_state = np.array([1.0, 2.0, 3.0])
        agent.last_action = 1
        agent.last_reward = 0.5

        initial_episode_count = agent.episode_count

        # Reset episode
        agent.reset_episode()

        assert agent.current_state is None
        assert agent.last_action is None
        assert agent.last_reward is None
        assert agent.episode_count == initial_episode_count + 1

    def test_agent_action_confidence(self):
        """Test agent action confidence calculation."""
        config = AgentConfig()
        agent = MockRLAgent("test_agent", config)

        state = np.array([1.0, 2.0, 3.0])

        # Before network initialization
        confidence = agent.get_action_confidence(state, 0)
        assert confidence == 0.0

        # After network initialization
        agent.select_action(state)  # This initializes networks
        confidence = agent.get_action_confidence(state, 0)
        assert 0.0 <= confidence <= 1.0
