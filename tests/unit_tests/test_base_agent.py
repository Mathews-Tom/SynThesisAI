"""Unit tests for BaseRLAgent."""

# Import the old AgentConfig directly
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from core.marl.agents.base_agent import BaseRLAgent

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "core" / "marl"))
from config import AgentConfig


class ConcreteRLAgent(BaseRLAgent):
    """Concrete implementation of BaseRLAgent for testing."""

    def get_action_space(self):
        """Return action space size."""
        return list(range(4))  # Return list of actions

    def get_state_representation(self, context):
        """Return state representation."""
        return np.random.random(10)  # Fixed size state

    def calculate_reward(self, action, result, context):
        """Calculate reward for action."""
        return np.random.random()


class TestBaseRLAgent:
    """Test BaseRLAgent functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = AgentConfig()
        self.agent = ConcreteRLAgent("test_agent", self.config)

    def test_agent_initialization(self):
        """Test agent initialization."""
        assert self.agent.agent_id == "test_agent"
        assert self.agent.config == self.config
        assert self.agent.action_size == 4

    def test_get_action(self):
        """Test action selection."""
        state = np.random.random(10)
        action = self.agent.get_action(state)

        assert isinstance(action, int)
        assert 0 <= action < 4

    def test_update_experience(self):
        """Test experience update."""
        state = np.random.random(10)
        action = 1
        reward = 0.5
        next_state = np.random.random(10)
        done = False

        # Should not raise an exception
        self.agent.update(state, action, reward, next_state, done)

    def test_epsilon_decay(self):
        """Test epsilon decay functionality."""
        initial_epsilon = self.agent.epsilon

        # Simulate multiple updates to trigger epsilon decay
        for _ in range(100):
            state = np.random.random(10)
            action = 1
            reward = 0.5
            next_state = np.random.random(10)
            done = False
            self.agent.update(state, action, reward, next_state, done)

        # Epsilon should have decayed
        assert self.agent.epsilon <= initial_epsilon

    def test_get_metrics(self):
        """Test metrics collection."""
        metrics = self.agent.get_metrics()

        assert isinstance(metrics, dict)
        assert "total_actions" in metrics
        assert "total_rewards" in metrics
        assert "epsilon" in metrics

    def test_save_load_model(self):
        """Test model saving and loading."""
        # Test that save/load methods exist and don't crash
        try:
            self.agent.save_model("test_model.pth")
            self.agent.load_model("test_model.pth")
        except NotImplementedError:
            # Expected for base class
            pass
        except Exception as e:
            pytest.fail(f"Unexpected exception: {e}")

    def test_reset_agent(self):
        """Test agent reset functionality."""
        # Perform some actions first
        state = np.random.random(10)
        self.agent.get_action(state)

        # Reset agent
        self.agent.reset()

        # Check that metrics are reset
        metrics = self.agent.get_metrics()
        assert metrics["total_actions"] == 0
        assert metrics["total_rewards"] == 0.0


class TestAgentConfig:
    """Test AgentConfig functionality."""

    def test_config_creation(self):
        """Test config creation with default values."""
        config = AgentConfig()

        assert config.learning_rate == 0.001  # default
        assert config.gamma == 0.99  # default
        assert config.buffer_size == 100000  # default

    def test_config_validation(self):
        """Test config validation."""
        # Valid config
        config = AgentConfig()
        assert config.learning_rate > 0
        assert config.gamma > 0
        assert config.buffer_size > 0

        # Test validation method
        try:
            config.validate()
        except Exception:
            pytest.fail("Valid config should not raise exception")

    def test_config_serialization(self):
        """Test config serialization."""
        config = AgentConfig()

        # Test that config can be converted to dict
        config_dict = config.__dict__
        assert isinstance(config_dict, dict)
        assert config_dict["learning_rate"] == 0.001
        assert config_dict["gamma"] == 0.99


@pytest.mark.integration
class TestBaseRLAgentIntegration:
    """Integration tests for BaseRLAgent."""

    def test_agent_learning_loop(self):
        """Test complete learning loop."""
        config = AgentConfig()
        agent = ConcreteRLAgent("integration_test", config)

        # Simulate learning episodes
        for episode in range(10):
            state = np.random.random(4)

            for step in range(20):
                action = agent.get_action(state)
                next_state = np.random.random(4)
                reward = np.random.random()
                done = step == 19  # Last step

                agent.update(state, action, reward, next_state, done)
                state = next_state

                if done:
                    break

        # Check that agent has learned something
        metrics = agent.get_metrics()
        assert metrics["total_actions"] > 0
        assert metrics["total_rewards"] > 0

    def test_agent_performance_tracking(self):
        """Test agent performance tracking over time."""
        config = AgentConfig()
        agent = ConcreteRLAgent("performance_test", config)

        initial_metrics = agent.get_metrics()

        # Perform some actions
        for _ in range(50):
            state = np.random.random(4)
            action = agent.get_action(state)
            next_state = np.random.random(4)
            reward = 1.0  # Positive reward
            done = False

            agent.update(state, action, reward, next_state, done)

        final_metrics = agent.get_metrics()

        # Check that metrics have been updated
        assert final_metrics["total_actions"] > initial_metrics["total_actions"]
        assert final_metrics["total_rewards"] > initial_metrics["total_rewards"]
        assert final_metrics["average_reward"] > 0


if __name__ == "__main__":
    pytest.main([__file__])
