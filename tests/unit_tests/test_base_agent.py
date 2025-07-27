"""Unit tests for BaseRLAgent."""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from core.marl.agents.base_agent import BaseRLAgent

# Import AgentConfig directly from the config.py file to avoid import conflicts
config_path = Path(__file__).parent.parent.parent / "core" / "marl" / "config_legacy.py"
spec = importlib.util.spec_from_file_location("marl_config", config_path)
marl_config_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(marl_config_module)
AgentConfig = marl_config_module.AgentConfig


class ConcreteRLAgent(BaseRLAgent):
    """Concrete implementation of BaseRLAgent for testing."""

    def get_action_space(self):
        """Return action space."""
        from core.marl.agents.base_agent import ActionSpace

        return ActionSpace(actions=["action_0", "action_1", "action_2", "action_3"])

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
        action = self.agent.select_action(state)

        assert isinstance(action, int)
        assert 0 <= action < 4

    def test_update_experience(self):
        """Test experience update."""
        state = np.random.random(10)
        action = 1
        reward = 0.5
        next_state = np.random.random(10)
        done = False

        # Initialize networks by selecting an action first
        self.agent.select_action(state)

        # Should not raise an exception
        self.agent.update_policy(state, action, reward, next_state, done)

    def test_epsilon_decay(self):
        """Test epsilon decay functionality."""
        initial_epsilon = self.agent.epsilon

        # Initialize networks first
        state = np.random.random(10)
        self.agent.select_action(state)

        # Simulate multiple updates to trigger epsilon decay
        for _ in range(100):
            state = np.random.random(10)
            action = 1
            reward = 0.5
            next_state = np.random.random(10)
            done = False
            self.agent.update_policy(state, action, reward, next_state, done)

        # Epsilon should have decayed
        assert self.agent.epsilon <= initial_epsilon

    def test_get_metrics(self):
        """Test metrics collection."""
        metrics = self.agent.get_agent_summary()

        assert isinstance(metrics, dict)
        assert "epsilon" in metrics
        assert "learning_metrics" in metrics
        learning_metrics = metrics["learning_metrics"]
        assert "steps" in learning_metrics
        assert "avg_reward" in learning_metrics

    def test_save_load_model(self):
        """Test model saving and loading."""
        # Test that save/load methods exist and don't crash
        try:
            self.agent.save_checkpoint("test_model.pth")
            self.agent.load_checkpoint("test_model.pth")
        except NotImplementedError:
            # Expected for base class
            pass
        except Exception as e:
            pytest.fail(f"Unexpected exception: {e}")

    def test_reset_agent(self):
        """Test agent reset functionality."""
        # Perform some actions first
        state = np.random.random(10)
        self.agent.select_action(state)

        # Reset agent
        self.agent.reset_episode()

        # Check that metrics are reset
        metrics = self.agent.get_agent_summary()
        learning_metrics = metrics["learning_metrics"]
        assert learning_metrics["steps"] == 0
        assert learning_metrics["avg_reward"] == 0.0


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
                action = agent.select_action(state)
                next_state = np.random.random(4)
                reward = np.random.random()
                done = step == 19  # Last step

                agent.update_policy(state, action, reward, next_state, done)
                state = next_state

                if done:
                    break

        # Check that agent has learned something
        metrics = agent.get_agent_summary()
        learning_metrics = metrics["learning_metrics"]
        assert learning_metrics["steps"] > 0
        # Note: avg_reward might be 0 if no rewards were given

    def test_agent_performance_tracking(self):
        """Test agent performance tracking over time."""
        config = AgentConfig()
        agent = ConcreteRLAgent("performance_test", config)

        initial_metrics = agent.get_agent_summary()

        # Perform some actions
        for _ in range(50):
            state = np.random.random(4)
            action = agent.select_action(state)
            next_state = np.random.random(4)
            reward = 1.0  # Positive reward
            done = False

            agent.update_policy(state, action, reward, next_state, done)

        final_metrics = agent.get_agent_summary()

        # Check that metrics have been updated
        initial_learning = initial_metrics["learning_metrics"]
        final_learning = final_metrics["learning_metrics"]
        assert final_learning["steps"] > initial_learning["steps"]
        # avg_reward should be positive since we're giving positive rewards
        assert final_learning["avg_reward"] > 0


if __name__ == "__main__":
    pytest.main([__file__])
