"""
Base RL Agent Framework.

This module provides the abstract base class for reinforcement learning agents
with common RL functionality including deep Q-learning, experience replay,
and epsilon-greedy exploration strategies.
"""

# Standard Library
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Third-Party Library
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# SynThesisAI Modules
from ..config_legacy import AgentConfig
from ..exceptions import AgentFailureError, PolicyNetworkError
from ..logging_config import get_marl_logger
from .experience import Experience
from .learning_metrics import LearningMetrics
from .neural_networks import build_q_network, build_target_network
from .replay_buffer import ReplayBuffer

logger = logging.getLogger(__name__)


@dataclass
class ActionSpace:
    """Defines the action space for an RL agent."""

    actions: List[str] = field(default_factory=list)

    def __len__(self) -> int:
        """Return the number of actions in the space."""
        return len(self.actions)

    def __getitem__(self, index: int) -> str:
        """Get action by index."""
        if 0 <= index < len(self.actions):
            return self.actions[index]
        raise IndexError(f"Action index {index} out of range [0, {len(self.actions)})")

    def get_action_index(self, action: str) -> int:
        """Get index of an action by its name."""
        try:
            return self.actions.index(action)
        except ValueError as e:
            raise ValueError(f"Action '{action}' not found in action space") from e

    def sample(self) -> int:
        """Sample a random action index."""
        return np.random.randint(0, len(self.actions))


class BaseRLAgent(ABC):
    """
    Abstract base class for reinforcement learning agents.

    Provides common RL functionality including deep Q-learning with experience
    replay, epsilon-greedy exploration, and policy optimization mechanisms.
    """

    def __init__(self, agent_id: str, config: AgentConfig) -> None:
        """
        Initialize the base RL agent.

        Args:
            agent_id: A unique identifier for this agent.
            config: The configuration parameters for the agent.
        """
        self.agent_id = agent_id
        self.config = config
        self.logger = get_marl_logger(f"agent.{agent_id}", None)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and config.gpu_enabled else "cpu"
        )

        # Neural network components
        self.state_size: Optional[int] = None
        self.action_size = len(self.get_action_space())
        self.q_network: Optional[nn.Module] = None
        self.target_network: Optional[nn.Module] = None
        self.optimizer: Optional[optim.Optimizer] = None

        # Learning components
        self.replay_buffer = ReplayBuffer(config.buffer_size)
        self.epsilon = config.epsilon_initial
        self.epsilon_decay = config.epsilon_decay
        self.epsilon_min = config.epsilon_min

        # Performance tracking
        self.learning_metrics = LearningMetrics()
        self.training_step = 0
        self.episode_count = 0

        # State tracking
        self.current_state: Optional[np.ndarray] = None
        self.last_action: Optional[int] = None
        self.last_reward: Optional[float] = None

        self.logger.log_agent_action(
            self.agent_id,
            "initialized",
            1.0,
            f"Device: {self.device}, Action space: {self.action_size}",
        )

    def _initialize_networks(self, state_size: int) -> None:
        """Initialize neural networks with the given state size."""
        if self.q_network is not None:
            return

        try:
            self.state_size = state_size

            # Build Q-network
            self.q_network = build_q_network(
                state_size=state_size,
                action_size=self.action_size,
                hidden_layers=self.config.hidden_layers,
                activation=self.config.activation,
            ).to(self.device)

            # Build target network
            self.target_network = build_target_network(self.q_network).to(self.device)
            self.optimizer = optim.Adam(
                self.q_network.parameters(), lr=self.config.learning_rate
            )
            self.logger.log_agent_action(
                self.agent_id,
                "networks_initialized",
                1.0,
                f"State size: {state_size}, Hidden layers: {self.config.hidden_layers}",
            )
        except Exception as e:
            error_msg = f"Failed to initialize networks for agent {self.agent_id}"
            self.logger.log_error_with_context(
                e,
                {
                    "agent_id": self.agent_id,
                    "state_size": state_size,
                    "action_size": self.action_size,
                },
            )
            raise PolicyNetworkError(
                error_msg,
                agent_id=self.agent_id,
                network_type="initialization",
                operation="build",
            ) from e

    @abstractmethod
    def get_state_representation(self, environment_state: Dict[str, Any]) -> np.ndarray:
        """
        Convert environment state to an agent-specific representation.

        Args:
            environment_state: The raw state from the environment.

        Returns:
            A NumPy array representing the state for this agent.
        """

    @abstractmethod
    def get_action_space(self) -> ActionSpace:
        """
        Define the agent-specific action space.

        Returns:
            An ActionSpace object defining available actions for this agent.
        """

    @abstractmethod
    def calculate_reward(
        self, state: np.ndarray, action: int, result: Dict[str, Any]
    ) -> float:
        """
        Calculate an agent-specific reward for the given state-action-result.

        Args:
            state: The state representation.
            action: The action taken.
            result: The result of the action.

        Returns:
            The reward value for this agent.
        """

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select an action using an epsilon-greedy strategy.

        Args:
            state: The current state representation.
            training: Whether the agent is in training mode.

        Returns:
            The selected action index.
        """
        try:
            # Initialize networks if needed
            if self.q_network is None:
                self._initialize_networks(len(state))

            # Epsilon-greedy action selection
            if training and np.random.random() <= self.epsilon:
                # Explore: random action
                action = self.get_action_space().sample()
                confidence = 0.0
            else:
                # Exploit: best action according to Q-notebook
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    q_values = self.q_network(state_tensor)
                    action = q_values.argmax().item()
                    confidence = torch.softmax(q_values, dim=1).max().item()

            # Log action selection
            action_name = self.get_action_space()[action]
            state_summary = f"dim={len(state)}, norm={np.linalg.norm(state):.3f}"
            self.logger.log_agent_action(
                self.agent_id, action_name, confidence, state_summary
            )
            return action

        except Exception as e:
            error_msg = f"Action selection failed for agent {self.agent_id}"
            self.logger.log_error_with_context(
                e,
                {
                    "agent_id": self.agent_id,
                    "state_shape": state.shape,
                    "training": training,
                    "epsilon": self.epsilon,
                },
            )
            raise AgentFailureError(
                error_msg,
                agent_id=self.agent_id,
                failure_type="action_selection",
                agent_state={
                    "epsilon": self.epsilon,
                    "training_step": self.training_step,
                },
            ) from e

    def update_policy(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        Update the agent's policy using Q-learning.

        Args:
            state: The current state.
            action: The action taken.
            reward: The reward received.
            next_state: The next state.
            done: Whether the episode has finished.
        """
        try:
            experience = Experience(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                metadata={"agent_id": self.agent_id, "episode": self.episode_count},
            )
            self.replay_buffer.add(experience)

            # Train if enough experiences available
            if len(self.replay_buffer) >= self.config.batch_size:
                loss = self.train_on_batch()

                # Update learning metrics
                self.learning_metrics.record_training_step(
                    loss=loss,
                    reward=reward,
                    epsilon=self.epsilon,
                    q_values_mean=self._get_mean_q_values(state),
                )

                # Log learning update
                self.logger.log_learning_update(
                    self.agent_id, self.episode_count, reward, loss, self.epsilon
                )

            # Update epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            # Update target network periodically
            if self.training_step % self.config.target_update_freq == 0:
                self.update_target_network()

            self.training_step += 1

        except Exception as e:
            error_msg = f"Policy update failed for agent {self.agent_id}"
            self.logger.log_error_with_context(
                e,
                {
                    "agent_id": self.agent_id,
                    "training_step": self.training_step,
                    "buffer_size": len(self.replay_buffer),
                    "epsilon": self.epsilon,
                },
            )
            raise AgentFailureError(
                error_msg,
                agent_id=self.agent_id,
                failure_type="policy_update",
                agent_state={
                    "training_step": self.training_step,
                    "epsilon": self.epsilon,
                    "buffer_size": len(self.replay_buffer),
                },
            ) from e

    def train_on_batch(self) -> float:
        """
        Train the Q-network on a batch of experiences.

        Returns:
            The training loss value.
        """
        try:
            # Sample batch from replay buffer
            experiences = self.replay_buffer.sample(self.config.batch_size)

            # Convert to tensors (optimized to avoid numpy array list warning)
            states = torch.FloatTensor(np.array([exp.state for exp in experiences])).to(
                self.device
            )
            actions = torch.LongTensor(
                np.array([exp.action for exp in experiences])
            ).to(self.device)
            rewards = torch.FloatTensor(
                np.array([exp.reward for exp in experiences])
            ).to(self.device)
            next_states = torch.FloatTensor(
                np.array([exp.next_state for exp in experiences])
            ).to(self.device)
            dones = torch.BoolTensor(np.array([exp.done for exp in experiences])).to(
                self.device
            )

            # Calculate current Q-values
            current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

            # Calculate target Q-values
            with torch.no_grad():
                next_q_values = self.target_network(next_states).max(1)[0]
                target_q_values = rewards + (self.config.gamma * next_q_values * ~dones)

            # Calculate loss
            loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

            # Optimize
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
            self.optimizer.step()
            return loss.item()

        except Exception as e:
            error_msg = f"Batch training failed for agent {self.agent_id}"
            self.logger.log_error_with_context(
                e,
                {
                    "agent_id": self.agent_id,
                    "batch_size": self.config.batch_size,
                    "buffer_size": len(self.replay_buffer),
                },
            )
            raise PolicyNetworkError(
                error_msg,
                agent_id=self.agent_id,
                network_type="q_network",
                operation="train",
            ) from e

    def update_target_network(self) -> None:
        """Update the target network with the current Q-network weights."""
        try:
            self.target_network.load_state_dict(self.q_network.state_dict())
            self.logger.log_agent_action(
                self.agent_id,
                "target_network_updated",
                1.0,
                f"Step: {self.training_step}",
            )
        except Exception as e:
            error_msg = f"Target network update failed for agent {self.agent_id}"
            raise PolicyNetworkError(
                error_msg,
                agent_id=self.agent_id,
                network_type="target_network",
                operation="update",
            ) from e

    def get_action_confidence(self, state: np.ndarray, action: int) -> float:
        """
        Get the confidence score for a specific action in the given state.

        Args:
            state: The state representation.
            action: The action index.

        Returns:
            A confidence score between 0 and 1.
        """
        try:
            if self.q_network is None:
                return 0.0
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                probabilities = torch.softmax(q_values, dim=1)
                return probabilities[0, action].item()
        except Exception as e:
            self.logger.log_error_with_context(
                e,
                {
                    "agent_id": self.agent_id,
                    "state_shape": state.shape,
                    "action": action,
                },
            )
            return 0.0

    def _get_mean_q_values(self, state: np.ndarray) -> float:
        """Get the mean Q-values for the given state."""
        try:
            if self.q_network is None:
                return 0.0
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                return q_values.mean().item()
        except Exception:
            return 0.0

    def save_checkpoint(self, checkpoint_path: str) -> None:
        """
        Save the agent's checkpoint.

        Args:
            checkpoint_path: The path to save the checkpoint to.
        """
        try:
            checkpoint = {
                "agent_id": self.agent_id,
                "episode_count": self.episode_count,
                "training_step": self.training_step,
                "epsilon": self.epsilon,
                "q_network_state_dict": (
                    self.q_network.state_dict() if self.q_network else None
                ),
                "target_network_state_dict": (
                    self.target_network.state_dict() if self.target_network else None
                ),
                "optimizer_state_dict": (
                    self.optimizer.state_dict() if self.optimizer else None
                ),
                "learning_metrics": self.learning_metrics.to_dict(),
                "config": self.config.__dict__,
            }
            torch.save(checkpoint, checkpoint_path)
            self.logger.log_checkpoint_save(
                checkpoint_path, self.episode_count, [self.agent_id]
            )
        except Exception as e:
            error_msg = f"Checkpoint save failed for agent {self.agent_id}"
            self.logger.log_error_with_context(
                e, {"agent_id": self.agent_id, "checkpoint_path": checkpoint_path}
            )
            raise AgentFailureError(
                error_msg, agent_id=self.agent_id, failure_type="checkpoint_save"
            ) from e

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load an agent's checkpoint.

        Args:
            checkpoint_path: The path to load the checkpoint from.
        """
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # Restore training state
            self.episode_count = checkpoint.get("episode_count", 0)
            self.training_step = checkpoint.get("training_step", 0)
            self.epsilon = checkpoint.get("epsilon", self.config.epsilon_initial)

            # Restore networks if they exist
            if checkpoint.get("q_network_state_dict") and self.q_network:
                self.q_network.load_state_dict(checkpoint["q_network_state_dict"])
            if checkpoint.get("target_network_state_dict") and self.target_network:
                self.target_network.load_state_dict(
                    checkpoint["target_network_state_dict"]
                )
            if checkpoint.get("optimizer_state_dict") and self.optimizer:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            # Restore learning metrics
            if checkpoint.get("learning_metrics"):
                self.learning_metrics.from_dict(checkpoint["learning_metrics"])

            self.logger.log_agent_action(
                self.agent_id,
                "checkpoint_loaded",
                1.0,
                f"Episode: {self.episode_count}, Step: {self.training_step}",
            )
        except Exception as e:
            error_msg = f"Checkpoint load failed for agent {self.agent_id}"
            self.logger.log_error_with_context(
                e, {"agent_id": self.agent_id, "checkpoint_path": checkpoint_path}
            )
            raise AgentFailureError(
                error_msg, agent_id=self.agent_id, failure_type="checkpoint_load"
            ) from e

    def reset_episode(self) -> None:
        """Reset the agent's state for a new episode."""
        self.current_state = None
        self.last_action = None
        self.last_reward = None
        self.episode_count += 1

    def get_agent_summary(self) -> Dict[str, Any]:
        """Get a summary of the agent's state and performance."""
        return {
            "agent_id": self.agent_id,
            "episode_count": self.episode_count,
            "training_step": self.training_step,
            "epsilon": self.epsilon,
            "buffer_size": len(self.replay_buffer),
            "action_space_size": self.action_size,
            "state_size": self.state_size,
            "device": str(self.device),
            "learning_metrics": self.learning_metrics.get_summary(),
            "networks_initialized": self.q_network is not None,
        }
