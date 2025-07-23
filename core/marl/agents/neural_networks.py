"""
Neural Network Architectures for RL Agents

This module provides neural network implementations for deep Q-learning,
including Q-networks and target networks with configurable architectures
following the development standards.
"""

import logging
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..exceptions import PolicyNetworkError

logger = logging.getLogger(__name__)


class QNetwork(nn.Module):
    """
    Deep Q-Network for value function approximation.

    Implements a fully connected neural network that maps states to Q-values
    for all possible actions. Supports configurable hidden layers and
    activation functions.
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_layers: List[int],
        activation: str = "relu",
    ):
        """
        Initialize Q-Network.

        Args:
            state_size: Dimension of input state
            action_size: Number of possible actions
            hidden_layers: List of hidden layer sizes
            activation: Activation function name
        """
        super(QNetwork, self).__init__()

        if state_size <= 0:
            raise PolicyNetworkError(
                f"State size must be positive, got {state_size}",
                agent_id="unknown",
                network_type="q_network",
                operation="initialize",
            )

        if action_size <= 0:
            raise PolicyNetworkError(
                f"Action size must be positive, got {action_size}",
                agent_id="unknown",
                network_type="q_network",
                operation="initialize",
            )

        if not hidden_layers:
            raise PolicyNetworkError(
                "Hidden layers list cannot be empty",
                agent_id="unknown",
                network_type="q_network",
                operation="initialize",
            )

        self.state_size = state_size
        self.action_size = action_size
        self.hidden_layers = hidden_layers
        self.activation_name = activation

        # Get activation function
        self.activation = self._get_activation_function(activation)

        # Build network layers
        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(nn.Linear(state_size, hidden_layers[0]))

        # Hidden layers
        for i in range(len(hidden_layers) - 1):
            self.layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))

        # Output layer
        self.layers.append(nn.Linear(hidden_layers[-1], action_size))

        # Initialize weights
        self._initialize_weights()

        logger.info(
            "Initialized Q-Network: state_size=%d, action_size=%d, "
            "hidden_layers=%s, activation=%s",
            state_size,
            action_size,
            hidden_layers,
            activation,
        )

    def _get_activation_function(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        activation_functions = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "leaky_relu": nn.LeakyReLU(),
            "elu": nn.ELU(),
            "gelu": nn.GELU(),
        }

        if activation.lower() not in activation_functions:
            raise PolicyNetworkError(
                f"Unknown activation function: {activation}",
                agent_id="unknown",
                network_type="q_network",
                operation="initialize",
            )

        return activation_functions[activation.lower()]

    def _initialize_weights(self) -> None:
        """Initialize network weights using Xavier initialization."""
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            state: Input state tensor

        Returns:
            Q-values for all actions
        """
        try:
            x = state

            # Forward through hidden layers with activation
            for layer in self.layers[:-1]:
                x = self.activation(layer(x))

            # Output layer (no activation)
            x = self.layers[-1](x)

            return x

        except Exception as e:
            error_msg = "Forward pass failed in Q-Network"
            logger.error("%s: %s", error_msg, str(e))
            raise PolicyNetworkError(
                error_msg,
                agent_id="unknown",
                network_type="q_network",
                operation="forward",
            ) from e

    def get_network_info(self) -> dict:
        """Get information about the network architecture."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "state_size": self.state_size,
            "action_size": self.action_size,
            "hidden_layers": self.hidden_layers,
            "activation": self.activation_name,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "num_layers": len(self.layers),
        }


class DuelingQNetwork(nn.Module):
    """
    Dueling Deep Q-Network architecture.

    Separates the value function and advantage function to improve learning
    stability and performance, especially in environments where many actions
    have similar values.
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_layers: List[int],
        activation: str = "relu",
    ):
        """
        Initialize Dueling Q-Network.

        Args:
            state_size: Dimension of input state
            action_size: Number of possible actions
            hidden_layers: List of hidden layer sizes
            activation: Activation function name
        """
        super(DuelingQNetwork, self).__init__()

        if len(hidden_layers) < 2:
            raise PolicyNetworkError(
                "Dueling network requires at least 2 hidden layers",
                agent_id="unknown",
                network_type="dueling_q_network",
                operation="initialize",
            )

        self.state_size = state_size
        self.action_size = action_size
        self.hidden_layers = hidden_layers
        self.activation_name = activation

        # Get activation function
        activation_fn = self._get_activation_function(activation)

        # Shared feature layers
        self.feature_layers = nn.ModuleList()
        self.feature_layers.append(nn.Linear(state_size, hidden_layers[0]))

        for i in range(len(hidden_layers) - 2):
            self.feature_layers.append(
                nn.Linear(hidden_layers[i], hidden_layers[i + 1])
            )

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_layers[-2], hidden_layers[-1]),
            activation_fn,
            nn.Linear(hidden_layers[-1], 1),
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_layers[-2], hidden_layers[-1]),
            activation_fn,
            nn.Linear(hidden_layers[-1], action_size),
        )

        self.activation = activation_fn
        self._initialize_weights()

        logger.info(
            "Initialized Dueling Q-Network: state_size=%d, action_size=%d, "
            "hidden_layers=%s, activation=%s",
            state_size,
            action_size,
            hidden_layers,
            activation,
        )

    def _get_activation_function(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        activation_functions = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "leaky_relu": nn.LeakyReLU(),
            "elu": nn.ELU(),
            "gelu": nn.GELU(),
        }

        if activation.lower() not in activation_functions:
            raise PolicyNetworkError(
                f"Unknown activation function: {activation}",
                agent_id="unknown",
                network_type="dueling_q_network",
                operation="initialize",
            )

        return activation_functions[activation.lower()]

    def _initialize_weights(self) -> None:
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the dueling network.

        Args:
            state: Input state tensor

        Returns:
            Q-values for all actions
        """
        try:
            # Forward through shared feature layers
            x = state
            for layer in self.feature_layers:
                x = self.activation(layer(x))

            # Separate value and advantage streams
            value = self.value_stream(x)
            advantage = self.advantage_stream(x)

            # Combine value and advantage
            # Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
            q_values = value + advantage - advantage.mean(dim=1, keepdim=True)

            return q_values

        except Exception as e:
            error_msg = "Forward pass failed in Dueling Q-Network"
            logger.error("%s: %s", error_msg, str(e))
            raise PolicyNetworkError(
                error_msg,
                agent_id="unknown",
                network_type="dueling_q_network",
                operation="forward",
            ) from e


def build_q_network(
    state_size: int,
    action_size: int,
    hidden_layers: List[int],
    activation: str = "relu",
    network_type: str = "standard",
) -> nn.Module:
    """
    Factory function to build Q-network.

    Args:
        state_size: Dimension of input state
        action_size: Number of possible actions
        hidden_layers: List of hidden layer sizes
        activation: Activation function name
        network_type: Type of network ("standard" or "dueling")

    Returns:
        Initialized Q-network
    """
    try:
        if network_type.lower() == "dueling":
            return DuelingQNetwork(state_size, action_size, hidden_layers, activation)
        else:
            return QNetwork(state_size, action_size, hidden_layers, activation)

    except Exception as e:
        error_msg = f"Failed to build Q-network of type {network_type}"
        logger.error("%s: %s", error_msg, str(e))
        raise PolicyNetworkError(
            error_msg, agent_id="unknown", network_type=network_type, operation="build"
        ) from e


def build_target_network(q_network: nn.Module) -> nn.Module:
    """
    Build target network as a copy of the Q-network.

    Args:
        q_network: Source Q-network to copy

    Returns:
        Target network with same architecture and weights
    """
    try:
        # Create a copy of the network
        target_network = type(q_network)(
            q_network.state_size,
            q_network.action_size,
            q_network.hidden_layers,
            q_network.activation_name,
        )

        # Copy weights
        target_network.load_state_dict(q_network.state_dict())

        # Set to evaluation mode and freeze parameters
        target_network.eval()
        for param in target_network.parameters():
            param.requires_grad = False

        logger.info("Built target network from Q-network")
        return target_network

    except Exception as e:
        error_msg = "Failed to build target network"
        logger.error("%s: %s", error_msg, str(e))
        raise PolicyNetworkError(
            error_msg,
            agent_id="unknown",
            network_type="target_network",
            operation="build",
        ) from e


def get_network_summary(network: nn.Module) -> str:
    """
    Get a summary string of the network architecture.

    Args:
        network: Neural network to summarize

    Returns:
        String summary of the network
    """
    try:
        total_params = sum(p.numel() for p in network.parameters())
        trainable_params = sum(
            p.numel() for p in network.parameters() if p.requires_grad
        )

        summary = f"Network Summary:\n"
        summary += f"  Type: {type(network).__name__}\n"
        summary += f"  Total Parameters: {total_params:,}\n"
        summary += f"  Trainable Parameters: {trainable_params:,}\n"

        if hasattr(network, "get_network_info"):
            info = network.get_network_info()
            summary += f"  State Size: {info['state_size']}\n"
            summary += f"  Action Size: {info['action_size']}\n"
            summary += f"  Hidden Layers: {info['hidden_layers']}\n"
            summary += f"  Activation: {info['activation']}\n"

        return summary

    except Exception as e:
        logger.error("Failed to generate network summary: %s", str(e))
        return f"Network: {type(network).__name__} (summary unavailable)"
