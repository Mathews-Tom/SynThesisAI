"""
MARL Agents Module.

This module provides the base RL agent framework and specialized agent
implementations for multi-agent reinforcement learning coordination in the
SynThesisAI platform.
"""

# SynThesisAI Modules
from .base_agent import ActionSpace, BaseRLAgent
from .experience import Experience
from .learning_metrics import LearningMetrics
from .neural_networks import QNetwork, build_q_network, build_target_network
from .replay_buffer import PrioritizedReplayBuffer, ReplayBuffer

__all__ = [
    "BaseRLAgent",
    "ActionSpace",
    "Experience",
    "ReplayBuffer",
    "PrioritizedReplayBuffer",
    "QNetwork",
    "build_q_network",
    "build_target_network",
    "LearningMetrics",
]
