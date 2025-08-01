"""
Multi-Agent Reinforcement Learning (MARL) Coordination Module.

This module provides the foundation for multi-agent reinforcement learning
coordination in the SynThesisAI platform, enabling sophisticated collaborative
decision-making between Generator, Validator, and Curriculum agents.
"""

# SynThesisAI Modules
from .config import MARLConfig
from .exceptions import (
    AgentFailureError,
    CoordinationError,
    LearningDivergenceError,
    MARLError,
    OptimizationFailureError,
)

__all__ = [
    "MARLConfig",
    "MARLError",
    "CoordinationError",
    "AgentFailureError",
    "OptimizationFailureError",
    "LearningDivergenceError",
]
