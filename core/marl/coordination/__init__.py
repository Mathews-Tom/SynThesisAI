"""
MARL Coordination Module

This module provides coordination mechanisms for multi-agent reinforcement learning,
including coordination policies, consensus mechanisms, and communication protocols.
"""

from .communication_protocol import (
    AgentCommunicationProtocol,
    AgentMessage,
    MessageResponse,
)
from .conflict_resolver import ConflictResolver
from .consensus_mechanism import ConsensusMechanism
from .coordination_policy import CoordinatedAction, CoordinationPolicy

__all__ = [
    "CoordinationPolicy",
    "CoordinatedAction",
    "ConflictResolver",
    "ConsensusMechanism",
    "AgentCommunicationProtocol",
    "AgentMessage",
    "MessageResponse",
]
