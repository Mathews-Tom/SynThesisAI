"""
Experience Data Structures.

This module provides data structures for storing and managing experiences
in reinforcement learning, including the Experience class and related utilities.
"""

# Standard Library
import time
from dataclasses import dataclass, field
from typing import Any, Dict

# Third-Party Library
import numpy as np


@dataclass
class Experience:
    """Represents a single experience tuple for RL training."""

    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the experience to a dictionary for serialization.

        Returns:
            A dictionary representation of the experience.
        """
        return {
            "state": self.state.tolist(),
            "action": self.action,
            "reward": self.reward,
            "next_state": self.next_state.tolist(),
            "done": self.done,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Experience":
        """
        Create an Experience object from a dictionary.

        Args:
            data: The dictionary to create the object from.

        Returns:
            An Experience object.
        """
        return cls(
            state=np.array(data["state"]),
            action=data["action"],
            reward=data["reward"],
            next_state=np.array(data["next_state"]),
            done=data["done"],
            metadata=data.get("metadata", {}),
            timestamp=data.get("timestamp", time.time()),
        )
