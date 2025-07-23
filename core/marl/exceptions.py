"""
MARL-Specific Exception Classes

This module defines custom exception classes for multi-agent reinforcement learning
operations, following the development standards for proper exception handling
with specific exception types and explicit exception chaining.
"""

from typing import Any, Dict, Optional


class MARLError(Exception):
    """Base exception class for MARL-related errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """
        Initialize MARL error.

        Args:
            message: Error message
            details: Additional error details and context
        """
        super().__init__(message)
        self.details = details or {}


class CoordinationError(MARLError):
    """Exception raised when multi-agent coordination fails."""

    def __init__(
        self,
        message: str,
        coordination_type: str = "unknown",
        agent_states: Optional[Dict[str, Any]] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize coordination error.

        Args:
            message: Error message
            coordination_type: Type of coordination that failed
            agent_states: Current states of agents involved
            details: Additional error details
        """
        super().__init__(message, details)
        self.coordination_type = coordination_type
        self.agent_states = agent_states or {}


class AgentFailureError(MARLError):
    """Exception raised when an individual RL agent fails."""

    def __init__(
        self,
        message: str,
        agent_id: str,
        failure_type: str = "unknown",
        agent_state: Optional[Dict[str, Any]] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize agent failure error.

        Args:
            message: Error message
            agent_id: ID of the failed agent
            failure_type: Type of failure (e.g., 'policy_divergence', 'memory_overflow')
            agent_state: Current state of the failed agent
            details: Additional error details
        """
        super().__init__(message, details)
        self.agent_id = agent_id
        self.failure_type = failure_type
        self.agent_state = agent_state or {}


class OptimizationFailureError(MARLError):
    """Exception raised when RL optimization fails."""

    def __init__(
        self,
        message: str,
        optimizer_type: str = "unknown",
        optimization_params: Optional[Dict[str, Any]] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize optimization failure error.

        Args:
            message: Error message
            optimizer_type: Type of optimizer that failed
            optimization_params: Parameters used during optimization
            details: Additional error details including performance metrics
        """
        super().__init__(message, details)
        self.optimizer_type = optimizer_type
        self.optimization_params = optimization_params or {}


class LearningDivergenceError(MARLError):
    """Exception raised when agent learning diverges or becomes unstable."""

    def __init__(
        self,
        message: str,
        agent_id: str,
        divergence_metrics: Optional[Dict[str, float]] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize learning divergence error.

        Args:
            message: Error message
            agent_id: ID of the agent with divergent learning
            divergence_metrics: Metrics indicating divergence (loss, reward variance, etc.)
            details: Additional error details
        """
        super().__init__(message, details)
        self.agent_id = agent_id
        self.divergence_metrics = divergence_metrics or {}


class ConsensusTimeoutError(CoordinationError):
    """Exception raised when consensus building times out."""

    def __init__(
        self,
        message: str,
        timeout_duration: float,
        partial_consensus: Optional[Dict[str, Any]] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize consensus timeout error.

        Args:
            message: Error message
            timeout_duration: Duration of timeout in seconds
            partial_consensus: Partial consensus reached before timeout
            details: Additional error details
        """
        super().__init__(message, "consensus_timeout", details=details)
        self.timeout_duration = timeout_duration
        self.partial_consensus = partial_consensus or {}


class CommunicationError(MARLError):
    """Exception raised when inter-agent communication fails."""

    def __init__(
        self,
        message: str,
        sender_id: str,
        receiver_id: str,
        message_type: str = "unknown",
        details: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize communication error.

        Args:
            message: Error message
            sender_id: ID of the sending agent
            receiver_id: ID of the receiving agent
            message_type: Type of message that failed
            details: Additional error details
        """
        super().__init__(message, details)
        self.sender_id = sender_id
        self.receiver_id = receiver_id
        self.message_type = message_type


class ExperienceBufferError(MARLError):
    """Exception raised when experience buffer operations fail."""

    def __init__(
        self,
        message: str,
        buffer_type: str = "unknown",
        buffer_size: Optional[int] = None,
        operation: str = "unknown",
        details: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize experience buffer error.

        Args:
            message: Error message
            buffer_type: Type of buffer (shared, agent-specific, etc.)
            buffer_size: Current buffer size
            operation: Operation that failed (add, sample, etc.)
            details: Additional error details
        """
        super().__init__(message, details)
        self.buffer_type = buffer_type
        self.buffer_size = buffer_size
        self.operation = operation


class PolicyNetworkError(MARLError):
    """Exception raised when neural network policy operations fail."""

    def __init__(
        self,
        message: str,
        agent_id: str,
        network_type: str = "q_network",
        operation: str = "unknown",
        details: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize policy network error.

        Args:
            message: Error message
            agent_id: ID of the agent whose network failed
            network_type: Type of network (q_network, target_network, etc.)
            operation: Operation that failed (forward, backward, update, etc.)
            details: Additional error details
        """
        super().__init__(message, details)
        self.agent_id = agent_id
        self.network_type = network_type
        self.operation = operation
