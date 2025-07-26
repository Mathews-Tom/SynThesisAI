"""
MARL Error Types.

This module defines specialized error types for the multi-agent
reinforcement learning coordination system.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


class MARLError(Exception):
    """
    Base exception class for MARL-related errors.

    Provides common functionality for error tracking, context preservation,
    and recovery strategy hints.
    """

    def __init__(
        self,
        message: str,
        error_code: str = "MARL_UNKNOWN",
        context: Optional[Dict[str, Any]] = None,
        recovery_hint: Optional[str] = None,
        severity: str = "ERROR",
    ):
        """
        Initialize MARL error.

        Args:
            message: Human-readable error description
            error_code: Unique error code for classification
            context: Additional context information
            recovery_hint: Suggested recovery strategy
            severity: Error severity level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or {}
        self.recovery_hint = recovery_hint
        self.severity = severity
        self.timestamp = datetime.now()
        self.error_id = self._generate_error_id()

    def _generate_error_id(self) -> str:
        """Generate unique error ID for tracking."""
        return f"{self.error_code}_{self.timestamp.strftime('%Y%m%d_%H%M%S_%f')}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for logging and analysis."""
        return {
            "error_id": self.error_id,
            "error_code": self.error_code,
            "message": self.message,
            "severity": self.severity,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context,
            "recovery_hint": self.recovery_hint,
            "error_type": self.__class__.__name__,
        }

    def add_context(self, key: str, value: Any) -> None:
        """Add additional context information."""
        self.context[key] = value

    def get_context(self, key: str, default: Any = None) -> Any:
        """Get context value by key."""
        return self.context.get(key, default)


class AgentError(MARLError):
    """Errors related to individual RL agents."""

    def __init__(
        self,
        message: str,
        agent_id: str,
        agent_type: Optional[str] = None,
        error_code: str = "AGENT_ERROR",
        **kwargs,
    ):
        super().__init__(message, error_code, **kwargs)
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.add_context("agent_id", agent_id)
        if agent_type:
            self.add_context("agent_type", agent_type)


class AgentInitializationError(AgentError):
    """Error during agent initialization."""

    def __init__(self, message: str, agent_id: str, **kwargs):
        super().__init__(
            message,
            agent_id,
            error_code="AGENT_INIT_ERROR",
            recovery_hint="Check agent configuration and dependencies",
            **kwargs,
        )


class AgentTrainingError(AgentError):
    """Error during agent training/learning."""

    def __init__(
        self, message: str, agent_id: str, training_step: Optional[int] = None, **kwargs
    ):
        super().__init__(
            message,
            agent_id,
            error_code="AGENT_TRAINING_ERROR",
            recovery_hint="Check learning parameters and experience buffer",
            **kwargs,
        )
        if training_step is not None:
            self.add_context("training_step", training_step)


class AgentActionError(AgentError):
    """Error during agent action selection or execution."""

    def __init__(
        self,
        message: str,
        agent_id: str,
        action: Optional[Any] = None,
        state: Optional[Any] = None,
        **kwargs,
    ):
        super().__init__(
            message,
            agent_id,
            error_code="AGENT_ACTION_ERROR",
            recovery_hint="Validate action space and state representation",
            **kwargs,
        )
        if action is not None:
            self.add_context("action", action)
        if state is not None:
            self.add_context("state_shape", getattr(state, "shape", "unknown"))


class CoordinationError(MARLError):
    """Errors related to multi-agent coordination."""

    def __init__(
        self,
        message: str,
        coordination_id: Optional[str] = None,
        participating_agents: Optional[List[str]] = None,
        error_code: str = "COORDINATION_ERROR",
        **kwargs,
    ):
        super().__init__(message, error_code, **kwargs)
        self.coordination_id = coordination_id
        self.participating_agents = participating_agents or []

        if coordination_id:
            self.add_context("coordination_id", coordination_id)
        if participating_agents:
            self.add_context("participating_agents", participating_agents)


class CoordinationTimeoutError(CoordinationError):
    """Coordination process timed out."""

    def __init__(self, message: str, timeout_duration: float, **kwargs):
        super().__init__(
            message,
            error_code="COORDINATION_TIMEOUT",
            recovery_hint="Increase coordination timeout or check agent responsiveness",
            **kwargs,
        )
        self.add_context("timeout_duration", timeout_duration)


class CoordinationDeadlockError(CoordinationError):
    """Coordination deadlock detected."""

    def __init__(self, message: str, deadlock_type: str = "unknown", **kwargs):
        super().__init__(
            message,
            error_code="COORDINATION_DEADLOCK",
            recovery_hint="Reset coordination state and use fallback strategy",
            **kwargs,
        )
        self.add_context("deadlock_type", deadlock_type)


class ConsensusError(MARLError):
    """Errors related to consensus mechanisms."""

    def __init__(
        self,
        message: str,
        consensus_type: Optional[str] = None,
        proposal_id: Optional[str] = None,
        error_code: str = "CONSENSUS_ERROR",
        **kwargs,
    ):
        super().__init__(message, error_code, **kwargs)
        self.consensus_type = consensus_type
        self.proposal_id = proposal_id

        if consensus_type:
            self.add_context("consensus_type", consensus_type)
        if proposal_id:
            self.add_context("proposal_id", proposal_id)


class ConsensusFailureError(ConsensusError):
    """Failed to reach consensus."""

    def __init__(
        self, message: str, votes_received: int = 0, votes_required: int = 0, **kwargs
    ):
        super().__init__(
            message,
            error_code="CONSENSUS_FAILURE",
            recovery_hint="Lower consensus threshold or use fallback decision",
            **kwargs,
        )
        self.add_context("votes_received", votes_received)
        self.add_context("votes_required", votes_required)


class CommunicationError(MARLError):
    """Errors related to agent communication."""

    def __init__(
        self,
        message: str,
        sender: Optional[str] = None,
        receiver: Optional[str] = None,
        message_type: Optional[str] = None,
        error_code: str = "COMMUNICATION_ERROR",
        **kwargs,
    ):
        super().__init__(message, error_code, **kwargs)
        self.sender = sender
        self.receiver = receiver
        self.message_type = message_type

        if sender:
            self.add_context("sender", sender)
        if receiver:
            self.add_context("receiver", receiver)
        if message_type:
            self.add_context("message_type", message_type)


class MessageDeliveryError(CommunicationError):
    """Failed to deliver message between agents."""

    def __init__(self, message: str, retry_count: int = 0, **kwargs):
        super().__init__(
            message,
            error_code="MESSAGE_DELIVERY_ERROR",
            recovery_hint="Check agent connectivity and message queue status",
            **kwargs,
        )
        self.add_context("retry_count", retry_count)


class LearningError(MARLError):
    """Errors related to learning processes."""

    def __init__(
        self,
        message: str,
        learning_component: Optional[str] = None,
        error_code: str = "LEARNING_ERROR",
        **kwargs,
    ):
        super().__init__(message, error_code, **kwargs)
        self.learning_component = learning_component

        if learning_component:
            self.add_context("learning_component", learning_component)


class LearningDivergenceError(LearningError):
    """Learning process has diverged."""

    def __init__(
        self,
        message: str,
        divergence_metric: Optional[str] = None,
        divergence_value: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(
            message,
            error_code="LEARNING_DIVERGENCE",
            recovery_hint="Reset learning parameters and restart training",
            **kwargs,
        )
        if divergence_metric:
            self.add_context("divergence_metric", divergence_metric)
        if divergence_value is not None:
            self.add_context("divergence_value", divergence_value)


class ExperienceBufferError(LearningError):
    """Error with experience buffer operations."""

    def __init__(
        self,
        message: str,
        buffer_type: Optional[str] = None,
        buffer_size: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            message,
            learning_component="experience_buffer",
            error_code="EXPERIENCE_BUFFER_ERROR",
            recovery_hint="Check buffer capacity and memory usage",
            **kwargs,
        )
        if buffer_type:
            self.add_context("buffer_type", buffer_type)
        if buffer_size is not None:
            self.add_context("buffer_size", buffer_size)


class PerformanceError(MARLError):
    """Errors related to system performance."""

    def __init__(
        self,
        message: str,
        performance_metric: Optional[str] = None,
        threshold_value: Optional[float] = None,
        actual_value: Optional[float] = None,
        error_code: str = "PERFORMANCE_ERROR",
        **kwargs,
    ):
        super().__init__(message, error_code, **kwargs)
        self.performance_metric = performance_metric
        self.threshold_value = threshold_value
        self.actual_value = actual_value

        if performance_metric:
            self.add_context("performance_metric", performance_metric)
        if threshold_value is not None:
            self.add_context("threshold_value", threshold_value)
        if actual_value is not None:
            self.add_context("actual_value", actual_value)


class PerformanceDegradationError(PerformanceError):
    """System performance has degraded below acceptable levels."""

    def __init__(
        self, message: str, degradation_percentage: Optional[float] = None, **kwargs
    ):
        super().__init__(
            message,
            error_code="PERFORMANCE_DEGRADATION",
            recovery_hint="Analyze performance metrics and optimize system parameters",
            **kwargs,
        )
        if degradation_percentage is not None:
            self.add_context("degradation_percentage", degradation_percentage)


class ResourceExhaustionError(PerformanceError):
    """System resources have been exhausted."""

    def __init__(
        self,
        message: str,
        resource_type: str,
        usage_percentage: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(
            message,
            error_code="RESOURCE_EXHAUSTION",
            recovery_hint="Free up resources or increase system capacity",
            **kwargs,
        )
        self.add_context("resource_type", resource_type)
        if usage_percentage is not None:
            self.add_context("usage_percentage", usage_percentage)


class ConfigurationError(MARLError):
    """Errors related to system configuration."""

    def __init__(
        self,
        message: str,
        config_section: Optional[str] = None,
        config_parameter: Optional[str] = None,
        error_code: str = "CONFIGURATION_ERROR",
        **kwargs,
    ):
        super().__init__(message, error_code, **kwargs)
        self.config_section = config_section
        self.config_parameter = config_parameter

        if config_section:
            self.add_context("config_section", config_section)
        if config_parameter:
            self.add_context("config_parameter", config_parameter)


class InvalidConfigurationError(ConfigurationError):
    """Configuration parameter is invalid."""

    def __init__(
        self,
        message: str,
        expected_type: Optional[str] = None,
        actual_value: Optional[Any] = None,
        **kwargs,
    ):
        super().__init__(
            message,
            error_code="INVALID_CONFIGURATION",
            recovery_hint="Check configuration documentation and fix invalid parameters",
            **kwargs,
        )
        if expected_type:
            self.add_context("expected_type", expected_type)
        if actual_value is not None:
            self.add_context("actual_value", str(actual_value))


@dataclass
class ErrorPattern:
    """
    Represents a pattern of errors for analysis and prediction.

    Used by the error analyzer to identify recurring error patterns
    and suggest preventive measures.
    """

    pattern_id: str
    error_codes: List[str]
    frequency: int = 0
    last_occurrence: Optional[datetime] = None
    context_patterns: Dict[str, Any] = field(default_factory=dict)
    recovery_success_rate: float = 0.0

    def matches(self, error: MARLError) -> bool:
        """Check if error matches this pattern."""
        return error.error_code in self.error_codes

    def update_frequency(self) -> None:
        """Update pattern frequency and last occurrence."""
        self.frequency += 1
        self.last_occurrence = datetime.now()

    def update_recovery_success(self, success: bool) -> None:
        """Update recovery success rate."""
        current_total = self.recovery_success_rate * (self.frequency - 1)
        new_success = 1.0 if success else 0.0
        self.recovery_success_rate = (current_total + new_success) / self.frequency


@dataclass
class ErrorStatistics:
    """
    Statistics about error occurrences and recovery.

    Used for monitoring system health and identifying
    areas for improvement.
    """

    total_errors: int = 0
    errors_by_type: Dict[str, int] = field(default_factory=dict)
    errors_by_severity: Dict[str, int] = field(default_factory=dict)
    recovery_attempts: int = 0
    successful_recoveries: int = 0
    average_recovery_time: float = 0.0
    most_common_errors: List[str] = field(default_factory=list)

    def record_error(self, error: MARLError) -> None:
        """Record error occurrence."""
        self.total_errors += 1

        # Update error type count
        error_type = error.__class__.__name__
        self.errors_by_type[error_type] = self.errors_by_type.get(error_type, 0) + 1

        # Update severity count
        self.errors_by_severity[error.severity] = (
            self.errors_by_severity.get(error.severity, 0) + 1
        )

        # Update most common errors
        self._update_most_common_errors()

    def record_recovery_attempt(self, success: bool, recovery_time: float) -> None:
        """Record recovery attempt."""
        self.recovery_attempts += 1

        if success:
            self.successful_recoveries += 1

        # Update average recovery time
        current_total = self.average_recovery_time * (self.recovery_attempts - 1)
        self.average_recovery_time = (
            current_total + recovery_time
        ) / self.recovery_attempts

    def get_recovery_success_rate(self) -> float:
        """Get overall recovery success rate."""
        if self.recovery_attempts == 0:
            return 0.0
        return self.successful_recoveries / self.recovery_attempts

    def _update_most_common_errors(self) -> None:
        """Update list of most common errors."""
        sorted_errors = sorted(
            self.errors_by_type.items(), key=lambda x: x[1], reverse=True
        )
        self.most_common_errors = [error_type for error_type, _ in sorted_errors[:5]]

    def to_dict(self) -> Dict[str, Any]:
        """Convert statistics to dictionary."""
        return {
            "total_errors": self.total_errors,
            "errors_by_type": self.errors_by_type,
            "errors_by_severity": self.errors_by_severity,
            "recovery_attempts": self.recovery_attempts,
            "successful_recoveries": self.successful_recoveries,
            "recovery_success_rate": self.get_recovery_success_rate(),
            "average_recovery_time": self.average_recovery_time,
            "most_common_errors": self.most_common_errors,
        }
