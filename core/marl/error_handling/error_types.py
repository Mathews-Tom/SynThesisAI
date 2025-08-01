"""MARL Error Types.

This module defines specialized error types for the multi-agent
reinforcement learning coordination system.
"""

# Standard Library
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


class MARLError(Exception):
    """Base exception class for MARL-related errors.

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
        """Initializes a MARLError instance.

        Args:
            message: Human-readable error description.
            error_code: Unique error code for classification.
            context: Additional context information.
            recovery_hint: Suggested recovery strategy.
            severity: Error severity level (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL).
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
        """Generates a unique error ID for tracking."""
        return f"{self.error_code}_{self.timestamp.strftime('%Y%m%d_%H%M%S_%f')}"

    def to_dict(self) -> Dict[str, Any]:
        """Converts the error to a dictionary for logging and analysis.

        Returns:
            A dictionary representation of the error.
        """
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
        """Adds additional context information to the error.

        Args:
            key: The context key.
            value: The context value.
        """
        self.context[key] = value

    def get_context(self, key: str, default: Any = None) -> Any:
        """Gets a context value by its key.

        Args:
            key: The context key.
            default: The default value to return if the key is not found.

        Returns:
            The context value.
        """
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
        """Initializes an AgentError instance.

        Args:
            message: Human-readable error description.
            agent_id: The ID of the agent that caused the error.
            agent_type: The type of the agent.
            error_code: The specific error code.
            **kwargs: Additional arguments for the base MARLError.
        """
        super().__init__(message, error_code, **kwargs)
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.add_context("agent_id", agent_id)
        if agent_type:
            self.add_context("agent_type", agent_type)


class AgentInitializationError(AgentError):
    """Error during agent initialization."""

    def __init__(self, message: str, agent_id: str, **kwargs):
        """Initializes an AgentInitializationError instance.

        Args:
            message: Human-readable error description.
            agent_id: The ID of the agent that failed to initialize.
            **kwargs: Additional arguments for the base AgentError.
        """
        super().__init__(
            message,
            agent_id,
            error_code="AGENT_INIT_ERROR",
            recovery_hint="Check agent configuration and dependencies",
            **kwargs,
        )


class AgentTrainingError(AgentError):
    """Error during agent training/learning."""

    def __init__(self, message: str, agent_id: str, training_step: Optional[int] = None, **kwargs):
        """Initializes an AgentTrainingError instance.

        Args:
            message: Human-readable error description.
            agent_id: The ID of the agent that failed during training.
            training_step: The training step at which the error occurred.
            **kwargs: Additional arguments for the base AgentError.
        """
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
        """Initializes an AgentActionError instance.

        Args:
            message: Human-readable error description.
            agent_id: The ID of the agent that failed to act.
            action: The action that caused the error.
            state: The state at which the error occurred.
            **kwargs: Additional arguments for the base AgentError.
        """
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
        """Initializes a CoordinationError instance.

        Args:
            message: Human-readable error description.
            coordination_id: The ID of the coordination process.
            participating_agents: A list of agent IDs involved in the coordination.
            error_code: The specific error code.
            **kwargs: Additional arguments for the base MARLError.
        """
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
        """Initializes a CoordinationTimeoutError instance.

        Args:
            message: Human-readable error description.
            timeout_duration: The timeout duration in seconds.
            **kwargs: Additional arguments for the base CoordinationError.
        """
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
        """Initializes a CoordinationDeadlockError instance.

        Args:
            message: Human-readable error description.
            deadlock_type: The type of deadlock detected.
            **kwargs: Additional arguments for the base CoordinationError.
        """
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
        """Initializes a ConsensusError instance.

        Args:
            message: Human-readable error description.
            consensus_type: The type of consensus mechanism used.
            proposal_id: The ID of the proposal that failed consensus.
            error_code: The specific error code.
            **kwargs: Additional arguments for the base MARLError.
        """
        super().__init__(message, error_code, **kwargs)
        self.consensus_type = consensus_type
        self.proposal_id = proposal_id

        if consensus_type:
            self.add_context("consensus_type", consensus_type)
        if proposal_id:
            self.add_context("proposal_id", proposal_id)


class ConsensusFailureError(ConsensusError):
    """Failed to reach consensus."""

    def __init__(self, message: str, votes_received: int = 0, votes_required: int = 0, **kwargs):
        """Initializes a ConsensusFailureError instance.

        Args:
            message: Human-readable error description.
            votes_received: The number of votes received.
            votes_required: The number of votes required for consensus.
            **kwargs: Additional arguments for the base ConsensusError.
        """
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
        """Initializes a CommunicationError instance.

        Args:
            message: Human-readable error description.
            sender: The ID of the sending agent.
            receiver: The ID of the receiving agent.
            message_type: The type of the message.
            error_code: The specific error code.
            **kwargs: Additional arguments for the base MARLError.
        """
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
    """Failed to deliver a message between agents."""

    def __init__(self, message: str, retry_count: int = 0, **kwargs):
        """Initializes a MessageDeliveryError instance.

        Args:
            message: Human-readable error description.
            retry_count: The number of times delivery was retried.
            **kwargs: Additional arguments for the base CommunicationError.
        """
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
        """Initializes a LearningError instance.

        Args:
            message: Human-readable error description.
            learning_component: The learning component where the error occurred.
            error_code: The specific error code.
            **kwargs: Additional arguments for the base MARLError.
        """
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
        """Initializes a LearningDivergenceError instance.

        Args:
            message: Human-readable error description.
            divergence_metric: The metric used to measure divergence.
            divergence_value: The value of the divergence metric.
            **kwargs: Additional arguments for the base LearningError.
        """
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
        """Initializes an ExperienceBufferError instance.

        Args:
            message: Human-readable error description.
            buffer_type: The type of the experience buffer.
            buffer_size: The size of the experience buffer.
            **kwargs: Additional arguments for the base LearningError.
        """
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
        """Initializes a PerformanceError instance.

        Args:
            message: Human-readable error description.
            performance_metric: The performance metric that triggered the error.
            threshold_value: The threshold value for the metric.
            actual_value: The actual value of the metric.
            error_code: The specific error code.
            **kwargs: Additional arguments for the base MARLError.
        """
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

    def __init__(self, message: str, degradation_percentage: Optional[float] = None, **kwargs):
        """Initializes a PerformanceDegradationError instance.

        Args:
            message: Human-readable error description.
            degradation_percentage: The percentage of performance degradation.
            **kwargs: Additional arguments for the base PerformanceError.
        """
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
        """Initializes a ResourceExhaustionError instance.

        Args:
            message: Human-readable error description.
            resource_type: The type of resource that was exhausted.
            usage_percentage: The usage percentage of the resource.
            **kwargs: Additional arguments for the base PerformanceError.
        """
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
        """Initializes a ConfigurationError instance.

        Args:
            message: Human-readable error description.
            config_section: The configuration section where the error occurred.
            config_parameter: The configuration parameter that caused the error.
            error_code: The specific error code.
            **kwargs: Additional arguments for the base MARLError.
        """
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
        """Initializes an InvalidConfigurationError instance.

        Args:
            message: Human-readable error description.
            expected_type: The expected type of the configuration value.
            actual_value: The actual value that was provided.
            **kwargs: Additional arguments for the base ConfigurationError.
        """
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
    """Represents a pattern of errors for analysis and prediction.

    Used by the error analyzer to identify recurring error patterns
    and suggest preventive measures.

    Attributes:
        pattern_id: A unique identifier for the pattern.
        error_codes: A list of error codes that define the pattern.
        frequency: The number of times this pattern has occurred.
        last_occurrence: The timestamp of the last occurrence.
        context_patterns: A dictionary of context patterns to match against.
        recovery_success_rate: The success rate of recovery actions for this pattern.
    """

    pattern_id: str
    error_codes: List[str]
    frequency: int = 0
    last_occurrence: Optional[datetime] = None
    context_patterns: Dict[str, Any] = field(default_factory=dict)
    recovery_success_rate: float = 0.0

    def matches(self, error: MARLError) -> bool:
        """Checks if the given error matches this pattern.

        Args:
            error: The MARLError instance to check.

        Returns:
            True if the error matches the pattern, False otherwise.
        """
        return error.error_code in self.error_codes

    def update_frequency(self) -> None:
        """Updates the pattern frequency and last occurrence timestamp."""
        self.frequency += 1
        self.last_occurrence = datetime.now()

    def update_recovery_success(self, success: bool) -> None:
        """Updates the recovery success rate for this pattern.

        Args:
            success: A boolean indicating whether the recovery was successful.
        """
        current_total = self.recovery_success_rate * (self.frequency - 1)
        new_success = 1.0 if success else 0.0
        self.recovery_success_rate = (current_total + new_success) / self.frequency


@dataclass
class ErrorStatistics:
    """Statistics about error occurrences and recovery.

    Used for monitoring system health and identifying areas for improvement.

    Attributes:
        total_errors: The total number of errors recorded.
        errors_by_type: A dictionary mapping error types to their counts.
        errors_by_severity: A dictionary mapping severity levels to their counts.
        recovery_attempts: The total number of recovery attempts.
        successful_recoveries: The number of successful recovery attempts.
        average_recovery_time: The average time taken for recovery.
        most_common_errors: A list of the most common error types.
    """

    total_errors: int = 0
    errors_by_type: Dict[str, int] = field(default_factory=dict)
    errors_by_severity: Dict[str, int] = field(default_factory=dict)
    recovery_attempts: int = 0
    successful_recoveries: int = 0
    average_recovery_time: float = 0.0
    most_common_errors: List[str] = field(default_factory=list)

    def record_error(self, error: MARLError) -> None:
        """Records the occurrence of an error.

        Args:
            error: The MARLError instance to record.
        """
        self.total_errors += 1

        error_type = error.__class__.__name__
        self.errors_by_type[error_type] = self.errors_by_type.get(error_type, 0) + 1

        self.errors_by_severity[error.severity] = self.errors_by_severity.get(error.severity, 0) + 1

        self._update_most_common_errors()

    def record_recovery_attempt(self, success: bool, recovery_time: float) -> None:
        """Records a recovery attempt.

        Args:
            success: A boolean indicating whether the recovery was successful.
            recovery_time: The time taken for the recovery attempt.
        """
        self.recovery_attempts += 1

        if success:
            self.successful_recoveries += 1

        current_total = self.average_recovery_time * (self.recovery_attempts - 1)
        self.average_recovery_time = (current_total + recovery_time) / self.recovery_attempts

    def get_recovery_success_rate(self) -> float:
        """Calculates the overall recovery success rate.

        Returns:
            The recovery success rate as a float.
        """
        if self.recovery_attempts == 0:
            return 0.0
        return self.successful_recoveries / self.recovery_attempts

    def _update_most_common_errors(self) -> None:
        """Updates the list of the most common errors."""
        sorted_errors = sorted(self.errors_by_type.items(), key=lambda x: x[1], reverse=True)
        self.most_common_errors = [error_type for error_type, _ in sorted_errors[:5]]

    def to_dict(self) -> Dict[str, Any]:
        """Converts the statistics to a dictionary.

        Returns:
            A dictionary representation of the error statistics.
        """
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
