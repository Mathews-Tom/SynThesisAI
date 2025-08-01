"""MARL Error Handler.

This module provides the main error handling system for the multi-agent
reinforcement learning coordination system, including error classification,
recovery strategy selection, and error logging.
"""

# Standard Library
import asyncio
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Type

# SynThesisAI Modules
from utils.logging_config import get_logger

from .error_analyzer import ErrorAnalyzer
from .error_types import (
    AgentError,
    CommunicationError,
    ConfigurationError,
    ConsensusError,
    CoordinationError,
    ErrorStatistics,
    LearningError,
    MARLError,
    PerformanceError,
)
from .recovery_strategies import RecoveryStrategyManager


class MARLErrorHandler:
    """Main error handler for the MARL coordination system.

    Provides comprehensive error handling, including classification,
    recovery strategy selection, pattern recognition, and logging.

    Attributes:
        recovery_manager: Manages recovery strategies.
        error_analyzer: Analyzes error patterns.
        max_recovery_attempts: Maximum recovery attempts per error.
        recovery_timeout: Timeout for recovery operations.
        enable_pattern_learning: Flag to enable error pattern learning.
        logger: The logger for this class.
        error_statistics: Tracks error statistics.
        active_errors: A dictionary of active errors.
        recovery_history: A list of recovery attempt records.
        error_handlers: A dictionary mapping error types to handler functions.
        recovery_callbacks: A list of callbacks to notify on recovery.
    """

    def __init__(
        self,
        recovery_manager: Optional[RecoveryStrategyManager] = None,
        error_analyzer: Optional[ErrorAnalyzer] = None,
        max_recovery_attempts: int = 3,
        recovery_timeout: float = 30.0,
        enable_pattern_learning: bool = True,
    ):
        """Initializes the MARLErrorHandler.

        Args:
            recovery_manager: An optional recovery strategy manager.
            error_analyzer: An optional error pattern analyzer.
            max_recovery_attempts: The maximum number of recovery attempts.
            recovery_timeout: The timeout for a recovery operation in seconds.
            enable_pattern_learning: Whether to enable error pattern learning.
        """
        self.logger = get_logger(__name__)

        # Core components
        self.recovery_manager = recovery_manager or RecoveryStrategyManager()
        self.error_analyzer = error_analyzer or ErrorAnalyzer()

        # Configuration
        self.max_recovery_attempts = max_recovery_attempts
        self.recovery_timeout = recovery_timeout
        self.enable_pattern_learning = enable_pattern_learning

        # Error tracking
        self.error_statistics = ErrorStatistics()
        self.active_errors: Dict[str, MARLError] = {}
        self.recovery_history: List[Dict[str, Any]] = []

        # Error handlers by type
        self.error_handlers: Dict[Type[MARLError], Callable] = {
            AgentError: self._handle_agent_error,
            CoordinationError: self._handle_coordination_error,
            LearningError: self._handle_learning_error,
            CommunicationError: self._handle_communication_error,
            ConsensusError: self._handle_consensus_error,
            PerformanceError: self._handle_performance_error,
            ConfigurationError: self._handle_configuration_error,
        }

        # Recovery callbacks
        self.recovery_callbacks: List[Callable] = []

        self.logger.info("MARL error handler initialized")

    async def handle_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        source_component: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Handles an error with the appropriate recovery strategy.

        Args:
            error: The error to handle.
            context: Additional context information about the error.
            source_component: The component that generated the error.

        Returns:
            A dictionary containing the recovery result and metadata.
        """
        start_time = time.time()
        context = context or {}

        try:
            marl_error = self._convert_to_marl_error(error, context, source_component)

            self.error_statistics.record_error(marl_error)
            self.active_errors[marl_error.error_id] = marl_error

            self.logger.error(
                "Handling MARL error: %s [%s] - %s",
                marl_error.error_code,
                marl_error.error_id,
                marl_error.message,
            )

            if self.enable_pattern_learning:
                await self.error_analyzer.analyze_error(marl_error)

            recovery_result = await self._attempt_recovery(marl_error)

            recovery_time = time.time() - start_time
            self.error_statistics.record_recovery_attempt(recovery_result["success"], recovery_time)

            self._update_recovery_history(marl_error, recovery_result, recovery_time)
            await self._notify_recovery_callbacks(marl_error, recovery_result)

            if recovery_result["success"]:
                self.active_errors.pop(marl_error.error_id, None)

            return recovery_result

        except Exception as recovery_error:
            self.logger.exception("Error during error handling: %s", recovery_error)
            return {
                "success": False,
                "error_id": getattr(error, "error_id", "unknown"),
                "recovery_strategy": "none",
                "recovery_error": str(recovery_error),
                "recovery_time": time.time() - start_time,
            }

    def _convert_to_marl_error(
        self, error: Exception, context: Dict[str, Any], source_component: Optional[str]
    ) -> MARLError:
        """Converts a generic exception to a MARLError."""
        if isinstance(error, MARLError):
            if source_component:
                error.add_context("source_component", source_component)
            return error

        error_type = self._classify_error_type(error, context, source_component)
        error_message = str(error)

        error_map = {
            "agent": AgentError(
                error_message, agent_id=context.get("agent_id", "unknown"), context=context
            ),
            "coordination": CoordinationError(
                error_message,
                coordination_id=context.get("coordination_id"),
                participating_agents=context.get("participating_agents"),
                context=context,
            ),
            "learning": LearningError(
                error_message,
                learning_component=context.get("learning_component"),
                context=context,
            ),
            "communication": CommunicationError(
                error_message,
                sender=context.get("sender"),
                receiver=context.get("receiver"),
                context=context,
            ),
            "consensus": ConsensusError(
                error_message,
                consensus_type=context.get("consensus_type"),
                proposal_id=context.get("proposal_id"),
                context=context,
            ),
            "performance": PerformanceError(
                error_message,
                performance_metric=context.get("performance_metric"),
                context=context,
            ),
            "configuration": ConfigurationError(
                error_message,
                config_section=context.get("config_section"),
                config_parameter=context.get("config_parameter"),
                context=context,
            ),
        }

        return error_map.get(error_type, MARLError(error_message, context=context))

    def _classify_error_type(
        self, error: Exception, context: Dict[str, Any], source_component: Optional[str]
    ) -> str:
        """Classifies the error type based on context and error characteristics."""
        error_message = str(error).lower()

        if "agent_id" in context or (source_component and "agent" in source_component):
            return "agent"
        if "coordination_id" in context or "coordination" in error_message:
            return "coordination"
        if "learning" in error_message or "training" in error_message:
            return "learning"
        if "message" in error_message or "communication" in error_message:
            return "communication"
        if "consensus" in error_message or "vote" in error_message:
            return "consensus"
        if "performance" in error_message or "timeout" in error_message:
            return "performance"
        if "config" in error_message or "parameter" in error_message:
            return "configuration"

        if isinstance(error, (ValueError, TypeError)):
            return "configuration"
        if isinstance(error, TimeoutError):
            return "performance"
        if isinstance(error, ConnectionError):
            return "communication"

        return "generic"

    async def _attempt_recovery(self, error: MARLError) -> Dict[str, Any]:
        """Attempts to recover from an error using an appropriate strategy."""
        last_recovery_error = None

        for attempt in range(self.max_recovery_attempts):
            try:
                strategy = await self.recovery_manager.get_recovery_strategy(error)
                if not strategy:
                    self.logger.warning("No recovery strategy for error: %s", error.error_code)
                    break

                self.logger.info(
                    "Attempt %d/%d for error %s with strategy: %s",
                    attempt + 1,
                    self.max_recovery_attempts,
                    error.error_id,
                    strategy.strategy_name,
                )

                recovery_result = await asyncio.wait_for(
                    strategy.execute_recovery(error), timeout=self.recovery_timeout
                )

                if recovery_result.success:
                    self.logger.info(
                        "Recovery successful for error %s after %d attempts",
                        error.error_id,
                        attempt + 1,
                    )
                    return {
                        "success": True,
                        "error_id": error.error_id,
                        "recovery_strategy": strategy.strategy_name,
                        "recovery_attempts": attempt + 1,
                        "recovery_details": recovery_result.to_dict(),
                    }
                else:
                    last_recovery_error = recovery_result.error or "Recovery failed"
                    self.logger.warning(
                        "Recovery attempt %d failed for error %s: %s",
                        attempt + 1,
                        error.error_id,
                        last_recovery_error,
                    )

            except asyncio.TimeoutError:
                last_recovery_error = f"Recovery timeout after {self.recovery_timeout}s"
                self.logger.warning(
                    "Recovery attempt %d timed out for error %s", attempt + 1, error.error_id
                )
            except Exception as recovery_error:
                last_recovery_error = str(recovery_error)
                self.logger.exception(
                    "Recovery attempt %d for error %s failed: %s",
                    attempt + 1,
                    error.error_id,
                    last_recovery_error,
                )

            if attempt < self.max_recovery_attempts - 1:
                await asyncio.sleep(min(2 ** (attempt + 1), 10))

        self.logger.error("All recovery attempts failed for error %s", error.error_id)
        return {
            "success": False,
            "error_id": error.error_id,
            "recovery_strategy": "failed",
            "recovery_attempts": self.max_recovery_attempts,
            "last_error": last_recovery_error,
        }

    async def _handle_agent_error(self, error: AgentError) -> Dict[str, Any]:
        """Handles agent-specific errors."""
        self.logger.debug("Handling agent error for agent: %s", error.agent_id)
        recovery_actions = []
        if await self._check_agent_health(error.agent_id):
            recovery_actions.append("agent_restart")
        else:
            recovery_actions.extend(["agent_isolation", "coordination_adjustment"])
        return {
            "error_type": "agent",
            "agent_id": error.agent_id,
            "recovery_actions": recovery_actions,
        }

    async def _handle_coordination_error(self, error: CoordinationError) -> Dict[str, Any]:
        """Handles coordination-specific errors."""
        self.logger.debug("Handling coordination error: %s", error.coordination_id or "unknown")
        recovery_actions = ["reset_coordination_state"]
        if error.participating_agents:
            healthy_agents = [
                agent_id
                for agent_id in error.participating_agents
                if await self._check_agent_health(agent_id)
            ]
            if len(healthy_agents) >= 2:
                recovery_actions.append("continue_with_healthy_agents")
            else:
                recovery_actions.append("fallback_coordination")
        return {
            "error_type": "coordination",
            "coordination_id": error.coordination_id,
            "recovery_actions": recovery_actions,
        }

    async def _handle_learning_error(self, error: LearningError) -> Dict[str, Any]:
        """Handles learning-specific errors."""
        self.logger.debug(
            "Handling learning error in component: %s", error.learning_component or "unknown"
        )
        recovery_actions = ["reset_learning_state"]
        if "divergence" in error.message.lower():
            recovery_actions.extend(["adjust_learning_rate", "reset_experience_buffer"])
        elif "buffer" in error.message.lower():
            recovery_actions.extend(["clear_experience_buffer", "adjust_buffer_size"])
        return {
            "error_type": "learning",
            "learning_component": error.learning_component,
            "recovery_actions": recovery_actions,
        }

    async def _handle_communication_error(self, error: CommunicationError) -> Dict[str, Any]:
        """Handles communication-specific errors."""
        self.logger.debug(
            "Handling communication error between %s and %s",
            error.sender or "unknown",
            error.receiver or "unknown",
        )
        recovery_actions = ["retry_message_delivery"]
        if error.sender and error.receiver:
            sender_healthy = await self._check_agent_health(error.sender)
            receiver_healthy = await self._check_agent_health(error.receiver)
            if not sender_healthy or not receiver_healthy:
                recovery_actions.append("isolate_unresponsive_agents")
            else:
                recovery_actions.append("reset_communication_channel")
        return {
            "error_type": "communication",
            "sender": error.sender,
            "receiver": error.receiver,
            "recovery_actions": recovery_actions,
        }

    async def _handle_consensus_error(self, error: ConsensusError) -> Dict[str, Any]:
        """Handles consensus-specific errors."""
        self.logger.debug(
            "Handling consensus error for proposal: %s", error.proposal_id or "unknown"
        )
        recovery_actions = ["lower_consensus_threshold"]
        if "timeout" in error.message.lower():
            recovery_actions.append("extend_consensus_timeout")
        elif "failure" in error.message.lower():
            recovery_actions.extend(["use_fallback_decision", "reset_consensus_state"])
        return {
            "error_type": "consensus",
            "proposal_id": error.proposal_id,
            "recovery_actions": recovery_actions,
        }

    async def _handle_performance_error(self, error: PerformanceError) -> Dict[str, Any]:
        """Handles performance-specific errors."""
        self.logger.debug(
            "Handling performance error for metric: %s", error.performance_metric or "unknown"
        )
        recovery_actions = ["optimize_system_resources"]
        if "memory" in error.message.lower():
            recovery_actions.extend(["garbage_collection", "reduce_buffer_sizes"])
        elif "timeout" in error.message.lower():
            recovery_actions.extend(["increase_timeouts", "optimize_algorithms"])
        return {
            "error_type": "performance",
            "performance_metric": error.performance_metric,
            "recovery_actions": recovery_actions,
        }

    async def _handle_configuration_error(self, error: ConfigurationError) -> Dict[str, Any]:
        """Handles configuration-specific errors."""
        self.logger.debug(
            "Handling configuration error in section: %s", error.config_section or "unknown"
        )
        recovery_actions = ["validate_configuration"]
        if error.config_parameter:
            recovery_actions.extend(["reset_parameter_to_default", "reload_configuration"])
        return {
            "error_type": "configuration",
            "config_section": error.config_section,
            "config_parameter": error.config_parameter,
            "recovery_actions": recovery_actions,
        }

    async def _check_agent_health(self, agent_id: str) -> bool:
        """Checks if an agent is healthy and responsive."""
        try:
            # This would typically ping the agent or check its status.
            # For now, we'll simulate a health check.
            await asyncio.sleep(0.1)  # Simulate health check delay
            return True  # Assume healthy for this simulation
        except Exception:
            return False

    def _update_recovery_history(
        self, error: MARLError, recovery_result: Dict[str, Any], recovery_time: float
    ) -> None:
        """Updates the recovery history for analysis."""
        history_entry = {
            "timestamp": datetime.now().isoformat(),
            "error_id": error.error_id,
            "error_code": error.error_code,
            "error_type": error.__class__.__name__,
            "recovery_success": recovery_result["success"],
            "recovery_strategy": recovery_result.get("recovery_strategy", "unknown"),
            "recovery_attempts": recovery_result.get("recovery_attempts", 0),
            "recovery_time": recovery_time,
            "context": error.context,
        }
        self.recovery_history.append(history_entry)
        if len(self.recovery_history) > 1000:
            self.recovery_history = self.recovery_history[-1000:]

    async def _notify_recovery_callbacks(
        self, error: MARLError, recovery_result: Dict[str, Any]
    ) -> None:
        """Notifies registered recovery callbacks."""
        for callback in self.recovery_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(error, recovery_result)
                else:
                    callback(error, recovery_result)
            except Exception as callback_error:
                self.logger.warning("Recovery callback failed: %s", callback_error)

    def add_recovery_callback(self, callback: Callable) -> None:
        """Adds a callback to be notified of recovery attempts."""
        self.recovery_callbacks.append(callback)

    def remove_recovery_callback(self, callback: Callable) -> None:
        """Removes a recovery callback."""
        try:
            self.recovery_callbacks.remove(callback)
        except ValueError:
            self.logger.warning("Attempted to remove a non-existent callback.")

    def get_error_statistics(self) -> Dict[str, Any]:
        """Gets the current error statistics."""
        return self.error_statistics.to_dict()

    def get_active_errors(self) -> List[Dict[str, Any]]:
        """Gets a list of currently active errors."""
        return [error.to_dict() for error in self.active_errors.values()]

    def get_recovery_history(
        self, limit: Optional[int] = None, since: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Gets the recovery history.

        Args:
            limit: The maximum number of entries to return.
            since: The timestamp to return entries after.

        Returns:
            A list of recovery history entries.
        """
        history = self.recovery_history
        if since:
            history = [
                entry for entry in history if datetime.fromisoformat(entry["timestamp"]) >= since
            ]
        if limit is not None:
            history = history[-limit:] if limit > 0 else []
        return history

    def clear_error_history(self) -> None:
        """Clears the error and recovery history."""
        self.recovery_history.clear()
        self.error_statistics = ErrorStatistics()
        self.logger.info("Error history cleared")

    async def shutdown(self) -> None:
        """Shuts down the error handler and cleans up resources."""
        self.logger.info("Shutting down MARL error handler")
        self.active_errors.clear()
        self.recovery_callbacks.clear()
        if hasattr(self.recovery_manager, "shutdown"):
            await self.recovery_manager.shutdown()
        if hasattr(self.error_analyzer, "shutdown"):
            await self.error_analyzer.shutdown()
        self.logger.info("MARL error handler shutdown complete")


class ErrorHandlerFactory:
    """Factory for creating configured error handlers."""

    @staticmethod
    def create_default() -> MARLErrorHandler:
        """Creates an error handler with the default configuration."""
        return MARLErrorHandler()

    @staticmethod
    def create_with_config(config: Dict[str, Any]) -> MARLErrorHandler:
        """Creates an error handler with a custom configuration."""
        return MARLErrorHandler(
            max_recovery_attempts=config.get("max_recovery_attempts", 3),
            recovery_timeout=config.get("recovery_timeout", 30.0),
            enable_pattern_learning=config.get("enable_pattern_learning", True),
        )

    @staticmethod
    def create_for_testing() -> MARLErrorHandler:
        """Creates an error handler optimized for testing."""
        return MARLErrorHandler(
            max_recovery_attempts=1, recovery_timeout=5.0, enable_pattern_learning=False
        )
