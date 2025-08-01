"""MARL Recovery Strategies.

This module provides recovery strategies for different types of errors
in the multi-agent reinforcement learning coordination system.
"""

# Standard Library
import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# Third-Party Library
# (No third-party libraries in this file)

# SynThesisAI Modules
from utils.logging_config import get_logger
from .error_types import (
    AgentError,
    CommunicationError,
    CoordinationError,
    LearningError,
    MARLError,
)


@dataclass
class RecoveryResult:
    """Result of a recovery attempt.

    Attributes:
        success: Whether the recovery was successful.
        strategy_name: The name of the strategy used.
        recovery_time: The time taken for the recovery in seconds.
        details: A dictionary with additional details about the recovery.
        error: A string describing an error if the recovery failed.
    """

    success: bool
    strategy_name: str
    recovery_time: float = 0.0
    details: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    def __post_init__(self):
        """Post-initialization to set default for details."""
        if self.details is None:
            self.details = {}

    def to_dict(self) -> Dict[str, Any]:
        """Converts the RecoveryResult to a dictionary.

        Returns:
            A dictionary representation of the recovery result.
        """
        return {
            "success": self.success,
            "strategy_name": self.strategy_name,
            "recovery_time": self.recovery_time,
            "details": self.details,
            "error": self.error,
        }


class RecoveryStrategy(ABC):
    """Base class for recovery strategies.

    Attributes:
        strategy_name: Name of the recovery strategy.
        priority: Priority level (1-10, higher is more important).
        logger: The logger for this strategy.
        attempts: The number of times this strategy has been attempted.
        successes: The number of successful attempts.
        total_recovery_time: The cumulative time spent in recovery.
    """

    def __init__(self, strategy_name: str, priority: int = 5):
        """Initializes the recovery strategy.

        Args:
            strategy_name: Name of the recovery strategy.
            priority: Priority level (1-10, higher is more important).
        """
        self.strategy_name = strategy_name
        self.priority = priority
        self.logger = get_logger(f"{__name__}.{self.strategy_name}")

        # Strategy statistics
        self.attempts = 0
        self.successes = 0
        self.total_recovery_time = 0.0

    @abstractmethod
    async def can_handle(self, error: MARLError) -> bool:
        """Checks if this strategy can handle the given error.

        Args:
            error: The MARLError to check.

        Returns:
            True if the strategy can handle the error, False otherwise.
        """
        pass

    @abstractmethod
    async def execute_recovery(self, error: MARLError) -> RecoveryResult:
        """Executes the recovery strategy.

        Args:
            error: The MARLError to handle.

        Returns:
            A RecoveryResult object detailing the outcome.
        """
        pass

    def get_success_rate(self) -> float:
        """Gets the success rate of this strategy.

        Returns:
            The success rate as a float between 0.0 and 1.0.
        """
        if self.attempts == 0:
            return 0.0
        return self.successes / self.attempts

    def get_average_recovery_time(self) -> float:
        """Gets the average recovery time for this strategy.

        Returns:
            The average recovery time in seconds.
        """
        if self.attempts == 0:
            return 0.0
        return self.total_recovery_time / self.attempts

    def record_attempt(self, success: bool, recovery_time: float) -> None:
        """Records a recovery attempt.

        Args:
            success: Whether the attempt was successful.
            recovery_time: The time taken for the attempt.
        """
        self.attempts += 1
        if success:
            self.successes += 1
        self.total_recovery_time += recovery_time

    def get_statistics(self) -> Dict[str, Any]:
        """Gets statistics for this strategy.

        Returns:
            A dictionary containing strategy statistics.
        """
        return {
            "strategy_name": self.strategy_name,
            "priority": self.priority,
            "attempts": self.attempts,
            "successes": self.successes,
            "success_rate": self.get_success_rate(),
            "average_recovery_time": self.get_average_recovery_time(),
        }


class AgentRestartStrategy(RecoveryStrategy):
    """Strategy to restart failed agents."""

    def __init__(self):
        """Initializes the AgentRestartStrategy."""
        super().__init__("agent_restart", priority=8)

    async def can_handle(self, error: MARLError) -> bool:
        """Checks if this is an agent error that can be resolved by restart.

        Args:
            error: The MARLError to check.

        Returns:
            True if the error is a restartable AgentError, False otherwise.
        """
        return isinstance(error, AgentError) and error.error_code in {
            "AGENT_INIT_ERROR",
            "AGENT_TRAINING_ERROR",
            "AGENT_ACTION_ERROR",
        }

    async def execute_recovery(self, error: MARLError) -> RecoveryResult:
        """Restarts the failed agent.

        Args:
            error: The AgentError to handle.

        Returns:
            A RecoveryResult object detailing the outcome.
        """
        start_time = time.time()
        agent_id = error.get_context("agent_id")

        if not agent_id:
            return RecoveryResult(
                success=False,
                strategy_name=self.strategy_name,
                error="No agent_id in error context",
            )

        self.logger.info("Attempting to restart agent: %s", agent_id)
        try:
            # Simulate agent restart process
            await self._restart_agent(agent_id)

            # Verify agent is responsive
            if await self._verify_agent_health(agent_id):
                recovery_time = time.time() - start_time
                self.record_attempt(True, recovery_time)
                return RecoveryResult(
                    success=True,
                    strategy_name=self.strategy_name,
                    recovery_time=recovery_time,
                    details={"agent_id": agent_id, "action": "restart"},
                )
            else:
                recovery_time = time.time() - start_time
                self.record_attempt(False, recovery_time)
                return RecoveryResult(
                    success=False,
                    strategy_name=self.strategy_name,
                    recovery_time=recovery_time,
                    error="Agent failed health check after restart",
                )
        except Exception as e:
            recovery_time = time.time() - start_time
            self.record_attempt(False, recovery_time)
            self.logger.error("Error during agent restart for %s: %s", agent_id, e, exc_info=True)
            return RecoveryResult(
                success=False,
                strategy_name=self.strategy_name,
                recovery_time=recovery_time,
                error=str(e),
            )

    async def _restart_agent(self, agent_id: str) -> None:
        """Simulates restarting the specified agent.

        Args:
            agent_id: The ID of the agent to restart.
        """
        # This would typically involve:
        # 1. Stopping the agent
        # 2. Clearing its state
        # 3. Reinitializing the agent
        # 4. Starting the agent
        self.logger.debug("Simulating restart for agent %s...", agent_id)
        await asyncio.sleep(2.0)
        self.logger.debug("Agent %s restarted", agent_id)

    async def _verify_agent_health(self, agent_id: str) -> bool:
        """Verifies agent is healthy after restart.

        Args:
            agent_id: The ID of the agent to check.

        Returns:
            True if the agent is healthy, False otherwise.
        """
        try:
            # Simulate health check
            self.logger.debug("Verifying health of agent %s...", agent_id)
            await asyncio.sleep(0.5)
            return True  # Assume success for now
        except Exception as e:
            self.logger.warning("Health check failed for agent %s: %s", agent_id, e, exc_info=True)
            return False


class CoordinationResetStrategy(RecoveryStrategy):
    """Strategy to reset coordination state."""

    def __init__(self):
        """Initializes the CoordinationResetStrategy."""
        super().__init__("coordination_reset", priority=7)

    async def can_handle(self, error: MARLError) -> bool:
        """Checks if this is a coordination error.

        Args:
            error: The MARLError to check.

        Returns:
            True if the error is a CoordinationError, False otherwise.
        """
        return isinstance(error, CoordinationError)

    async def execute_recovery(self, error: MARLError) -> RecoveryResult:
        """Resets coordination state.

        Args:
            error: The CoordinationError to handle.

        Returns:
            A RecoveryResult object detailing the outcome.
        """
        start_time = time.time()
        coordination_id = error.get_context("coordination_id")
        participating_agents = error.get_context("participating_agents", [])

        self.logger.info("Resetting coordination state for: %s", coordination_id or "unknown")
        try:
            # Reset coordination state
            await self._reset_coordination_state(coordination_id, participating_agents)

            # Verify agents are responsive
            healthy_agents = await self._check_agent_health(participating_agents)
            recovery_time = time.time() - start_time

            if len(healthy_agents) >= 2:  # Need at least 2 agents for coordination
                self.record_attempt(True, recovery_time)
                return RecoveryResult(
                    success=True,
                    strategy_name=self.strategy_name,
                    recovery_time=recovery_time,
                    details={
                        "coordination_id": coordination_id,
                        "healthy_agents": healthy_agents,
                        "action": "reset_coordination",
                    },
                )
            else:
                self.record_attempt(False, recovery_time)
                return RecoveryResult(
                    success=False,
                    strategy_name=self.strategy_name,
                    recovery_time=recovery_time,
                    error=f"Insufficient healthy agents: {len(healthy_agents)}",
                )
        except Exception as e:
            recovery_time = time.time() - start_time
            self.record_attempt(False, recovery_time)
            self.logger.error(
                "Error during coordination reset for %s: %s",
                coordination_id or "unknown",
                e,
                exc_info=True,
            )
            return RecoveryResult(
                success=False,
                strategy_name=self.strategy_name,
                recovery_time=recovery_time,
                error=str(e),
            )

    async def _reset_coordination_state(
        self, coordination_id: Optional[str], participating_agents: List[str]
    ) -> None:
        """Simulates resetting coordination state.

        Args:
            coordination_id: The ID of the coordination group.
            participating_agents: A list of agent IDs involved.
        """
        # This would typically involve:
        # 1. Clearing coordination locks
        # 2. Resetting agent coordination states
        # 3. Clearing message queues
        # 4. Reinitializing coordination protocols
        self.logger.debug(
            "Simulating coordination state reset for %s", coordination_id or "all agents"
        )
        await asyncio.sleep(1.0)
        self.logger.debug("Coordination state reset for %s", coordination_id)

    async def _check_agent_health(self, agent_ids: List[str]) -> List[str]:
        """Checks health of participating agents.

        Args:
            agent_ids: A list of agent IDs to check.

        Returns:
            A list of healthy agent IDs.
        """
        healthy_agents = []
        for agent_id in agent_ids:
            try:
                # Simulate health check
                await asyncio.sleep(0.1)
                healthy_agents.append(agent_id)
            except Exception:
                self.logger.warning("Agent %s failed health check", agent_id)
        return healthy_agents


class LearningResetStrategy(RecoveryStrategy):
    """Strategy to reset learning state when divergence occurs."""

    def __init__(self):
        """Initializes the LearningResetStrategy."""
        super().__init__("learning_reset", priority=6)

    async def can_handle(self, error: MARLError) -> bool:
        """Checks if this is a learning error that requires reset.

        Args:
            error: The MARLError to check.

        Returns:
            True if the error is a LearningError indicating divergence.
        """
        return isinstance(error, LearningError) and "divergence" in error.message.lower()

    async def execute_recovery(self, error: MARLError) -> RecoveryResult:
        """Resets learning state.

        Args:
            error: The LearningError to handle.

        Returns:
            A RecoveryResult object detailing the outcome.
        """
        start_time = time.time()
        learning_component = error.get_context("learning_component")
        agent_id = error.get_context("agent_id")

        self.logger.info(
            "Resetting learning state for component: %s, agent: %s",
            learning_component or "unknown",
            agent_id or "unknown",
        )
        try:
            # Reset learning parameters
            await self._reset_learning_parameters(agent_id, learning_component)
            # Clear experience buffers
            await self._clear_experience_buffers(agent_id)
            # Reinitialize learning state
            await self._reinitialize_learning_state(agent_id)

            recovery_time = time.time() - start_time
            self.record_attempt(True, recovery_time)
            return RecoveryResult(
                success=True,
                strategy_name=self.strategy_name,
                recovery_time=recovery_time,
                details={
                    "agent_id": agent_id,
                    "learning_component": learning_component,
                    "action": "learning_reset",
                },
            )
        except Exception as e:
            recovery_time = time.time() - start_time
            self.record_attempt(False, recovery_time)
            self.logger.error(
                "Error during learning reset for agent %s: %s",
                agent_id or "unknown",
                e,
                exc_info=True,
            )
            return RecoveryResult(
                success=False,
                strategy_name=self.strategy_name,
                recovery_time=recovery_time,
                error=str(e),
            )

    async def _reset_learning_parameters(
        self, agent_id: Optional[str], learning_component: Optional[str]
    ) -> None:
        """Simulates resetting learning parameters to safe defaults."""
        self.logger.debug(
            "Resetting learning parameters for component %s of agent %s",
            learning_component or "all",
            agent_id or "unknown",
        )
        await asyncio.sleep(0.5)
        self.logger.debug("Learning parameters reset for %s", agent_id)

    async def _clear_experience_buffers(self, agent_id: Optional[str]) -> None:
        """Simulates clearing experience buffers."""
        self.logger.debug("Clearing experience buffers for %s", agent_id or "unknown")
        await asyncio.sleep(0.3)
        self.logger.debug("Experience buffers cleared for %s", agent_id)

    async def _reinitialize_learning_state(self, agent_id: Optional[str]) -> None:
        """Simulates reinitializing learning state."""
        self.logger.debug("Reinitializing learning state for %s", agent_id or "unknown")
        await asyncio.sleep(0.5)
        self.logger.debug("Learning state reinitialized for %s", agent_id)


class CommunicationRetryStrategy(RecoveryStrategy):
    """Strategy to retry failed communications."""

    def __init__(self, max_retries: int = 3):
        """Initializes the CommunicationRetryStrategy.

        Args:
            max_retries: The maximum number of retry attempts.
        """
        super().__init__("communication_retry", priority=5)
        self.max_retries = max_retries

    async def can_handle(self, error: MARLError) -> bool:
        """Checks if this is a communication error.

        Args:
            error: The MARLError to check.

        Returns:
            True if the error is a CommunicationError, False otherwise.
        """
        return isinstance(error, CommunicationError)

    async def execute_recovery(self, error: MARLError) -> RecoveryResult:
        """Retries failed communication.

        Args:
            error: The CommunicationError to handle.

        Returns:
            A RecoveryResult object detailing the outcome.
        """
        start_time = time.time()
        sender = error.get_context("sender")
        receiver = error.get_context("receiver")
        message_type = error.get_context("message_type")

        self.logger.info(
            "Retrying communication from %s to %s (type: %s)",
            sender or "unknown",
            receiver or "unknown",
            message_type or "unknown",
        )

        for attempt in range(self.max_retries):
            try:
                success = await self._retry_communication(sender, receiver, message_type)
                if success:
                    recovery_time = time.time() - start_time
                    self.record_attempt(True, recovery_time)
                    return RecoveryResult(
                        success=True,
                        strategy_name=self.strategy_name,
                        recovery_time=recovery_time,
                        details={
                            "sender": sender,
                            "receiver": receiver,
                            "message_type": message_type,
                            "retry_attempt": attempt + 1,
                            "action": "communication_retry",
                        },
                    )
                # Wait before next retry
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2**attempt)  # Exponential backoff
            except Exception as retry_error:
                self.logger.warning(
                    "Retry attempt %d failed: %s", attempt + 1, retry_error, exc_info=True
                )

        # All retries failed
        recovery_time = time.time() - start_time
        self.record_attempt(False, recovery_time)
        return RecoveryResult(
            success=False,
            strategy_name=self.strategy_name,
            recovery_time=recovery_time,
            error=f"All {self.max_retries} retry attempts failed",
        )

    async def _retry_communication(
        self,
        sender: Optional[str],
        receiver: Optional[str],
        message_type: Optional[str],
    ) -> bool:
        """Simulates retrying the communication.

        Args:
            sender: The ID of the sending agent.
            receiver: The ID of the receiving agent.
            message_type: The type of message being sent.

        Returns:
            True if the communication succeeds, False otherwise.
        """
        # This would typically involve:
        # 1. Checking agent connectivity
        # 2. Resending the message
        # 3. Verifying delivery
        self.logger.debug("Attempting to resend message from %s to %s", sender, receiver)
        await asyncio.sleep(0.5)  # Simulate retry
        return True  # Assume success for now


class FallbackStrategy(RecoveryStrategy):
    """Fallback strategy for unhandled errors."""

    def __init__(self):
        """Initializes the FallbackStrategy."""
        super().__init__("fallback", priority=1)

    async def can_handle(self, error: MARLError) -> bool:
        """Can handle any error as a fallback.

        Args:
            error: The MARLError to check.

        Returns:
            Always returns True.
        """
        return True

    async def execute_recovery(self, error: MARLError) -> RecoveryResult:
        """Executes fallback recovery.

        Args:
            error: The MARLError to handle.

        Returns:
            A RecoveryResult object detailing the outcome.
        """
        start_time = time.time()
        try:
            self.logger.warning("Using fallback recovery for error: %s", error.error_code)
            # Basic fallback actions
            await self._log_error_details(error)
            await self._notify_administrators(error)
            await self._enable_safe_mode()

            recovery_time = time.time() - start_time
            self.record_attempt(True, recovery_time)
            return RecoveryResult(
                success=True,
                strategy_name=self.strategy_name,
                recovery_time=recovery_time,
                details={
                    "error_code": error.error_code,
                    "action": "fallback_recovery",
                    "safe_mode_enabled": True,
                },
            )
        except Exception as e:
            recovery_time = time.time() - start_time
            self.record_attempt(False, recovery_time)
            self.logger.error("Fallback strategy failed: %s", e, exc_info=True)
            return RecoveryResult(
                success=False,
                strategy_name=self.strategy_name,
                recovery_time=recovery_time,
                error=str(e),
            )

    async def _log_error_details(self, error: MARLError) -> None:
        """Logs detailed error information."""
        self.logger.error("Fallback recovery for error %s: %s", error.error_id, error.to_dict())

    async def _notify_administrators(self, error: MARLError) -> None:
        """Simulates notifying system administrators."""
        # This would typically send alerts/notifications
        self.logger.info("Notifying administrators about error %s", error.error_id)
        await asyncio.sleep(0.1)
        self.logger.info("Administrator notification sent for error %s", error.error_id)

    async def _enable_safe_mode(self) -> None:
        """Simulates enabling safe mode operation."""
        # This would typically:
        # 1. Disable non-essential features
        # 2. Use conservative parameters
        # 3. Enable additional monitoring
        self.logger.info("Enabling safe mode...")
        await asyncio.sleep(0.2)
        self.logger.info("Safe mode enabled")


class RecoveryStrategyManager:
    """Manages and coordinates recovery strategies.

    Attributes:
        logger: The logger for this manager.
        strategies: A list of available recovery strategies.
    """

    def __init__(self):
        """Initializes the RecoveryStrategyManager."""
        self.logger = get_logger(__name__)
        self.strategies: List[RecoveryStrategy] = self._initialize_strategies()
        self._sort_strategies()
        self.logger.info(
            "Recovery strategy manager initialized with %d strategies",
            len(self.strategies),
        )

    def _initialize_strategies(self) -> List[RecoveryStrategy]:
        """Initializes the default set of recovery strategies.

        Returns:
            A list of default recovery strategy instances.
        """
        return [
            AgentRestartStrategy(),
            CoordinationResetStrategy(),
            LearningResetStrategy(),
            CommunicationRetryStrategy(),
            FallbackStrategy(),
        ]

    def _sort_strategies(self) -> None:
        """Sorts strategies by priority (highest first)."""
        self.strategies.sort(key=lambda s: s.priority, reverse=True)

    async def get_recovery_strategy(self, error: MARLError) -> Optional[RecoveryStrategy]:
        """Gets the best recovery strategy for the given error.

        Args:
            error: The MARLError to handle.

        Returns:
            The highest-priority strategy that can handle the error, or None.
        """
        for strategy in self.strategies:
            try:
                if await strategy.can_handle(error):
                    self.logger.debug(
                        "Selected recovery strategy: %s for error: %s",
                        strategy.strategy_name,
                        error.error_code,
                    )
                    return strategy
            except Exception as e:
                self.logger.warning(
                    "Error checking strategy %s: %s", strategy.strategy_name, e, exc_info=True
                )
        self.logger.warning("No recovery strategy found for error: %s", error.error_code)
        return None

    def add_strategy(self, strategy: RecoveryStrategy) -> None:
        """Adds a custom recovery strategy and resorts the list.

        Args:
            strategy: The recovery strategy to add.
        """
        self.strategies.append(strategy)
        self._sort_strategies()
        self.logger.info(
            "Added recovery strategy: %s (priority: %d)",
            strategy.strategy_name,
            strategy.priority,
        )

    def remove_strategy(self, strategy_name: str) -> bool:
        """Removes a recovery strategy by its name.

        Args:
            strategy_name: The name of the strategy to remove.

        Returns:
            True if the strategy was found and removed, False otherwise.
        """
        strategy_to_remove = self.get_strategy_by_name(strategy_name)
        if strategy_to_remove:
            self.strategies.remove(strategy_to_remove)
            self.logger.info("Removed recovery strategy: %s", strategy_name)
            return True
        self.logger.warning("Strategy not found for removal: %s", strategy_name)
        return False

    def get_strategy_statistics(self) -> List[Dict[str, Any]]:
        """Gets statistics for all managed strategies.

        Returns:
            A list of dictionaries, each containing statistics for a strategy.
        """
        return [strategy.get_statistics() for strategy in self.strategies]

    def get_strategy_by_name(self, strategy_name: str) -> Optional[RecoveryStrategy]:
        """Gets a strategy by its name.

        Args:
            strategy_name: The name of the strategy to find.

        Returns:
            The strategy instance if found, otherwise None.
        """
        for strategy in self.strategies:
            if strategy.strategy_name == strategy_name:
                return strategy
        return None

    async def shutdown(self) -> None:
        """Shuts down the recovery strategy manager."""
        self.logger.info("Recovery strategy manager shutdown complete.")
