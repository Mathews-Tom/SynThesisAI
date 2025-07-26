"""
MARL Recovery Strategies.

This module provides recovery strategies for different types of errors
in the multi-agent reinforcement learning coordination system.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol

from utils.logging_config import get_logger

from .error_types import (
    AgentError,
    CommunicationError,
    ConfigurationError,
    ConsensusError,
    CoordinationError,
    LearningError,
    MARLError,
    PerformanceError,
)


class RecoveryResult:
    """Result of a recovery attempt."""

    def __init__(
        self,
        success: bool,
        strategy_name: str,
        recovery_time: float = 0.0,
        details: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ):
        self.success = success
        self.strategy_name = strategy_name
        self.recovery_time = recovery_time
        self.details = details or {}
        self.error = error

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "strategy_name": self.strategy_name,
            "recovery_time": self.recovery_time,
            "details": self.details,
            "error": self.error,
        }


class RecoveryStrategy(ABC):
    """Base class for recovery strategies."""

    def __init__(self, strategy_name: str, priority: int = 5):
        """
        Initialize recovery strategy.

        Args:
            strategy_name: Name of the recovery strategy
            priority: Priority level (1-10, higher is more important)
        """
        self.strategy_name = strategy_name
        self.priority = priority
        self.logger = get_logger(f"{__name__}.{strategy_name}")

        # Strategy statistics
        self.attempts = 0
        self.successes = 0
        self.total_recovery_time = 0.0

    @abstractmethod
    async def can_handle(self, error: MARLError) -> bool:
        """Check if this strategy can handle the given error."""
        pass

    @abstractmethod
    async def execute_recovery(self, error: MARLError) -> RecoveryResult:
        """Execute the recovery strategy."""
        pass

    def get_success_rate(self) -> float:
        """Get success rate of this strategy."""
        if self.attempts == 0:
            return 0.0
        return self.successes / self.attempts

    def get_average_recovery_time(self) -> float:
        """Get average recovery time."""
        if self.attempts == 0:
            return 0.0
        return self.total_recovery_time / self.attempts

    def record_attempt(self, success: bool, recovery_time: float) -> None:
        """Record a recovery attempt."""
        self.attempts += 1
        if success:
            self.successes += 1
        self.total_recovery_time += recovery_time

    def get_statistics(self) -> Dict[str, Any]:
        """Get strategy statistics."""
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
        super().__init__("agent_restart", priority=8)

    async def can_handle(self, error: MARLError) -> bool:
        """Check if this is an agent error that can be resolved by restart."""
        return isinstance(error, AgentError) and error.error_code in [
            "AGENT_INIT_ERROR",
            "AGENT_TRAINING_ERROR",
            "AGENT_ACTION_ERROR",
        ]

    async def execute_recovery(self, error: MARLError) -> RecoveryResult:
        """Restart the failed agent."""
        start_time = time.time()

        try:
            agent_id = error.get_context("agent_id")
            if not agent_id:
                return RecoveryResult(
                    False, self.strategy_name, error="No agent_id in error context"
                )

            self.logger.info("Attempting to restart agent: %s", agent_id)

            # Simulate agent restart process
            await self._restart_agent(agent_id)

            # Verify agent is responsive
            if await self._verify_agent_health(agent_id):
                recovery_time = time.time() - start_time
                self.record_attempt(True, recovery_time)

                return RecoveryResult(
                    True,
                    self.strategy_name,
                    recovery_time,
                    {"agent_id": agent_id, "action": "restart"},
                )
            else:
                return RecoveryResult(
                    False,
                    self.strategy_name,
                    error="Agent failed health check after restart",
                )

        except Exception as e:
            recovery_time = time.time() - start_time
            self.record_attempt(False, recovery_time)

            return RecoveryResult(
                False, self.strategy_name, recovery_time, error=str(e)
            )

    async def _restart_agent(self, agent_id: str) -> None:
        """Restart the specified agent."""
        # This would typically involve:
        # 1. Stopping the agent
        # 2. Clearing its state
        # 3. Reinitializing the agent
        # 4. Starting the agent

        # Simulate restart process
        await asyncio.sleep(2.0)
        self.logger.debug("Agent %s restarted", agent_id)

    async def _verify_agent_health(self, agent_id: str) -> bool:
        """Verify agent is healthy after restart."""
        try:
            # Simulate health check
            await asyncio.sleep(0.5)
            return True  # Assume success for now
        except Exception:
            return False


class CoordinationResetStrategy(RecoveryStrategy):
    """Strategy to reset coordination state."""

    def __init__(self):
        super().__init__("coordination_reset", priority=7)

    async def can_handle(self, error: MARLError) -> bool:
        """Check if this is a coordination error."""
        return isinstance(error, CoordinationError)

    async def execute_recovery(self, error: MARLError) -> RecoveryResult:
        """Reset coordination state."""
        start_time = time.time()

        try:
            coordination_id = error.get_context("coordination_id")
            participating_agents = error.get_context("participating_agents", [])

            self.logger.info(
                "Resetting coordination state for: %s", coordination_id or "unknown"
            )

            # Reset coordination state
            await self._reset_coordination_state(coordination_id, participating_agents)

            # Verify agents are responsive
            healthy_agents = await self._check_agent_health(participating_agents)

            recovery_time = time.time() - start_time

            if len(healthy_agents) >= 2:  # Need at least 2 agents for coordination
                self.record_attempt(True, recovery_time)

                return RecoveryResult(
                    True,
                    self.strategy_name,
                    recovery_time,
                    {
                        "coordination_id": coordination_id,
                        "healthy_agents": healthy_agents,
                        "action": "reset_coordination",
                    },
                )
            else:
                self.record_attempt(False, recovery_time)

                return RecoveryResult(
                    False,
                    self.strategy_name,
                    recovery_time,
                    error=f"Insufficient healthy agents: {len(healthy_agents)}",
                )

        except Exception as e:
            recovery_time = time.time() - start_time
            self.record_attempt(False, recovery_time)

            return RecoveryResult(
                False, self.strategy_name, recovery_time, error=str(e)
            )

    async def _reset_coordination_state(
        self, coordination_id: Optional[str], participating_agents: List[str]
    ) -> None:
        """Reset coordination state."""
        # This would typically involve:
        # 1. Clearing coordination locks
        # 2. Resetting agent coordination states
        # 3. Clearing message queues
        # 4. Reinitializing coordination protocols

        await asyncio.sleep(1.0)
        self.logger.debug("Coordination state reset for %s", coordination_id)

    async def _check_agent_health(self, agent_ids: List[str]) -> List[str]:
        """Check health of participating agents."""
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
        super().__init__("learning_reset", priority=6)

    async def can_handle(self, error: MARLError) -> bool:
        """Check if this is a learning error that requires reset."""
        return (
            isinstance(error, LearningError) and "divergence" in error.message.lower()
        )

    async def execute_recovery(self, error: MARLError) -> RecoveryResult:
        """Reset learning state."""
        start_time = time.time()

        try:
            learning_component = error.get_context("learning_component")
            agent_id = error.get_context("agent_id")

            self.logger.info(
                "Resetting learning state for component: %s, agent: %s",
                learning_component or "unknown",
                agent_id or "unknown",
            )

            # Reset learning parameters
            await self._reset_learning_parameters(agent_id, learning_component)

            # Clear experience buffers
            await self._clear_experience_buffers(agent_id)

            # Reinitialize learning state
            await self._reinitialize_learning_state(agent_id)

            recovery_time = time.time() - start_time
            self.record_attempt(True, recovery_time)

            return RecoveryResult(
                True,
                self.strategy_name,
                recovery_time,
                {
                    "agent_id": agent_id,
                    "learning_component": learning_component,
                    "action": "learning_reset",
                },
            )

        except Exception as e:
            recovery_time = time.time() - start_time
            self.record_attempt(False, recovery_time)

            return RecoveryResult(
                False, self.strategy_name, recovery_time, error=str(e)
            )

    async def _reset_learning_parameters(
        self, agent_id: Optional[str], learning_component: Optional[str]
    ) -> None:
        """Reset learning parameters to safe defaults."""
        await asyncio.sleep(0.5)
        self.logger.debug("Learning parameters reset for %s", agent_id)

    async def _clear_experience_buffers(self, agent_id: Optional[str]) -> None:
        """Clear experience buffers."""
        await asyncio.sleep(0.3)
        self.logger.debug("Experience buffers cleared for %s", agent_id)

    async def _reinitialize_learning_state(self, agent_id: Optional[str]) -> None:
        """Reinitialize learning state."""
        await asyncio.sleep(0.5)
        self.logger.debug("Learning state reinitialized for %s", agent_id)


class CommunicationRetryStrategy(RecoveryStrategy):
    """Strategy to retry failed communications."""

    def __init__(self, max_retries: int = 3):
        super().__init__("communication_retry", priority=5)
        self.max_retries = max_retries

    async def can_handle(self, error: MARLError) -> bool:
        """Check if this is a communication error."""
        return isinstance(error, CommunicationError)

    async def execute_recovery(self, error: MARLError) -> RecoveryResult:
        """Retry failed communication."""
        start_time = time.time()

        try:
            sender = error.get_context("sender")
            receiver = error.get_context("receiver")
            message_type = error.get_context("message_type")

            self.logger.info(
                "Retrying communication from %s to %s (type: %s)",
                sender or "unknown",
                receiver or "unknown",
                message_type or "unknown",
            )

            # Attempt retries
            for attempt in range(self.max_retries):
                try:
                    success = await self._retry_communication(
                        sender, receiver, message_type
                    )

                    if success:
                        recovery_time = time.time() - start_time
                        self.record_attempt(True, recovery_time)

                        return RecoveryResult(
                            True,
                            self.strategy_name,
                            recovery_time,
                            {
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
                        "Retry attempt %d failed: %s", attempt + 1, str(retry_error)
                    )

            # All retries failed
            recovery_time = time.time() - start_time
            self.record_attempt(False, recovery_time)

            return RecoveryResult(
                False,
                self.strategy_name,
                recovery_time,
                error=f"All {self.max_retries} retry attempts failed",
            )

        except Exception as e:
            recovery_time = time.time() - start_time
            self.record_attempt(False, recovery_time)

            return RecoveryResult(
                False, self.strategy_name, recovery_time, error=str(e)
            )

    async def _retry_communication(
        self,
        sender: Optional[str],
        receiver: Optional[str],
        message_type: Optional[str],
    ) -> bool:
        """Attempt to retry the communication."""
        # This would typically involve:
        # 1. Checking agent connectivity
        # 2. Resending the message
        # 3. Verifying delivery

        await asyncio.sleep(0.5)  # Simulate retry
        return True  # Assume success for now


class FallbackStrategy(RecoveryStrategy):
    """Fallback strategy for unhandled errors."""

    def __init__(self):
        super().__init__("fallback", priority=1)

    async def can_handle(self, error: MARLError) -> bool:
        """Can handle any error as fallback."""
        return True

    async def execute_recovery(self, error: MARLError) -> RecoveryResult:
        """Execute fallback recovery."""
        start_time = time.time()

        try:
            self.logger.warning(
                "Using fallback recovery for error: %s", error.error_code
            )

            # Basic fallback actions
            await self._log_error_details(error)
            await self._notify_administrators(error)
            await self._enable_safe_mode()

            recovery_time = time.time() - start_time
            self.record_attempt(True, recovery_time)

            return RecoveryResult(
                True,
                self.strategy_name,
                recovery_time,
                {
                    "error_code": error.error_code,
                    "action": "fallback_recovery",
                    "safe_mode_enabled": True,
                },
            )

        except Exception as e:
            recovery_time = time.time() - start_time
            self.record_attempt(False, recovery_time)

            return RecoveryResult(
                False, self.strategy_name, recovery_time, error=str(e)
            )

    async def _log_error_details(self, error: MARLError) -> None:
        """Log detailed error information."""
        self.logger.error(
            "Fallback recovery for error %s: %s", error.error_id, error.to_dict()
        )

    async def _notify_administrators(self, error: MARLError) -> None:
        """Notify system administrators."""
        # This would typically send alerts/notifications
        await asyncio.sleep(0.1)
        self.logger.info("Administrator notification sent for error %s", error.error_id)

    async def _enable_safe_mode(self) -> None:
        """Enable safe mode operation."""
        # This would typically:
        # 1. Disable non-essential features
        # 2. Use conservative parameters
        # 3. Enable additional monitoring

        await asyncio.sleep(0.2)
        self.logger.info("Safe mode enabled")


class RecoveryStrategyManager:
    """Manages and coordinates recovery strategies."""

    def __init__(self):
        self.logger = get_logger(__name__)

        # Initialize default strategies
        self.strategies: List[RecoveryStrategy] = [
            AgentRestartStrategy(),
            CoordinationResetStrategy(),
            LearningResetStrategy(),
            CommunicationRetryStrategy(),
            FallbackStrategy(),
        ]

        # Sort strategies by priority (highest first)
        self.strategies.sort(key=lambda s: s.priority, reverse=True)

        self.logger.info(
            "Recovery strategy manager initialized with %d strategies",
            len(self.strategies),
        )

    async def get_recovery_strategy(
        self, error: MARLError
    ) -> Optional[RecoveryStrategy]:
        """Get the best recovery strategy for the given error."""
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
                    "Error checking strategy %s: %s", strategy.strategy_name, str(e)
                )

        self.logger.warning(
            "No recovery strategy found for error: %s", error.error_code
        )
        return None

    def add_strategy(self, strategy: RecoveryStrategy) -> None:
        """Add a custom recovery strategy."""
        self.strategies.append(strategy)
        self.strategies.sort(key=lambda s: s.priority, reverse=True)

        self.logger.info(
            "Added recovery strategy: %s (priority: %d)",
            strategy.strategy_name,
            strategy.priority,
        )

    def remove_strategy(self, strategy_name: str) -> bool:
        """Remove a recovery strategy."""
        for i, strategy in enumerate(self.strategies):
            if strategy.strategy_name == strategy_name:
                del self.strategies[i]
                self.logger.info("Removed recovery strategy: %s", strategy_name)
                return True

        self.logger.warning("Strategy not found: %s", strategy_name)
        return False

    def get_strategy_statistics(self) -> List[Dict[str, Any]]:
        """Get statistics for all strategies."""
        return [strategy.get_statistics() for strategy in self.strategies]

    def get_strategy_by_name(self, strategy_name: str) -> Optional[RecoveryStrategy]:
        """Get strategy by name."""
        for strategy in self.strategies:
            if strategy.strategy_name == strategy_name:
                return strategy
        return None

    async def shutdown(self) -> None:
        """Shutdown recovery strategy manager."""
        self.logger.info("Recovery strategy manager shutdown complete")
