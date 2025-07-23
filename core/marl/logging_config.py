"""
MARL Logging Configuration

This module provides specialized logging configuration for multi-agent reinforcement
learning components, following the development standards for proper logging with
lazy % formatting and structured log messages.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from .config import MARLConfig


class MARLLogger:
    """Specialized logger for MARL components with structured logging."""

    def __init__(self, name: str, config: Optional[MARLConfig] = None):
        """
        Initialize MARL logger.

        Args:
            name: Logger name
            config: MARL configuration for logging settings
        """
        self.logger = logging.getLogger(name)
        self.config = config

        if config:
            self._configure_logger()

    def _configure_logger(self) -> None:
        """Configure logger based on MARL configuration."""
        # Set log level
        log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        self.logger.setLevel(log_level)

        # Create formatter with MARL-specific format
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - [MARL] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Console handler
        if not self.logger.handlers:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

            # File handler for debug mode
            if self.config.debug_mode:
                log_dir = Path("logs/marl")
                log_dir.mkdir(parents=True, exist_ok=True)

                log_file = (
                    log_dir / f"marl_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
                )
                file_handler = logging.FileHandler(log_file)
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)

    def log_coordination_start(self, request_id: str, agents: list) -> None:
        """Log coordination start with proper lazy formatting."""
        self.logger.info(
            "Starting coordination for request %s with agents: %s",
            request_id,
            ", ".join(agents),
        )

    def log_coordination_success(
        self, request_id: str, coordination_time: float, success_rate: float
    ) -> None:
        """Log successful coordination with metrics."""
        self.logger.info(
            "Coordination completed for request %s in %.2fs (success rate: %.2f)",
            request_id,
            coordination_time,
            success_rate,
        )

    def log_coordination_failure(
        self, request_id: str, error_type: str, failure_reason: str
    ) -> None:
        """Log coordination failure with error details."""
        self.logger.error(
            "Coordination failed for request %s - %s: %s",
            request_id,
            error_type,
            failure_reason,
        )

    def log_agent_action(
        self, agent_id: str, action: str, confidence: float, state_summary: str
    ) -> None:
        """Log agent action selection with context."""
        self.logger.debug(
            "Agent %s selected action '%s' (confidence: %.2f) - State: %s",
            agent_id,
            action,
            confidence,
            state_summary,
        )

    def log_learning_update(
        self, agent_id: str, episode: int, reward: float, loss: float, epsilon: float
    ) -> None:
        """Log agent learning update with metrics."""
        self.logger.debug(
            "Agent %s learning update - Episode: %d, Reward: %.3f, Loss: %.4f, Epsilon: %.3f",
            agent_id,
            episode,
            reward,
            loss,
            epsilon,
        )

    def log_performance_metrics(self, metrics: Dict[str, float]) -> None:
        """Log performance metrics with proper formatting."""
        metrics_str = ", ".join([f"{k}: {v:.3f}" for k, v in metrics.items()])
        self.logger.info("Performance metrics - %s", metrics_str)

    def log_error_with_context(self, error: Exception, context: Dict[str, Any]) -> None:
        """Log error with full context information."""
        context_str = ", ".join([f"{k}: {v}" for k, v in context.items()])
        self.logger.error(
            "MARL error occurred: %s - Context: %s",
            str(error),
            context_str,
            exc_info=True,
        )

    def log_consensus_building(self, proposals: Dict[str, Any], strategy: str) -> None:
        """Log consensus building process."""
        agent_count = len(proposals)
        self.logger.debug(
            "Building consensus from %d agent proposals using %s strategy",
            agent_count,
            strategy,
        )

    def log_experience_sharing(
        self, agent_id: str, experience_type: str, reward: float, shared: bool
    ) -> None:
        """Log experience sharing decisions."""
        action = "shared" if shared else "kept local"
        self.logger.debug(
            "Agent %s %s %s experience (reward: %.3f)",
            agent_id,
            action,
            experience_type,
            reward,
        )

    def log_checkpoint_save(
        self, checkpoint_path: str, episode: int, agents_saved: list
    ) -> None:
        """Log checkpoint saving."""
        self.logger.info(
            "Saved checkpoint at episode %d to %s for agents: %s",
            episode,
            checkpoint_path,
            ", ".join(agents_saved),
        )

    def log_distributed_training(
        self, worker_id: int, node_count: int, sync_status: str
    ) -> None:
        """Log distributed training status."""
        self.logger.info(
            "Distributed training - Worker %d/%d, Sync status: %s",
            worker_id,
            node_count,
            sync_status,
        )


def setup_marl_logging(config: MARLConfig) -> Dict[str, MARLLogger]:
    """
    Set up logging for all MARL components.

    Args:
        config: MARL configuration

    Returns:
        Dictionary of configured loggers for different components
    """
    loggers = {}

    # Main MARL components
    component_names = [
        "marl.coordinator",
        "marl.generator_agent",
        "marl.validator_agent",
        "marl.curriculum_agent",
        "marl.coordination_policy",
        "marl.consensus_mechanism",
        "marl.communication_protocol",
        "marl.experience_manager",
        "marl.performance_monitor",
        "marl.error_handler",
    ]

    for component_name in component_names:
        loggers[component_name] = MARLLogger(component_name, config)

    return loggers


def get_marl_logger(name: str, config: Optional[MARLConfig] = None) -> MARLLogger:
    """
    Get a MARL logger instance.

    Args:
        name: Logger name
        config: Optional MARL configuration

    Returns:
        Configured MARL logger
    """
    return MARLLogger(f"marl.{name}", config)
