"""
MARL Optimization Infrastructure

This module provides advanced optimization algorithms and learning rate scheduling
for multi-agent reinforcement learning, following the development standards for
comprehensive optimization with proper error handling and logging.
"""

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ..config import AgentConfig
from ..exceptions import LearningDivergenceError, OptimizationFailureError
from ..logging_config import get_marl_logger

logger = logging.getLogger(__name__)


@dataclass
class OptimizationMetrics:
    """Metrics for tracking optimization performance."""

    total_updates: int = 0
    average_loss: float = 0.0
    loss_variance: float = 0.0
    gradient_norm: float = 0.0
    learning_rate: float = 0.0
    convergence_score: float = 0.0
    recent_losses: List[float] = field(default_factory=list)

    def update(self, loss: float, grad_norm: float, lr: float) -> None:
        """Update optimization metrics."""
        self.total_updates += 1
        self.recent_losses.append(loss)

        # Keep only recent losses for efficiency
        if len(self.recent_losses) > 1000:
            self.recent_losses = self.recent_losses[-1000:]

        # Update running averages
        self.average_loss = np.mean(self.recent_losses)
        self.loss_variance = (
            np.var(self.recent_losses) if len(self.recent_losses) > 1 else 0.0
        )
        self.gradient_norm = grad_norm
        self.learning_rate = lr

        # Calculate convergence score (lower variance = better convergence)
        if self.loss_variance > 0:
            self.convergence_score = 1.0 / (1.0 + self.loss_variance)
        else:
            self.convergence_score = 1.0

    def get_summary(self) -> Dict[str, Any]:
        """Get optimization metrics summary."""
        return {
            "total_updates": self.total_updates,
            "average_loss": self.average_loss,
            "loss_variance": self.loss_variance,
            "gradient_norm": self.gradient_norm,
            "learning_rate": self.learning_rate,
            "convergence_score": self.convergence_score,
            "recent_loss_count": len(self.recent_losses),
        }


class MARLOptimizer:
    """
    Advanced optimizer for MARL agents with adaptive learning and stability monitoring.

    Provides enhanced optimization capabilities including gradient clipping,
    learning rate scheduling, and convergence monitoring for stable training.
    """

    def __init__(self, network: nn.Module, config: AgentConfig, agent_id: str):
        """
        Initialize MARL optimizer.

        Args:
            network: Neural network to optimize
            config: Agent configuration
            agent_id: ID of the agent being optimized
        """
        self.network = network
        self.config = config
        self.agent_id = agent_id
        self.logger = get_marl_logger(f"optimizer.{agent_id}")

        # Create base optimizer
        self.optimizer = self._create_optimizer()

        # Optimization tracking
        self.metrics = OptimizationMetrics()
        self.step_count = 0

        # Stability monitoring
        self.divergence_threshold = 10.0  # Loss threshold for divergence detection
        self.gradient_clip_value = 1.0

        self.logger.log_agent_action(
            agent_id,
            "optimizer_initialized",
            1.0,
            f"Type: {type(self.optimizer).__name__}, LR: {config.learning_rate}",
        )

    def _create_optimizer(self) -> optim.Optimizer:
        """Create the base optimizer."""
        optimizer_type = getattr(self.config, "optimizer_type", "adam")

        if optimizer_type.lower() == "adam":
            return optim.Adam(
                self.network.parameters(),
                lr=self.config.learning_rate,
                weight_decay=getattr(self.config, "weight_decay", 1e-5),
            )
        elif optimizer_type.lower() == "rmsprop":
            return optim.RMSprop(
                self.network.parameters(),
                lr=self.config.learning_rate,
                alpha=0.95,
                eps=1e-7,
            )
        elif optimizer_type.lower() == "sgd":
            return optim.SGD(
                self.network.parameters(), lr=self.config.learning_rate, momentum=0.9
            )
        else:
            # Default to Adam
            return optim.Adam(self.network.parameters(), lr=self.config.learning_rate)

    def step(self, loss: torch.Tensor) -> Dict[str, float]:
        """
        Perform optimization step with stability monitoring.

        Args:
            loss: Loss tensor to optimize

        Returns:
            Dictionary with optimization metrics

        Raises:
            OptimizationFailureError: If optimization fails
            LearningDivergenceError: If learning diverges
        """
        try:
            # Check for divergence
            loss_value = loss.item()
            if loss_value > self.divergence_threshold or math.isnan(loss_value):
                raise LearningDivergenceError(
                    f"Learning diverged for agent {self.agent_id}",
                    agent_id=self.agent_id,
                    divergence_metrics={"loss": loss_value, "step": self.step_count},
                )

            # Zero gradients
            self.optimizer.zero_grad()

            # Backward pass
            loss.backward()

            # Calculate gradient norm before clipping
            grad_norm = self._calculate_gradient_norm()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(
                self.network.parameters(), max_norm=self.gradient_clip_value
            )

            # Optimization step
            self.optimizer.step()

            # Update metrics
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.metrics.update(loss_value, grad_norm, current_lr)
            self.step_count += 1

            # Log progress periodically
            if self.step_count % 1000 == 0:
                self.logger.log_learning_update(
                    self.agent_id, self.step_count, 0.0, loss_value, current_lr
                )

            return {
                "loss": loss_value,
                "gradient_norm": grad_norm,
                "learning_rate": current_lr,
                "convergence_score": self.metrics.convergence_score,
            }

        except Exception as e:
            if isinstance(e, (LearningDivergenceError, OptimizationFailureError)):
                raise

            error_msg = f"Optimization step failed for agent {self.agent_id}"
            self.logger.log_error_with_context(
                e,
                {
                    "agent_id": self.agent_id,
                    "step_count": self.step_count,
                    "loss_value": loss_value if "loss_value" in locals() else "unknown",
                },
            )
            raise OptimizationFailureError(
                error_msg,
                optimizer_type=type(self.optimizer).__name__,
                optimization_params={
                    "learning_rate": self.config.learning_rate,
                    "step_count": self.step_count,
                },
            ) from e

    def _calculate_gradient_norm(self) -> float:
        """Calculate the norm of gradients."""
        total_norm = 0.0
        param_count = 0

        for param in self.network.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1

        return (total_norm**0.5) if param_count > 0 else 0.0

    def adjust_learning_rate(self, factor: float) -> None:
        """
        Adjust learning rate by a factor.

        Args:
            factor: Multiplication factor for learning rate
        """
        for param_group in self.optimizer.param_groups:
            old_lr = param_group["lr"]
            param_group["lr"] *= factor
            new_lr = param_group["lr"]

        self.logger.log_agent_action(
            self.agent_id,
            "learning_rate_adjusted",
            1.0,
            f"Old LR: {old_lr:.6f}, New LR: {new_lr:.6f}, Factor: {factor}",
        )

    def get_optimization_state(self) -> Dict[str, Any]:
        """Get current optimization state."""
        return {
            "step_count": self.step_count,
            "metrics": self.metrics.get_summary(),
            "optimizer_state": self.optimizer.state_dict(),
            "gradient_clip_value": self.gradient_clip_value,
            "divergence_threshold": self.divergence_threshold,
        }

    def load_optimization_state(self, state: Dict[str, Any]) -> None:
        """Load optimization state."""
        self.step_count = state.get("step_count", 0)
        self.gradient_clip_value = state.get("gradient_clip_value", 1.0)
        self.divergence_threshold = state.get("divergence_threshold", 10.0)

        if "optimizer_state" in state:
            self.optimizer.load_state_dict(state["optimizer_state"])

        self.logger.log_agent_action(
            self.agent_id, "optimization_state_loaded", 1.0, f"Step: {self.step_count}"
        )


class AdaptiveLearningRateScheduler:
    """
    Adaptive learning rate scheduler for MARL agents.

    Automatically adjusts learning rates based on training progress and
    performance metrics to maintain stable and efficient learning.
    """

    def __init__(self, optimizer: MARLOptimizer, config: AgentConfig):
        """
        Initialize adaptive learning rate scheduler.

        Args:
            optimizer: MARL optimizer to schedule
            config: Agent configuration
        """
        self.optimizer = optimizer
        self.config = config
        self.agent_id = optimizer.agent_id
        self.logger = get_marl_logger(f"scheduler.{self.agent_id}")

        # Scheduling parameters
        self.initial_lr = config.learning_rate
        self.min_lr = getattr(config, "min_learning_rate", 1e-6)
        self.patience = getattr(config, "lr_patience", 100)
        self.factor = getattr(config, "lr_decay_factor", 0.5)
        self.threshold = getattr(config, "lr_threshold", 1e-4)

        # State tracking
        self.best_loss = float("inf")
        self.patience_counter = 0
        self.last_improvement_step = 0

        self.logger.log_agent_action(
            self.agent_id,
            "scheduler_initialized",
            1.0,
            f"Initial LR: {self.initial_lr}, Min LR: {self.min_lr}, Patience: {self.patience}",
        )

    def step(self, current_loss: float) -> bool:
        """
        Update learning rate based on current loss.

        Args:
            current_loss: Current training loss

        Returns:
            True if learning rate was adjusted
        """
        adjusted = False

        # Check for improvement
        if current_loss < self.best_loss - self.threshold:
            self.best_loss = current_loss
            self.patience_counter = 0
            self.last_improvement_step = self.optimizer.step_count
        else:
            self.patience_counter += 1

        # Reduce learning rate if no improvement
        if self.patience_counter >= self.patience:
            current_lr = self.optimizer.optimizer.param_groups[0]["lr"]

            if current_lr > self.min_lr:
                new_lr = max(current_lr * self.factor, self.min_lr)
                self.optimizer.adjust_learning_rate(self.factor)

                self.patience_counter = 0
                adjusted = True

                self.logger.log_agent_action(
                    self.agent_id,
                    "learning_rate_reduced",
                    1.0,
                    f"New LR: {new_lr:.6f}, Best loss: {self.best_loss:.6f}",
                )

        return adjusted

    def get_scheduler_state(self) -> Dict[str, Any]:
        """Get scheduler state."""
        return {
            "best_loss": self.best_loss,
            "patience_counter": self.patience_counter,
            "last_improvement_step": self.last_improvement_step,
            "initial_lr": self.initial_lr,
            "min_lr": self.min_lr,
            "patience": self.patience,
            "factor": self.factor,
            "threshold": self.threshold,
        }

    def load_scheduler_state(self, state: Dict[str, Any]) -> None:
        """Load scheduler state."""
        self.best_loss = state.get("best_loss", float("inf"))
        self.patience_counter = state.get("patience_counter", 0)
        self.last_improvement_step = state.get("last_improvement_step", 0)

        self.logger.log_agent_action(
            self.agent_id,
            "scheduler_state_loaded",
            1.0,
            f"Best loss: {self.best_loss:.6f}, Patience: {self.patience_counter}",
        )


class LearningStabilizer:
    """
    Learning stabilizer for detecting and correcting training instabilities.

    Monitors training progress and applies corrective measures when
    learning becomes unstable or divergent.
    """

    def __init__(self, agent_id: str):
        """
        Initialize learning stabilizer.

        Args:
            agent_id: ID of the agent to stabilize
        """
        self.agent_id = agent_id
        self.logger = get_marl_logger(f"stabilizer.{agent_id}")

        # Stability monitoring
        self.loss_history = []
        self.gradient_history = []
        self.stability_window = 50

        # Thresholds
        self.loss_spike_threshold = 5.0  # Multiple of average loss
        self.gradient_explosion_threshold = 10.0
        self.oscillation_threshold = 0.8  # Correlation threshold

    def check_stability(self, loss: float, gradient_norm: float) -> Dict[str, Any]:
        """
        Check training stability and return recommendations.

        Args:
            loss: Current training loss
            gradient_norm: Current gradient norm

        Returns:
            Dictionary with stability analysis and recommendations
        """
        # Update history
        self.loss_history.append(loss)
        self.gradient_history.append(gradient_norm)

        # Keep only recent history
        if len(self.loss_history) > self.stability_window:
            self.loss_history = self.loss_history[-self.stability_window :]
            self.gradient_history = self.gradient_history[-self.stability_window :]

        stability_report = {"stable": True, "issues": [], "recommendations": []}

        if len(self.loss_history) < 10:
            return stability_report

        # Check for loss spikes
        recent_losses = self.loss_history[-10:]
        avg_loss = (
            np.mean(self.loss_history[:-10])
            if len(self.loss_history) > 10
            else np.mean(recent_losses)
        )

        if loss > avg_loss * self.loss_spike_threshold:
            stability_report["stable"] = False
            stability_report["issues"].append("loss_spike")
            stability_report["recommendations"].append("reduce_learning_rate")

        # Check for gradient explosion
        if gradient_norm > self.gradient_explosion_threshold:
            stability_report["stable"] = False
            stability_report["issues"].append("gradient_explosion")
            stability_report["recommendations"].append("clip_gradients")

        # Check for oscillations
        if len(self.loss_history) >= 20:
            recent_trend = np.corrcoef(range(20), self.loss_history[-20:])[0, 1]
            if abs(recent_trend) < 0.1:  # No clear trend
                oscillation_score = self._calculate_oscillation_score(
                    self.loss_history[-20:]
                )
                if oscillation_score > self.oscillation_threshold:
                    stability_report["stable"] = False
                    stability_report["issues"].append("oscillation")
                    stability_report["recommendations"].append("adjust_learning_rate")

        # Log stability issues
        if not stability_report["stable"]:
            issues_str = ", ".join(stability_report["issues"])
            self.logger.log_agent_action(
                self.agent_id,
                "stability_issue_detected",
                0.0,
                f"Issues: {issues_str}, Loss: {loss:.4f}, Grad norm: {gradient_norm:.4f}",
            )

        return stability_report

    def _calculate_oscillation_score(self, losses: List[float]) -> float:
        """Calculate oscillation score for loss sequence."""
        if len(losses) < 4:
            return 0.0

        # Count direction changes
        direction_changes = 0
        for i in range(2, len(losses)):
            prev_diff = losses[i - 1] - losses[i - 2]
            curr_diff = losses[i] - losses[i - 1]

            if prev_diff * curr_diff < 0:  # Sign change
                direction_changes += 1

        # Normalize by sequence length
        return direction_changes / (len(losses) - 2)

    def get_stability_summary(self) -> Dict[str, Any]:
        """Get stability analysis summary."""
        if not self.loss_history:
            return {"status": "insufficient_data"}

        return {
            "status": "stable" if len(self.loss_history) > 0 else "insufficient_data",
            "loss_trend": np.corrcoef(range(len(self.loss_history)), self.loss_history)[
                0, 1
            ]
            if len(self.loss_history) > 1
            else 0.0,
            "loss_variance": np.var(self.loss_history),
            "gradient_variance": np.var(self.gradient_history)
            if self.gradient_history
            else 0.0,
            "recent_loss": self.loss_history[-1] if self.loss_history else 0.0,
            "recent_gradient": self.gradient_history[-1]
            if self.gradient_history
            else 0.0,
        }
