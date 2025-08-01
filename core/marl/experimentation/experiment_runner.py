"""MARL Experiment Runner.

This module provides experiment execution capabilities for MARL systems,
including parallel execution, progress tracking, and result collection.
"""

# Standard Library

import asyncio
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

# Third-Party Library

# SynThesisAI Modules

from utils.logging_config import get_logger
from .experiment_manager import Experiment, ExperimentStatus


class ExperimentExecutionError(Exception):
    """Exception raised during experiment execution."""

    pass


class ExperimentRunner:
    """
    Experiment runner for executing MARL experiments.

    Handles the actual execution of experimental conditions,
    including parallel execution, progress tracking, and result collection.
    """

    def __init__(
        self,
        max_parallel_experiments: int = 4,
        progress_callback: Optional[Callable] = None,
    ) -> None:
        """Initialize the experiment runner.

        Args:
            max_parallel_experiments (int): Maximum number of parallel experiments.
            progress_callback (Optional[Callable]): Optional callback for progress updates.

        Returns:
            None
        """
        self.logger = get_logger(__name__)
        self.max_parallel_experiments = max_parallel_experiments
        self.progress_callback = progress_callback

        # Execution state
        self._running_experiments: Dict[str, asyncio.Task] = {}
        self._execution_semaphore = asyncio.Semaphore(max_parallel_experiments)

        self.logger.info(
            "Experiment runner initialized with max %d parallel experiments",
            max_parallel_experiments,
        )

    async def run_experiment(
        self,
        experiment: Experiment,
        marl_system_factory: Callable,
        environment_factory: Callable,
    ) -> bool:
        """Run a complete experiment.

        Args:
            experiment (Experiment): Experiment to run.
            marl_system_factory (Callable): Factory function for creating MARL systems.
            environment_factory (Callable): Factory function for creating environments.

        Returns:
            bool: True if experiment completed successfully.
        """
        if experiment.status == ExperimentStatus.RUNNING:
            self.logger.warning("Experiment %s is already running", experiment.experiment_id)
            return False

        self.logger.info("Starting experiment: %s", experiment.name)

        # Update experiment status
        experiment.status = ExperimentStatus.RUNNING
        experiment.started_at = datetime.now()

        try:
            if experiment.parallel_execution:
                success = await self._run_experiment_parallel(
                    experiment, marl_system_factory, environment_factory
                )
            else:
                success = await self._run_experiment_sequential(
                    experiment, marl_system_factory, environment_factory
                )

            # Update final status
            if success:
                experiment.status = ExperimentStatus.COMPLETED
                experiment.completed_at = datetime.now()
                self.logger.info("Experiment completed successfully: %s", experiment.name)
            else:
                experiment.status = ExperimentStatus.FAILED
                self.logger.error("Experiment failed: %s", experiment.name)

            return success

        except Exception as e:
            experiment.status = ExperimentStatus.FAILED
            self.logger.error("Experiment execution error: %s", str(e))
            return False

    async def _run_experiment_parallel(
        self,
        experiment: Experiment,
        marl_system_factory: Callable,
        environment_factory: Callable,
    ) -> bool:
        """Run experiment conditions in parallel.

        Args:
            experiment (Experiment): Experiment to run.
            marl_system_factory (Callable): Factory for MARL systems.
            environment_factory (Callable): Factory for environments.

        Returns:
            bool: True if at least one condition succeeded.
        """
        self.logger.info("Running experiment in parallel mode: %s", experiment.name)

        # Create tasks for each condition
        tasks = []
        for condition in experiment.conditions:
            task = asyncio.create_task(
                self._run_condition(
                    experiment,
                    condition.condition_id,
                    marl_system_factory,
                    environment_factory,
                )
            )
            tasks.append(task)

        # Wait for all conditions to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Check results
        success_count = 0
        for i, result in enumerate(results):
            condition_id = experiment.conditions[i].condition_id

            if isinstance(result, Exception):
                self.logger.error("Condition %s failed: %s", condition_id, str(result))
                experiment.results[condition_id].status = ExperimentStatus.FAILED
                experiment.results[condition_id].error_message = str(result)
            elif result:
                success_count += 1

        return success_count > 0

    async def _run_experiment_sequential(
        self,
        experiment: Experiment,
        marl_system_factory: Callable,
        environment_factory: Callable,
    ) -> bool:
        """Run experiment conditions sequentially.

        Args:
            experiment (Experiment): Experiment to run.
            marl_system_factory (Callable): Factory for MARL systems.
            environment_factory (Callable): Factory for environments.

        Returns:
            bool: True if at least one condition succeeded.
        """
        self.logger.info("Running experiment in sequential mode: %s", experiment.name)

        success_count = 0

        for condition in experiment.conditions:
            try:
                success = await self._run_condition(
                    experiment,
                    condition.condition_id,
                    marl_system_factory,
                    environment_factory,
                )

                if success:
                    success_count += 1
                else:
                    self.logger.warning("Condition %s failed", condition.condition_id)

            except Exception as e:
                self.logger.error("Condition %s error: %s", condition.condition_id, str(e))
                experiment.results[condition.condition_id].status = ExperimentStatus.FAILED
                experiment.results[condition.condition_id].error_message = str(e)

        return success_count > 0

    async def _run_condition(
        self,
        experiment: Experiment,
        condition_id: str,
        marl_system_factory: Callable,
        environment_factory: Callable,
    ) -> bool:
        """Run a single experimental condition.

        Args:
            experiment (Experiment): Parent experiment.
            condition_id (str): Condition identifier.
            marl_system_factory (Callable): Factory for MARL systems.
            environment_factory (Callable): Factory for environments.

        Returns:
            bool: True if condition completed successfully.
        """
        async with self._execution_semaphore:
            condition = experiment.get_condition(condition_id)
            result = experiment.get_result(condition_id)

            if not condition or not result:
                return False

            self.logger.info("Running condition: %s", condition.name)

            # Update result status
            result.status = ExperimentStatus.RUNNING
            result.start_time = datetime.now()

            try:
                # Create MARL system with condition configuration
                marl_system = marl_system_factory(condition.config)

                # Apply parameter overrides
                if condition.parameters:
                    await self._apply_parameters(marl_system, condition.parameters)

                # Create environment
                environment = environment_factory()

                # Set random seed if specified
                if experiment.random_seed is not None:
                    await self._set_random_seed(marl_system, experiment.random_seed)

                # Run training episodes
                success = await self._run_training_episodes(
                    experiment, condition_id, marl_system, environment
                )

                if success:
                    result.status = ExperimentStatus.COMPLETED
                    self.logger.info("Condition completed: %s", condition.name)
                else:
                    result.status = ExperimentStatus.FAILED
                    self.logger.warning("Condition failed: %s", condition.name)

                result.end_time = datetime.now()
                result.duration_seconds = (result.end_time - result.start_time).total_seconds()

                return success

            except Exception as e:
                result.status = ExperimentStatus.FAILED
                result.error_message = str(e)
                result.end_time = datetime.now()
                if result.start_time:
                    result.duration_seconds = (result.end_time - result.start_time).total_seconds()

                self.logger.error("Condition execution error: %s", str(e))
                return False

    async def _apply_parameters(self, marl_system: Any, parameters: Dict[str, Any]) -> None:
        """Apply parameter overrides to a MARL system.

        Args:
            marl_system: MARL system instance with a parameter_manager attribute.
            parameters: Dictionary of parameters to apply.

        Returns:
            None
        """
        try:
            # This would depend on the specific MARL system interface
            # For now, assume the system has a parameter manager
            if hasattr(marl_system, "parameter_manager"):
                success, errors = marl_system.parameter_manager.set_parameters(parameters)
                if not success:
                    self.logger.warning("Parameter application errors: %s", "; ".join(errors))

        except Exception as e:
            self.logger.warning("Failed to apply parameters: %s", str(e))

    async def _set_random_seed(self, marl_system: Any, seed: int) -> None:
        """Set random seed for reproducibility.

        Args:
            marl_system: MARL system instance to set seed in.
            seed: Seed value for random number generators.

        Returns:
            None
        """
        try:
            import random

            import numpy as np

            # Set Python random seed
            random.seed(seed)

            # Set NumPy random seed
            np.random.seed(seed)

            # Set PyTorch seed if available
            try:
                import torch

                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
            except ImportError:
                pass

            # Set system-specific seed if supported
            if hasattr(marl_system, "set_random_seed"):
                marl_system.set_random_seed(seed)

        except Exception as e:
            self.logger.warning("Failed to set random seed: %s", str(e))

    async def _run_training_episodes(
        self,
        experiment: Experiment,
        condition_id: str,
        marl_system: Any,
        environment: Any,
    ) -> bool:
        """Run training episodes for a condition.

        Args:
            experiment (Experiment): Parent experiment.
            condition_id (str): Condition identifier.
            marl_system (Any): MARL system instance.
            environment (Any): Environment instance.

        Returns:
            bool: True if training completed successfully.
        """
        result = experiment.get_result(condition_id)
        if not result:
            return False

        max_episodes = experiment.max_episodes_per_condition
        max_duration = experiment.max_duration_per_condition

        start_time = time.time()
        episode_rewards = []
        coordination_success_rates = []

        try:
            for episode in range(max_episodes):
                # Check time limit
                if max_duration and (time.time() - start_time) > max_duration:
                    self.logger.info("Condition %s reached time limit", condition_id)
                    break

                # Run single episode
                episode_result = await self._run_single_episode(marl_system, environment, episode)

                if episode_result is None:
                    self.logger.warning("Episode %d failed for condition %s", episode, condition_id)
                    continue

                # Collect metrics
                episode_rewards.append(episode_result.get("total_reward", 0.0))
                coordination_success_rates.append(
                    episode_result.get("coordination_success_rate", 0.0)
                )

                # Record metrics in result
                result.add_metric("episode_reward", episode_result.get("total_reward", 0.0))
                result.add_metric(
                    "coordination_success_rate",
                    episode_result.get("coordination_success_rate", 0.0),
                )

                # Progress callback
                if self.progress_callback:
                    progress = {
                        "experiment_id": experiment.experiment_id,
                        "condition_id": condition_id,
                        "episode": episode,
                        "total_episodes": max_episodes,
                        "progress_percentage": (episode / max_episodes) * 100,
                        "current_reward": episode_result.get("total_reward", 0.0),
                        "average_reward": sum(episode_rewards) / len(episode_rewards),
                    }
                    self.progress_callback(progress)

                # Log progress periodically
                if episode % 100 == 0:
                    avg_reward = sum(episode_rewards[-100:]) / min(len(episode_rewards), 100)
                    avg_coordination = sum(coordination_success_rates[-100:]) / min(
                        len(coordination_success_rates), 100
                    )

                    self.logger.info(
                        "Condition %s - Episode %d: Avg Reward=%.3f, Avg Coordination=%.3f",
                        condition_id,
                        episode,
                        avg_reward,
                        avg_coordination,
                    )

            # Store final performance data
            result.add_performance_data("episode_rewards", episode_rewards)
            result.add_performance_data("coordination_success_rates", coordination_success_rates)

            # Calculate final metrics
            if episode_rewards:
                result.add_metric(
                    "final_average_reward", sum(episode_rewards) / len(episode_rewards)
                )
                result.add_metric("final_max_reward", max(episode_rewards))
                result.add_metric("final_min_reward", min(episode_rewards))

            if coordination_success_rates:
                result.add_metric(
                    "final_average_coordination_success",
                    sum(coordination_success_rates) / len(coordination_success_rates),
                )

            result.add_metric("total_episodes_completed", len(episode_rewards))
            result.add_metric("total_training_time", time.time() - start_time)

            return True

        except Exception as e:
            self.logger.error("Training episodes failed for condition %s: %s", condition_id, str(e))
            result.error_message = str(e)
            return False

    async def _run_single_episode(
        self, marl_system: Any, environment: Any, episode_num: int
    ) -> Optional[Dict[str, Any]]:
        """Run a single training episode.

        Args:
            marl_system (Any): MARL system instance.
            environment (Any): Environment instance.
            episode_num (int): Episode number.

        Returns:
            Optional[Dict[str, Any]]: Episode results dictionary or None if failed.
        """
        try:
            # This is a simplified interface - actual implementation would depend
            # on the specific MARL system and environment interfaces

            # Reset environment
            if hasattr(environment, "reset"):
                state = environment.reset()
            else:
                state = None

            total_reward = 0.0
            coordination_attempts = 0
            successful_coordinations = 0
            step = 0
            max_steps = 1000  # Default max steps per episode

            done = False
            while not done and step < max_steps:
                # Get actions from MARL system
                if hasattr(marl_system, "get_actions"):
                    actions = marl_system.get_actions(state)
                else:
                    # Fallback: random actions
                    actions = {"agent_0": 0}  # Simplified

                # Execute actions in environment
                if hasattr(environment, "step"):
                    next_state, rewards, done, info = environment.step(actions)
                else:
                    # Simplified fallback
                    next_state = state
                    rewards = {"agent_0": 0.1}
                    done = step >= max_steps - 1
                    info = {}

                # Update MARL system
                if hasattr(marl_system, "update"):
                    marl_system.update(state, actions, rewards, next_state, done)

                # Track coordination
                if info.get("coordination_attempted", False):
                    coordination_attempts += 1
                    if info.get("coordination_successful", False):
                        successful_coordinations += 1

                # Accumulate rewards
                if isinstance(rewards, dict):
                    total_reward += sum(rewards.values())
                else:
                    total_reward += rewards

                state = next_state
                step += 1

            # Calculate coordination success rate
            coordination_success_rate = (
                successful_coordinations / coordination_attempts
                if coordination_attempts > 0
                else 0.0
            )

            return {
                "total_reward": total_reward,
                "coordination_success_rate": coordination_success_rate,
                "steps": step,
                "coordination_attempts": coordination_attempts,
                "successful_coordinations": successful_coordinations,
            }

        except Exception as e:
            self.logger.error("Single episode failed: %s", str(e))
            return None

    def cancel_experiment(self, experiment_id: str) -> bool:
        """Cancel a running experiment.

        Args:
            experiment_id (str): Experiment ID to cancel.

        Returns:
            bool: True if cancelled successfully.
        """
        if experiment_id in self._running_experiments:
            task = self._running_experiments[experiment_id]
            task.cancel()
            del self._running_experiments[experiment_id]

            self.logger.info("Cancelled experiment: %s", experiment_id)
            return True

        return False

    def get_running_experiments(self) -> List[str]:
        """Get list of currently running experiment IDs.

        Returns:
            List[str]: List of experiment IDs currently running.
        """
        return list(self._running_experiments.keys())

    def is_experiment_running(self, experiment_id: str) -> bool:
        """Check if an experiment is currently running.

        Args:
            experiment_id (str): Experiment ID to check.

        Returns:
            bool: True if the experiment is running, False otherwise.
        """
        return experiment_id in self._running_experiments


class ExperimentRunnerFactory:
    """Factory for creating ExperimentRunner instances."""

    @staticmethod
    def create(max_parallel_experiments: int = 4) -> ExperimentRunner:
        """Create an ExperimentRunner.

        Args:
            max_parallel_experiments (int): Maximum number of parallel experiments.

        Returns:
            ExperimentRunner: A new ExperimentRunner instance.
        """
        return ExperimentRunner(max_parallel_experiments)

    @staticmethod
    def create_with_progress_callback(
        max_parallel_experiments: int = 4, progress_callback: Optional[Callable] = None
    ) -> ExperimentRunner:
        """Create an ExperimentRunner with a progress callback.

        Args:
            max_parallel_experiments (int): Maximum number of parallel experiments.
            progress_callback (Optional[Callable]): Callback function for progress updates.

        Returns:
            ExperimentRunner: A new ExperimentRunner instance with progress callback.
        """
        return ExperimentRunner(max_parallel_experiments, progress_callback)
