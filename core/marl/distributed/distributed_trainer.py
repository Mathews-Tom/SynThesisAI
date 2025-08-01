"""Distributed MARL Training Infrastructure.

This module provides distributed training capabilities for multi-agent
reinforcement learning, supporting multi-GPU and multi-node deployment.
"""

# Standard Library
import asyncio
import os
import time
from dataclasses import dataclass, field
from datetime import timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# Third-Party Libraries
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# SynThesisAI Modules
from utils.logging_config import get_logger


class TrainingMode(Enum):
    """Enumeration for supported training modes."""

    SINGLE_GPU = "single_gpu"
    MULTI_GPU = "multi_gpu"
    MULTI_NODE = "multi_node"
    CPU_ONLY = "cpu_only"


class SynchronizationStrategy(Enum):
    """Enumeration for synchronization strategies in distributed training."""

    SYNCHRONOUS = "synchronous"
    ASYNCHRONOUS = "asynchronous"
    SEMI_SYNCHRONOUS = "semi_synchronous"


@dataclass
class DistributedTrainingConfig:
    """Configuration for distributed MARL training.

    Attributes:
        training_mode: The training mode to use.
        world_size: The total number of processes participating in training.
        rank: The unique identifier of the current process.
        local_rank: The local rank of the process on a node.
        master_addr: The address of the master node for process coordination.
        master_port: The port on the master node for process coordination.
        backend: The distributed communication backend to use ('nccl' or 'gloo').
        sync_strategy: The strategy for synchronizing gradients and parameters.
        sync_frequency: The number of steps between synchronizations.
        gradient_clipping: The maximum norm for gradient clipping.
        gpu_ids: A list of GPU IDs to use for training.
        cpu_workers: The number of CPU workers for data loading.
        memory_limit_gb: The memory limit per GPU in gigabytes.
        batch_size_per_gpu: The batch size for each GPU.
        accumulation_steps: The number of steps to accumulate gradients over.
        mixed_precision: Whether to use mixed-precision training.
        checkpoint_frequency: The frequency (in epochs) for saving checkpoints.
        checkpoint_dir: The directory to save checkpoints in.
        save_optimizer_state: Whether to save the optimizer state in checkpoints.
        communication_timeout: The timeout for distributed communication in seconds.
        max_retries: The maximum number of retries for failed communication operations.
    """

    training_mode: TrainingMode = TrainingMode.SINGLE_GPU
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    master_addr: str = "localhost"
    master_port: str = "12355"
    backend: str = "nccl"
    sync_strategy: SynchronizationStrategy = SynchronizationStrategy.SYNCHRONOUS
    sync_frequency: int = 10
    gradient_clipping: float = 1.0
    gpu_ids: List[int] = field(default_factory=list)
    cpu_workers: int = 4
    memory_limit_gb: float = 8.0
    batch_size_per_gpu: int = 32
    accumulation_steps: int = 1
    mixed_precision: bool = True
    checkpoint_frequency: int = 1000
    checkpoint_dir: str = "checkpoints/distributed"
    save_optimizer_state: bool = True
    communication_timeout: float = 30.0
    max_retries: int = 3

    def __post_init__(self) -> None:
        """Validates the configuration after initialization.

        Raises:
            ValueError: If the configuration is invalid.
        """
        if self.world_size <= 0:
            raise ValueError("World size must be a positive integer.")

        if not 0 <= self.rank < self.world_size:
            raise ValueError(
                f"Rank must be between 0 and {self.world_size - 1}, but got {self.rank}."
            )

        if self.training_mode == TrainingMode.MULTI_GPU and not self.gpu_ids:
            if torch.cuda.is_available():
                self.gpu_ids = list(range(torch.cuda.device_count()))
            else:
                raise ValueError("Multi-GPU mode requires at least one available GPU.")


class DistributedMARLTrainer:
    """Manages distributed training for multi-agent reinforcement learning.

    This class orchestrates the entire distributed training workflow, including
    environment setup, agent management, training loops, synchronization, and
    checkpointing across multiple GPUs and nodes.

    Attributes:
        config: The distributed training configuration object.
        logger: The logger instance for this class.
        is_initialized: A flag indicating if the distributed environment is initialized.
        is_training: A flag indicating if training is currently active.
        current_epoch: The current training epoch.
        global_step: The total number of training steps performed.
        process_group: The distributed process group.
        device: The primary computation device (e.g., 'cuda:0' or 'cpu').
        local_device: The device for the local process.
        agents: A dictionary of registered agents.
        distributed_agents: A dictionary of agents wrapped for distributed training.
        training_metrics: A dictionary to store various training metrics.
        training_callbacks: A list of callbacks to be executed during training.
        synchronization_callbacks: A list of callbacks for synchronization events.
    """

    def __init__(self, config: DistributedTrainingConfig) -> None:
        """Initializes the DistributedMARLTrainer.

        Args:
            config: The configuration object for distributed training.
        """
        self.config: DistributedTrainingConfig = config
        self.logger = get_logger(__name__)

        self.is_initialized: bool = False
        self.is_training: bool = False
        self.current_epoch: int = 0
        self.global_step: int = 0

        self.process_group: Optional[dist.ProcessGroup] = None
        self.device: Optional[torch.device] = None
        self.local_device: Optional[torch.device] = None

        self.agents: Dict[str, Any] = {}
        self.distributed_agents: Dict[str, DDP] = {}

        self.training_metrics: Dict[str, Any] = {
            "total_steps": 0,
            "total_episodes": 0,
            "average_reward": 0.0,
            "training_time": 0.0,
            "communication_time": 0.0,
            "synchronization_time": 0.0,
        }

        self.training_callbacks: List[Callable[[int, Dict[str, Any]], None]] = []
        self.synchronization_callbacks: List[Callable[[], None]] = []

        self.logger.info("Distributed MARL trainer initialized with config: %s", self.config)

    async def initialize_distributed_training(self) -> None:
        """Initializes the distributed training environment.

        This method sets up the necessary environment variables, devices, and
        process groups required for distributed operation.

        Raises:
            RuntimeError: If initialization fails.
        """
        if self.is_initialized:
            self.logger.warning("Distributed training is already initialized.")
            return

        try:
            self.logger.info(
                "Initializing distributed training (rank %d/%d)...",
                self.config.rank,
                self.config.world_size,
            )
            self._setup_distributed_environment()
            self._setup_devices()
            await self._setup_process_group()
            await self._setup_distributed_agents()
            self.is_initialized = True
            self.logger.info("Distributed training initialization complete.")
        except Exception as e:
            self.logger.exception("Failed to initialize distributed training.")
            raise RuntimeError("Distributed training initialization failed.") from e

    def _setup_distributed_environment(self) -> None:
        """Sets up environment variables for distributed training."""
        os.environ["MASTER_ADDR"] = self.config.master_addr
        os.environ["MASTER_PORT"] = self.config.master_port
        os.environ["WORLD_SIZE"] = str(self.config.world_size)
        os.environ["RANK"] = str(self.config.rank)
        os.environ["LOCAL_RANK"] = str(self.config.local_rank)

        if self.config.training_mode != TrainingMode.CPU_ONLY:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.config.gpu_ids))
        self.logger.debug("Distributed environment variables set.")

    def _setup_devices(self) -> None:
        """Sets up the training devices (CPU or GPU)."""
        if self.config.training_mode == TrainingMode.CPU_ONLY:
            self.device = torch.device("cpu")
        elif torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.config.local_rank}")
            torch.cuda.set_device(self.device)
        else:
            self.logger.warning("CUDA not available, falling back to CPU.")
            self.device = torch.device("cpu")

        self.local_device = self.device
        self.logger.info("Training device set to: %s", self.device)

    async def _setup_process_group(self) -> None:
        """Initializes the distributed process group."""
        if self.config.world_size > 1:
            self.logger.info("Initializing process group with backend: %s", self.config.backend)
            dist.init_process_group(
                backend=self.config.backend,
                init_method=f"tcp://{self.config.master_addr}:{self.config.master_port}",
                world_size=self.config.world_size,
                rank=self.config.rank,
                timeout=timedelta(seconds=self.config.communication_timeout),
            )
            self.process_group = dist.group.WORLD
            self.logger.info("Process group initialized successfully.")
        else:
            self.logger.info("Single-process training, no process group needed.")

    async def _setup_distributed_agents(self) -> None:
        """Wraps agents with DistributedDataParallel (DDP)."""
        for agent_id, agent in self.agents.items():
            try:
                agent.to(self.device)
                if self.config.world_size > 1:
                    device_ids = [self.config.local_rank] if self.device.type == "cuda" else None
                    self.distributed_agents[agent_id] = DDP(
                        agent,
                        device_ids=device_ids,
                        output_device=(
                            self.config.local_rank if self.device.type == "cuda" else None
                        ),
                        find_unused_parameters=True,
                    )
                else:
                    self.distributed_agents[agent_id] = agent
                self.logger.debug("Set up distributed agent: %s", agent_id)
            except Exception:
                self.logger.exception("Failed to set up distributed agent %s.", agent_id)
                raise

    def register_agent(self, agent_id: str, agent: Any) -> None:
        """Registers an agent for distributed training.

        Args:
            agent_id: A unique identifier for the agent.
            agent: The agent instance to register.
        """
        if agent_id in self.agents:
            self.logger.warning("Agent with ID '%s' is already registered. Overwriting.", agent_id)
        self.agents[agent_id] = agent
        self.logger.info("Registered agent for distributed training: %s", agent_id)

    def unregister_agent(self, agent_id: str) -> None:
        """Unregisters an agent from the trainer.

        Args:
            agent_id: The identifier of the agent to unregister.
        """
        if agent_id in self.agents:
            del self.agents[agent_id]
            if agent_id in self.distributed_agents:
                del self.distributed_agents[agent_id]
            self.logger.info("Unregistered agent: %s", agent_id)
        else:
            self.logger.warning("Attempted to unregister non-existent agent: %s", agent_id)

    async def start_distributed_training(
        self,
        num_epochs: int,
        steps_per_epoch: int,
        training_data_loader: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Starts the distributed training loop.

        Args:
            num_epochs: The total number of epochs to train for.
            steps_per_epoch: The number of training steps in each epoch.
            training_data_loader: An optional data loader for training data.

        Returns:
            A dictionary containing the final training metrics.

        Raises:
            RuntimeError: If training is already in progress or fails.
        """
        if not self.is_initialized:
            await self.initialize_distributed_training()

        if self.is_training:
            self.logger.warning("Training is already in progress.")
            return self.training_metrics

        self.logger.info(
            "Starting distributed training for %d epochs with %d steps per epoch.",
            num_epochs,
            steps_per_epoch,
        )
        self.is_training = True
        start_time = time.monotonic()

        try:
            for epoch in range(self.current_epoch, num_epochs):
                self.current_epoch = epoch
                epoch_results = await self._train_epoch(
                    epoch, steps_per_epoch, training_data_loader
                )
                self._update_training_metrics(epoch_results)

                if (epoch + 1) % self.config.sync_frequency == 0:
                    await self._synchronize_agents()

                if (epoch + 1) % self.config.checkpoint_frequency == 0:
                    await self._save_checkpoint(epoch)

                await self._notify_training_callbacks(epoch, epoch_results)
                self.logger.info(
                    "Epoch %d complete: Avg Reward=%.3f, Steps=%d",
                    epoch,
                    epoch_results.get("average_reward", 0.0),
                    epoch_results.get("steps", 0),
                )

            await self._synchronize_agents()
            await self._save_checkpoint(num_epochs)

        except Exception as e:
            self.logger.exception(
                "Distributed training failed during epoch %d.", self.current_epoch
            )
            raise RuntimeError("Training failed.") from e
        finally:
            self.is_training = False
            self.training_metrics["training_time"] = time.monotonic() - start_time
            self.logger.info(
                "Distributed training finished in %.2f seconds. Total steps: %d",
                self.training_metrics["training_time"],
                self.training_metrics["total_steps"],
            )

        return self.training_metrics

    async def _train_epoch(
        self, epoch: int, steps_per_epoch: int, data_loader: Optional[Any]
    ) -> Dict[str, Any]:
        """Runs a single training epoch.

        Args:
            epoch: The current epoch number.
            steps_per_epoch: The number of steps to perform in this epoch.
            data_loader: The data loader for training batches.

        Returns:
            A dictionary containing metrics for the epoch.
        """
        epoch_metrics = {
            "steps": 0,
            "total_reward": 0.0,
            "losses": [],
            "communication_time": 0.0,
        }

        for step in range(steps_per_epoch):
            step_results = await self._training_step(step)
            epoch_metrics["steps"] += 1
            epoch_metrics["total_reward"] += step_results.get("reward", 0.0)
            if "loss" in step_results:
                epoch_metrics["losses"].append(step_results["loss"])

            if self.config.sync_strategy == SynchronizationStrategy.SYNCHRONOUS:
                comm_start = time.monotonic()
                await self._synchronize_gradients()
                epoch_metrics["communication_time"] += time.monotonic() - comm_start
            elif (
                self.config.sync_strategy == SynchronizationStrategy.SEMI_SYNCHRONOUS
                and (step + 1) % self.config.sync_frequency == 0
            ):
                comm_start = time.monotonic()
                await self._synchronize_gradients()
                epoch_metrics["communication_time"] += time.monotonic() - comm_start

            self.global_step += 1

        if epoch_metrics["steps"] > 0:
            epoch_metrics["average_reward"] = epoch_metrics["total_reward"] / epoch_metrics["steps"]
        else:
            epoch_metrics["average_reward"] = 0.0

        return epoch_metrics

    async def _training_step(self, step: int) -> Dict[str, Any]:
        """Executes a single training step for all agents.

        Args:
            step: The current step number.

        Returns:
            A dictionary containing results from the training step.
        """
        step_results = {"reward": 0.0, "loss": 0.0, "agent_results": {}}
        num_agents = len(self.distributed_agents)

        if num_agents == 0:
            return step_results

        for agent_id, agent in self.distributed_agents.items():
            try:
                # This is a placeholder for actual agent training logic.
                # Typically involves environment interaction, forward/backward passes.
                agent_reward = np.random.normal(0.5, 0.1)
                agent_loss = np.random.exponential(0.1)

                step_results["agent_results"][agent_id] = {
                    "reward": agent_reward,
                    "loss": agent_loss,
                }
                step_results["reward"] += agent_reward
                step_results["loss"] += agent_loss
            except Exception:
                self.logger.exception("Training step failed for agent %s.", agent_id)
                raise

        step_results["reward"] /= num_agents
        step_results["loss"] /= num_agents
        return step_results

    async def _synchronize_gradients(self) -> None:
        """Synchronizes gradients across all distributed processes."""
        if self.config.world_size <= 1:
            return

        try:
            for agent in self.distributed_agents.values():
                if self.config.gradient_clipping > 0:
                    torch.nn.utils.clip_grad_norm_(
                        agent.parameters(), self.config.gradient_clipping
                    )
                for param in agent.parameters():
                    if param.grad is not None:
                        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                        param.grad.data /= self.config.world_size
            self.logger.debug("Gradient synchronization complete.")
        except Exception:
            self.logger.exception("Gradient synchronization failed.")
            raise

    async def _synchronize_agents(self) -> None:
        """Synchronizes agent model parameters across all processes."""
        if self.config.world_size <= 1:
            return

        start_time = time.monotonic()
        try:
            for agent in self.distributed_agents.values():
                for param in agent.parameters():
                    dist.broadcast(param.data, src=0)
            sync_time = time.monotonic() - start_time
            self.training_metrics["synchronization_time"] += sync_time
            await self._notify_synchronization_callbacks()
            self.logger.debug("Agent parameter synchronization complete (%.3fs).", sync_time)
        except Exception:
            self.logger.exception("Agent synchronization failed.")
            raise

    async def _save_checkpoint(self, epoch: int) -> None:
        """Saves a training checkpoint.

        Args:
            epoch: The current epoch number.
        """
        if self.config.rank != 0:
            return

        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"

        checkpoint_data = {
            "epoch": epoch,
            "global_step": self.global_step,
            "training_metrics": self.training_metrics,
            "config": self.config.__dict__,
            "agents": {
                agent_id: agent.module.state_dict()
                for agent_id, agent in self.distributed_agents.items()
            },
        }

        try:
            torch.save(checkpoint_data, checkpoint_path)
            self.logger.info("Checkpoint saved to %s", checkpoint_path)
        except IOError as e:
            self.logger.error("Failed to save checkpoint to %s: %s", checkpoint_path, e)

    async def load_checkpoint(self, checkpoint_path: str) -> None:
        """Loads a training checkpoint.

        Args:
            checkpoint_path: The path to the checkpoint file.

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
            RuntimeError: If loading the checkpoint fails.
        """
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.current_epoch = checkpoint["epoch"]
            self.global_step = checkpoint["global_step"]
            self.training_metrics = checkpoint["training_metrics"]

            for agent_id, state_dict in checkpoint["agents"].items():
                if agent_id in self.distributed_agents:
                    self.distributed_agents[agent_id].module.load_state_dict(state_dict)
            self.logger.info("Checkpoint loaded from %s", checkpoint_path)
        except Exception as e:
            self.logger.exception("Failed to load checkpoint from %s.", checkpoint_path)
            raise RuntimeError("Failed to load checkpoint.") from e

    def _update_training_metrics(self, epoch_results: Dict[str, Any]) -> None:
        """Updates the main training metrics with results from an epoch.

        Args:
            epoch_results: A dictionary of metrics from the completed epoch.
        """
        self.training_metrics["total_steps"] += epoch_results.get("steps", 0)
        self.training_metrics["total_episodes"] += 1
        new_reward = epoch_results.get("average_reward", 0.0)
        alpha = 0.1
        self.training_metrics["average_reward"] = (
            alpha * new_reward + (1 - alpha) * self.training_metrics["average_reward"]
        )
        self.training_metrics["communication_time"] += epoch_results.get("communication_time", 0.0)

    async def _notify_training_callbacks(self, epoch: int, results: Dict[str, Any]) -> None:
        """Notifies all registered training callbacks.

        Args:
            epoch: The completed epoch number.
            results: The results from the completed epoch.
        """
        for callback in self.training_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(epoch, results)
                else:
                    callback(epoch, results)
            except Exception:
                self.logger.exception("Error in training callback.")

    async def _notify_synchronization_callbacks(self) -> None:
        """Notifies all registered synchronization callbacks."""
        for callback in self.synchronization_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback()
                else:
                    callback()
            except Exception:
                self.logger.exception("Error in synchronization callback.")

    def add_training_callback(self, callback: Callable[[int, Dict[str, Any]], None]) -> None:
        """Adds a callback to be executed after each training epoch.

        Args:
            callback: The callback function to add.
        """
        self.training_callbacks.append(callback)

    def add_synchronization_callback(self, callback: Callable[[], None]) -> None:
        """Adds a callback to be executed after agent synchronization.

        Args:
            callback: The callback function to add.
        """
        self.synchronization_callbacks.append(callback)

    def get_training_metrics(self) -> Dict[str, Any]:
        """Returns a copy of the current training metrics.

        Returns:
            A dictionary of training metrics.
        """
        return self.training_metrics.copy()

    def get_distributed_info(self) -> Dict[str, Any]:
        """Returns information about the distributed training setup.

        Returns:
            A dictionary containing distributed training details.
        """
        return {
            "world_size": self.config.world_size,
            "rank": self.config.rank,
            "local_rank": self.config.local_rank,
            "device": str(self.device),
            "training_mode": self.config.training_mode.value,
            "sync_strategy": self.config.sync_strategy.value,
            "is_initialized": self.is_initialized,
            "is_training": self.is_training,
            "registered_agents": list(self.agents.keys()),
        }

    async def stop_training(self) -> None:
        """Stops the current training loop gracefully."""
        if self.is_training:
            self.is_training = False
            self.logger.info("Training stop signal received.")

    async def shutdown(self) -> None:
        """Shuts down the distributed training system."""
        await self.stop_training()
        if self.config.world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()
        self.agents.clear()
        self.distributed_agents.clear()
        self.training_callbacks.clear()
        self.synchronization_callbacks.clear()
        self.is_initialized = False
        self.logger.info("Distributed MARL trainer has been shut down.")


class DistributedTrainerFactory:
    """A factory for creating instances of `DistributedMARLTrainer`."""

    @staticmethod
    def create_single_gpu_trainer() -> DistributedMARLTrainer:
        """Creates a trainer configured for single-GPU training.

        Returns:
            A `DistributedMARLTrainer` instance.
        """
        config = DistributedTrainingConfig(
            training_mode=TrainingMode.SINGLE_GPU, world_size=1, rank=0
        )
        return DistributedMARLTrainer(config)

    @staticmethod
    def create_multi_gpu_trainer(gpu_ids: List[int]) -> DistributedMARLTrainer:
        """Creates a trainer configured for multi-GPU training.

        Args:
            gpu_ids: A list of GPU IDs to use.

        Returns:
            A `DistributedMARLTrainer` instance.
        """
        config = DistributedTrainingConfig(
            training_mode=TrainingMode.MULTI_GPU,
            world_size=len(gpu_ids),
            gpu_ids=gpu_ids,
        )
        return DistributedMARLTrainer(config)

    @staticmethod
    def create_multi_node_trainer(
        world_size: int, rank: int, master_addr: str, master_port: str = "12355"
    ) -> DistributedMARLTrainer:
        """Creates a trainer for multi-node distributed training.

        Args:
            world_size: The total number of processes.
            rank: The rank of the current process.
            master_addr: The address of the master node.
            master_port: The port of the master node.

        Returns:
            A `DistributedMARLTrainer` instance.
        """
        config = DistributedTrainingConfig(
            training_mode=TrainingMode.MULTI_NODE,
            world_size=world_size,
            rank=rank,
            master_addr=master_addr,
            master_port=master_port,
        )
        return DistributedMARLTrainer(config)

    @staticmethod
    def create_cpu_trainer(cpu_workers: int = 4) -> DistributedMARLTrainer:
        """Creates a trainer configured for CPU-only training.

        Args:
            cpu_workers: The number of CPU workers to use.

        Returns:
            A `DistributedMARLTrainer` instance.
        """
        config = DistributedTrainingConfig(
            training_mode=TrainingMode.CPU_ONLY,
            world_size=1,
            rank=0,
            cpu_workers=cpu_workers,
            backend="gloo",
        )
        return DistributedMARLTrainer(config)


def setup_distributed_environment(
    rank: int,
    world_size: int,
    master_addr: str = "localhost",
    master_port: str = "12355",
) -> None:
    """A utility function to set up the distributed environment.

    Args:
        rank: The rank of the current process.
        world_size: The total number of processes.
        master_addr: The address of the master node.
        master_port: The port of the master node.
    """
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port


def cleanup_distributed_environment() -> None:
    """Cleans up the distributed environment resources."""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_distributed_info() -> Dict[str, Any]:
    """Retrieves information about the current distributed setup.

    Returns:
        A dictionary with distributed training information.
    """
    if not dist.is_initialized():
        return {"initialized": False, "world_size": 1, "rank": 0, "local_rank": 0}
    return {
        "initialized": True,
        "world_size": dist.get_world_size(),
        "rank": dist.get_rank(),
        "local_rank": int(os.environ.get("LOCAL_RANK", 0)),
    }


def is_main_process() -> bool:
    """Checks if the current process is the main process (rank 0).

    Returns:
        True if it is the main process, False otherwise.
    """
    return not dist.is_initialized() or dist.get_rank() == 0


def synchronize_processes() -> None:
    """Synchronizes all processes by creating a barrier."""
    if dist.is_initialized():
        dist.barrier()
