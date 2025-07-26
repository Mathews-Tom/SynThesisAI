"""Distributed MARL Training Infrastructure.

This module provides distributed training capabilities for multi-agent
reinforcement learning, supporting multi-GPU and multi-node deployment.
"""

import asyncio
import json
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.logging_config import get_logger


class TrainingMode(Enum):
    """Training mode enumeration."""

    SINGLE_GPU = "single_gpu"
    MULTI_GPU = "multi_gpu"
    MULTI_NODE = "multi_node"
    CPU_ONLY = "cpu_only"


class SynchronizationStrategy(Enum):
    """Synchronization strategy for distributed training."""

    SYNCHRONOUS = "synchronous"
    ASYNCHRONOUS = "asynchronous"
    SEMI_SYNCHRONOUS = "semi_synchronous"


@dataclass
class DistributedTrainingConfig:
    """Configuration for distributed MARL training."""

    # Training mode
    training_mode: TrainingMode = TrainingMode.SINGLE_GPU

    # Distributed settings
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    master_addr: str = "localhost"
    master_port: str = "12355"
    backend: str = "nccl"  # nccl for GPU, gloo for CPU

    # Synchronization
    sync_strategy: SynchronizationStrategy = SynchronizationStrategy.SYNCHRONOUS
    sync_frequency: int = 10  # Steps between synchronization
    gradient_clipping: float = 1.0

    # Resource allocation
    gpu_ids: List[int] = field(default_factory=list)
    cpu_workers: int = 4
    memory_limit_gb: float = 8.0

    # Training parameters
    batch_size_per_gpu: int = 32
    accumulation_steps: int = 1
    mixed_precision: bool = True

    # Checkpointing
    checkpoint_frequency: int = 1000
    checkpoint_dir: str = "checkpoints/distributed"
    save_optimizer_state: bool = True

    # Communication
    communication_timeout: float = 30.0
    max_retries: int = 3

    def __post_init__(self):
        """Validate configuration."""
        if self.world_size <= 0:
            raise ValueError("World size must be positive")

        if self.rank < 0 or self.rank >= self.world_size:
            raise ValueError(f"Rank must be between 0 and {self.world_size - 1}")

        if self.training_mode == TrainingMode.MULTI_GPU and not self.gpu_ids:
            # Auto-detect GPUs
            if torch.cuda.is_available():
                self.gpu_ids = list(range(torch.cuda.device_count()))
            else:
                raise ValueError("Multi-GPU mode requires available GPUs")


class DistributedMARLTrainer:
    """Distributed training system for MARL agents.

    Provides distributed training capabilities across multiple GPUs and nodes,
    with support for various synchronization strategies and resource management.
    """

    def __init__(self, config: DistributedTrainingConfig):
        """
        Initialize distributed MARL trainer.

        Args:
            config: Distributed training configuration
        """
        self.config = config
        self.logger = get_logger(__name__)

        # Training state
        self.is_initialized = False
        self.is_training = False
        self.current_epoch = 0
        self.global_step = 0

        # Distributed state
        self.process_group = None
        self.device = None
        self.local_device = None

        # Agent management
        self.agents: Dict[str, Any] = {}
        self.distributed_agents: Dict[str, Any] = {}

        # Training metrics
        self.training_metrics = {
            "total_steps": 0,
            "total_episodes": 0,
            "average_reward": 0.0,
            "training_time": 0.0,
            "communication_time": 0.0,
            "synchronization_time": 0.0,
        }

        # Communication queues
        self.gradient_queue = queue.Queue()
        self.parameter_queue = queue.Queue()
        self.metric_queue = queue.Queue()

        # Callbacks
        self.training_callbacks: List[Callable] = []
        self.synchronization_callbacks: List[Callable] = []

        self.logger.info("Distributed MARL trainer initialized")

    async def initialize_distributed_training(self) -> None:
        """Initialize distributed training environment."""
        if self.is_initialized:
            self.logger.warning("Distributed training already initialized")
            return

        try:
            self.logger.info(
                "Initializing distributed training (rank %d/%d)",
                self.config.rank,
                self.config.world_size,
            )

            # Set up distributed environment
            await self._setup_distributed_environment()

            # Initialize devices
            self._setup_devices()

            # Set up process group
            await self._setup_process_group()

            # Initialize distributed agents
            await self._setup_distributed_agents()

            self.is_initialized = True
            self.logger.info("Distributed training initialization complete")

        except Exception as e:
            self.logger.error("Failed to initialize distributed training: %s", str(e))
            raise

    async def _setup_distributed_environment(self) -> None:
        """Set up distributed training environment variables."""
        import os

        # Set environment variables for distributed training
        os.environ["MASTER_ADDR"] = self.config.master_addr
        os.environ["MASTER_PORT"] = self.config.master_port
        os.environ["WORLD_SIZE"] = str(self.config.world_size)
        os.environ["RANK"] = str(self.config.rank)
        os.environ["LOCAL_RANK"] = str(self.config.local_rank)

        # CUDA settings
        if self.config.training_mode != TrainingMode.CPU_ONLY:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.config.gpu_ids))

        self.logger.debug("Distributed environment variables set")

    def _setup_devices(self) -> None:
        """Set up training devices."""
        if self.config.training_mode == TrainingMode.CPU_ONLY:
            self.device = torch.device("cpu")
            self.local_device = torch.device("cpu")
        else:
            if torch.cuda.is_available():
                self.local_device = torch.device(f"cuda:{self.config.local_rank}")
                self.device = self.local_device
                torch.cuda.set_device(self.local_device)
            else:
                self.logger.warning("CUDA not available, falling back to CPU")
                self.device = torch.device("cpu")
                self.local_device = torch.device("cpu")

        self.logger.info("Training device: %s", self.device)

    async def _setup_process_group(self) -> None:
        """Set up distributed process group."""
        if self.config.world_size > 1:
            try:
                # Initialize process group
                dist.init_process_group(
                    backend=self.config.backend,
                    world_size=self.config.world_size,
                    rank=self.config.rank,
                    timeout=datetime.timedelta(
                        seconds=self.config.communication_timeout
                    ),
                )

                self.process_group = dist.group.WORLD
                self.logger.info("Process group initialized")

            except Exception as e:
                self.logger.error("Failed to initialize process group: %s", str(e))
                raise
        else:
            self.logger.info("Single process training, no process group needed")

    async def _setup_distributed_agents(self) -> None:
        """Set up distributed versions of agents."""
        for agent_id, agent in self.agents.items():
            try:
                # Move agent to device
                if hasattr(agent, "to"):
                    agent.to(self.device)

                # Wrap with DistributedDataParallel if multi-process
                if self.config.world_size > 1 and hasattr(agent, "parameters"):
                    distributed_agent = DDP(
                        agent,
                        device_ids=[self.config.local_rank]
                        if self.device.type == "cuda"
                        else None,
                        output_device=self.config.local_rank
                        if self.device.type == "cuda"
                        else None,
                        find_unused_parameters=True,
                    )
                    self.distributed_agents[agent_id] = distributed_agent
                else:
                    self.distributed_agents[agent_id] = agent

                self.logger.debug("Set up distributed agent: %s", agent_id)

            except Exception as e:
                self.logger.error(
                    "Failed to set up distributed agent %s: %s", agent_id, str(e)
                )
                raise

    def register_agent(self, agent_id: str, agent: Any) -> None:
        """Register an agent for distributed training."""
        self.agents[agent_id] = agent
        self.logger.info("Registered agent for distributed training: %s", agent_id)

    def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent from distributed training."""
        if agent_id in self.agents:
            del self.agents[agent_id]
        if agent_id in self.distributed_agents:
            del self.distributed_agents[agent_id]
        self.logger.info("Unregistered agent from distributed training: %s", agent_id)

    async def start_distributed_training(
        self, num_epochs: int, steps_per_epoch: int, training_data_loader: Any = None
    ) -> Dict[str, Any]:
        """Start distributed training process.

        Args:
            num_epochs: Number of training epochs
            steps_per_epoch: Steps per epoch
            training_data_loader: Data loader for training

        Returns:
            Training results dictionary
        """
        if not self.is_initialized:
            await self.initialize_distributed_training()

        if self.is_training:
            self.logger.warning("Training already in progress")
            return self.training_metrics

        self.logger.info(
            "Starting distributed training: %d epochs, %d steps/epoch",
            num_epochs,
            steps_per_epoch,
        )

        self.is_training = True
        training_start_time = time.time()

        try:
            for epoch in range(num_epochs):
                self.current_epoch = epoch

                epoch_results = await self._train_epoch(
                    epoch, steps_per_epoch, training_data_loader
                )

                # Update metrics
                self._update_training_metrics(epoch_results)

                # Synchronize across processes
                if epoch % self.config.sync_frequency == 0:
                    await self._synchronize_agents()

                # Save checkpoint
                if epoch % self.config.checkpoint_frequency == 0:
                    await self._save_checkpoint(epoch)

                # Notify callbacks
                await self._notify_training_callbacks(epoch, epoch_results)

                self.logger.info(
                    "Epoch %d complete: avg_reward=%.3f, steps=%d",
                    epoch,
                    epoch_results.get("average_reward", 0.0),
                    epoch_results.get("steps", 0),
                )

            # Final synchronization
            await self._synchronize_agents()

            # Save final checkpoint
            await self._save_checkpoint(num_epochs)

            training_time = time.time() - training_start_time
            self.training_metrics["training_time"] = training_time

            self.logger.info(
                "Distributed training complete: %.2f seconds, %d total steps",
                training_time,
                self.training_metrics["total_steps"],
            )

            return self.training_metrics

        except Exception as e:
            self.logger.error("Distributed training failed: %s", str(e))
            raise
        finally:
            self.is_training = False

    async def _train_epoch(
        self, epoch: int, steps_per_epoch: int, data_loader: Any = None
    ) -> Dict[str, Any]:
        """Train for one epoch."""
        epoch_metrics = {
            "epoch": epoch,
            "steps": 0,
            "total_reward": 0.0,
            "average_reward": 0.0,
            "losses": [],
            "communication_time": 0.0,
        }

        for step in range(steps_per_epoch):
            step_start_time = time.time()

            # Training step for each agent
            step_results = await self._training_step(step)

            # Update metrics
            epoch_metrics["steps"] += 1
            epoch_metrics["total_reward"] += step_results.get("reward", 0.0)
            if "loss" in step_results:
                epoch_metrics["losses"].append(step_results["loss"])

            # Gradient synchronization
            if self.config.sync_strategy == SynchronizationStrategy.SYNCHRONOUS:
                comm_start = time.time()
                await self._synchronize_gradients()
                epoch_metrics["communication_time"] += time.time() - comm_start

            self.global_step += 1

            # Periodic synchronization for semi-synchronous mode
            if (
                self.config.sync_strategy == SynchronizationStrategy.SEMI_SYNCHRONOUS
                and step % self.config.sync_frequency == 0
            ):
                comm_start = time.time()
                await self._synchronize_gradients()
                epoch_metrics["communication_time"] += time.time() - comm_start

        # Calculate average reward
        if epoch_metrics["steps"] > 0:
            epoch_metrics["average_reward"] = (
                epoch_metrics["total_reward"] / epoch_metrics["steps"]
            )

        return epoch_metrics

    async def _training_step(self, step: int) -> Dict[str, Any]:
        """Execute one training step."""
        step_results = {"step": step, "reward": 0.0, "loss": 0.0, "agent_results": {}}

        # Train each agent
        for agent_id, agent in self.distributed_agents.items():
            try:
                # This would typically involve:
                # 1. Getting batch of experiences
                # 2. Forward pass
                # 3. Loss calculation
                # 4. Backward pass
                # 5. Gradient clipping

                # Simulate training step
                import numpy as np

                agent_reward = np.random.normal(0.5, 0.1)
                agent_loss = np.random.exponential(0.1)

                step_results["agent_results"][agent_id] = {
                    "reward": agent_reward,
                    "loss": agent_loss,
                }

                step_results["reward"] += agent_reward
                step_results["loss"] += agent_loss

            except Exception as e:
                self.logger.error(
                    "Training step failed for agent %s: %s", agent_id, str(e)
                )
                raise

        # Average across agents
        if len(self.distributed_agents) > 0:
            step_results["reward"] /= len(self.distributed_agents)
            step_results["loss"] /= len(self.distributed_agents)

        return step_results

    async def _synchronize_gradients(self) -> None:
        """Synchronize gradients across processes."""
        if self.config.world_size <= 1:
            return

        try:
            for agent_id, agent in self.distributed_agents.items():
                if hasattr(agent, "parameters"):
                    # Gradient clipping
                    if self.config.gradient_clipping > 0:
                        torch.nn.utils.clip_grad_norm_(
                            agent.parameters(), self.config.gradient_clipping
                        )

                    # All-reduce gradients
                    for param in agent.parameters():
                        if param.grad is not None:
                            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                            param.grad.data /= self.config.world_size

            self.logger.debug("Gradient synchronization complete")

        except Exception as e:
            self.logger.error("Gradient synchronization failed: %s", str(e))
            raise

    async def _synchronize_agents(self) -> None:
        """Synchronize agent parameters across processes."""
        if self.config.world_size <= 1:
            return

        sync_start_time = time.time()

        try:
            for agent_id, agent in self.distributed_agents.items():
                if hasattr(agent, "parameters"):
                    # Synchronize parameters
                    for param in agent.parameters():
                        dist.broadcast(param.data, src=0)

            sync_time = time.time() - sync_start_time
            self.training_metrics["synchronization_time"] += sync_time

            # Notify synchronization callbacks
            await self._notify_synchronization_callbacks()

            self.logger.debug("Agent synchronization complete (%.3fs)", sync_time)

        except Exception as e:
            self.logger.error("Agent synchronization failed: %s", str(e))
            raise

    async def _save_checkpoint(self, epoch: int) -> None:
        """Save training checkpoint."""
        if self.config.rank != 0:  # Only master process saves checkpoints
            return

        try:
            checkpoint_dir = Path(self.config.checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"

            checkpoint_data = {
                "epoch": epoch,
                "global_step": self.global_step,
                "training_metrics": self.training_metrics,
                "config": self.config.__dict__,
                "agents": {},
            }

            # Save agent states
            for agent_id, agent in self.agents.items():
                if hasattr(agent, "state_dict"):
                    checkpoint_data["agents"][agent_id] = agent.state_dict()

            torch.save(checkpoint_data, checkpoint_path)

            self.logger.info("Checkpoint saved: %s", checkpoint_path)

        except Exception as e:
            self.logger.error("Failed to save checkpoint: %s", str(e))

    async def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load training checkpoint."""
        try:
            checkpoint_data = torch.load(checkpoint_path, map_location=self.device)

            self.current_epoch = checkpoint_data["epoch"]
            self.global_step = checkpoint_data["global_step"]
            self.training_metrics = checkpoint_data["training_metrics"]

            # Load agent states
            for agent_id, state_dict in checkpoint_data["agents"].items():
                if agent_id in self.agents and hasattr(
                    self.agents[agent_id], "load_state_dict"
                ):
                    self.agents[agent_id].load_state_dict(state_dict)

            self.logger.info("Checkpoint loaded: %s", checkpoint_path)

        except Exception as e:
            self.logger.error("Failed to load checkpoint: %s", str(e))
            raise

    def _update_training_metrics(self, epoch_results: Dict[str, Any]) -> None:
        """Update training metrics."""
        self.training_metrics["total_steps"] += epoch_results.get("steps", 0)
        self.training_metrics["total_episodes"] += 1

        # Update average reward (exponential moving average)
        new_reward = epoch_results.get("average_reward", 0.0)
        alpha = 0.1  # Smoothing factor
        self.training_metrics["average_reward"] = (
            alpha * new_reward + (1 - alpha) * self.training_metrics["average_reward"]
        )

        # Update communication time
        self.training_metrics["communication_time"] += epoch_results.get(
            "communication_time", 0.0
        )

    async def _notify_training_callbacks(
        self, epoch: int, results: Dict[str, Any]
    ) -> None:
        """Notify training callbacks."""
        for callback in self.training_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(epoch, results)
                else:
                    callback(epoch, results)
            except Exception as e:
                self.logger.error("Training callback error: %s", str(e))

    async def _notify_synchronization_callbacks(self) -> None:
        """Notify synchronization callbacks."""
        for callback in self.synchronization_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback()
                else:
                    callback()
            except Exception as e:
                self.logger.error("Synchronization callback error: %s", str(e))

    def add_training_callback(self, callback: Callable) -> None:
        """Add training callback."""
        self.training_callbacks.append(callback)

    def add_synchronization_callback(self, callback: Callable) -> None:
        """Add synchronization callback."""
        self.synchronization_callbacks.append(callback)

    def get_training_metrics(self) -> Dict[str, Any]:
        """Get current training metrics."""
        return self.training_metrics.copy()

    def get_distributed_info(self) -> Dict[str, Any]:
        """Get distributed training information."""
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
        """Stop distributed training."""
        if self.is_training:
            self.is_training = False
            self.logger.info("Distributed training stopped")

    async def shutdown(self) -> None:
        """Shutdown distributed training system."""
        await self.stop_training()

        # Cleanup distributed resources
        if self.config.world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()

        # Clear agents
        self.agents.clear()
        self.distributed_agents.clear()

        # Clear callbacks
        self.training_callbacks.clear()
        self.synchronization_callbacks.clear()

        self.is_initialized = False
        self.logger.info("Distributed MARL trainer shutdown complete")


class DistributedTrainerFactory:
    """Factory for creating distributed MARL trainers."""

    @staticmethod
    def create_single_gpu_trainer() -> DistributedMARLTrainer:
        """Create single GPU trainer."""
        config = DistributedTrainingConfig(
            training_mode=TrainingMode.SINGLE_GPU, world_size=1, rank=0
        )
        return DistributedMARLTrainer(config)

    @staticmethod
    def create_multi_gpu_trainer(gpu_ids: List[int]) -> DistributedMARLTrainer:
        """Create multi-GPU trainer."""
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
        """Create multi-node trainer."""
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
        """Create CPU-only trainer."""
        config = DistributedTrainingConfig(
            training_mode=TrainingMode.CPU_ONLY,
            world_size=1,
            rank=0,
            cpu_workers=cpu_workers,
            backend="gloo",  # Use gloo backend for CPU
        )
        return DistributedMARLTrainer(config)


# Utility functions for distributed training
def setup_distributed_environment(
    rank: int,
    world_size: int,
    master_addr: str = "localhost",
    master_port: str = "12355",
) -> None:
    """Set up distributed training environment."""

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)


def cleanup_distributed_environment() -> None:
    """Clean up distributed training environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_distributed_info() -> Dict[str, Any]:
    """Get current distributed training information."""
    if not dist.is_initialized():
        return {"initialized": False, "world_size": 1, "rank": 0, "local_rank": 0}

    return {
        "initialized": True,
        "world_size": dist.get_world_size(),
        "rank": dist.get_rank(),
        "local_rank": int(os.environ.get("LOCAL_RANK", 0)),
    }


def is_main_process() -> bool:
    """Check if current process is the main process."""
    return not dist.is_initialized() or dist.get_rank() == 0


def synchronize_processes() -> None:
    """Synchronize all processes."""
    if dist.is_initialized():
        dist.barrier()
