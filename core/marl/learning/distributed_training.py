"""
Distributed Training Infrastructure

This module provides distributed training capabilities for multi-agent
reinforcement learning, enabling scalable training across multiple nodes
and GPUs while maintaining coordination effectiveness.
"""

import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from ..config import MARLConfig
from ..exceptions import AgentFailureError, MARLError
from ..logging_config import get_marl_logger

logger = logging.getLogger(__name__)


class TrainingNodeType(Enum):
    """Types of training nodes in distributed setup."""

    MASTER = "master"
    WORKER = "worker"
    PARAMETER_SERVER = "parameter_server"


@dataclass
class TrainingNode:
    """Represents a node in the distributed training setup."""

    node_id: str
    node_type: TrainingNodeType
    rank: int
    world_size: int
    gpu_id: Optional[int] = None
    ip_address: str = "localhost"
    port: int = 29500
    status: str = "initialized"
    last_heartbeat: float = field(default_factory=time.time)

    def __post_init__(self):
        """Post-initialization setup."""
        if self.gpu_id is not None and torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.gpu_id}")
        else:
            self.device = torch.device("cpu")

    def update_heartbeat(self) -> None:
        """Update the last heartbeat timestamp."""
        self.last_heartbeat = time.time()

    def is_alive(self, timeout_seconds: float = 30.0) -> bool:
        """Check if node is alive based on heartbeat."""
        return time.time() - self.last_heartbeat < timeout_seconds


class DistributedTrainingManager:
    """
    Manager for distributed multi-agent reinforcement learning training.

    Coordinates training across multiple nodes while maintaining the
    effectiveness of multi-agent coordination and learning.
    """

    def __init__(self, config: MARLConfig, node_config: TrainingNode):
        """
        Initialize distributed training manager.

        Args:
            config: MARL configuration
            node_config: Configuration for this training node
        """
        self.config = config
        self.node = node_config
        self.logger = get_marl_logger(f"distributed.{node_config.node_id}")

        # Distributed training state
        self.is_initialized = False
        self.training_active = False
        self.nodes: Dict[str, TrainingNode] = {node_config.node_id: node_config}

        # Synchronization
        self.sync_frequency = getattr(config, "sync_frequency", 100)
        self.last_sync_step = 0
        self.sync_lock = threading.Lock()

        # Performance tracking
        self.training_metrics = {
            "total_steps": 0,
            "sync_operations": 0,
            "communication_time": 0.0,
            "training_time": 0.0,
            "throughput": 0.0,
        }

        self.logger.log_distributed_training(
            node_config.rank, config.num_workers, "initialized"
        )

    def initialize_distributed_training(self) -> None:
        """Initialize distributed training environment."""
        try:
            if self.config.distributed_training and self.node.world_size > 1:
                # Initialize process group
                dist.init_process_group(
                    backend="nccl" if torch.cuda.is_available() else "gloo",
                    init_method=f"tcp://{self.node.ip_address}:{self.node.port}",
                    world_size=self.node.world_size,
                    rank=self.node.rank,
                )

                self.is_initialized = True
                self.node.status = "connected"

                self.logger.log_distributed_training(
                    self.node.rank, self.node.world_size, "connected"
                )
            else:
                # Single node training
                self.is_initialized = True
                self.node.status = "single_node"

                self.logger.log_distributed_training(0, 1, "single_node")

        except Exception as e:
            error_msg = f"Failed to initialize distributed training for node {self.node.node_id}"
            self.logger.log_error_with_context(
                e,
                {
                    "node_id": self.node.node_id,
                    "rank": self.node.rank,
                    "world_size": self.node.world_size,
                },
            )
            raise MARLError(error_msg) from e

    def start_training(self, training_function: Callable, *args, **kwargs) -> None:
        """
        Start distributed training.

        Args:
            training_function: Function to execute for training
            *args: Arguments for training function
            **kwargs: Keyword arguments for training function
        """
        if not self.is_initialized:
            self.initialize_distributed_training()

        try:
            self.training_active = True
            self.node.status = "training"

            # Start heartbeat thread
            heartbeat_thread = threading.Thread(
                target=self._heartbeat_loop, daemon=True
            )
            heartbeat_thread.start()

            # Execute training
            start_time = time.time()
            result = training_function(*args, **kwargs)
            training_time = time.time() - start_time

            # Update metrics
            self.training_metrics["training_time"] += training_time
            self.training_metrics["throughput"] = (
                self.training_metrics["total_steps"] / training_time
                if training_time > 0
                else 0.0
            )

            self.node.status = "completed"
            self.training_active = False

            self.logger.log_distributed_training(
                self.node.rank, self.node.world_size, "completed"
            )

            return result

        except Exception as e:
            self.training_active = False
            self.node.status = "failed"

            error_msg = f"Distributed training failed for node {self.node.node_id}"
            self.logger.log_error_with_context(
                e,
                {
                    "node_id": self.node.node_id,
                    "training_metrics": self.training_metrics,
                },
            )
            raise AgentFailureError(
                error_msg,
                agent_id=self.node.node_id,
                failure_type="distributed_training",
            ) from e

    def synchronize_parameters(self, model: torch.nn.Module) -> None:
        """
        Synchronize model parameters across all nodes.

        Args:
            model: PyTorch model to synchronize
        """
        if not self.is_initialized or self.node.world_size <= 1:
            return

        try:
            with self.sync_lock:
                sync_start = time.time()

                # Average gradients across all nodes
                for param in model.parameters():
                    if param.grad is not None:
                        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                        param.grad.data /= self.node.world_size

                sync_time = time.time() - sync_start

                # Update metrics
                self.training_metrics["sync_operations"] += 1
                self.training_metrics["communication_time"] += sync_time
                self.last_sync_step = self.training_metrics["total_steps"]

                self.logger.log_distributed_training(
                    self.node.rank, self.node.world_size, "synchronized"
                )

        except Exception as e:
            error_msg = f"Parameter synchronization failed for node {self.node.node_id}"
            self.logger.log_error_with_context(
                e,
                {
                    "node_id": self.node.node_id,
                    "sync_step": self.training_metrics["total_steps"],
                },
            )
            raise MARLError(error_msg) from e

    def should_synchronize(self) -> bool:
        """Check if parameters should be synchronized."""
        steps_since_sync = self.training_metrics["total_steps"] - self.last_sync_step
        return steps_since_sync >= self.sync_frequency

    def broadcast_coordination_state(self, coordination_state: Dict[str, Any]) -> None:
        """
        Broadcast coordination state to all nodes.

        Args:
            coordination_state: State to broadcast
        """
        if not self.is_initialized or self.node.world_size <= 1:
            return

        try:
            # Convert state to tensor for broadcasting
            state_tensor = self._dict_to_tensor(coordination_state)

            # Broadcast from master node
            if self.node.node_type == TrainingNodeType.MASTER:
                dist.broadcast(state_tensor, src=0)
            else:
                dist.broadcast(state_tensor, src=0)
                # Convert back to dict
                coordination_state.update(self._tensor_to_dict(state_tensor))

            self.logger.log_distributed_training(
                self.node.rank, self.node.world_size, "coordination_broadcast"
            )

        except Exception as e:
            error_msg = (
                f"Coordination state broadcast failed for node {self.node.node_id}"
            )
            self.logger.log_error_with_context(
                e,
                {
                    "node_id": self.node.node_id,
                    "state_keys": list(coordination_state.keys()),
                },
            )
            # Don't raise exception for coordination broadcast failures
            # as training can continue without perfect coordination

    def gather_training_metrics(self) -> Dict[str, Any]:
        """Gather training metrics from all nodes."""
        if not self.is_initialized or self.node.world_size <= 1:
            return self.training_metrics

        try:
            # Create tensor with local metrics
            metrics_tensor = torch.tensor(
                [
                    self.training_metrics["total_steps"],
                    self.training_metrics["sync_operations"],
                    self.training_metrics["communication_time"],
                    self.training_metrics["training_time"],
                    self.training_metrics["throughput"],
                ],
                dtype=torch.float32,
            )

            # Gather metrics from all nodes
            gathered_metrics = [
                torch.zeros_like(metrics_tensor) for _ in range(self.node.world_size)
            ]
            dist.all_gather(gathered_metrics, metrics_tensor)

            # Aggregate metrics
            aggregated_metrics = {
                "total_steps": sum(m[0].item() for m in gathered_metrics),
                "sync_operations": sum(m[1].item() for m in gathered_metrics),
                "communication_time": sum(m[2].item() for m in gathered_metrics),
                "training_time": sum(m[3].item() for m in gathered_metrics),
                "average_throughput": sum(m[4].item() for m in gathered_metrics)
                / len(gathered_metrics),
                "node_count": len(gathered_metrics),
            }

            return aggregated_metrics

        except Exception as e:
            self.logger.log_error_with_context(
                e, {"node_id": self.node.node_id, "operation": "gather_metrics"}
            )
            return self.training_metrics

    def _heartbeat_loop(self) -> None:
        """Heartbeat loop for node health monitoring."""
        while self.training_active:
            try:
                self.node.update_heartbeat()

                # Check other nodes' health (if master)
                if self.node.node_type == TrainingNodeType.MASTER:
                    self._check_node_health()

                time.sleep(5.0)  # Heartbeat every 5 seconds

            except Exception as e:
                self.logger.log_error_with_context(
                    e, {"node_id": self.node.node_id, "operation": "heartbeat"}
                )
                break

    def _check_node_health(self) -> None:
        """Check health of all nodes (master node only)."""
        unhealthy_nodes = []

        for node_id, node in self.nodes.items():
            if not node.is_alive() and node.node_id != self.node.node_id:
                unhealthy_nodes.append(node_id)
                node.status = "unhealthy"

        if unhealthy_nodes:
            self.logger.log_distributed_training(
                self.node.rank,
                self.node.world_size,
                f"unhealthy_nodes: {', '.join(unhealthy_nodes)}",
            )

    def _dict_to_tensor(self, data: Dict[str, Any]) -> torch.Tensor:
        """Convert dictionary to tensor for broadcasting."""
        # Simple implementation - can be enhanced for complex data structures
        values = []
        for key in sorted(data.keys()):
            if isinstance(data[key], (int, float)):
                values.append(float(data[key]))
            elif isinstance(data[key], bool):
                values.append(float(data[key]))

        return torch.tensor(values, dtype=torch.float32)

    def _tensor_to_dict(self, tensor: torch.Tensor) -> Dict[str, Any]:
        """Convert tensor back to dictionary."""
        # Simple implementation - would need enhancement for real use
        return {"broadcast_data": tensor.tolist()}

    def cleanup(self) -> None:
        """Clean up distributed training resources."""
        try:
            self.training_active = False

            if self.is_initialized and dist.is_initialized():
                dist.destroy_process_group()

            self.node.status = "cleanup"

            self.logger.log_distributed_training(
                self.node.rank, self.node.world_size, "cleanup"
            )

        except Exception as e:
            self.logger.log_error_with_context(
                e, {"node_id": self.node.node_id, "operation": "cleanup"}
            )

    def get_training_status(self) -> Dict[str, Any]:
        """Get comprehensive training status."""
        return {
            "node_info": {
                "node_id": self.node.node_id,
                "node_type": self.node.node_type.value,
                "rank": self.node.rank,
                "world_size": self.node.world_size,
                "status": self.node.status,
                "device": str(self.node.device),
            },
            "training_state": {
                "is_initialized": self.is_initialized,
                "training_active": self.training_active,
                "last_sync_step": self.last_sync_step,
                "sync_frequency": self.sync_frequency,
            },
            "metrics": self.training_metrics,
            "node_health": {
                node_id: {
                    "status": node.status,
                    "last_heartbeat": node.last_heartbeat,
                    "is_alive": node.is_alive(),
                }
                for node_id, node in self.nodes.items()
            },
        }


class DistributedCoordinationManager:
    """
    Manager for coordinating multi-agent learning across distributed nodes.

    Ensures that coordination mechanisms work effectively even when agents
    are distributed across multiple training nodes.
    """

    def __init__(self, training_manager: DistributedTrainingManager):
        """
        Initialize distributed coordination manager.

        Args:
            training_manager: Distributed training manager
        """
        self.training_manager = training_manager
        self.node = training_manager.node
        self.logger = get_marl_logger(f"coord_dist.{self.node.node_id}")

        # Coordination state
        self.global_coordination_state = {}
        self.local_coordination_state = {}
        self.coordination_lock = threading.Lock()

        # Consensus tracking
        self.consensus_proposals = {}
        self.consensus_votes = {}

    def propose_coordination_action(
        self, agent_id: str, action_proposal: Dict[str, Any]
    ) -> str:
        """
        Propose a coordination action across distributed nodes.

        Args:
            agent_id: ID of the proposing agent
            action_proposal: Proposed action details

        Returns:
            Proposal ID for tracking
        """
        proposal_id = f"{self.node.node_id}_{agent_id}_{int(time.time() * 1000)}"

        with self.coordination_lock:
            self.consensus_proposals[proposal_id] = {
                "agent_id": agent_id,
                "node_id": self.node.node_id,
                "proposal": action_proposal,
                "timestamp": time.time(),
                "votes": {},
                "status": "proposed",
            }

        # Broadcast proposal to other nodes
        if self.training_manager.is_initialized:
            broadcast_data = {
                "type": "coordination_proposal",
                "proposal_id": proposal_id,
                "agent_id": agent_id,
                "node_id": self.node.node_id,
            }
            self.training_manager.broadcast_coordination_state(broadcast_data)

        self.logger.log_coordination_start(proposal_id, [agent_id])

        return proposal_id

    def vote_on_proposal(self, proposal_id: str, agent_id: str, vote: bool) -> None:
        """
        Vote on a coordination proposal.

        Args:
            proposal_id: ID of the proposal
            agent_id: ID of the voting agent
            vote: True for approve, False for reject
        """
        with self.coordination_lock:
            if proposal_id in self.consensus_proposals:
                self.consensus_proposals[proposal_id]["votes"][agent_id] = vote

                # Check if consensus is reached
                proposal = self.consensus_proposals[proposal_id]
                votes = proposal["votes"]

                if len(votes) >= self.training_manager.node.world_size:
                    # Determine consensus
                    approve_votes = sum(1 for v in votes.values() if v)
                    total_votes = len(votes)

                    if approve_votes > total_votes / 2:
                        proposal["status"] = "approved"
                        self.logger.log_coordination_success(
                            proposal_id,
                            time.time() - proposal["timestamp"],
                            approve_votes / total_votes,
                        )
                    else:
                        proposal["status"] = "rejected"
                        self.logger.log_coordination_failure(
                            proposal_id, "consensus", "Insufficient votes"
                        )

    def get_coordination_consensus(
        self, proposal_id: str, timeout_seconds: float = 30.0
    ) -> Optional[bool]:
        """
        Get consensus result for a proposal.

        Args:
            proposal_id: ID of the proposal
            timeout_seconds: Maximum time to wait for consensus

        Returns:
            True if approved, False if rejected, None if timeout
        """
        start_time = time.time()

        while time.time() - start_time < timeout_seconds:
            with self.coordination_lock:
                if proposal_id in self.consensus_proposals:
                    proposal = self.consensus_proposals[proposal_id]

                    if proposal["status"] == "approved":
                        return True
                    elif proposal["status"] == "rejected":
                        return False

            time.sleep(0.1)  # Check every 100ms

        # Timeout
        self.logger.log_coordination_failure(
            proposal_id, "timeout", "Consensus timeout"
        )
        return None

    def update_global_coordination_state(self, state_update: Dict[str, Any]) -> None:
        """
        Update global coordination state.

        Args:
            state_update: State updates to apply
        """
        with self.coordination_lock:
            self.global_coordination_state.update(state_update)

        # Broadcast update to other nodes
        if self.training_manager.is_initialized:
            broadcast_data = {
                "type": "state_update",
                "updates": state_update,
                "node_id": self.node.node_id,
            }
            self.training_manager.broadcast_coordination_state(broadcast_data)

    def get_coordination_status(self) -> Dict[str, Any]:
        """Get coordination status across distributed nodes."""
        with self.coordination_lock:
            active_proposals = {
                pid: proposal
                for pid, proposal in self.consensus_proposals.items()
                if proposal["status"] == "proposed"
            }

            return {
                "node_id": self.node.node_id,
                "global_state_size": len(self.global_coordination_state),
                "local_state_size": len(self.local_coordination_state),
                "active_proposals": len(active_proposals),
                "total_proposals": len(self.consensus_proposals),
                "recent_proposals": list(active_proposals.keys()),
            }
