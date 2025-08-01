"""Distributed MARL Module.

This module provides distributed training and deployment capabilities
for the multi-agent reinforcement learning coordination system.
"""

from .distributed_coordinator import (
    DistributedCoordinationConfig,
    DistributedCoordinator,
)
from .distributed_trainer import DistributedMARLTrainer, DistributedTrainingConfig
from .network_coordinator import NetworkConfig, NetworkCoordinator
from .resource_manager import ResourceConfig, ResourceManager
from .scalability_manager import ScalabilityManager, ScalingConfig

__all__ = [
    "DistributedMARLTrainer",
    "DistributedTrainingConfig",
    "DistributedCoordinator",
    "DistributedCoordinationConfig",
    "ResourceManager",
    "ResourceConfig",
    "NetworkCoordinator",
    "NetworkConfig",
    "ScalabilityManager",
    "ScalingConfig",
]
