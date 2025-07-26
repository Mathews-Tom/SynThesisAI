"""Unit tests for distributed MARL components."""

import asyncio
import time
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch

from core.marl.distributed.distributed_coordinator import (
    DistributedCoordinationConfig,
    DistributedCoordinationMode,
    DistributedCoordinator,
    NetworkTopology,
)
from core.marl.distributed.distributed_trainer import (
    DistributedMARLTrainer,
    DistributedTrainerFactory,
    DistributedTrainingConfig,
    SynchronizationStrategy,
    TrainingMode,
)
from core.marl.distributed.network_coordinator import (
    MessageType,
    NetworkConfig,
    NetworkCoordinator,
    NetworkMessage,
    NetworkProtocol,
)
from core.marl.distributed.resource_manager import (
    AllocationStrategy,
    ResourceConfig,
    ResourceManager,
    ResourceType,
)
from core.marl.distributed.scalability_manager import (
    DeploymentStrategy,
    ScalabilityManager,
    ScalingConfig,
    ScalingDirection,
    ScalingTrigger,
)


class TestDistributedTrainingConfig:
    """Test distributed training configuration."""

    def test_default_config(self):
        """Test default configuration."""
        config = DistributedTrainingConfig()

        assert config.training_mode == TrainingMode.SINGLE_GPU
        assert config.world_size == 1
        assert config.rank == 0
        assert config.sync_strategy == SynchronizationStrategy.SYNCHRONOUS
        assert config.backend == "nccl"

    def test_config_validation(self):
        """Test configuration validation."""
        # Test invalid world size
        with pytest.raises(ValueError, match="World size must be positive"):
            DistributedTrainingConfig(world_size=0)

        # Test invalid rank
        with pytest.raises(ValueError, match="Rank must be between"):
            DistributedTrainingConfig(world_size=2, rank=2)

    def test_multi_gpu_auto_detection(self):
        """Test multi-GPU auto-detection."""
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.device_count", return_value=4),
        ):
            config = DistributedTrainingConfig(
                training_mode=TrainingMode.MULTI_GPU, world_size=4
            )

            assert config.gpu_ids == [0, 1, 2, 3]


class TestDistributedMARLTrainer:
    """Test distributed MARL trainer."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return DistributedTrainingConfig(
            training_mode=TrainingMode.CPU_ONLY, world_size=1, backend="gloo"
        )

    @pytest.fixture
    def trainer(self, config):
        """Create test trainer."""
        return DistributedMARLTrainer(config)

    def test_initialization(self, trainer):
        """Test trainer initialization."""
        assert not trainer.is_initialized
        assert not trainer.is_training
        assert trainer.current_epoch == 0
        assert trainer.global_step == 0
        assert len(trainer.agents) == 0

    def test_agent_registration(self, trainer):
        """Test agent registration."""
        mock_agent = Mock()

        trainer.register_agent("test_agent", mock_agent)

        assert "test_agent" in trainer.agents
        assert trainer.agents["test_agent"] == mock_agent

    def test_agent_unregistration(self, trainer):
        """Test agent unregistration."""
        mock_agent = Mock()
        trainer.register_agent("test_agent", mock_agent)

        trainer.unregister_agent("test_agent")

        assert "test_agent" not in trainer.agents

    @pytest.mark.asyncio
    async def test_initialization_process(self, trainer):
        """Test initialization process."""
        with (
            patch.object(trainer, "_setup_distributed_environment"),
            patch.object(trainer, "_setup_devices"),
            patch.object(trainer, "_setup_process_group"),
            patch.object(trainer, "_setup_distributed_agents"),
        ):
            await trainer.initialize_distributed_training()

            assert trainer.is_initialized

    def test_training_metrics(self, trainer):
        """Test training metrics."""
        metrics = trainer.get_training_metrics()

        assert "total_steps" in metrics
        assert "total_episodes" in metrics
        assert "average_reward" in metrics
        assert "training_time" in metrics

    def test_distributed_info(self, trainer):
        """Test distributed info."""
        info = trainer.get_distributed_info()

        assert "world_size" in info
        assert "rank" in info
        assert "training_mode" in info
        assert "is_initialized" in info


class TestDistributedTrainerFactory:
    """Test distributed trainer factory."""

    def test_single_gpu_trainer(self):
        """Test single GPU trainer creation."""
        trainer = DistributedTrainerFactory.create_single_gpu_trainer()

        assert trainer.config.training_mode == TrainingMode.SINGLE_GPU
        assert trainer.config.world_size == 1

    def test_multi_gpu_trainer(self):
        """Test multi-GPU trainer creation."""
        gpu_ids = [0, 1, 2, 3]
        trainer = DistributedTrainerFactory.create_multi_gpu_trainer(gpu_ids)

        assert trainer.config.training_mode == TrainingMode.MULTI_GPU
        assert trainer.config.world_size == len(gpu_ids)
        assert trainer.config.gpu_ids == gpu_ids

    def test_cpu_trainer(self):
        """Test CPU trainer creation."""
        trainer = DistributedTrainerFactory.create_cpu_trainer(cpu_workers=8)

        assert trainer.config.training_mode == TrainingMode.CPU_ONLY
        assert trainer.config.cpu_workers == 8
        assert trainer.config.backend == "gloo"


class TestDistributedCoordinationConfig:
    """Test distributed coordination configuration."""

    def test_default_config(self):
        """Test default configuration."""
        config = DistributedCoordinationConfig()

        assert config.coordination_mode == DistributedCoordinationMode.CENTRALIZED
        assert config.network_topology == NetworkTopology.STAR
        assert config.consensus_threshold == 0.7
        assert config.enable_fault_tolerance is True

    def test_config_validation(self):
        """Test configuration validation."""
        # Test invalid consensus threshold
        with pytest.raises(
            ValueError, match="Consensus threshold must be between 0 and 1"
        ):
            DistributedCoordinationConfig(consensus_threshold=1.5)

        # Test invalid coordination timeout
        with pytest.raises(ValueError, match="Coordination timeout must be positive"):
            DistributedCoordinationConfig(coordination_timeout=-1.0)


class TestDistributedCoordinator:
    """Test distributed coordinator."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return DistributedCoordinationConfig(
            coordination_mode=DistributedCoordinationMode.CENTRALIZED,
            network_topology=NetworkTopology.STAR,
        )

    @pytest.fixture
    def coordinator(self, config):
        """Create test coordinator."""
        return DistributedCoordinator(config)

    def test_initialization(self, coordinator):
        """Test coordinator initialization."""
        assert not coordinator.is_initialized
        assert not coordinator.is_active
        assert len(coordinator.connected_nodes) == 0
        assert len(coordinator.active_coordinations) == 0

    @pytest.mark.asyncio
    async def test_initialization_process(self, coordinator):
        """Test initialization process."""
        with (
            patch.object(coordinator, "_initialize_coordination_components"),
            patch.object(coordinator, "_setup_network_topology"),
            patch.object(coordinator, "_start_network_services"),
            patch.object(coordinator, "_determine_master_node"),
            patch.object(coordinator, "_start_coordination_services"),
        ):
            await coordinator.initialize_distributed_coordination()

            assert coordinator.is_initialized
            assert coordinator.is_active

    def test_coordination_metrics(self, coordinator):
        """Test coordination metrics."""
        metrics = coordinator.get_coordination_metrics()

        assert "total_coordinations" in metrics
        assert "successful_coordinations" in metrics
        assert "failed_coordinations" in metrics
        assert "average_coordination_time" in metrics

    def test_network_status(self, coordinator):
        """Test network status."""
        status = coordinator.get_network_status()

        assert "node_id" in status
        assert "is_master" in status
        assert "connected_nodes" in status
        assert "network_topology" in status


class TestResourceConfig:
    """Test resource configuration."""

    def test_default_config(self):
        """Test default configuration."""
        config = ResourceConfig()

        assert config.max_cpu_percent == 80.0
        assert config.max_memory_percent == 80.0
        assert config.allocation_strategy == AllocationStrategy.LOAD_BALANCED
        assert config.enable_auto_scaling is True

    def test_config_validation(self):
        """Test configuration validation."""
        # Test invalid percentage
        with pytest.raises(
            ValueError, match="Resource percentages must be between 0 and 100"
        ):
            ResourceConfig(max_cpu_percent=150.0)


class TestResourceManager:
    """Test resource manager."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return ResourceConfig(
            max_cpu_percent=70.0, max_memory_percent=70.0, monitoring_interval=5.0
        )

    @pytest.fixture
    def manager(self, config):
        """Create test resource manager."""
        return ResourceManager(config)

    def test_initialization(self, manager):
        """Test manager initialization."""
        assert not manager.is_active
        assert len(manager.resource_allocations) == 0
        assert len(manager.resource_metrics_history) == 0

    def test_system_info(self, manager):
        """Test system information collection."""
        system_info = manager.system_info

        assert "cpu_count" in system_info
        assert "memory_total" in system_info
        assert "gpu_count" in system_info

    def test_available_resources(self, manager):
        """Test available resources."""
        resources = manager.available_resources

        assert "cpu_cores" in resources
        assert "memory_gb" in resources
        assert "storage_gb" in resources
        assert "gpu_count" in resources

    @pytest.mark.asyncio
    async def test_resource_allocation(self, manager):
        """Test resource allocation."""
        allocation_id = await manager.allocate_resource(
            ResourceType.CPU, 2.0, "test_owner", priority=1
        )

        assert allocation_id is not None
        assert allocation_id in manager.resource_allocations

        allocation = manager.resource_allocations[allocation_id]
        assert allocation.resource_type == ResourceType.CPU
        assert allocation.allocated_amount == 2.0
        assert allocation.owner == "test_owner"

    @pytest.mark.asyncio
    async def test_resource_deallocation(self, manager):
        """Test resource deallocation."""
        allocation_id = await manager.allocate_resource(
            ResourceType.MEMORY, 4.0, "test_owner"
        )

        success = await manager.deallocate_resource(allocation_id)

        assert success
        assert allocation_id not in manager.resource_allocations

    def test_resource_usage(self, manager):
        """Test resource usage information."""
        usage = manager.get_resource_usage()

        assert "usage_by_type" in usage
        assert "total_allocations" in usage
        assert "allocation_details" in usage


class TestNetworkConfig:
    """Test network configuration."""

    def test_default_config(self):
        """Test default configuration."""
        config = NetworkConfig()

        assert config.host == "localhost"
        assert config.port == 8080
        assert config.protocol == NetworkProtocol.TCP
        assert config.max_connections == 100

    def test_config_validation(self):
        """Test configuration validation."""
        # Test invalid port
        with pytest.raises(ValueError, match="Port must be between 1 and 65535"):
            NetworkConfig(port=0)

        # Test invalid max connections
        with pytest.raises(ValueError, match="Max connections must be positive"):
            NetworkConfig(max_connections=0)


class TestNetworkMessage:
    """Test network message."""

    def test_message_creation(self):
        """Test message creation."""
        message = NetworkMessage(
            message_id="test_id",
            message_type=MessageType.HEARTBEAT,
            sender_id="sender",
            recipient_id="recipient",
            payload={"data": "test"},
            timestamp=time.time(),
        )

        assert message.message_id == "test_id"
        assert message.message_type == MessageType.HEARTBEAT
        assert message.sender_id == "sender"
        assert message.recipient_id == "recipient"
        assert message.payload == {"data": "test"}

    def test_message_serialization(self):
        """Test message serialization."""
        message = NetworkMessage(
            message_id="test_id",
            message_type=MessageType.COORDINATION_REQUEST,
            sender_id="sender",
            recipient_id=None,
            payload={"action": "test"},
            timestamp=time.time(),
        )

        # Test to_dict
        message_dict = message.to_dict()
        assert message_dict["message_id"] == "test_id"
        assert message_dict["message_type"] == "coordination_request"

        # Test from_dict
        reconstructed = NetworkMessage.from_dict(message_dict)
        assert reconstructed.message_id == message.message_id
        assert reconstructed.message_type == message.message_type


class TestNetworkCoordinator:
    """Test network coordinator."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return NetworkConfig(host="localhost", port=8080, protocol=NetworkProtocol.TCP)

    @pytest.fixture
    def coordinator(self, config):
        """Create test network coordinator."""
        return NetworkCoordinator(config, "test_node")

    def test_initialization(self, coordinator):
        """Test coordinator initialization."""
        assert not coordinator.is_active
        assert len(coordinator.connections) == 0
        assert coordinator.node_id == "test_node"

    @pytest.mark.asyncio
    async def test_message_sending(self, coordinator):
        """Test message sending."""
        message_id = await coordinator.send_message(
            MessageType.HEARTBEAT, {"status": "active"}, recipient_id="target_node"
        )

        assert message_id is not None

    def test_message_handler_registration(self, coordinator):
        """Test message handler registration."""
        handler = Mock()

        coordinator.add_message_handler(MessageType.HEARTBEAT, handler)

        assert handler in coordinator.message_handlers[MessageType.HEARTBEAT]

    def test_network_status(self, coordinator):
        """Test network status."""
        status = coordinator.get_network_status()

        assert "node_id" in status
        assert "is_active" in status
        assert "server_address" in status
        assert "active_connections" in status

    def test_network_metrics(self, coordinator):
        """Test network metrics."""
        metrics = coordinator.get_network_metrics()

        assert "total_messages_sent" in metrics
        assert "total_messages_received" in metrics
        assert "failed_messages" in metrics
        assert "average_latency" in metrics


class TestScalingConfig:
    """Test scaling configuration."""

    def test_default_config(self):
        """Test default configuration."""
        config = ScalingConfig()

        assert config.enable_auto_scaling is True
        assert config.deployment_strategy == DeploymentStrategy.HYBRID
        assert config.min_nodes == 1
        assert config.max_nodes == 10
        assert config.scale_up_threshold == 0.8

    def test_config_validation(self):
        """Test configuration validation."""
        # Test invalid min nodes
        with pytest.raises(ValueError, match="Minimum nodes must be positive"):
            ScalingConfig(min_nodes=0)

        # Test invalid max nodes
        with pytest.raises(ValueError, match="Maximum nodes must be >= minimum nodes"):
            ScalingConfig(min_nodes=5, max_nodes=3)

        # Test invalid thresholds
        with pytest.raises(
            ValueError, match="Scale up threshold must be between 0 and 1"
        ):
            ScalingConfig(scale_up_threshold=1.5)


class TestScalabilityManager:
    """Test scalability manager."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return ScalingConfig(
            min_nodes=1, max_nodes=5, scale_up_threshold=0.8, scale_down_threshold=0.3
        )

    @pytest.fixture
    def manager(self, config):
        """Create test scalability manager."""
        return ScalabilityManager(config)

    def test_initialization(self, manager):
        """Test manager initialization."""
        assert not manager.is_active
        assert manager.current_nodes == 1
        assert len(manager.active_nodes) == 0
        assert len(manager.scaling_actions) == 0

    @pytest.mark.asyncio
    async def test_manual_scaling(self, manager):
        """Test manual scaling."""
        success = await manager.manual_scale(3, "Test scaling")

        assert success
        # Note: In real implementation, this would actually scale

    def test_scaling_status(self, manager):
        """Test scaling status."""
        status = manager.get_scaling_status()

        assert "is_active" in status
        assert "current_nodes" in status
        assert "target_range" in status
        assert "deployment_strategy" in status

    def test_scaling_metrics(self, manager):
        """Test scaling metrics."""
        metrics = manager.get_scaling_metrics()

        assert "total_scaling_actions" in metrics
        assert "successful_scaling_actions" in metrics
        assert "failed_scaling_actions" in metrics
        assert "current_efficiency" in metrics

    def test_node_templates(self, manager):
        """Test node templates."""
        assert "default" in manager.node_templates
        assert "high_performance" in manager.node_templates
        assert "lightweight" in manager.node_templates

        default_template = manager.node_templates["default"]
        assert "cpu_cores" in default_template
        assert "memory_gb" in default_template
        assert "deployment_config" in default_template


class TestDistributedIntegration:
    """Test integration between distributed components."""

    @pytest.fixture
    def resource_manager(self):
        """Create resource manager."""
        config = ResourceConfig()
        return ResourceManager(config)

    @pytest.fixture
    def network_coordinator(self):
        """Create network coordinator."""
        config = NetworkConfig()
        return NetworkCoordinator(config, "test_node")

    @pytest.fixture
    def scalability_manager(self, resource_manager, network_coordinator):
        """Create scalability manager with dependencies."""
        config = ScalingConfig()
        return ScalabilityManager(config, resource_manager, network_coordinator)

    def test_component_integration(self, scalability_manager):
        """Test component integration."""
        assert scalability_manager.resource_manager is not None
        assert scalability_manager.network_coordinator is not None

    @pytest.mark.asyncio
    async def test_distributed_coordination_flow(self):
        """Test distributed coordination flow."""
        # Create distributed coordinator
        coord_config = DistributedCoordinationConfig()
        coordinator = DistributedCoordinator(coord_config)

        # Test coordination request handling
        request = {
            "coordination_id": "test_coord",
            "action_request": {"action": "test"},
            "timestamp": time.time(),
        }

        response = await coordinator.handle_coordination_request(request, "sender_node")

        assert "coordination_id" in response
        assert "node_id" in response
        assert "response" in response


if __name__ == "__main__":
    pytest.main([__file__])
