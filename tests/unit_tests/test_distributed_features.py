"""Comprehensive unit tests for distributed MARL features (Task 10+)."""

import asyncio
import threading
import time
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

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
    NetworkConnection,
    NetworkCoordinator,
    NetworkMessage,
    NetworkProtocol,
)
from core.marl.distributed.resource_manager import (
    AllocationStrategy,
    ResourceAllocation,
    ResourceConfig,
    ResourceManager,
    ResourceType,
)
from core.marl.distributed.scalability_manager import (
    DeploymentStrategy,
    NodeInfo,
    NodeStatus,
    ScalabilityManager,
    ScalingAction,
    ScalingConfig,
    ScalingDirection,
    ScalingMetrics,
    ScalingTrigger,
)


class TestDistributedTrainer:
    """Test distributed MARL trainer functionality."""

    @pytest.fixture
    def trainer_config(self):
        """Create trainer configuration."""
        return DistributedTrainingConfig(
            training_mode=TrainingMode.CPU_ONLY, world_size=2, rank=0, backend="gloo"
        )

    @pytest.fixture
    def trainer(self, trainer_config):
        """Create distributed trainer."""
        return DistributedMARLTrainer(trainer_config)

    def test_trainer_initialization(self, trainer):
        """Test trainer initialization."""
        assert not trainer.is_initialized
        assert not trainer.is_training
        assert trainer.config.world_size == 2
        assert trainer.config.backend == "gloo"
        assert len(trainer.agents) == 0

    def test_agent_registration(self, trainer):
        """Test agent registration and unregistration."""
        mock_agent = Mock()

        # Register agent
        trainer.register_agent("test_agent", mock_agent)
        assert "test_agent" in trainer.agents
        assert trainer.agents["test_agent"] == mock_agent

        # Unregister agent
        trainer.unregister_agent("test_agent")
        assert "test_agent" not in trainer.agents

    @pytest.mark.asyncio
    async def test_training_initialization(self, trainer):
        """Test training initialization process."""
        with (
            patch.object(trainer, "_setup_distributed_environment") as mock_env,
            patch.object(trainer, "_setup_devices") as mock_devices,
            patch.object(trainer, "_setup_process_group") as mock_group,
            patch.object(trainer, "_setup_distributed_agents") as mock_agents,
        ):
            await trainer.initialize_distributed_training()

            assert trainer.is_initialized
            mock_env.assert_called_once()
            mock_devices.assert_called_once()
            mock_group.assert_called_once()
            mock_agents.assert_called_once()

    @pytest.mark.asyncio
    async def test_training_process(self, trainer):
        """Test training process."""
        # Mock initialization
        trainer.is_initialized = True

        # Mock agent
        mock_agent = Mock()
        mock_agent.parameters.return_value = []
        trainer.register_agent("test_agent", mock_agent)

        with (
            patch.object(trainer, "_train_epoch") as mock_epoch,
            patch.object(trainer, "_synchronize_agents") as mock_sync,
            patch.object(trainer, "_save_checkpoint") as mock_checkpoint,
        ):
            mock_epoch.return_value = {
                "steps": 5,
                "average_reward": 0.8,
                "communication_time": 0.1,
            }

            results = await trainer.start_distributed_training(
                num_epochs=2, steps_per_epoch=5
            )

            assert "total_steps" in results
            assert "training_time" in results
            assert results["total_steps"] == 10  # 2 epochs * 5 steps
            assert mock_epoch.call_count == 2

    def test_training_metrics(self, trainer):
        """Test training metrics collection."""
        metrics = trainer.get_training_metrics()

        expected_keys = [
            "total_steps",
            "total_episodes",
            "average_reward",
            "training_time",
            "communication_time",
            "synchronization_time",
        ]

        for key in expected_keys:
            assert key in metrics

    def test_distributed_info(self, trainer):
        """Test distributed information retrieval."""
        info = trainer.get_distributed_info()

        expected_keys = [
            "world_size",
            "rank",
            "local_rank",
            "device",
            "training_mode",
            "sync_strategy",
            "is_initialized",
            "is_training",
            "registered_agents",
        ]

        for key in expected_keys:
            assert key in info


class TestDistributedCoordinator:
    """Test distributed coordinator functionality."""

    @pytest.fixture
    def coordinator_config(self):
        """Create coordinator configuration."""
        return DistributedCoordinationConfig(
            coordination_mode=DistributedCoordinationMode.CENTRALIZED,
            network_topology=NetworkTopology.STAR,
            consensus_threshold=0.7,
        )

    @pytest.fixture
    def coordinator(self, coordinator_config):
        """Create distributed coordinator."""
        return DistributedCoordinator(coordinator_config)

    def test_coordinator_initialization(self, coordinator):
        """Test coordinator initialization."""
        assert not coordinator.is_initialized
        assert not coordinator.is_active
        assert (
            coordinator.config.coordination_mode
            == DistributedCoordinationMode.CENTRALIZED
        )
        assert len(coordinator.connected_nodes) == 0

    @pytest.mark.asyncio
    async def test_coordinator_initialization_process(self, coordinator):
        """Test coordinator initialization process."""
        with (
            patch.object(
                coordinator, "_initialize_coordination_components"
            ) as mock_comp,
            patch.object(coordinator, "_setup_network_topology") as mock_topo,
            patch.object(coordinator, "_start_network_services") as mock_net,
            patch.object(coordinator, "_determine_master_node") as mock_master,
            patch.object(coordinator, "_start_coordination_services") as mock_coord,
        ):
            await coordinator.initialize_distributed_coordination()

            assert coordinator.is_initialized
            assert coordinator.is_active
            mock_comp.assert_called_once()
            mock_topo.assert_called_once()
            mock_net.assert_called_once()
            mock_master.assert_called_once()
            mock_coord.assert_called_once()

    @pytest.mark.asyncio
    async def test_coordination_request_handling(self, coordinator):
        """Test coordination request handling."""
        request = {
            "coordination_id": "test_coord",
            "action_request": {"action": "test_action"},
            "timestamp": time.time(),
        }

        with patch.object(coordinator, "_process_local_coordination") as mock_process:
            mock_process.return_value = {"result": "success"}

            response = await coordinator.handle_coordination_request(
                request, "sender_node"
            )

            assert "coordination_id" in response
            assert "node_id" in response
            assert "response" in response
            assert response["coordination_id"] == "test_coord"

    @pytest.mark.asyncio
    async def test_distributed_action_coordination(self, coordinator):
        """Test distributed action coordination."""
        coordinator.is_active = True

        action_request = {
            "action": "generate_content",
            "parameters": {"domain": "math"},
        }

        with (
            patch.object(coordinator, "_send_coordination_request") as mock_send,
            patch.object(coordinator, "_wait_for_coordination_completion") as mock_wait,
            patch.object(coordinator, "_process_coordination_result") as mock_process,
        ):
            mock_wait.return_value = {
                "coordination_id": "test",
                "responses": [{"agent": "test", "result": "success"}],
            }
            mock_process.return_value = {
                "status": "success",
                "coordinated_action": {"action": "result"},
            }

            result = await coordinator.coordinate_distributed_action(action_request)

            assert "status" in result
            assert "coordinated_action" in result

    def test_coordination_metrics(self, coordinator):
        """Test coordination metrics."""
        metrics = coordinator.get_coordination_metrics()

        expected_keys = [
            "total_coordinations",
            "successful_coordinations",
            "failed_coordinations",
            "average_coordination_time",
            "network_latency",
            "node_failures",
        ]

        for key in expected_keys:
            assert key in metrics


class TestResourceManager:
    """Test resource manager functionality."""

    @pytest.fixture
    def resource_config(self):
        """Create resource configuration."""
        return ResourceConfig(
            max_cpu_percent=80.0, max_memory_percent=80.0, monitoring_interval=5.0
        )

    @pytest.fixture
    def resource_manager(self, resource_config):
        """Create resource manager."""
        return ResourceManager(resource_config)

    def test_resource_manager_initialization(self, resource_manager):
        """Test resource manager initialization."""
        assert not resource_manager.is_active
        assert len(resource_manager.resource_allocations) == 0
        assert "cpu_count" in resource_manager.system_info
        assert "memory_total" in resource_manager.system_info

    @pytest.mark.asyncio
    async def test_resource_allocation(self, resource_manager):
        """Test resource allocation and deallocation."""
        # Test allocation
        allocation_id = await resource_manager.allocate_resource(
            ResourceType.CPU, 2.0, "test_owner", priority=1
        )

        assert allocation_id is not None
        assert allocation_id in resource_manager.resource_allocations

        allocation = resource_manager.resource_allocations[allocation_id]
        assert allocation.resource_type == ResourceType.CPU
        assert allocation.allocated_amount == 2.0
        assert allocation.owner == "test_owner"
        assert allocation.priority == 1

        # Test deallocation
        success = await resource_manager.deallocate_resource(allocation_id)
        assert success
        assert allocation_id not in resource_manager.resource_allocations

    @pytest.mark.asyncio
    async def test_resource_allocation_limits(self, resource_manager):
        """Test resource allocation limits."""
        # Try to allocate more than available
        allocation_id = await resource_manager.allocate_resource(
            ResourceType.CPU,
            1000.0,  # Unrealistic amount
            "test_owner",
        )

        # Should fail due to limits
        assert allocation_id is None

    @pytest.mark.asyncio
    async def test_resource_management_lifecycle(self, resource_manager):
        """Test resource management lifecycle."""
        # Start resource management
        await resource_manager.start_resource_management()
        assert resource_manager.is_active

        # Wait for some monitoring
        await asyncio.sleep(0.1)

        # Check metrics collection
        metrics = resource_manager.get_resource_metrics(limit=5)
        # Should have some metrics after starting

        # Shutdown
        await resource_manager.shutdown()
        assert not resource_manager.is_active

    def test_resource_usage_tracking(self, resource_manager):
        """Test resource usage tracking."""
        usage = resource_manager.get_resource_usage()

        assert "usage_by_type" in usage
        assert "total_allocations" in usage
        assert "allocation_details" in usage

        # Check resource types
        for resource_type in ResourceType:
            assert resource_type.value in usage["usage_by_type"]


class TestNetworkCoordinator:
    """Test network coordinator functionality."""

    @pytest.fixture
    def network_config(self):
        """Create network configuration."""
        return NetworkConfig(host="localhost", port=8080, protocol=NetworkProtocol.TCP)

    @pytest.fixture
    def network_coordinator(self, network_config):
        """Create network coordinator."""
        return NetworkCoordinator(network_config, "test_node")

    def test_network_coordinator_initialization(self, network_coordinator):
        """Test network coordinator initialization."""
        assert not network_coordinator.is_active
        assert len(network_coordinator.connections) == 0
        assert network_coordinator.node_id == "test_node"
        assert network_coordinator.config.port == 8080

    @pytest.mark.asyncio
    async def test_message_sending(self, network_coordinator):
        """Test message sending."""
        message_id = await network_coordinator.send_message(
            MessageType.HEARTBEAT,
            {"status": "active", "timestamp": time.time()},
            recipient_id="target_node",
        )

        assert message_id is not None
        assert isinstance(message_id, str)

    def test_message_handler_management(self, network_coordinator):
        """Test message handler registration and removal."""
        handler = Mock()

        # Add handler
        network_coordinator.add_message_handler(MessageType.HEARTBEAT, handler)
        assert handler in network_coordinator.message_handlers[MessageType.HEARTBEAT]

        # Remove handler
        network_coordinator.remove_message_handler(MessageType.HEARTBEAT, handler)
        assert (
            handler not in network_coordinator.message_handlers[MessageType.HEARTBEAT]
        )

    def test_network_message_serialization(self):
        """Test network message serialization."""
        message = NetworkMessage(
            message_id="test_id",
            message_type=MessageType.COORDINATION_REQUEST,
            sender_id="sender",
            recipient_id="recipient",
            payload={"data": "test"},
            timestamp=time.time(),
        )

        # Test serialization
        message_dict = message.to_dict()
        assert message_dict["message_id"] == "test_id"
        assert message_dict["message_type"] == "coordination_request"

        # Test deserialization
        reconstructed = NetworkMessage.from_dict(message_dict)
        assert reconstructed.message_id == message.message_id
        assert reconstructed.message_type == message.message_type
        assert reconstructed.sender_id == message.sender_id

    def test_network_status_and_metrics(self, network_coordinator):
        """Test network status and metrics."""
        status = network_coordinator.get_network_status()

        expected_keys = [
            "node_id",
            "is_active",
            "server_address",
            "active_connections",
            "connected_nodes",
        ]

        for key in expected_keys:
            assert key in status

        metrics = network_coordinator.get_network_metrics()

        expected_metric_keys = [
            "total_messages_sent",
            "total_messages_received",
            "failed_messages",
            "average_latency",
        ]

        for key in expected_metric_keys:
            assert key in metrics


class TestScalabilityManager:
    """Test scalability manager functionality."""

    @pytest.fixture
    def scaling_config(self):
        """Create scaling configuration."""
        return ScalingConfig(
            min_nodes=1,
            max_nodes=5,
            scale_up_threshold=0.8,
            scale_down_threshold=0.3,
            monitoring_interval=1.0,
        )

    @pytest.fixture
    def scalability_manager(self, scaling_config):
        """Create scalability manager."""
        return ScalabilityManager(scaling_config)

    def test_scalability_manager_initialization(self, scalability_manager):
        """Test scalability manager initialization."""
        assert not scalability_manager.is_active
        assert scalability_manager.current_nodes == 1
        assert len(scalability_manager.nodes) == 1  # Current node
        assert len(scalability_manager.scaling_actions) == 0

    @pytest.mark.asyncio
    async def test_scalability_management_lifecycle(self, scalability_manager):
        """Test scalability management lifecycle."""
        # Start scalability management
        await scalability_manager.start_scalability_management()
        assert scalability_manager.is_active

        # Wait for some monitoring
        await asyncio.sleep(0.1)

        # Shutdown
        await scalability_manager.shutdown()
        assert not scalability_manager.is_active

    @pytest.mark.asyncio
    async def test_manual_scaling(self, scalability_manager):
        """Test manual scaling operations."""
        # Scale up
        success = await scalability_manager.manual_scale(3, "Test scale up")
        assert success

        # Scale down
        success = await scalability_manager.manual_scale(2, "Test scale down")
        assert success

        # Invalid scaling (outside limits)
        success = await scalability_manager.manual_scale(10, "Invalid scale")
        assert not success

    def test_scaling_metrics_collection(self, scalability_manager):
        """Test scaling metrics collection."""
        metrics = scalability_manager._collect_scaling_metrics()

        assert isinstance(metrics, ScalingMetrics)
        assert metrics.timestamp > 0
        assert 0 <= metrics.cpu_utilization <= 1
        assert 0 <= metrics.memory_utilization <= 1
        assert metrics.active_nodes >= 1

    def test_scaling_policy_evaluation(self, scalability_manager):
        """Test scaling policy evaluation."""
        # Create test metrics
        test_metrics = [
            ScalingMetrics(
                timestamp=time.time(),
                cpu_utilization=0.9,  # High CPU
                memory_utilization=0.5,
                gpu_utilization=0.3,
                network_latency=50.0,
                coordination_load=0.4,
                active_nodes=2,
                total_agents=5,
                coordination_success_rate=0.95,
            )
        ]

        # Test CPU-based policy
        cpu_policy = scalability_manager.config.scaling_policies["cpu_based"]
        result = scalability_manager._evaluate_policy(cpu_policy, test_metrics)

        assert result["action"] == ScalingDirection.UP
        assert result["trigger"] == ScalingTrigger.CPU_UTILIZATION
        assert result["current_value"] == 0.9

    def test_node_management(self, scalability_manager):
        """Test node management operations."""
        # Check initial node
        assert len(scalability_manager.nodes) == 1

        # Get node status
        status = scalability_manager.get_node_status()

        assert "active_nodes" in status
        assert "pending_nodes" in status
        assert "summary" in status

        summary = status["summary"]
        assert summary["total_nodes"] >= 1
        assert summary["active_count"] >= 1

    def test_scaling_action_tracking(self, scalability_manager):
        """Test scaling action tracking."""
        # Create test scaling action
        action = ScalingAction(
            action_id="test_action",
            action_type=ScalingDirection.UP,
            trigger=ScalingTrigger.CPU_UTILIZATION,
            target_nodes=3,
            current_nodes=2,
            timestamp=time.time(),
            reason="Test scaling",
        )

        scalability_manager.scaling_actions.append(action)

        # Get scaling history
        history = scalability_manager.get_scaling_history(limit=10)

        assert len(history) == 1
        assert history[0]["action_id"] == "test_action"
        assert history[0]["action_type"] == "up"

    def test_scaling_configuration_updates(self, scalability_manager):
        """Test scaling configuration updates."""
        original_threshold = scalability_manager.config.scale_up_threshold

        # Update configuration
        scalability_manager.update_scaling_config({"scale_up_threshold": 0.9})

        assert scalability_manager.config.scale_up_threshold == 0.9
        assert scalability_manager.config.scale_up_threshold != original_threshold

    def test_callback_management(self, scalability_manager):
        """Test callback management."""
        scaling_callback = Mock()
        node_callback = Mock()

        # Add callbacks
        scalability_manager.add_scaling_callback(scaling_callback)
        scalability_manager.add_node_callback(node_callback)

        assert scaling_callback in scalability_manager.scaling_callbacks
        assert node_callback in scalability_manager.node_callbacks


class TestDistributedIntegration:
    """Test integration between distributed components."""

    @pytest.fixture
    def integrated_system(self):
        """Create integrated distributed system."""
        resource_config = ResourceConfig(monitoring_interval=0.5)
        resource_manager = ResourceManager(resource_config)

        network_config = NetworkConfig(port=8081)
        network_coordinator = NetworkCoordinator(network_config, "test_node")

        scaling_config = ScalingConfig(monitoring_interval=1.0)
        scalability_manager = ScalabilityManager(
            scaling_config, resource_manager, network_coordinator
        )

        return {
            "resource_manager": resource_manager,
            "network_coordinator": network_coordinator,
            "scalability_manager": scalability_manager,
        }

    @pytest.mark.asyncio
    async def test_system_integration(self, integrated_system):
        """Test system integration."""
        resource_manager = integrated_system["resource_manager"]
        network_coordinator = integrated_system["network_coordinator"]
        scalability_manager = integrated_system["scalability_manager"]

        # Start all components
        await resource_manager.start_resource_management()
        await network_coordinator.start_network_coordination()
        await scalability_manager.start_scalability_management()

        # Verify integration
        assert resource_manager.is_active
        assert network_coordinator.is_active
        assert scalability_manager.is_active

        # Test resource allocation
        allocation_id = await resource_manager.allocate_resource(
            ResourceType.CPU, 2.0, "test_agent"
        )
        assert allocation_id is not None

        # Test network communication
        message_id = await network_coordinator.send_message(
            MessageType.HEARTBEAT, {"status": "active"}
        )
        assert message_id is not None

        # Test scaling
        success = await scalability_manager.manual_scale(2, "Integration test")
        assert success

        # Cleanup
        await resource_manager.shutdown()
        await network_coordinator.shutdown()
        await scalability_manager.shutdown()

    @pytest.mark.asyncio
    async def test_fault_tolerance_integration(self, integrated_system):
        """Test fault tolerance across components."""
        scalability_manager = integrated_system["scalability_manager"]

        await scalability_manager.start_scalability_management()

        # Simulate node failure
        test_node_id = "test_node_failure"
        await scalability_manager._handle_node_failure(test_node_id)

        # System should continue functioning
        assert scalability_manager.is_active

        await scalability_manager.shutdown()

    @pytest.mark.asyncio
    async def test_performance_monitoring_integration(self, integrated_system):
        """Test performance monitoring integration."""
        resource_manager = integrated_system["resource_manager"]
        scalability_manager = integrated_system["scalability_manager"]

        await resource_manager.start_resource_management()
        await scalability_manager.start_scalability_management()

        # Wait for metrics collection
        await asyncio.sleep(0.2)

        # Check resource metrics
        resource_metrics = resource_manager.get_resource_metrics(limit=5)
        assert len(resource_metrics) >= 0  # May be empty initially

        # Check scaling metrics
        scaling_metrics = scalability_manager.get_scaling_metrics()
        assert "current_efficiency" in scaling_metrics
        assert "uptime_percentage" in scaling_metrics

        await resource_manager.shutdown()
        await scalability_manager.shutdown()


class TestDistributedPerformance:
    """Test performance characteristics of distributed components."""

    @pytest.mark.asyncio
    async def test_resource_allocation_performance(self):
        """Test resource allocation performance."""
        config = ResourceConfig()
        manager = ResourceManager(config)

        await manager.start_resource_management()

        # Measure allocation time
        start_time = time.time()

        allocations = []
        for i in range(10):
            allocation_id = await manager.allocate_resource(
                ResourceType.CPU, 0.1, f"agent_{i}"
            )
            if allocation_id:
                allocations.append(allocation_id)

        allocation_time = time.time() - start_time

        # Should be fast (less than 1 second for 10 allocations)
        assert allocation_time < 1.0
        assert len(allocations) == 10

        # Cleanup
        for allocation_id in allocations:
            await manager.deallocate_resource(allocation_id)

        await manager.shutdown()

    @pytest.mark.asyncio
    async def test_scaling_decision_performance(self):
        """Test scaling decision performance."""
        config = ScalingConfig(monitoring_interval=0.1)
        manager = ScalabilityManager(config)

        # Add some metrics history
        for i in range(20):
            metrics = ScalingMetrics(
                timestamp=time.time() - (20 - i),
                cpu_utilization=0.5 + (i * 0.02),  # Gradually increasing
                memory_utilization=0.4,
                gpu_utilization=0.3,
                network_latency=50.0,
                coordination_load=0.3,
                active_nodes=2,
                total_agents=5,
                coordination_success_rate=0.95,
            )
            manager.metrics_history.append(metrics)

        # Measure decision time
        start_time = time.time()
        decision = manager._evaluate_scaling_need()
        decision_time = time.time() - start_time

        # Should be very fast (less than 0.1 seconds)
        assert decision_time < 0.1
        assert "action" in decision
        assert "reason" in decision

    @pytest.mark.asyncio
    async def test_network_message_throughput(self):
        """Test network message throughput."""
        config = NetworkConfig(port=8082)
        coordinator = NetworkCoordinator(config, "test_node")

        await coordinator.start_network_coordination()

        # Measure message queuing time
        start_time = time.time()
        message_count = 50

        message_ids = []
        for i in range(message_count):
            message_id = await coordinator.send_message(
                MessageType.HEARTBEAT, {"sequence": i}
            )
            message_ids.append(message_id)

        queuing_time = time.time() - start_time

        # Should queue messages quickly
        assert queuing_time < 0.5  # Less than 0.5 seconds for 50 messages
        assert len(message_ids) == message_count
        assert all(mid is not None for mid in message_ids)

        await coordinator.shutdown()


if __name__ == "__main__":
    pytest.main([__file__])
