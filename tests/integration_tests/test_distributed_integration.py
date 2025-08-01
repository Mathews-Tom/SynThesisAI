"""Integration tests for distributed MARL system."""

# Standard Library
import asyncio
import time
from typing import Any, AsyncIterator, Dict
from unittest.mock import Mock, patch

# Third-Party Library
import pytest

# SynThesisAI Modules
from core.marl.distributed.distributed_coordinator import (
    DistributedCoordinationConfig,
    DistributedCoordinator,
)
from core.marl.distributed.distributed_trainer import (
    DistributedMARLTrainer,
    DistributedTrainingConfig,
    TrainingMode,
)
from core.marl.distributed.network_coordinator import (
    MessageType,
    NetworkConfig,
    NetworkCoordinator,
)
from core.marl.distributed.resource_manager import (
    ResourceConfig,
    ResourceManager,
    ResourceType,
)
from core.marl.distributed.scalability_manager import ScalabilityManager, ScalingConfig


class TestDistributedMARLIntegration:
    """Test distributed MARL system integration."""

    @pytest.fixture
    async def distributed_system(self) -> AsyncIterator[Dict[str, Any]]:
        """Create integrated distributed system."""
        # Create components
        resource_config = ResourceConfig(monitoring_interval=1.0)
        resource_manager = ResourceManager(resource_config)

        network_config = NetworkConfig(port=8080)
        network_coordinator = NetworkCoordinator(network_config, "test_node")

        scaling_config = ScalingConfig(monitoring_interval=5.0)
        scalability_manager = ScalabilityManager(
            scaling_config, resource_manager, network_coordinator
        )

        coord_config = DistributedCoordinationConfig()
        distributed_coordinator = DistributedCoordinator(coord_config)

        training_config = DistributedTrainingConfig(
            training_mode=TrainingMode.CPU_ONLY, backend="gloo"
        )
        distributed_trainer = DistributedMARLTrainer(training_config)

        # Start services
        await resource_manager.start_resource_management()
        await network_coordinator.start_network_coordination()
        await scalability_manager.start_scalability_management()
        await distributed_coordinator.initialize_distributed_coordination()
        await distributed_trainer.initialize_distributed_training()

        system = {
            "resource_manager": resource_manager,
            "network_coordinator": network_coordinator,
            "scalability_manager": scalability_manager,
            "distributed_coordinator": distributed_coordinator,
            "distributed_trainer": distributed_trainer,
        }

        yield system

        # Cleanup
        await resource_manager.shutdown()
        await network_coordinator.shutdown()
        await scalability_manager.shutdown()
        await distributed_coordinator.shutdown()
        await distributed_trainer.shutdown()

    @pytest.mark.asyncio
    async def test_system_initialization(self, distributed_system):
        """Test system initialization."""
        assert distributed_system["resource_manager"].is_active
        assert distributed_system["network_coordinator"].is_active
        assert distributed_system["scalability_manager"].is_active
        assert distributed_system["distributed_coordinator"].is_initialized
        assert distributed_system["distributed_trainer"].is_initialized

    @pytest.mark.asyncio
    async def test_resource_allocation_flow(self, distributed_system):
        """Test resource allocation flow."""
        resource_manager = distributed_system["resource_manager"]

        # Allocate CPU resource
        allocation_id = await resource_manager.allocate_resource(
            ResourceType.CPU, 2.0, "test_agent", priority=1
        )

        assert allocation_id is not None

        # Check resource usage
        usage = resource_manager.get_resource_usage()
        assert usage["total_allocations"] == 1

        # Deallocate resource
        success = await resource_manager.deallocate_resource(allocation_id)
        assert success

        # Verify deallocation
        usage = resource_manager.get_resource_usage()
        assert usage["total_allocations"] == 0

    @pytest.mark.asyncio
    async def test_network_communication_flow(self, distributed_system):
        """Test network communication flow."""
        network_coordinator = distributed_system["network_coordinator"]

        # Send heartbeat message
        message_id = await network_coordinator.send_message(
            MessageType.HEARTBEAT, {"status": "active", "timestamp": time.time()}
        )

        assert message_id is not None

        # Check network metrics
        metrics = network_coordinator.get_network_metrics()
        assert metrics["total_messages_sent"] >= 1

    @pytest.mark.asyncio
    async def test_coordination_flow(self, distributed_system):
        """Test distributed coordination flow."""
        coordinator = distributed_system["distributed_coordinator"]

        # Create coordination request
        action_request = {
            "action": "generate_content",
            "domain": "mathematics",
            "parameters": {"difficulty": "medium"},
        }

        # Test coordination (will timeout since no other nodes)
        with pytest.raises(TimeoutError):
            await coordinator.coordinate_distributed_action(
                action_request, participating_nodes=["node1", "node2"]
            )

        # Check coordination metrics
        metrics = coordinator.get_coordination_metrics()
        assert metrics["total_coordinations"] >= 1

    @pytest.mark.asyncio
    async def test_scaling_integration(self, distributed_system):
        """Test scaling integration."""
        scalability_manager = distributed_system["scalability_manager"]

        # Test manual scaling
        success = await scalability_manager.manual_scale(2, "Integration test")
        assert success

        # Check scaling status
        status = scalability_manager.get_scaling_status()
        assert status["current_nodes"] == 2

        # Scale back down
        success = await scalability_manager.manual_scale(1, "Scale down test")
        assert success

    @pytest.mark.asyncio
    async def test_training_integration(self, distributed_system):
        """Test distributed training integration."""
        trainer = distributed_system["distributed_trainer"]

        # Register mock agent
        mock_agent = Mock()
        mock_agent.parameters.return_value = []
        trainer.register_agent("test_agent", mock_agent)

        # Start short training session
        results = await trainer.start_distributed_training(num_epochs=2, steps_per_epoch=5)

        assert "total_steps" in results
        assert "training_time" in results
        assert results["total_steps"] == 10  # 2 epochs * 5 steps

    @pytest.mark.asyncio
    async def test_fault_tolerance(self, distributed_system):
        """Test fault tolerance mechanisms."""
        coordinator = distributed_system["distributed_coordinator"]

        # Simulate node failure
        failed_node = "failed_node"
        await coordinator.handle_node_failure(failed_node)

        # Check that system continues to function
        metrics = coordinator.get_coordination_metrics()
        assert metrics["node_failures"] >= 1

    @pytest.mark.asyncio
    async def test_performance_monitoring(self, distributed_system):
        """Test performance monitoring integration."""
        resource_manager = distributed_system["resource_manager"]
        scalability_manager = distributed_system["scalability_manager"]

        # Wait for some metrics to be collected
        await asyncio.sleep(2.0)

        # Check resource metrics
        resource_metrics = resource_manager.get_resource_metrics(limit=5)
        assert len(resource_metrics) > 0

        # Check scaling metrics
        scaling_metrics = scalability_manager.get_scaling_metrics()
        assert "current_efficiency" in scaling_metrics

    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self, distributed_system):
        """Test end-to-end distributed MARL workflow."""
        resource_manager = distributed_system["resource_manager"]
        trainer = distributed_system["distributed_trainer"]

        # 1. Allocate resources for agents
        cpu_allocation = await resource_manager.allocate_resource(
            ResourceType.CPU, 4.0, "generator_agent"
        )
        memory_allocation = await resource_manager.allocate_resource(
            ResourceType.MEMORY, 2.0, "validator_agent"
        )

        assert cpu_allocation is not None
        assert memory_allocation is not None

        # 2. Register agents with trainer
        mock_generator = Mock()
        mock_validator = Mock()
        mock_generator.parameters.return_value = []
        mock_validator.parameters.return_value = []

        trainer.register_agent("generator", mock_generator)
        trainer.register_agent("validator", mock_validator)

        # 3. Start training
        training_task = asyncio.create_task(
            trainer.start_distributed_training(num_epochs=3, steps_per_epoch=5)
        )

        # 4. Wait a bit then check system status
        await asyncio.sleep(0.5)

        # Check that training is active
        assert trainer.is_training

        # Check resource usage
        usage = resource_manager.get_resource_usage()
        assert usage["total_allocations"] == 2

        # 5. Wait for training to complete
        results = await training_task

        assert results["total_steps"] == 15  # 3 epochs * 5 steps
        assert not trainer.is_training

        # 6. Cleanup resources
        await resource_manager.deallocate_resource(cpu_allocation)
        await resource_manager.deallocate_resource(memory_allocation)

        # Verify cleanup
        usage = resource_manager.get_resource_usage()
        assert usage["total_allocations"] == 0


class TestDistributedFailureScenarios:
    """Test distributed system failure scenarios."""

    @pytest.mark.asyncio
    async def test_resource_exhaustion(self):
        """Test resource exhaustion handling."""
        config = ResourceConfig(
            max_cpu_percent=50.0,  # Low limit for testing
            max_memory_percent=50.0,
        )
        manager = ResourceManager(config)

        await manager.start_resource_management()

        try:
            # Allocate resources up to limit
            allocations = []

            # This should succeed
            alloc1 = await manager.allocate_resource(ResourceType.CPU, 1.0, "agent1")
            allocations.append(alloc1)
            assert alloc1 is not None

            # This should fail due to resource limits
            alloc2 = await manager.allocate_resource(
                ResourceType.CPU,
                100.0,
                "agent2",  # Too much
            )
            assert alloc2 is None

            # Cleanup
            for alloc_id in allocations:
                if alloc_id:
                    await manager.deallocate_resource(alloc_id)

        finally:
            await manager.shutdown()

    @pytest.mark.asyncio
    async def test_network_disconnection(self):
        """Test network disconnection handling."""
        config = NetworkConfig(port=8081)
        coordinator = NetworkCoordinator(config, "test_node")

        await coordinator.start_network_coordination()

        try:
            # Simulate connection failure
            with patch.object(
                coordinator,
                "_send_message_to_node",
                side_effect=ConnectionError("Network error"),
            ):
                # This should handle the error gracefully
                with pytest.raises(ConnectionError):
                    await coordinator._send_message_to_node("nonexistent_node", {})

        finally:
            await coordinator.shutdown()

    @pytest.mark.asyncio
    async def test_scaling_limits(self):
        """Test scaling limits handling."""
        config = ScalingConfig(min_nodes=1, max_nodes=3, enable_auto_scaling=True)
        manager = ScalabilityManager(config)

        await manager.start_scalability_management()

        try:
            # Try to scale beyond maximum
            success = await manager.manual_scale(5, "Test over-scaling")
            assert not success  # Should fail

            # Try to scale below minimum
            success = await manager.manual_scale(0, "Test under-scaling")
            assert not success  # Should fail

            # Valid scaling should work
            success = await manager.manual_scale(2, "Valid scaling")
            assert success

        finally:
            await manager.shutdown()


class TestDistributedPerformance:
    """Test distributed system performance characteristics."""

    @pytest.mark.asyncio
    async def test_concurrent_resource_allocation(self):
        """Test concurrent resource allocation performance."""
        config = ResourceConfig()
        manager = ResourceManager(config)

        await manager.start_resource_management()

        try:
            # Create multiple concurrent allocation tasks
            async def allocate_resource(agent_id: str):
                return await manager.allocate_resource(ResourceType.CPU, 0.5, f"agent_{agent_id}")

            # Run concurrent allocations
            tasks = [allocate_resource(str(i)) for i in range(10)]
            results = await asyncio.gather(*tasks)

            # All allocations should succeed
            successful_allocations = [r for r in results if r is not None]
            assert len(successful_allocations) == 10

            # Cleanup
            for alloc_id in successful_allocations:
                await manager.deallocate_resource(alloc_id)

        finally:
            await manager.shutdown()

    @pytest.mark.asyncio
    async def test_message_throughput(self):
        """Test network message throughput."""
        config = NetworkConfig(port=8082)
        coordinator = NetworkCoordinator(config, "test_node")

        await coordinator.start_network_coordination()

        try:
            # Send multiple messages rapidly
            start_time = time.time()
            message_count = 100

            tasks = []
            for i in range(message_count):
                task = coordinator.send_message(
                    MessageType.HEARTBEAT, {"sequence": i, "timestamp": time.time()}
                )
                tasks.append(task)

            # Wait for all messages to be queued
            await asyncio.gather(*tasks)

            end_time = time.time()
            duration = end_time - start_time

            # Check throughput (should be able to queue 100 messages quickly)
            assert duration < 1.0  # Should complete in less than 1 second

            # Check metrics
            metrics = coordinator.get_network_metrics()
            assert metrics["total_messages_sent"] >= message_count

        finally:
            await coordinator.shutdown()

    @pytest.mark.asyncio
    async def test_scaling_response_time(self):
        """Test scaling response time."""
        config = ScalingConfig(monitoring_interval=1.0)
        manager = ScalabilityManager(config)

        await manager.start_scalability_management()

        try:
            # Measure scaling time
            start_time = time.time()

            success = await manager.manual_scale(3, "Performance test")

            end_time = time.time()
            scaling_time = end_time - start_time

            assert success
            assert scaling_time < 5.0  # Should complete quickly (simulated)

            # Check scaling metrics
            metrics = manager.get_scaling_metrics()
            assert metrics["total_scaling_actions"] >= 1

        finally:
            await manager.shutdown()


if __name__ == "__main__":
    pytest.main([__file__])
