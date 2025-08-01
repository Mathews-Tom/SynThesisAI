"""Integration tests for distributed MARL system (Task 10+)."""

# Standard Library
import asyncio
import logging
import time
from typing import Any, AsyncGenerator, Dict
from unittest.mock import Mock

# Third-Party
import pytest

# SynThesisAI Modules
from core.marl.distributed.distributed_coordinator import (
    DistributedCoordinationConfig,
    DistributedCoordinationMode,
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

logger = logging.getLogger(__name__)


class TestDistributedMARLSystemIntegration:
    """Test complete distributed MARL system integration."""

    @pytest.fixture
    async def distributed_system(self) -> AsyncGenerator[Dict[str, Any], None]:
        """Create complete distributed MARL system."""
        # Create components
        resource_config = ResourceConfig(
            monitoring_interval=0.5, max_cpu_percent=70.0, max_memory_percent=70.0
        )
        resource_manager = ResourceManager(resource_config)

        network_config = NetworkConfig(port=8083, heartbeat_interval=1.0)
        network_coordinator = NetworkCoordinator(network_config, "integration_test_node")

        scaling_config = ScalingConfig(
            min_nodes=1, max_nodes=4, monitoring_interval=1.0, scaling_cooldown=5.0
        )
        scalability_manager = ScalabilityManager(
            scaling_config, resource_manager, network_coordinator
        )

        coord_config = DistributedCoordinationConfig(
            coordination_mode=DistributedCoordinationMode.CENTRALIZED,
            coordination_timeout=10.0,
        )
        distributed_coordinator = DistributedCoordinator(coord_config)

        training_config = DistributedTrainingConfig(
            training_mode=TrainingMode.CPU_ONLY, world_size=1, backend="gloo"
        )
        distributed_trainer = DistributedMARLTrainer(training_config)

        # Start all services
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
        await distributed_trainer.shutdown()
        await distributed_coordinator.shutdown()
        await scalability_manager.shutdown()
        await network_coordinator.shutdown()
        await resource_manager.shutdown()

    @pytest.mark.asyncio
    async def test_complete_system_initialization(self, distributed_system) -> None:
        """Test complete system initialization."""
        assert distributed_system["resource_manager"].is_active
        assert distributed_system["network_coordinator"].is_active
        assert distributed_system["scalability_manager"].is_active
        assert distributed_system["distributed_coordinator"].is_initialized
        assert distributed_system["distributed_trainer"].is_initialized

    @pytest.mark.asyncio
    async def test_resource_allocation_workflow(self, distributed_system) -> None:
        """Test complete resource allocation workflow."""
        resource_manager = distributed_system["resource_manager"]
        scalability_manager = distributed_system["scalability_manager"]

        # Allocate resources for multiple agents
        allocations = []

        # Generator agent
        cpu_alloc = await resource_manager.allocate_resource(
            ResourceType.CPU, 2.0, "generator_agent", priority=1
        )
        allocations.append(cpu_alloc)

        # Validator agent
        memory_alloc = await resource_manager.allocate_resource(
            ResourceType.MEMORY, 4.0, "validator_agent", priority=2
        )
        allocations.append(memory_alloc)

        # Curriculum agent
        gpu_alloc = await resource_manager.allocate_resource(
            ResourceType.GPU, 1.0, "curriculum_agent", priority=1
        )
        allocations.append(gpu_alloc)

        # Verify allocations
        assert all(alloc is not None for alloc in allocations)

        # Check resource usage
        usage = resource_manager.get_resource_usage()
        assert usage["total_allocations"] == 3

        # Check if scaling manager detects resource pressure
        await asyncio.sleep(1.5)  # Wait for monitoring

        scaling_status = scalability_manager.get_scaling_status()
        assert scaling_status["is_active"]

        # Cleanup allocations
        for alloc_id in allocations:
            if alloc_id:
                await resource_manager.deallocate_resource(alloc_id)

        # Verify cleanup
        usage = resource_manager.get_resource_usage()
        assert usage["total_allocations"] == 0

    @pytest.mark.asyncio
    async def test_distributed_coordination_workflow(self, distributed_system) -> None:
        """Test distributed coordination workflow."""
        coordinator = distributed_system["distributed_coordinator"]
        network_coordinator = distributed_system["network_coordinator"]

        # Test coordination request
        action_request = {
            "action": "coordinate_generation",
            "domain": "mathematics",
            "difficulty": "medium",
            "agents": ["generator", "validator", "curriculum"],
        }

        # Since we don't have actual other nodes, this will timeout
        # but we can test the coordination setup
        try:
            await asyncio.wait_for(
                coordinator.coordinate_distributed_action(
                    action_request, participating_nodes=["node1"]
                ),
                timeout=2.0,
            )
        except asyncio.TimeoutError:
            # Expected since no actual nodes are connected
            pass

        # Check coordination metrics
        metrics = coordinator.get_coordination_metrics()
        assert metrics["total_coordinations"] >= 1

        # Check network activity
        network_metrics = network_coordinator.get_network_metrics()
        assert "total_messages_sent" in network_metrics

    @pytest.mark.asyncio
    async def test_scaling_workflow(self, distributed_system) -> None:
        """Test scaling workflow."""
        scalability_manager = distributed_system["scalability_manager"]

        # Get initial state
        initial_status = scalability_manager.get_scaling_status()
        initial_nodes = initial_status["current_nodes"]

        # Trigger manual scaling up
        success = await scalability_manager.manual_scale(
            initial_nodes + 1, "Integration test scale up"
        )
        assert success

        # Verify scaling
        new_status = scalability_manager.get_scaling_status()
        assert new_status["current_nodes"] == initial_nodes + 1

        # Check scaling history
        history = scalability_manager.get_scaling_history(limit=5)
        assert len(history) >= 1
        assert history[-1]["action_type"] == "up"

        # Scale back down
        success = await scalability_manager.manual_scale(
            initial_nodes, "Integration test scale down"
        )
        assert success

        # Verify scale down
        final_status = scalability_manager.get_scaling_status()
        assert final_status["current_nodes"] == initial_nodes

    @pytest.mark.asyncio
    async def test_training_integration(self, distributed_system) -> None:
        """Test distributed training integration."""
        trainer = distributed_system["distributed_trainer"]
        resource_manager = distributed_system["resource_manager"]

        # Register mock agents
        mock_generator = Mock()
        mock_validator = Mock()
        mock_generator.parameters.return_value = []
        mock_validator.parameters.return_value = []

        trainer.register_agent("generator", mock_generator)
        trainer.register_agent("validator", mock_validator)

        # Allocate resources for training
        cpu_alloc = await resource_manager.allocate_resource(
            ResourceType.CPU, 4.0, "training_process"
        )
        memory_alloc = await resource_manager.allocate_resource(
            ResourceType.MEMORY, 8.0, "training_process"
        )

        assert cpu_alloc is not None
        assert memory_alloc is not None

        # Start short training session
        training_task = asyncio.create_task(
            trainer.start_distributed_training(num_epochs=2, steps_per_epoch=3)
        )

        # Wait a bit then check training status
        await asyncio.sleep(0.1)
        assert trainer.is_training

        # Wait for training completion
        results = await training_task

        # Verify training results
        assert "total_steps" in results
        assert "training_time" in results
        assert results["total_steps"] == 6  # 2 epochs * 3 steps
        assert not trainer.is_training

        # Cleanup resources
        await resource_manager.deallocate_resource(cpu_alloc)
        await resource_manager.deallocate_resource(memory_alloc)

    @pytest.mark.asyncio
    async def test_fault_tolerance_workflow(self, distributed_system) -> None:
        """Test fault tolerance across the system."""
        scalability_manager = distributed_system["scalability_manager"]
        coordinator = distributed_system["distributed_coordinator"]

        # Simulate node failure
        failed_node_id = "failed_test_node"

        # Add the node to coordinator's connected nodes (simulate)
        coordinator.connected_nodes[failed_node_id] = {
            "node_id": failed_node_id,
            "connected_at": time.time(),
            "status": "connected",
        }

        # Handle node failure
        await coordinator.handle_node_failure(failed_node_id)

        # Verify failure handling
        assert failed_node_id not in coordinator.connected_nodes

        # Check metrics
        coord_metrics = coordinator.get_coordination_metrics()
        assert coord_metrics["node_failures"] >= 1

        # Test scalability manager fault tolerance
        await scalability_manager._handle_node_failure("test_node_failure")

        # System should continue functioning
        assert scalability_manager.is_active

    @pytest.mark.asyncio
    async def test_performance_monitoring_integration(self, distributed_system) -> None:
        """Test performance monitoring across all components."""
        resource_manager = distributed_system["resource_manager"]
        network_coordinator = distributed_system["network_coordinator"]
        scalability_manager = distributed_system["scalability_manager"]

        # Wait for metrics collection
        await asyncio.sleep(2.0)

        # Check resource metrics
        management_metrics = resource_manager.get_management_metrics()

        assert "total_allocations" in management_metrics
        assert "average_utilization" in management_metrics

        # Check network metrics
        network_metrics = network_coordinator.get_network_metrics()
        network_status = network_coordinator.get_network_status()

        assert "total_messages_sent" in network_metrics
        assert "node_id" in network_status

        # Check scaling metrics
        scaling_metrics = scalability_manager.get_scaling_metrics()
        scaling_status = scalability_manager.get_scaling_status()

        assert "current_efficiency" in scaling_metrics
        assert "current_nodes" in scaling_status

        # Verify metrics are being updated
        initial_time = time.time()
        await asyncio.sleep(1.0)

        # Get updated metrics
        new_resource_metrics = resource_manager.get_resource_metrics(limit=1)
        if new_resource_metrics:
            assert new_resource_metrics[0]["timestamp"] > initial_time

    @pytest.mark.asyncio
    async def test_end_to_end_marl_workflow(self, distributed_system) -> None:
        """Test complete end-to-end MARL workflow."""
        resource_manager = distributed_system["resource_manager"]
        coordinator = distributed_system["distributed_coordinator"]
        trainer = distributed_system["distributed_trainer"]
        scalability_manager = distributed_system["scalability_manager"]

        # Step 1: Allocate resources for MARL agents
        allocations = {}

        allocations["generator_cpu"] = await resource_manager.allocate_resource(
            ResourceType.CPU, 2.0, "generator_agent"
        )
        allocations["validator_memory"] = await resource_manager.allocate_resource(
            ResourceType.MEMORY, 3.0, "validator_agent"
        )
        allocations["curriculum_gpu"] = await resource_manager.allocate_resource(
            ResourceType.GPU, 1.0, "curriculum_agent"
        )

        # Verify allocations
        assert all(alloc is not None for alloc in allocations.values())

        # Step 2: Register agents with trainer
        mock_agents = {}
        for agent_type in ["generator", "validator", "curriculum"]:
            mock_agent = Mock()
            mock_agent.parameters.return_value = []
            mock_agents[agent_type] = mock_agent
            trainer.register_agent(agent_type, mock_agent)

        # Step 3: Scale system if needed
        initial_nodes = scalability_manager.get_scaling_status()["current_nodes"]
        if initial_nodes < 2:
            await scalability_manager.manual_scale(2, "MARL workflow scaling")

        # Step 4: Start distributed training
        training_task = asyncio.create_task(
            trainer.start_distributed_training(num_epochs=3, steps_per_epoch=4)
        )

        # Step 5: Monitor system during training
        await asyncio.sleep(0.2)  # Let training start

        # Check system status
        assert trainer.is_training
        assert resource_manager.is_active
        assert scalability_manager.is_active

        # Step 6: Complete training
        training_results = await training_task

        # Verify training completion
        assert training_results["total_steps"] == 12  # 3 epochs * 4 steps
        assert not trainer.is_training

        # Step 7: Cleanup resources
        for alloc_id in allocations.values():
            if alloc_id:
                await resource_manager.deallocate_resource(alloc_id)

        # Step 8: Verify system state
        usage = resource_manager.get_resource_usage()
        assert usage["total_allocations"] == 0

        scaling_metrics = scalability_manager.get_scaling_metrics()
        assert scaling_metrics["total_scaling_actions"] >= 0

        # Step 9: Generate system report
        system_report = {
            "training_results": training_results,
            "resource_metrics": resource_manager.get_management_metrics(),
            "scaling_metrics": scalability_manager.get_scaling_metrics(),
            "coordination_metrics": coordinator.get_coordination_metrics(),
            "network_status": distributed_system["network_coordinator"].get_network_status(),
        }

        # Verify report completeness
        assert all(
            key in system_report
            for key in [
                "training_results",
                "resource_metrics",
                "scaling_metrics",
                "coordination_metrics",
                "network_status",
            ]
        )


class TestDistributedMARLFailureScenarios:
    """Test distributed MARL system under failure conditions."""

    @pytest.mark.asyncio
    async def test_resource_exhaustion_handling(self) -> None:
        """Test system behavior under resource exhaustion."""
        config = ResourceConfig(
            max_cpu_percent=30.0,  # Very low limit
            max_memory_percent=30.0,
        )
        resource_manager = ResourceManager(config)

        scaling_config = ScalingConfig(min_nodes=1, max_nodes=3)
        scalability_manager = ScalabilityManager(scaling_config, resource_manager)

        await resource_manager.start_resource_management()
        await scalability_manager.start_scalability_management()

        try:
            # Try to allocate resources beyond limits
            allocations = []

            # This should succeed
            alloc1 = await resource_manager.allocate_resource(ResourceType.CPU, 0.5, "agent1")
            allocations.append(alloc1)
            assert alloc1 is not None

            # This should trigger scaling or fail gracefully
            alloc2 = await resource_manager.allocate_resource(
                ResourceType.CPU,
                10.0,
                "agent2",  # Excessive request
            )

            # Should either succeed (if scaled) or fail gracefully
            if alloc2 is None:
                # Failed gracefully - system should still be functional
                assert resource_manager.is_active
                assert scalability_manager.is_active
            else:
                allocations.append(alloc2)

            # Cleanup
            for alloc_id in allocations:
                if alloc_id:
                    await resource_manager.deallocate_resource(alloc_id)

        finally:
            await scalability_manager.shutdown()
            await resource_manager.shutdown()

    @pytest.mark.asyncio
    async def test_network_partition_handling(self) -> None:
        """Test system behavior under network partitions."""
        network_config = NetworkConfig(port=8084, message_timeout=2.0)
        network_coordinator = NetworkCoordinator(network_config, "partition_test_node")

        coord_config = DistributedCoordinationConfig(coordination_timeout=3.0)
        distributed_coordinator = DistributedCoordinator(coord_config)

        await network_coordinator.start_network_coordination()
        await distributed_coordinator.initialize_distributed_coordination()

        try:
            # Simulate network partition by trying to coordinate with unreachable nodes
            action_request = {"action": "test_partition"}

            start_time = time.time()
            try:
                await distributed_coordinator.coordinate_distributed_action(
                    action_request,
                    participating_nodes=["unreachable_node1", "unreachable_node2"],
                )
            except Exception:
                # Should handle partition gracefully
                coordination_time = time.time() - start_time
                assert coordination_time <= 5.0  # Should timeout reasonably

            # System should still be functional
            assert distributed_coordinator.is_active
            assert network_coordinator.is_active

        finally:
            await distributed_coordinator.shutdown()
            await network_coordinator.shutdown()

    @pytest.mark.asyncio
    async def test_scaling_limits_handling(self) -> None:
        """Test system behavior at scaling limits."""
        scaling_config = ScalingConfig(
            min_nodes=1,
            max_nodes=2,  # Very low limit
            scale_up_threshold=0.5,
            scale_down_threshold=0.2,
        )
        scalability_manager = ScalabilityManager(scaling_config)

        await scalability_manager.start_scalability_management()

        try:
            # Scale to maximum
            success = await scalability_manager.manual_scale(2, "Test max scaling")
            assert success

            # Try to scale beyond maximum
            success = await scalability_manager.manual_scale(5, "Test over-scaling")
            assert not success  # Should fail gracefully

            # Verify we're still at maximum
            status = scalability_manager.get_scaling_status()
            assert status["current_nodes"] == 2

            # Scale to minimum
            success = await scalability_manager.manual_scale(1, "Test min scaling")
            assert success

            # Try to scale below minimum
            success = await scalability_manager.manual_scale(0, "Test under-scaling")
            assert not success  # Should fail gracefully

            # Verify we're still at minimum
            status = scalability_manager.get_scaling_status()
            assert status["current_nodes"] == 1

        finally:
            await scalability_manager.shutdown()


class TestDistributedMARLPerformance:
    """Test performance characteristics of distributed MARL system."""

    @pytest.mark.asyncio
    async def test_concurrent_resource_operations(self) -> None:
        """Test concurrent resource operations performance."""
        config = ResourceConfig()
        resource_manager = ResourceManager(config)

        await resource_manager.start_resource_management()

        try:
            # Create concurrent allocation tasks
            async def allocate_and_deallocate(agent_id: str):
                allocation_id = await resource_manager.allocate_resource(
                    ResourceType.CPU, 0.5, f"agent_{agent_id}"
                )
                if allocation_id:
                    await asyncio.sleep(0.01)  # Simulate work
                    await resource_manager.deallocate_resource(allocation_id)
                return allocation_id is not None

            # Run concurrent operations
            start_time = time.time()
            tasks = [allocate_and_deallocate(str(i)) for i in range(20)]
            results = await asyncio.gather(*tasks)
            operation_time = time.time() - start_time

            # Verify performance
            assert operation_time < 2.0  # Should complete in reasonable time
            assert all(results)  # All operations should succeed

            # Verify no resource leaks
            usage = resource_manager.get_resource_usage()
            assert usage["total_allocations"] == 0

        finally:
            await resource_manager.shutdown()

    @pytest.mark.asyncio
    async def test_scaling_decision_latency(self) -> None:
        """Test scaling decision latency."""
        scaling_config = ScalingConfig(monitoring_interval=0.1)
        scalability_manager = ScalabilityManager(scaling_config)

        # Populate metrics history
        for i in range(50):
            metrics = scalability_manager._collect_scaling_metrics()
            scalability_manager.metrics_history.append(metrics)

        # Measure decision time
        start_time = time.time()
        decision = scalability_manager._evaluate_scaling_need()
        decision_time = time.time() - start_time

        # Should be very fast
        assert decision_time < 0.05  # Less than 50ms
        assert "action" in decision

    @pytest.mark.asyncio
    async def test_message_processing_throughput(self) -> None:
        """Test message processing throughput."""
        network_config = NetworkConfig(port=8085)
        network_coordinator = NetworkCoordinator(network_config, "throughput_test")

        await network_coordinator.start_network_coordination()

        try:
            # Queue many messages rapidly
            start_time = time.time()
            message_count = 100

            message_ids = []
            for i in range(message_count):
                message_id = await network_coordinator.send_message(
                    MessageType.HEARTBEAT, {"sequence": i, "timestamp": time.time()}
                )
                message_ids.append(message_id)

            queuing_time = time.time() - start_time

            # Verify throughput
            assert queuing_time < 1.0  # Should queue 100 messages in less than 1 second
            assert len(message_ids) == message_count
            assert all(mid is not None for mid in message_ids)

            # Check metrics
            metrics = network_coordinator.get_network_metrics()
            assert metrics["total_messages_sent"] >= message_count

        finally:
            await network_coordinator.shutdown()


if __name__ == "__main__":
    pytest.main([__file__])
