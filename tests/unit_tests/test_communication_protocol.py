"""
Unit tests for Agent Communication Protocol

Tests the communication protocol, message passing, broadcasting,
and response handling for multi-agent RL coordination.
"""

# Standard Library
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

# Third-Party Library
import pytest

# SynThesisAI Modules
from core.marl.config_legacy import CoordinationConfig
from core.marl.coordination.communication_protocol import (
    AgentCommunicationProtocol,
    AgentMessage,
    MessageResponse,
)
from core.marl.exceptions import CommunicationError


@pytest.fixture
def config() -> CoordinationConfig:
    """
    Create test coordination configuration.

    Returns:
        CoordinationConfig: configuration with message_queue_size and communication_timeout
    """
    return CoordinationConfig(
        message_queue_size=100,
        communication_timeout=5.0,
    )


@pytest.fixture
def protocol(config: CoordinationConfig) -> AgentCommunicationProtocol:
    """
    Create test AgentCommunicationProtocol.

    Args:
        config (CoordinationConfig): coordination configuration fixture

    Returns:
        AgentCommunicationProtocol: protocol instance with patched logger
    """
    with patch("core.marl.coordination.communication_protocol.get_marl_logger"):
        return AgentCommunicationProtocol(config)


@pytest.fixture
def sample_message() -> AgentMessage:
    """
    Create sample agent message.

    Returns:
        AgentMessage: sample coordination_request message with default priority
    """
    return AgentMessage(
        message_type="coordination_request",
        content={"strategy": "step_by_step", "confidence": 0.8},
        priority=1,
    )


class TestAgentMessage:
    """Test AgentMessage dataclass."""

    def test_initialization(self):
        """Test AgentMessage initialization."""
        message = AgentMessage(
            message_type="coordination_request",
            content={"strategy": "step_by_step", "confidence": 0.8},
            priority=2,
            sender="generator",
            receiver="validator",
        )

        assert message.message_type == "coordination_request"
        assert message.content["strategy"] == "step_by_step"
        assert message.priority == 2
        assert message.sender == "generator"
        assert message.receiver == "validator"
        assert isinstance(message.timestamp, datetime)
        assert message.message_id.startswith("msg_")

    def test_default_values(self):
        """Test AgentMessage default values."""
        message = AgentMessage(
            message_type="test",
            content={"data": "test"},
        )

        assert message.priority == 1
        assert message.sender == ""
        assert message.receiver == ""
        assert message.requires_response == False
        assert message.response_timeout == 10.0
        assert message.metadata == {}

    def test_message_id_generation(self):
        """Test unique message ID generation."""
        message1 = AgentMessage(message_type="test1", content={})
        message2 = AgentMessage(message_type="test2", content={})

        assert message1.message_id != message2.message_id
        assert message1.message_id.startswith("msg_")
        assert message2.message_id.startswith("msg_")


class TestMessageResponse:
    """Test MessageResponse dataclass."""

    def test_initialization(self):
        """Test MessageResponse initialization."""
        response = MessageResponse(
            original_message_id="msg_123",
            response_content={"result": "success", "data": {"value": 42}},
            sender="validator",
        )

        assert response.original_message_id == "msg_123"
        assert response.response_content["result"] == "success"
        assert response.sender == "validator"
        assert isinstance(response.timestamp, datetime)
        assert response.success == True
        assert response.error_message is None

    def test_error_response(self):
        """Test error response creation."""
        response = MessageResponse(
            original_message_id="msg_456",
            response_content={},
            sender="curriculum",
            success=False,
            error_message="Processing failed",
        )

        assert response.success == False
        assert response.error_message == "Processing failed"


class TestAgentCommunicationProtocol:
    """Test AgentCommunicationProtocol class."""

    def test_initialization(self, protocol, config):
        """Test AgentCommunicationProtocol initialization."""
        assert protocol.config == config
        assert isinstance(protocol.message_queue, asyncio.Queue)
        assert protocol.agent_channels == {}
        assert protocol.registered_agents == set()
        assert protocol.agent_status == {}
        assert protocol.communication_metrics["total_messages_sent"] == 0

    @pytest.mark.asyncio
    async def test_register_agent(self, protocol):
        """Test agent registration."""
        await protocol.register_agent("generator")

        assert "generator" in protocol.registered_agents
        assert "generator" in protocol.agent_channels
        assert protocol.agent_status["generator"] == "online"
        assert isinstance(protocol.agent_channels["generator"], asyncio.Queue)

    @pytest.mark.asyncio
    async def test_register_multiple_agents(self, protocol):
        """Test multiple agent registration."""
        agents = ["generator", "validator", "curriculum"]

        for agent in agents:
            await protocol.register_agent(agent)

        assert len(protocol.registered_agents) == 3
        assert all(agent in protocol.registered_agents for agent in agents)
        assert all(protocol.agent_status[agent] == "online" for agent in agents)

    @pytest.mark.asyncio
    async def test_unregister_agent(self, protocol):
        """Test agent unregistration."""
        await protocol.register_agent("generator")
        assert "generator" in protocol.registered_agents

        await protocol.unregister_agent("generator")
        assert "generator" not in protocol.registered_agents
        assert protocol.agent_status["generator"] == "offline"

    @pytest.mark.asyncio
    async def test_send_message_success(self, protocol, sample_message):
        """Test successful message sending."""
        # Register agents
        await protocol.register_agent("generator")
        await protocol.register_agent("validator")

        # Send message
        response = await protocol.send_message("generator", "validator", sample_message)

        assert response is None  # No response expected for this message
        assert protocol.communication_metrics["total_messages_sent"] == 1
        assert len(protocol.message_history) == 1

        # Check message was added to receiver's queue
        validator_queue = protocol.agent_channels["validator"]
        assert validator_queue.qsize() == 1

        # Verify message content
        received_message = await validator_queue.get()
        assert received_message.sender == "generator"
        assert received_message.receiver == "validator"
        assert received_message.message_type == "coordination_request"

    @pytest.mark.asyncio
    async def test_send_message_unregistered_sender(self, protocol, sample_message):
        """Test sending message from unregistered sender."""
        await protocol.register_agent("validator")

        with pytest.raises(CommunicationError) as exc_info:
            await protocol.send_message("generator", "validator", sample_message)

        assert "Sender agent not registered" in str(exc_info.value)
        assert protocol.communication_metrics["failed_deliveries"] == 1

    @pytest.mark.asyncio
    async def test_send_message_unregistered_receiver(self, protocol, sample_message):
        """Test sending message to unregistered receiver."""
        await protocol.register_agent("generator")

        with pytest.raises(CommunicationError) as exc_info:
            await protocol.send_message("generator", "validator", sample_message)

        assert "Receiver agent not registered" in str(exc_info.value)
        assert protocol.communication_metrics["failed_deliveries"] == 1

    @pytest.mark.asyncio
    async def test_broadcast_message(self, protocol):
        """Test message broadcasting."""
        # Register multiple agents
        agents = ["generator", "validator", "curriculum"]
        for agent in agents:
            await protocol.register_agent(agent)

        # Create broadcast message
        broadcast_message = AgentMessage(
            message_type="system_announcement",
            content={"announcement": "System update available"},
        )

        # Broadcast from generator
        receivers = await protocol.broadcast_message("generator", broadcast_message)

        # Should receive by all except sender
        expected_receivers = ["validator", "curriculum"]
        assert set(receivers) == set(expected_receivers)
        assert protocol.communication_metrics["total_broadcasts"] == 1
        assert protocol.communication_metrics["total_messages_sent"] == 2

        # Check each receiver got the message
        for receiver in expected_receivers:
            queue = protocol.agent_channels[receiver]
            assert queue.qsize() == 1

            received_message = await queue.get()
            assert received_message.sender == "generator"
            assert received_message.receiver == "all"
            assert received_message.message_type == "system_announcement"

    @pytest.mark.asyncio
    async def test_broadcast_unregistered_sender(self, protocol):
        """Test broadcasting from unregistered sender."""
        await protocol.register_agent("validator")

        broadcast_message = AgentMessage(message_type="test", content={"data": "test"})

        with pytest.raises(CommunicationError) as exc_info:
            await protocol.broadcast_message("generator", broadcast_message)

        assert "Sender agent not registered" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_receive_messages(self, protocol):
        """Test message receiving."""
        # Register agents
        await protocol.register_agent("generator")
        await protocol.register_agent("validator")

        # Send multiple messages
        messages = [
            AgentMessage(message_type="request1", content={"data": 1}),
            AgentMessage(message_type="request2", content={"data": 2}),
            AgentMessage(message_type="request3", content={"data": 3}),
        ]

        for message in messages:
            await protocol.send_message("generator", "validator", message)

        # Receive messages
        received_messages = await protocol.receive_messages("validator", timeout=1.0)

        assert len(received_messages) == 3
        assert protocol.communication_metrics["total_messages_received"] == 3

        # Check message order and content
        for i, received_message in enumerate(received_messages):
            assert received_message.message_type == f"request{i + 1}"
            assert received_message.content["data"] == i + 1

    @pytest.mark.asyncio
    async def test_receive_messages_unregistered_agent(self, protocol):
        """Test receiving messages for unregistered agent."""
        with pytest.raises(CommunicationError) as exc_info:
            await protocol.receive_messages("generator")

        assert "Agent not registered" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_receive_messages_timeout(self, protocol):
        """Test message receiving with timeout."""
        await protocol.register_agent("validator")

        # No messages sent, should timeout quickly
        received_messages = await protocol.receive_messages("validator", timeout=0.1)

        assert len(received_messages) == 0

    @pytest.mark.asyncio
    async def test_send_response(self, protocol):
        """Test sending response to a message."""
        # Register agents
        await protocol.register_agent("generator")
        await protocol.register_agent("validator")

        # Send original message
        original_message = AgentMessage(
            message_type="validation_request",
            content={"content": "test content"},
            requires_response=True,
        )

        await protocol.send_message("generator", "validator", original_message)

        # Receive the message
        received_messages = await protocol.receive_messages("validator", timeout=1.0)
        assert len(received_messages) == 1

        original_received = received_messages[0]

        # Send response
        response_content = {"validation_result": "passed", "score": 0.85}
        await protocol.send_response("validator", original_received, response_content)

        # Check response was sent back to generator
        response_messages = await protocol.receive_messages("generator", timeout=1.0)
        assert len(response_messages) == 1

        response_message = response_messages[0]
        assert response_message.message_type == "response"
        assert response_message.sender == "validator"
        assert response_message.receiver == "generator"
        assert response_message.content["response"]["validation_result"] == "passed"

    @pytest.mark.asyncio
    async def test_message_with_response_requirement(self, protocol):
        """Test message that requires response."""
        # Register agents
        await protocol.register_agent("generator")
        await protocol.register_agent("validator")

        # Create message requiring response
        message = AgentMessage(
            message_type="validation_request",
            content={"content": "test content"},
            requires_response=True,
            response_timeout=2.0,
        )

        # Send message and wait for response (will timeout since no response sent)
        response = await protocol.send_message(
            "generator", "validator", message, wait_for_response=True
        )

        # Should timeout and return None
        assert response is None

    def test_add_routing_rule(self, protocol):
        """Test adding message routing rules."""
        protocol.add_routing_rule("validation_request", "validator")
        protocol.add_routing_rule("curriculum_advice", "curriculum")

        assert protocol.routing_rules["validation_request"] == "validator"
        assert protocol.routing_rules["curriculum_advice"] == "curriculum"

    def test_add_message_filter(self, protocol):
        """Test adding message filters."""

        def priority_filter(message):
            return message.priority >= 2

        def type_filter(message):
            return message.message_type != "spam"

        protocol.add_message_filter("priority", priority_filter)
        protocol.add_message_filter("type", type_filter)

        assert "priority" in protocol.message_filters
        assert "type" in protocol.message_filters

        # Test filter application
        high_priority_msg = AgentMessage(message_type="test", content={}, priority=3)
        low_priority_msg = AgentMessage(message_type="test", content={}, priority=1)
        spam_msg = AgentMessage(message_type="spam", content={}, priority=2)

        assert protocol._apply_message_filters(high_priority_msg) == True
        assert protocol._apply_message_filters(low_priority_msg) == False
        assert protocol._apply_message_filters(spam_msg) == False

    def test_agent_status_management(self, protocol):
        """Test agent status management."""
        # Initially unknown
        assert protocol.get_agent_status("generator") == "unknown"

        # Set status
        protocol.set_agent_status("generator", "busy")
        assert protocol.get_agent_status("generator") == "busy"

        protocol.set_agent_status("generator", "online")
        assert protocol.get_agent_status("generator") == "online"

    def test_message_history(self, protocol):
        """Test message history functionality."""
        # Add some messages to history
        messages = [
            AgentMessage(
                message_type="type1",
                content={},
                sender="generator",
                receiver="validator",
            ),
            AgentMessage(
                message_type="type2",
                content={},
                sender="validator",
                receiver="curriculum",
            ),
            AgentMessage(
                message_type="type1",
                content={},
                sender="curriculum",
                receiver="generator",
            ),
        ]

        protocol.message_history.extend(messages)

        # Get all history
        all_history = protocol.get_message_history()
        assert len(all_history) == 3

        # Filter by agent
        generator_history = protocol.get_message_history(agent_id="generator")
        assert len(generator_history) == 2  # generator as sender or receiver

        # Filter by message type
        type1_history = protocol.get_message_history(message_type="type1")
        assert len(type1_history) == 2

        # Filter by both
        generator_type1 = protocol.get_message_history(
            agent_id="generator", message_type="type1"
        )
        assert len(generator_type1) == 2

        # Test limit
        limited_history = protocol.get_message_history(limit=2)
        assert len(limited_history) == 2

    def test_communication_metrics(self, protocol):
        """Test communication metrics collection."""
        # Simulate some activity
        protocol.communication_metrics["total_messages_sent"] = 10
        protocol.communication_metrics["failed_deliveries"] = 1
        protocol.communication_metrics["total_broadcasts"] = 2
        protocol.communication_metrics["message_types"]["request"] = 5
        protocol.communication_metrics["message_types"]["response"] = 3

        protocol.registered_agents.update(["generator", "validator", "curriculum"])
        protocol.agent_status.update(
            {"generator": "online", "validator": "busy", "curriculum": "online"}
        )

        metrics = protocol.get_communication_metrics()

        assert metrics["success_rate"] == 0.9  # (10-1)/10
        assert metrics["registered_agents"] == ["curriculum", "generator", "validator"]
        assert metrics["performance_status"] == "good"  # 90% success rate
        assert "communication_metrics" in metrics
        assert "agent_status" in metrics

    @pytest.mark.asyncio
    async def test_shutdown(self, protocol):
        """Test protocol shutdown."""
        # Register agents and add some data
        await protocol.register_agent("generator")
        await protocol.register_agent("validator")

        # Add some messages
        message = AgentMessage(message_type="test", content={})
        await protocol.send_message("generator", "validator", message)

        # Shutdown
        await protocol.shutdown()

        # Check cleanup
        assert len(protocol.registered_agents) == 0
        assert len(protocol.agent_status) == 0
        assert len(protocol.pending_responses) == 0
        assert len(protocol.response_futures) == 0

    def test_message_routing(self, protocol):
        """Test message routing functionality."""
        # Add routing rule
        protocol.add_routing_rule("validation_request", "validator")

        # Test routing
        message = AgentMessage(
            message_type="validation_request", content={}, receiver="generator"
        )
        routed_receiver = protocol._route_message(message)

        assert routed_receiver == "validator"  # Should be routed to validator

        # Test message without routing rule
        message2 = AgentMessage(
            message_type="unknown_type", content={}, receiver="curriculum"
        )
        routed_receiver2 = protocol._route_message(message2)

        assert routed_receiver2 == "curriculum"  # Should use original receiver

    def test_message_type_metrics_update(self, protocol):
        """Test message type metrics updating."""
        protocol._update_message_type_metrics("request")
        protocol._update_message_type_metrics("request")
        protocol._update_message_type_metrics("response")

        assert protocol.communication_metrics["message_types"]["request"] == 2
        assert protocol.communication_metrics["message_types"]["response"] == 1

    def test_message_history_maintenance(self, protocol):
        """Test message history size maintenance."""
        # Add many messages to trigger maintenance
        for i in range(15000):  # More than max_history_size (10000)
            message = AgentMessage(message_type=f"test_{i}", content={})
            protocol.message_history.append(message)

        protocol._maintain_message_history()

        # Should be trimmed to max size
        assert len(protocol.message_history) == 10000
        # Should keep most recent messages
        assert protocol.message_history[-1].message_type == "test_14999"


@pytest.mark.integration
class TestCommunicationProtocolIntegration:
    """Integration tests for AgentCommunicationProtocol."""

    @pytest.fixture
    def config(self):
        """Create test coordination configuration."""
        return CoordinationConfig(
            message_queue_size=50,
            communication_timeout=2.0,
        )

    @pytest.fixture
    def protocol(self, config):
        """Create test AgentCommunicationProtocol."""
        with patch("core.marl.coordination.communication_protocol.get_marl_logger"):
            return AgentCommunicationProtocol(config)

    @pytest.mark.asyncio
    async def test_full_communication_workflow(self, protocol):
        """Test complete communication workflow."""
        # Register agents
        agents = ["generator", "validator", "curriculum"]
        for agent in agents:
            await protocol.register_agent(agent)

        # Generator requests validation
        validation_request = AgentMessage(
            message_type="validation_request",
            content={"content": "Mathematical proof", "domain": "mathematics"},
            requires_response=True,
            response_timeout=5.0,
        )

        # Send request (don't wait for response in this test)
        await protocol.send_message("generator", "validator", validation_request)

        # Validator receives and processes request
        validator_messages = await protocol.receive_messages("validator", timeout=1.0)
        assert len(validator_messages) == 1

        received_request = validator_messages[0]
        assert received_request.message_type == "validation_request"
        assert received_request.content["domain"] == "mathematics"

        # Validator sends response
        validation_response = {
            "validation_result": "approved",
            "quality_score": 0.92,
            "feedback": "Excellent mathematical reasoning",
        }

        await protocol.send_response("validator", received_request, validation_response)

        # Generator receives response
        generator_messages = await protocol.receive_messages("generator", timeout=1.0)
        assert len(generator_messages) == 1

        response_message = generator_messages[0]
        assert response_message.message_type == "response"
        assert response_message.content["response"]["validation_result"] == "approved"

        # Broadcast system message
        system_message = AgentMessage(
            message_type="system_update",
            content={"update": "New quality standards implemented"},
        )

        receivers = await protocol.broadcast_message("validator", system_message)
        assert set(receivers) == {"generator", "curriculum"}

        # Check metrics
        metrics = protocol.get_communication_metrics()
        assert metrics["communication_metrics"]["total_messages_sent"] >= 3
        assert metrics["communication_metrics"]["total_broadcasts"] == 1
        assert metrics["success_rate"] > 0.9

    @pytest.mark.asyncio
    async def test_concurrent_communication(self, protocol):
        """Test concurrent message handling."""
        # Register agents
        agents = ["generator", "validator", "curriculum"]
        for agent in agents:
            await protocol.register_agent(agent)

        # Create multiple concurrent communication tasks
        async def send_messages(sender, receiver, count):
            tasks = []
            for i in range(count):
                message = AgentMessage(
                    message_type=f"message_{i}",
                    content={"data": i, "sender": sender},
                )
                task = protocol.send_message(sender, receiver, message)
                tasks.append(task)
            await asyncio.gather(*tasks)

        # Send messages concurrently
        await asyncio.gather(
            send_messages("generator", "validator", 5),
            send_messages("validator", "curriculum", 3),
            send_messages("curriculum", "generator", 4),
        )

        # Receive all messages
        validator_messages = await protocol.receive_messages("validator", timeout=2.0)
        curriculum_messages = await protocol.receive_messages("curriculum", timeout=2.0)
        generator_messages = await protocol.receive_messages("generator", timeout=2.0)

        assert len(validator_messages) == 5
        assert len(curriculum_messages) == 3
        assert len(generator_messages) == 4

        # Check total metrics
        metrics = protocol.get_communication_metrics()
        assert metrics["communication_metrics"]["total_messages_sent"] == 12

    @pytest.mark.asyncio
    async def test_error_recovery(self, protocol):
        """Test error recovery and resilience."""
        # Register agents
        await protocol.register_agent("generator")
        await protocol.register_agent("validator")

        # Send valid message
        valid_message = AgentMessage(message_type="test", content={"valid": True})
        await protocol.send_message("generator", "validator", valid_message)

        # Try to send to unregistered agent (should fail gracefully)
        with pytest.raises(CommunicationError):
            await protocol.send_message("generator", "unknown", valid_message)

        # System should still be functional
        another_message = AgentMessage(message_type="test2", content={"valid": True})
        await protocol.send_message("generator", "validator", another_message)

        # Receive messages
        messages = await protocol.receive_messages("validator", timeout=1.0)
        assert len(messages) == 2  # Both valid messages should be received

        # Check that failed delivery was recorded
        metrics = protocol.get_communication_metrics()
        assert metrics["communication_metrics"]["failed_deliveries"] == 1


if __name__ == "__main__":
    pytest.main([__file__])
