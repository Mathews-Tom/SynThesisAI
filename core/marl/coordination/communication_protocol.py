"""
Agent Communication Protocol

This module implements the communication protocol for multi-agent RL coordination,
including message queuing, routing, broadcasting, and message history management.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from ..config import CoordinationConfig
from ..exceptions import CommunicationError
from ..logging_config import get_marl_logger

logger = logging.getLogger(__name__)


@dataclass
class AgentMessage:
    """Represents a message between agents."""

    message_type: str
    content: Dict[str, Any]
    priority: int = 1
    sender: str = ""
    receiver: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    message_id: str = field(default_factory=lambda: f"msg_{int(time.time() * 1000000)}")
    requires_response: bool = False
    response_timeout: float = 10.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Post-initialization processing."""
        if not self.message_id:
            self.message_id = f"msg_{int(time.time() * 1000000)}"


@dataclass
class MessageResponse:
    """Represents a response to an agent message."""

    original_message_id: str
    response_content: Dict[str, Any]
    sender: str
    timestamp: datetime = field(default_factory=datetime.now)
    success: bool = True
    error_message: Optional[str] = None


class AgentCommunicationProtocol:
    """
    Communication protocol system for multi-agent coordination.

    This class manages message passing, queuing, routing, and history
    for communication between RL agents in the coordination system.
    """

    def __init__(self, config: CoordinationConfig):
        """
        Initialize agent communication protocol.

        Args:
            config: Coordination configuration parameters
        """
        self.config = config
        self.logger = get_marl_logger("communication_protocol")

        # Message queuing system
        self.message_queue = asyncio.Queue(maxsize=config.message_queue_size)
        self.agent_channels: Dict[str, asyncio.Queue] = {}
        self.broadcast_channel = asyncio.Queue()

        # Message history and tracking
        self.message_history: List[AgentMessage] = []
        self.pending_responses: Dict[str, AgentMessage] = {}
        self.response_futures: Dict[str, asyncio.Future] = {}

        # Agent registration and status
        self.registered_agents: Set[str] = set()
        self.agent_status: Dict[str, str] = {}  # online, offline, busy

        # Communication metrics
        self.communication_metrics = {
            "total_messages_sent": 0,
            "total_messages_received": 0,
            "total_broadcasts": 0,
            "failed_deliveries": 0,
            "average_response_time": 0.0,
            "message_types": {},
        }

        # Message routing rules
        self.routing_rules: Dict[str, str] = {}
        self.message_filters: Dict[str, callable] = {}

        self.logger.log_agent_action(
            "communication_protocol",
            "initialized",
            1.0,
            "Queue size: %d" % config.message_queue_size,
        )

    async def register_agent(self, agent_id: str) -> None:
        """
        Register an agent for communication.

        Args:
            agent_id: Unique identifier for the agent
        """
        try:
            if agent_id not in self.agent_channels:
                self.agent_channels[agent_id] = asyncio.Queue()

            self.registered_agents.add(agent_id)
            self.agent_status[agent_id] = "online"

            self.logger.log_coordination_event(
                "agent_registered",
                {"agent_id": agent_id, "total_agents": len(self.registered_agents)},
            )

        except Exception as e:
            self.logger.log_error_with_context(
                e, {"agent_id": agent_id, "operation": "register_agent"}
            )
            raise CommunicationError(
                "Failed to register agent for communication",
                sender_id=agent_id,
                receiver_id="system",
                operation="registration",
            ) from e

    async def unregister_agent(self, agent_id: str) -> None:
        """
        Unregister an agent from communication.

        Args:
            agent_id: Unique identifier for the agent
        """
        try:
            if agent_id in self.registered_agents:
                self.registered_agents.remove(agent_id)

            self.agent_status[agent_id] = "offline"

            # Clear pending messages for this agent
            if agent_id in self.agent_channels:
                while not self.agent_channels[agent_id].empty():
                    try:
                        self.agent_channels[agent_id].get_nowait()
                    except asyncio.QueueEmpty:
                        break

            self.logger.log_coordination_event(
                "agent_unregistered",
                {"agent_id": agent_id, "remaining_agents": len(self.registered_agents)},
            )

        except Exception as e:
            self.logger.log_error_with_context(
                e, {"agent_id": agent_id, "operation": "unregister_agent"}
            )

    async def send_message(
        self,
        sender: str,
        receiver: str,
        message: AgentMessage,
        wait_for_response: bool = False,
    ) -> Optional[MessageResponse]:
        """
        Send message between agents.

        Args:
            sender: ID of sending agent
            receiver: ID of receiving agent
            message: Message to send
            wait_for_response: Whether to wait for a response

        Returns:
            MessageResponse if wait_for_response is True, None otherwise

        Raises:
            CommunicationError: If message sending fails
        """
        try:
            # Validate agents
            if sender not in self.registered_agents:
                self.communication_metrics["failed_deliveries"] += 1
                raise CommunicationError(
                    "Sender agent not registered",
                    sender_id=sender,
                    receiver_id="unknown",
                )

            if receiver not in self.registered_agents:
                self.communication_metrics["failed_deliveries"] += 1
                raise CommunicationError(
                    "Receiver agent not registered",
                    sender_id="unknown",
                    receiver_id=receiver,
                )

            # Set message metadata
            message.sender = sender
            message.receiver = receiver
            message.timestamp = datetime.now()

            # Apply message filters
            if not self._apply_message_filters(message):
                self.logger.log_coordination_event(
                    "message_filtered",
                    {
                        "sender": sender,
                        "receiver": receiver,
                        "message_type": message.message_type,
                    },
                )
                return None

            # Route message
            actual_receiver = self._route_message(message)

            # Add to message queue
            await self.message_queue.put(message)

            # Add to receiver's channel
            if actual_receiver in self.agent_channels:
                await self.agent_channels[actual_receiver].put(message)

            # Update metrics
            self.communication_metrics["total_messages_sent"] += 1
            self._update_message_type_metrics(message.message_type)

            # Add to history
            self.message_history.append(message)
            self._maintain_message_history()

            # Handle response waiting
            response = None
            if wait_for_response or message.requires_response:
                response = await self._wait_for_response(message)

            self.logger.log_coordination_event(
                "message_sent",
                {
                    "sender": sender,
                    "receiver": actual_receiver,
                    "message_type": message.message_type,
                    "message_id": message.message_id,
                    "requires_response": message.requires_response,
                },
            )

            return response

        except CommunicationError:
            # Re-raise communication errors as-is
            raise
        except Exception as e:
            self.communication_metrics["failed_deliveries"] += 1
            self.logger.log_error_with_context(
                e,
                {
                    "sender": sender,
                    "receiver": receiver,
                    "message_type": message.message_type,
                },
            )
            raise CommunicationError(
                "Failed to send message",
                sender_id=sender,
                receiver_id=receiver,
                message_type=message.message_type,
            ) from e

    async def broadcast_message(self, sender: str, message: AgentMessage) -> List[str]:
        """
        Broadcast message to all registered agents.

        Args:
            sender: ID of sending agent
            message: Message to broadcast

        Returns:
            List of agent IDs that received the message

        Raises:
            CommunicationError: If broadcast fails
        """
        try:
            if sender not in self.registered_agents:
                self.communication_metrics["failed_deliveries"] += 1
                raise CommunicationError(
                    "Sender agent not registered",
                    sender_id=sender,
                    receiver_id="broadcast",
                )

            # Set message metadata
            message.sender = sender
            message.receiver = "all"
            message.timestamp = datetime.now()

            # Get list of receivers (all except sender)
            receivers = [
                agent_id
                for agent_id in self.registered_agents
                if agent_id != sender and self.agent_status.get(agent_id) == "online"
            ]

            # Send to broadcast channel
            await self.broadcast_channel.put(message)

            # Send to each agent's channel
            successful_deliveries = []
            for receiver in receivers:
                try:
                    if receiver in self.agent_channels:
                        await self.agent_channels[receiver].put(message)
                        successful_deliveries.append(receiver)
                except Exception as e:
                    self.logger.log_error_with_context(
                        e,
                        {"receiver": receiver, "broadcast_message": message.message_id},
                    )

            # Update metrics
            self.communication_metrics["total_broadcasts"] += 1
            self.communication_metrics["total_messages_sent"] += len(
                successful_deliveries
            )
            self._update_message_type_metrics(message.message_type)

            # Add to history
            self.message_history.append(message)
            self._maintain_message_history()

            self.logger.log_coordination_event(
                "message_broadcast",
                {
                    "sender": sender,
                    "receivers": successful_deliveries,
                    "message_type": message.message_type,
                    "message_id": message.message_id,
                },
            )

            return successful_deliveries

        except CommunicationError:
            # Re-raise communication errors as-is
            raise
        except Exception as e:
            self.communication_metrics["failed_deliveries"] += 1
            self.logger.log_error_with_context(
                e, {"sender": sender, "message_type": message.message_type}
            )
            raise CommunicationError(
                "Failed to broadcast message",
                sender_id=sender,
                receiver_id="broadcast",
                message_type=message.message_type,
            ) from e

    async def receive_messages(
        self, agent_id: str, timeout: Optional[float] = None
    ) -> List[AgentMessage]:
        """
        Receive messages for specific agent.

        Args:
            agent_id: ID of receiving agent
            timeout: Timeout for receiving messages

        Returns:
            List of received messages

        Raises:
            CommunicationError: If message receiving fails
        """
        try:
            if agent_id not in self.registered_agents:
                self.communication_metrics["failed_deliveries"] += 1
                raise CommunicationError(
                    "Agent not registered", sender_id="system", receiver_id=agent_id
                )

            messages = []
            agent_queue = self.agent_channels.get(agent_id)

            if not agent_queue:
                return messages

            # Collect all available messages
            timeout_per_message = (
                timeout / 10 if timeout else 0.1
            )  # Small timeout per message

            while True:
                try:
                    message = await asyncio.wait_for(
                        agent_queue.get(), timeout=timeout_per_message
                    )
                    messages.append(message)

                    # Update metrics
                    self.communication_metrics["total_messages_received"] += 1

                    # Handle response messages
                    if message.message_type == "response":
                        await self._handle_response_message(message)

                except asyncio.TimeoutError:
                    break  # No more messages available

            self.logger.log_coordination_event(
                "messages_received",
                {
                    "agent_id": agent_id,
                    "message_count": len(messages),
                    "message_types": [m.message_type for m in messages],
                },
            )

            return messages

        except CommunicationError:
            # Re-raise communication errors as-is
            raise
        except Exception as e:
            self.logger.log_error_with_context(
                e, {"agent_id": agent_id, "operation": "receive_messages"}
            )
            raise CommunicationError(
                "Failed to receive messages", sender_id="system", receiver_id=agent_id
            ) from e

    async def send_response(
        self,
        responder: str,
        original_message: AgentMessage,
        response_content: Dict[str, Any],
    ) -> None:
        """
        Send response to a message.

        Args:
            responder: ID of responding agent
            original_message: Original message being responded to
            response_content: Content of the response
        """
        try:
            response = MessageResponse(
                original_message_id=original_message.message_id,
                response_content=response_content,
                sender=responder,
            )

            # Create response message
            response_message = AgentMessage(
                message_type="response",
                content={
                    "response": response_content,
                    "original_message_id": original_message.message_id,
                },
                sender=responder,
                receiver=original_message.sender,
                priority=original_message.priority,
            )

            # Send response
            await self.send_message(
                responder, original_message.sender, response_message
            )

            self.logger.log_coordination_event(
                "response_sent",
                {
                    "responder": responder,
                    "original_sender": original_message.sender,
                    "original_message_id": original_message.message_id,
                },
            )

        except Exception as e:
            self.logger.log_error_with_context(
                e,
                {
                    "responder": responder,
                    "original_message_id": original_message.message_id,
                },
            )
            raise CommunicationError(
                "Failed to send response",
                sender_id=responder,
                receiver_id="system",
                operation="send_response",
            ) from e

    def add_routing_rule(self, message_type: str, target_agent: str) -> None:
        """
        Add message routing rule.

        Args:
            message_type: Type of message to route
            target_agent: Target agent for this message type
        """
        self.routing_rules[message_type] = target_agent
        self.logger.log_coordination_event(
            "routing_rule_added",
            {"message_type": message_type, "target_agent": target_agent},
        )

    def add_message_filter(self, filter_name: str, filter_func: callable) -> None:
        """
        Add message filter.

        Args:
            filter_name: Name of the filter
            filter_func: Function to filter messages (returns bool)
        """
        self.message_filters[filter_name] = filter_func
        self.logger.log_coordination_event(
            "message_filter_added", {"filter_name": filter_name}
        )

    def get_agent_status(self, agent_id: str) -> str:
        """Get status of an agent."""
        return self.agent_status.get(agent_id, "unknown")

    def set_agent_status(self, agent_id: str, status: str) -> None:
        """Set status of an agent."""
        self.agent_status[agent_id] = status
        self.logger.log_coordination_event(
            "agent_status_changed", {"agent_id": agent_id, "status": status}
        )

    def get_message_history(
        self,
        agent_id: Optional[str] = None,
        message_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[AgentMessage]:
        """
        Get message history with optional filtering.

        Args:
            agent_id: Filter by specific agent (sender or receiver)
            message_type: Filter by message type
            limit: Maximum number of messages to return

        Returns:
            List of filtered messages
        """
        filtered_messages = self.message_history

        if agent_id:
            filtered_messages = [
                msg
                for msg in filtered_messages
                if msg.sender == agent_id or msg.receiver == agent_id
            ]

        if message_type:
            filtered_messages = [
                msg for msg in filtered_messages if msg.message_type == message_type
            ]

        # Return most recent messages first
        return filtered_messages[-limit:]

    def _route_message(self, message: AgentMessage) -> str:
        """Route message based on routing rules."""
        if message.message_type in self.routing_rules:
            return self.routing_rules[message.message_type]
        return message.receiver

    def _apply_message_filters(self, message: AgentMessage) -> bool:
        """Apply message filters to determine if message should be sent."""
        for filter_name, filter_func in self.message_filters.items():
            try:
                if not filter_func(message):
                    return False
            except Exception as e:
                self.logger.log_error_with_context(
                    e, {"filter_name": filter_name, "message_id": message.message_id}
                )
                # Continue with other filters if one fails

        return True

    async def _wait_for_response(
        self, message: AgentMessage
    ) -> Optional[MessageResponse]:
        """Wait for response to a message."""
        if not message.requires_response:
            return None

        try:
            # Create future for response
            response_future = asyncio.Future()
            self.response_futures[message.message_id] = response_future
            self.pending_responses[message.message_id] = message

            # Wait for response with timeout
            response = await asyncio.wait_for(
                response_future, timeout=message.response_timeout
            )

            return response

        except asyncio.TimeoutError:
            self.logger.log_coordination_event(
                "response_timeout",
                {
                    "message_id": message.message_id,
                    "sender": message.sender,
                    "receiver": message.receiver,
                },
            )
            return None
        finally:
            # Clean up
            self.response_futures.pop(message.message_id, None)
            self.pending_responses.pop(message.message_id, None)

    async def _handle_response_message(self, message: AgentMessage) -> None:
        """Handle incoming response message."""
        original_message_id = message.content.get("original_message_id")

        if original_message_id in self.response_futures:
            response = MessageResponse(
                original_message_id=original_message_id,
                response_content=message.content.get("response", {}),
                sender=message.sender,
            )

            # Complete the future
            future = self.response_futures[original_message_id]
            if not future.done():
                future.set_result(response)

    def _update_message_type_metrics(self, message_type: str) -> None:
        """Update metrics for message types."""
        if message_type not in self.communication_metrics["message_types"]:
            self.communication_metrics["message_types"][message_type] = 0
        self.communication_metrics["message_types"][message_type] += 1

    def _maintain_message_history(self) -> None:
        """Maintain message history size."""
        max_history_size = 10000
        if len(self.message_history) > max_history_size:
            # Keep most recent messages
            self.message_history = self.message_history[-max_history_size:]

    def get_communication_metrics(self) -> Dict[str, Any]:
        """Get comprehensive communication metrics."""
        total_messages = self.communication_metrics["total_messages_sent"]
        failed_deliveries = self.communication_metrics["failed_deliveries"]

        success_rate = (
            (total_messages - failed_deliveries) / max(total_messages, 1)
            if total_messages > 0
            else 0.0
        )

        return {
            "communication_metrics": self.communication_metrics.copy(),
            "registered_agents": sorted(list(self.registered_agents)),
            "agent_status": self.agent_status.copy(),
            "success_rate": success_rate,
            "pending_responses": len(self.pending_responses),
            "message_queue_size": self.message_queue.qsize(),
            "performance_status": "excellent"
            if success_rate > 0.95
            else "good"
            if success_rate > 0.85
            else "needs_improvement",
        }

    async def shutdown(self) -> None:
        """Shutdown communication protocol and clean up resources."""
        try:
            # Clear all queues
            while not self.message_queue.empty():
                try:
                    self.message_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

            for agent_id, queue in self.agent_channels.items():
                while not queue.empty():
                    try:
                        queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break

            # Cancel pending response futures
            for future in self.response_futures.values():
                if not future.done():
                    future.cancel()

            # Clear data structures
            self.registered_agents.clear()
            self.agent_status.clear()
            self.pending_responses.clear()
            self.response_futures.clear()

            self.logger.log_coordination_event(
                "communication_protocol_shutdown",
                {"final_metrics": self.get_communication_metrics()},
            )

        except Exception as e:
            self.logger.log_error_with_context(e, {"operation": "shutdown"})
