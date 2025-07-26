"""Network Coordination System for Distributed MARL.

This module provides network-level coordination capabilities for distributed
MARL deployment, including network topology management and communication optimization.
"""

import asyncio
import json
import socket
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from utils.logging_config import get_logger


class NetworkProtocol(Enum):
    """Network protocol enumeration."""

    TCP = "tcp"
    UDP = "udp"
    HTTP = "http"
    WEBSOCKET = "websocket"


class MessageType(Enum):
    """Message type enumeration."""

    HEARTBEAT = "heartbeat"
    COORDINATION_REQUEST = "coordination_request"
    COORDINATION_RESPONSE = "coordination_response"
    RESOURCE_UPDATE = "resource_update"
    TOPOLOGY_UPDATE = "topology_update"
    AGENT_MESSAGE = "agent_message"


@dataclass
class NetworkConfig:
    """Configuration for network coordination."""

    # Network settings
    host: str = "localhost"
    port: int = 8080
    protocol: NetworkProtocol = NetworkProtocol.TCP
    max_connections: int = 100

    # Communication settings
    message_timeout: float = 30.0
    heartbeat_interval: float = 5.0
    connection_retry_attempts: int = 3
    connection_retry_delay: float = 1.0

    # Performance settings
    buffer_size: int = 8192
    compression_enabled: bool = True
    encryption_enabled: bool = False

    # Topology settings
    enable_auto_discovery: bool = True
    discovery_port: int = 8081
    discovery_interval: float = 10.0

    # Quality of Service
    enable_qos: bool = True
    priority_levels: int = 3
    bandwidth_limit_mbps: Optional[float] = None

    def __post_init__(self):
        """Validate configuration."""
        if self.port <= 0 or self.port > 65535:
            raise ValueError("Port must be between 1 and 65535")

        if self.max_connections <= 0:
            raise ValueError("Max connections must be positive")


@dataclass
class NetworkMessage:
    """Network message structure."""

    message_id: str
    message_type: MessageType
    sender_id: str
    recipient_id: Optional[str]
    payload: Dict[str, Any]
    timestamp: float
    priority: int = 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "message_id": self.message_id,
            "message_type": self.message_type.value,
            "sender_id": self.sender_id,
            "recipient_id": self.recipient_id,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "priority": self.priority,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NetworkMessage":
        """Create from dictionary."""
        return cls(
            message_id=data["message_id"],
            message_type=MessageType(data["message_type"]),
            sender_id=data["sender_id"],
            recipient_id=data.get("recipient_id"),
            payload=data["payload"],
            timestamp=data["timestamp"],
            priority=data.get("priority", 1),
        )


@dataclass
class NetworkConnection:
    """Network connection information."""

    connection_id: str
    node_id: str
    host: str
    port: int
    protocol: NetworkProtocol
    connected_at: float
    last_heartbeat: float
    status: str = "connected"
    latency: float = 0.0
    bandwidth_usage: float = 0.0

    @property
    def is_alive(self) -> bool:
        """Check if connection is alive."""
        return (
            self.status == "connected"
            and time.time() - self.last_heartbeat < 30.0  # 30 second timeout
        )


class NetworkCoordinator:
    """Network coordination system for distributed MARL.

    Manages network-level coordination including topology management,
    message routing, and communication optimization.
    """

    def __init__(self, config: NetworkConfig, node_id: str):
        """
        Initialize network coordinator.

        Args:
            config: Network configuration
            node_id: Unique identifier for this node
        """
        self.config = config
        self.node_id = node_id
        self.logger = get_logger(__name__)

        # Network state
        self.is_active = False
        self.server_socket = None
        self.connections: Dict[str, NetworkConnection] = {}
        self.message_handlers: Dict[MessageType, List[callable]] = {
            msg_type: [] for msg_type in MessageType
        }

        # Message management
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.pending_messages: Dict[str, NetworkMessage] = {}
        self.message_history: List[NetworkMessage] = []

        # Network topology
        self.network_topology: Dict[str, List[str]] = {}
        self.discovered_nodes: Set[str] = set()

        # Threading
        self.server_thread = None
        self.heartbeat_thread = None
        self.discovery_thread = None
        self.message_processor_thread = None
        self.network_lock = threading.RLock()

        # Metrics
        self.network_metrics = {
            "total_messages_sent": 0,
            "total_messages_received": 0,
            "failed_messages": 0,
            "average_latency": 0.0,
            "bandwidth_usage": 0.0,
            "active_connections": 0,
            "topology_updates": 0,
        }

        self.logger.info("Network coordinator initialized (node: %s)", self.node_id)

    async def start_network_coordination(self) -> None:
        """Start network coordination services."""
        if self.is_active:
            self.logger.warning("Network coordination already active")
            return

        try:
            self.logger.info("Starting network coordination services")

            # Start server
            await self._start_network_server()

            # Start message processing
            await self._start_message_processing()

            # Start heartbeat service
            await self._start_heartbeat_service()

            # Start node discovery
            if self.config.enable_auto_discovery:
                await self._start_node_discovery()

            self.is_active = True
            self.logger.info("Network coordination services started")

        except Exception as e:
            self.logger.error("Failed to start network coordination: %s", str(e))
            raise

    async def _start_network_server(self) -> None:
        """Start network server for incoming connections."""

        def server_worker():
            try:
                # Create server socket
                self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.server_socket.bind((self.config.host, self.config.port))
                self.server_socket.listen(self.config.max_connections)

                self.logger.info(
                    "Network server listening on %s:%d",
                    self.config.host,
                    self.config.port,
                )

                while self.is_active:
                    try:
                        # Accept connections
                        client_socket, client_address = self.server_socket.accept()

                        # Handle connection in separate thread
                        connection_thread = threading.Thread(
                            target=self._handle_connection,
                            args=(client_socket, client_address),
                            daemon=True,
                        )
                        connection_thread.start()

                    except Exception as e:
                        if self.is_active:
                            self.logger.error("Server accept error: %s", str(e))

            except Exception as e:
                self.logger.error("Network server error: %s", str(e))

        self.server_thread = threading.Thread(target=server_worker, daemon=True)
        self.server_thread.start()

        self.logger.debug("Network server started")

    def _handle_connection(
        self, client_socket: socket.socket, client_address: Tuple[str, int]
    ) -> None:
        """Handle incoming network connection."""
        connection_id = str(uuid.uuid4())

        try:
            self.logger.info(
                "New connection from %s:%d", client_address[0], client_address[1]
            )

            # Create connection record
            connection = NetworkConnection(
                connection_id=connection_id,
                node_id="",  # Will be set during handshake
                host=client_address[0],
                port=client_address[1],
                protocol=self.config.protocol,
                connected_at=time.time(),
                last_heartbeat=time.time(),
            )

            # Perform handshake
            node_id = self._perform_handshake(client_socket)
            if node_id:
                connection.node_id = node_id

                with self.network_lock:
                    self.connections[connection_id] = connection

                # Handle messages from this connection
                self._handle_connection_messages(client_socket, connection)

        except Exception as e:
            self.logger.error("Connection handling error: %s", str(e))
        finally:
            try:
                client_socket.close()
            except Exception:
                pass

            # Remove connection
            with self.network_lock:
                if connection_id in self.connections:
                    del self.connections[connection_id]

    def _perform_handshake(self, client_socket: socket.socket) -> Optional[str]:
        """Perform connection handshake."""
        try:
            # Send handshake request
            handshake_request = {
                "type": "handshake_request",
                "node_id": self.node_id,
                "timestamp": time.time(),
            }

            message_data = json.dumps(handshake_request).encode("utf-8")
            client_socket.send(len(message_data).to_bytes(4, byteorder="big"))
            client_socket.send(message_data)

            # Receive handshake response
            response_length = int.from_bytes(client_socket.recv(4), byteorder="big")
            response_data = client_socket.recv(response_length)
            handshake_response = json.loads(response_data.decode("utf-8"))

            if handshake_response.get("type") == "handshake_response":
                return handshake_response.get("node_id")

            return None

        except Exception as e:
            self.logger.error("Handshake error: %s", str(e))
            return None

    def _handle_connection_messages(
        self, client_socket: socket.socket, connection: NetworkConnection
    ) -> None:
        """Handle messages from a connection."""
        try:
            while self.is_active and connection.is_alive:
                try:
                    # Receive message length
                    length_data = client_socket.recv(4)
                    if not length_data:
                        break

                    message_length = int.from_bytes(length_data, byteorder="big")

                    # Receive message data
                    message_data = client_socket.recv(message_length)
                    if not message_data:
                        break

                    # Parse message
                    message_dict = json.loads(message_data.decode("utf-8"))
                    message = NetworkMessage.from_dict(message_dict)

                    # Update connection heartbeat
                    connection.last_heartbeat = time.time()

                    # Process message
                    await self._process_received_message(message, connection)

                except Exception as e:
                    self.logger.error("Message handling error: %s", str(e))
                    break

        except Exception as e:
            self.logger.error("Connection message handling error: %s", str(e))

    async def _start_message_processing(self) -> None:
        """Start message processing service."""

        async def message_processor():
            while self.is_active:
                try:
                    # Process message queue
                    try:
                        message = await asyncio.wait_for(
                            self.message_queue.get(), timeout=1.0
                        )
                        await self._route_message(message)
                    except asyncio.TimeoutError:
                        continue

                except Exception as e:
                    self.logger.error("Message processing error: %s", str(e))

        # Start message processor task
        asyncio.create_task(message_processor())

        self.logger.debug("Message processing started")

    async def _start_heartbeat_service(self) -> None:
        """Start heartbeat service."""

        def heartbeat_worker():
            while self.is_active:
                try:
                    # Send heartbeats to all connections
                    self._send_heartbeats()

                    # Check for dead connections
                    self._cleanup_dead_connections()

                    time.sleep(self.config.heartbeat_interval)

                except Exception as e:
                    self.logger.error("Heartbeat service error: %s", str(e))

        self.heartbeat_thread = threading.Thread(target=heartbeat_worker, daemon=True)
        self.heartbeat_thread.start()

        self.logger.debug("Heartbeat service started")

    async def _start_node_discovery(self) -> None:
        """Start node discovery service."""

        def discovery_worker():
            while self.is_active:
                try:
                    # Broadcast discovery message
                    self._broadcast_discovery()

                    time.sleep(self.config.discovery_interval)

                except Exception as e:
                    self.logger.error("Node discovery error: %s", str(e))

        self.discovery_thread = threading.Thread(target=discovery_worker, daemon=True)
        self.discovery_thread.start()

        self.logger.debug("Node discovery started")

    def _send_heartbeats(self) -> None:
        """Send heartbeat messages to all connections."""
        heartbeat_message = NetworkMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.HEARTBEAT,
            sender_id=self.node_id,
            recipient_id=None,
            payload={"timestamp": time.time(), "status": "active"},
            timestamp=time.time(),
        )

        with self.network_lock:
            for connection in self.connections.values():
                try:
                    self._send_message_to_connection(heartbeat_message, connection)
                except Exception as e:
                    self.logger.error(
                        "Failed to send heartbeat to %s: %s", connection.node_id, str(e)
                    )

    def _cleanup_dead_connections(self) -> None:
        """Clean up dead connections."""
        with self.network_lock:
            dead_connections = [
                conn_id
                for conn_id, conn in self.connections.items()
                if not conn.is_alive
            ]

            for conn_id in dead_connections:
                connection = self.connections[conn_id]
                self.logger.info("Removing dead connection: %s", connection.node_id)
                del self.connections[conn_id]

    def _broadcast_discovery(self) -> None:
        """Broadcast node discovery message."""
        try:
            # Create discovery message
            discovery_message = {
                "type": "node_discovery",
                "node_id": self.node_id,
                "host": self.config.host,
                "port": self.config.port,
                "timestamp": time.time(),
            }

            # Broadcast on discovery port
            discovery_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            discovery_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

            message_data = json.dumps(discovery_message).encode("utf-8")
            discovery_socket.sendto(
                message_data, ("<broadcast>", self.config.discovery_port)
            )
            discovery_socket.close()

        except Exception as e:
            self.logger.error("Discovery broadcast error: %s", str(e))

    async def _process_received_message(
        self, message: NetworkMessage, connection: NetworkConnection
    ) -> None:
        """Process received message."""
        try:
            self.network_metrics["total_messages_received"] += 1

            # Add to message history
            self.message_history.append(message)
            if len(self.message_history) > 1000:  # Limit history size
                self.message_history.pop(0)

            # Handle message based on type
            handlers = self.message_handlers.get(message.message_type, [])
            for handler in handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(message, connection)
                    else:
                        handler(message, connection)
                except Exception as e:
                    self.logger.error("Message handler error: %s", str(e))

            # Update latency metrics
            if message.message_type == MessageType.HEARTBEAT:
                latency = time.time() - message.timestamp
                connection.latency = latency
                self._update_average_latency(latency)

        except Exception as e:
            self.logger.error("Message processing error: %s", str(e))

    async def _route_message(self, message: NetworkMessage) -> None:
        """Route message to appropriate destination."""
        try:
            if message.recipient_id:
                # Send to specific recipient
                await self._send_message_to_node(message, message.recipient_id)
            else:
                # Broadcast to all connections
                await self._broadcast_message(message)

            self.network_metrics["total_messages_sent"] += 1

        except Exception as e:
            self.logger.error("Message routing error: %s", str(e))
            self.network_metrics["failed_messages"] += 1

    async def _send_message_to_node(
        self, message: NetworkMessage, node_id: str
    ) -> None:
        """Send message to specific node."""
        with self.network_lock:
            # Find connection for node
            target_connection = None
            for connection in self.connections.values():
                if connection.node_id == node_id:
                    target_connection = connection
                    break

            if target_connection:
                self._send_message_to_connection(message, target_connection)
            else:
                raise ConnectionError(f"No connection to node {node_id}")

    def _send_message_to_connection(
        self, message: NetworkMessage, connection: NetworkConnection
    ) -> None:
        """Send message to specific connection."""
        try:
            # This would typically send the message over the network
            # For now, we'll simulate it
            self.logger.debug(
                "Sending message %s to %s",
                message.message_type.value,
                connection.node_id,
            )

        except Exception as e:
            self.logger.error("Failed to send message to connection: %s", str(e))
            raise

    async def _broadcast_message(self, message: NetworkMessage) -> None:
        """Broadcast message to all connections."""
        with self.network_lock:
            for connection in self.connections.values():
                try:
                    self._send_message_to_connection(message, connection)
                except Exception as e:
                    self.logger.error(
                        "Failed to broadcast to %s: %s", connection.node_id, str(e)
                    )

    def _update_average_latency(self, latency: float) -> None:
        """Update average latency metric."""
        current_avg = self.network_metrics["average_latency"]
        total_messages = self.network_metrics["total_messages_received"]

        if total_messages > 0:
            self.network_metrics["average_latency"] = (
                current_avg * (total_messages - 1) + latency
            ) / total_messages

    async def send_message(
        self,
        message_type: MessageType,
        payload: Dict[str, Any],
        recipient_id: Optional[str] = None,
        priority: int = 1,
    ) -> str:
        """Send message through network.

        Args:
            message_type: Type of message
            payload: Message payload
            recipient_id: Specific recipient (None for broadcast)
            priority: Message priority

        Returns:
            Message ID
        """
        message = NetworkMessage(
            message_id=str(uuid.uuid4()),
            message_type=message_type,
            sender_id=self.node_id,
            recipient_id=recipient_id,
            payload=payload,
            timestamp=time.time(),
            priority=priority,
        )

        # Add to message queue
        await self.message_queue.put(message)

        return message.message_id

    async def connect_to_node(self, host: str, port: int) -> bool:
        """Connect to another node.

        Args:
            host: Target host
            port: Target port

        Returns:
            True if successful, False otherwise
        """
        try:
            # Create connection
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.connect((host, port))

            # Perform handshake
            node_id = self._perform_client_handshake(client_socket)
            if node_id:
                # Create connection record
                connection = NetworkConnection(
                    connection_id=str(uuid.uuid4()),
                    node_id=node_id,
                    host=host,
                    port=port,
                    protocol=self.config.protocol,
                    connected_at=time.time(),
                    last_heartbeat=time.time(),
                )

                with self.network_lock:
                    self.connections[connection.connection_id] = connection

                # Start handling messages from this connection
                connection_thread = threading.Thread(
                    target=self._handle_connection_messages,
                    args=(client_socket, connection),
                    daemon=True,
                )
                connection_thread.start()

                self.logger.info("Connected to node %s at %s:%d", node_id, host, port)
                return True

            return False

        except Exception as e:
            self.logger.error("Failed to connect to %s:%d: %s", host, port, str(e))
            return False

    def _perform_client_handshake(self, server_socket: socket.socket) -> Optional[str]:
        """Perform client-side handshake."""
        try:
            # Receive handshake request
            request_length = int.from_bytes(server_socket.recv(4), byteorder="big")
            request_data = server_socket.recv(request_length)
            handshake_request = json.loads(request_data.decode("utf-8"))

            # Send handshake response
            handshake_response = {
                "type": "handshake_response",
                "node_id": self.node_id,
                "timestamp": time.time(),
            }

            response_data = json.dumps(handshake_response).encode("utf-8")
            server_socket.send(len(response_data).to_bytes(4, byteorder="big"))
            server_socket.send(response_data)

            return handshake_request.get("node_id")

        except Exception as e:
            self.logger.error("Client handshake error: %s", str(e))
            return None

    def add_message_handler(self, message_type: MessageType, handler: callable) -> None:
        """Add message handler for specific message type."""
        self.message_handlers[message_type].append(handler)

    def remove_message_handler(
        self, message_type: MessageType, handler: callable
    ) -> None:
        """Remove message handler."""
        if handler in self.message_handlers[message_type]:
            self.message_handlers[message_type].remove(handler)

    def get_network_status(self) -> Dict[str, Any]:
        """Get network status information."""
        with self.network_lock:
            return {
                "node_id": self.node_id,
                "is_active": self.is_active,
                "server_address": f"{self.config.host}:{self.config.port}",
                "active_connections": len(self.connections),
                "connected_nodes": [conn.node_id for conn in self.connections.values()],
                "network_topology": self.network_topology,
                "discovered_nodes": list(self.discovered_nodes),
            }

    def get_network_metrics(self) -> Dict[str, Any]:
        """Get network metrics."""
        with self.network_lock:
            self.network_metrics["active_connections"] = len(self.connections)

            # Calculate bandwidth usage
            total_bandwidth = sum(
                conn.bandwidth_usage for conn in self.connections.values()
            )
            self.network_metrics["bandwidth_usage"] = total_bandwidth

            return self.network_metrics.copy()

    def get_connection_info(self) -> List[Dict[str, Any]]:
        """Get information about all connections."""
        with self.network_lock:
            return [
                {
                    "connection_id": conn.connection_id,
                    "node_id": conn.node_id,
                    "host": conn.host,
                    "port": conn.port,
                    "protocol": conn.protocol.value,
                    "connected_at": conn.connected_at,
                    "last_heartbeat": conn.last_heartbeat,
                    "status": conn.status,
                    "latency": conn.latency,
                    "bandwidth_usage": conn.bandwidth_usage,
                    "is_alive": conn.is_alive,
                }
                for conn in self.connections.values()
            ]

    async def shutdown(self) -> None:
        """Shutdown network coordinator."""
        self.logger.info("Shutting down network coordinator")

        self.is_active = False

        # Close server socket
        if self.server_socket:
            try:
                self.server_socket.close()
            except Exception:
                pass

        # Stop threads
        if self.server_thread and self.server_thread.is_alive():
            self.server_thread.join(timeout=5.0)

        if self.heartbeat_thread and self.heartbeat_thread.is_alive():
            self.heartbeat_thread.join(timeout=5.0)

        if self.discovery_thread and self.discovery_thread.is_alive():
            self.discovery_thread.join(timeout=5.0)

        # Clear connections
        with self.network_lock:
            self.connections.clear()
            self.discovered_nodes.clear()

        # Clear message handlers
        for handlers in self.message_handlers.values():
            handlers.clear()

        self.logger.info("Network coordinator shutdown complete")
