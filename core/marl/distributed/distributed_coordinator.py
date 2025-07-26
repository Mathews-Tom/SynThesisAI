"""Distributed MARL Coordination System.

This module provides distributed coordination capabilities for multi-agent
reinforcement learning across network boundaries and multiple nodes.
"""

import asyncio
import time
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import threading
from pathlib import Path
import uuid

from utils.logging_config import get_logger
from core.marl.coordination.coordination_policy import CoordinationPolicy
from core.marl.coordination.consensus_mechanism import ConsensusMechanism
from core.marl.coordination.communication_protocol import AgentCommunicationProtocol


class DistributedCoordinationMode(Enum):
    """Distributed coordination mode enumeration."""
    CENTRALIZED = "centralized"
    DECENTRALIZED = "decentralized"
    HIERARCHICAL = "hierarchical"
    FEDERATED = "federated"


class NetworkTopology(Enum):
    """Network topology for distributed coordination."""
    STAR = "star"
    RING = "ring"
    MESH = "mesh"
    TREE = "tree"
    HYBRID = "hybrid"


@dataclass
class DistributedCoordinationConfig:
    """Configuration for distributed MARL coordination."""
    
    # Coordination mode
    coordination_mode: DistributedCoordinationMode = DistributedCoordinationMode.CENTRALIZED
    network_topology: NetworkTopology = NetworkTopology.STAR
    
    # Network settings
    node_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    master_node_id: Optional[str] = None
    network_port: int = 8080
    heartbeat_interval: float = 5.0
    connection_timeout: float = 30.0
    
    # Coordination parameters
    consensus_threshold: float = 0.7
    coordination_timeout: float = 60.0
    max_coordination_rounds: int = 10
    conflict_resolution_strategy: str = "majority_vote"
    
    # Performance settings
    batch_coordination: bool = True
    batch_size: int = 32
    coordination_cache_size: int = 1000
    enable_compression: bool = True
    
    # Fault tolerance
    enable_fault_tolerance: bool = True
    max_node_failures: int = 2
    recovery_timeout: float = 120.0
    backup_coordination_nodes: List[str] = field(default_factory=list)
    
    # Security
    enable_encryption: bool = True
    authentication_required: bool = True
    security_token: Optional[str] = None
    
    def __post_init__(self):
        """Validate configuration."""
        if self.consensus_threshold <= 0 or self.consensus_threshold > 1:
            raise ValueError("Consensus threshold must be between 0 and 1")
        
        if self.coordination_timeout <= 0:
            raise ValueError("Coordination timeout must be positive")
        
        if self.max_coordination_rounds <= 0:
            raise ValueError("Max coordination rounds must be positive")


class DistributedCoordinator:
    """Distributed coordination system for MARL agents.
    
    Provides distributed coordination capabilities across network boundaries,
    supporting various coordination modes and network topologies.
    """
    
    def __init__(self, config: DistributedCoordinationConfig):
        """
        Initialize distributed coordinator.
        
        Args:
            config: Distributed coordination configuration
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # Coordination state
        self.is_initialized = False
        self.is_active = False
        self.is_master_node = False
        
        # Network state
        self.connected_nodes: Dict[str, Dict[str, Any]] = {}
        self.node_agents: Dict[str, List[str]] = {}
        self.network_topology_map: Dict[str, List[str]] = {}
        
        # Coordination components
        self.coordination_policy = None
        self.consensus_mechanism = None
        self.communication_protocol = None
        
        # Coordination tracking
        self.active_coordinations: Dict[str, Dict[str, Any]] = {}
        self.coordination_history: List[Dict[str, Any]] = []
        self.coordination_metrics = {
            "total_coordinations": 0,
            "successful_coordinations": 0,
            "failed_coordinations": 0,
            "average_coordination_time": 0.0,
            "network_latency": 0.0,
            "node_failures": 0
        }
        
        # Threading and synchronization
        self.coordination_lock = threading.RLock()
        self.network_lock = threading.RLock()
        self.heartbeat_thread = None
        self.coordination_thread = None
        
        # Event handlers
        self.node_join_handlers: List[callable] = []
        self.node_leave_handlers: List[callable] = []
        self.coordination_handlers: List[callable] = []
        
        self.logger.info("Distributed coordinator initialized (node: %s)", self.config.node_id)
    
    async def initialize_distributed_coordination(self) -> None:
        """Initialize distributed coordination system."""
        if self.is_initialized:
            self.logger.warning("Distributed coordination already initialized")
            return
        
        try:
            self.logger.info("Initializing distributed coordination system")
            
            # Initialize coordination components
            await self._initialize_coordination_components()
            
            # Set up network topology
            await self._setup_network_topology()
            
            # Start network services
            await self._start_network_services()
            
            # Determine master node
            await self._determine_master_node()
            
            # Start coordination services
            await self._start_coordination_services()
            
            self.is_initialized = True
            self.is_active = True
            
            self.logger.info("Distributed coordination initialization complete")
            
        except Exception as e:
            self.logger.error("Failed to initialize distributed coordination: %s", str(e))
            raise
    
    async def _initialize_coordination_components(self) -> None:
        """Initialize coordination components."""
        # Initialize coordination policy
        self.coordination_policy = CoordinationPolicy()        
        # Initialize consensus mechanism
        self.consensus_mechanism = ConsensusMechanism()        
        # Initialize communication protocol
        self.communication_protocol = AgentCommunicationProtocol()
        
        self.logger.debug("Coordination components initialized")
    
    async def _setup_network_topology(self) -> None:
        """Set up network topology based on configuration."""
        try:
            if self.config.network_topology == NetworkTopology.STAR:
                await self._setup_star_topology()
            elif self.config.network_topology == NetworkTopology.RING:
                await self._setup_ring_topology()
            elif self.config.network_topology == NetworkTopology.MESH:
                await self._setup_mesh_topology()
            elif self.config.network_topology == NetworkTopology.TREE:
                await self._setup_tree_topology()
            else:
                await self._setup_hybrid_topology()
            
            self.logger.info("Network topology set up: %s", self.config.network_topology.value)
            
        except Exception as e:
            self.logger.error("Failed to set up network topology: %s", str(e))
            raise
    
    async def _setup_star_topology(self) -> None:
        """Set up star network topology."""
        # In star topology, all nodes connect to master node
        if self.config.master_node_id:
            self.network_topology_map[self.config.node_id] = [self.config.master_node_id]
        else:
            # This node becomes master
            self.network_topology_map[self.config.node_id] = []
            self.is_master_node = True
    
    async def _setup_ring_topology(self) -> None:
        """Set up ring network topology."""
        # In ring topology, each node connects to next node in ring
        # This is a simplified implementation
        self.network_topology_map[self.config.node_id] = []
    
    async def _setup_mesh_topology(self) -> None:
        """Set up mesh network topology."""
        # In mesh topology, all nodes connect to all other nodes
        self.network_topology_map[self.config.node_id] = []
    
    async def _setup_tree_topology(self) -> None:
        """Set up tree network topology."""
        # In tree topology, nodes form hierarchical structure
        self.network_topology_map[self.config.node_id] = []
    
    async def _setup_hybrid_topology(self) -> None:
        """Set up hybrid network topology."""
        # Hybrid topology combines multiple topologies
        self.network_topology_map[self.config.node_id] = []
    
    async def _start_network_services(self) -> None:
        """Start network services for distributed coordination."""
        try:
            # Start heartbeat service
            await self._start_heartbeat_service()
            
            # Start network listener
            await self._start_network_listener()
            
            # Connect to other nodes
            await self._connect_to_nodes()
            
            self.logger.info("Network services started")
            
        except Exception as e:
            self.logger.error("Failed to start network services: %s", str(e))
            raise
    
    async def _start_heartbeat_service(self) -> None:
        """Start heartbeat service for node health monitoring."""
        def heartbeat_worker():
            while self.is_active:
                try:
                    # Send heartbeat to connected nodes
                    self._send_heartbeat()
                    time.sleep(self.config.heartbeat_interval)
                except Exception as e:
                    self.logger.error("Heartbeat error: %s", str(e))
        
        self.heartbeat_thread = threading.Thread(target=heartbeat_worker, daemon=True)
        self.heartbeat_thread.start()
        
        self.logger.debug("Heartbeat service started")
    
    async def _start_network_listener(self) -> None:
        """Start network listener for incoming connections."""
        # This would typically start a network server
        # For now, we'll simulate it
        self.logger.debug("Network listener started on port %d", self.config.network_port)
    
    async def _connect_to_nodes(self) -> None:
        """Connect to other nodes in the network."""
        # Connect based on network topology
        target_nodes = self.network_topology_map.get(self.config.node_id, [])
        
        for node_id in target_nodes:
            try:
                await self._connect_to_node(node_id)
            except Exception as e:
                self.logger.error("Failed to connect to node %s: %s", node_id, str(e))
    
    async def _connect_to_node(self, node_id: str) -> None:
        """Connect to a specific node."""
        # Simulate connection
        connection_info = {
            "node_id": node_id,
            "connected_at": time.time(),
            "status": "connected",
            "latency": 0.0,
            "agents": []
        }
        
        with self.network_lock:
            self.connected_nodes[node_id] = connection_info
            self.node_agents[node_id] = []
        
        # Notify handlers
        await self._notify_node_join_handlers(node_id)
        
        self.logger.info("Connected to node: %s", node_id)
    
    async def _determine_master_node(self) -> None:
        """Determine which node should be the master."""
        if self.config.coordination_mode == DistributedCoordinationMode.CENTRALIZED:
            if self.config.master_node_id:
                self.is_master_node = (self.config.node_id == self.config.master_node_id)
            else:
                # Elect master based on node ID (lexicographic order)
                all_nodes = [self.config.node_id] + list(self.connected_nodes.keys())
                master_node = min(all_nodes)
                self.is_master_node = (self.config.node_id == master_node)
        else:
            # In decentralized modes, no single master
            self.is_master_node = False
        
        self.logger.info("Master node determination: is_master=%s", self.is_master_node)
    
    async def _start_coordination_services(self) -> None:
        """Start coordination services."""
        def coordination_worker():
            while self.is_active:
                try:
                    # Process pending coordinations
                    self._process_pending_coordinations()
                    time.sleep(0.1)  # Small delay to prevent busy waiting
                except Exception as e:
                    self.logger.error("Coordination worker error: %s", str(e))
        
        self.coordination_thread = threading.Thread(target=coordination_worker, daemon=True)
        self.coordination_thread.start()
        
        self.logger.debug("Coordination services started")
    
    def _send_heartbeat(self) -> None:
        """Send heartbeat to connected nodes."""
        heartbeat_message = {
            "type": "heartbeat",
            "node_id": self.config.node_id,
            "timestamp": time.time(),
            "status": "active",
            "agents": list(self.node_agents.get(self.config.node_id, []))
        }
        
        # Send to all connected nodes
        for node_id in self.connected_nodes:
            try:
                # Simulate sending heartbeat
                self._update_node_latency(node_id)
            except Exception as e:
                self.logger.error("Failed to send heartbeat to %s: %s", node_id, str(e))
    
    def _update_node_latency(self, node_id: str) -> None:
        """Update network latency for a node."""
        if node_id in self.connected_nodes:
            # Simulate latency measurement
            import random
            latency = random.uniform(0.001, 0.1)  # 1-100ms
            self.connected_nodes[node_id]["latency"] = latency
            
            # Update average network latency
            total_latency = sum(node["latency"] for node in self.connected_nodes.values())
            self.coordination_metrics["network_latency"] = total_latency / len(self.connected_nodes)
    
    def _process_pending_coordinations(self) -> None:
        """Process pending coordination requests."""
        with self.coordination_lock:
            # Process active coordinations
            completed_coordinations = []
            
            for coord_id, coordination in self.active_coordinations.items():
                if self._is_coordination_complete(coordination):
                    completed_coordinations.append(coord_id)
                elif self._is_coordination_timeout(coordination):
                    self._handle_coordination_timeout(coordination)
                    completed_coordinations.append(coord_id)
            
            # Remove completed coordinations
            for coord_id in completed_coordinations:
                del self.active_coordinations[coord_id]
    
    def _is_coordination_complete(self, coordination: Dict[str, Any]) -> bool:
        """Check if coordination is complete."""
        required_responses = coordination.get("required_responses", 0)
        received_responses = len(coordination.get("responses", []))
        return received_responses >= required_responses
    
    def _is_coordination_timeout(self, coordination: Dict[str, Any]) -> bool:
        """Check if coordination has timed out."""
        start_time = coordination.get("start_time", 0)
        return time.time() - start_time > self.config.coordination_timeout
    
    def _handle_coordination_timeout(self, coordination: Dict[str, Any]) -> None:
        """Handle coordination timeout."""
        coord_id = coordination.get("coordination_id")
        self.logger.warning("Coordination timeout: %s", coord_id)
        
        # Update metrics
        self.coordination_metrics["failed_coordinations"] += 1
        
        # Add to history
        coordination["status"] = "timeout"
        coordination["end_time"] = time.time()
        self.coordination_history.append(coordination.copy())
    
    async def coordinate_distributed_action(
        self,
        action_request: Dict[str, Any],
        participating_nodes: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Coordinate action across distributed nodes.
        
        Args:
            action_request: Action request to coordinate
            participating_nodes: Specific nodes to include in coordination
            
        Returns:
            Coordination result
        """
        if not self.is_active:
            raise RuntimeError("Distributed coordinator not active")
        
        coordination_id = str(uuid.uuid4())
        start_time = time.time()
        
        self.logger.info("Starting distributed coordination: %s", coordination_id)
        
        try:
            # Determine participating nodes
            if participating_nodes is None:
                participating_nodes = list(self.connected_nodes.keys())
            
            # Create coordination record
            coordination = {
                "coordination_id": coordination_id,
                "action_request": action_request,
                "participating_nodes": participating_nodes,
                "start_time": start_time,
                "status": "active",
                "responses": [],
                "required_responses": len(participating_nodes)
            }
            
            with self.coordination_lock:
                self.active_coordinations[coordination_id] = coordination
            
            # Send coordination request to nodes
            await self._send_coordination_request(coordination)
            
            # Wait for responses or timeout
            result = await self._wait_for_coordination_completion(coordination_id)
            
            # Process coordination result
            coordination_result = await self._process_coordination_result(result)
            
            # Update metrics
            self._update_coordination_metrics(coordination_result, start_time)
            
            self.logger.info("Distributed coordination complete: %s", coordination_id)
            
            return coordination_result
            
        except Exception as e:
            self.logger.error("Distributed coordination failed: %s", str(e))
            
            # Update failure metrics
            self.coordination_metrics["failed_coordinations"] += 1
            
            raise
    
    async def _send_coordination_request(self, coordination: Dict[str, Any]) -> None:
        """Send coordination request to participating nodes."""
        request_message = {
            "type": "coordination_request",
            "coordination_id": coordination["coordination_id"],
            "action_request": coordination["action_request"],
            "sender_node": self.config.node_id,
            "timestamp": time.time()
        }
        
        for node_id in coordination["participating_nodes"]:
            try:
                await self._send_message_to_node(node_id, request_message)
            except Exception as e:
                self.logger.error("Failed to send coordination request to %s: %s", node_id, str(e))
    
    async def _send_message_to_node(self, node_id: str, message: Dict[str, Any]) -> None:
        """Send message to a specific node."""
        # Simulate message sending
        if node_id in self.connected_nodes:
            # Add some network delay
            await asyncio.sleep(0.01)
            self.logger.debug("Message sent to %s: %s", node_id, message["type"])
        else:
            raise ConnectionError(f"Node {node_id} not connected")
    
    async def _wait_for_coordination_completion(self, coordination_id: str) -> Dict[str, Any]:
        """Wait for coordination to complete."""
        timeout = self.config.coordination_timeout
        check_interval = 0.1
        elapsed_time = 0
        
        while elapsed_time < timeout:
            with self.coordination_lock:
                if coordination_id not in self.active_coordinations:
                    # Coordination completed or removed
                    break
                
                coordination = self.active_coordinations[coordination_id]
                if self._is_coordination_complete(coordination):
                    return coordination
            
            await asyncio.sleep(check_interval)
            elapsed_time += check_interval
        
        # Timeout occurred
        raise TimeoutError(f"Coordination {coordination_id} timed out")
    
    async def _process_coordination_result(self, coordination: Dict[str, Any]) -> Dict[str, Any]:
        """Process coordination result and generate final action."""
        responses = coordination.get("responses", [])
        
        if not responses:
            return {
                "coordination_id": coordination["coordination_id"],
                "status": "failed",
                "reason": "No responses received",
                "coordinated_action": None
            }
        
        try:
            # Use consensus mechanism to determine final action
            consensus_result = await self.consensus_mechanism.reach_consensus(responses)
            
            coordinated_action = {
                "coordination_id": coordination["coordination_id"],
                "status": "success",
                "consensus_result": consensus_result,
                "coordinated_action": consensus_result.get("action"),
                "participating_nodes": coordination["participating_nodes"],
                "response_count": len(responses)
            }
            
            return coordinated_action
            
        except Exception as e:
            self.logger.error("Failed to process coordination result: %s", str(e))
            return {
                "coordination_id": coordination["coordination_id"],
                "status": "failed",
                "reason": f"Processing error: {str(e)}",
                "coordinated_action": None
            }
    
    def _update_coordination_metrics(self, result: Dict[str, Any], start_time: float) -> None:
        """Update coordination metrics."""
        coordination_time = time.time() - start_time
        
        self.coordination_metrics["total_coordinations"] += 1
        
        if result["status"] == "success":
            self.coordination_metrics["successful_coordinations"] += 1
        else:
            self.coordination_metrics["failed_coordinations"] += 1
        
        # Update average coordination time
        total_coords = self.coordination_metrics["total_coordinations"]
        current_avg = self.coordination_metrics["average_coordination_time"]
        self.coordination_metrics["average_coordination_time"] = (
            (current_avg * (total_coords - 1) + coordination_time) / total_coords
        )
    
    async def handle_coordination_request(
        self,
        request: Dict[str, Any],
        sender_node: str
    ) -> Dict[str, Any]:
        """Handle incoming coordination request from another node.
        
        Args:
            request: Coordination request
            sender_node: Node that sent the request
            
        Returns:
            Coordination response
        """
        try:
            coordination_id = request.get("coordination_id")
            action_request = request.get("action_request")
            
            self.logger.info("Handling coordination request: %s from %s", coordination_id, sender_node)
            
            # Process action request locally
            local_response = await self._process_local_coordination(action_request)
            
            # Create response
            response = {
                "coordination_id": coordination_id,
                "node_id": self.config.node_id,
                "response": local_response,
                "timestamp": time.time()
            }
            
            # Send response back to sender
            await self._send_coordination_response(sender_node, response)
            
            return response
            
        except Exception as e:
            self.logger.error("Failed to handle coordination request: %s", str(e))
            raise
    
    async def _process_local_coordination(self, action_request: Dict[str, Any]) -> Dict[str, Any]:
        """Process coordination request locally."""
        # This would typically involve:
        # 1. Evaluating the action request
        # 2. Consulting local agents
        # 3. Generating local response
        
        # Simulate local processing
        import random
        
        local_response = {
            "action": action_request.get("action", "default"),
            "confidence": random.uniform(0.5, 1.0),
            "local_state": {"status": "ready"},
            "processing_time": random.uniform(0.01, 0.1)
        }
        
        return local_response
    
    async def _send_coordination_response(
        self,
        target_node: str,
        response: Dict[str, Any]
    ) -> None:
        """Send coordination response to target node."""
        response_message = {
            "type": "coordination_response",
            "response": response,
            "sender_node": self.config.node_id,
            "timestamp": time.time()
        }
        
        await self._send_message_to_node(target_node, response_message)
    
    async def handle_coordination_response(
        self,
        response_message: Dict[str, Any],
        sender_node: str
    ) -> None:
        """Handle incoming coordination response."""
        try:
            response = response_message.get("response")
            coordination_id = response.get("coordination_id")
            
            with self.coordination_lock:
                if coordination_id in self.active_coordinations:
                    coordination = self.active_coordinations[coordination_id]
                    coordination["responses"].append(response)
                    
                    self.logger.debug(
                        "Received coordination response: %s from %s",
                        coordination_id,
                        sender_node
                    )
                else:
                    self.logger.warning(
                        "Received response for unknown coordination: %s",
                        coordination_id
                    )
            
        except Exception as e:
            self.logger.error("Failed to handle coordination response: %s", str(e))
    
    async def handle_node_failure(self, failed_node: str) -> None:
        """Handle node failure in distributed coordination."""
        self.logger.warning("Handling node failure: %s", failed_node)
        
        try:
            # Remove failed node from connected nodes
            with self.network_lock:
                if failed_node in self.connected_nodes:
                    del self.connected_nodes[failed_node]
                if failed_node in self.node_agents:
                    del self.node_agents[failed_node]
            
            # Update coordination requirements
            with self.coordination_lock:
                for coordination in self.active_coordinations.values():
                    if failed_node in coordination["participating_nodes"]:
                        coordination["participating_nodes"].remove(failed_node)
                        coordination["required_responses"] -= 1
            
            # Update metrics
            self.coordination_metrics["node_failures"] += 1
            
            # Notify handlers
            await self._notify_node_leave_handlers(failed_node)
            
            # Trigger recovery if needed
            if self.config.enable_fault_tolerance:
                await self._trigger_fault_recovery(failed_node)
            
        except Exception as e:
            self.logger.error("Failed to handle node failure: %s", str(e))
    
    async def _trigger_fault_recovery(self, failed_node: str) -> None:
        """Trigger fault recovery procedures."""
        # Check if we need to elect new master
        if failed_node == self.config.master_node_id:
            await self._elect_new_master()
        
        # Redistribute agents if needed
        await self._redistribute_agents(failed_node)
        
        # Update network topology
        await self._update_network_topology_after_failure(failed_node)
    
    async def _elect_new_master(self) -> None:
        """Elect new master node after master failure."""
        if self.config.coordination_mode == DistributedCoordinationMode.CENTRALIZED:
            # Elect new master based on node ID
            all_nodes = [self.config.node_id] + list(self.connected_nodes.keys())
            new_master = min(all_nodes)
            
            if new_master == self.config.node_id:
                self.is_master_node = True
                self.logger.info("Elected as new master node")
            else:
                self.is_master_node = False
                self.config.master_node_id = new_master
                self.logger.info("New master node elected: %s", new_master)
    
    async def _redistribute_agents(self, failed_node: str) -> None:
        """Redistribute agents from failed node."""
        # This would typically involve reassigning agents to other nodes
        self.logger.info("Redistributing agents from failed node: %s", failed_node)
    
    async def _update_network_topology_after_failure(self, failed_node: str) -> None:
        """Update network topology after node failure."""
        # Remove failed node from topology map
        for node_id, connections in self.network_topology_map.items():
            if failed_node in connections:
                connections.remove(failed_node)
        
        if failed_node in self.network_topology_map:
            del self.network_topology_map[failed_node]
    
    async def _notify_node_join_handlers(self, node_id: str) -> None:
        """Notify node join handlers."""
        for handler in self.node_join_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(node_id)
                else:
                    handler(node_id)
            except Exception as e:
                self.logger.error("Node join handler error: %s", str(e))
    
    async def _notify_node_leave_handlers(self, node_id: str) -> None:
        """Notify node leave handlers."""
        for handler in self.node_leave_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(node_id)
                else:
                    handler(node_id)
            except Exception as e:
                self.logger.error("Node leave handler error: %s", str(e))
    
    def add_node_join_handler(self, handler: callable) -> None:
        """Add node join event handler."""
        self.node_join_handlers.append(handler)
    
    def add_node_leave_handler(self, handler: callable) -> None:
        """Add node leave event handler."""
        self.node_leave_handlers.append(handler)
    
    def add_coordination_handler(self, handler: callable) -> None:
        """Add coordination event handler."""
        self.coordination_handlers.append(handler)
    
    def get_coordination_metrics(self) -> Dict[str, Any]:
        """Get coordination metrics."""
        return self.coordination_metrics.copy()
    
    def get_network_status(self) -> Dict[str, Any]:
        """Get network status information."""
        return {
            "node_id": self.config.node_id,
            "is_master": self.is_master_node,
            "connected_nodes": len(self.connected_nodes),
            "network_topology": self.config.network_topology.value,
            "coordination_mode": self.config.coordination_mode.value,
            "active_coordinations": len(self.active_coordinations),
            "network_latency": self.coordination_metrics["network_latency"]
        }
    
    def get_distributed_info(self) -> Dict[str, Any]:
        """Get distributed coordination information."""
        return {
            "config": {
                "node_id": self.config.node_id,
                "coordination_mode": self.config.coordination_mode.value,
                "network_topology": self.config.network_topology.value,
                "is_master": self.is_master_node
            },
            "network": {
                "connected_nodes": list(self.connected_nodes.keys()),
                "node_count": len(self.connected_nodes),
                "topology_map": self.network_topology_map
            },
            "coordination": {
                "active_coordinations": len(self.active_coordinations),
                "total_coordinations": self.coordination_metrics["total_coordinations"],
                "success_rate": (
                    self.coordination_metrics["successful_coordinations"] /
                    max(1, self.coordination_metrics["total_coordinations"])
                )
            },
            "status": {
                "is_initialized": self.is_initialized,
                "is_active": self.is_active
            }
        }
    
    async def shutdown(self) -> None:
        """Shutdown distributed coordination system."""
        self.logger.info("Shutting down distributed coordinator")
        
        self.is_active = False
        
        # Stop threads
        if self.heartbeat_thread and self.heartbeat_thread.is_alive():
            self.heartbeat_thread.join(timeout=5.0)
        
        if self.coordination_thread and self.coordination_thread.is_alive():
            self.coordination_thread.join(timeout=5.0)
        
        # Disconnect from nodes
        with self.network_lock:
            self.connected_nodes.clear()
            self.node_agents.clear()
        
        # Clear active coordinations
        with self.coordination_lock:
            self.active_coordinations.clear()
        
        # Clear handlers
        self.node_join_handlers.clear()
        self.node_leave_handlers.clear()
        self.coordination_handlers.clear()
        
        self.is_initialized = False
        
        self.logger.info("Distributed coordinator shutdown complete")