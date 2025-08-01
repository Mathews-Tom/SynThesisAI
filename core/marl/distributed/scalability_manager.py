"""Scalability Management System for Distributed MARL.

This module provides auto-scaling and resource optimization capabilities
for distributed MARL deployment across multiple nodes and environments.
"""

# Standard Library
import asyncio
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

# Third-Party Library
import psutil

# SynThesisAI Modules
from utils.logging_config import get_logger
from .network_coordinator import NetworkCoordinator
from .resource_manager import ResourceManager


class ScalingDirection(Enum):
    """Scaling direction enumeration."""

    UP = "up"
    DOWN = "down"
    STABLE = "stable"


class ScalingTrigger(Enum):
    """Scaling trigger enumeration."""

    CPU_UTILIZATION = "cpu_utilization"
    MEMORY_UTILIZATION = "memory_utilization"
    GPU_UTILIZATION = "gpu_utilization"
    NETWORK_LATENCY = "network_latency"
    COORDINATION_LOAD = "coordination_load"
    AGENT_COUNT = "agent_count"
    CUSTOM_METRIC = "custom_metric"


class DeploymentStrategy(Enum):
    """Deployment strategy enumeration."""

    HORIZONTAL = "horizontal"  # Add more nodes
    VERTICAL = "vertical"  # Add more resources to existing nodes
    HYBRID = "hybrid"  # Combination of both


class NodeStatus(Enum):
    """Node status enumeration."""

    PENDING = "pending"
    STARTING = "starting"
    ACTIVE = "active"
    STOPPING = "stopping"
    FAILED = "failed"
    TERMINATED = "terminated"


@dataclass
class ScalingConfig:
    """Configuration for scalability management."""

    # Scaling settings
    enable_auto_scaling: bool = True
    deployment_strategy: DeploymentStrategy = DeploymentStrategy.HYBRID
    min_nodes: int = 1
    max_nodes: int = 10

    # Scaling thresholds
    scale_up_threshold: float = 0.8
    scale_down_threshold: float = 0.3
    scaling_cooldown: float = 300.0  # 5 minutes

    # Monitoring settings
    monitoring_interval: float = 30.0
    metric_window_size: int = 10
    stability_threshold: float = 0.1

    # Resource limits per node
    max_cpu_per_node: int = 16
    max_memory_per_node_gb: float = 64.0
    max_gpu_per_node: int = 4
    max_agents_per_node: int = 10

    # Performance targets
    target_cpu_utilization: float = 0.7
    target_memory_utilization: float = 0.7
    target_network_latency_ms: float = 100.0
    target_coordination_success_rate: float = 0.95

    # Scaling policies
    scaling_policies: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: {
            "cpu_based": {
                "trigger": ScalingTrigger.CPU_UTILIZATION,
                "scale_up_threshold": 0.8,
                "scale_down_threshold": 0.3,
                "weight": 1.0,
                "enabled": True,
            },
            "memory_based": {
                "trigger": ScalingTrigger.MEMORY_UTILIZATION,
                "scale_up_threshold": 0.8,
                "scale_down_threshold": 0.3,
                "weight": 1.0,
                "enabled": True,
            },
            "coordination_based": {
                "trigger": ScalingTrigger.COORDINATION_LOAD,
                "scale_up_threshold": 0.9,
                "scale_down_threshold": 0.4,
                "weight": 1.5,
                "enabled": True,
            },
            "agent_count_based": {
                "trigger": ScalingTrigger.AGENT_COUNT,
                "scale_up_threshold": 8,  # agents per node
                "scale_down_threshold": 2,
                "weight": 0.8,
                "enabled": True,
            },
        }
    )

    # Node templates
    node_templates: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: {
            "default": {
                "cpu_cores": 4,
                "memory_gb": 8.0,
                "gpu_count": 1,
                "storage_gb": 100.0,
                "max_agents": 5,
                "deployment_config": {
                    "image": "marl-node:latest",
                    "environment": {},
                    "resources": {"cpu": "4000m", "memory": "8Gi"},
                },
            },
            "high_performance": {
                "cpu_cores": 16,
                "memory_gb": 32.0,
                "gpu_count": 4,
                "storage_gb": 500.0,
                "max_agents": 15,
                "deployment_config": {
                    "image": "marl-node-gpu:latest",
                    "environment": {"CUDA_VISIBLE_DEVICES": "0,1,2,3"},
                    "resources": {
                        "cpu": "16000m",
                        "memory": "32Gi",
                        "nvidia.com/gpu": "4",
                    },
                },
            },
            "lightweight": {
                "cpu_cores": 2,
                "memory_gb": 4.0,
                "gpu_count": 0,
                "storage_gb": 50.0,
                "max_agents": 3,
                "deployment_config": {
                    "image": "marl-node-cpu:latest",
                    "environment": {},
                    "resources": {"cpu": "2000m", "memory": "4Gi"},
                },
            },
        }
    )

    def __post_init__(self):
        """Validate configuration."""
        if self.min_nodes <= 0:
            raise ValueError("Minimum nodes must be positive")

        if self.max_nodes < self.min_nodes:
            raise ValueError("Maximum nodes must be >= minimum nodes")

        if not (0 < self.scale_up_threshold <= 1):
            raise ValueError("Scale up threshold must be between 0 and 1")

        if not (0 < self.scale_down_threshold <= 1):
            raise ValueError("Scale down threshold must be between 0 and 1")


@dataclass
class ScalingMetrics:
    """Scaling metrics data."""

    timestamp: float
    cpu_utilization: float
    memory_utilization: float
    gpu_utilization: float
    network_latency: float
    coordination_load: float
    active_nodes: int
    total_agents: int
    coordination_success_rate: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "cpu_utilization": self.cpu_utilization,
            "memory_utilization": self.memory_utilization,
            "gpu_utilization": self.gpu_utilization,
            "network_latency": self.network_latency,
            "coordination_load": self.coordination_load,
            "active_nodes": self.active_nodes,
            "total_agents": self.total_agents,
            "coordination_success_rate": self.coordination_success_rate,
        }


@dataclass
class ScalingAction:
    """Scaling action information."""

    action_id: str
    action_type: ScalingDirection
    trigger: ScalingTrigger
    target_nodes: int
    current_nodes: int
    timestamp: float
    reason: str
    status: str = "pending"
    completion_time: Optional[float] = None
    nodes_affected: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "action_id": self.action_id,
            "action_type": self.action_type.value,
            "trigger": self.trigger.value,
            "target_nodes": self.target_nodes,
            "current_nodes": self.current_nodes,
            "timestamp": self.timestamp,
            "reason": self.reason,
            "status": self.status,
            "completion_time": self.completion_time,
            "nodes_affected": self.nodes_affected,
        }


@dataclass
class NodeInfo:
    """Node information."""

    node_id: str
    status: NodeStatus
    template: str
    created_at: float
    last_heartbeat: float
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    gpu_usage: float = 0.0
    agent_count: int = 0
    coordination_load: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "node_id": self.node_id,
            "status": self.status.value,
            "template": self.template,
            "created_at": self.created_at,
            "last_heartbeat": self.last_heartbeat,
            "cpu_usage": self.cpu_usage,
            "memory_usage": self.memory_usage,
            "gpu_usage": self.gpu_usage,
            "agent_count": self.agent_count,
            "coordination_load": self.coordination_load,
        }


class ScalabilityManager:
    """Scalability management system for distributed MARL.

    Provides auto-scaling capabilities, resource optimization, and
    deployment consistency across different environments.
    """

    def __init__(
        self,
        config: ScalingConfig,
        resource_manager: Optional[ResourceManager] = None,
        network_coordinator: Optional[NetworkCoordinator] = None,
    ):
        """
        Initialize scalability manager.

        Args:
            config: Scaling configuration
            resource_manager: Resource manager instance
            network_coordinator: Network coordinator instance
        """
        self.config = config
        self.resource_manager = resource_manager
        self.network_coordinator = network_coordinator
        self.logger = get_logger(__name__)

        # Scaling state
        self.is_active = False
        self.current_nodes = 1
        self.last_scaling_action = 0

        # Node management
        self.nodes: Dict[str, NodeInfo] = {}
        self.pending_nodes: Dict[str, NodeInfo] = {}

        # Metrics tracking
        self.metrics_history: List[ScalingMetrics] = []
        self.scaling_actions: List[ScalingAction] = []

        # Threading
        self.monitoring_thread = None
        self.scaling_thread = None
        self.node_health_thread = None
        self.scaling_lock = threading.RLock()

        # Callbacks
        self.scaling_callbacks: List[Callable] = []
        self.node_callbacks: List[Callable] = []

        # Metrics
        self.scaling_metrics = {
            "total_scaling_actions": 0,
            "successful_scaling_actions": 0,
            "failed_scaling_actions": 0,
            "average_scaling_time": 0.0,
            "current_efficiency": 0.0,
            "cost_optimization": 0.0,
            "uptime_percentage": 100.0,
        }

        # Initialize with current node
        self._initialize_current_node()

        self.logger.info("Scalability manager initialized")

    def _initialize_current_node(self) -> None:
        """Initialize current node information."""
        current_node_id = f"node_{uuid.uuid4().hex[:8]}"

        current_node = NodeInfo(
            node_id=current_node_id,
            status=NodeStatus.ACTIVE,
            template="default",
            created_at=time.time(),
            last_heartbeat=time.time(),
        )

        self.nodes[current_node_id] = current_node
        self.logger.debug("Initialized current node: %s", current_node_id)

    async def start_scalability_management(self) -> None:
        """Start scalability management services."""
        if self.is_active:
            self.logger.warning("Scalability management already active")
            return

        try:
            self.logger.info("Starting scalability management services")

            # Start monitoring
            await self._start_scaling_monitoring()

            # Start scaling engine
            if self.config.enable_auto_scaling:
                await self._start_scaling_engine()

            # Start node health monitoring
            await self._start_node_health_monitoring()

            self.is_active = True
            self.logger.info("Scalability management services started")

        except Exception as e:
            self.logger.error("Failed to start scalability management: %s", str(e))
            raise

    async def _start_scaling_monitoring(self) -> None:
        """Start scaling monitoring thread."""

        def monitoring_worker():
            while self.is_active:
                try:
                    # Collect scaling metrics
                    metrics = self._collect_scaling_metrics()

                    # Store metrics
                    with self.scaling_lock:
                        self.metrics_history.append(metrics)

                        # Limit history size
                        if len(self.metrics_history) > self.config.metric_window_size * 10:
                            self.metrics_history = self.metrics_history[
                                -self.config.metric_window_size * 5 :
                            ]

                    time.sleep(self.config.monitoring_interval)

                except Exception as e:
                    self.logger.error("Scaling monitoring error: %s", str(e))

        self.monitoring_thread = threading.Thread(target=monitoring_worker, daemon=True)
        self.monitoring_thread.start()

        self.logger.debug("Scaling monitoring started")

    async def _start_scaling_engine(self) -> None:
        """Start scaling engine thread."""

        def scaling_worker():
            while self.is_active:
                try:
                    # Check if scaling is needed
                    scaling_decision = self._evaluate_scaling_need()

                    if scaling_decision["action"] != ScalingDirection.STABLE:
                        # Execute scaling action
                        asyncio.run(self._execute_scaling_action(scaling_decision))

                    time.sleep(30.0)  # Check every 30 seconds

                except Exception as e:
                    self.logger.error("Scaling engine error: %s", str(e))

        self.scaling_thread = threading.Thread(target=scaling_worker, daemon=True)
        self.scaling_thread.start()

        self.logger.debug("Scaling engine started")

    async def _start_node_health_monitoring(self) -> None:
        """Start node health monitoring thread."""

        def health_worker():
            while self.is_active:
                try:
                    # Check node health
                    self._check_node_health()

                    # Update node metrics
                    self._update_node_metrics()

                    time.sleep(10.0)  # Check every 10 seconds

                except Exception as e:
                    self.logger.error("Node health monitoring error: %s", str(e))

        self.node_health_thread = threading.Thread(target=health_worker, daemon=True)
        self.node_health_thread.start()

        self.logger.debug("Node health monitoring started")

    def _collect_scaling_metrics(self) -> ScalingMetrics:
        """Collect current scaling metrics."""
        try:
            # Get resource metrics
            cpu_utilization = 0.0
            memory_utilization = 0.0
            gpu_utilization = 0.0

            if self.resource_manager:
                resource_metrics = self.resource_manager.get_resource_metrics(limit=1)
                if resource_metrics:
                    latest_metrics = resource_metrics[-1]
                    cpu_utilization = latest_metrics.get("cpu_percent", 0.0) / 100.0
                    memory_utilization = latest_metrics.get("memory_percent", 0.0) / 100.0
                    gpu_utilization = latest_metrics.get("gpu_percent", 0.0) / 100.0
            else:
                # Fallback to direct system metrics
                cpu_utilization = psutil.cpu_percent(interval=1) / 100.0
                memory_utilization = psutil.virtual_memory().percent / 100.0

            # Get network metrics
            network_latency = 0.0
            if self.network_coordinator:
                network_metrics = self.network_coordinator.get_network_metrics()
                network_latency = (
                    network_metrics.get("average_latency", 0.0) * 1000
                )  # Convert to ms

            # Calculate coordination load
            coordination_load = self._calculate_coordination_load()

            # Get node and agent counts
            active_nodes = len([n for n in self.nodes.values() if n.status == NodeStatus.ACTIVE])
            total_agents = sum(n.agent_count for n in self.nodes.values())

            # Calculate coordination success rate
            coordination_success_rate = self._calculate_coordination_success_rate()

            return ScalingMetrics(
                timestamp=time.time(),
                cpu_utilization=cpu_utilization,
                memory_utilization=memory_utilization,
                gpu_utilization=gpu_utilization,
                network_latency=network_latency,
                coordination_load=coordination_load,
                active_nodes=active_nodes,
                total_agents=total_agents,
                coordination_success_rate=coordination_success_rate,
            )

        except Exception as e:
            self.logger.error("Failed to collect scaling metrics: %s", str(e))
            # Return default metrics
            return ScalingMetrics(
                timestamp=time.time(),
                cpu_utilization=0.0,
                memory_utilization=0.0,
                gpu_utilization=0.0,
                network_latency=0.0,
                coordination_load=0.0,
                active_nodes=1,
                total_agents=0,
                coordination_success_rate=1.0,
            )

    def _calculate_coordination_load(self) -> float:
        """Calculate current coordination load."""
        # Simple coordination load calculation
        active_nodes = len([n for n in self.nodes.values() if n.status == NodeStatus.ACTIVE])
        total_agents = sum(n.agent_count for n in self.nodes.values())

        if active_nodes == 0:
            return 0.0

        # Load increases with more agents and coordination complexity
        agents_per_node = total_agents / active_nodes
        coordination_complexity = (active_nodes * (active_nodes - 1)) / 2  # Pairwise coordination

        # Normalize to [0, 1]
        load = min(
            1.0,
            (agents_per_node / self.config.max_agents_per_node) * 0.7
            + (coordination_complexity / 100.0) * 0.3,
        )

        return load

    def _calculate_coordination_success_rate(self) -> float:
        """Calculate coordination success rate."""
        # This would typically come from the coordination system
        # For now, simulate based on system load
        if not self.metrics_history:
            return 1.0

        recent_metrics = (
            self.metrics_history[-5:] if len(self.metrics_history) >= 5 else self.metrics_history
        )

        # Success rate decreases with high resource utilization
        avg_cpu = sum(m.cpu_utilization for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_utilization for m in recent_metrics) / len(recent_metrics)
        avg_coordination_load = sum(m.coordination_load for m in recent_metrics) / len(
            recent_metrics
        )

        # Calculate success rate based on resource pressure
        resource_pressure = (avg_cpu + avg_memory + avg_coordination_load) / 3.0
        success_rate = max(
            0.5, 1.0 - (resource_pressure - 0.7) * 2.0
        )  # Degrades after 70% utilization

        return min(1.0, success_rate)

    def _check_node_health(self) -> None:
        """Check health of all nodes."""
        current_time = time.time()
        unhealthy_nodes = []

        with self.scaling_lock:
            for node_id, node in self.nodes.items():
                # Check heartbeat timeout
                if current_time - node.last_heartbeat > 60.0:  # 1 minute timeout
                    if node.status == NodeStatus.ACTIVE:
                        node.status = NodeStatus.FAILED
                        unhealthy_nodes.append(node_id)
                        self.logger.warning("Node %s marked as failed (heartbeat timeout)", node_id)

        # Handle unhealthy nodes
        for node_id in unhealthy_nodes:
            asyncio.run(self._handle_node_failure(node_id))

    def _update_node_metrics(self) -> None:
        """Update metrics for all nodes."""
        with self.scaling_lock:
            for node in self.nodes.values():
                if node.status == NodeStatus.ACTIVE:
                    # Update heartbeat for active nodes (simulate)
                    node.last_heartbeat = time.time()

                    # Update resource usage (simulate)
                    node.cpu_usage = min(1.0, node.cpu_usage + (time.time() % 0.1 - 0.05))
                    node.memory_usage = min(1.0, node.memory_usage + (time.time() % 0.08 - 0.04))
                    node.coordination_load = self._calculate_coordination_load()

    async def _handle_node_failure(self, node_id: str) -> None:
        """Handle node failure."""
        try:
            self.logger.info("Handling failure of node: %s", node_id)

            with self.scaling_lock:
                if node_id in self.nodes:
                    failed_node = self.nodes[node_id]
                    failed_node.status = NodeStatus.FAILED

                    # Redistribute agents from failed node
                    if failed_node.agent_count > 0:
                        await self._redistribute_agents(node_id, failed_node.agent_count)

                    # Remove failed node after cleanup
                    del self.nodes[node_id]
                    self.current_nodes = len(
                        [n for n in self.nodes.values() if n.status == NodeStatus.ACTIVE]
                    )

            # Trigger scaling if needed
            if self.current_nodes < self.config.min_nodes:
                await self.manual_scale(
                    self.config.min_nodes, f"Recovery from node {node_id} failure"
                )

        except Exception as e:
            self.logger.error("Failed to handle node failure %s: %s", node_id, str(e))

    async def _redistribute_agents(self, failed_node_id: str, agent_count: int) -> None:
        """Redistribute agents from failed node."""
        self.logger.info(
            "Redistributing %d agents from failed node %s", agent_count, failed_node_id
        )

        # Find nodes with capacity
        available_nodes = [
            n
            for n in self.nodes.values()
            if n.status == NodeStatus.ACTIVE and n.agent_count < self.config.max_agents_per_node
        ]

        if not available_nodes:
            # Need to scale up
            await self.manual_scale(
                self.current_nodes + 1, "Agent redistribution requires new node"
            )
            return

        # Distribute agents across available nodes
        agents_per_node = agent_count // len(available_nodes)
        remaining_agents = agent_count % len(available_nodes)

        for i, node in enumerate(available_nodes):
            additional_agents = agents_per_node + (1 if i < remaining_agents else 0)
            node.agent_count += additional_agents

            self.logger.debug("Redistributed %d agents to node %s", additional_agents, node.node_id)

    def _evaluate_scaling_need(self) -> Dict[str, Any]:
        """Evaluate if scaling is needed."""
        try:
            with self.scaling_lock:
                if len(self.metrics_history) < self.config.metric_window_size:
                    return {
                        "action": ScalingDirection.STABLE,
                        "reason": "Insufficient metrics",
                    }

                # Check cooldown period
                if time.time() - self.last_scaling_action < self.config.scaling_cooldown:
                    return {
                        "action": ScalingDirection.STABLE,
                        "reason": "Cooldown period",
                    }

                # Get recent metrics
                recent_metrics = self.metrics_history[-self.config.metric_window_size :]

                # Evaluate each scaling policy
                scaling_scores = []

                for policy_name, policy in self.config.scaling_policies.items():
                    if not policy.get("enabled", True):
                        continue

                    score = self._evaluate_policy(policy, recent_metrics)
                    if score["action"] != ScalingDirection.STABLE:
                        score["policy"] = policy_name
                        score["weight"] = policy.get("weight", 1.0)
                        scaling_scores.append(score)

                if not scaling_scores:
                    return {
                        "action": ScalingDirection.STABLE,
                        "reason": "All metrics within thresholds",
                    }

                # Weighted decision
                up_scores = [s for s in scaling_scores if s["action"] == ScalingDirection.UP]
                down_scores = [s for s in scaling_scores if s["action"] == ScalingDirection.DOWN]

                up_weight = sum(s["weight"] * s.get("urgency", 1.0) for s in up_scores)
                down_weight = sum(s["weight"] * s.get("urgency", 1.0) for s in down_scores)

                if up_weight > down_weight and up_weight > 1.0:
                    # Scale up
                    target_nodes = min(self.config.max_nodes, self.current_nodes + 1)
                    primary_reason = max(up_scores, key=lambda x: x["weight"])

                    return {
                        "action": ScalingDirection.UP,
                        "trigger": primary_reason["trigger"],
                        "target_nodes": target_nodes,
                        "reason": f"Weighted scaling decision: {primary_reason['reason']}",
                        "urgency": primary_reason.get("urgency", 1.0),
                    }

                elif down_weight > up_weight and down_weight > 0.5:
                    # Scale down
                    target_nodes = max(self.config.min_nodes, self.current_nodes - 1)
                    primary_reason = max(down_scores, key=lambda x: x["weight"])

                    return {
                        "action": ScalingDirection.DOWN,
                        "trigger": primary_reason["trigger"],
                        "target_nodes": target_nodes,
                        "reason": f"Weighted scaling decision: {primary_reason['reason']}",
                        "urgency": primary_reason.get("urgency", 1.0),
                    }

                return {
                    "action": ScalingDirection.STABLE,
                    "reason": "Conflicting scaling signals",
                }

        except Exception as e:
            self.logger.error("Failed to evaluate scaling need: %s", str(e))
            return {
                "action": ScalingDirection.STABLE,
                "reason": f"Evaluation error: {str(e)}",
            }

    def _evaluate_policy(
        self, policy: Dict[str, Any], metrics: List[ScalingMetrics]
    ) -> Dict[str, Any]:
        """Evaluate a specific scaling policy."""
        trigger = policy["trigger"]
        scale_up_threshold = policy["scale_up_threshold"]
        scale_down_threshold = policy["scale_down_threshold"]

        # Get metric values based on trigger
        metric_values = []
        for metric in metrics:
            if trigger == ScalingTrigger.CPU_UTILIZATION:
                metric_values.append(metric.cpu_utilization)
            elif trigger == ScalingTrigger.MEMORY_UTILIZATION:
                metric_values.append(metric.memory_utilization)
            elif trigger == ScalingTrigger.GPU_UTILIZATION:
                metric_values.append(metric.gpu_utilization)
            elif trigger == ScalingTrigger.NETWORK_LATENCY:
                # Normalize latency (higher is worse)
                normalized_latency = min(
                    1.0, metric.network_latency / self.config.target_network_latency_ms
                )
                metric_values.append(normalized_latency)
            elif trigger == ScalingTrigger.COORDINATION_LOAD:
                metric_values.append(metric.coordination_load)
            elif trigger == ScalingTrigger.AGENT_COUNT:
                # Agents per node
                agents_per_node = metric.total_agents / max(1, metric.active_nodes)
                metric_values.append(agents_per_node)

        if not metric_values:
            return {"action": ScalingDirection.STABLE, "reason": "No metric values"}

        # Calculate average and trend
        avg_metric = sum(metric_values) / len(metric_values)

        # Calculate trend (recent vs older values)
        if len(metric_values) >= 5:
            recent_avg = sum(metric_values[-3:]) / 3
            older_avg = sum(metric_values[:3]) / 3
            trend = (recent_avg - older_avg) / max(older_avg, 0.001)
        else:
            trend = 0.0

        # Calculate urgency based on how far from threshold and trend
        urgency = 1.0

        # Determine scaling action
        if trigger == ScalingTrigger.AGENT_COUNT:
            # Special handling for agent count (absolute values)
            if avg_metric > scale_up_threshold and self.current_nodes < self.config.max_nodes:
                urgency = min(2.0, avg_metric / scale_up_threshold)
                return {
                    "action": ScalingDirection.UP,
                    "trigger": trigger,
                    "current_value": avg_metric,
                    "threshold": scale_up_threshold,
                    "urgency": urgency,
                    "reason": f"{trigger.value} ({avg_metric:.1f}) > threshold ({scale_up_threshold})",
                }
            elif avg_metric < scale_down_threshold and self.current_nodes > self.config.min_nodes:
                urgency = min(2.0, scale_down_threshold / max(avg_metric, 0.1))
                return {
                    "action": ScalingDirection.DOWN,
                    "trigger": trigger,
                    "current_value": avg_metric,
                    "threshold": scale_down_threshold,
                    "urgency": urgency,
                    "reason": f"{trigger.value} ({avg_metric:.1f}) < threshold ({scale_down_threshold})",
                }
        else:
            # Percentage-based thresholds
            if avg_metric > scale_up_threshold and self.current_nodes < self.config.max_nodes:
                urgency = min(2.0, 1.0 + (avg_metric - scale_up_threshold) * 2.0)
                if trend > 0.1:  # Increasing trend
                    urgency *= 1.2

                return {
                    "action": ScalingDirection.UP,
                    "trigger": trigger,
                    "current_value": avg_metric,
                    "threshold": scale_up_threshold,
                    "urgency": urgency,
                    "reason": f"{trigger.value} ({avg_metric:.3f}) > threshold ({scale_up_threshold})",
                }
            elif avg_metric < scale_down_threshold and self.current_nodes > self.config.min_nodes:
                urgency = min(2.0, 1.0 + (scale_down_threshold - avg_metric) * 2.0)
                if trend < -0.1:  # Decreasing trend
                    urgency *= 1.2

                return {
                    "action": ScalingDirection.DOWN,
                    "trigger": trigger,
                    "current_value": avg_metric,
                    "threshold": scale_down_threshold,
                    "urgency": urgency,
                    "reason": f"{trigger.value} ({avg_metric:.3f}) < threshold ({scale_down_threshold})",
                }

        return {"action": ScalingDirection.STABLE, "reason": "Metric within thresholds"}

    async def _execute_scaling_action(self, decision: Dict[str, Any]) -> None:
        """Execute scaling action."""
        try:
            action_id = str(uuid.uuid4())

            scaling_action = ScalingAction(
                action_id=action_id,
                action_type=decision["action"],
                trigger=decision["trigger"],
                target_nodes=decision["target_nodes"],
                current_nodes=self.current_nodes,
                timestamp=time.time(),
                reason=decision["reason"],
            )

            self.logger.info(
                "Executing scaling action: %s from %d to %d nodes (%s)",
                decision["action"].value,
                self.current_nodes,
                decision["target_nodes"],
                decision["reason"],
            )

            # Record scaling action
            with self.scaling_lock:
                self.scaling_actions.append(scaling_action)
                self.scaling_metrics["total_scaling_actions"] += 1

            # Execute scaling based on strategy
            success = False
            if self.config.deployment_strategy == DeploymentStrategy.HORIZONTAL:
                success = await self._execute_horizontal_scaling(scaling_action)
            elif self.config.deployment_strategy == DeploymentStrategy.VERTICAL:
                success = await self._execute_vertical_scaling(scaling_action)
            else:
                success = await self._execute_hybrid_scaling(scaling_action)

            # Update scaling action status
            scaling_action.status = "completed" if success else "failed"
            scaling_action.completion_time = time.time()

            if success:
                self.current_nodes = decision["target_nodes"]
                self.last_scaling_action = time.time()
                self.scaling_metrics["successful_scaling_actions"] += 1

                # Update average scaling time
                scaling_time = scaling_action.completion_time - scaling_action.timestamp
                self._update_average_scaling_time(scaling_time)
            else:
                self.scaling_metrics["failed_scaling_actions"] += 1

            # Notify callbacks
            await self._notify_scaling_callbacks(scaling_action)

        except Exception as e:
            self.logger.error("Failed to execute scaling action: %s", str(e))

    async def _execute_horizontal_scaling(self, action: ScalingAction) -> bool:
        """Execute horizontal scaling (add/remove nodes)."""
        try:
            if action.action_type == ScalingDirection.UP:
                # Add new nodes
                nodes_to_add = action.target_nodes - action.current_nodes
                for i in range(nodes_to_add):
                    node_id = await self._deploy_new_node()
                    if node_id:
                        action.nodes_affected.append(node_id)
                    else:
                        return False

            elif action.action_type == ScalingDirection.DOWN:
                # Remove nodes
                nodes_to_remove = action.current_nodes - action.target_nodes
                nodes_to_terminate = self._select_nodes_for_termination(nodes_to_remove)

                for node_id in nodes_to_terminate:
                    if await self._terminate_node(node_id):
                        action.nodes_affected.append(node_id)
                    else:
                        return False

            return True

        except Exception as e:
            self.logger.error("Horizontal scaling failed: %s", str(e))
            return False

    async def _execute_vertical_scaling(self, action: ScalingAction) -> bool:
        """Execute vertical scaling (add/remove resources to existing nodes)."""
        try:
            # This would typically involve resizing existing nodes
            # For now, we'll simulate it by updating node templates

            if action.action_type == ScalingDirection.UP:
                # Upgrade nodes to higher performance template
                for node in self.nodes.values():
                    if node.status == NodeStatus.ACTIVE and node.template == "default":
                        node.template = "high_performance"
                        action.nodes_affected.append(node.node_id)

            elif action.action_type == ScalingDirection.DOWN:
                # Downgrade nodes to lower performance template
                for node in self.nodes.values():
                    if node.status == NodeStatus.ACTIVE and node.template == "high_performance":
                        node.template = "default"
                        action.nodes_affected.append(node.node_id)

            self.logger.info("Vertical scaling completed (simulated)")
            return True

        except Exception as e:
            self.logger.error("Vertical scaling failed: %s", str(e))
            return False

    async def _execute_hybrid_scaling(self, action: ScalingAction) -> bool:
        """Execute hybrid scaling (combination of horizontal and vertical)."""
        try:
            # Decide between horizontal and vertical based on current state
            active_nodes = len([n for n in self.nodes.values() if n.status == NodeStatus.ACTIVE])

            if active_nodes < self.config.max_nodes // 2:
                # Prefer horizontal scaling when we have few nodes
                return await self._execute_horizontal_scaling(action)
            else:
                # Prefer vertical scaling when we have many nodes
                return await self._execute_vertical_scaling(action)

        except Exception as e:
            self.logger.error("Hybrid scaling failed: %s", str(e))
            return False

    async def _deploy_new_node(self, template: str = "default") -> Optional[str]:
        """Deploy a new node."""
        try:
            node_id = f"node_{uuid.uuid4().hex[:8]}"

            self.logger.info("Deploying new node: %s (template: %s)", node_id, template)

            # Create node info
            new_node = NodeInfo(
                node_id=node_id,
                status=NodeStatus.STARTING,
                template=template,
                created_at=time.time(),
                last_heartbeat=time.time(),
            )

            # Add to pending nodes
            with self.scaling_lock:
                self.pending_nodes[node_id] = new_node

            # Simulate deployment process
            await asyncio.sleep(2.0)  # Deployment time

            # Move to active nodes
            with self.scaling_lock:
                if node_id in self.pending_nodes:
                    new_node.status = NodeStatus.ACTIVE
                    self.nodes[node_id] = new_node
                    del self.pending_nodes[node_id]

            # Notify node callbacks
            await self._notify_node_callbacks("deployed", node_id, new_node.to_dict())

            self.logger.info("Node deployed successfully: %s", node_id)
            return node_id

        except Exception as e:
            self.logger.error("Failed to deploy node: %s", str(e))
            return None

    def _select_nodes_for_termination(self, count: int) -> List[str]:
        """Select nodes for termination."""
        # Select nodes with lowest agent count first
        active_nodes = [n for n in self.nodes.values() if n.status == NodeStatus.ACTIVE]
        active_nodes.sort(
            key=lambda x: (x.agent_count, x.created_at)
        )  # Lowest agent count, oldest first

        return [n.node_id for n in active_nodes[:count]]

    async def _terminate_node(self, node_id: str) -> bool:
        """Terminate a node."""
        try:
            self.logger.info("Terminating node: %s", node_id)

            with self.scaling_lock:
                if node_id not in self.nodes:
                    self.logger.warning("Node %s not found for termination", node_id)
                    return False

                node = self.nodes[node_id]
                node.status = NodeStatus.STOPPING

                # Redistribute agents if any
                if node.agent_count > 0:
                    await self._redistribute_agents(node_id, node.agent_count)

                # Simulate termination delay
                await asyncio.sleep(1.0)

                # Remove node
                node.status = NodeStatus.TERMINATED
                del self.nodes[node_id]

            # Notify node callbacks
            await self._notify_node_callbacks("terminated", node_id, {"status": "terminated"})

            self.logger.info("Node terminated successfully: %s", node_id)
            return True

        except Exception as e:
            self.logger.error("Failed to terminate node %s: %s", node_id, str(e))
            return False

    def _update_average_scaling_time(self, scaling_time: float) -> None:
        """Update average scaling time metric."""
        current_avg = self.scaling_metrics["average_scaling_time"]
        total_actions = self.scaling_metrics["successful_scaling_actions"]

        if total_actions > 0:
            self.scaling_metrics["average_scaling_time"] = (
                current_avg * (total_actions - 1) + scaling_time
            ) / total_actions

    async def _notify_scaling_callbacks(self, action: ScalingAction) -> None:
        """Notify scaling callbacks."""
        for callback in self.scaling_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(action)
                else:
                    callback(action)
            except Exception as e:
                self.logger.error("Scaling callback error: %s", str(e))

    async def _notify_node_callbacks(
        self, event: str, node_id: str, node_info: Dict[str, Any]
    ) -> None:
        """Notify node callbacks."""
        for callback in self.node_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event, node_id, node_info)
                else:
                    callback(event, node_id, node_info)
            except Exception as e:
                self.logger.error("Node callback error: %s", str(e))

    async def manual_scale(self, target_nodes: int, reason: str = "Manual scaling") -> bool:
        """Manually trigger scaling action.

        Args:
            target_nodes: Target number of nodes
            reason: Reason for scaling

        Returns:
            True if successful, False otherwise
        """
        try:
            if target_nodes < self.config.min_nodes or target_nodes > self.config.max_nodes:
                self.logger.error(
                    "Target nodes %d outside allowed range [%d, %d]",
                    target_nodes,
                    self.config.min_nodes,
                    self.config.max_nodes,
                )
                return False

            if target_nodes == self.current_nodes:
                self.logger.info("Target nodes equals current nodes, no scaling needed")
                return True

            # Create scaling decision
            action_type = (
                ScalingDirection.UP if target_nodes > self.current_nodes else ScalingDirection.DOWN
            )

            decision = {
                "action": action_type,
                "trigger": ScalingTrigger.CUSTOM_METRIC,
                "target_nodes": target_nodes,
                "reason": reason,
            }

            # Execute scaling
            await self._execute_scaling_action(decision)

            return True

        except Exception as e:
            self.logger.error("Manual scaling failed: %s", str(e))
            return False

    def get_scaling_status(self) -> Dict[str, Any]:
        """Get current scaling status."""
        with self.scaling_lock:
            recent_metrics = self.metrics_history[-1] if self.metrics_history else None

            return {
                "is_active": self.is_active,
                "current_nodes": self.current_nodes,
                "target_range": [self.config.min_nodes, self.config.max_nodes],
                "deployment_strategy": self.config.deployment_strategy.value,
                "auto_scaling_enabled": self.config.enable_auto_scaling,
                "last_scaling_action": self.last_scaling_action,
                "active_nodes": [
                    n.node_id for n in self.nodes.values() if n.status == NodeStatus.ACTIVE
                ],
                "pending_nodes": list(self.pending_nodes.keys()),
                "recent_metrics": recent_metrics.to_dict() if recent_metrics else None,
                "scaling_policies": {
                    name: policy
                    for name, policy in self.config.scaling_policies.items()
                    if policy.get("enabled", True)
                },
            }

    def get_scaling_metrics(self) -> Dict[str, Any]:
        """Get scaling metrics."""
        # Calculate current efficiency
        if self.metrics_history:
            recent_metrics = self.metrics_history[-5:]  # Last 5 measurements
            avg_cpu = sum(m.cpu_utilization for m in recent_metrics) / len(recent_metrics)
            avg_memory = sum(m.memory_utilization for m in recent_metrics) / len(recent_metrics)

            # Efficiency is how close we are to target utilization
            cpu_efficiency = 1.0 - abs(avg_cpu - self.config.target_cpu_utilization)
            memory_efficiency = 1.0 - abs(avg_memory - self.config.target_memory_utilization)
            self.scaling_metrics["current_efficiency"] = (cpu_efficiency + memory_efficiency) / 2.0

        # Calculate uptime percentage
        total_nodes = len(self.nodes) + len(self.pending_nodes)
        active_nodes = len([n for n in self.nodes.values() if n.status == NodeStatus.ACTIVE])
        self.scaling_metrics["uptime_percentage"] = (active_nodes / max(1, total_nodes)) * 100.0

        return self.scaling_metrics.copy()

    def get_node_status(self) -> Dict[str, Any]:
        """Get status of all nodes."""
        with self.scaling_lock:
            return {
                "active_nodes": {
                    node_id: node.to_dict()
                    for node_id, node in self.nodes.items()
                    if node.status == NodeStatus.ACTIVE
                },
                "pending_nodes": {
                    node_id: node.to_dict() for node_id, node in self.pending_nodes.items()
                },
                "failed_nodes": {
                    node_id: node.to_dict()
                    for node_id, node in self.nodes.items()
                    if node.status == NodeStatus.FAILED
                },
                "summary": {
                    "total_nodes": len(self.nodes) + len(self.pending_nodes),
                    "active_count": len(
                        [n for n in self.nodes.values() if n.status == NodeStatus.ACTIVE]
                    ),
                    "pending_count": len(self.pending_nodes),
                    "failed_count": len(
                        [n for n in self.nodes.values() if n.status == NodeStatus.FAILED]
                    ),
                    "total_agents": sum(n.agent_count for n in self.nodes.values()),
                },
            }

    def get_scaling_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get scaling action history.

        Args:
            limit: Maximum number of actions to return

        Returns:
            List of scaling actions
        """
        with self.scaling_lock:
            actions = self.scaling_actions.copy()

            if limit:
                actions = actions[-limit:]

            return [action.to_dict() for action in actions]

    def get_metrics_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get metrics history.

        Args:
            limit: Maximum number of metrics to return

        Returns:
            List of scaling metrics
        """
        with self.scaling_lock:
            metrics = self.metrics_history.copy()

            if limit:
                metrics = metrics[-limit:]

            return [metric.to_dict() for metric in metrics]

    def add_scaling_callback(self, callback: Callable) -> None:
        """Add scaling event callback."""
        self.scaling_callbacks.append(callback)

    def add_node_callback(self, callback: Callable) -> None:
        """Add node event callback."""
        self.node_callbacks.append(callback)

    def update_scaling_config(self, config_updates: Dict[str, Any]) -> None:
        """Update scaling configuration.

        Args:
            config_updates: Configuration updates to apply
        """
        try:
            for key, value in config_updates.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
                    self.logger.info("Updated scaling config: %s = %s", key, value)
                else:
                    self.logger.warning("Unknown config key: %s", key)

        except Exception as e:
            self.logger.error("Failed to update scaling config: %s", str(e))

    async def shutdown(self) -> None:
        """Shutdown scalability manager."""
        self.logger.info("Shutting down scalability manager")

        self.is_active = False

        # Stop threads
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5.0)

        if self.scaling_thread and self.scaling_thread.is_alive():
            self.scaling_thread.join(timeout=5.0)

        if self.node_health_thread and self.node_health_thread.is_alive():
            self.node_health_thread.join(timeout=5.0)

        # Terminate all managed nodes (except current)
        with self.scaling_lock:
            nodes_to_terminate = [
                node_id
                for node_id, node in self.nodes.items()
                if node.status == NodeStatus.ACTIVE and len(self.nodes) > 1
            ]

            for node_id in nodes_to_terminate:
                await self._terminate_node(node_id)

            self.nodes.clear()
            self.pending_nodes.clear()
            self.metrics_history.clear()

        # Clear callbacks
        self.scaling_callbacks.clear()
        self.node_callbacks.clear()

        self.logger.info("Scalability manager shutdown complete")
