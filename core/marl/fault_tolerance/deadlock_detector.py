"""
Deadlock Detection and Resolution.

This module provides deadlock detection and resolution mechanisms
for the multi-agent reinforcement learning coordination system.
"""

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from utils.logging_config import get_logger


class DeadlockType(Enum):
    """Types of deadlocks that can occur."""

    COORDINATION_DEADLOCK = "coordination_deadlock"
    RESOURCE_DEADLOCK = "resource_deadlock"
    COMMUNICATION_DEADLOCK = "communication_deadlock"
    CONSENSUS_DEADLOCK = "consensus_deadlock"
    CIRCULAR_WAIT = "circular_wait"
    UNKNOWN = "unknown"


@dataclass
class DeadlockEvent:
    """Represents a deadlock event."""

    deadlock_id: str
    deadlock_type: DeadlockType
    involved_agents: List[str]
    involved_resources: List[str] = field(default_factory=list)
    detection_time: datetime = field(default_factory=datetime.now)
    resolution_time: Optional[datetime] = None
    resolved: bool = False
    resolution_strategy: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "deadlock_id": self.deadlock_id,
            "deadlock_type": self.deadlock_type.value,
            "involved_agents": self.involved_agents,
            "involved_resources": self.involved_resources,
            "detection_time": self.detection_time.isoformat(),
            "resolution_time": self.resolution_time.isoformat()
            if self.resolution_time
            else None,
            "resolved": self.resolved,
            "resolution_strategy": self.resolution_strategy,
            "duration_seconds": (
                (self.resolution_time or datetime.now()) - self.detection_time
            ).total_seconds(),
            "metadata": self.metadata,
        }


@dataclass
class WaitingState:
    """Represents an agent waiting for a resource or other agent."""

    agent_id: str
    waiting_for: str  # Resource or agent ID
    wait_type: str  # "resource", "agent", "consensus", etc.
    start_time: datetime = field(default_factory=datetime.now)
    timeout: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_timed_out(self) -> bool:
        """Check if wait has timed out."""
        if not self.timeout:
            return False

        elapsed = (datetime.now() - self.start_time).total_seconds()
        return elapsed > self.timeout

    def get_wait_duration(self) -> float:
        """Get wait duration in seconds."""
        return (datetime.now() - self.start_time).total_seconds()


class DeadlockDetector:
    """
    Detects and resolves deadlocks in the MARL system.

    Monitors agent interactions, resource usage, and coordination
    patterns to identify potential deadlocks and automatically
    resolve them using various strategies.
    """

    def __init__(
        self,
        detection_interval: float = 5.0,
        deadlock_timeout: float = 30.0,
        max_wait_time: float = 60.0,
        enable_auto_resolution: bool = True,
    ):
        """
        Initialize deadlock detector.

        Args:
            detection_interval: Interval between deadlock checks (seconds)
            deadlock_timeout: Time before considering a wait as potential deadlock
            max_wait_time: Maximum allowed wait time before forced resolution
            enable_auto_resolution: Enable automatic deadlock resolution
        """
        self.logger = get_logger(__name__)

        # Configuration
        self.detection_interval = detection_interval
        self.deadlock_timeout = deadlock_timeout
        self.max_wait_time = max_wait_time
        self.enable_auto_resolution = enable_auto_resolution

        # State tracking
        self.waiting_states: Dict[str, WaitingState] = {}
        self.resource_owners: Dict[str, str] = {}  # resource_id -> agent_id
        self.agent_dependencies: Dict[str, Set[str]] = defaultdict(set)

        # Deadlock tracking
        self.active_deadlocks: Dict[str, DeadlockEvent] = {}
        self.deadlock_history: List[DeadlockEvent] = []
        self.deadlock_counter = 0

        # Detection state
        self.is_detecting = False
        self.detection_task: Optional[asyncio.Task] = None

        # Resolution strategies
        self.resolution_strategies = {
            DeadlockType.COORDINATION_DEADLOCK: self._resolve_coordination_deadlock,
            DeadlockType.RESOURCE_DEADLOCK: self._resolve_resource_deadlock,
            DeadlockType.COMMUNICATION_DEADLOCK: self._resolve_communication_deadlock,
            DeadlockType.CONSENSUS_DEADLOCK: self._resolve_consensus_deadlock,
            DeadlockType.CIRCULAR_WAIT: self._resolve_circular_wait,
        }

        # Callbacks
        self.deadlock_callbacks: List[callable] = []
        self.resolution_callbacks: List[callable] = []

        self.logger.info("Deadlock detector initialized")

    async def start_detection(self) -> None:
        """Start deadlock detection."""
        if self.is_detecting:
            self.logger.warning("Deadlock detection already running")
            return

        self.is_detecting = True
        self.detection_task = asyncio.create_task(self._detection_loop())

        self.logger.info("Deadlock detection started")

    async def stop_detection(self) -> None:
        """Stop deadlock detection."""
        if not self.is_detecting:
            return

        self.is_detecting = False

        if self.detection_task:
            self.detection_task.cancel()
            try:
                await self.detection_task
            except asyncio.CancelledError:
                pass

        self.logger.info("Deadlock detection stopped")

    async def _detection_loop(self) -> None:
        """Main detection loop."""
        while self.is_detecting:
            try:
                await self._check_for_deadlocks()
                await asyncio.sleep(self.detection_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(
                    "Error in deadlock detection loop: %s", str(e), exc_info=True
                )
                await asyncio.sleep(5.0)  # Wait before retrying

    async def _check_for_deadlocks(self) -> None:
        """Check for various types of deadlocks."""
        # Clean up expired waiting states
        self._cleanup_expired_waits()

        # Check for different types of deadlocks
        await self._check_circular_waits()
        await self._check_coordination_deadlocks()
        await self._check_resource_deadlocks()
        await self._check_communication_deadlocks()
        await self._check_consensus_deadlocks()

        # Check for timeout-based deadlocks
        await self._check_timeout_deadlocks()

    def _cleanup_expired_waits(self) -> None:
        """Clean up expired waiting states."""
        current_time = datetime.now()
        expired_waits = []

        for agent_id, wait_state in self.waiting_states.items():
            if wait_state.is_timed_out():
                expired_waits.append(agent_id)

        for agent_id in expired_waits:
            self.logger.debug("Removing expired wait for agent: %s", agent_id)
            del self.waiting_states[agent_id]

            # Clean up dependencies
            if agent_id in self.agent_dependencies:
                del self.agent_dependencies[agent_id]

    async def _check_circular_waits(self) -> None:
        """Check for circular wait deadlocks."""
        # Build dependency graph
        dependency_graph = self._build_dependency_graph()

        # Find cycles in the graph
        cycles = self._find_cycles(dependency_graph)

        for cycle in cycles:
            if len(cycle) > 1:  # Actual cycle
                deadlock_id = f"circular_wait_{self.deadlock_counter}"
                self.deadlock_counter += 1

                # Check if this is a new deadlock
                if not self._is_existing_deadlock(cycle):
                    await self._handle_detected_deadlock(
                        deadlock_id,
                        DeadlockType.CIRCULAR_WAIT,
                        cycle,
                        metadata={"cycle_length": len(cycle)},
                    )

    def _build_dependency_graph(self) -> Dict[str, Set[str]]:
        """Build dependency graph from waiting states."""
        graph = defaultdict(set)

        for agent_id, wait_state in self.waiting_states.items():
            if wait_state.wait_type == "agent":
                graph[agent_id].add(wait_state.waiting_for)
            elif wait_state.wait_type == "resource":
                # If waiting for resource, depend on resource owner
                resource_owner = self.resource_owners.get(wait_state.waiting_for)
                if resource_owner and resource_owner != agent_id:
                    graph[agent_id].add(resource_owner)

        return dict(graph)

    def _find_cycles(self, graph: Dict[str, Set[str]]) -> List[List[str]]:
        """Find cycles in dependency graph using DFS."""
        cycles = []
        visited = set()
        rec_stack = set()
        path = []

        def dfs(node: str) -> None:
            if node in rec_stack:
                # Found cycle
                cycle_start = path.index(node)
                cycle = path[cycle_start:] + [node]
                cycles.append(cycle)
                return

            if node in visited:
                return

            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in graph.get(node, set()):
                dfs(neighbor)

            rec_stack.remove(node)
            path.pop()

        for node in graph:
            if node not in visited:
                dfs(node)

        return cycles

    async def _check_coordination_deadlocks(self) -> None:
        """Check for coordination deadlocks."""
        # Look for agents waiting for coordination responses
        coordination_waits = [
            (agent_id, wait_state)
            for agent_id, wait_state in self.waiting_states.items()
            if wait_state.wait_type == "coordination"
        ]

        # Group by coordination ID
        coordination_groups = defaultdict(list)
        for agent_id, wait_state in coordination_waits:
            coord_id = wait_state.metadata.get("coordination_id", "unknown")
            coordination_groups[coord_id].append(agent_id)

        # Check for potential deadlocks
        for coord_id, agents in coordination_groups.items():
            if len(agents) > 1:
                # Check if all agents are waiting for each other
                if self._are_agents_mutually_waiting(agents):
                    deadlock_id = f"coordination_deadlock_{self.deadlock_counter}"
                    self.deadlock_counter += 1

                    await self._handle_detected_deadlock(
                        deadlock_id,
                        DeadlockType.COORDINATION_DEADLOCK,
                        agents,
                        metadata={"coordination_id": coord_id},
                    )

    async def _check_resource_deadlocks(self) -> None:
        """Check for resource deadlocks."""
        # Look for resource contention patterns
        resource_waits = defaultdict(list)

        for agent_id, wait_state in self.waiting_states.items():
            if wait_state.wait_type == "resource":
                resource_waits[wait_state.waiting_for].append(agent_id)

        # Check for deadlock patterns
        for resource_id, waiting_agents in resource_waits.items():
            if len(waiting_agents) > 1:
                # Check if resource owner is also waiting
                owner = self.resource_owners.get(resource_id)
                if owner and owner in self.waiting_states:
                    # Potential resource deadlock
                    involved_agents = waiting_agents + [owner]

                    deadlock_id = f"resource_deadlock_{self.deadlock_counter}"
                    self.deadlock_counter += 1

                    await self._handle_detected_deadlock(
                        deadlock_id,
                        DeadlockType.RESOURCE_DEADLOCK,
                        involved_agents,
                        involved_resources=[resource_id],
                        metadata={"resource_id": resource_id, "owner": owner},
                    )

    async def _check_communication_deadlocks(self) -> None:
        """Check for communication deadlocks."""
        # Look for agents waiting for messages from each other
        message_waits = [
            (agent_id, wait_state)
            for agent_id, wait_state in self.waiting_states.items()
            if wait_state.wait_type == "message"
        ]

        # Check for mutual message waiting
        for i, (agent1, wait1) in enumerate(message_waits):
            for j, (agent2, wait2) in enumerate(message_waits[i + 1 :], i + 1):
                if wait1.waiting_for == agent2 and wait2.waiting_for == agent1:
                    # Mutual message waiting detected

                    deadlock_id = f"communication_deadlock_{self.deadlock_counter}"
                    self.deadlock_counter += 1

                    await self._handle_detected_deadlock(
                        deadlock_id,
                        DeadlockType.COMMUNICATION_DEADLOCK,
                        [agent1, agent2],
                        metadata={
                            "agent1_waiting_for": wait1.waiting_for,
                            "agent2_waiting_for": wait2.waiting_for,
                        },
                    )

    async def _check_consensus_deadlocks(self) -> None:
        """Check for consensus deadlocks."""
        # Look for agents waiting for consensus
        consensus_waits = [
            (agent_id, wait_state)
            for agent_id, wait_state in self.waiting_states.items()
            if wait_state.wait_type == "consensus"
        ]

        # Group by proposal ID
        proposal_groups = defaultdict(list)
        for agent_id, wait_state in consensus_waits:
            proposal_id = wait_state.metadata.get("proposal_id", "unknown")
            proposal_groups[proposal_id].append(agent_id)

        # Check for stalled consensus
        for proposal_id, agents in proposal_groups.items():
            if len(agents) > 1:
                # Check if consensus has been waiting too long
                max_wait_time = max(
                    wait_state.get_wait_duration()
                    for _, wait_state in consensus_waits
                    if wait_state.metadata.get("proposal_id") == proposal_id
                )

                if max_wait_time > self.deadlock_timeout:
                    deadlock_id = f"consensus_deadlock_{self.deadlock_counter}"
                    self.deadlock_counter += 1

                    await self._handle_detected_deadlock(
                        deadlock_id,
                        DeadlockType.CONSENSUS_DEADLOCK,
                        agents,
                        metadata={
                            "proposal_id": proposal_id,
                            "wait_time": max_wait_time,
                        },
                    )

    async def _check_timeout_deadlocks(self) -> None:
        """Check for timeout-based deadlocks."""
        current_time = datetime.now()

        for agent_id, wait_state in self.waiting_states.items():
            wait_duration = wait_state.get_wait_duration()

            if wait_duration > self.max_wait_time:
                # Force resolution of long waits
                deadlock_id = f"timeout_deadlock_{self.deadlock_counter}"
                self.deadlock_counter += 1

                await self._handle_detected_deadlock(
                    deadlock_id,
                    DeadlockType.UNKNOWN,
                    [agent_id],
                    metadata={
                        "wait_type": wait_state.wait_type,
                        "waiting_for": wait_state.waiting_for,
                        "wait_duration": wait_duration,
                    },
                )

    def _are_agents_mutually_waiting(self, agents: List[str]) -> bool:
        """Check if agents are mutually waiting for each other."""
        for agent in agents:
            if agent not in self.waiting_states:
                return False

            wait_state = self.waiting_states[agent]
            if wait_state.waiting_for not in agents:
                return False

        return True

    def _is_existing_deadlock(self, involved_agents: List[str]) -> bool:
        """Check if deadlock with same agents already exists."""
        agent_set = set(involved_agents)

        for deadlock in self.active_deadlocks.values():
            if set(deadlock.involved_agents) == agent_set:
                return True

        return False

    async def _handle_detected_deadlock(
        self,
        deadlock_id: str,
        deadlock_type: DeadlockType,
        involved_agents: List[str],
        involved_resources: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Handle a detected deadlock."""
        deadlock_event = DeadlockEvent(
            deadlock_id=deadlock_id,
            deadlock_type=deadlock_type,
            involved_agents=involved_agents,
            involved_resources=involved_resources or [],
            metadata=metadata or {},
        )

        self.active_deadlocks[deadlock_id] = deadlock_event

        self.logger.warning(
            "Deadlock detected: %s (type: %s, agents: %s)",
            deadlock_id,
            deadlock_type.value,
            involved_agents,
        )

        # Notify callbacks
        await self._notify_deadlock_callbacks(deadlock_event)

        # Attempt resolution if enabled
        if self.enable_auto_resolution:
            await self._resolve_deadlock(deadlock_event)

    async def _resolve_deadlock(self, deadlock_event: DeadlockEvent) -> None:
        """Resolve a deadlock using appropriate strategy."""
        try:
            strategy_func = self.resolution_strategies.get(deadlock_event.deadlock_type)

            if strategy_func:
                success = await strategy_func(deadlock_event)

                if success:
                    deadlock_event.resolved = True
                    deadlock_event.resolution_time = datetime.now()
                    deadlock_event.resolution_strategy = strategy_func.__name__

                    self.logger.info(
                        "Deadlock resolved: %s using strategy: %s",
                        deadlock_event.deadlock_id,
                        deadlock_event.resolution_strategy,
                    )

                    # Move to history
                    self.deadlock_history.append(deadlock_event)
                    del self.active_deadlocks[deadlock_event.deadlock_id]

                    # Notify resolution callbacks
                    await self._notify_resolution_callbacks(deadlock_event)
                else:
                    self.logger.warning(
                        "Failed to resolve deadlock: %s", deadlock_event.deadlock_id
                    )
            else:
                self.logger.warning(
                    "No resolution strategy for deadlock type: %s",
                    deadlock_event.deadlock_type.value,
                )

        except Exception as e:
            self.logger.error(
                "Error resolving deadlock %s: %s",
                deadlock_event.deadlock_id,
                str(e),
                exc_info=True,
            )

    async def _resolve_coordination_deadlock(
        self, deadlock_event: DeadlockEvent
    ) -> bool:
        """Resolve coordination deadlock."""
        try:
            # Reset coordination state for involved agents
            for agent_id in deadlock_event.involved_agents:
                if agent_id in self.waiting_states:
                    del self.waiting_states[agent_id]

            # Clear agent dependencies
            for agent_id in deadlock_event.involved_agents:
                if agent_id in self.agent_dependencies:
                    del self.agent_dependencies[agent_id]

            self.logger.info("Coordination deadlock resolved by resetting state")
            return True

        except Exception as e:
            self.logger.error("Failed to resolve coordination deadlock: %s", str(e))
            return False

    async def _resolve_resource_deadlock(self, deadlock_event: DeadlockEvent) -> bool:
        """Resolve resource deadlock."""
        try:
            # Release resources held by involved agents
            for resource_id in deadlock_event.involved_resources:
                if resource_id in self.resource_owners:
                    owner = self.resource_owners[resource_id]
                    if owner in deadlock_event.involved_agents:
                        del self.resource_owners[resource_id]
                        self.logger.debug(
                            "Released resource %s from agent %s", resource_id, owner
                        )

            # Clear waiting states for involved agents
            for agent_id in deadlock_event.involved_agents:
                if agent_id in self.waiting_states:
                    del self.waiting_states[agent_id]

            self.logger.info("Resource deadlock resolved by releasing resources")
            return True

        except Exception as e:
            self.logger.error("Failed to resolve resource deadlock: %s", str(e))
            return False

    async def _resolve_communication_deadlock(
        self, deadlock_event: DeadlockEvent
    ) -> bool:
        """Resolve communication deadlock."""
        try:
            # Break communication cycle by clearing one side
            if len(deadlock_event.involved_agents) >= 2:
                # Clear waiting state for first agent
                agent_to_clear = deadlock_event.involved_agents[0]
                if agent_to_clear in self.waiting_states:
                    del self.waiting_states[agent_to_clear]
                    self.logger.debug(
                        "Cleared communication wait for agent %s", agent_to_clear
                    )

            self.logger.info("Communication deadlock resolved by breaking cycle")
            return True

        except Exception as e:
            self.logger.error("Failed to resolve communication deadlock: %s", str(e))
            return False

    async def _resolve_consensus_deadlock(self, deadlock_event: DeadlockEvent) -> bool:
        """Resolve consensus deadlock."""
        try:
            # Force consensus timeout for involved agents
            proposal_id = deadlock_event.metadata.get("proposal_id")

            for agent_id in deadlock_event.involved_agents:
                if agent_id in self.waiting_states:
                    wait_state = self.waiting_states[agent_id]
                    if wait_state.metadata.get("proposal_id") == proposal_id:
                        del self.waiting_states[agent_id]
                        self.logger.debug(
                            "Forced consensus timeout for agent %s", agent_id
                        )

            self.logger.info("Consensus deadlock resolved by forcing timeout")
            return True

        except Exception as e:
            self.logger.error("Failed to resolve consensus deadlock: %s", str(e))
            return False

    async def _resolve_circular_wait(self, deadlock_event: DeadlockEvent) -> bool:
        """Resolve circular wait deadlock."""
        try:
            # Break the cycle by removing one dependency
            if len(deadlock_event.involved_agents) > 1:
                # Remove waiting state for one agent to break cycle
                agent_to_break = deadlock_event.involved_agents[0]
                if agent_to_break in self.waiting_states:
                    del self.waiting_states[agent_to_break]
                    self.logger.debug(
                        "Broke circular wait by clearing agent %s", agent_to_break
                    )

            self.logger.info("Circular wait deadlock resolved by breaking cycle")
            return True

        except Exception as e:
            self.logger.error("Failed to resolve circular wait: %s", str(e))
            return False

    async def _notify_deadlock_callbacks(self, deadlock_event: DeadlockEvent) -> None:
        """Notify deadlock detection callbacks."""
        for callback in self.deadlock_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(deadlock_event)
                else:
                    callback(deadlock_event)
            except Exception as e:
                self.logger.error("Error in deadlock callback: %s", str(e))

    async def _notify_resolution_callbacks(self, deadlock_event: DeadlockEvent) -> None:
        """Notify deadlock resolution callbacks."""
        for callback in self.resolution_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(deadlock_event)
                else:
                    callback(deadlock_event)
            except Exception as e:
                self.logger.error("Error in resolution callback: %s", str(e))

    # Public interface methods

    def record_agent_wait(
        self,
        agent_id: str,
        waiting_for: str,
        wait_type: str,
        timeout: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record that an agent is waiting for something."""
        wait_state = WaitingState(
            agent_id=agent_id,
            waiting_for=waiting_for,
            wait_type=wait_type,
            timeout=timeout,
            metadata=metadata or {},
        )

        self.waiting_states[agent_id] = wait_state

        # Update dependencies
        if wait_type == "agent":
            self.agent_dependencies[agent_id].add(waiting_for)

        self.logger.debug(
            "Recorded wait: agent %s waiting for %s (%s)",
            agent_id,
            waiting_for,
            wait_type,
        )

    def clear_agent_wait(self, agent_id: str) -> None:
        """Clear an agent's waiting state."""
        if agent_id in self.waiting_states:
            del self.waiting_states[agent_id]

        if agent_id in self.agent_dependencies:
            del self.agent_dependencies[agent_id]

        self.logger.debug("Cleared wait for agent: %s", agent_id)

    def record_resource_ownership(self, resource_id: str, owner_agent_id: str) -> None:
        """Record resource ownership."""
        self.resource_owners[resource_id] = owner_agent_id
        self.logger.debug("Resource %s owned by agent %s", resource_id, owner_agent_id)

    def release_resource(self, resource_id: str) -> None:
        """Release a resource."""
        if resource_id in self.resource_owners:
            owner = self.resource_owners[resource_id]
            del self.resource_owners[resource_id]
            self.logger.debug("Released resource %s from agent %s", resource_id, owner)

    def get_active_deadlocks(self) -> List[DeadlockEvent]:
        """Get list of active deadlocks."""
        return list(self.active_deadlocks.values())

    def get_deadlock_history(self) -> List[DeadlockEvent]:
        """Get deadlock history."""
        return self.deadlock_history.copy()

    def get_waiting_agents(self) -> List[str]:
        """Get list of agents currently waiting."""
        return list(self.waiting_states.keys())

    def get_deadlock_statistics(self) -> Dict[str, Any]:
        """Get deadlock statistics."""
        total_deadlocks = len(self.deadlock_history) + len(self.active_deadlocks)
        resolved_deadlocks = len(self.deadlock_history)

        # Count by type
        type_counts = defaultdict(int)
        for deadlock in self.deadlock_history + list(self.active_deadlocks.values()):
            type_counts[deadlock.deadlock_type.value] += 1

        # Calculate average resolution time
        resolution_times = [
            (deadlock.resolution_time - deadlock.detection_time).total_seconds()
            for deadlock in self.deadlock_history
            if deadlock.resolution_time
        ]

        avg_resolution_time = (
            sum(resolution_times) / len(resolution_times) if resolution_times else 0.0
        )

        return {
            "total_deadlocks": total_deadlocks,
            "active_deadlocks": len(self.active_deadlocks),
            "resolved_deadlocks": resolved_deadlocks,
            "resolution_rate": resolved_deadlocks / total_deadlocks
            if total_deadlocks > 0
            else 0.0,
            "average_resolution_time": avg_resolution_time,
            "deadlocks_by_type": dict(type_counts),
            "currently_waiting_agents": len(self.waiting_states),
        }

    def add_deadlock_callback(self, callback: callable) -> None:
        """Add deadlock detection callback."""
        self.deadlock_callbacks.append(callback)

    def add_resolution_callback(self, callback: callable) -> None:
        """Add deadlock resolution callback."""
        self.resolution_callbacks.append(callback)

    def remove_deadlock_callback(self, callback: callable) -> None:
        """Remove deadlock detection callback."""
        if callback in self.deadlock_callbacks:
            self.deadlock_callbacks.remove(callback)

    def remove_resolution_callback(self, callback: callable) -> None:
        """Remove deadlock resolution callback."""
        if callback in self.resolution_callbacks:
            self.resolution_callbacks.remove(callback)

    async def shutdown(self) -> None:
        """Shutdown deadlock detector."""
        await self.stop_detection()

        # Clear all state
        self.waiting_states.clear()
        self.resource_owners.clear()
        self.agent_dependencies.clear()
        self.active_deadlocks.clear()
        self.deadlock_callbacks.clear()
        self.resolution_callbacks.clear()

        self.logger.info("Deadlock detector shutdown complete")
