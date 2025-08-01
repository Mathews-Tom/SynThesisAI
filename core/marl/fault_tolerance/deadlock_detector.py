"""Deadlock Detection and Resolution for the SynThesisAI MARL Framework.

This module provides robust deadlock detection and resolution mechanisms tailored for
the multi-agent reinforcement learning (MARL) coordination system. It identifies and
mitigates various forms of deadlocks, including circular waits, resource contention,
and communication standstills, ensuring system stability and continuous operation.
"""

# Standard Library
import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

# SynThesisAI Modules
from utils.logging_config import get_logger


class DeadlockType(Enum):
    """Enumerates the types of deadlocks that can be detected.

    Attributes:
        COORDINATION_DEADLOCK: Deadlock arising from failed coordination attempts.
        RESOURCE_DEADLOCK: Deadlock due to circular dependencies on shared resources.
        COMMUNICATION_DEADLOCK: Deadlock caused by agents waiting for messages from each other.
        CONSENSUS_DEADLOCK: Deadlock where agents cannot reach a required consensus.
        CIRCULAR_WAIT: A classic deadlock condition where two or more agents wait for each other in a cycle.
        UNKNOWN: A deadlock of an undetermined or unexpected type.
    """

    COORDINATION_DEADLOCK = "coordination_deadlock"
    RESOURCE_DEADLOCK = "resource_deadlock"
    COMMUNICATION_DEADLOCK = "communication_deadlock"
    CONSENSUS_DEADLOCK = "consensus_deadlock"
    CIRCULAR_WAIT = "circular_wait"
    UNKNOWN = "unknown"


@dataclass
class DeadlockEvent:
    """Represents a single, detected deadlock event.

    This data class captures all relevant information about a detected deadlock,
    including the agents and resources involved, timing, and resolution status.

    Attributes:
        deadlock_id: A unique identifier for the deadlock event.
        deadlock_type: The type of deadlock, as defined by `DeadlockType`.
        involved_agents: A list of agent IDs involved in the deadlock.
        involved_resources: A list of resource IDs involved in the deadlock.
        detection_time: The timestamp when the deadlock was detected.
        resolution_time: The timestamp when the deadlock was resolved.
        resolved: A boolean flag indicating whether the deadlock has been resolved.
        resolution_strategy: The name of the strategy used to resolve the deadlock.
        metadata: A dictionary for storing additional context-specific information.
    """

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
        """Serializes the deadlock event to a dictionary.

        Returns:
            A dictionary representation of the deadlock event, suitable for logging or reporting.
        """
        return {
            "deadlock_id": self.deadlock_id,
            "deadlock_type": self.deadlock_type.value,
            "involved_agents": self.involved_agents,
            "involved_resources": self.involved_resources,
            "detection_time": self.detection_time.isoformat(),
            "resolution_time": (self.resolution_time.isoformat() if self.resolution_time else None),
            "resolved": self.resolved,
            "resolution_strategy": self.resolution_strategy,
            "duration_seconds": (
                (self.resolution_time or datetime.now()) - self.detection_time
            ).total_seconds(),
            "metadata": self.metadata,
        }


@dataclass
class WaitingState:
    """Represents an agent's state of waiting for a resource or another agent.

    Attributes:
        agent_id: The ID of the waiting agent.
        waiting_for: The ID of the resource or agent being waited for.
        wait_type: The category of the wait (e.g., "resource", "agent").
        start_time: The timestamp when the agent started waiting.
        timeout: The optional duration in seconds before the wait is considered timed out.
        metadata: A dictionary for additional wait-specific context.
    """

    agent_id: str
    waiting_for: str
    wait_type: str
    start_time: datetime = field(default_factory=datetime.now)
    timeout: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_timed_out(self) -> bool:
        """Checks if the waiting state has exceeded its timeout.

        Returns:
            True if the wait has timed out, False otherwise.
        """
        if self.timeout is None:
            return False
        return self.get_wait_duration() > self.timeout

    def get_wait_duration(self) -> float:
        """Calculates the total duration of the current wait in seconds.

        Returns:
            The elapsed time in seconds since the wait began.
        """
        return (datetime.now() - self.start_time).total_seconds()


class DeadlockDetector:
    """Detects and resolves deadlocks in the MARL system.

    This class monitors agent interactions, resource allocations, and coordination
    patterns to identify potential deadlocks. When a deadlock is detected, it can
    trigger automatic resolution strategies to restore system fluidity.

    Attributes:
        detection_interval: The frequency (in seconds) of deadlock checks.
        deadlock_timeout: The duration (in seconds) after which a wait is considered a potential deadlock.
        max_wait_time: The maximum time (in seconds) an agent can wait before forced resolution.
        enable_auto_resolution: A flag to enable or disable automatic deadlock resolution.
    """

    def __init__(
        self,
        detection_interval: float = 5.0,
        deadlock_timeout: float = 30.0,
        max_wait_time: float = 60.0,
        enable_auto_resolution: bool = True,
    ):
        """Initializes the DeadlockDetector.

        Args:
            detection_interval: Interval in seconds between deadlock detection cycles.
            deadlock_timeout: Time in seconds before a prolonged wait is considered a potential deadlock.
            max_wait_time: Maximum allowed wait time in seconds before triggering forced resolution.
            enable_auto_resolution: If True, automatically attempts to resolve detected deadlocks.
        """
        self.logger = get_logger(__name__)

        # Configuration
        self.detection_interval: float = detection_interval
        self.deadlock_timeout: float = deadlock_timeout
        self.max_wait_time: float = max_wait_time
        self.enable_auto_resolution: bool = enable_auto_resolution

        # State tracking
        self.waiting_states: Dict[str, WaitingState] = {}
        self.resource_owners: Dict[str, str] = {}  # resource_id -> agent_id
        self.agent_dependencies: Dict[str, Set[str]] = defaultdict(set)

        # Deadlock tracking
        self.active_deadlocks: Dict[str, DeadlockEvent] = {}
        self.deadlock_history: List[DeadlockEvent] = []
        self.deadlock_counter: int = 0

        # Detection state
        self.is_detecting: bool = False
        self.detection_task: Optional[asyncio.Task] = None

        # Resolution strategies
        self.resolution_strategies: Dict[DeadlockType, Callable] = {
            DeadlockType.COORDINATION_DEADLOCK: self._resolve_coordination_deadlock,
            DeadlockType.RESOURCE_DEADLOCK: self._resolve_resource_deadlock,
            DeadlockType.COMMUNICATION_DEADLOCK: self._resolve_communication_deadlock,
            DeadlockType.CONSENSUS_DEADLOCK: self._resolve_consensus_deadlock,
            DeadlockType.CIRCULAR_WAIT: self._resolve_circular_wait,
        }

        # Callbacks
        self.deadlock_callbacks: List[Callable] = []
        self.resolution_callbacks: List[Callable] = []

        self.logger.info("Deadlock detector initialized")

    async def start_detection(self) -> None:
        """Starts the asynchronous deadlock detection loop."""
        if self.is_detecting:
            self.logger.warning("Deadlock detection is already running.")
            return

        self.is_detecting = True
        self.detection_task = asyncio.create_task(self._detection_loop())
        self.logger.info("Deadlock detection started.")

    async def stop_detection(self) -> None:
        """Stops the asynchronous deadlock detection loop."""
        if not self.is_detecting:
            return

        self.is_detecting = False
        if self.detection_task:
            self.detection_task.cancel()
            try:
                await self.detection_task
            except asyncio.CancelledError:
                pass  # Expected on cancellation
        self.logger.info("Deadlock detection stopped.")

    async def _detection_loop(self) -> None:
        """The main loop that periodically runs deadlock checks."""
        while self.is_detecting:
            try:
                await self._check_for_deadlocks()
                await asyncio.sleep(self.detection_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Error in deadlock detection loop: %s", e, exc_info=True)
                await asyncio.sleep(5.0)  # Cooldown period before retrying

    async def _check_for_deadlocks(self) -> None:
        """Runs a suite of checks for different types of deadlocks."""
        self._cleanup_expired_waits()
        await self._check_circular_waits()
        await self._check_coordination_deadlocks()
        await self._check_resource_deadlocks()
        await self._check_communication_deadlocks()
        await self._check_consensus_deadlocks()
        await self._check_timeout_deadlocks()

    def _cleanup_expired_waits(self) -> None:
        """Removes waiting states that have exceeded their specified timeout."""
        expired_waits = [
            agent_id
            for agent_id, wait_state in self.waiting_states.items()
            if wait_state.is_timed_out()
        ]

        for agent_id in expired_waits:
            self.logger.debug("Removing expired wait for agent: %s", agent_id)
            if agent_id in self.waiting_states:
                del self.waiting_states[agent_id]
            if agent_id in self.agent_dependencies:
                del self.agent_dependencies[agent_id]

    async def _check_circular_waits(self) -> None:
        """Detects deadlocks caused by circular dependencies between agents."""
        dependency_graph = self._build_dependency_graph()
        cycles = self._find_cycles(dependency_graph)

        for cycle in cycles:
            if len(cycle) > 1 and not self._is_existing_deadlock(cycle):
                deadlock_id = f"circular_wait_{self.deadlock_counter}"
                self.deadlock_counter += 1
                await self._handle_detected_deadlock(
                    deadlock_id,
                    DeadlockType.CIRCULAR_WAIT,
                    cycle,
                    metadata={"cycle_length": len(cycle)},
                )

    def _build_dependency_graph(self) -> Dict[str, Set[str]]:
        """Constructs a dependency graph from the current waiting states.

        Returns:
            A dictionary representing the dependency graph, where keys are agent IDs
            and values are sets of agent IDs they depend on.
        """
        graph = defaultdict(set)
        for agent_id, wait_state in self.waiting_states.items():
            if wait_state.wait_type == "agent":
                graph[agent_id].add(wait_state.waiting_for)
            elif wait_state.wait_type == "resource":
                resource_owner = self.resource_owners.get(wait_state.waiting_for)
                if resource_owner and resource_owner != agent_id:
                    graph[agent_id].add(resource_owner)
        return dict(graph)

    def _find_cycles(self, graph: Dict[str, Set[str]]) -> List[List[str]]:
        """Finds all cycles in a directed graph using Depth First Search (DFS).

        Args:
            graph: The dependency graph to search for cycles.

        Returns:
            A list of lists, where each inner list represents a detected cycle of agent IDs.
        """
        cycles = []
        path = []
        recursion_stack = set()
        visited = set()

        def _dfs(node: str):
            """Recursive DFS helper to find cycles."""
            recursion_stack.add(node)
            path.append(node)

            for neighbor in graph.get(node, set()):
                if neighbor in recursion_stack:
                    try:
                        # Cycle detected
                        cycle_start_index = path.index(neighbor)
                        cycles.append(path[cycle_start_index:])
                    except ValueError:
                        # This should not happen in a correct graph traversal
                        pass
                elif neighbor not in visited:
                    _dfs(neighbor)

            path.pop()
            recursion_stack.remove(node)
            visited.add(node)

        for node in list(graph):
            if node not in visited:
                _dfs(node)

        return cycles

    async def _check_coordination_deadlocks(self) -> None:
        """Detects deadlocks related to failed multi-agent coordination."""
        coordination_groups = defaultdict(list)
        for agent_id, wait_state in self.waiting_states.items():
            if wait_state.wait_type == "coordination":
                coord_id = wait_state.metadata.get("coordination_id", "unknown")
                coordination_groups[coord_id].append(agent_id)

        for coord_id, agents in coordination_groups.items():
            if len(agents) > 1 and self._are_agents_mutually_waiting(agents):
                deadlock_id = f"coordination_deadlock_{self.deadlock_counter}"
                self.deadlock_counter += 1
                await self._handle_detected_deadlock(
                    deadlock_id,
                    DeadlockType.COORDINATION_DEADLOCK,
                    agents,
                    metadata={"coordination_id": coord_id},
                )

    async def _check_resource_deadlocks(self) -> None:
        """Detects deadlocks arising from resource contention."""
        resource_waits = defaultdict(list)
        for agent_id, wait_state in self.waiting_states.items():
            if wait_state.wait_type == "resource":
                resource_waits[wait_state.waiting_for].append(agent_id)

        for resource_id, waiting_agents in resource_waits.items():
            owner = self.resource_owners.get(resource_id)
            if owner and owner in self.waiting_states and len(waiting_agents) > 0:
                involved_agents = waiting_agents + [owner]
                if not self._is_existing_deadlock(involved_agents):
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
        """Detects deadlocks where agents are mutually waiting for messages."""
        message_waits = [
            (agent_id, ws)
            for agent_id, ws in self.waiting_states.items()
            if ws.wait_type == "message"
        ]
        for i, (agent1, wait1) in enumerate(message_waits):
            for agent2, wait2 in message_waits[i + 1 :]:
                if wait1.waiting_for == agent2 and wait2.waiting_for == agent1:
                    involved = [agent1, agent2]
                    if not self._is_existing_deadlock(involved):
                        deadlock_id = f"communication_deadlock_{self.deadlock_counter}"
                        self.deadlock_counter += 1
                        await self._handle_detected_deadlock(
                            deadlock_id,
                            DeadlockType.COMMUNICATION_DEADLOCK,
                            involved,
                            metadata={
                                "agent1_waiting_for": wait1.waiting_for,
                                "agent2_waiting_for": wait2.waiting_for,
                            },
                        )

    async def _check_consensus_deadlocks(self) -> None:
        """Detects deadlocks where consensus cannot be reached in time."""
        proposal_groups = defaultdict(list)
        for agent_id, wait_state in self.waiting_states.items():
            if wait_state.wait_type == "consensus":
                proposal_id = wait_state.metadata.get("proposal_id", "unknown")
                proposal_groups[proposal_id].append((agent_id, wait_state))

        for proposal_id, agent_waits in proposal_groups.items():
            if len(agent_waits) > 1:
                max_wait = max(ws.get_wait_duration() for _, ws in agent_waits)
                if max_wait > self.deadlock_timeout:
                    agents = [agent_id for agent_id, _ in agent_waits]
                    if not self._is_existing_deadlock(agents):
                        deadlock_id = f"consensus_deadlock_{self.deadlock_counter}"
                        self.deadlock_counter += 1
                        await self._handle_detected_deadlock(
                            deadlock_id,
                            DeadlockType.CONSENSUS_DEADLOCK,
                            agents,
                            metadata={
                                "proposal_id": proposal_id,
                                "wait_time": max_wait,
                            },
                        )

    async def _check_timeout_deadlocks(self) -> None:
        """Identifies deadlocks based on maximum wait time violations."""
        for agent_id, wait_state in self.waiting_states.items():
            wait_duration = wait_state.get_wait_duration()
            if wait_duration > self.max_wait_time:
                if not self._is_existing_deadlock([agent_id]):
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
        """Checks if a group of agents are all waiting for another agent within the same group.

        Args:
            agents: A list of agent IDs to check for mutual waiting.

        Returns:
            True if a mutual wait condition exists, False otherwise.
        """
        agent_set = set(agents)
        for agent_id in agents:
            wait_state = self.waiting_states.get(agent_id)
            if not wait_state or wait_state.waiting_for not in agent_set:
                return False
        return True

    def _is_existing_deadlock(self, involved_agents: List[str]) -> bool:
        """Checks if an active deadlock with the same set of agents already exists.

        Args:
            involved_agents: A list of agent IDs involved in the potential deadlock.

        Returns:
            True if an identical deadlock is already active, False otherwise.
        """
        agent_set = set(involved_agents)
        return any(
            set(deadlock.involved_agents) == agent_set
            for deadlock in self.active_deadlocks.values()
        )

    async def _handle_detected_deadlock(
        self,
        deadlock_id: str,
        deadlock_type: DeadlockType,
        involved_agents: List[str],
        involved_resources: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Central handler for newly detected deadlocks.

        This method logs the deadlock, creates a `DeadlockEvent`, notifies callbacks,
        and triggers the resolution process.

        Args:
            deadlock_id: The unique ID for the new deadlock.
            deadlock_type: The type of the detected deadlock.
            involved_agents: A list of agent IDs involved.
            involved_resources: An optional list of resource IDs involved.
            metadata: Optional dictionary with additional context.
        """
        event = DeadlockEvent(
            deadlock_id=deadlock_id,
            deadlock_type=deadlock_type,
            involved_agents=involved_agents,
            involved_resources=involved_resources or [],
            metadata=metadata or {},
        )
        self.active_deadlocks[deadlock_id] = event

        self.logger.warning(
            "Deadlock detected: %s (type: %s, agents: %s)",
            deadlock_id,
            deadlock_type.value,
            involved_agents,
        )

        await self._notify_deadlock_callbacks(event)

        if self.enable_auto_resolution:
            await self._resolve_deadlock(event)

    async def _resolve_deadlock(self, deadlock_event: DeadlockEvent) -> None:
        """Attempts to resolve a deadlock using a predefined strategy.

        Args:
            deadlock_event: The `DeadlockEvent` to be resolved.
        """
        strategy_func = self.resolution_strategies.get(deadlock_event.deadlock_type)
        if not strategy_func:
            self.logger.warning(
                "No resolution strategy found for deadlock type: %s",
                deadlock_event.deadlock_type.value,
            )
            return

        try:
            success = await strategy_func(deadlock_event)
            if success:
                deadlock_event.resolved = True
                deadlock_event.resolution_time = datetime.now()
                deadlock_event.resolution_strategy = strategy_func.__name__
                self.logger.info(
                    "Deadlock %s resolved using strategy: %s",
                    deadlock_event.deadlock_id,
                    deadlock_event.resolution_strategy,
                )
                self.deadlock_history.append(deadlock_event)
                del self.active_deadlocks[deadlock_event.deadlock_id]
                await self._notify_resolution_callbacks(deadlock_event)
            else:
                self.logger.warning(
                    "Failed to resolve deadlock %s with strategy %s",
                    deadlock_event.deadlock_id,
                    strategy_func.__name__,
                )
        except Exception as e:
            self.logger.error(
                "Error resolving deadlock %s: %s",
                deadlock_event.deadlock_id,
                e,
                exc_info=True,
            )

    async def _resolve_coordination_deadlock(self, event: DeadlockEvent) -> bool:
        """Resolution strategy for coordination deadlocks."""
        self.logger.info("Attempting to resolve coordination deadlock by resetting state.")
        for agent_id in event.involved_agents:
            self.clear_agent_wait(agent_id)
        return True

    async def _resolve_resource_deadlock(self, event: DeadlockEvent) -> bool:
        """Resolution strategy for resource deadlocks."""
        self.logger.info("Attempting to resolve resource deadlock by releasing resources.")
        for resource_id in event.involved_resources:
            if resource_id in self.resource_owners:
                owner = self.resource_owners[resource_id]
                if owner in event.involved_agents:
                    self.release_resource(resource_id)
        for agent_id in event.involved_agents:
            self.clear_agent_wait(agent_id)
        return True

    async def _resolve_communication_deadlock(self, event: DeadlockEvent) -> bool:
        """Resolution strategy for communication deadlocks."""
        self.logger.info("Attempting to resolve communication deadlock by breaking the cycle.")
        if event.involved_agents:
            # Break the cycle by clearing one agent's wait state
            agent_to_clear = event.involved_agents[0]
            self.clear_agent_wait(agent_to_clear)
        return True

    async def _resolve_consensus_deadlock(self, event: DeadlockEvent) -> bool:
        """Resolution strategy for consensus deadlocks."""
        self.logger.info("Attempting to resolve consensus deadlock by forcing a timeout.")
        proposal_id = event.metadata.get("proposal_id")
        for agent_id in event.involved_agents:
            wait_state = self.waiting_states.get(agent_id)
            if (
                wait_state
                and wait_state.wait_type == "consensus"
                and wait_state.metadata.get("proposal_id") == proposal_id
            ):
                self.clear_agent_wait(agent_id)
        return True

    async def _resolve_circular_wait(self, event: DeadlockEvent) -> bool:
        """Resolution strategy for circular wait deadlocks."""
        self.logger.info("Attempting to resolve circular wait by breaking the dependency cycle.")
        if event.involved_agents:
            # Break the cycle by clearing one agent's dependency
            agent_to_break = event.involved_agents[0]
            self.clear_agent_wait(agent_to_break)
        return True

    async def _notify_deadlock_callbacks(self, deadlock_event: DeadlockEvent) -> None:
        """Notifies registered callbacks about a new deadlock."""
        for callback in self.deadlock_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(deadlock_event)
                else:
                    callback(deadlock_event)
            except Exception as e:
                self.logger.error("Error in deadlock callback: %s", e, exc_info=True)

    async def _notify_resolution_callbacks(self, deadlock_event: DeadlockEvent) -> None:
        """Notifies registered callbacks about a resolved deadlock."""
        for callback in self.resolution_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(deadlock_event)
                else:
                    callback(deadlock_event)
            except Exception as e:
                self.logger.error("Error in resolution callback: %s", e, exc_info=True)

    # Public Interface Methods

    def record_agent_wait(
        self,
        agent_id: str,
        waiting_for: str,
        wait_type: str,
        timeout: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Records that an agent has entered a waiting state.

        Args:
            agent_id: The ID of the agent that is waiting.
            waiting_for: The ID of the resource or agent being waited for.
            wait_type: A string indicating the type of wait (e.g., 'resource', 'agent').
            timeout: An optional timeout in seconds for this wait.
            metadata: Optional dictionary for extra data.
        """
        self.waiting_states[agent_id] = WaitingState(
            agent_id=agent_id,
            waiting_for=waiting_for,
            wait_type=wait_type,
            timeout=timeout,
            metadata=metadata or {},
        )
        if wait_type == "agent":
            self.agent_dependencies[agent_id].add(waiting_for)
        self.logger.debug(
            "Recorded wait: agent %s waiting for %s (%s)",
            agent_id,
            waiting_for,
            wait_type,
        )

    def clear_agent_wait(self, agent_id: str) -> None:
        """Clears an agent's waiting state, typically when the wait is over.

        Args:
            agent_id: The ID of the agent whose wait state should be cleared.
        """
        if agent_id in self.waiting_states:
            del self.waiting_states[agent_id]
        if agent_id in self.agent_dependencies:
            del self.agent_dependencies[agent_id]
        self.logger.debug("Cleared wait for agent: %s", agent_id)

    def record_resource_ownership(self, resource_id: str, owner_agent_id: str) -> None:
        """Records that a resource is now owned by a specific agent.

        Args:
            resource_id: The ID of the resource.
            owner_agent_id: The ID of the agent that now owns the resource.
        """
        self.resource_owners[resource_id] = owner_agent_id
        self.logger.debug("Resource %s owned by agent %s", resource_id, owner_agent_id)

    def release_resource(self, resource_id: str) -> None:
        """Records that a resource has been released.

        Args:
            resource_id: The ID of the resource to release.
        """
        if resource_id in self.resource_owners:
            owner = self.resource_owners.pop(resource_id)
            self.logger.debug("Released resource %s from agent %s", resource_id, owner)

    def get_active_deadlocks(self) -> List[DeadlockEvent]:
        """Returns a list of all currently active (unresolved) deadlocks.

        Returns:
            A list of `DeadlockEvent` objects.
        """
        return list(self.active_deadlocks.values())

    def get_deadlock_history(self) -> List[DeadlockEvent]:
        """Returns a list of all resolved deadlocks.

        Returns:
            A list of `DeadlockEvent` objects.
        """
        return self.deadlock_history.copy()

    def get_waiting_agents(self) -> List[str]:
        """Returns a list of IDs of all agents currently in a waiting state.

        Returns:
            A list of agent ID strings.
        """
        return list(self.waiting_states.keys())

    def get_deadlock_statistics(self) -> Dict[str, Any]:
        """Provides a summary of deadlock statistics.

        Returns:
            A dictionary containing statistics such as total deadlocks, resolution rate,
            and average resolution time.
        """
        total_deadlocks = len(self.deadlock_history) + len(self.active_deadlocks)
        resolved_deadlocks = len(self.deadlock_history)
        type_counts = defaultdict(int)
        all_deadlocks = self.deadlock_history + list(self.active_deadlocks.values())
        for deadlock in all_deadlocks:
            type_counts[deadlock.deadlock_type.value] += 1

        resolution_times = [
            (d.resolution_time - d.detection_time).total_seconds()
            for d in self.deadlock_history
            if d.resolution_time
        ]
        avg_resolution_time = (
            sum(resolution_times) / len(resolution_times) if resolution_times else 0.0
        )

        return {
            "total_deadlocks": total_deadlocks,
            "active_deadlocks": len(self.active_deadlocks),
            "resolved_deadlocks": resolved_deadlocks,
            "resolution_rate": (
                resolved_deadlocks / total_deadlocks if total_deadlocks > 0 else 1.0
            ),
            "average_resolution_time": avg_resolution_time,
            "deadlocks_by_type": dict(type_counts),
            "currently_waiting_agents": len(self.waiting_states),
        }

    def add_deadlock_callback(self, callback: Callable[[DeadlockEvent], None]) -> None:
        """Registers a callback function to be invoked when a deadlock is detected.

        Args:
            callback: The function to call. It can be synchronous or asynchronous.
        """
        self.deadlock_callbacks.append(callback)

    def add_resolution_callback(self, callback: Callable[[DeadlockEvent], None]) -> None:
        """Registers a callback function to be invoked when a deadlock is resolved.

        Args:
            callback: The function to call. It can be synchronous or asynchronous.
        """
        self.resolution_callbacks.append(callback)

    def remove_deadlock_callback(self, callback: Callable[[DeadlockEvent], None]) -> None:
        """Removes a previously registered deadlock detection callback.

        Args:
            callback: The callback function to remove.
        """
        try:
            self.deadlock_callbacks.remove(callback)
        except ValueError:
            self.logger.warning("Attempted to remove a non-existent deadlock callback.")

    def remove_resolution_callback(self, callback: Callable[[DeadlockEvent], None]) -> None:
        """Removes a previously registered deadlock resolution callback.

        Args:
            callback: The callback function to remove.
        """
        try:
            self.resolution_callbacks.remove(callback)
        except ValueError:
            self.logger.warning("Attempted to remove a non-existent resolution callback.")

    async def shutdown(self) -> None:
        """Gracefully shuts down the deadlock detector.

        Stops the detection loop and clears all internal states.
        """
        await self.stop_detection()
        self.waiting_states.clear()
        self.resource_owners.clear()
        self.agent_dependencies.clear()
        self.active_deadlocks.clear()
        self.deadlock_callbacks.clear()
        self.resolution_callbacks.clear()
        self.logger.info("Deadlock detector shutdown complete.")
