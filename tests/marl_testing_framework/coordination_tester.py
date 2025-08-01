"""Coordination Testing Framework for MARL.

This module provides specialized testing capabilities for MARL coordination
mechanisms, including conflict resolution, consensus building, and communication.
"""

# Standard Library
import asyncio
import random
import statistics
import time
import uuid

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

# SynThesisAI Modules
from utils.logging_config import get_logger


class CoordinationTestType(Enum):
    """Coordination test type enumeration."""

    CONSENSUS_BUILDING = "consensus_building"
    CONFLICT_RESOLUTION = "conflict_resolution"
    COMMUNICATION_RELIABILITY = "communication_reliability"
    DEADLOCK_PREVENTION = "deadlock_prevention"
    SCALABILITY = "scalability"
    FAULT_TOLERANCE = "fault_tolerance"


class ConflictScenario(Enum):
    """Conflict scenario enumeration."""

    SIMPLE_DISAGREEMENT = "simple_disagreement"
    RESOURCE_COMPETITION = "resource_competition"
    PRIORITY_CONFLICT = "priority_conflict"
    CIRCULAR_DEPENDENCY = "circular_dependency"
    TIMING_CONFLICT = "timing_conflict"


@dataclass
class CoordinationTestConfig:
    """
    Configuration for coordination testing.

    Args:
        test_type: Type of test to run.
        num_agents: Number of agents participating in the test.
        num_test_rounds: Total test rounds to execute.
        timeout_per_round: Timeout duration (seconds) per test round.
        conflict_scenario: Scenario type for conflict simulation.
        conflict_probability: Probability of conflict occurrence (0-1).
        conflict_intensity: Intensity level for generated conflicts (0-1).
        consensus_threshold: Minimum consensus score to declare success.
        max_consensus_rounds: Maximum rounds for consensus building.
        consensus_timeout: Timeout duration (seconds) for consensus process.
        message_loss_probability: Probability of losing a message (0-1).
        message_delay_range: Tuple specifying min and max message delay (seconds).
        network_partition_probability: Probability of partitioning two agents.
        min_success_rate: Minimum success rate threshold for tests.
        max_average_resolution_time: Maximum average resolution time threshold (seconds).
        max_communication_overhead: Maximum allowed communication overhead.
        max_agents_for_scalability: Max agents to test scalability.
        scalability_step_size: Step size to increase agents per scalability test.
    """

    # Test settings
    test_type: CoordinationTestType = CoordinationTestType.CONSENSUS_BUILDING
    num_agents: int = 3
    num_test_rounds: int = 50
    timeout_per_round: float = 10.0

    # Conflict settings
    conflict_scenario: ConflictScenario = ConflictScenario.SIMPLE_DISAGREEMENT
    conflict_probability: float = 0.3
    conflict_intensity: float = 0.5

    # Consensus settings
    consensus_threshold: float = 0.7
    max_consensus_rounds: int = 10
    consensus_timeout: float = 5.0

    # Communication settings
    message_loss_probability: float = 0.05
    message_delay_range: Tuple[float, float] = (0.01, 0.1)
    network_partition_probability: float = 0.02

    # Success criteria
    min_success_rate: float = 0.85
    max_average_resolution_time: float = 3.0
    max_communication_overhead: float = 0.2

    # Scalability settings
    max_agents_for_scalability: int = 20
    scalability_step_size: int = 2

    def __post_init__(self) -> None:
        """
        Validate configuration.

        Raises:
            ValueError: If num_agents < 2 or conflict_probability not between 0 and 1.
        """
        if self.num_agents < 2:
            raise ValueError("Need at least 2 agents for coordination testing")
        if not (0 <= self.conflict_probability <= 1):
            raise ValueError("Conflict probability must be between 0 and 1")


@dataclass
class CoordinationTestResult:
    """Result of a coordination test."""

    test_id: str
    test_type: CoordinationTestType
    success: bool
    resolution_time: float
    num_rounds: int
    consensus_achieved: bool
    communication_overhead: float
    agent_satisfaction: Dict[str, float]
    conflict_resolved: bool
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "test_id": self.test_id,
            "test_type": self.test_type.value,
            "success": self.success,
            "resolution_time": self.resolution_time,
            "num_rounds": self.num_rounds,
            "consensus_achieved": self.consensus_achieved,
            "communication_overhead": self.communication_overhead,
            "agent_satisfaction": self.agent_satisfaction,
            "conflict_resolved": self.conflict_resolved,
            "error_message": self.error_message,
        }


class MockAgent:
    """Mock agent for coordination testing."""

    def __init__(self, agent_id: str, agent_type: str = "default") -> None:
        """
        Initialize mock agent.

        Args:
            agent_id: Unique identifier for the agent.
            agent_type: Type/category of the agent.
        """
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.preferences = self._generate_preferences()
        self.satisfaction = 0.0
        self.messages_sent = 0
        self.messages_received = 0
        self.is_active = True

    def _generate_preferences(self) -> Dict[str, float]:
        """
        Generate random preferences for testing.

        Returns:
            A dictionary mapping preference names to values.
        """
        return {
            "action_preference": random.uniform(0, 1),
            "priority_weight": random.uniform(0.1, 1.0),
            "cooperation_tendency": random.uniform(0.3, 0.9),
            "compromise_willingness": random.uniform(0.2, 0.8),
        }

    async def propose_action(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Propose an action based on the provided context.

        Args:
            context: Contextual information for action proposal.

        Returns:
            A proposal dictionary containing action details.
        """
        # Simulate decision making
        await asyncio.sleep(random.uniform(0.01, 0.05))

        proposal = {
            "agent_id": self.agent_id,
            "action": random.randint(0, 10),
            "confidence": random.uniform(0.5, 1.0),
            "priority": self.preferences["priority_weight"],
            "reasoning": f"Agent {self.agent_id} proposal based on preferences",
        }

        self.messages_sent += 1
        return proposal

    async def evaluate_proposal(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate another agent's proposal.

        Args:
            proposal: Proposal dictionary from another agent.

        Returns:
            An evaluation dictionary with satisfaction and support info.
        """
        await asyncio.sleep(random.uniform(0.005, 0.02))

        # Calculate satisfaction based on preferences
        action_diff = abs(proposal.get("action", 0) - self.preferences["action_preference"] * 10)
        satisfaction = max(0.0, 1.0 - (action_diff / 10.0))

        # Apply cooperation tendency
        satisfaction *= self.preferences["cooperation_tendency"]

        evaluation = {
            "evaluator_id": self.agent_id,
            "proposal_id": proposal.get("agent_id"),
            "satisfaction": satisfaction,
            "support": satisfaction > 0.5,
            "suggested_modifications": [] if satisfaction > 0.7 else ["reduce_action_value"],
        }

        self.messages_sent += 1
        return evaluation

    async def receive_message(self, message: Dict[str, Any]) -> None:
        """
        Receive and process a message.

        Args:
            message: The message dictionary received.
        """
        self.messages_received += 1

        # Update satisfaction based on message content
        if "final_decision" in message:
            decision = message["final_decision"]
            action_diff = abs(
                decision.get("action", 0) - self.preferences["action_preference"] * 10
            )
            self.satisfaction = max(0.0, 1.0 - (action_diff / 10.0))

    def get_status(self) -> Dict[str, Any]:
        """
        Get agent status.

        Returns:
            A dictionary containing current status of the agent.
        """
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "is_active": self.is_active,
            "satisfaction": self.satisfaction,
            "messages_sent": self.messages_sent,
            "messages_received": self.messages_received,
            "preferences": self.preferences,
        }


class CoordinationTester:
    """Coordination testing framework for MARL.

    Provides comprehensive testing of coordination mechanisms including
    consensus building, conflict resolution, and communication reliability.
    """

    def __init__(self, config: CoordinationTestConfig) -> None:
        """
        Initialize coordination tester.

        Args:
            config: Coordination test configuration.
        """
        self.config = config
        self.logger = get_logger(__name__)

        # Test state
        self.test_results: List[CoordinationTestResult] = []
        self.current_test_id: Optional[str] = None

        # Mock agents
        self.agents: Dict[str, MockAgent] = {}

        # Communication simulation
        self.message_queue: List[Dict[str, Any]] = []
        self.network_partitions: List[Tuple[str, str]] = []  # Pairs of disconnected agents

        self.logger.info("Coordination tester initialized")

    def _create_mock_agents(self) -> None:
        """Create mock agents for testing."""
        self.agents.clear()

        agent_types = ["generator", "validator", "curriculum"]

        for i in range(self.config.num_agents):
            agent_id = f"agent_{i}"
            agent_type = agent_types[i % len(agent_types)]

            self.agents[agent_id] = MockAgent(agent_id, agent_type)

        self.logger.debug("Created %d mock agents", len(self.agents))

    async def run_coordination_test(
        self, coordination_system: Optional[Any] = None
    ) -> CoordinationTestResult:
        """
        Run a single coordination test.

        Args:
            coordination_system: Optional coordination system to test.

        Returns:
            CoordinationTestResult: Result object for this test.
        """
        test_id = str(uuid.uuid4())
        self.current_test_id = test_id

        self.logger.info("Starting coordination test: %s", test_id)

        start_time = time.time()

        try:
            # Create mock agents
            self._create_mock_agents()

            # Run test based on type
            if self.config.test_type == CoordinationTestType.CONSENSUS_BUILDING:
                result = await self._test_consensus_building()
            elif self.config.test_type == CoordinationTestType.CONFLICT_RESOLUTION:
                result = await self._test_conflict_resolution()
            elif self.config.test_type == CoordinationTestType.COMMUNICATION_RELIABILITY:
                result = await self._test_communication_reliability()
            elif self.config.test_type == CoordinationTestType.DEADLOCK_PREVENTION:
                result = await self._test_deadlock_prevention()
            elif self.config.test_type == CoordinationTestType.SCALABILITY:
                result = await self._test_scalability()
            elif self.config.test_type == CoordinationTestType.FAULT_TOLERANCE:
                result = await self._test_fault_tolerance()
            else:
                raise ValueError(f"Unknown test type: {self.config.test_type}")

            result.test_id = test_id
            result.resolution_time = time.time() - start_time

            self.test_results.append(result)

            self.logger.info(
                "Coordination test completed: %s (%.3fs)",
                test_id,
                result.resolution_time,
            )

            return result

        except Exception as e:
            error_result = CoordinationTestResult(
                test_id=test_id,
                test_type=self.config.test_type,
                success=False,
                resolution_time=time.time() - start_time,
                num_rounds=0,
                consensus_achieved=False,
                communication_overhead=0.0,
                agent_satisfaction={},
                conflict_resolved=False,
                error_message=str(e),
            )

            self.test_results.append(error_result)

            self.logger.error("Coordination test failed: %s", str(e))
            return error_result

    async def _test_consensus_building(self) -> CoordinationTestResult:
        """
        Test consensus building mechanism.

        Returns:
            CoordinationTestResult: Result of the consensus building test.
        """
        self.logger.debug("Testing consensus building")

        consensus_achieved = False
        num_rounds = 0
        total_messages = 0

        # Initial proposals
        proposals = {}
        for agent_id, agent in self.agents.items():
            context = {"round": 0, "other_agents": list(self.agents.keys())}
            proposal = await agent.propose_action(context)
            proposals[agent_id] = proposal
            total_messages += 1

        # Consensus rounds
        for round_num in range(self.config.max_consensus_rounds):
            num_rounds += 1

            # Evaluate proposals
            evaluations = {}
            for agent_id, agent in self.agents.items():
                agent_evaluations = []
                for proposal_id, proposal in proposals.items():
                    if proposal_id != agent_id:  # Don't evaluate own proposal
                        evaluation = await agent.evaluate_proposal(proposal)
                        agent_evaluations.append(evaluation)
                        total_messages += 1

                evaluations[agent_id] = agent_evaluations

            # Check for consensus
            consensus_score = self._calculate_consensus_score(proposals, evaluations)

            if consensus_score >= self.config.consensus_threshold:
                consensus_achieved = True
                break

            # Generate new proposals based on feedback
            new_proposals = {}
            for agent_id, agent in self.agents.items():
                # Modify proposal based on feedback
                context = {
                    "round": round_num + 1,
                    "evaluations": evaluations,
                    "consensus_score": consensus_score,
                }
                new_proposal = await agent.propose_action(context)
                new_proposals[agent_id] = new_proposal
                total_messages += 1

            proposals = new_proposals

        # Calculate final metrics
        agent_satisfaction = {
            agent_id: agent.satisfaction for agent_id, agent in self.agents.items()
        }

        communication_overhead = total_messages / (self.config.num_agents * num_rounds)

        return CoordinationTestResult(
            test_id="",  # Will be set by caller
            test_type=self.config.test_type,
            success=consensus_achieved,
            resolution_time=0.0,  # Will be set by caller
            num_rounds=num_rounds,
            consensus_achieved=consensus_achieved,
            communication_overhead=communication_overhead,
            agent_satisfaction=agent_satisfaction,
            conflict_resolved=consensus_achieved,
        )

    def _calculate_consensus_score(
        self,
        proposals: Dict[str, Dict[str, Any]],
        evaluations: Dict[str, List[Dict[str, Any]]],
    ) -> float:
        """
        Calculate consensus score from proposals and evaluations.

        Args:
            proposals: Mapping of agent IDs to their proposals.
            evaluations: Mapping of agent IDs to lists of evaluation results.

        Returns:
            float: Consensus score (0.0 to 1.0).
        """
        if not evaluations:
            return 0.0

        total_satisfaction = 0.0
        total_evaluations = 0

        for agent_evaluations in evaluations.values():
            for evaluation in agent_evaluations:
                total_satisfaction += evaluation.get("satisfaction", 0.0)
                total_evaluations += 1

        return total_satisfaction / max(1, total_evaluations)

    async def _test_conflict_resolution(self) -> CoordinationTestResult:
        """Test conflict resolution mechanism."""
        self.logger.debug("Testing conflict resolution")

        # Create conflicting scenario
        conflict_created = await self._create_conflict_scenario()

        if not conflict_created:
            return CoordinationTestResult(
                test_id="",
                test_type=self.config.test_type,
                success=False,
                resolution_time=0.0,
                num_rounds=0,
                consensus_achieved=False,
                communication_overhead=0.0,
                agent_satisfaction={},
                conflict_resolved=False,
                error_message="Failed to create conflict scenario",
            )

        # Attempt conflict resolution
        resolution_start = time.time()
        conflict_resolved = False
        num_rounds = 0
        total_messages = 0

        # Conflict resolution rounds
        for round_num in range(self.config.max_consensus_rounds):
            num_rounds += 1

            # Each agent proposes resolution
            resolutions = {}
            for agent_id, agent in self.agents.items():
                if agent.is_active:
                    context = {
                        "conflict_round": round_num,
                        "conflict_type": self.config.conflict_scenario,
                    }
                    resolution = await agent.propose_action(context)
                    resolutions[agent_id] = resolution
                    total_messages += 1

            # Evaluate resolutions
            resolution_scores = []
            for agent_id, agent in self.agents.items():
                if agent.is_active:
                    for resolution_id, resolution in resolutions.items():
                        if resolution_id != agent_id:
                            evaluation = await agent.evaluate_proposal(resolution)
                            resolution_scores.append(evaluation.get("satisfaction", 0.0))
                            total_messages += 1

            # Check if conflict is resolved
            if resolution_scores:
                avg_satisfaction = sum(resolution_scores) / len(resolution_scores)
                if avg_satisfaction >= self.config.consensus_threshold:
                    conflict_resolved = True
                    break

        resolution_time = time.time() - resolution_start

        # Calculate metrics
        agent_satisfaction = {
            agent_id: agent.satisfaction for agent_id, agent in self.agents.items()
        }

        communication_overhead = total_messages / (
            len([a for a in self.agents.values() if a.is_active]) * num_rounds
        )

        return CoordinationTestResult(
            test_id="",
            test_type=self.config.test_type,
            success=conflict_resolved,
            resolution_time=resolution_time,
            num_rounds=num_rounds,
            consensus_achieved=conflict_resolved,
            communication_overhead=communication_overhead,
            agent_satisfaction=agent_satisfaction,
            conflict_resolved=conflict_resolved,
        )

    async def _create_conflict_scenario(self) -> bool:
        """
        Create a conflict scenario based on configuration.

        Returns:
            bool: True if conflict scenario was successfully created, False otherwise.
        """
        try:
            if self.config.conflict_scenario == ConflictScenario.SIMPLE_DISAGREEMENT:
                # Make agents have opposing preferences
                agent_list = list(self.agents.values())
                for i, agent in enumerate(agent_list):
                    agent.preferences["action_preference"] = i / len(agent_list)
                    agent.preferences["cooperation_tendency"] *= 1 - self.config.conflict_intensity

            elif self.config.conflict_scenario == ConflictScenario.RESOURCE_COMPETITION:
                # All agents want the same resource
                for agent in self.agents.values():
                    agent.preferences["action_preference"] = 0.5  # Same preference
                    agent.preferences["priority_weight"] = random.uniform(0.8, 1.0)  # High priority

            elif self.config.conflict_scenario == ConflictScenario.PRIORITY_CONFLICT:
                # Agents have conflicting priorities
                priorities = [1.0, 0.9, 0.8]
                for i, agent in enumerate(self.agents.values()):
                    agent.preferences["priority_weight"] = priorities[i % len(priorities)]
                    agent.preferences["compromise_willingness"] *= (
                        1 - self.config.conflict_intensity
                    )

            return True

        except Exception as e:
            self.logger.error("Failed to create conflict scenario: %s", str(e))
            return False

    async def _test_communication_reliability(self) -> CoordinationTestResult:
        """Test communication reliability under network issues."""
        self.logger.debug("Testing communication reliability")

        # Simulate network issues
        self._simulate_network_issues()

        # Run communication test
        successful_messages = 0
        total_messages = 0

        for round_num in range(self.config.num_test_rounds):
            # Each agent sends a message to all others
            for sender_id, sender in self.agents.items():
                for receiver_id, receiver in self.agents.items():
                    if sender_id != receiver_id:
                        total_messages += 1

                        # Check if message should be lost
                        if random.random() > self.config.message_loss_probability:
                            # Check for network partition
                            if not self._is_partitioned(sender_id, receiver_id):
                                # Simulate message delay
                                delay = random.uniform(*self.config.message_delay_range)
                                await asyncio.sleep(delay)

                                # Deliver message
                                message = {
                                    "sender": sender_id,
                                    "content": f"message_{round_num}",
                                }
                                await receiver.receive_message(message)
                                successful_messages += 1

        # Calculate reliability
        reliability = successful_messages / max(1, total_messages)

        return CoordinationTestResult(
            test_id="",
            test_type=self.config.test_type,
            success=reliability >= 0.9,  # 90% reliability threshold
            resolution_time=0.0,
            num_rounds=self.config.num_test_rounds,
            consensus_achieved=reliability >= 0.95,
            communication_overhead=1.0 - reliability,
            agent_satisfaction={
                agent_id: 1.0 if reliability >= 0.9 else 0.5 for agent_id in self.agents
            },
            conflict_resolved=True,
        )

    def _simulate_network_issues(self) -> None:
        """Simulate network issues like partitions and message loss."""
        # Create random network partitions
        agent_ids = list(self.agents.keys())
        for i in range(len(agent_ids)):
            for j in range(i + 1, len(agent_ids)):
                if random.random() < self.config.network_partition_probability:
                    self.network_partitions.append((agent_ids[i], agent_ids[j]))

    def _is_partitioned(self, agent1_id: str, agent2_id: str) -> bool:
        """
        Check if two agents are partitioned.

        Args:
            agent1_id: Identifier of first agent.
            agent2_id: Identifier of second agent.

        Returns:
            bool: True if agents are in a network partition, False otherwise.
        """
        return (agent1_id, agent2_id) in self.network_partitions or (
            agent2_id,
            agent1_id,
        ) in self.network_partitions

    async def _test_deadlock_prevention(self) -> CoordinationTestResult:
        """
        Test deadlock prevention mechanism.

        Returns:
            CoordinationTestResult: Result of deadlock prevention test.
        """
        self.logger.debug("Testing deadlock prevention")

        # Create potential deadlock scenario
        deadlock_resolved = False
        num_rounds = 0

        # Simulate circular waiting
        agent_list = list(self.agents.keys())
        waiting_for = {}

        for i, agent_id in enumerate(agent_list):
            next_agent = agent_list[(i + 1) % len(agent_list)]
            waiting_for[agent_id] = next_agent

        # Deadlock detection and resolution
        for round_num in range(self.config.max_consensus_rounds):
            num_rounds += 1

            # Check for deadlock
            if self._detect_deadlock(waiting_for):
                # Deadlock detected

                # Attempt resolution by breaking one dependency
                victim_agent = random.choice(agent_list)
                if victim_agent in waiting_for:
                    del waiting_for[victim_agent]
                    deadlock_resolved = True
                    break

            # Simulate some progress
            if random.random() < 0.3:  # 30% chance to resolve dependency
                if waiting_for:
                    resolved_agent = random.choice(list(waiting_for.keys()))
                    del waiting_for[resolved_agent]

            if not waiting_for:  # All dependencies resolved
                deadlock_resolved = True
                break

        return CoordinationTestResult(
            test_id="",
            test_type=self.config.test_type,
            success=deadlock_resolved,
            resolution_time=0.0,
            num_rounds=num_rounds,
            consensus_achieved=deadlock_resolved,
            communication_overhead=0.1,  # Low overhead for deadlock detection
            agent_satisfaction={
                agent_id: 1.0 if deadlock_resolved else 0.0 for agent_id in self.agents
            },
            conflict_resolved=deadlock_resolved,
        )

    def _detect_deadlock(self, waiting_for: Dict[str, str]) -> bool:
        """
        Detect circular dependencies (deadlock).

        Args:
            waiting_for: Mapping of agent ID to the ID it is waiting for.

        Returns:
            bool: True if a cycle (deadlock) is detected, False otherwise.
        """
        if not waiting_for:
            return False

        # Simple cycle detection
        visited = set()

        for start_agent in waiting_for:
            if start_agent in visited:
                continue

            current = start_agent
            path = set()

            while current in waiting_for:
                if current in path:
                    return True  # Cycle detected

                path.add(current)
                visited.add(current)
                current = waiting_for[current]

        return False

    async def _test_scalability(self) -> CoordinationTestResult:
        """
        Test coordination scalability with increasing number of agents.

        Returns:
            CoordinationTestResult: Scalability test results and metrics.
        """
        self.logger.debug("Testing scalability")

        scalability_results = []

        for num_agents in range(
            2,
            self.config.max_agents_for_scalability + 1,
            self.config.scalability_step_size,
        ):
            # Create agents for this test
            test_agents = {}
            for i in range(num_agents):
                agent_id = f"scale_agent_{i}"
                test_agents[agent_id] = MockAgent(agent_id)

            # Measure coordination time
            start_time = time.time()

            # Simple coordination task
            proposals = {}
            for agent_id, agent in test_agents.items():
                proposal = await agent.propose_action({"scale_test": True})
                proposals[agent_id] = proposal

            coordination_time = time.time() - start_time

            scalability_results.append(
                {
                    "num_agents": num_agents,
                    "coordination_time": coordination_time,
                    "time_per_agent": coordination_time / num_agents,
                }
            )

        # Analyze scalability
        if len(scalability_results) >= 2:
            # Check if coordination time grows linearly or worse
            time_growth_rate = (
                scalability_results[-1]["coordination_time"]
                - scalability_results[0]["coordination_time"]
            ) / (scalability_results[-1]["num_agents"] - scalability_results[0]["num_agents"])

            # Good scalability if time growth is sub-linear
            good_scalability = time_growth_rate < 0.1  # Less than 0.1 seconds per additional agent
        else:
            good_scalability = True

        return CoordinationTestResult(
            test_id="",
            test_type=self.config.test_type,
            success=good_scalability,
            resolution_time=sum(r["coordination_time"] for r in scalability_results),
            num_rounds=len(scalability_results),
            consensus_achieved=good_scalability,
            communication_overhead=time_growth_rate if len(scalability_results) >= 2 else 0.0,
            agent_satisfaction={"scalability_score": 1.0 if good_scalability else 0.5},
            conflict_resolved=True,
        )

    async def _test_fault_tolerance(self) -> CoordinationTestResult:
        """
        Test fault tolerance with agent failures.

        Returns:
            CoordinationTestResult: Result of fault tolerance test.
        """
        self.logger.debug("Testing fault tolerance")

        # Randomly fail some agents
        num_failures = max(1, len(self.agents) // 3)  # Fail 1/3 of agents
        failed_agents = random.sample(list(self.agents.keys()), num_failures)

        for agent_id in failed_agents:
            self.agents[agent_id].is_active = False

        active_agents = {aid: agent for aid, agent in self.agents.items() if agent.is_active}

        # Test if remaining agents can still coordinate
        if len(active_agents) < 2:
            return CoordinationTestResult(
                test_id="",
                test_type=self.config.test_type,
                success=False,
                resolution_time=0.0,
                num_rounds=0,
                consensus_achieved=False,
                communication_overhead=0.0,
                agent_satisfaction={},
                conflict_resolved=False,
                error_message="Too many agent failures",
            )

        # Run coordination with remaining agents
        coordination_successful = True
        num_rounds = 0

        try:
            # Simple coordination test
            proposals = {}
            for agent_id, agent in active_agents.items():
                proposal = await agent.propose_action({"fault_tolerance_test": True})
                proposals[agent_id] = proposal
                num_rounds += 1

            # Check if coordination is still possible
            if len(proposals) >= 2:
                # Calculate agreement
                actions = [p.get("action", 0) for p in proposals.values()]
                action_std = statistics.stdev(actions) if len(actions) > 1 else 0.0
                coordination_successful = action_std < 3.0  # Reasonable agreement

        except Exception:
            coordination_successful = False

        return CoordinationTestResult(
            test_id="",
            test_type=self.config.test_type,
            success=coordination_successful,
            resolution_time=0.0,
            num_rounds=num_rounds,
            consensus_achieved=coordination_successful,
            communication_overhead=num_failures / len(self.agents),
            agent_satisfaction={
                aid: 1.0 if coordination_successful else 0.0 for aid in active_agents
            },
            conflict_resolved=coordination_successful,
        )

    async def run_comprehensive_test_suite(self) -> Dict[str, Any]:
        """Run comprehensive coordination test suite.

        Returns:
            Comprehensive test results
        """
        self.logger.info("Running comprehensive coordination test suite")

        test_suite_results = {}
        overall_success = True

        # Test all coordination types
        test_types = [
            CoordinationTestType.CONSENSUS_BUILDING,
            CoordinationTestType.CONFLICT_RESOLUTION,
            CoordinationTestType.COMMUNICATION_RELIABILITY,
            CoordinationTestType.DEADLOCK_PREVENTION,
            CoordinationTestType.SCALABILITY,
            CoordinationTestType.FAULT_TOLERANCE,
        ]

        for test_type in test_types:
            # Update config for this test type
            original_test_type = self.config.test_type
            self.config.test_type = test_type

            try:
                # Run multiple rounds of this test type
                test_results = []
                for _ in range(5):  # 5 rounds per test type
                    result = await self.run_coordination_test()
                    test_results.append(result)

                # Analyze results for this test type
                success_rate = sum(1 for r in test_results if r.success) / len(test_results)
                avg_resolution_time = sum(r.resolution_time for r in test_results) / len(
                    test_results
                )
                avg_communication_overhead = sum(
                    r.communication_overhead for r in test_results
                ) / len(test_results)

                test_type_result = {
                    "test_type": test_type.value,
                    "success_rate": success_rate,
                    "average_resolution_time": avg_resolution_time,
                    "average_communication_overhead": avg_communication_overhead,
                    "meets_criteria": (
                        success_rate >= self.config.min_success_rate
                        and avg_resolution_time <= self.config.max_average_resolution_time
                        and avg_communication_overhead <= self.config.max_communication_overhead
                    ),
                    "individual_results": [r.to_dict() for r in test_results],
                }

                test_suite_results[test_type.value] = test_type_result

                if not test_type_result["meets_criteria"]:
                    overall_success = False

            except Exception as e:
                self.logger.error("Test type %s failed: %s", test_type.value, str(e))
                test_suite_results[test_type.value] = {
                    "test_type": test_type.value,
                    "success_rate": 0.0,
                    "error": str(e),
                    "meets_criteria": False,
                }
                overall_success = False

            finally:
                # Restore original config
                self.config.test_type = original_test_type

        # Generate overall summary
        summary = {
            "overall_success": overall_success,
            "total_test_types": len(test_types),
            "successful_test_types": sum(
                1 for r in test_suite_results.values() if r.get("meets_criteria", False)
            ),
            "test_results": test_suite_results,
            "recommendations": self._generate_coordination_recommendations(test_suite_results),
        }

        self.logger.info(
            "Comprehensive coordination test suite completed: %s",
            "PASS" if overall_success else "FAIL",
        )

        return summary

    def _generate_coordination_recommendations(self, test_results: Dict[str, Any]) -> List[str]:
        """
        Generate recommendations based on test results.

        Args:
            test_results: Aggregated test suite results mapping test types to metrics.

        Returns:
            List[str]: List of recommendations for improvement or confirmation.
        """
        recommendations = []

        for test_type, result in test_results.items():
            if not result.get("meets_criteria", False):
                if test_type == "consensus_building":
                    recommendations.append(
                        "Improve consensus algorithms and reduce consensus timeout"
                    )
                elif test_type == "conflict_resolution":
                    recommendations.append("Enhance conflict detection and resolution strategies")
                elif test_type == "communication_reliability":
                    recommendations.append(
                        "Implement message retry mechanisms and improve network resilience"
                    )
                elif test_type == "deadlock_prevention":
                    recommendations.append("Add deadlock detection and prevention mechanisms")
                elif test_type == "scalability":
                    recommendations.append(
                        "Optimize coordination algorithms for better scalability"
                    )
                elif test_type == "fault_tolerance":
                    recommendations.append("Improve fault detection and recovery mechanisms")

        if not recommendations:
            recommendations.append("All coordination tests passed. System is performing well.")

        return recommendations

    def get_test_summary(self) -> Dict[str, Any]:
        """Get summary of all coordination tests.

        Returns:
            Test summary
        """
        if not self.test_results:
            return {"total_tests": 0, "message": "No tests run yet"}

        total_tests = len(self.test_results)
        successful_tests = sum(1 for r in self.test_results if r.success)

        avg_resolution_time = sum(r.resolution_time for r in self.test_results) / total_tests
        avg_communication_overhead = (
            sum(r.communication_overhead for r in self.test_results) / total_tests
        )

        return {
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "success_rate": successful_tests / total_tests,
            "average_resolution_time": avg_resolution_time,
            "average_communication_overhead": avg_communication_overhead,
            "test_types_covered": list(set(r.test_type.value for r in self.test_results)),
        }

    def clear_test_results(self) -> None:
        """
        Clear all test results and reset internal state.

        This empties test_results, resets current_test_id, and clears network partitions and message queue.
        """
        self.test_results.clear()
        self.current_test_id = None
        self.network_partitions.clear()
        self.message_queue.clear()

        self.logger.info("Coordination test results cleared")
