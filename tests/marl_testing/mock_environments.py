"""Mock Environments for MARL Testing.

This module provides mock environments and agents for isolated testing
of MARL components without external dependencies.
"""

# Standard Library
import asyncio
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

# SynThesisAI Modules
from utils.logging_config import get_logger


class MockAgentType(Enum):
    """Mock agent type enumeration."""

    GENERATOR = "generator"
    VALIDATOR = "validator"
    CURRICULUM = "curriculum"


class MockEnvironmentState(Enum):
    """Mock environment state enumeration."""

    IDLE = "idle"
    ACTIVE = "active"
    COORDINATING = "coordinating"
    ERROR = "error"


@dataclass
class MockAction:
    """Mock action for testing."""

    agent_id: str
    action_type: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    confidence: float = 1.0


@dataclass
class MockObservation:
    """Mock observation for testing."""

    state: Dict[str, Any] = field(default_factory=dict)
    reward: float = 0.0
    done: bool = False
    info: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class MockExperience:
    """Mock experience for testing."""

    state: Dict[str, Any]
    action: MockAction
    reward: float
    next_state: Dict[str, Any]
    done: bool
    timestamp: float = field(default_factory=time.time)


class MockAgent:
    """Mock agent for MARL testing.

    Provides a controllable agent implementation for testing
    coordination mechanisms without real RL complexity.
    """

    def __init__(
        self,
        agent_id: str,
        agent_type: MockAgentType,
        behavior_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize mock agent.

        Args:
            agent_id: Unique agent identifier
            agent_type: Type of agent (generator, validator, curriculum)
            behavior_config: Configuration for agent behavior
        """
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.behavior_config = behavior_config or {}
        self.logger = get_logger(f"{__name__}.{agent_id}")

        # Agent state
        self.is_active = False
        self.current_state = {}
        self.action_history: List[MockAction] = []
        self.experience_buffer: List[MockExperience] = []

        # Behavior configuration
        self.response_delay = self.behavior_config.get("response_delay", 0.1)
        self.error_probability = self.behavior_config.get("error_probability", 0.0)
        self.action_success_rate = self.behavior_config.get("action_success_rate", 0.9)
        self.cooperation_level = self.behavior_config.get("cooperation_level", 0.8)

        # Performance metrics
        self.metrics = {
            "actions_taken": 0,
            "successful_actions": 0,
            "coordination_attempts": 0,
            "successful_coordinations": 0,
            "total_reward": 0.0,
            "average_response_time": 0.0,
        }

        # Callbacks for testing
        self.action_callbacks: List[Callable] = []
        self.coordination_callbacks: List[Callable] = []

        self.logger.info("Mock agent initialized: %s (%s)", agent_id, agent_type.value)

    async def observe(self, observation: MockObservation) -> None:
        """Process observation from environment."""
        self.current_state = observation.state
        self.metrics["total_reward"] += observation.reward

        # Simulate processing delay
        if self.response_delay > 0:
            await asyncio.sleep(self.response_delay)

        self.logger.debug("Agent %s observed state", self.agent_id)

    async def select_action(self, state: Dict[str, Any]) -> MockAction:
        """Select action based on current state."""
        start_time = time.time()

        # Simulate error probability
        if random.random() < self.error_probability:
            raise RuntimeError(f"Mock error in agent {self.agent_id}")

        # Generate action based on agent type
        action = self._generate_action_by_type(state)

        # Update metrics
        self.metrics["actions_taken"] += 1
        if random.random() < self.action_success_rate:
            self.metrics["successful_actions"] += 1

        # Update response time
        response_time = time.time() - start_time
        self._update_average_response_time(response_time)

        # Store action
        self.action_history.append(action)

        # Notify callbacks
        await self._notify_action_callbacks(action)

        self.logger.debug("Agent %s selected action: %s", self.agent_id, action.action_type)
        return action

    def _generate_action_by_type(self, state: Dict[str, Any]) -> MockAction:
        """Generate action based on agent type."""
        if self.agent_type == MockAgentType.GENERATOR:
            return MockAction(
                agent_id=self.agent_id,
                action_type="generate_content",
                parameters={
                    "content_type": random.choice(["text", "image", "video"]),
                    "quality_target": random.uniform(0.7, 1.0),
                    "creativity_level": random.uniform(0.5, 1.0),
                },
                confidence=random.uniform(0.6, 1.0),
            )

        elif self.agent_type == MockAgentType.VALIDATOR:
            return MockAction(
                agent_id=self.agent_id,
                action_type="validate_content",
                parameters={
                    "validation_criteria": ["quality", "accuracy", "relevance"],
                    "threshold": random.uniform(0.6, 0.9),
                    "feedback_detail": random.choice(["basic", "detailed", "comprehensive"]),
                },
                confidence=random.uniform(0.7, 1.0),
            )

        elif self.agent_type == MockAgentType.CURRICULUM:
            return MockAction(
                agent_id=self.agent_id,
                action_type="suggest_curriculum",
                parameters={
                    "difficulty_level": random.uniform(0.3, 0.8),
                    "learning_objectives": random.randint(3, 8),
                    "progression_strategy": random.choice(["linear", "adaptive", "branching"]),
                },
                confidence=random.uniform(0.6, 0.9),
            )

        else:
            return MockAction(
                agent_id=self.agent_id,
                action_type="default_action",
                parameters={},
                confidence=0.5,
            )

    async def coordinate_with_agents(
        self, other_agents: List["MockAgent"], coordination_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Coordinate with other agents."""
        self.metrics["coordination_attempts"] += 1

        # Simulate coordination delay
        await asyncio.sleep(self.response_delay * len(other_agents))

        # Determine coordination success based on cooperation level
        coordination_success = random.random() < self.cooperation_level

        if coordination_success:
            self.metrics["successful_coordinations"] += 1

        # Generate coordination result
        coordination_result = {
            "agent_id": self.agent_id,
            "success": coordination_success,
            "contribution": self._generate_coordination_contribution(),
            "conflicts": self._detect_mock_conflicts(other_agents),
            "suggestions": self._generate_coordination_suggestions(),
        }

        # Notify callbacks
        await self._notify_coordination_callbacks(coordination_result)

        self.logger.debug(
            "Agent %s coordination result: %s",
            self.agent_id,
            "success" if coordination_success else "failed",
        )

        return coordination_result

    def _generate_coordination_contribution(self) -> Dict[str, Any]:
        """Generate mock coordination contribution."""
        if self.agent_type == MockAgentType.GENERATOR:
            return {
                "type": "content_generation",
                "quality_score": random.uniform(0.6, 1.0),
                "novelty_score": random.uniform(0.4, 0.9),
                "efficiency_score": random.uniform(0.5, 0.95),
            }

        elif self.agent_type == MockAgentType.VALIDATOR:
            return {
                "type": "content_validation",
                "accuracy_score": random.uniform(0.7, 1.0),
                "feedback_quality": random.uniform(0.6, 0.95),
                "validation_confidence": random.uniform(0.8, 1.0),
            }

        elif self.agent_type == MockAgentType.CURRICULUM:
            return {
                "type": "curriculum_design",
                "pedagogical_score": random.uniform(0.6, 0.95),
                "coherence_score": random.uniform(0.7, 1.0),
                "progression_quality": random.uniform(0.5, 0.9),
            }

        return {"type": "generic", "score": random.uniform(0.5, 0.8)}

    def _detect_mock_conflicts(self, other_agents: List["MockAgent"]) -> List[Dict[str, Any]]:
        """Detect mock conflicts with other agents."""
        conflicts = []

        for other_agent in other_agents:
            # Simulate conflict probability based on agent types
            conflict_probability = self._calculate_conflict_probability(other_agent)

            if random.random() < conflict_probability:
                conflicts.append(
                    {
                        "with_agent": other_agent.agent_id,
                        "conflict_type": random.choice(
                            [
                                "resource_contention",
                                "strategy_disagreement",
                                "priority_conflict",
                                "quality_threshold_mismatch",
                            ]
                        ),
                        "severity": random.uniform(0.1, 0.8),
                    }
                )

        return conflicts

    def _calculate_conflict_probability(self, other_agent: "MockAgent") -> float:
        """Calculate conflict probability with another agent."""
        # Same type agents have higher conflict probability
        if self.agent_type == other_agent.agent_type:
            return 0.3

        # Different cooperation levels increase conflict probability
        cooperation_diff = abs(self.cooperation_level - other_agent.cooperation_level)
        return 0.1 + (cooperation_diff * 0.2)

    def _generate_coordination_suggestions(self) -> List[str]:
        """Generate mock coordination suggestions."""
        suggestions = [
            "Increase communication frequency",
            "Adjust quality thresholds",
            "Implement priority-based scheduling",
            "Use consensus-based decision making",
            "Add conflict resolution mechanisms",
        ]

        return random.sample(suggestions, random.randint(1, 3))

    def _update_average_response_time(self, response_time: float) -> None:
        """Update average response time metric."""
        current_avg = self.metrics["average_response_time"]
        total_actions = self.metrics["actions_taken"]

        if total_actions == 1:
            self.metrics["average_response_time"] = response_time
        else:
            self.metrics["average_response_time"] = (
                current_avg * (total_actions - 1) + response_time
            ) / total_actions

    async def _notify_action_callbacks(self, action: MockAction) -> None:
        """Notify action callbacks."""
        for callback in self.action_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(self, action)
                else:
                    callback(self, action)
            except Exception as e:
                self.logger.error("Action callback error: %s", str(e))

    async def _notify_coordination_callbacks(self, result: Dict[str, Any]) -> None:
        """Notify coordination callbacks."""
        for callback in self.coordination_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(self, result)
                else:
                    callback(self, result)
            except Exception as e:
                self.logger.error("Coordination callback error: %s", str(e))

    def add_action_callback(self, callback: Callable) -> None:
        """Add action callback."""
        self.action_callbacks.append(callback)

    def add_coordination_callback(self, callback: Callable) -> None:
        """Add coordination callback."""
        self.coordination_callbacks.append(callback)

    def get_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics."""
        return self.metrics.copy()

    def reset_metrics(self) -> None:
        """Reset agent metrics."""
        self.metrics = {
            "actions_taken": 0,
            "successful_actions": 0,
            "coordination_attempts": 0,
            "successful_coordinations": 0,
            "total_reward": 0.0,
            "average_response_time": 0.0,
        }

    def get_state(self) -> Dict[str, Any]:
        """Get current agent state."""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type.value,
            "is_active": self.is_active,
            "current_state": self.current_state,
            "metrics": self.metrics,
            "action_history_length": len(self.action_history),
            "experience_buffer_length": len(self.experience_buffer),
        }


class MockCoordinationScenario:
    """Mock coordination scenario for testing.

    Provides predefined scenarios for testing different
    coordination situations and conflict resolution.
    """

    def __init__(
        self,
        scenario_id: str,
        scenario_type: str,
        agents: List[MockAgent],
        scenario_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize mock coordination scenario.

        Args:
            scenario_id: Unique scenario identifier
            scenario_type: Type of scenario (cooperation, conflict, mixed)
            agents: List of agents participating in scenario
            scenario_config: Configuration for scenario behavior
        """
        self.scenario_id = scenario_id
        self.scenario_type = scenario_type
        self.agents = agents
        self.scenario_config = scenario_config or {}
        self.logger = get_logger(f"{__name__}.{scenario_id}")

        # Scenario state
        self.is_running = False
        self.current_step = 0
        self.max_steps = self.scenario_config.get("max_steps", 100)

        # Scenario results
        self.results = {
            "coordination_success_rate": 0.0,
            "conflict_resolution_rate": 0.0,
            "average_response_time": 0.0,
            "total_actions": 0,
            "successful_coordinations": 0,
            "conflicts_detected": 0,
            "conflicts_resolved": 0,
        }

        # Event tracking
        self.events: List[Dict[str, Any]] = []

        self.logger.info("Mock coordination scenario initialized: %s", scenario_id)

    async def run_scenario(self) -> Dict[str, Any]:
        """Run the coordination scenario."""
        if self.is_running:
            self.logger.warning("Scenario already running")
            return self.results

        self.logger.info("Starting coordination scenario: %s", self.scenario_id)
        self.is_running = True
        scenario_start_time = time.time()

        try:
            # Initialize scenario
            await self._initialize_scenario()

            # Run scenario steps
            for step in range(self.max_steps):
                self.current_step = step

                step_results = await self._execute_scenario_step()
                self._update_scenario_results(step_results)

                # Check termination conditions
                if await self._should_terminate():
                    break

                # Add step delay
                step_delay = self.scenario_config.get("step_delay", 0.05)
                if step_delay > 0:
                    await asyncio.sleep(step_delay)

            # Finalize scenario
            await self._finalize_scenario()

            scenario_duration = time.time() - scenario_start_time
            self.results["scenario_duration"] = scenario_duration

            self.logger.info(
                "Coordination scenario complete: %s (%.2fs, %d steps)",
                self.scenario_id,
                scenario_duration,
                self.current_step + 1,
            )

            return self.results

        except Exception as e:
            self.logger.error("Scenario execution failed: %s", str(e))
            raise
        finally:
            self.is_running = False

    async def _initialize_scenario(self) -> None:
        """Initialize scenario state."""
        # Reset agent metrics
        for agent in self.agents:
            agent.reset_metrics()

        # Set up scenario-specific conditions
        if self.scenario_type == "conflict":
            await self._setup_conflict_conditions()
        elif self.scenario_type == "cooperation":
            await self._setup_cooperation_conditions()
        elif self.scenario_type == "mixed":
            await self._setup_mixed_conditions()

        self.logger.debug("Scenario initialized: %s", self.scenario_type)

    async def _setup_conflict_conditions(self) -> None:
        """Set up conditions that promote conflicts."""
        for agent in self.agents:
            # Reduce cooperation level
            agent.cooperation_level *= 0.6
            # Increase error probability
            agent.error_probability = min(agent.error_probability + 0.1, 0.3)

    async def _setup_cooperation_conditions(self) -> None:
        """Set up conditions that promote cooperation."""
        for agent in self.agents:
            # Increase cooperation level
            agent.cooperation_level = min(agent.cooperation_level * 1.2, 1.0)
            # Reduce error probability
            agent.error_probability *= 0.5

    async def _setup_mixed_conditions(self) -> None:
        """Set up mixed conditions."""
        for i, agent in enumerate(self.agents):
            if i % 2 == 0:
                # Cooperative agents
                agent.cooperation_level = min(agent.cooperation_level * 1.1, 1.0)
            else:
                # Less cooperative agents
                agent.cooperation_level *= 0.8

    async def _execute_scenario_step(self) -> Dict[str, Any]:
        """Execute one scenario step."""
        step_results = {
            "step": self.current_step,
            "actions": [],
            "coordinations": [],
            "conflicts": [],
            "resolutions": [],
        }

        # Generate actions for each agent
        for agent in self.agents:
            try:
                action = await agent.select_action(agent.current_state)
                step_results["actions"].append(action)
            except Exception as e:
                self.logger.warning("Agent %s action failed: %s", agent.agent_id, str(e))

        # Perform coordination
        coordination_pairs = self._generate_coordination_pairs()
        for agent1, agent2 in coordination_pairs:
            try:
                coordination_result = await agent1.coordinate_with_agents(
                    [agent2], {"step": self.current_step, "scenario": self.scenario_id}
                )
                step_results["coordinations"].append(coordination_result)

                # Check for conflicts
                if coordination_result.get("conflicts"):
                    step_results["conflicts"].extend(coordination_result["conflicts"])

                    # Attempt conflict resolution
                    for conflict in coordination_result["conflicts"]:
                        resolution = await self._attempt_conflict_resolution(conflict)
                        if resolution:
                            step_results["resolutions"].append(resolution)

            except Exception as e:
                self.logger.warning("Coordination failed: %s", str(e))

        # Record event
        self.events.append(
            {
                "step": self.current_step,
                "timestamp": time.time(),
                "results": step_results,
            }
        )

        return step_results

    def _generate_coordination_pairs(self) -> List[Tuple[MockAgent, MockAgent]]:
        """Generate pairs of agents for coordination."""
        pairs = []

        # Simple round-robin pairing
        for i in range(len(self.agents)):
            for j in range(i + 1, len(self.agents)):
                pairs.append((self.agents[i], self.agents[j]))

        # Limit pairs per step to avoid overwhelming
        max_pairs = self.scenario_config.get("max_coordination_pairs", 3)
        return random.sample(pairs, min(len(pairs), max_pairs))

    async def _attempt_conflict_resolution(
        self, conflict: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Attempt to resolve a conflict."""
        resolution_probability = self.scenario_config.get("conflict_resolution_probability", 0.7)

        if random.random() < resolution_probability:
            return {
                "conflict_id": conflict.get("with_agent", "unknown"),
                "conflict_type": conflict.get("conflict_type", "unknown"),
                "resolution_strategy": random.choice(
                    [
                        "compromise",
                        "priority_override",
                        "resource_reallocation",
                        "consensus_building",
                    ]
                ),
                "success": True,
                "resolution_time": random.uniform(0.1, 0.5),
            }

        return None

    def _update_scenario_results(self, step_results: Dict[str, Any]) -> None:
        """Update scenario results with step data."""
        # Count actions
        self.results["total_actions"] += len(step_results["actions"])

        # Count successful coordinations
        successful_coords = sum(
            1 for coord in step_results["coordinations"] if coord.get("success", False)
        )
        self.results["successful_coordinations"] += successful_coords

        # Count conflicts and resolutions
        self.results["conflicts_detected"] += len(step_results["conflicts"])
        self.results["conflicts_resolved"] += len(step_results["resolutions"])

        # Update rates
        total_coordinations = sum(agent.metrics["coordination_attempts"] for agent in self.agents)
        if total_coordinations > 0:
            self.results["coordination_success_rate"] = (
                self.results["successful_coordinations"] / total_coordinations
            )

        if self.results["conflicts_detected"] > 0:
            self.results["conflict_resolution_rate"] = (
                self.results["conflicts_resolved"] / self.results["conflicts_detected"]
            )

        # Update average response time
        total_response_time = sum(agent.metrics["average_response_time"] for agent in self.agents)
        if len(self.agents) > 0:
            self.results["average_response_time"] = total_response_time / len(self.agents)

    async def _should_terminate(self) -> bool:
        """Check if scenario should terminate early."""
        # Terminate if all agents have high success rates
        if self.scenario_type == "cooperation":
            avg_success_rate = sum(
                agent.metrics["successful_actions"] / max(agent.metrics["actions_taken"], 1)
                for agent in self.agents
            ) / len(self.agents)

            if avg_success_rate > 0.95:
                return True

        # Terminate if too many conflicts in conflict scenario
        if self.scenario_type == "conflict":
            if self.results["conflicts_detected"] > self.max_steps * 0.8:
                return True

        return False

    async def _finalize_scenario(self) -> None:
        """Finalize scenario and compute final metrics."""
        # Aggregate final metrics from agents
        for agent in self.agents:
            agent_metrics = agent.get_metrics()
            self.logger.debug("Agent %s final metrics: %s", agent.agent_id, agent_metrics)

        self.logger.debug("Scenario finalized: %s", self.scenario_id)

    def get_scenario_results(self) -> Dict[str, Any]:
        """Get scenario results."""
        return {
            "scenario_id": self.scenario_id,
            "scenario_type": self.scenario_type,
            "results": self.results,
            "agent_count": len(self.agents),
            "total_steps": self.current_step + 1,
            "events_recorded": len(self.events),
        }

    def get_detailed_events(self) -> List[Dict[str, Any]]:
        """Get detailed event history."""
        return self.events.copy()


class MockMARLEnvironment:
    """Mock MARL environment for comprehensive testing.

    Provides a controlled environment for testing MARL coordination
    without external dependencies or complex RL dynamics.
    """

    def __init__(self, environment_config: Optional[Dict[str, Any]] = None):
        """
        Initialize mock MARL environment.

        Args:
            environment_config: Configuration for environment behavior
        """
        self.environment_config = environment_config or {}
        self.logger = get_logger(__name__)

        # Environment state
        self.state = MockEnvironmentState.IDLE
        self.current_step = 0
        self.agents: Dict[str, MockAgent] = {}
        self.scenarios: Dict[str, MockCoordinationScenario] = {}

        # Environment metrics
        self.metrics = {
            "total_steps": 0,
            "total_episodes": 0,
            "average_episode_length": 0.0,
            "coordination_success_rate": 0.0,
            "conflict_resolution_rate": 0.0,
            "environment_uptime": 0.0,
        }

        # Environment history
        self.episode_history: List[Dict[str, Any]] = []
        self.start_time = time.time()

        self.logger.info("Mock MARL environment initialized")

    def add_agent(self, agent: MockAgent) -> None:
        """Add agent to environment."""
        self.agents[agent.agent_id] = agent
        self.logger.info("Added agent to environment: %s", agent.agent_id)

    def remove_agent(self, agent_id: str) -> None:
        """Remove agent from environment."""
        if agent_id in self.agents:
            del self.agents[agent_id]
            self.logger.info("Removed agent from environment: %s", agent_id)

    def add_scenario(self, scenario: MockCoordinationScenario) -> None:
        """Add coordination scenario to environment."""
        self.scenarios[scenario.scenario_id] = scenario
        self.logger.info("Added scenario to environment: %s", scenario.scenario_id)

    async def run_episode(
        self, episode_length: int = 100, scenario_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run one episode in the environment."""
        if self.state != MockEnvironmentState.IDLE:
            self.logger.warning("Environment not idle, cannot start episode")
            return {}

        self.logger.info("Starting environment episode (length: %d)", episode_length)
        self.state = MockEnvironmentState.ACTIVE
        episode_start_time = time.time()

        episode_results = {
            "episode_id": f"episode_{self.metrics['total_episodes']}",
            "episode_length": episode_length,
            "scenario_id": scenario_id,
            "steps_completed": 0,
            "coordination_events": 0,
            "conflicts_detected": 0,
            "conflicts_resolved": 0,
            "agent_performance": {},
        }

        try:
            # Run scenario if specified
            if scenario_id and scenario_id in self.scenarios:
                scenario_results = await self.scenarios[scenario_id].run_scenario()
                episode_results.update(scenario_results)
            else:
                # Run standard episode
                for step in range(episode_length):
                    self.current_step = step

                    step_results = await self._execute_environment_step()
                    self._update_episode_results(episode_results, step_results)

                    # Check termination conditions
                    if await self._should_terminate_episode():
                        break

            episode_results["steps_completed"] = self.current_step + 1

            # Collect agent performance
            for agent_id, agent in self.agents.items():
                episode_results["agent_performance"][agent_id] = agent.get_metrics()

            episode_duration = time.time() - episode_start_time
            episode_results["episode_duration"] = episode_duration

            # Update environment metrics
            self._update_environment_metrics(episode_results)

            # Store episode history
            self.episode_history.append(episode_results)

            self.logger.info(
                "Episode complete: %d steps, %.2fs duration",
                episode_results["steps_completed"],
                episode_duration,
            )

            return episode_results

        except Exception as e:
            self.logger.error("Episode execution failed: %s", str(e))
            raise
        finally:
            self.state = MockEnvironmentState.IDLE

    async def _execute_environment_step(self) -> Dict[str, Any]:
        """Execute one environment step."""
        step_results = {
            "step": self.current_step,
            "agent_actions": {},
            "coordination_events": [],
            "conflicts": [],
            "environment_events": [],
        }

        # Generate observations for agents
        observations = self._generate_observations()

        # Process agent actions
        for agent_id, agent in self.agents.items():
            try:
                # Send observation
                await agent.observe(observations[agent_id])

                # Get action
                action = await agent.select_action(observations[agent_id].state)
                step_results["agent_actions"][agent_id] = action

            except Exception as e:
                self.logger.warning("Agent %s step failed: %s", agent_id, str(e))

        # Simulate coordination events
        if len(self.agents) > 1 and random.random() < 0.3:
            coordination_event = await self._simulate_coordination_event()
            if coordination_event:
                step_results["coordination_events"].append(coordination_event)

        # Simulate environment events
        if random.random() < 0.1:
            env_event = self._generate_environment_event()
            step_results["environment_events"].append(env_event)

        return step_results

    def _generate_observations(self) -> Dict[str, MockObservation]:
        """Generate observations for all agents."""
        observations = {}

        for agent_id, agent in self.agents.items():
            # Generate agent-specific observation
            observation = MockObservation(
                state={
                    "step": self.current_step,
                    "agent_id": agent_id,
                    "environment_state": self.state.value,
                    "other_agents": [aid for aid in self.agents.keys() if aid != agent_id],
                    "random_factor": random.uniform(0, 1),
                },
                reward=random.uniform(-0.1, 0.1),  # Small random reward
                done=False,
                info={
                    "environment_step": self.current_step,
                    "total_agents": len(self.agents),
                },
            )

            observations[agent_id] = observation

        return observations

    async def _simulate_coordination_event(self) -> Optional[Dict[str, Any]]:
        """Simulate a coordination event between random agents."""
        if len(self.agents) < 2:
            return None

        # Select random agents for coordination
        agent_ids = random.sample(list(self.agents.keys()), 2)
        agent1 = self.agents[agent_ids[0]]
        agent2 = self.agents[agent_ids[1]]

        try:
            coordination_result = await agent1.coordinate_with_agents(
                [agent2],
                {"step": self.current_step, "event_type": "random_coordination"},
            )

            return {
                "type": "coordination",
                "participants": agent_ids,
                "result": coordination_result,
                "timestamp": time.time(),
            }

        except Exception as e:
            self.logger.warning("Coordination event failed: %s", str(e))
            return None

    def _generate_environment_event(self) -> Dict[str, Any]:
        """Generate random environment event."""
        event_types = [
            "resource_change",
            "difficulty_adjustment",
            "new_objective",
            "system_update",
            "external_disturbance",
        ]

        return {
            "type": random.choice(event_types),
            "impact": random.uniform(0.1, 0.5),
            "duration": random.randint(1, 10),
            "timestamp": time.time(),
            "step": self.current_step,
        }

    def _update_episode_results(
        self, episode_results: Dict[str, Any], step_results: Dict[str, Any]
    ) -> None:
        """Update episode results with step data."""
        episode_results["coordination_events"] += len(step_results["coordination_events"])
        episode_results["conflicts_detected"] += len(step_results["conflicts"])

        # Count resolved conflicts (simplified)
        for event in step_results["coordination_events"]:
            if event.get("result", {}).get("success", False):
                episode_results["conflicts_resolved"] += len(
                    event.get("result", {}).get("conflicts", [])
                )

    async def _should_terminate_episode(self) -> bool:
        """Check if episode should terminate early."""
        # Terminate if all agents are performing very well
        if len(self.agents) > 0:
            avg_success_rate = sum(
                agent.metrics["successful_actions"] / max(agent.metrics["actions_taken"], 1)
                for agent in self.agents.values()
            ) / len(self.agents)

            if avg_success_rate > 0.98:
                return True

        return False

    def _update_environment_metrics(self, episode_results: Dict[str, Any]) -> None:
        """Update environment metrics with episode data."""
        self.metrics["total_episodes"] += 1
        self.metrics["total_steps"] += episode_results["steps_completed"]

        # Update average episode length
        total_episodes = self.metrics["total_episodes"]
        current_avg = self.metrics["average_episode_length"]
        new_length = episode_results["steps_completed"]

        self.metrics["average_episode_length"] = (
            current_avg * (total_episodes - 1) + new_length
        ) / total_episodes

        # Update coordination success rate
        if episode_results["coordination_events"] > 0:
            episode_success_rate = episode_results["conflicts_resolved"] / max(
                episode_results["conflicts_detected"], 1
            )

            # Exponential moving average
            alpha = 0.1
            self.metrics["coordination_success_rate"] = (
                alpha * episode_success_rate
                + (1 - alpha) * self.metrics["coordination_success_rate"]
            )

        # Update environment uptime
        self.metrics["environment_uptime"] = time.time() - self.start_time

    def get_environment_metrics(self) -> Dict[str, Any]:
        """Get environment performance metrics."""
        return self.metrics.copy()

    def get_environment_state(self) -> Dict[str, Any]:
        """Get current environment state."""
        return {
            "state": self.state.value,
            "current_step": self.current_step,
            "agent_count": len(self.agents),
            "scenario_count": len(self.scenarios),
            "metrics": self.metrics,
            "uptime": time.time() - self.start_time,
        }

    def reset_environment(self) -> None:
        """Reset environment to initial state."""
        self.state = MockEnvironmentState.IDLE
        self.current_step = 0

        # Reset agent metrics
        for agent in self.agents.values():
            agent.reset_metrics()

        # Reset environment metrics
        self.metrics = {
            "total_steps": 0,
            "total_episodes": 0,
            "average_episode_length": 0.0,
            "coordination_success_rate": 0.0,
            "conflict_resolution_rate": 0.0,
            "environment_uptime": 0.0,
        }

        # Clear history
        self.episode_history.clear()
        self.start_time = time.time()

        self.logger.info("Environment reset complete")

    def get_episode_history(self) -> List[Dict[str, Any]]:
        """Get episode history."""
        return self.episode_history.copy()
