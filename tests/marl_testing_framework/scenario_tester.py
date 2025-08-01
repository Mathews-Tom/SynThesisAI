"""Scenario Testing Framework for MARL.

This module provides comprehensive scenario testing capabilities for MARL
coordination, including conflict situations, edge cases, and performance scenarios.
"""

# Standard Library
import random
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

# SynThesisAI Modules
from utils.logging_config import get_logger

from .mock_environment import (
    DifficultyLevel,
    EnvironmentType,
    MockEnvironmentConfig,
    MockMARLEnvironment,
)


class ScenarioType(Enum):
    """Scenario type enumeration."""

    COORDINATION_CONFLICT = "coordination_conflict"
    RESOURCE_CONTENTION = "resource_contention"
    AGENT_FAILURE = "agent_failure"
    PERFORMANCE_STRESS = "performance_stress"
    LEARNING_CONVERGENCE = "learning_convergence"
    EDGE_CASE = "edge_case"
    INTEGRATION = "integration"


class ConflictType(Enum):
    """Conflict type enumeration."""

    ACTION_CONFLICT = "action_conflict"
    GOAL_CONFLICT = "goal_conflict"
    RESOURCE_CONFLICT = "resource_conflict"
    TIMING_CONFLICT = "timing_conflict"
    PRIORITY_CONFLICT = "priority_conflict"


@dataclass
class ScenarioConfig:
    """Configuration for scenario testing."""

    # Scenario settings
    scenario_type: ScenarioType = ScenarioType.COORDINATION_CONFLICT
    scenario_name: str = "default_scenario"
    description: str = ""

    # Test parameters
    num_runs: int = 10
    max_episodes_per_run: int = 50
    timeout_per_run: float = 300.0  # 5 minutes

    # Environment settings
    environment_config: Optional[MockEnvironmentConfig] = None

    # Conflict settings (for conflict scenarios)
    conflict_type: ConflictType = ConflictType.ACTION_CONFLICT
    conflict_intensity: float = 0.5  # 0.0 to 1.0
    conflict_frequency: float = 0.3  # Probability per step

    # Stress settings (for performance scenarios)
    stress_factor: float = 1.0
    concurrent_agents: int = 10
    high_frequency_actions: bool = False

    # Success criteria
    min_success_rate: float = 0.8
    max_failure_rate: float = 0.2
    performance_threshold: float = 0.7

    # Validation settings
    validate_coordination: bool = True
    validate_performance: bool = True
    validate_learning: bool = True

    def __post_init__(self):
        """Initialize default environment config if not provided."""
        if self.environment_config is None:
            self.environment_config = MockEnvironmentConfig()


@dataclass
class TestScenario:
    """Test scenario definition."""

    scenario_id: str
    config: ScenarioConfig
    setup_function: Optional[Callable] = None
    teardown_function: Optional[Callable] = None
    validation_function: Optional[Callable] = None

    # Scenario state
    is_running: bool = False
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    # Results
    run_results: List[Dict[str, Any]] = field(default_factory=list)
    overall_result: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Initialize scenario ID if not provided."""
        if not self.scenario_id:
            self.scenario_id = str(uuid.uuid4())


class ScenarioTester:
    """Scenario testing framework for MARL coordination.

    Provides comprehensive testing of various coordination scenarios including
    conflict situations, edge cases, and performance stress tests.
    """

    def __init__(self):
        """Initialize scenario tester."""
        self.logger = get_logger(__name__)

        # Test management
        self.registered_scenarios: Dict[str, TestScenario] = {}
        self.test_results: Dict[str, Dict[str, Any]] = {}

        # Built-in scenarios
        self._register_builtin_scenarios()

        self.logger.info("Scenario tester initialized")

    def _register_builtin_scenarios(self) -> None:
        """Register built-in test scenarios."""
        # Coordination conflict scenarios
        self.register_scenario(self._create_action_conflict_scenario())
        self.register_scenario(self._create_goal_conflict_scenario())
        self.register_scenario(self._create_resource_conflict_scenario())

        # Performance scenarios
        self.register_scenario(self._create_stress_test_scenario())
        self.register_scenario(self._create_high_load_scenario())

        # Edge case scenarios
        self.register_scenario(self._create_agent_failure_scenario())
        self.register_scenario(self._create_timeout_scenario())

        # Learning scenarios
        self.register_scenario(self._create_convergence_scenario())
        self.register_scenario(self._create_adaptation_scenario())

    def _create_action_conflict_scenario(self) -> TestScenario:
        """Create action conflict scenario."""
        config = ScenarioConfig(
            scenario_type=ScenarioType.COORDINATION_CONFLICT,
            scenario_name="action_conflict",
            description="Test coordination when agents choose conflicting actions",
            conflict_type=ConflictType.ACTION_CONFLICT,
            conflict_intensity=0.7,
            conflict_frequency=0.5,
            num_runs=15,
            environment_config=MockEnvironmentConfig(
                environment_type=EnvironmentType.COORDINATION,
                difficulty_level=DifficultyLevel.MEDIUM,
                coordination_required=True,
                consensus_threshold=0.8,
            ),
        )

        return TestScenario(
            scenario_id="action_conflict",
            config=config,
            setup_function=self._setup_conflict_scenario,
            validation_function=self._validate_conflict_resolution,
        )

    def _create_goal_conflict_scenario(self) -> TestScenario:
        """Create goal conflict scenario."""
        config = ScenarioConfig(
            scenario_type=ScenarioType.COORDINATION_CONFLICT,
            scenario_name="goal_conflict",
            description="Test coordination when agents have conflicting goals",
            conflict_type=ConflictType.GOAL_CONFLICT,
            conflict_intensity=0.6,
            num_runs=12,
            environment_config=MockEnvironmentConfig(
                environment_type=EnvironmentType.MIXED,
                num_agents=4,
                agent_types=["generator", "validator", "curriculum", "optimizer"],
                coordination_required=True,
            ),
        )

        return TestScenario(
            scenario_id="goal_conflict",
            config=config,
            setup_function=self._setup_goal_conflict,
            validation_function=self._validate_goal_resolution,
        )

    def _create_resource_conflict_scenario(self) -> TestScenario:
        """Create resource conflict scenario."""
        config = ScenarioConfig(
            scenario_type=ScenarioType.RESOURCE_CONTENTION,
            scenario_name="resource_conflict",
            description="Test coordination under resource constraints",
            conflict_type=ConflictType.RESOURCE_CONFLICT,
            conflict_intensity=0.8,
            num_runs=10,
            environment_config=MockEnvironmentConfig(
                environment_type=EnvironmentType.COORDINATION,
                num_agents=5,
                max_steps_per_episode=30,
            ),
        )

        return TestScenario(
            scenario_id="resource_conflict",
            config=config,
            setup_function=self._setup_resource_conflict,
            validation_function=self._validate_resource_management,
        )

    def _create_stress_test_scenario(self) -> TestScenario:
        """Create stress test scenario."""
        config = ScenarioConfig(
            scenario_type=ScenarioType.PERFORMANCE_STRESS,
            scenario_name="stress_test",
            description="Test system performance under high load",
            stress_factor=2.0,
            concurrent_agents=15,
            high_frequency_actions=True,
            num_runs=5,
            max_episodes_per_run=20,
            environment_config=MockEnvironmentConfig(
                environment_type=EnvironmentType.MIXED,
                num_agents=15,
                max_steps_per_episode=100,
                coordination_required=True,
            ),
        )

        return TestScenario(
            scenario_id="stress_test",
            config=config,
            setup_function=self._setup_stress_test,
            validation_function=self._validate_performance_under_stress,
        )

    def _create_high_load_scenario(self) -> TestScenario:
        """Create high load scenario."""
        config = ScenarioConfig(
            scenario_type=ScenarioType.PERFORMANCE_STRESS,
            scenario_name="high_load",
            description="Test coordination with many agents and rapid decisions",
            concurrent_agents=20,
            high_frequency_actions=True,
            num_runs=3,
            timeout_per_run=600.0,  # 10 minutes
            environment_config=MockEnvironmentConfig(
                num_agents=20, max_steps_per_episode=200, coordination_timeout=5.0
            ),
        )

        return TestScenario(
            scenario_id="high_load",
            config=config,
            setup_function=self._setup_high_load,
            validation_function=self._validate_high_load_performance,
        )

    def _create_agent_failure_scenario(self) -> TestScenario:
        """Create agent failure scenario."""
        config = ScenarioConfig(
            scenario_type=ScenarioType.AGENT_FAILURE,
            scenario_name="agent_failure",
            description="Test system resilience when agents fail",
            num_runs=8,
            environment_config=MockEnvironmentConfig(
                num_agents=4,
                coordination_required=True,
                consensus_threshold=0.5,  # Lower threshold to handle failures
            ),
        )

        return TestScenario(
            scenario_id="agent_failure",
            config=config,
            setup_function=self._setup_agent_failure,
            validation_function=self._validate_failure_recovery,
        )

    def _create_timeout_scenario(self) -> TestScenario:
        """Create timeout scenario."""
        config = ScenarioConfig(
            scenario_type=ScenarioType.EDGE_CASE,
            scenario_name="timeout_handling",
            description="Test handling of coordination timeouts",
            num_runs=6,
            environment_config=MockEnvironmentConfig(
                coordination_timeout=2.0,  # Short timeout
                coordination_required=True,
            ),
        )

        return TestScenario(
            scenario_id="timeout_handling",
            config=config,
            setup_function=self._setup_timeout_scenario,
            validation_function=self._validate_timeout_handling,
        )

    def _create_convergence_scenario(self) -> TestScenario:
        """Create learning convergence scenario."""
        config = ScenarioConfig(
            scenario_type=ScenarioType.LEARNING_CONVERGENCE,
            scenario_name="learning_convergence",
            description="Test agent learning convergence",
            num_runs=5,
            max_episodes_per_run=100,
            environment_config=MockEnvironmentConfig(
                difficulty_level=DifficultyLevel.ADAPTIVE, max_episodes=100
            ),
        )

        return TestScenario(
            scenario_id="learning_convergence",
            config=config,
            setup_function=self._setup_convergence_test,
            validation_function=self._validate_learning_convergence,
        )

    def _create_adaptation_scenario(self) -> TestScenario:
        """Create adaptation scenario."""
        config = ScenarioConfig(
            scenario_type=ScenarioType.LEARNING_CONVERGENCE,
            scenario_name="adaptation_test",
            description="Test agent adaptation to changing conditions",
            num_runs=7,
            environment_config=MockEnvironmentConfig(
                difficulty_level=DifficultyLevel.HARD,
                environment_type=EnvironmentType.MIXED,
            ),
        )

        return TestScenario(
            scenario_id="adaptation_test",
            config=config,
            setup_function=self._setup_adaptation_test,
            validation_function=self._validate_adaptation,
        )

    def register_scenario(self, scenario: TestScenario) -> None:
        """Register a test scenario."""
        self.registered_scenarios[scenario.scenario_id] = scenario
        self.logger.info("Registered scenario: %s", scenario.scenario_id)

    def unregister_scenario(self, scenario_id: str) -> None:
        """Unregister a test scenario."""
        if scenario_id in self.registered_scenarios:
            del self.registered_scenarios[scenario_id]
            self.logger.info("Unregistered scenario: %s", scenario_id)

    async def run_scenario(
        self, scenario_id: str, agent_policies: Dict[str, Callable]
    ) -> Dict[str, Any]:
        """Run a specific test scenario.

        Args:
            scenario_id: ID of scenario to run
            agent_policies: Dictionary mapping agent types to policy functions

        Returns:
            Scenario test results
        """
        if scenario_id not in self.registered_scenarios:
            raise ValueError(f"Scenario {scenario_id} not registered")

        scenario = self.registered_scenarios[scenario_id]

        if scenario.is_running:
            raise RuntimeError(f"Scenario {scenario_id} is already running")

        self.logger.info("Starting scenario: %s", scenario_id)

        scenario.is_running = True
        scenario.start_time = time.time()
        scenario.run_results.clear()

        try:
            # Setup scenario
            if scenario.setup_function:
                await scenario.setup_function(scenario)

            # Run multiple test runs
            for run_idx in range(scenario.config.num_runs):
                self.logger.debug(
                    "Running scenario %s, run %d/%d",
                    scenario_id,
                    run_idx + 1,
                    scenario.config.num_runs,
                )

                run_result = await self._run_single_scenario_run(
                    scenario, agent_policies, run_idx
                )
                scenario.run_results.append(run_result)

                # Check for early termination conditions
                if run_result.get("critical_failure", False):
                    self.logger.warning(
                        "Critical failure in scenario %s, run %d",
                        scenario_id,
                        run_idx + 1,
                    )
                    break

            # Analyze overall results
            scenario.overall_result = self._analyze_scenario_results(scenario)

            # Validate results
            if scenario.validation_function:
                validation_result = await scenario.validation_function(scenario)
                scenario.overall_result["validation"] = validation_result

            # Teardown scenario
            if scenario.teardown_function:
                await scenario.teardown_function(scenario)

            scenario.end_time = time.time()
            scenario.is_running = False

            # Store results
            self.test_results[scenario_id] = scenario.overall_result

            self.logger.info(
                "Completed scenario: %s (%.2fs)",
                scenario_id,
                scenario.end_time - scenario.start_time,
            )

            return scenario.overall_result

        except Exception as e:
            scenario.is_running = False
            scenario.end_time = time.time()

            error_result = {
                "scenario_id": scenario_id,
                "success": False,
                "error": str(e),
                "completed_runs": len(scenario.run_results),
                "total_runs": scenario.config.num_runs,
            }

            self.test_results[scenario_id] = error_result

            self.logger.error("Scenario %s failed: %s", scenario_id, str(e))
            raise

    async def _run_single_scenario_run(
        self, scenario: TestScenario, agent_policies: Dict[str, Callable], run_idx: int
    ) -> Dict[str, Any]:
        """Run a single scenario run."""
        run_start_time = time.time()

        # Create environment for this run
        environment = MockMARLEnvironment(scenario.config.environment_config)

        # Apply scenario-specific modifications
        await self._apply_scenario_modifications(environment, scenario, run_idx)

        run_data = {
            "run_index": run_idx,
            "start_time": run_start_time,
            "episodes": [],
            "total_reward": 0.0,
            "coordination_successes": 0,
            "coordination_attempts": 0,
            "critical_failure": False,
        }

        try:
            # Run episodes
            for episode_idx in range(scenario.config.max_episodes_per_run):
                episode_start = time.time()

                # Check timeout
                if time.time() - run_start_time > scenario.config.timeout_per_run:
                    self.logger.warning("Scenario run %d timed out", run_idx)
                    break

                # Run episode
                episode_result = await environment.run_episode(agent_policies)

                # Record episode data
                episode_data = {
                    "episode_index": episode_idx,
                    "duration": time.time() - episode_start,
                    "total_reward": episode_result.get("total_reward", 0.0),
                    "coordination_success_rate": episode_result.get(
                        "coordination_success_rate", 0.0
                    ),
                    "final_state_quality": episode_result.get(
                        "final_state_quality", 0.0
                    ),
                }

                run_data["episodes"].append(episode_data)
                run_data["total_reward"] += episode_data["total_reward"]

                # Track coordination
                coord_attempts = episode_result.get("coordination_attempts", 0)
                coord_successes = episode_result.get("coordination_successes", 0)
                run_data["coordination_attempts"] += coord_attempts
                run_data["coordination_successes"] += coord_successes

                # Check for critical failures
                if episode_result.get("critical_failure", False):
                    run_data["critical_failure"] = True
                    break

            # Calculate run metrics
            run_data["end_time"] = time.time()
            run_data["duration"] = run_data["end_time"] - run_start_time
            run_data["average_reward"] = run_data["total_reward"] / max(
                1, len(run_data["episodes"])
            )
            run_data["coordination_success_rate"] = run_data[
                "coordination_successes"
            ] / max(1, run_data["coordination_attempts"])

            return run_data

        except Exception as e:
            run_data["error"] = str(e)
            run_data["critical_failure"] = True
            run_data["end_time"] = time.time()
            run_data["duration"] = run_data["end_time"] - run_start_time

            self.logger.error("Scenario run %d failed: %s", run_idx, str(e))
            return run_data

    async def _apply_scenario_modifications(
        self, environment: MockMARLEnvironment, scenario: TestScenario, run_idx: int
    ) -> None:
        """Apply scenario-specific modifications to environment."""
        config = scenario.config

        if config.scenario_type == ScenarioType.COORDINATION_CONFLICT:
            # Inject conflicts based on configuration
            await self._inject_coordination_conflicts(environment, config)

        elif config.scenario_type == ScenarioType.PERFORMANCE_STRESS:
            # Apply stress factors
            await self._apply_stress_factors(environment, config)

        elif config.scenario_type == ScenarioType.AGENT_FAILURE:
            # Set up agent failure conditions
            await self._setup_agent_failures(environment, config, run_idx)

        elif config.scenario_type == ScenarioType.EDGE_CASE:
            # Configure edge case conditions
            await self._configure_edge_cases(environment, config)

    async def _inject_coordination_conflicts(
        self, environment: MockMARLEnvironment, config: ScenarioConfig
    ) -> None:
        """Inject coordination conflicts into environment."""
        # Modify environment to create conflicts
        if config.conflict_type == ConflictType.ACTION_CONFLICT:
            # Increase action space diversity
            environment.config.action_space_size *= 2

        elif config.conflict_type == ConflictType.GOAL_CONFLICT:
            # Modify reward functions to create conflicting goals
            environment.config.base_reward *= config.conflict_intensity

        elif config.conflict_type == ConflictType.RESOURCE_CONFLICT:
            # Reduce available resources
            environment.config.state_space_size = int(
                environment.config.state_space_size * (1 - config.conflict_intensity)
            )

    async def _apply_stress_factors(
        self, environment: MockMARLEnvironment, config: ScenarioConfig
    ) -> None:
        """Apply stress factors to environment."""
        # Increase complexity
        environment.config.state_space_size = int(
            environment.config.state_space_size * config.stress_factor
        )
        environment.config.action_space_size = int(
            environment.config.action_space_size * config.stress_factor
        )

        # Reduce timeouts if high frequency actions
        if config.high_frequency_actions:
            environment.config.coordination_timeout *= 0.5

    async def _setup_agent_failures(
        self, environment: MockMARLEnvironment, config: ScenarioConfig, run_idx: int
    ) -> None:
        """Set up agent failure conditions."""
        # Randomly disable agents during the run
        failure_probability = 0.1 + (
            run_idx * 0.05
        )  # Increase failure rate with run index

        for agent_id in environment.agents:
            if random.random() < failure_probability:
                environment.agents[agent_id]["active"] = False

    async def _configure_edge_cases(
        self, environment: MockMARLEnvironment, config: ScenarioConfig
    ) -> None:
        """Configure edge case conditions."""
        # Set extreme values to test edge cases
        environment.config.observation_noise *= 2.0
        environment.config.reward_noise *= 2.0
        environment.config.consensus_threshold = 0.9  # Very high threshold

    def _analyze_scenario_results(self, scenario: TestScenario) -> Dict[str, Any]:
        """Analyze overall scenario results."""
        if not scenario.run_results:
            return {"success": False, "error": "No run results"}

        # Calculate aggregate metrics
        total_runs = len(scenario.run_results)
        successful_runs = len(
            [r for r in scenario.run_results if not r.get("critical_failure", False)]
        )

        # Average metrics across runs
        avg_reward = (
            sum(r.get("average_reward", 0.0) for r in scenario.run_results) / total_runs
        )
        avg_coordination_success = (
            sum(r.get("coordination_success_rate", 0.0) for r in scenario.run_results)
            / total_runs
        )
        avg_duration = (
            sum(r.get("duration", 0.0) for r in scenario.run_results) / total_runs
        )

        # Success criteria evaluation
        success_rate = successful_runs / total_runs
        meets_success_criteria = success_rate >= scenario.config.min_success_rate
        meets_performance_criteria = avg_reward >= scenario.config.performance_threshold

        overall_success = meets_success_criteria and meets_performance_criteria

        return {
            "scenario_id": scenario.scenario_id,
            "scenario_name": scenario.config.scenario_name,
            "success": overall_success,
            "total_runs": total_runs,
            "successful_runs": successful_runs,
            "success_rate": success_rate,
            "metrics": {
                "average_reward": avg_reward,
                "average_coordination_success_rate": avg_coordination_success,
                "average_duration": avg_duration,
            },
            "criteria_evaluation": {
                "meets_success_criteria": meets_success_criteria,
                "meets_performance_criteria": meets_performance_criteria,
                "required_success_rate": scenario.config.min_success_rate,
                "required_performance_threshold": scenario.config.performance_threshold,
            },
            "run_details": scenario.run_results,
        }

    # Setup functions for different scenario types
    async def _setup_conflict_scenario(self, scenario: TestScenario) -> None:
        """Set up conflict scenario."""
        self.logger.debug("Setting up conflict scenario: %s", scenario.scenario_id)

    async def _setup_goal_conflict(self, scenario: TestScenario) -> None:
        """Set up goal conflict scenario."""
        self.logger.debug("Setting up goal conflict scenario: %s", scenario.scenario_id)

    async def _setup_resource_conflict(self, scenario: TestScenario) -> None:
        """Set up resource conflict scenario."""
        self.logger.debug(
            "Setting up resource conflict scenario: %s", scenario.scenario_id
        )

    async def _setup_stress_test(self, scenario: TestScenario) -> None:
        """Set up stress test scenario."""
        self.logger.debug("Setting up stress test scenario: %s", scenario.scenario_id)

    async def _setup_high_load(self, scenario: TestScenario) -> None:
        """Set up high load scenario."""
        self.logger.debug("Setting up high load scenario: %s", scenario.scenario_id)

    async def _setup_agent_failure(self, scenario: TestScenario) -> None:
        """Set up agent failure scenario."""
        self.logger.debug("Setting up agent failure scenario: %s", scenario.scenario_id)

    async def _setup_timeout_scenario(self, scenario: TestScenario) -> None:
        """Set up timeout scenario."""
        self.logger.debug("Setting up timeout scenario: %s", scenario.scenario_id)

    async def _setup_convergence_test(self, scenario: TestScenario) -> None:
        """Set up convergence test scenario."""
        self.logger.debug(
            "Setting up convergence test scenario: %s", scenario.scenario_id
        )

    async def _setup_adaptation_test(self, scenario: TestScenario) -> None:
        """Set up adaptation test scenario."""
        self.logger.debug(
            "Setting up adaptation test scenario: %s", scenario.scenario_id
        )

    # Validation functions for different scenario types
    async def _validate_conflict_resolution(
        self, scenario: TestScenario
    ) -> Dict[str, Any]:
        """Validate conflict resolution."""
        results = scenario.run_results

        # Check if conflicts were resolved successfully
        conflict_resolution_rate = sum(
            1 for r in results if r.get("coordination_success_rate", 0.0) > 0.5
        ) / len(results)

        return {
            "conflict_resolution_rate": conflict_resolution_rate,
            "validation_passed": conflict_resolution_rate >= 0.7,
            "details": "Conflict resolution validation",
        }

    async def _validate_goal_resolution(self, scenario: TestScenario) -> Dict[str, Any]:
        """Validate goal resolution."""
        results = scenario.run_results

        # Check if agents achieved balanced goals
        avg_reward = sum(r.get("average_reward", 0.0) for r in results) / len(results)

        return {
            "goal_balance_score": avg_reward,
            "validation_passed": avg_reward >= 0.6,
            "details": "Goal resolution validation",
        }

    async def _validate_resource_management(
        self, scenario: TestScenario
    ) -> Dict[str, Any]:
        """Validate resource management."""
        results = scenario.run_results

        # Check resource utilization efficiency
        efficiency_score = sum(
            r.get("coordination_success_rate", 0.0) for r in results
        ) / len(results)

        return {
            "resource_efficiency": efficiency_score,
            "validation_passed": efficiency_score >= 0.6,
            "details": "Resource management validation",
        }

    async def _validate_performance_under_stress(
        self, scenario: TestScenario
    ) -> Dict[str, Any]:
        """Validate performance under stress."""
        results = scenario.run_results

        # Check if system maintained performance under stress
        performance_maintained = sum(
            1
            for r in results
            if r.get("average_reward", 0.0) >= scenario.config.performance_threshold
        ) / len(results)

        return {
            "performance_maintenance_rate": performance_maintained,
            "validation_passed": performance_maintained >= 0.8,
            "details": "Performance under stress validation",
        }

    async def _validate_high_load_performance(
        self, scenario: TestScenario
    ) -> Dict[str, Any]:
        """Validate high load performance."""
        results = scenario.run_results

        # Check system stability under high load
        stability_score = sum(
            1 for r in results if not r.get("critical_failure", False)
        ) / len(results)

        return {
            "stability_score": stability_score,
            "validation_passed": stability_score >= 0.9,
            "details": "High load performance validation",
        }

    async def _validate_failure_recovery(
        self, scenario: TestScenario
    ) -> Dict[str, Any]:
        """Validate failure recovery."""
        results = scenario.run_results

        # Check recovery from agent failures
        recovery_rate = sum(
            1
            for r in results
            if r.get("coordination_success_rate", 0.0)
            > 0.3  # Lower threshold due to failures
        ) / len(results)

        return {
            "recovery_rate": recovery_rate,
            "validation_passed": recovery_rate >= 0.6,
            "details": "Failure recovery validation",
        }

    async def _validate_timeout_handling(
        self, scenario: TestScenario
    ) -> Dict[str, Any]:
        """Validate timeout handling."""
        results = scenario.run_results

        # Check graceful timeout handling
        timeout_handling_score = sum(
            1 for r in results if not r.get("critical_failure", False)
        ) / len(results)

        return {
            "timeout_handling_score": timeout_handling_score,
            "validation_passed": timeout_handling_score >= 0.8,
            "details": "Timeout handling validation",
        }

    async def _validate_learning_convergence(
        self, scenario: TestScenario
    ) -> Dict[str, Any]:
        """Validate learning convergence."""
        results = scenario.run_results

        # Check if learning converged (improving rewards over time)
        convergence_score = 0.0
        for result in results:
            episodes = result.get("episodes", [])
            if len(episodes) >= 10:
                early_rewards = [ep["total_reward"] for ep in episodes[:5]]
                late_rewards = [ep["total_reward"] for ep in episodes[-5:]]

                if sum(late_rewards) > sum(early_rewards):
                    convergence_score += 1.0

        convergence_rate = convergence_score / len(results)

        return {
            "convergence_rate": convergence_rate,
            "validation_passed": convergence_rate >= 0.7,
            "details": "Learning convergence validation",
        }

    async def _validate_adaptation(self, scenario: TestScenario) -> Dict[str, Any]:
        """Validate adaptation."""
        results = scenario.run_results

        # Check adaptation to changing conditions
        adaptation_score = sum(
            r.get("coordination_success_rate", 0.0) for r in results
        ) / len(results)

        return {
            "adaptation_score": adaptation_score,
            "validation_passed": adaptation_score
            >= 0.5,  # Lower threshold for hard adaptation
            "details": "Adaptation validation",
        }

    async def run_all_scenarios(
        self, agent_policies: Dict[str, Callable]
    ) -> Dict[str, Dict[str, Any]]:
        """Run all registered scenarios.

        Args:
            agent_policies: Dictionary mapping agent types to policy functions

        Returns:
            Dictionary of all scenario results
        """
        self.logger.info(
            "Running all scenarios (%d total)", len(self.registered_scenarios)
        )

        all_results = {}

        for scenario_id in self.registered_scenarios:
            try:
                result = await self.run_scenario(scenario_id, agent_policies)
                all_results[scenario_id] = result
            except Exception as e:
                self.logger.error("Failed to run scenario %s: %s", scenario_id, str(e))
                all_results[scenario_id] = {"success": False, "error": str(e)}

        return all_results

    def get_scenario_list(self) -> List[Dict[str, Any]]:
        """Get list of registered scenarios."""
        return [
            {
                "scenario_id": scenario.scenario_id,
                "name": scenario.config.scenario_name,
                "type": scenario.config.scenario_type.value,
                "description": scenario.config.description,
                "num_runs": scenario.config.num_runs,
            }
            for scenario in self.registered_scenarios.values()
        ]

    def get_test_results(self, scenario_id: Optional[str] = None) -> Dict[str, Any]:
        """Get test results.

        Args:
            scenario_id: Specific scenario ID, or None for all results

        Returns:
            Test results
        """
        if scenario_id:
            return self.test_results.get(scenario_id, {})
        else:
            return self.test_results.copy()

    def clear_results(self) -> None:
        """Clear all test results."""
        self.test_results.clear()

        # Clear individual scenario results
        for scenario in self.registered_scenarios.values():
            scenario.run_results.clear()
            scenario.overall_result = None
