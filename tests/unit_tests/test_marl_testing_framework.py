"""Unit tests for MARL testing framework components."""

import asyncio
import random
import time
from unittest.mock import Mock, patch

import pytest

from tests.marl_testing_framework.coordination_tester import (
    ConflictScenario,
    CoordinationTestConfig,
    CoordinationTester,
    CoordinationTestType,
    MockAgent,
)
from tests.marl_testing_framework.mock_environment import (
    ActionResult,
    DifficultyLevel,
    EnvironmentState,
    EnvironmentType,
    MockEnvironmentConfig,
    MockMARLEnvironment,
)
from tests.marl_testing_framework.performance_validator import (
    PerformanceConfig,
    PerformanceMetric,
    PerformanceResult,
    PerformanceValidator,
)
from tests.marl_testing_framework.scenario_tester import (
    ConflictType,
    ScenarioConfig,
    ScenarioTester,
    ScenarioType,
    TestScenario,
)


class TestMockEnvironment:
    """Test mock MARL environment."""

    @pytest.fixture
    def env_config(self):
        """Create environment configuration."""
        return MockEnvironmentConfig(
            environment_type=EnvironmentType.MIXED,
            difficulty_level=DifficultyLevel.MEDIUM,
            num_agents=3,
            max_episodes=10,
            max_steps_per_episode=20,
        )

    @pytest.fixture
    def environment(self, env_config):
        """Create mock environment."""
        return MockMARLEnvironment(env_config)

    def test_environment_initialization(self, environment):
        """Test environment initialization."""
        assert environment.config.num_agents == 3
        assert len(environment.agents) == 3
        assert len(environment.agent_states) == 3
        assert environment.current_episode == 0
        assert environment.current_step == 0

    @pytest.mark.asyncio
    async def test_environment_reset(self, environment):
        """Test environment reset."""
        observations = await environment.reset()

        assert len(observations) == 3
        assert environment.current_step == 0
        assert environment.is_active
        assert not environment.episode_done

        # Check observation structure
        for agent_id, obs in observations.items():
            assert agent_id in environment.agents
            assert len(obs) == environment.config.state_space_size

    @pytest.mark.asyncio
    async def test_environment_step(self, environment):
        """Test environment step execution."""
        await environment.reset()

        # Create actions for all agents
        actions = {
            agent_id: random.randint(0, environment.config.action_space_size - 1)
            for agent_id in environment.agents
        }

        observations, rewards, done, info = await environment.step(actions)

        # Check return values
        assert len(observations) == 3
        assert len(rewards) == 3
        assert isinstance(done, bool)
        assert isinstance(info, dict)

        # Check info structure
        assert "episode" in info
        assert "step" in info
        assert "performance_metrics" in info

        # Check that step was recorded
        assert environment.current_step == 1
        assert len(environment.action_history) == 1

    @pytest.mark.asyncio
    async def test_episode_completion(self, environment):
        """Test complete episode execution."""
        observations = await environment.reset()

        episode_steps = 0
        while (
            not environment.episode_done
            and episode_steps < environment.config.max_steps_per_episode
        ):
            actions = {
                agent_id: random.randint(0, environment.config.action_space_size - 1)
                for agent_id in environment.agents
            }

            observations, rewards, done, info = await environment.step(actions)
            episode_steps += 1

            if done:
                break

        # Episode should complete
        assert (
            environment.episode_done
            or episode_steps == environment.config.max_steps_per_episode
        )

        # Check episode summary
        if "episode_summary" in info:
            summary = info["episode_summary"]
            assert "total_steps" in summary
            assert "total_reward" in summary
            assert "coordination_success_rate" in summary

    @pytest.mark.asyncio
    async def test_coordination_mechanism(self, environment):
        """Test coordination mechanism."""
        environment.config.coordination_required = True
        await environment.reset()

        # Execute step with coordination
        actions = {
            agent_id: 5 for agent_id in environment.agents
        }  # Same action for coordination

        observations, rewards, done, info = await environment.step(actions)

        # Check coordination tracking
        assert len(environment.coordination_requests) >= 1

        # Rewards should include coordination bonus/penalty
        for reward in rewards.values():
            assert isinstance(reward, float)

    def test_performance_metrics_tracking(self, environment):
        """Test performance metrics tracking."""
        metrics = environment.performance_metrics

        expected_keys = [
            "total_episodes",
            "total_steps",
            "average_reward",
            "coordination_success_rate",
            "learning_progress",
            "quality_score",
        ]

        for key in expected_keys:
            assert key in metrics

    def test_environment_info(self, environment):
        """Test environment information retrieval."""
        info = environment.get_environment_info()

        assert "config" in info
        assert "current_state" in info
        assert "agents" in info
        assert "performance_metrics" in info

        # Check config info
        config_info = info["config"]
        assert config_info["num_agents"] == 3
        assert config_info["environment_type"] == EnvironmentType.MIXED.value


class TestScenarioTester:
    """Test scenario testing framework."""

    @pytest.fixture
    def scenario_tester(self):
        """Create scenario tester."""
        return ScenarioTester()

    def test_scenario_tester_initialization(self, scenario_tester):
        """Test scenario tester initialization."""
        assert len(scenario_tester.registered_scenarios) > 0  # Built-in scenarios
        assert len(scenario_tester.test_results) == 0

    def test_builtin_scenarios_registration(self, scenario_tester):
        """Test built-in scenarios are registered."""
        scenario_list = scenario_tester.get_scenario_list()

        # Check for expected built-in scenarios
        scenario_names = [s["name"] for s in scenario_list]

        expected_scenarios = [
            "action_conflict",
            "goal_conflict",
            "resource_conflict",
            "stress_test",
            "agent_failure",
            "learning_convergence",
        ]

        for expected in expected_scenarios:
            assert expected in scenario_names

    def test_custom_scenario_registration(self, scenario_tester):
        """Test custom scenario registration."""
        config = ScenarioConfig(
            scenario_type=ScenarioType.COORDINATION_CONFLICT,
            scenario_name="custom_test",
            description="Custom test scenario",
            num_runs=5,
        )

        scenario = TestScenario(scenario_id="custom_test", config=config)

        scenario_tester.register_scenario(scenario)

        assert "custom_test" in scenario_tester.registered_scenarios

        # Unregister
        scenario_tester.unregister_scenario("custom_test")
        assert "custom_test" not in scenario_tester.registered_scenarios

    @pytest.mark.asyncio
    async def test_scenario_execution(self, scenario_tester):
        """Test scenario execution."""

        # Create simple agent policies
        def simple_policy(observation):
            return random.randint(0, 9)

        agent_policies = {
            "generator": simple_policy,
            "validator": simple_policy,
            "curriculum": simple_policy,
        }

        # Run a simple scenario
        result = await scenario_tester.run_scenario("action_conflict", agent_policies)

        # Check result structure
        assert "scenario_id" in result
        assert "success" in result
        assert "total_runs" in result
        assert "metrics" in result
        assert result["scenario_id"] == "action_conflict"

    def test_scenario_configuration(self, scenario_tester):
        """Test scenario configuration."""
        scenario = scenario_tester.registered_scenarios["action_conflict"]
        config = scenario.config

        assert config.scenario_type == ScenarioType.COORDINATION_CONFLICT
        assert config.conflict_type == ConflictType.ACTION_CONFLICT
        assert config.num_runs > 0
        assert config.environment_config is not None


class TestPerformanceValidator:
    """Test performance validation framework."""

    @pytest.fixture
    def perf_config(self):
        """Create performance configuration."""
        return PerformanceConfig(
            improvement_threshold=0.30, confidence_level=0.95, min_sample_size=10
        )

    @pytest.fixture
    def validator(self, perf_config):
        """Create performance validator."""
        return PerformanceValidator(perf_config)

    def test_validator_initialization(self, validator):
        """Test validator initialization."""
        assert validator.config.improvement_threshold == 0.30
        assert len(validator.current_performance_data) == len(PerformanceMetric)
        assert len(validator.validation_results) == 0

    def test_performance_sample_addition(self, validator):
        """Test adding performance samples."""
        # Add samples for coordination success rate
        for i in range(20):
            validator.add_performance_sample(
                PerformanceMetric.COORDINATION_SUCCESS_RATE,
                0.8 + (i * 0.01),  # Gradually improving
            )

        samples = validator.current_performance_data[
            PerformanceMetric.COORDINATION_SUCCESS_RATE
        ]
        assert len(samples) == 20
        assert samples[0] == 0.8
        assert samples[-1] == 0.99

    def test_baseline_data_setting(self, validator):
        """Test setting baseline data."""
        baseline_data = {
            PerformanceMetric.COORDINATION_SUCCESS_RATE: [0.6, 0.65, 0.7],
            PerformanceMetric.AVERAGE_REWARD: [0.4, 0.45, 0.5],
        }

        validator.set_baseline_data(baseline_data)

        assert (
            len(
                validator.baseline_performance_data[
                    PerformanceMetric.COORDINATION_SUCCESS_RATE
                ]
            )
            == 3
        )
        assert (
            len(validator.baseline_performance_data[PerformanceMetric.AVERAGE_REWARD])
            == 3
        )

    @pytest.mark.asyncio
    async def test_performance_validation(self, validator):
        """Test performance validation."""
        # Add current performance data
        current_data = {
            PerformanceMetric.COORDINATION_SUCCESS_RATE: [0.85, 0.87, 0.89, 0.91, 0.93]
            * 4,  # 20 samples
            PerformanceMetric.AVERAGE_REWARD: [0.75, 0.77, 0.79, 0.81, 0.83] * 4,
        }

        validator.add_performance_batch(current_data)

        # Set baseline data
        baseline_data = {
            PerformanceMetric.COORDINATION_SUCCESS_RATE: [0.65] * 20,
            PerformanceMetric.AVERAGE_REWARD: [0.55] * 20,
        }

        validator.set_baseline_data(baseline_data)

        # Run validation
        results = await validator.validate_performance()

        # Check results structure
        assert "overall_pass" in results
        assert "metrics" in results
        assert "summary" in results

        # Check individual metric results
        coord_result = results["metrics"][
            PerformanceMetric.COORDINATION_SUCCESS_RATE.value
        ]
        assert "current_value" in coord_result
        assert "improvement_percent" in coord_result
        assert "meets_threshold" in coord_result

    @pytest.mark.asyncio
    async def test_performance_report_generation(self, validator):
        """Test performance report generation."""
        # Add some data
        validator.add_performance_sample(
            PerformanceMetric.COORDINATION_SUCCESS_RATE, 0.9
        )
        validator.add_performance_sample(PerformanceMetric.AVERAGE_REWARD, 0.8)

        # Generate report
        report = await validator.generate_performance_report()

        # Check report structure
        assert "report_timestamp" in report
        assert "validation_result" in report
        assert "detailed_metrics" in report
        assert "statistical_analysis" in report
        assert "recommendations" in report

    def test_performance_summary(self, validator):
        """Test performance summary."""
        # Add some data
        validator.add_performance_sample(
            PerformanceMetric.COORDINATION_SUCCESS_RATE, 0.85
        )
        validator.add_performance_sample(PerformanceMetric.AVERAGE_REWARD, 0.75)

        summary = validator.get_performance_summary()

        assert "data_points" in summary
        assert "current_averages" in summary
        assert "validation_status" in summary


class TestCoordinationTester:
    """Test coordination testing framework."""

    @pytest.fixture
    def coord_config(self):
        """Create coordination test configuration."""
        return CoordinationTestConfig(
            test_type=CoordinationTestType.CONSENSUS_BUILDING,
            num_agents=3,
            num_test_rounds=10,
            consensus_threshold=0.7,
        )

    @pytest.fixture
    def coordination_tester(self, coord_config):
        """Create coordination tester."""
        return CoordinationTester(coord_config)

    def test_coordination_tester_initialization(self, coordination_tester):
        """Test coordination tester initialization."""
        assert coordination_tester.config.num_agents == 3
        assert len(coordination_tester.test_results) == 0
        assert coordination_tester.current_test_id is None

    def test_mock_agent_creation(self, coordination_tester):
        """Test mock agent creation."""
        coordination_tester._create_mock_agents()

        assert len(coordination_tester.agents) == 3

        # Check agent properties
        for agent_id, agent in coordination_tester.agents.items():
            assert isinstance(agent, MockAgent)
            assert agent.agent_id == agent_id
            assert "action_preference" in agent.preferences
            assert agent.is_active

    @pytest.mark.asyncio
    async def test_mock_agent_behavior(self):
        """Test mock agent behavior."""
        agent = MockAgent("test_agent", "generator")

        # Test proposal
        context = {"round": 1, "other_agents": ["agent2", "agent3"]}
        proposal = await agent.propose_action(context)

        assert "agent_id" in proposal
        assert "action" in proposal
        assert "confidence" in proposal
        assert proposal["agent_id"] == "test_agent"

        # Test evaluation
        test_proposal = {"agent_id": "other_agent", "action": 5, "confidence": 0.8}
        evaluation = await agent.evaluate_proposal(test_proposal)

        assert "evaluator_id" in evaluation
        assert "satisfaction" in evaluation
        assert "support" in evaluation
        assert evaluation["evaluator_id"] == "test_agent"

    @pytest.mark.asyncio
    async def test_consensus_building_test(self, coordination_tester):
        """Test consensus building test."""
        result = await coordination_tester.run_coordination_test()

        # Check result structure
        assert result.test_type == CoordinationTestType.CONSENSUS_BUILDING
        assert isinstance(result.success, bool)
        assert result.resolution_time >= 0
        assert result.num_rounds > 0
        assert isinstance(result.consensus_achieved, bool)
        assert len(result.agent_satisfaction) == coordination_tester.config.num_agents

    @pytest.mark.asyncio
    async def test_conflict_resolution_test(self, coordination_tester):
        """Test conflict resolution test."""
        coordination_tester.config.test_type = CoordinationTestType.CONFLICT_RESOLUTION
        coordination_tester.config.conflict_scenario = (
            ConflictScenario.SIMPLE_DISAGREEMENT
        )

        result = await coordination_tester.run_coordination_test()

        assert result.test_type == CoordinationTestType.CONFLICT_RESOLUTION
        assert isinstance(result.conflict_resolved, bool)
        assert result.resolution_time >= 0

    @pytest.mark.asyncio
    async def test_comprehensive_test_suite(self, coordination_tester):
        """Test comprehensive coordination test suite."""
        results = await coordination_tester.run_comprehensive_test_suite()

        # Check results structure
        assert "overall_success" in results
        assert "total_test_types" in results
        assert "test_results" in results
        assert "recommendations" in results

        # Check that all test types were covered
        test_results = results["test_results"]
        expected_test_types = [
            "consensus_building",
            "conflict_resolution",
            "communication_reliability",
            "deadlock_prevention",
            "scalability",
            "fault_tolerance",
        ]

        for test_type in expected_test_types:
            assert test_type in test_results

    def test_coordination_test_summary(self, coordination_tester):
        """Test coordination test summary."""
        # Add some mock results
        from tests.marl_testing_framework.coordination_tester import (
            CoordinationTestResult,
        )

        test_result = CoordinationTestResult(
            test_id="test_1",
            test_type=CoordinationTestType.CONSENSUS_BUILDING,
            success=True,
            resolution_time=2.5,
            num_rounds=5,
            consensus_achieved=True,
            communication_overhead=0.1,
            agent_satisfaction={"agent_0": 0.8, "agent_1": 0.9},
            conflict_resolved=True,
        )

        coordination_tester.test_results.append(test_result)

        summary = coordination_tester.get_test_summary()

        assert summary["total_tests"] == 1
        assert summary["successful_tests"] == 1
        assert summary["success_rate"] == 1.0
        assert "average_resolution_time" in summary


class TestMARLTestingFrameworkIntegration:
    """Test integration between testing framework components."""

    @pytest.mark.asyncio
    async def test_environment_with_performance_validation(self):
        """Test mock environment with performance validation."""
        # Create environment
        env_config = MockEnvironmentConfig(
            environment_type=EnvironmentType.COORDINATION,
            num_agents=3,
            max_episodes=5,
            max_steps_per_episode=10,
        )
        environment = MockMARLEnvironment(env_config)

        # Create performance validator
        perf_config = PerformanceConfig(min_sample_size=5)
        validator = PerformanceValidator(perf_config)

        # Simple agent policies
        def simple_policy(observation):
            return random.randint(0, 9)

        agent_policies = {f"agent_{i}": simple_policy for i in range(3)}

        # Run episodes and collect performance data
        for episode in range(5):
            episode_result = await environment.run_episode(agent_policies)

            # Extract performance metrics
            validator.add_performance_sample(
                PerformanceMetric.COORDINATION_SUCCESS_RATE,
                episode_result.get("coordination_success_rate", 0.0),
            )
            validator.add_performance_sample(
                PerformanceMetric.AVERAGE_REWARD,
                episode_result.get("average_reward", 0.0),
            )

        # Validate performance
        results = await validator.validate_performance()

        assert "overall_pass" in results
        assert "metrics" in results

    @pytest.mark.asyncio
    async def test_scenario_with_coordination_testing(self):
        """Test scenario testing with coordination testing."""
        # Create scenario tester
        scenario_tester = ScenarioTester()

        # Create coordination tester
        coord_config = CoordinationTestConfig(
            test_type=CoordinationTestType.CONSENSUS_BUILDING,
            num_agents=3,
            num_test_rounds=5,
        )
        coordination_tester = CoordinationTester(coord_config)

        # Simple agent policies
        def simple_policy(observation):
            return random.randint(0, 9)

        agent_policies = {
            "generator": simple_policy,
            "validator": simple_policy,
            "curriculum": simple_policy,
        }

        # Run scenario test
        scenario_result = await scenario_tester.run_scenario(
            "action_conflict", agent_policies
        )

        # Run coordination test
        coordination_result = await coordination_tester.run_coordination_test()

        # Both should complete successfully
        assert "success" in scenario_result
        assert coordination_result.test_type == CoordinationTestType.CONSENSUS_BUILDING

    @pytest.mark.asyncio
    async def test_complete_testing_workflow(self):
        """Test complete testing workflow."""
        # Step 1: Create environment and run episodes
        env_config = MockEnvironmentConfig(
            environment_type=EnvironmentType.MIXED,
            num_agents=3,
            max_episodes=3,
            coordination_required=True,
        )
        environment = MockMARLEnvironment(env_config)

        # Step 2: Create performance validator
        perf_config = PerformanceConfig(min_sample_size=3)
        validator = PerformanceValidator(perf_config)

        # Step 3: Create coordination tester
        coord_config = CoordinationTestConfig(num_agents=3, num_test_rounds=3)
        coordination_tester = CoordinationTester(coord_config)

        # Step 4: Run testing workflow
        def test_policy(observation):
            return random.randint(0, 9)

        agent_policies = {f"agent_{i}": test_policy for i in range(3)}

        # Collect performance data
        performance_data = await validator.collect_performance_data(
            environment, agent_policies, num_episodes=3
        )

        # Run coordination tests
        coordination_result = await coordination_tester.run_coordination_test()

        # Validate performance
        validation_results = await validator.validate_performance()

        # Generate comprehensive report
        performance_report = await validator.generate_performance_report()

        # Verify workflow completion
        assert len(performance_data) > 0
        assert coordination_result.test_type == CoordinationTestType.CONSENSUS_BUILDING
        assert "overall_pass" in validation_results
        assert "recommendations" in performance_report


if __name__ == "__main__":
    pytest.main([__file__])
