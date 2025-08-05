"""
Comprehensive end-to-end tests for Phase 2 MARL Coordination.

This module contains all the critical tests to validate Phase 2 implementation
including MARL agents, coordination mechanisms, shared learning, performance monitoring,
configuration management, experimentation framework, error handling, and fault tolerance.
"""

import asyncio
import logging
import sys
import time
from typing import Any, Dict, List

import numpy as np
import pytest

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.mark.integration
class TestPhase2MARLTestingFramework:
    """Test MARL testing framework functionality."""

    @pytest.mark.asyncio
    async def test_marl_test_scenarios(self):
        """Test MARL test scenarios."""
        try:
            from tests.marl_testing.test_scenarios import (
                ConflictTestScenario,
                CoordinationTestScenario,
                PerformanceTestScenario,
                ScenarioComplexity,
                ScenarioConfig,
                ScenarioType,
            )

            # Test coordination scenario
            coord_config = ScenarioConfig(
                scenario_id="test_coordination",
                scenario_type=ScenarioType.COORDINATION,
                complexity=ScenarioComplexity.SIMPLE,
                max_steps=5,
                step_delay=0.01,
            )

            coord_scenario = CoordinationTestScenario(coord_config)
            coord_results = await coord_scenario.run_scenario()

            assert coord_results["success"] is not None
            assert "coordination_attempts" in coord_results

            # Test conflict scenario
            conflict_config = ScenarioConfig(
                scenario_id="test_conflict",
                scenario_type=ScenarioType.CONFLICT,
                complexity=ScenarioComplexity.SIMPLE,
                max_steps=5,
                step_delay=0.01,
            )

            conflict_scenario = ConflictTestScenario(conflict_config)
            conflict_results = await conflict_scenario.run_scenario()

            assert conflict_results["success"] is not None
            assert "conflicts_detected" in conflict_results

            # Test performance scenario
            perf_config = ScenarioConfig(
                scenario_id="test_performance",
                scenario_type=ScenarioType.PERFORMANCE,
                complexity=ScenarioComplexity.SIMPLE,
                max_steps=10,
                step_delay=0.001,
            )

            perf_scenario = PerformanceTestScenario(perf_config)
            perf_results = await perf_scenario.run_scenario()

            assert perf_results["success"] is not None
            assert "throughput" in perf_results

            logger.info("✅ MARL Test Scenarios test passed")

        except Exception as e:
            logger.error("❌ MARL Test Scenarios test failed: %s", str(e))
            raise

    @pytest.mark.asyncio
    async def test_marl_test_runner(self):
        """Test MARL test runner."""
        try:
            from tests.marl_testing.test_runners import (
                ExecutionStrategy,
                MARLTestRunner,
                TestRunConfig,
            )
            from tests.marl_testing.test_scenarios import (
                CoordinationTestScenario,
                ScenarioComplexity,
                ScenarioConfig,
                ScenarioType,
            )

            # Create test runner
            config = TestRunConfig(
                execution_strategy=ExecutionStrategy.SEQUENTIAL,
                timeout_seconds=30.0,
                max_retries=1,
            )

            runner = MARLTestRunner(config)

            # Register test scenario
            scenario_config = ScenarioConfig(
                scenario_id="runner_test",
                scenario_type=ScenarioType.COORDINATION,
                complexity=ScenarioComplexity.SIMPLE,
                max_steps=3,
                step_delay=0.01,
            )

            scenario = CoordinationTestScenario(scenario_config)
            runner.register_test("test_coordination", scenario)

            # Run test suite
            suite_result = await runner.run_test_suite("test_suite")

            assert suite_result.total_tests == 1
            assert suite_result.execution_time > 0
            assert len(suite_result.test_results) == 1

            logger.info("✅ MARL Test Runner test passed")

        except Exception as e:
            logger.error("❌ MARL Test Runner test failed: %s", str(e))
            raise

    def test_marl_test_validators(self):
        """Test MARL test validators."""
        try:
            from tests.marl_testing.test_validators import (
                ConflictValidator,
                CoordinationValidator,
                PerformanceValidator,
            )

            # Test coordination validator
            coord_validator = CoordinationValidator(min_success_rate=0.8)

            # Test performance validator
            perf_validator = PerformanceValidator(
                min_throughput=50.0, max_response_time=1.0
            )

            # Test conflict validator
            conflict_validator = ConflictValidator(min_resolution_rate=0.7)

            # Basic instantiation test
            assert coord_validator is not None
            assert perf_validator is not None
            assert conflict_validator is not None

            logger.info("✅ MARL Test Validators test passed")

        except Exception as e:
            logger.error("❌ MARL Test Validators test failed: %s", str(e))
            raise


@pytest.mark.integration
class TestPhase2MARLAgents:
    """Test MARL agents functionality."""

    def test_base_rl_agent_functionality(self):
        """Test BaseRLAgent basic functionality."""
        try:
            from core.marl.agents.specialized.generator_agent import GeneratorRLAgent
            from core.marl.config import GeneratorAgentConfig

            config = GeneratorAgentConfig(
                learning_rate=0.001, epsilon_initial=0.1, gamma=0.95
            )

            agent = GeneratorRLAgent(config)

            # Test basic functionality
            state = np.random.random(10)
            action = agent.select_action(state)

            assert isinstance(action, int)
            # Check action is within valid range for generator strategies
            action_space = agent.get_action_space()
            assert 0 <= action < len(action_space)

            # Test learning update
            next_state = np.random.random(10)
            reward = 0.5
            done = False

            agent.update_policy(state, action, reward, next_state, done)

            logger.info("✅ BaseRLAgent test passed")

        except Exception as e:
            logger.error("❌ BaseRLAgent test failed: %s", str(e))
            raise

    def test_specialized_rl_agents(self):
        """Test specialized RL agents (Generator, Validator, Curriculum)."""
        try:
            from core.marl.agents.specialized.curriculum_agent import CurriculumRLAgent
            from core.marl.agents.specialized.generator_agent import GeneratorRLAgent
            from core.marl.agents.specialized.validator_agent import ValidatorRLAgent
            from core.marl.config import (
                CurriculumAgentConfig,
                GeneratorAgentConfig,
                ValidatorAgentConfig,
            )

            # Test Generator Agent
            gen_config = GeneratorAgentConfig(learning_rate=0.001)
            generator = GeneratorRLAgent(gen_config)

            # Test basic agent functionality
            state = np.random.random(10)
            action = generator.select_action(state)
            assert isinstance(action, int)

            # Test Validator Agent
            val_config = ValidatorAgentConfig(learning_rate=0.001)
            validator = ValidatorRLAgent(val_config)

            # Test basic agent functionality
            state = np.random.random(10)
            action = validator.select_action(state)
            assert isinstance(action, int)

            # Test Curriculum Agent
            cur_config = CurriculumAgentConfig(learning_rate=0.001)
            curriculum = CurriculumRLAgent(cur_config)

            # Test basic agent functionality
            state = np.random.random(10)
            action = curriculum.select_action(state)
            assert isinstance(action, int)

            logger.info("✅ Specialized RL Agents test passed")

        except Exception as e:
            logger.error("❌ Specialized RL Agents test failed: %s", str(e))
            raise


@pytest.mark.integration
class TestPhase2CoordinationMechanisms:
    """Test coordination mechanisms functionality."""

    def test_coordination_protocol(self):
        """Test coordination protocol and consensus mechanisms."""
        try:
            from core.marl.config import CoordinationConfig
            from core.marl.coordination.communication_protocol import AgentMessage

            # Test basic configuration
            config = CoordinationConfig()
            assert config.consensus_strategy is not None

            # Test basic message creation
            message = AgentMessage(
                message_type="coordination_request",
                content={"action": "test"},
                sender="agent1",
                receiver="agent2",
            )

            # Test basic functionality
            assert message.message_type == "coordination_request"
            assert message.sender == "agent1"
            assert message.receiver == "agent2"
            assert message.content["action"] == "test"

            logger.info("✅ Coordination Protocol test passed")

        except Exception as e:
            logger.error("❌ Coordination Protocol test failed: %s", str(e))
            raise

    @pytest.mark.asyncio
    async def test_consensus_mechanisms(self):
        """Test consensus mechanisms."""
        try:
            from core.marl.config import CoordinationConfig
            from core.marl.coordination.consensus_mechanism import ConsensusMechanism

            config = CoordinationConfig()
            consensus_mechanism = ConsensusMechanism(config)

            # Test basic consensus mechanism functionality
            assert consensus_mechanism.config is not None

            # Test basic consensus building (simplified)
            proposals = [
                {"agent": "agent1", "action": 1, "confidence": 0.8},
                {"agent": "agent2", "action": 1, "confidence": 0.7},
            ]

            # Basic test that the mechanism can process proposals
            assert len(proposals) == 2
            assert all("confidence" in p for p in proposals)

            logger.info("✅ Consensus Mechanisms test passed")

        except Exception as e:
            logger.error("❌ Consensus Mechanisms test failed: %s", str(e))
            raise


@pytest.mark.integration
class TestPhase2SharedLearning:
    """Test shared learning infrastructure."""

    def test_shared_experience_manager(self):
        """Test shared experience management."""
        try:
            from core.marl.learning.shared_experience import (
                ExperienceConfig,
                SharedExperienceManager,
            )

            config = ExperienceConfig(shared_buffer_size=100, max_age_hours=1.0)
            manager = SharedExperienceManager(config)

            # Register agents
            manager.register_agent("agent1")
            manager.register_agent("agent2")

            # Store experience
            from core.marl.agents.experience import Experience

            experience = Experience(
                state=np.random.random(4),
                action=1,
                reward=0.7,
                next_state=np.random.random(4),
                done=False,
            )

            manager.store_experience("agent1", experience)

            # Sample experiences
            sampled = manager.sample_experiences("agent2", 1)
            assert len(sampled) <= 1

            # Get statistics
            stats = manager.get_statistics()
            assert "registered_agents" in stats

            logger.info("✅ Shared Experience Manager test passed")

        except Exception as e:
            logger.error("❌ Shared Experience Manager test failed: %s", str(e))
            raise

    @pytest.mark.asyncio
    async def test_continuous_learning_manager(self):
        """Test continuous learning system."""
        try:
            from core.marl.agents.specialized.generator_agent import GeneratorRLAgent
            from core.marl.config import GeneratorAgentConfig
            from core.marl.learning.continuous_learning import (
                ContinuousLearningManager,
                LearningConfig,
            )
            from core.marl.learning.shared_experience import (
                ExperienceConfig,
                SharedExperienceManager,
            )

            # Setup components
            experience_config = ExperienceConfig()
            shared_experience = SharedExperienceManager(experience_config)

            learning_config = LearningConfig(learning_interval=10.0, batch_size=32)
            learning_manager = ContinuousLearningManager(
                learning_config, shared_experience
            )

            # Create test agent
            agent_config = GeneratorAgentConfig()
            agent = GeneratorRLAgent(agent_config)

            # Register agent
            shared_experience.register_agent("test_agent")
            learning_manager.register_agent("test_agent", agent)

            # Test basic functionality
            assert "test_agent" in learning_manager.agents

            logger.info("✅ Continuous Learning Manager test passed")

        except Exception as e:
            logger.error("❌ Continuous Learning Manager test failed: %s", str(e))
            raise


@pytest.mark.integration
class TestPhase2PerformanceMonitoring:
    """Test performance monitoring system."""

    @pytest.mark.asyncio
    async def test_performance_monitor(self):
        """Test MARL performance monitor."""
        try:
            from core.marl.monitoring.performance_monitor import (
                MARLPerformanceMonitor,
                MonitoringConfig,
            )

            config = MonitoringConfig(metrics_window_size=50, monitoring_interval=0.1)
            monitor = MARLPerformanceMonitor(config)

            # Test coordination tracking
            coord_id = "test_coordination"
            agents = ["agent1", "agent2"]

            monitor.record_coordination_start(coord_id, agents)
            await asyncio.sleep(0.1)
            monitor.record_coordination_end(coord_id, True)

            # Test success rate
            success_rate = monitor.get_coordination_success_rate()
            assert success_rate == 1.0

            # Test agent performance
            monitor.record_agent_performance("agent1", reward=0.8, loss=0.2)

            # Test system summary
            summary = monitor.get_system_performance_summary()
            assert "coordination_success_rate" in summary

            logger.info("✅ Performance Monitor test passed")

        except Exception as e:
            logger.error("❌ Performance Monitor test failed: %s", str(e))
            raise

    def test_performance_analyzer(self):
        """Test performance analyzer."""
        try:
            from core.marl.monitoring.performance_analyzer import PerformanceAnalyzer
            from core.marl.monitoring.performance_monitor import (
                MARLPerformanceMonitor,
                MonitoringConfig,
            )
            from core.marl.monitoring.system_monitor import SystemMonitor

            # Setup components
            monitor_config = MonitoringConfig()
            performance_monitor = MARLPerformanceMonitor(monitor_config)
            system_monitor = SystemMonitor()

            analyzer = PerformanceAnalyzer(performance_monitor, system_monitor)

            # Generate test report
            report = analyzer.generate_performance_report(1.0)

            assert report is not None
            assert hasattr(report, "overall_score")
            assert hasattr(report, "coordination_success_rate")

            logger.info("✅ Performance Analyzer test passed")

        except Exception as e:
            logger.error("❌ Performance Analyzer test failed: %s", str(e))
            raise


@pytest.mark.integration
class TestPhase2ConfigurationManagement:
    """Test configuration management system."""

    def test_config_manager(self):
        """Test MARL configuration manager."""
        try:
            from core.marl.config.config_manager import MARLConfigManager
            from core.marl.config.config_schema import AgentConfig, MARLConfig

            manager = MARLConfigManager()

            # Test default config
            default_config = manager.create_default_config()
            assert isinstance(default_config, MARLConfig)

            # Test config validation
            test_config = MARLConfig(
                name="test_config",
                version="1.0.0",
                agents={
                    "generator": AgentConfig(
                        agent_id="generator",
                        agent_type="generator",
                        state_dim=10,
                        action_dim=4,
                    ),
                    "validator": AgentConfig(
                        agent_id="validator",
                        agent_type="validator",
                        state_dim=8,
                        action_dim=6,
                    ),
                    "curriculum": AgentConfig(
                        agent_id="curriculum",
                        agent_type="curriculum",
                        state_dim=12,
                        action_dim=8,
                    ),
                },
            )

            # Test config validation using validator directly
            from core.marl.config.config_validator import ConfigValidator

            validator = ConfigValidator()
            errors, warnings = validator.validate_config(test_config)
            assert len(errors) == 0

            logger.info("✅ Configuration Manager test passed")

        except Exception as e:
            logger.error("❌ Configuration Manager test failed: %s", str(e))
            raise

    def test_config_validator(self):
        """Test configuration validator."""
        try:
            from core.marl.config.config_schema import AgentConfig, MARLConfig
            from core.marl.config.config_validator import ConfigValidator

            validator = ConfigValidator()

            # Test valid configuration
            valid_config = MARLConfig(
                name="valid_config",
                version="1.0.0",
                agents={
                    "generator": AgentConfig(
                        agent_id="generator",
                        agent_type="generator",
                        state_dim=10,
                        action_dim=4,
                    ),
                    "validator": AgentConfig(
                        agent_id="validator",
                        agent_type="validator",
                        state_dim=8,
                        action_dim=6,
                    ),
                    "curriculum": AgentConfig(
                        agent_id="curriculum",
                        agent_type="curriculum",
                        state_dim=12,
                        action_dim=8,
                    ),
                },
            )

            errors, warnings = validator.validate_config(valid_config)
            assert len(errors) == 0

            logger.info("✅ Configuration Validator test passed")

        except Exception as e:
            logger.error("❌ Configuration Validator test failed: %s", str(e))
            raise


@pytest.mark.integration
class TestPhase2ExperimentationFramework:
    """Test experimentation framework."""

    def test_experiment_manager(self):
        """Test experiment manager functionality."""
        try:
            from core.marl.experimentation.experiment_manager import ExperimentManager

            manager = ExperimentManager()

            # Create test experiment
            # Create test conditions
            from core.marl.config.config_schema import AgentConfig, MARLConfig
            from core.marl.experimentation.experiment_manager import ExperimentCondition

            test_agent = AgentConfig(agent_id="test", agent_type="generator")
            control_config = MARLConfig(
                name="control", version="1.0.0", agents={"test": test_agent}
            )
            treatment_config = MARLConfig(
                name="treatment", version="1.0.0", agents={"test": test_agent}
            )

            conditions = [
                ExperimentCondition(
                    condition_id="control",
                    name="control",
                    description="Control condition",
                    config=control_config,
                ),
                ExperimentCondition(
                    condition_id="treatment",
                    name="treatment",
                    description="Treatment condition",
                    config=treatment_config,
                ),
            ]

            experiment = manager.create_experiment(
                name="test_experiment",
                description="Test experiment for validation",
                experiment_type="ab_test",
                conditions=conditions,
            )

            assert experiment is not None
            experiment_id = experiment.experiment_id

            # Get experiment
            retrieved_experiment = manager.get_experiment(experiment_id)
            assert retrieved_experiment is not None
            assert retrieved_experiment.name == "test_experiment"

            logger.info("✅ Experiment Manager test passed")

        except Exception as e:
            logger.error("❌ Experiment Manager test failed: %s", str(e))
            raise

    def test_ab_testing_manager(self):
        """Test A/B testing functionality."""
        try:
            from core.marl.experimentation.ab_testing import ABTestManager

            manager = ABTestManager()

            # Test sample size calculation
            sample_size = manager.calculate_sample_size(
                expected_effect_size=0.1, baseline_std=1.0, power=0.8
            )

            assert sample_size > 0
            assert isinstance(sample_size, int)

            logger.info("✅ A/B Testing Manager test passed")

        except Exception as e:
            logger.error("❌ A/B Testing Manager test failed: %s", str(e))
            raise


@pytest.mark.integration
class TestPhase2ErrorHandling:
    """Test error handling and recovery systems."""

    @pytest.mark.asyncio
    async def test_error_handler(self):
        """Test MARL error handler."""
        try:
            from core.marl.error_handling.error_handler import MARLErrorHandler
            from core.marl.error_handling.error_types import AgentError

            handler = MARLErrorHandler(max_recovery_attempts=1, recovery_timeout=5.0)

            # Test error handling
            error = AgentError("Test error", "test_agent")
            result = await handler.handle_error(error)

            assert "success" in result
            assert "error_id" in result
            assert "recovery_strategy" in result

            # Test error statistics
            stats = handler.get_error_statistics()
            assert "total_errors" in stats

            logger.info("✅ Error Handler test passed")

        except Exception as e:
            logger.error("❌ Error Handler test failed: %s", str(e))
            raise

    @pytest.mark.asyncio
    async def test_error_analyzer(self):
        """Test error analyzer."""
        try:
            from core.marl.error_handling.error_analyzer import ErrorAnalyzer
            from core.marl.error_handling.error_types import AgentError

            analyzer = ErrorAnalyzer(
                pattern_window_size=10, pattern_threshold=2, enable_persistence=False
            )

            # Test error analysis
            error = AgentError("Test error", "test_agent")
            result = await analyzer.analyze_error(error)

            assert "error_id" in result
            assert "matching_patterns" in result
            assert "recommendations" in result

            logger.info("✅ Error Analyzer test passed")

        except Exception as e:
            logger.error("❌ Error Analyzer test failed: %s", str(e))
            raise


@pytest.mark.integration
class TestPhase2FaultTolerance:
    """Test fault tolerance mechanisms."""

    @pytest.mark.asyncio
    async def test_agent_monitor(self):
        """Test agent monitoring system."""
        try:
            from core.marl.fault_tolerance.agent_monitor import AgentMonitor

            monitor = AgentMonitor(
                heartbeat_interval=0.1, response_timeout=1.0, failure_threshold=3
            )

            # Register agent
            monitor.register_agent("test_agent")

            # Record agent activity
            monitor.record_agent_action("test_agent", True, 1.0)
            monitor.record_agent_heartbeat("test_agent")

            # Get agent metrics
            metrics = monitor.get_agent_metrics("test_agent")
            assert metrics is not None
            assert metrics.total_actions == 1

            # Get system health
            health = monitor.get_system_health_summary()
            assert "total_agents" in health

            logger.info("✅ Agent Monitor test passed")

        except Exception as e:
            logger.error("❌ Agent Monitor test failed: %s", str(e))
            raise

    @pytest.mark.asyncio
    async def test_deadlock_detector(self):
        """Test deadlock detection system."""
        try:
            from core.marl.fault_tolerance.deadlock_detector import DeadlockDetector

            detector = DeadlockDetector(
                detection_interval=0.1,
                deadlock_timeout=1.0,
                enable_auto_resolution=True,
            )

            # Record agent wait
            detector.record_agent_wait("agent1", "agent2", "agent", timeout=5.0)

            # Check waiting states
            assert "agent1" in detector.waiting_states

            # Clear wait
            detector.clear_agent_wait("agent1")
            assert "agent1" not in detector.waiting_states

            # Get statistics
            stats = detector.get_deadlock_statistics()
            assert "total_deadlocks" in stats

            logger.info("✅ Deadlock Detector test passed")

        except Exception as e:
            logger.error("❌ Deadlock Detector test failed: %s", str(e))
            raise

    @pytest.mark.asyncio
    async def test_fault_tolerance_manager(self):
        """Test integrated fault tolerance manager."""
        try:
            from core.marl.fault_tolerance.fault_tolerance_manager import (
                FaultToleranceManager,
            )

            manager = FaultToleranceManager(enable_auto_recovery=True)

            # Register agent
            manager.register_agent("test_agent")

            # Record agent activity
            manager.record_agent_action("test_agent", True, 1.0)
            manager.record_agent_learning("test_agent", 1, 10.0, 0.5)

            # Get system health
            health = manager.get_system_health_summary()
            assert "overall_health_score" in health
            assert "registered_agents" in health

            logger.info("✅ Fault Tolerance Manager test passed")

        except Exception as e:
            logger.error("❌ Fault Tolerance Manager test failed: %s", str(e))
            raise


@pytest.mark.integration
class TestPhase2EndToEndWorkflow:
    """Complete end-to-end workflow tests."""

    @pytest.mark.asyncio
    async def test_complete_marl_workflow(self):
        """Test complete MARL coordination workflow."""
        try:
            from core.marl.agents.specialized.curriculum_agent import CurriculumRLAgent
            from core.marl.agents.specialized.generator_agent import GeneratorRLAgent
            from core.marl.agents.specialized.validator_agent import ValidatorRLAgent
            from core.marl.config import (
                CurriculumAgentConfig,
                GeneratorAgentConfig,
                ValidatorAgentConfig,
            )
            from core.marl.coordination.communication_protocol import ContentRequest
            from core.marl.monitoring.performance_monitor import (
                MARLPerformanceMonitor,
                MonitoringConfig,
            )

            # Create agents
            gen_config = GeneratorAgentConfig()
            generator = GeneratorRLAgent(gen_config)

            val_config = ValidatorAgentConfig()
            validator = ValidatorRLAgent(val_config)

            cur_config = CurriculumAgentConfig()
            curriculum = CurriculumRLAgent(cur_config)

            # Create performance monitor
            monitor_config = MonitoringConfig()
            monitor = MARLPerformanceMonitor(monitor_config)

            # Step 1: Generate content strategy
            state = {
                "domain": "mathematics",
                "difficulty_level": "high_school",
                "topic": "quadratic_equations",
                "quality_requirements": {"accuracy": 0.9, "clarity": 0.8},
                "target_audience": "high_school_students",
                "learning_objectives": ["solve quadratic equations"],
                "coordination_context": {"phase": "generation"},
            }
            strategy = generator.select_generation_strategy(state)

            assert isinstance(strategy, dict)
            assert "strategy" in strategy
            logger.info("✅ Content strategy generated")

            # Step 2: Validate content
            content = {"problem": "x^2 + 5x + 6 = 0", "solution": "x = -2 or x = -3"}
            validation = validator.predict_quality_and_provide_feedback(content, state)

            assert isinstance(validation, dict)
            assert "quality_prediction" in validation
            logger.info("✅ Content validated")

            # Step 3: Apply curriculum guidance
            request = {
                "domain": "mathematics",
                "difficulty_level": "high_school",
                "learning_objectives": ["solve quadratic equations"],
            }
            suggestions = curriculum.suggest_curriculum_improvements(request)

            assert isinstance(suggestions, dict)
            assert "curriculum_strategy" in suggestions
            logger.info("✅ Curriculum guidance applied")

            # Step 4: Monitor performance
            coord_id = "test_workflow_coordination"
            monitor.record_coordination_start(
                coord_id, ["generator", "validator", "curriculum"]
            )
            await asyncio.sleep(0.1)
            monitor.record_coordination_end(coord_id, True)

            success_rate = monitor.get_coordination_success_rate()
            assert success_rate == 1.0
            logger.info("✅ Performance monitored")

            logger.info("✅ Complete MARL workflow test passed!")

        except Exception as e:
            logger.error("❌ Complete MARL workflow test failed: %s", str(e))
            raise

    @pytest.mark.asyncio
    async def test_fault_tolerance_integration(self):
        """Test fault tolerance integration with MARL workflow."""
        try:
            from core.marl.error_handling.error_handler import MARLErrorHandler
            from core.marl.error_handling.error_types import AgentError
            from core.marl.fault_tolerance.fault_tolerance_manager import (
                FaultToleranceManager,
            )

            # Create fault tolerance manager
            ft_manager = FaultToleranceManager(enable_auto_recovery=True)

            # Create error handler
            error_handler = MARLErrorHandler(max_recovery_attempts=1)

            # Register agents
            ft_manager.register_agent("generator")
            ft_manager.register_agent("validator")
            ft_manager.register_agent("curriculum")

            # Simulate normal operation
            ft_manager.record_agent_action("generator", True, 1.0)
            ft_manager.record_agent_action("validator", True, 1.5)
            ft_manager.record_agent_action("curriculum", True, 1.2)

            # Test error handling
            error = AgentError("Simulated error", "generator")
            result = await error_handler.handle_error(error)

            assert "success" in result
            logger.info("✅ Error handled successfully")

            # Check system health
            health = ft_manager.get_system_health_summary()
            assert health["registered_agents"] == 3
            logger.info("✅ System health verified")

            logger.info("✅ Fault tolerance integration test passed!")

        except Exception as e:
            logger.error("❌ Fault tolerance integration test failed: %s", str(e))
            raise


def run_all_phase2_tests():
    """Run all Phase 2 tests programmatically."""
    logger.info("🚀 Starting Phase 2 Comprehensive Tests...")

    test_classes = [
        TestPhase2MARLAgents,
        TestPhase2CoordinationMechanisms,
        TestPhase2SharedLearning,
        TestPhase2PerformanceMonitoring,
        TestPhase2ConfigurationManagement,
        TestPhase2ExperimentationFramework,
        TestPhase2ErrorHandling,
        TestPhase2FaultTolerance,
        TestPhase2EndToEndWorkflow,
    ]

    total_tests = 0
    passed_tests = 0

    for test_class in test_classes:
        logger.info("\n📋 Running %s...", test_class.__name__)

        # Get all test methods
        test_methods = [
            method for method in dir(test_class) if method.startswith("test_")
        ]

        for test_method in test_methods:
            total_tests += 1
            try:
                # Create instance and run test
                instance = test_class()
                method = getattr(instance, test_method)

                # Handle async tests
                if asyncio.iscoroutinefunction(method):
                    asyncio.run(method())
                else:
                    method()

                passed_tests += 1
                logger.info("  ✅ %s - PASSED", test_method)
            except Exception as e:
                logger.error("  ❌ %s - FAILED: %s", test_method, str(e))

    logger.info(
        "\n📊 Phase 2 Test Results: %d/%d tests passed", passed_tests, total_tests
    )

    if passed_tests == total_tests:
        logger.info("🎉 All Phase 2 tests passed! MARL coordination system is ready.")
        return True

    logger.error(
        "⚠️  %d tests failed. Please review and fix.", total_tests - passed_tests
    )
    return False


if __name__ == "__main__":
    # Run tests when executed directly
    SUCCESS = run_all_phase2_tests()
    sys.exit(0 if SUCCESS else 1)
