"""MARL Testing Framework.

This module provides comprehensive testing infrastructure for the
Multi-Agent Reinforcement Learning coordination system.
"""

# SynThesisAI Modules
from .mock_environments import (
    MockAction,
    MockAgent,
    MockAgentType,
    MockCoordinationScenario,
    MockEnvironmentState,
    MockExperience,
    MockMARLEnvironment,
    MockObservation,
)
from .test_runners import (
    ExecutionStrategy,
    MARLTestRunner,
    TestPriority,
    TestResult,
    TestRunConfig,
    TestSuiteResult,
)
from .test_scenarios import (
    BaseTestScenario,
    ConflictTestScenario,
    CoordinationTestScenario,
    PerformanceTestScenario,
    ScenarioComplexity,
    ScenarioConfig,
    ScenarioType,
)
from .test_validators import (
    ConflictValidator,
    CoordinationValidator,
    MARLTestValidator,
    PerformanceValidator,
    ValidationResult,
)

__all__: list[str] = [
    # mock_environments
    "MockAgentType",
    "MockEnvironmentState",
    "MockAction",
    "MockObservation",
    "MockExperience",
    "MockAgent",
    "MockCoordinationScenario",
    "MockMARLEnvironment",
    # test_runners
    "ExecutionStrategy",
    "TestPriority",
    "TestRunConfig",
    "TestResult",
    "TestSuiteResult",
    "MARLTestRunner",
    # test_scenarios
    "ScenarioType",
    "ScenarioComplexity",
    "ScenarioConfig",
    "BaseTestScenario",
    "CoordinationTestScenario",
    "ConflictTestScenario",
    "PerformanceTestScenario",
    # test_validators
    "MARLTestValidator",
    "ValidationResult",
    "CoordinationValidator",
    "PerformanceValidator",
    "ConflictValidator",
]
