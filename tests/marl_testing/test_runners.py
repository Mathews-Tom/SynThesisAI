"""Test Runners for MARL Testing Framework."""

# Standard Library
import asyncio
from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Any, Dict, List, Optional, Tuple

# SynThesisAI Modules
from .test_scenarios import BaseTestScenario
from utils.logging_config import get_logger


class ExecutionStrategy(Enum):
    """Test execution strategy enumeration."""

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    MIXED = "mixed"


class TestPriority(Enum):
    """Test priority levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class TestRunConfig:
    """Configuration for test runs."""

    execution_strategy: ExecutionStrategy = ExecutionStrategy.SEQUENTIAL
    max_parallel_tests: int = 4
    timeout_seconds: float = 300.0
    max_retries: int = 2
    retry_delay: float = 1.0


@dataclass
class TestResult:
    """Individual test result."""

    test_id: str
    scenario_type: str
    complexity: str
    success: bool
    execution_time: float
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestSuiteResult:
    """Test suite execution result."""

    suite_id: str
    start_time: float
    end_time: float
    total_tests: int
    passed_tests: int
    failed_tests: int
    skipped_tests: int
    execution_time: float
    test_results: List[TestResult] = field(default_factory=list)
    summary_metrics: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


class MARLTestRunner:
    """Main test runner for MARL testing framework."""

    def __init__(self, config: TestRunConfig):
        """Initialize MARL test runner."""
        self.config = config
        self.logger = get_logger(__name__)
        self.registered_tests: Dict[str, BaseTestScenario] = {}
        self.test_priorities: Dict[str, TestPriority] = {}
        self.is_running = False
        self.current_suite_result: Optional[TestSuiteResult] = None
        self.logger.info("MARL test runner initialized")

    def register_test(
        self,
        test_id: str,
        scenario: BaseTestScenario,
        priority: TestPriority = TestPriority.MEDIUM,
    ) -> None:
        """Register a test scenario."""
        self.registered_tests[test_id] = scenario
        self.test_priorities[test_id] = priority
        self.logger.info("Registered test: %s (priority: %s)", test_id, priority.value)

    async def run_test_suite(
        self, suite_id: str, test_ids: Optional[List[str]] = None
    ) -> TestSuiteResult:
        """Run a test suite."""
        if self.is_running:
            self.logger.warning("Test runner already running")
            return TestSuiteResult(
                suite_id=suite_id,
                start_time=time.time(),
                end_time=time.time(),
                total_tests=0,
                passed_tests=0,
                failed_tests=0,
                skipped_tests=0,
                execution_time=0.0,
            )

        self.logger.info("Starting test suite: %s", suite_id)
        self.is_running = True

        suite_start_time = time.time()
        self.current_suite_result = TestSuiteResult(
            suite_id=suite_id,
            start_time=suite_start_time,
            end_time=0.0,
            total_tests=0,
            passed_tests=0,
            failed_tests=0,
            skipped_tests=0,
            execution_time=0.0,
        )

        try:
            tests_to_run = self._select_tests_to_run(test_ids)
            self.current_suite_result.total_tests = len(tests_to_run)

            for test_id, scenario in tests_to_run:
                await self._run_single_test(test_id, scenario)

            suite_end_time = time.time()
            self.current_suite_result.end_time = suite_end_time
            self.current_suite_result.execution_time = suite_end_time - suite_start_time

            self._generate_suite_summary_metrics()

            self.logger.info(
                "Test suite complete: %s (%d/%d passed, %.2fs)",
                suite_id,
                self.current_suite_result.passed_tests,
                self.current_suite_result.total_tests,
                self.current_suite_result.execution_time,
            )

            return self.current_suite_result

        except Exception as e:
            self.logger.error("Test suite execution failed: %s", str(e))
            self.current_suite_result.errors.append(str(e))
            raise
        finally:
            self.is_running = False

    def _select_tests_to_run(
        self, test_ids: Optional[List[str]]
    ) -> List[Tuple[str, BaseTestScenario]]:
        """Select tests to run."""
        if test_ids:
            tests = [
                (test_id, self.registered_tests[test_id])
                for test_id in test_ids
                if test_id in self.registered_tests
            ]
        else:
            tests = list(self.registered_tests.items())

        self.logger.info("Selected %d tests to run", len(tests))
        return tests

    async def _run_single_test(self, test_id: str, scenario: BaseTestScenario) -> TestResult:
        """Run a single test."""
        self.logger.info("Starting test: %s", test_id)

        test_result = TestResult(
            test_id=test_id,
            scenario_type=scenario.config.scenario_type.value,
            complexity=scenario.config.complexity.value,
            success=False,
            execution_time=0.0,
        )

        for attempt in range(self.config.max_retries + 1):
            try:
                test_start_time = time.time()

                scenario_results = await asyncio.wait_for(
                    scenario.run_scenario(), timeout=self.config.timeout_seconds
                )

                test_execution_time = time.time() - test_start_time

                test_result.success = scenario_results.get("success", False)
                test_result.execution_time = test_execution_time
                test_result.metrics = scenario_results.get("metrics", {})

                if test_result.success:
                    break

                if attempt < self.config.max_retries:
                    self.logger.warning(
                        "Test %s failed (attempt %d/%d), retrying in %.1fs",
                        test_id,
                        attempt + 1,
                        self.config.max_retries + 1,
                        self.config.retry_delay,
                    )
                    await asyncio.sleep(self.config.retry_delay)

            except asyncio.TimeoutError:
                test_result.error_message = f"Test timed out after {self.config.timeout_seconds}s"
                self.logger.error("Test %s timed out", test_id)
                break

            except Exception as e:
                test_result.error_message = str(e)
                self.logger.error("Test %s failed: %s", test_id, str(e))

                if attempt == self.config.max_retries:
                    break

        if test_result.success:
            self.current_suite_result.passed_tests += 1
        else:
            self.current_suite_result.failed_tests += 1

        self.current_suite_result.test_results.append(test_result)

        self.logger.info(
            "Test complete: %s (success: %s, %.2fs)",
            test_id,
            test_result.success,
            test_result.execution_time,
        )

        return test_result

    def _generate_suite_summary_metrics(self) -> None:
        """Generate summary metrics."""
        if not self.current_suite_result:
            return

        total_tests = self.current_suite_result.total_tests
        passed_tests = self.current_suite_result.passed_tests
        failed_tests = self.current_suite_result.failed_tests

        summary_metrics = {
            "success_rate": passed_tests / max(total_tests, 1),
            "failure_rate": failed_tests / max(total_tests, 1),
            "average_execution_time": 0.0,
            "total_execution_time": self.current_suite_result.execution_time,
        }

        if self.current_suite_result.test_results:
            total_test_time = sum(
                result.execution_time for result in self.current_suite_result.test_results
            )
            summary_metrics["average_execution_time"] = total_test_time / len(
                self.current_suite_result.test_results
            )

        self.current_suite_result.summary_metrics = summary_metrics

    def get_current_status(self) -> Dict[str, Any]:
        """Get current runner status."""
        status = {
            "is_running": self.is_running,
            "registered_tests": len(self.registered_tests),
            "config": {
                "execution_strategy": self.config.execution_strategy.value,
                "max_parallel_tests": self.config.max_parallel_tests,
                "timeout_seconds": self.config.timeout_seconds,
                "max_retries": self.config.max_retries,
            },
        }

        if self.current_suite_result:
            status["current_suite"] = {
                "suite_id": self.current_suite_result.suite_id,
                "total_tests": self.current_suite_result.total_tests,
                "passed_tests": self.current_suite_result.passed_tests,
                "failed_tests": self.current_suite_result.failed_tests,
                "execution_time": (
                    time.time() - self.current_suite_result.start_time
                    if self.is_running
                    else self.current_suite_result.execution_time
                ),
            }

        return status
