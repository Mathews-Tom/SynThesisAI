"""Test Validators for MARL Testing Framework.

This module provides validation components for MARL test results
to ensure test quality and correctness.
"""

# Standard Library
import time

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# SynThesisAI Modules
from utils.logging_config import get_logger


@dataclass
class ValidationResult:
    """Result of test validation."""

    passed: bool
    errors: List[str]
    warnings: List[str]
    metrics: Dict[str, Any]
    validation_time: float


class MARLTestValidator(ABC):
    """Base class for MARL test validators."""

    def __init__(self):
        """Initialize validator."""
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")

    @abstractmethod
    async def validate_test_results(
        self, test_id: str, test_results: Dict[str, Any]
    ) -> ValidationResult:
        """Validate test results."""
        pass

    def _create_validation_result(
        self,
        passed: bool,
        errors: Optional[List[str]] = None,
        warnings: Optional[List[str]] = None,
        metrics: Optional[Dict[str, Any]] = None,
        validation_time: float = 0.0,
    ) -> ValidationResult:
        """Create validation result."""
        return ValidationResult(
            passed=passed,
            errors=errors or [],
            warnings=warnings or [],
            metrics=metrics or {},
            validation_time=validation_time,
        )


class CoordinationValidator(MARLTestValidator):
    """Validator for coordination test scenarios."""

    def __init__(self, min_success_rate: float = 0.8):
        """Initialize coordination validator."""
        super().__init__()
        self.min_success_rate = min_success_rate

    async def validate_test_results(
        self, test_id: str, test_results: Dict[str, Any]
    ) -> ValidationResult:
        """Validate coordination test results."""
        start_time = time.time()
        errors = []
        warnings = []
        metrics = {}

        # Check coordination success rate
        coordination_success_rate = test_results.get("coordination_success_rate", 0.0)
        if coordination_success_rate < self.min_success_rate:
            errors.append(
                f"Coordination success rate {coordination_success_rate:.2f} "
                f"below minimum {self.min_success_rate:.2f}"
            )

        # Check consensus achievement
        consensus_success_rate = test_results.get("consensus_success_rate", 0.0)
        if consensus_success_rate < 0.6:  # 60% minimum for consensus
            warnings.append(
                f"Consensus success rate {consensus_success_rate:.2f} " f"below recommended 0.60"
            )

        # Check response time
        avg_response_time = test_results.get("average_coordination_time", 0.0)
        if avg_response_time > 2.0:  # 2 second maximum
            warnings.append(
                f"Average response time {avg_response_time:.2f}s above recommended 2.0s"
            )

        # Collect metrics
        metrics.update(
            {
                "coordination_success_rate": coordination_success_rate,
                "consensus_success_rate": consensus_success_rate,
                "average_response_time": avg_response_time,
                "coordination_attempts": test_results.get("coordination_attempts", 0),
                "successful_coordinations": test_results.get("successful_coordinations", 0),
            }
        )

        validation_time = time.time() - start_time
        passed = len(errors) == 0

        self.logger.info(
            "Coordination validation complete for %s: %s (%d errors, %d warnings)",
            test_id,
            "PASSED" if passed else "FAILED",
            len(errors),
            len(warnings),
        )

        return self._create_validation_result(
            passed=passed,
            errors=errors,
            warnings=warnings,
            metrics=metrics,
            validation_time=validation_time,
        )


class PerformanceValidator(MARLTestValidator):
    """Validator for performance test scenarios."""

    def __init__(
        self,
        min_throughput: float = 50.0,
        max_response_time: float = 1.0,
        max_memory_mb: float = 1024.0,
        max_cpu_percent: float = 90.0,
    ):
        """Initialize performance validator."""
        super().__init__()
        self.min_throughput = min_throughput
        self.max_response_time = max_response_time
        self.max_memory_mb = max_memory_mb
        self.max_cpu_percent = max_cpu_percent

    async def validate_test_results(
        self, test_id: str, test_results: Dict[str, Any]
    ) -> ValidationResult:
        """Validate performance test results."""
        start_time = time.time()
        errors = []
        warnings = []
        metrics = {}

        # Check throughput
        throughput = test_results.get("throughput", 0.0)
        if throughput < self.min_throughput:
            errors.append(
                f"Throughput {throughput:.1f} ops/s "
                f"below minimum {self.min_throughput:.1f} ops/s"
            )

        # Check response time
        avg_response_time = test_results.get("average_response_time", 0.0)
        if avg_response_time > self.max_response_time:
            errors.append(
                f"Average response time {avg_response_time:.3f}s "
                f"above maximum {self.max_response_time:.3f}s"
            )

        # Check memory usage
        memory_usage = test_results.get("memory_usage", 0.0)
        if memory_usage > self.max_memory_mb:
            warnings.append(
                f"Memory usage {memory_usage:.1f}MB "
                f"above recommended {self.max_memory_mb:.1f}MB"
            )

        # Check CPU usage
        cpu_usage = test_results.get("cpu_usage", 0.0)
        if cpu_usage > self.max_cpu_percent:
            warnings.append(
                f"CPU usage {cpu_usage:.1f}% " f"above recommended {self.max_cpu_percent:.1f}%"
            )

        # Collect metrics
        metrics.update(
            {
                "throughput": throughput,
                "average_response_time": avg_response_time,
                "memory_usage": memory_usage,
                "cpu_usage": cpu_usage,
                "total_operations": test_results.get("total_operations", 0),
                "successful_operations": test_results.get("successful_operations", 0),
            }
        )

        validation_time = time.time() - start_time
        passed = len(errors) == 0

        self.logger.info(
            "Performance validation complete for %s: %s (%d errors, %d warnings)",
            test_id,
            "PASSED" if passed else "FAILED",
            len(errors),
            len(warnings),
        )

        return self._create_validation_result(
            passed=passed,
            errors=errors,
            warnings=warnings,
            metrics=metrics,
            validation_time=validation_time,
        )


class ConflictValidator(MARLTestValidator):
    """Validator for conflict resolution test scenarios."""

    def __init__(self, min_resolution_rate: float = 0.7):
        """Initialize conflict validator."""
        super().__init__()
        self.min_resolution_rate = min_resolution_rate

    async def validate_test_results(
        self, test_id: str, test_results: Dict[str, Any]
    ) -> ValidationResult:
        """Validate conflict resolution test results."""
        start_time = time.time()
        errors = []
        warnings = []
        metrics = {}

        # Check conflict resolution rate
        resolution_rate = test_results.get("conflict_resolution_rate", 0.0)
        if resolution_rate < self.min_resolution_rate:
            errors.append(
                f"Conflict resolution rate {resolution_rate:.2f} "
                f"below minimum {self.min_resolution_rate:.2f}"
            )

        # Check resolution time
        avg_resolution_time = test_results.get("average_resolution_time", 0.0)
        if avg_resolution_time > 3.0:  # 3 second maximum for conflict resolution
            warnings.append(
                f"Average resolution time {avg_resolution_time:.2f}s " f"above recommended 3.0s"
            )

        # Check unresolved conflicts
        unresolved_conflicts = test_results.get("unresolved_conflicts", 0)
        total_conflicts = test_results.get("conflicts_detected", 1)
        unresolved_rate = unresolved_conflicts / total_conflicts

        if unresolved_rate > 0.3:  # More than 30% unresolved
            warnings.append(
                f"Unresolved conflict rate {unresolved_rate:.2f} above recommended 0.30"
            )

        # Collect metrics
        metrics.update(
            {
                "conflict_resolution_rate": resolution_rate,
                "average_resolution_time": avg_resolution_time,
                "conflicts_detected": test_results.get("conflicts_detected", 0),
                "conflicts_resolved": test_results.get("conflicts_resolved", 0),
                "unresolved_conflicts": unresolved_conflicts,
                "unresolved_rate": unresolved_rate,
            }
        )

        validation_time = time.time() - start_time
        passed = len(errors) == 0

        self.logger.info(
            "Conflict validation complete for %s: %s (%d errors, %d warnings)",
            test_id,
            "PASSED" if passed else "FAILED",
            len(errors),
            len(warnings),
        )

        return self._create_validation_result(
            passed=passed,
            errors=errors,
            warnings=warnings,
            metrics=metrics,
            validation_time=validation_time,
        )
