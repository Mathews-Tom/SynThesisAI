#!/usr/bin/env python3
"""
Script to run all Phase 1 DSPy integration tests.

This script runs all unit tests and integration tests for Phase 1 of the DSPy integration.
"""

import logging
import subprocess
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_tests():
    """Run all Phase 1 tests."""
    logger.info("Running Phase 1 DSPy integration tests...")

    # Run unit tests
    logger.info("Running unit tests...")
    unit_test_result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-m",
            "pytest",
            "tests/unit_tests/test_dspy_adapter.py",
            "-v",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    # Run integration tests
    logger.info("Running integration tests...")
    integration_test_result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-m",
            "pytest",
            "tests/unit_tests/test_dspy_continuous_learning.py",
            "tests/unit_tests/test_dspy_feedback.py",
            "-v",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    # Run end-to-end tests
    logger.info("Running end-to-end tests...")
    e2e_test_result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-m",
            "pytest",
            "tests/end_to_end/test_phase1_comprehensive.py::TestPhase1QualityAssurance::test_quality_metrics_calculation",
            "-v",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    # Print results
    logger.info("Unit test results:")
    print(unit_test_result.stdout)
    if unit_test_result.stderr:
        print(unit_test_result.stderr)

    logger.info("Integration test results:")
    print(integration_test_result.stdout)
    if integration_test_result.stderr:
        print(integration_test_result.stderr)

    logger.info("End-to-end test results:")
    print(e2e_test_result.stdout)
    if e2e_test_result.stderr:
        print(e2e_test_result.stderr)

    # Check if all tests passed
    all_passed = (
        unit_test_result.returncode == 0
        and integration_test_result.returncode == 0
        and e2e_test_result.returncode == 0
    )

    if all_passed:
        logger.info("✅ All Phase 1 tests passed!")
        return 0

    logger.error("❌ Some tests failed. Please review the output.")
    return 1


if __name__ == "__main__":
    sys.exit(run_tests())
