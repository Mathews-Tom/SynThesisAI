#!/usr/bin/env python3
"""
Script to run all Phase 2 MARL coordination tests.

This script runs all unit tests and integration tests for Phase 2 of the MARL coordination system.
"""

import logging
import subprocess
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_tests():
    """Run all Phase 2 tests."""
    logger.info("🚀 Running Phase 2 MARL coordination tests...")

    # Define test categories and their corresponding test files
    test_categories = {
        "MARL Agents": [
            "tests/unit_tests/test_base_agent.py",
            "tests/unit_tests/test_generator_agent.py",
            "tests/unit_tests/test_validator_agent.py",
            "tests/unit_tests/test_curriculum_agent.py",
        ],
        "Coordination Mechanisms": [
            "tests/unit_tests/test_coordination_policy.py",
            "tests/unit_tests/test_consensus_manager.py",
            "tests/unit_tests/test_communication_protocol.py",
        ],
        "Shared Learning": [
            "tests/unit_tests/test_shared_experience.py",
            "tests/unit_tests/test_continuous_learning.py",
        ],
        "Performance Monitoring": [
            "tests/unit_tests/test_performance_monitor.py",
            "tests/unit_tests/test_performance_analyzer.py",
            "tests/unit_tests/test_performance_reporter.py",
        ],
        "Configuration Management": [
            "tests/unit_tests/test_config_manager.py",
            "tests/unit_tests/test_config_validator.py",
        ],
        "Experimentation Framework": [
            "tests/unit_tests/test_experiment_manager.py",
            "tests/unit_tests/test_ab_testing.py",
        ],
        "Error Handling": [
            "tests/unit_tests/test_error_handling.py",
        ],
        "Fault Tolerance": [
            "tests/unit_tests/test_fault_tolerance.py",
        ],
    }

    # Track overall results
    all_results = {}
    overall_success = True

    # Run tests by category
    for category, test_files in test_categories.items():
        logger.info(f"\n📋 Running {category} tests...")

        # Filter existing test files
        existing_files = []
        for test_file in test_files:
            if Path(test_file).exists():
                existing_files.append(test_file)
            else:
                logger.warning(f"⚠️  Test file not found: {test_file}")

        if not existing_files:
            logger.warning(f"⚠️  No test files found for {category}")
            continue

        # Run tests for this category
        result = subprocess.run(
            ["uv", "run", "python", "-m", "pytest"]
            + existing_files
            + ["-v", "--tb=short"],
            capture_output=True,
            text=True,
            check=False,
        )

        all_results[category] = result

        # Print results
        if result.returncode == 0:
            logger.info(f"✅ {category} tests PASSED")
        else:
            logger.error(f"❌ {category} tests FAILED")
            overall_success = False

        # Print test output (abbreviated)
        if result.stdout:
            lines = result.stdout.split("\n")
            # Show summary line
            for line in lines:
                if "passed" in line and "failed" in line:
                    logger.info(f"   {line.strip()}")
                    break
                elif "passed" in line and line.strip().endswith("passed"):
                    logger.info(f"   {line.strip()}")
                    break

        if result.stderr and result.returncode != 0:
            logger.error(f"   Error output: {result.stderr[:200]}...")

    # Run end-to-end tests
    logger.info("\n📋 Running End-to-End tests...")
    e2e_result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-m",
            "pytest",
            "tests/end_to_end/test_phase2_comprehensive.py",
            "-v",
            "--tb=short",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    all_results["End-to-End"] = e2e_result

    if e2e_result.returncode == 0:
        logger.info("✅ End-to-End tests PASSED")
    else:
        logger.error("❌ End-to-End tests FAILED")
        overall_success = False
        if e2e_result.stderr:
            logger.error(f"   Error output: {e2e_result.stderr[:200]}...")

    # Print comprehensive summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 PHASE 2 TEST SUMMARY")
    logger.info("=" * 60)

    total_categories = len(all_results)
    passed_categories = sum(
        1 for result in all_results.values() if result.returncode == 0
    )

    for category, result in all_results.items():
        status = "✅ PASSED" if result.returncode == 0 else "❌ FAILED"
        logger.info(f"{category:.<30} {status}")

    logger.info("-" * 60)
    logger.info(f"Categories passed: {passed_categories}/{total_categories}")

    if overall_success:
        logger.info("🎉 ALL PHASE 2 TESTS PASSED! MARL coordination system is ready!")
        logger.info("✨ Key achievements:")
        logger.info("   • Multi-agent RL coordination implemented")
        logger.info("   • Consensus mechanisms working")
        logger.info("   • Shared learning infrastructure operational")
        logger.info("   • Performance monitoring active")
        logger.info("   • Configuration management ready")
        logger.info("   • Experimentation framework functional")
        logger.info("   • Error handling and fault tolerance robust")
        return 0
    else:
        failed_categories = total_categories - passed_categories
        logger.error(
            f"⚠️  {failed_categories} test categories failed. Please review and fix."
        )
        logger.error("🔧 Common issues to check:")
        logger.error("   • Missing dependencies or imports")
        logger.error("   • Configuration file issues")
        logger.error("   • Async/await syntax problems")
        logger.error("   • Mock object setup")
        return 1


def print_detailed_results():
    """Print detailed test results for debugging."""
    logger.info("\n🔍 For detailed test output, run individual test categories:")
    logger.info("   uv run pytest tests/unit_tests/test_error_handling.py -v")
    logger.info("   uv run pytest tests/unit_tests/test_fault_tolerance.py -v")
    logger.info("   uv run pytest tests/unit_tests/test_experiment_manager.py -v")
    logger.info("   uv run pytest tests/end_to_end/test_phase2_comprehensive.py -v")


if __name__ == "__main__":
    exit_code = run_tests()
    if exit_code != 0:
        print_detailed_results()
    sys.exit(exit_code)
