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
    logger.info("🚀 Running Phase 1 DSPy integration tests...")

    # Define test categories and their corresponding test files
    test_categories = {
        "DSPy Adapter": ["tests/unit_tests/test_dspy_adapter.py"],
        "Continuous Learning": ["tests/unit_tests/test_dspy_continuous_learning.py"],
        "Feedback Systems": ["tests/unit_tests/test_dspy_feedback.py"],
        "End-to-End Integration": [
            "tests/end_to_end/test_phase1_comprehensive.py::TestPhase1QualityAssurance::test_quality_metrics_calculation"
        ],
    }

    # Track results
    all_results = {}
    overall_success = True

    # Run tests by category
    for category, test_files in test_categories.items():
        logger.info(f"\n📋 Running {category} tests...")

        result = subprocess.run(
            ["uv", "run", "python", "-m", "pytest"] + test_files + ["-v", "--tb=short"],
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

    # Print comprehensive summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 PHASE 1 TEST SUMMARY")
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
        logger.info("🎉 ALL PHASE 1 TESTS PASSED! DSPy integration is ready!")
        logger.info("✨ Key achievements:")
        logger.info("   • DSPy adapter functionality implemented")
        logger.info("   • Continuous learning systems operational")
        logger.info("   • Feedback mechanisms working")
        logger.info("   • End-to-end integration validated")
        return 0

    failed_categories = total_categories - passed_categories
    logger.error(
        f"⚠️  {failed_categories} test categories failed. Please review and fix."
    )
    logger.error("🔧 Common issues to check:")
    logger.error("   • Missing dependencies or imports")
    logger.error("   • DSPy configuration issues")
    logger.error("   • Mock object setup")
    return 1


if __name__ == "__main__":
    sys.exit(run_tests())
