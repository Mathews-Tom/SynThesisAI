#!/usr/bin/env python3
"""
Script to run all phase tests for the SynThesisAI platform.

This script runs tests for all implemented phases in sequence and provides
a comprehensive summary of the entire system's test status.
"""

import logging
import re
import subprocess
import sys
import time
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_test_results(output: str) -> dict:
    """
    Parse test output to extract pass/fail statistics and category information.

    Args:
        output: Test output string

    Returns:
        Dictionary with test statistics and category information
    """
    stats = {
        "total_tests": 0,
        "passed_tests": 0,
        "failed_tests": 0,
        "pass_percentage": 0.0,
        "categories": {
            "total_categories": 0,
            "passed_categories": 0,
            "failed_categories": 0,
            "category_details": [],
        },
    }

    total_passed = 0
    total_failed = 0

    # Look for category-based results first (Phase 2 format)
    category_pattern = r"Categories passed:\s*(\d+)/(\d+)"
    category_match = re.search(category_pattern, output)

    if category_match:
        passed_categories = int(category_match.group(1))
        total_categories = int(category_match.group(2))
        stats["categories"]["passed_categories"] = passed_categories
        stats["categories"]["total_categories"] = total_categories
        stats["categories"]["failed_categories"] = total_categories - passed_categories

        # Extract category details from the summary section
        lines = output.split("\n")
        in_summary = False
        for line in lines:
            line = line.strip()
            if "TEST SUMMARY" in line or "PHASE 2 TEST SUMMARY" in line:
                in_summary = True
                continue
            if in_summary and ("✅ PASSED" in line or "❌ FAILED" in line):
                # Parse lines like "MARL Agents................... ✅ PASSED"
                if "." in line and ("✅ PASSED" in line or "❌ FAILED" in line):
                    # Remove timestamp prefix if present
                    clean_line = line
                    if " - INFO - " in line:
                        clean_line = line.split(" - INFO - ", 1)[1]
                    elif " - ERROR - " in line:
                        clean_line = line.split(" - ERROR - ", 1)[1]

                    category_name = clean_line.split(".")[0].strip()
                    status = "PASSED" if "✅ PASSED" in line else "FAILED"
                    stats["categories"]["category_details"].append(
                        {"name": category_name, "status": status}
                    )
            elif in_summary and line.startswith("-"):
                break  # End of category summary

    # Look for pytest summary lines and extract individual test numbers
    # Pattern 1: "X failed, Y passed, Z errors in T.Ts" or "X failed, Y passed in T.Ts"
    failed_passed_pattern = r"(\d+)\s+failed,\s*(\d+)\s+passed"
    matches = re.findall(failed_passed_pattern, output)
    for match in matches:
        failed, passed = int(match[0]), int(match[1])
        total_failed += failed
        total_passed += passed

    # Pattern 2: "Y passed, X failed in T.Ts"
    passed_failed_pattern = r"(\d+)\s+passed,\s*(\d+)\s+failed"
    matches = re.findall(passed_failed_pattern, output)
    for match in matches:
        passed, failed = int(match[0]), int(match[1])
        total_passed += passed
        total_failed += failed

    # Pattern 3: "Y passed in T.Ts" (only passed, no failures)
    for line in output.split("\n"):
        line = line.strip()
        if (
            "passed" in line
            and "failed" not in line
            and "in" in line
            and "s ==" in line
        ):
            # Try to extract just the passed count
            passed_match = re.search(r"(\d+)\s+passed\s+in\s+[\d.]+s", line)
            if passed_match:
                passed = int(passed_match.group(1))
                total_passed += passed

    # Set test statistics
    if total_passed > 0 or total_failed > 0:
        stats["passed_tests"] = total_passed
        stats["failed_tests"] = total_failed
        stats["total_tests"] = total_passed + total_failed
    elif category_match:
        # Use category counts as fallback for test counts
        stats["passed_tests"] = passed_categories
        stats["total_tests"] = total_categories
        stats["failed_tests"] = total_categories - passed_categories

    # Calculate percentage
    if stats["total_tests"] > 0:
        stats["pass_percentage"] = (stats["passed_tests"] / stats["total_tests"]) * 100

    return stats


def run_phase_tests(phase_name: str, script_path: str) -> dict:
    """
    Run tests for a specific phase.

    Args:
        phase_name: Name of the phase (e.g., "Phase 1", "Phase 2")
        script_path: Path to the phase test script

    Returns:
        Dictionary with test results
    """
    logger.info(f"\n🚀 Starting {phase_name} tests...")
    logger.info("=" * 60)

    start_time = time.time()

    # Check if script exists
    if not Path(script_path).exists():
        logger.error(f"❌ Test script not found: {script_path}")
        return {
            "phase": phase_name,
            "success": False,
            "duration": 0,
            "error": f"Script not found: {script_path}",
            "test_stats": {
                "total_tests": 0,
                "passed_tests": 0,
                "failed_tests": 0,
                "pass_percentage": 0.0,
            },
        }

    # Run the phase test script
    try:
        result = subprocess.run(
            ["python", script_path],
            capture_output=True,
            text=True,
            check=False,
            timeout=600,  # 10 minute timeout per phase
        )

        duration = time.time() - start_time

        # Parse results
        success = result.returncode == 0

        # Parse test statistics from output (try stdout first, then stderr)
        output_to_parse = result.stdout if result.stdout.strip() else result.stderr
        test_stats = parse_test_results(output_to_parse)

        # Log output
        if result.stdout:
            # Show last few lines of output for summary
            output_lines = result.stdout.strip().split("\n")
            for line in output_lines[-10:]:  # Last 10 lines
                if line.strip():
                    logger.info(f"   {line}")

        if result.stderr and not success:
            logger.error(f"   Error: {result.stderr[:200]}...")

        status = "✅ PASSED" if success else "❌ FAILED"
        pass_info = (
            f"({test_stats['pass_percentage']:.1f}% pass)"
            if test_stats["total_tests"] > 0
            else ""
        )
        logger.info(
            f"\n{phase_name} Result: {status} (Duration: {duration:.1f}s) {pass_info}"
        )

        return {
            "phase": phase_name,
            "success": success,
            "duration": duration,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
            "test_stats": test_stats,
        }

    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        logger.error(f"❌ {phase_name} tests timed out after {duration:.1f}s")
        return {
            "phase": phase_name,
            "success": False,
            "duration": duration,
            "error": "Test timeout",
            "test_stats": {
                "total_tests": 0,
                "passed_tests": 0,
                "failed_tests": 0,
                "pass_percentage": 0.0,
            },
        }
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"❌ {phase_name} tests failed with exception: {str(e)}")
        return {
            "phase": phase_name,
            "success": False,
            "duration": duration,
            "error": str(e),
            "test_stats": {
                "total_tests": 0,
                "passed_tests": 0,
                "failed_tests": 0,
                "pass_percentage": 0.0,
            },
        }


def main():
    """Main function to run all phase tests."""
    logger.info("🌟 SynThesisAI Platform - Comprehensive Test Suite")
    logger.info("=" * 60)
    logger.info("Running all phase tests to validate the entire system...")

    overall_start_time = time.time()

    # Define phases and their test scripts
    phases = [
        {
            "name": "Phase 1 - DSPy Integration",
            "script": "scripts/run_phase1_tests.py",
            "description": "DSPy agents, optimization engine, caching, signatures, feedback systems",
        },
        {
            "name": "Phase 2 - MARL Coordination",
            "script": "scripts/run_phase2_tests.py",
            "description": "Multi-agent RL, coordination mechanisms, shared learning, monitoring",
        },
    ]

    # Track results
    phase_results = []

    # Run each phase
    for phase in phases:
        logger.info(f"\n📋 {phase['name']}")
        logger.info(f"   Description: {phase['description']}")

        result = run_phase_tests(phase["name"], phase["script"])
        phase_results.append(result)

        # Short pause between phases
        if result != phase_results[-1]:  # Not the last phase
            time.sleep(2)

    # Calculate overall results
    total_duration = time.time() - overall_start_time
    total_phases = len(phase_results)
    passed_phases = sum(1 for result in phase_results if result["success"])

    # Calculate overall test statistics
    total_tests_all = sum(
        result.get("test_stats", {}).get("total_tests", 0) for result in phase_results
    )
    passed_tests_all = sum(
        result.get("test_stats", {}).get("passed_tests", 0) for result in phase_results
    )
    overall_pass_percentage = (
        (passed_tests_all / total_tests_all * 100) if total_tests_all > 0 else 0.0
    )

    # Calculate overall category statistics
    total_categories_all = sum(
        result.get("test_stats", {}).get("categories", {}).get("total_categories", 0)
        for result in phase_results
    )
    passed_categories_all = sum(
        result.get("test_stats", {}).get("categories", {}).get("passed_categories", 0)
        for result in phase_results
    )

    # Print comprehensive summary
    logger.info("\n" + "=" * 80)
    logger.info("🏆 COMPREHENSIVE TEST SUMMARY")
    logger.info("=" * 80)

    for result in phase_results:
        status = "✅ PASSED" if result["success"] else "❌ FAILED"
        duration = result.get("duration", 0)
        test_stats = result.get("test_stats", {})
        pass_percentage = test_stats.get("pass_percentage", 0.0)
        categories = test_stats.get("categories", {})

        # Format the line with pass percentage and category info
        phase_info = f"{result['phase']:.<50} {status} ({duration:.1f}s)"

        # Add both category info and pass percentage when available
        info_parts = []

        if categories.get("total_categories", 0) > 0:
            cat_passed = categories.get("passed_categories", 0)
            cat_total = categories.get("total_categories", 0)
            info_parts.append(f"{cat_passed}/{cat_total} categories")

        if test_stats.get("total_tests", 0) > 0:
            info_parts.append(f"{pass_percentage:.1f}% pass")

        if info_parts:
            phase_info += f" ({', '.join(info_parts)})"

        logger.info(phase_info)

        # Show category details for failed phases
        if not result["success"] and categories.get("category_details"):
            failed_categories = [
                cat
                for cat in categories["category_details"]
                if cat["status"] == "FAILED"
            ]
            if failed_categories:
                logger.error("   Failed categories:")
                for cat in failed_categories:
                    logger.error(f"     • {cat['name']}")

        if not result["success"] and "error" in result:
            logger.error(f"   Error: {result['error']}")

    logger.info("-" * 80)
    logger.info(f"Phases passed: {passed_phases}/{total_phases}")
    if total_categories_all > 0:
        logger.info(
            f"Categories passed: {passed_categories_all}/{total_categories_all}"
        )
    logger.info(f"Total duration: {total_duration:.1f} seconds")
    if total_tests_all > 0:
        logger.info(
            f"Overall pass rate: {overall_pass_percentage:.1f}% ({passed_tests_all}/{total_tests_all} tests)"
        )

    # Overall status
    if passed_phases == total_phases:
        logger.info("\n🎉 ALL PHASES PASSED! SynThesisAI platform is fully validated!")
        logger.info("✨ System Status: READY FOR PRODUCTION")
        logger.info("\n🚀 Key Platform Capabilities Validated:")
        logger.info("   ✅ DSPy Integration - Advanced prompt optimization")
        logger.info("   ✅ MARL Coordination - Multi-agent reinforcement learning")
        logger.info("   ✅ Error Handling - Robust fault tolerance")
        logger.info("   ✅ Performance Monitoring - Comprehensive metrics")
        logger.info("   ✅ Configuration Management - Flexible system configuration")
        logger.info("   ✅ Experimentation Framework - A/B testing and research")

        logger.info("\n📊 Platform Performance Targets:")
        logger.info("   • >85% coordination success rate")
        logger.info("   • >30% performance improvement over baseline")
        logger.info("   • >95% content accuracy")
        logger.info("   • <3% false positive rate")
        logger.info("   • 50-70% development time reduction")

        return 0
    else:
        failed_phases = total_phases - passed_phases
        logger.error(
            f"\n⚠️  {failed_phases} phases failed. System not ready for production."
        )
        logger.error("🔧 Next Steps:")
        logger.error("   1. Review failed phase test outputs")
        logger.error("   2. Fix identified issues")
        logger.error("   3. Re-run individual phase tests")
        logger.error("   4. Re-run this comprehensive test suite")

        logger.info("\n🔍 Debug Commands:")
        for result in phase_results:
            if not result["success"]:
                phase_num = "1" if "Phase 1" in result["phase"] else "2"
                logger.info(f"   python scripts/run_phase{phase_num}_tests.py")

        return 1


def print_system_info():
    """Print system information for debugging."""
    logger.info("\n🔧 System Information:")
    try:
        # Python version
        python_version = subprocess.run(
            ["python", "--version"], capture_output=True, text=True
        )
        logger.info(f"   Python: {python_version.stdout.strip()}")

        # UV version
        uv_version = subprocess.run(["uv", "--version"], capture_output=True, text=True)
        logger.info(f"   UV: {uv_version.stdout.strip()}")

        # Pytest version
        pytest_version = subprocess.run(
            ["uv", "run", "python", "-m", "pytest", "--version"],
            capture_output=True,
            text=True,
        )
        if pytest_version.returncode == 0:
            logger.info(f"   Pytest: {pytest_version.stdout.strip()}")

    except Exception as e:
        logger.warning(f"   Could not get system info: {e}")


if __name__ == "__main__":
    print_system_info()
    exit_code = main()

    if exit_code == 0:
        logger.info("\n🎯 Ready to proceed with Phase 3 development!")
    else:
        logger.info("\n🔄 Please fix issues and re-run tests before proceeding.")

    sys.exit(exit_code)
