#!/usr/bin/env python3
"""
Script to run all phase tests for the SynThesisAI platform.

This script runs tests for all implemented phases in sequence and provides
a comprehensive summary of the entire system's test status.

Usage:
    uv run python scripts/run_all_phase_tests.py [--phase PHASE] [--verbose] [--coverage]
"""

# Standard Library
import argparse
import logging
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_test_results(output: str) -> Dict[str, Any]:
    """
    Parse test output to extract pass/fail statistics and category information.

    Args:
        output: Test output string.

    Returns:
        A dictionary containing test statistics and category details.
    """
    stats: Dict[str, Any] = {
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

    # Look for category-based results (Phase 1/2 format: "Categories passed:")
    category_pattern = r"Categories passed:\s*(\d+)/(\d+)"
    category_match = re.search(category_pattern, output)

    # Look for component-based results (Phase 3 format: "Components: X/Y passed")
    component_pattern = r"Components:\s*(\d+)/(\d+)\s+passed"
    component_match = re.search(component_pattern, output)

    if category_match or component_match:
        if category_match:
            passed_categories = int(category_match.group(1))
            total_categories = int(category_match.group(2))
        else:  # component_match
            passed_categories = int(component_match.group(1))
            total_categories = int(component_match.group(2))

        stats["categories"]["passed_categories"] = passed_categories
        stats["categories"]["total_categories"] = total_categories
        stats["categories"]["failed_categories"] = total_categories - passed_categories

        # Extract category/component details from the summary section
        lines = output.split("\n")
        in_summary = False
        for line in lines:
            clean_line = line.strip()
            if (
                "TEST SUMMARY" in clean_line
                or "PHASE 2 TEST SUMMARY" in clean_line
                or "PHASE 3 TEST SUMMARY" in clean_line
            ):
                in_summary = True
                continue
            if in_summary and ("✅" in clean_line or "❌" in clean_line):
                # Handle Phase 1/2 format (with dots)
                if "." in clean_line and (
                    "✅ PASSED" in clean_line or "❌ FAILED" in clean_line
                ):
                    entry = clean_line
                    if " - INFO - " in clean_line:
                        entry = clean_line.split(" - INFO - ", 1)[1]
                    elif " - ERROR - " in clean_line:
                        entry = clean_line.split(" - ERROR - ", 1)[1]
                    category_name = entry.split(".")[0].strip()
                    status = "PASSED" if "✅ PASSED" in clean_line else "FAILED"
                    stats["categories"]["category_details"].append(
                        {"name": category_name, "status": status}
                    )
                # Handle Phase 3 format (component details)
                elif "✅" in clean_line and (
                    "Validation" in clean_line or "Framework" in clean_line
                ):
                    # Extract component name from lines like "✅ Enhanced Chemistry Validation (Task 3.3)"
                    component_name = clean_line.replace("✅", "").strip()
                    if "(" in component_name:
                        component_name = component_name.split("(")[0].strip()
                    stats["categories"]["category_details"].append(
                        {"name": component_name, "status": "PASSED"}
                    )
                elif "❌" in clean_line and (
                    "Validation" in clean_line or "Framework" in clean_line
                ):
                    component_name = clean_line.replace("❌", "").strip()
                    if "(" in component_name:
                        component_name = component_name.split("(")[0].strip()
                    stats["categories"]["category_details"].append(
                        {"name": component_name, "status": "FAILED"}
                    )
            elif in_summary and clean_line.startswith("-"):
                break

    # Parse pytest summaries
    failed_passed_pattern = r"(\d+)\s+failed,\s*(\d+)\s+passed"
    for failed, passed in re.findall(failed_passed_pattern, output):
        total_failed += int(failed)
        total_passed += int(passed)

    passed_failed_pattern = r"(\d+)\s+passed,\s*(\d+)\s+failed"
    for passed, failed in re.findall(passed_failed_pattern, output):
        total_passed += int(passed)
        total_failed += int(failed)

    # Only passed pattern
    for line in output.split("\n"):
        line_stripped = line.strip()
        if (
            "passed" in line_stripped
            and "failed" not in line_stripped
            and "in" in line_stripped
            and "s ==" in line_stripped
        ):
            m = re.search(r"(\d+)\s+passed\s+in\s+[\d.]+s", line_stripped)
            if m:
                total_passed += int(m.group(1))

    if total_passed or total_failed:
        stats["passed_tests"] = total_passed
        stats["failed_tests"] = total_failed
        stats["total_tests"] = total_passed + total_failed
    elif category_match or component_match:
        stats["passed_tests"] = stats["categories"]["passed_categories"]
        stats["failed_tests"] = stats["categories"]["failed_categories"]
        stats["total_tests"] = stats["categories"]["total_categories"]

    if stats["total_tests"] > 0:
        stats["pass_percentage"] = (stats["passed_tests"] / stats["total_tests"]) * 100

    return stats


def run_phase_tests(phase_name: str, script_path: str) -> Dict[str, Any]:
    """
    Run tests for a specific phase.

    Args:
        phase_name: Name of the phase.
        script_path: Path to the phase test script.

    Returns:
        A dictionary with test results including success flag, duration,
        stdout, stderr, returncode, and test statistics.
    """
    logger.info("\n🚀 Starting %s tests...", phase_name)
    logger.info("%s", "=" * 60)
    start_time = time.time()

    if not Path(script_path).exists():
        logger.error("❌ Test script not found: %s", script_path)
        return {
            "phase": phase_name,
            "success": False,
            "duration": 0.0,
            "error": f"Script not found: {script_path}",
            "test_stats": {
                "total_tests": 0,
                "passed_tests": 0,
                "failed_tests": 0,
                "pass_percentage": 0.0,
            },
        }

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            check=False,
            timeout=600,
        )
        duration = time.time() - start_time
        success = result.returncode == 0
        output_to_parse = result.stdout or result.stderr
        test_stats = parse_test_results(output_to_parse)

        if result.stdout:
            for line in result.stdout.strip().split("\n")[-10:]:
                if line.strip():
                    logger.info("   %s", line)

        if result.stderr and not success:
            logger.error("   Error: %s...", result.stderr[:200])

        status = "✅ PASSED" if success else "❌ FAILED"
        pass_info = (
            "(%.1f%% pass)" % test_stats["pass_percentage"]
            if test_stats["total_tests"]
            else ""
        )
        logger.info(
            "\n%s Result: %s (Duration: %.1fs) %s",
            phase_name,
            status,
            duration,
            pass_info,
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
        logger.error("❌ %s tests timed out after %.1fs", phase_name, duration)
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
    except subprocess.SubprocessError as se:
        duration = time.time() - start_time
        logger.error("❌ %s tests failed: %s", phase_name, se)
        return {
            "phase": phase_name,
            "success": False,
            "duration": duration,
            "error": str(se),
            "test_stats": {
                "total_tests": 0,
                "passed_tests": 0,
                "failed_tests": 0,
                "pass_percentage": 0.0,
            },
        }


def main() -> int:
    """
    Main function to run phase tests.

    Returns:
        Exit code 0 if all phases passed, otherwise 1.
    """
    parser = argparse.ArgumentParser(
        description="Run SynThesisAI phase tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run python scripts/run_all_phase_tests.py                 # Run all phase tests
  uv run python scripts/run_all_phase_tests.py --phase phase1  # Run only Phase 1 tests
  uv run python scripts/run_all_phase_tests.py --phase phase2  # Run only Phase 2 tests
  uv run python scripts/run_all_phase_tests.py --phase phase3  # Run only Phase 3 tests
  uv run python scripts/run_all_phase_tests.py --verbose       # Run with verbose output
  uv run python scripts/run_all_phase_tests.py --coverage      # Run with coverage reporting
        """,
    )

    parser.add_argument(
        "--phase",
        choices=["phase1", "phase2", "phase3", "all"],
        default="all",
        help="Run tests for specific phase (default: all)",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose test output"
    )

    parser.add_argument(
        "--coverage", "-c", action="store_true", help="Enable coverage reporting"
    )

    args = parser.parse_args()

    logger.info("🌟 SynThesisAI Platform - Comprehensive Test Suite")
    logger.info("%s", "=" * 80)

    if args.phase == "all":
        logger.info("Running all phase tests to validate the entire system...")
    else:
        logger.info("Running %s tests...", args.phase.upper())

    overall_start = time.time()

    all_phases = [
        {
            "id": "phase1",
            "name": "Phase 1 - DSPy Integration",
            "script": "scripts/run_phase1_tests.py",
            "description": (
                "DSPy agents, optimization engine, caching, signatures, feedback systems"
            ),
        },
        {
            "id": "phase2",
            "name": "Phase 2 - MARL Coordination",
            "script": "scripts/run_phase2_tests.py",
            "description": (
                "Multi-agent RL, coordination mechanisms, shared learning, monitoring"
            ),
        },
        {
            "id": "phase3",
            "name": "Phase 3 - STREAM Domain Validation",
            "script": "scripts/run_phase3_tests.py",
            "description": (
                "Advanced domain validation with code execution, algorithm analysis, and best practices"
            ),
        },
    ]

    # Select phases to run based on arguments
    if args.phase == "all":
        phases = all_phases
    else:
        phases = [p for p in all_phases if p["id"] == args.phase]

    if not phases:
        logger.error("❌ No phases selected to run")
        return 1

    phase_results = []

    logger.info("Phases to test: %s", [p["id"] for p in phases])

    for idx, phase in enumerate(phases, start=1):
        logger.info("\n%s", "=" * 80)
        logger.info("📋 %s: %s", phase["id"].upper(), phase["name"])
        logger.info("%s", "=" * 80)
        logger.info("Description: %s", phase["description"])

        # Add components info for Phase 3
        if phase["id"] == "phase3":
            logger.info("Components:")
            components = [
                "• Enhanced Chemistry Validation (Task 3.3)",
                "• Advanced Biology Validation (Task 3.4)",
                "• Technology Validation Framework (Task 4.1)",
                "• Advanced Code Execution Validation (Task 4.2)",
                "• Algorithm Analysis Validation (Task 4.3)",
            ]
            for component in components:
                logger.info("  %s", component)

        result = run_phase_tests(phase["name"], phase["script"])
        phase_results.append(result)

        if result["success"]:
            logger.info("✅ %s completed successfully!", phase["id"])
        else:
            logger.error("❌ %s failed!", phase["id"])

        if idx < len(phases):
            time.sleep(2)

    total_time = time.time() - overall_start
    total_phases = len(phase_results)
    passed_phases = sum(1 for r in phase_results if r["success"])

    total_tests = sum(r["test_stats"].get("total_tests", 0) for r in phase_results)
    passed_tests = sum(r["test_stats"].get("passed_tests", 0) for r in phase_results)
    overall_pct = (passed_tests / total_tests * 100) if total_tests else 0.0

    total_cats = sum(
        r["test_stats"].get("categories", {}).get("total_categories", 0)
        for r in phase_results
    )
    passed_cats = sum(
        r["test_stats"].get("categories", {}).get("passed_categories", 0)
        for r in phase_results
    )

    logger.info("\n%s", "=" * 80)
    if args.phase == "all":
        logger.info("🏆 COMPREHENSIVE TEST SUMMARY")
    else:
        logger.info("🏆 %s TEST SUMMARY", args.phase.upper())
    logger.info("%s", "=" * 80)

    for r in phase_results:
        status = "✅ PASSED" if r["success"] else "❌ FAILED"
        dur = r.get("duration", 0.0)
        stats = r.get("test_stats", {})
        pct = stats.get("pass_percentage", 0.0)
        cats = stats.get("categories", {})
        parts = []
        if cats.get("total_categories"):
            parts.append(
                "%d/%d Categories"
                % (cats["passed_categories"], cats["total_categories"])
            )
        if stats.get("total_tests"):
            parts.append("%.1f%% PASS" % pct)
        info_str = ""
        if parts:
            info_str = " (%s)" % ", ".join(parts)
        logger.info("%s %s (%.1fs)%s", r["phase"], status, dur, info_str)

        if not r["success"] and cats.get("category_details"):
            failed = [c for c in cats["category_details"] if c["status"] == "FAILED"]
            if failed:
                logger.error("   Failed categories:")
                for c in failed:
                    logger.error("     • %s", c["name"])
        if not r["success"] and "error" in r:
            logger.error("   Error: %s", r["error"])

    logger.info("%s", "-" * 80)
    logger.info("Phases passed: %d/%d", passed_phases, total_phases)
    if total_cats:
        logger.info("Categories passed: %d/%d", passed_cats, total_cats)
    logger.info("Total duration: %.1f seconds", total_time)
    if total_tests:
        logger.info(
            "Overall pass rate: %.1f%% (%d/%d tests)",
            overall_pct,
            passed_tests,
            total_tests,
        )

    if passed_phases == total_phases:
        logger.info("\n🎉 ALL PHASES PASSED! SynThesisAI platform is fully validated!")
        logger.info("✨ System Status: READY FOR PRODUCTION")
        logger.info("\n🚀 Key Platform Capabilities Validated:")
        logger.info("   ✅ DSPy Integration - Advanced prompt optimization")
        logger.info("   ✅ MARL Coordination - Multi-agent reinforcement learning")
        logger.info("   ✅ STREAM Domain Validation - Comprehensive content validation")
        logger.info("   ✅ Code Execution - Multi-language sandboxed execution")
        logger.info("   ✅ Algorithm Analysis - Complexity and pattern detection")
        logger.info("   ✅ Best Practices Validation - Industry standards compliance")
        logger.info("   ✅ Error Handling - Robust fault tolerance")
        logger.info("   ✅ Performance Monitoring - Comprehensive metrics")
        logger.info("\n📊 Platform Performance Targets:")
        logger.info("   • >85%% coordination success rate")
        logger.info("   • >30%% performance improvement over baseline")
        logger.info("   • >95%% content accuracy")
        logger.info("   • <3%% false positive rate")
        logger.info("   • 50-70%% development time reduction")
        return 0
    logger.error(
        "\n⚠️  %d phases failed. System not ready for production.",
        total_phases - passed_phases,
    )
    logger.error("🔧 Next Steps:")
    logger.error("   1. Review failed phase test outputs")
    logger.error("   2. Fix identified issues")
    logger.error("   3. Re-run individual phase tests")
    logger.error("   4. Re-run this comprehensive test suite")
    logger.info("\n🔍 Debug Commands:")
    for r in phase_results:
        if not r["success"]:
            if "Phase 1" in r["phase"]:
                num = "1"
            elif "Phase 2" in r["phase"]:
                num = "2"
            elif "Phase 3" in r["phase"]:
                num = "3"
            else:
                num = "unknown"
            logger.info("   uv run python scripts/run_phase%s_tests.py", num)
    return 1


def print_system_info() -> None:
    """
    Print system information for debugging.
    """
    logger.info("\n🔧 System Information:")
    try:
        python_version = subprocess.run(
            [sys.executable, "--version"], capture_output=True, text=True, check=False
        )
        logger.info("   Python: %s", python_version.stdout.strip())
        uv_version = subprocess.run(
            ["uv", "--version"], capture_output=True, text=True, check=False
        )
        logger.info("   UV: %s", uv_version.stdout.strip())
        pytest_version = subprocess.run(
            ["uv", "run", sys.executable, "-m", "pytest", "--version"],
            capture_output=True,
            text=True,
            check=False,
        )
        if pytest_version.returncode == 0:
            logger.info("   Pytest: %s", pytest_version.stdout.strip())
    except (subprocess.SubprocessError, FileNotFoundError) as spe:
        logger.warning("   Could not get system info: %s", spe)


if __name__ == "__main__":
    print_system_info()
    exit_code = main()
    if exit_code == 0:
        logger.info(
            "\n🎯 All phases validated! SynThesisAI platform is ready for production!"
        )
    else:
        logger.info("\n🔄 Please fix issues and re-run tests before proceeding.")
    sys.exit(exit_code)
