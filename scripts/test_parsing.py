#!/usr/bin/env python3
"""Test script with the final parsing function."""

# Standard Library
import logging
import re
from typing import Any, Dict

logger = logging.getLogger(__name__)


def parse_test_results(output: str) -> Dict[str, Any]:
    """
    Parse test output to extract pass/fail statistics.

    Args:
        output: Test output string

    Returns:
        Dictionary with test statistics:
            {
                "total_tests": int,
                "passed_tests": int,
                "failed_tests": int,
                "pass_percentage": float
            }

    Raises:
        None
    """
    stats: Dict[str, Any] = {
        "total_tests": 0,
        "passed_tests": 0,
        "failed_tests": 0,
        "pass_percentage": 0.0,
    }

    total_passed = 0
    total_failed = 0

    logger.debug("=== PARSING DEBUG ===")
    logger.debug("Input length: %d chars", len(output))

    # Pattern 1: "X failed, Y passed, Z errors in T.Ts" or "X failed, Y passed in T.Ts"
    failed_passed_pattern = r"(\d+)\s+failed,\s*(\d+)\s+passed"
    matches = re.findall(failed_passed_pattern, output)
    logger.debug("Failed-Passed pattern matches: %s", matches)
    for failed_str, passed_str in matches:
        failed = int(failed_str)
        passed = int(passed_str)
        logger.debug("  -> Failed: %d, Passed: %d", failed, passed)
        total_failed += failed
        total_passed += passed

    # Pattern 2: "Y passed, X failed in T.Ts"
    passed_failed_pattern = r"(\d+)\s+passed,\s*(\d+)\s+failed"
    matches = re.findall(passed_failed_pattern, output)
    logger.debug("Passed-Failed pattern matches: %s", matches)
    for passed_str, failed_str in matches:
        passed = int(passed_str)
        failed = int(failed_str)
        logger.debug("  -> Passed: %d, Failed: %d", passed, failed)
        total_passed += passed
        total_failed += failed

    # Pattern 3: "Y passed in T.Ts" (only passed, no failures)
    logger.debug("Checking for passed-only lines")
    for line in output.splitlines():
        line = line.strip()
        if "passed" in line and "failed" not in line and "in" in line and "s ==" in line:
            passed_match = re.search(r"(\d+)\s+passed\s+in\s+[\d.]+s", line)
            if passed_match:
                passed = int(passed_match.group(1))
                logger.debug("  -> Passed only: %d (from line: %s)", passed, line)
                total_passed += passed

    logger.debug("Total accumulated - Passed: %d, Failed: %d", total_passed, total_failed)

    if total_passed > 0 or total_failed > 0:
        stats["passed_tests"] = total_passed
        stats["failed_tests"] = total_failed
        stats["total_tests"] = total_passed + total_failed
    else:
        category_pattern = r"Categories passed:\s*(\d+)/(\d+)"
        category_match = re.search(category_pattern, output)
        if category_match:
            passed_categories = int(category_match.group(1))
            total_categories = int(category_match.group(2))
            logger.debug("Category match: %d/%d", passed_categories, total_categories)
            stats["passed_tests"] = passed_categories
            stats["total_tests"] = total_categories
            stats["failed_tests"] = total_categories - passed_categories

    if stats["total_tests"] > 0:
        stats["pass_percentage"] = (stats["passed_tests"] / stats["total_tests"]) * 100

    logger.debug("Final stats: %s", stats)

    return stats


def _main() -> None:
    """
    Execute example parsing with sample Phase 1 and Phase 2 outputs.
    """
    # Test with Phase 1 output
    phase1_output = """
cachedir: .pytest_cache
rootdir: /Users/druk/WorkSpace/AetherForge/SynThesisAI
configfile: pytest.ini
plugins: anyio-4.9.0, asyncio-1.1.0
asyncio: mode=Mode.STRICT, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collecting ... collected 1 item
tests/end_to_end/test_phase1_comprehensive.py::TestPhase1QualityAssurance::test_quality_metrics_calculation PASSED [100%]
============================== 1 passed in 1.13s ===============================
"""
    logger.info("=== PHASE 1 TEST ===")
    result1 = parse_test_results(phase1_output)
    logger.info("Phase 1 Result: %s", result1)

    # Test with Phase 2 output (updated with recent output)
    phase2_output = """
🚀 Running Phase 2 MARL coordination tests...

📋 Running MARL Agents tests...
❌ MARL Agents tests FAILED
   =================== 23 failed, 74 passed, 7 errors in 1.43s ====================

📋 Running Coordination Mechanisms tests...
❌ Coordination Mechanisms tests FAILED

📋 Running Shared Learning tests...
❌ Shared Learning tests FAILED
   ========================= 2 failed, 75 passed in 2.36s =========================

📋 Running Performance Monitoring tests...
✅ Performance Monitoring tests PASSED

📋 Running Configuration Management tests...
❌ Configuration Management tests FAILED
   ======================== 11 failed, 36 passed in 0.18s =========================

📋 Running Experimentation Framework tests...
✅ Experimentation Framework tests PASSED

📋 Running Error Handling tests...
✅ Error Handling tests PASSED

📋 Running Fault Tolerance tests...
✅ Fault Tolerance tests PASSED

📋 Running End-to-End tests...
❌ End-to-End tests FAILED

============================================================
📊 PHASE 2 TEST SUMMARY
============================================================
MARL Agents................... ❌ FAILED
Coordination Mechanisms....... ❌ FAILED
Shared Learning............... ❌ FAILED
Performance Monitoring........ ✅ PASSED
Configuration Management...... ❌ FAILED
Experimentation Framework..... ✅ PASSED
Error Handling................ ✅ PASSED
Fault Tolerance............... ✅ PASSED
End-to-End.................... ❌ FAILED
------------------------------------------------------------
Categories passed: 4/9
"""
    logger.info("\n=== PHASE 2 TEST ===")
    result2 = parse_test_results(phase2_output)
    logger.info("Phase 2 Result: %s", result2)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    _main()
