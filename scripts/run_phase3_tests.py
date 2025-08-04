#!/usr/bin/env python3
"""
Phase 3 Test Runner for SynThesisAI Stream Domain Validation

This script runs comprehensive tests for Phase 3 components including:
- Enhanced Chemistry Validation (Task 3.3)
- Advanced Biology Validation (Task 3.4) 
- Technology Validation Framework (Task 4.1)
- Advanced Code Execution Validation (Task 4.2)
- Algorithm Analysis Validation (Task 4.3)

Usage:
    uv run python scripts/run_phase3_tests.py [--verbose] [--coverage] [--component COMPONENT]
"""

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class Phase3TestRunner:
    """
    Comprehensive test runner for Phase 3 components.

    Manages execution of all Phase 3 validation tests with detailed reporting
    and component-specific test selection.
    """

    def __init__(self):
        """Initialize the Phase 3 test runner."""
        self.project_root = Path(__file__).parent.parent

        # Phase 3 test components
        self.test_components = {
            "chemistry": {
                "name": "Enhanced Chemistry Validation",
                "task": "3.3",
                "test_paths": ["tests/unit/validation/domains/test_chemistry.py"],
                "description": "Chemical equation balancing, reaction mechanisms, safety protocols",
            },
            "biology": {
                "name": "Advanced Biology Validation",
                "task": "3.4",
                "test_paths": ["tests/unit/validation/domains/test_biology.py"],
                "description": "Biological processes, taxonomic accuracy, ethics validation",
            },
            "technology": {
                "name": "Technology Validation Framework",
                "task": "4.1",
                "test_paths": ["tests/unit/validation/domains/test_technology.py"],
                "description": "Code execution, algorithm analysis, security validation",
            },
            "code_execution": {
                "name": "Advanced Code Execution Validation",
                "task": "4.2",
                "test_paths": ["tests/unit/validation/test_code_execution.py"],
                "description": "Multi-language execution, performance analysis, security checks",
            },
            "algorithm_analysis": {
                "name": "Algorithm Analysis Validation",
                "task": "4.3",
                "test_paths": ["tests/unit/validation/test_algorithm_analysis.py"],
                "description": "Complexity analysis, pattern detection, optimization suggestions",
            },
        }

        logger.info(
            "Initialized Phase 3 Test Runner with %d components",
            len(self.test_components),
        )

    def run_component_tests(
        self, component: str, verbose: bool = False, coverage: bool = False
    ) -> Dict[str, any]:
        """
        Run tests for a specific Phase 3 component.

        Args:
            component: Component name to test
            verbose: Enable verbose output
            coverage: Enable coverage reporting

        Returns:
            Dictionary with test results
        """
        if component not in self.test_components:
            available = list(self.test_components.keys())
            raise ValueError(f"Unknown component: {component}. Available: {available}")

        component_info = self.test_components[component]
        logger.info(
            "Running tests for %s (Task %s)",
            component_info["name"],
            component_info["task"],
        )

        results = {
            "component": component,
            "name": component_info["name"],
            "task": component_info["task"],
            "description": component_info["description"],
            "test_paths": component_info["test_paths"],
            "passed": 0,
            "failed": 0,
            "total": 0,
            "duration": 0.0,
            "success": False,
            "output": "",
            "coverage": None,
        }

        start_time = time.time()

        try:
            # Build pytest command
            cmd = ["uv", "run", "python", "-m", "pytest"]

            # Add test paths
            cmd.extend(component_info["test_paths"])

            # Add options
            if verbose:
                cmd.append("-v")
            else:
                cmd.append("-q")

            cmd.extend(["--tb=short", "--no-header"])

            # Add coverage if requested
            if coverage:
                cmd.extend(
                    [
                        "--cov=core/validation",
                        "--cov-report=term-missing",
                        "--cov-report=json:coverage_phase3.json",
                    ]
                )

            # Run tests
            logger.debug("Executing command: %s", " ".join(cmd))
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            results["duration"] = time.time() - start_time
            results["output"] = result.stdout + result.stderr
            results["success"] = result.returncode == 0

            # Parse test results from output
            self._parse_test_output(results, result.stdout)

            if results["success"]:
                logger.info(
                    "✅ %s: %d/%d tests passed in %.2fs",
                    component_info["name"],
                    results["passed"],
                    results["total"],
                    results["duration"],
                )
            else:
                logger.error(
                    "❌ %s: %d/%d tests failed in %.2fs",
                    component_info["name"],
                    results["failed"],
                    results["total"],
                    results["duration"],
                )

        except subprocess.TimeoutExpired:
            results["duration"] = time.time() - start_time
            results["success"] = False
            results["output"] = "Test execution timed out after 5 minutes"
            logger.error("❌ %s: Tests timed out", component_info["name"])

        except Exception as e:
            results["duration"] = time.time() - start_time
            results["success"] = False
            results["output"] = f"Test execution failed: {str(e)}"
            logger.error("❌ %s: Test execution failed: %s", component_info["name"], e)

        return results

    def run_all_tests(
        self, verbose: bool = False, coverage: bool = False
    ) -> Dict[str, any]:
        """
        Run all Phase 3 component tests.

        Args:
            verbose: Enable verbose output
            coverage: Enable coverage reporting

        Returns:
            Dictionary with comprehensive test results
        """
        logger.info("🚀 Starting Phase 3 comprehensive test suite")
        logger.info(
            "Testing %d components: %s",
            len(self.test_components),
            ", ".join(self.test_components.keys()),
        )

        start_time = time.time()
        all_results = {
            "phase": "Phase 3",
            "components": {},
            "summary": {
                "total_components": len(self.test_components),
                "passed_components": 0,
                "failed_components": 0,
                "total_tests": 0,
                "passed_tests": 0,
                "failed_tests": 0,
                "total_duration": 0.0,
                "success": False,
            },
        }

        # Run tests for each component
        for component in self.test_components.keys():
            logger.info("\n%s", "=" * 60)
            logger.info(
                "Testing Component: %s", self.test_components[component]["name"]
            )
            logger.info("%s", "=" * 60)

            component_results = self.run_component_tests(component, verbose, coverage)
            all_results["components"][component] = component_results

            # Update summary
            if component_results["success"]:
                all_results["summary"]["passed_components"] += 1
            else:
                all_results["summary"]["failed_components"] += 1

            all_results["summary"]["total_tests"] += component_results["total"]
            all_results["summary"]["passed_tests"] += component_results["passed"]
            all_results["summary"]["failed_tests"] += component_results["failed"]

        all_results["summary"]["total_duration"] = time.time() - start_time
        all_results["summary"]["success"] = (
            all_results["summary"]["failed_components"] == 0
        )

        # Print comprehensive summary
        self._print_final_summary(all_results)

        return all_results

    def _parse_test_output(self, results: Dict, output: str) -> None:
        """
        Parse pytest output to extract test statistics.

        Args:
            results: Results dictionary to update
            output: Pytest output string
        """
        try:
            # Look for test summary line like "31 passed in 0.03s"
            import re

            # Pattern for "X passed, Y failed in Z.ZZs"
            summary_pattern = r"(\d+)\s+passed(?:,\s+(\d+)\s+failed)?.*in\s+([\d.]+)s"
            match = re.search(summary_pattern, output)

            if match:
                passed = int(match.group(1))
                failed = int(match.group(2)) if match.group(2) else 0

                results["passed"] = passed
                results["failed"] = failed
                results["total"] = passed + failed
            else:
                # Fallback: count test method calls
                test_lines = [line for line in output.split("\n") if "::test_" in line]
                results["total"] = len(test_lines)

                if "FAILED" in output:
                    failed_lines = [
                        line for line in output.split("\n") if "FAILED" in line
                    ]
                    results["failed"] = len(failed_lines)
                    results["passed"] = results["total"] - results["failed"]
                else:
                    results["passed"] = results["total"]
                    results["failed"] = 0

        except Exception as e:
            logger.warning("Failed to parse test output: %s", e)
            results["total"] = 0
            results["passed"] = 0
            results["failed"] = 0

    def _print_final_summary(self, results: Dict) -> None:
        """
        Print comprehensive test summary.

        Args:
            results: Complete test results dictionary
        """
        summary = results["summary"]

        print("\n" + "=" * 80)
        print("🎯 PHASE 3 TEST SUMMARY")
        print("=" * 80)

        print(
            f"📊 Components: {summary['passed_components']}/{summary['total_components']} passed"
        )
        print(f"🧪 Tests: {summary['passed_tests']}/{summary['total_tests']} passed")
        print(f"⏱️  Duration: {summary['total_duration']:.2f}s")

        if summary["success"]:
            print("\n🎉 ALL PHASE 3 TESTS PASSED! 🎉")
        else:
            print(f"\n❌ {summary['failed_components']} components failed")
            print(f"❌ {summary['failed_tests']} tests failed")

        print("\n📋 Component Details:")
        print("-" * 80)

        for component, comp_results in results["components"].items():
            status = "✅" if comp_results["success"] else "❌"
            print(f"{status} {comp_results['name']} (Task {comp_results['task']})")
            print(f"   Tests: {comp_results['passed']}/{comp_results['total']} passed")
            print(f"   Duration: {comp_results['duration']:.2f}s")
            print(f"   Description: {comp_results['description']}")

            if not comp_results["success"] and comp_results["failed"] > 0:
                print(f"   ⚠️  {comp_results['failed']} tests failed")
            print()

        print("=" * 80)


def main():
    """Main entry point for Phase 3 test runner."""
    parser = argparse.ArgumentParser(
        description="Run Phase 3 tests for SynThesisAI Stream Domain Validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run python scripts/run_phase3_tests.py                    # Run all Phase 3 tests
  uv run python scripts/run_phase3_tests.py --verbose          # Run with verbose output
  uv run python scripts/run_phase3_tests.py --coverage         # Run with coverage reporting
  uv run python scripts/run_phase3_tests.py --component chemistry  # Run only chemistry tests
  uv run python scripts/run_phase3_tests.py --component algorithm_analysis --verbose

Available components:
  - chemistry: Enhanced Chemistry Validation (Task 3.3)
  - biology: Advanced Biology Validation (Task 3.4)
  - technology: Technology Validation Framework (Task 4.1)
  - code_execution: Advanced Code Execution Validation (Task 4.2)
  - algorithm_analysis: Algorithm Analysis Validation (Task 4.3)
        """,
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose test output"
    )

    parser.add_argument(
        "--coverage", "-c", action="store_true", help="Enable coverage reporting"
    )

    parser.add_argument(
        "--component",
        choices=[
            "chemistry",
            "biology",
            "technology",
            "code_execution",
            "algorithm_analysis",
        ],
        help="Run tests for specific component only",
    )

    args = parser.parse_args()

    # Initialize test runner
    runner = Phase3TestRunner()

    try:
        if args.component:
            # Run specific component tests
            results = runner.run_component_tests(
                args.component, verbose=args.verbose, coverage=args.coverage
            )

            # Print component summary
            print("\n" + "=" * 60)
            print(f"🎯 {results['name']} (Task {results['task']})")
            print("=" * 60)
            print(f"Tests: {results['passed']}/{results['total']} passed")
            print(f"Duration: {results['duration']:.2f}s")

            if results["success"]:
                print("\n✅ Component tests passed!")
                return 0
            else:
                print(f"\n❌ {results['failed']} tests failed")
                if args.verbose:
                    print("\nTest Output:")
                    print("-" * 40)
                    print(results["output"])
                return 1
        else:
            # Run all Phase 3 tests
            results = runner.run_all_tests(verbose=args.verbose, coverage=args.coverage)

            return 0 if results["summary"]["success"] else 1

    except KeyboardInterrupt:
        logger.info("\n⚠️  Test execution interrupted by user")
        return 130
    except Exception as e:
        logger.error("❌ Test execution failed: %s", e)
        return 1


if __name__ == "__main__":
    exit(main())
