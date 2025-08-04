#!/usr/bin/env python3
"""
Script to run the comprehensive SynThesisAI user workflow test.

This script provides an easy way to run the end-to-end user workflow test
that simulates real-world usage of the SynThesisAI system.
"""

# Standard Library
import argparse
import logging
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Third-Party Library
import pytest

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main function to run the user workflow test."""
    parser = argparse.ArgumentParser(
        description="Run SynThesisAI User Workflow End-to-End Test"
    )

    parser.add_argument(
        "--test-type",
        choices=["basic", "dspy", "validation", "performance", "error", "full"],
        default="basic",
        help="Type of workflow test to run (default: basic)",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    parser.add_argument(
        "--output-dir", type=str, help="Directory to save test results (optional)"
    )

    args = parser.parse_args()

    # Map test types to specific test methods
    test_mapping = {
        "basic": "test_complete_user_workflow_basic",
        "dspy": "test_complete_user_workflow_with_dspy_optimization",
        "validation": "test_validation_integration_workflow",
        "performance": "test_performance_and_monitoring_workflow",
        "error": "test_error_handling_and_recovery_workflow",
        "full": "test_full_system_integration_workflow",
    }

    test_method = test_mapping[args.test_type]
    test_file = (
        project_root / "tests" / "end_to_end" / "test_synthesisai_user_workflow.py"
    )

    logger.info(f"🚀 Running {args.test_type} workflow test...")
    logger.info(f"📁 Test file: {test_file}")

    # Prepare pytest arguments
    pytest_args = [
        str(test_file) + f"::TestSynThesisAIUserWorkflow::{test_method}",
        "-s",  # Don't capture output
        "--tb=short",  # Short traceback format
    ]

    if args.verbose:
        pytest_args.append("-v")

    # Set environment variable for output directory if provided
    if args.output_dir:
        import os

        os.environ["TEST_OUTPUT_DIR"] = args.output_dir
        logger.info("📂 Test output directory: %s", args.output_dir)

    # Run the test
    logger.info("🔄 Starting test execution...")
    exit_code = pytest.main(pytest_args)

    if exit_code == 0:
        logger.info("✅ Test completed successfully!")
    else:
        logger.error("❌ Test failed!")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
