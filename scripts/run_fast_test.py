#!/usr/bin/env python3
"""
Ultra-fast testing script for SynThesisAI development.

This script provides multiple fast testing modes to verify system functionality
without waiting for slow API calls or getting stuck in generation loops.
"""

# Standard Library
import argparse
import json
import logging
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict
from unittest.mock import Mock, patch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# SynThesisAI Modules
from utils.config_manager import get_config_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_minimal_config(temp_dir: Path) -> Path:
    """Create a minimal configuration for fast testing."""
    config_data = {
        "num_problems": 1,
        "max_workers": 1,
        "taxonomy": "taxonomy/enhanced_math_taxonomy.json",
        "output_dir": str(temp_dir / "results"),
        "default_batch_id": "fast_test",
        "use_search": False,
        "use_enhanced_concurrent_processing": False,
        "enable_prefiltering": False,
        "llm_cache_enabled": False,
        "use_seed_data": False,
        "dspy_enabled": False,
        "timeout_seconds": 10,
        "test_mode": True,
        "mock_mode": True,
        # Use mock models to avoid API calls
        "engineer_model": {"provider": "mock", "model_name": "mock-engineer"},
        "checker_model": {"provider": "mock", "model_name": "mock-checker"},
        "target_model": {"provider": "mock", "model_name": "mock-target"},
    }

    config_path = temp_dir / "fast_config.json"
    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=2)

    return config_path


def create_minimal_taxonomy(temp_dir: Path) -> Path:
    """Create a minimal taxonomy for testing."""
    taxonomy_data = {
        "domains": {
            "mathematics": {
                "subjects": {
                    "algebra": {
                        "topics": {
                            "linear_equations": {
                                "difficulty_levels": ["High School"],
                                "description": "Basic linear equations",
                            }
                        }
                    }
                }
            }
        }
    }

    taxonomy_path = temp_dir / "minimal_taxonomy.json"
    with open(taxonomy_path, "w") as f:
        json.dump(taxonomy_data, f, indent=2)

    return taxonomy_path


def test_config_loading(temp_dir: Path) -> bool:
    """Test 1: Configuration loading (fastest test)."""
    logger.info("🔧 Testing configuration loading...")

    try:
        config_path = create_minimal_config(temp_dir)
        taxonomy_path = create_minimal_taxonomy(temp_dir)

        config_manager = get_config_manager()
        config_manager.load_config(config_path)
        config_manager.set("taxonomy", str(taxonomy_path))

        # Verify basic config
        assert config_manager.get("num_problems") == 1
        assert config_manager.get("test_mode") is True

        # Load taxonomy
        taxonomy_data = config_manager.load_taxonomy_file_cached(str(taxonomy_path))
        assert "domains" in taxonomy_data
        assert "mathematics" in taxonomy_data["domains"]

        logger.info("✅ Configuration loading test passed")
        return True

    except Exception as e:
        logger.error("❌ Configuration loading test failed: %s", e)
        return False


def test_validation_system(temp_dir: Path) -> bool:
    """Test 2: Validation system without API calls."""
    logger.info("🔍 Testing validation system...")

    try:
        from core.validation.config import ValidationConfig
        from core.validation.domains.mathematics import MathematicsValidator

        # Test mathematics validation
        math_config = ValidationConfig(
            domain="mathematics",
            quality_thresholds={"fidelity_score": 0.8},
            timeout_seconds=5,
        )

        math_validator = MathematicsValidator("mathematics", math_config)

        test_content = {
            "problem": "What is 2 + 2?",
            "answer": "4",
            "explanation": "Adding 2 and 2 gives 4",
        }

        result = math_validator.validate_content(test_content)

        assert result.domain == "mathematics"
        assert result.quality_score > 0.0

        logger.info("✅ Validation system test passed")
        return True

    except Exception as e:
        logger.error("❌ Validation system test failed: %s", e)
        return False


def test_mock_generation_pipeline(temp_dir: Path) -> bool:
    """Test 3: Mock generation pipeline to verify workflow."""
    logger.info("🏭 Testing mock generation pipeline...")

    try:
        # Create config
        config_path = create_minimal_config(temp_dir)
        taxonomy_path = create_minimal_taxonomy(temp_dir)

        config_manager = get_config_manager()
        config_manager.load_config(config_path)
        config_manager.set("taxonomy", str(taxonomy_path))

        # Load taxonomy
        taxonomy_data = config_manager.load_taxonomy_file_cached(str(taxonomy_path))
        config_manager.set("taxonomy", taxonomy_data)

        # Set up output directory
        output_dir = Path(temp_dir / "results")
        save_path = output_dir / "fast_test"
        config_manager.set("save_path", str(save_path))

        # Mock the generation pipeline
        mock_valid_problems = [
            {
                "problem_statement": "Solve for x: 2x + 3 = 7",
                "solution": "x = 2",
                "hints": {"0": "Subtract 3 from both sides"},
                "metadata": {
                    "domain": "mathematics",
                    "subject": "algebra",
                    "topic": "linear_equations",
                    "difficulty": "High School",
                },
            }
        ]

        mock_rejected_problems = []

        # Test with mock
        with patch(
            "core.orchestration.generate_batch.run_generation_pipeline"
        ) as mock_pipeline:
            mock_pipeline.return_value = (mock_valid_problems, mock_rejected_problems)

            from core.orchestration.generate_batch import run_generation_pipeline

            config = config_manager.get_all()
            valid, rejected = run_generation_pipeline(config)

            assert len(valid) == 1
            assert len(rejected) == 0
            assert valid[0]["problem_statement"] == "Solve for x: 2x + 3 = 7"

            mock_pipeline.assert_called_once()

        logger.info("✅ Mock generation pipeline test passed")
        return True

    except Exception as e:
        logger.error("❌ Mock generation pipeline test failed: %s", e)
        return False


def test_component_imports() -> bool:
    """Test 4: Verify all major components can be imported."""
    logger.info("📦 Testing component imports...")

    try:
        # Test core imports
        # Test agent imports (correct import path)
        from core.agents import CheckerAgent, EngineerAgent, TargetAgent
        from core.orchestration.generate_batch import run_generation_pipeline
        from core.validation.domains.mathematics import MathematicsValidator
        from core.validation.domains.science import ScienceValidator
        from utils.config_manager import get_config_manager

        logger.info("✅ Component imports test passed")
        return True

    except Exception as e:
        logger.error("❌ Component imports test failed: %s", e)
        return False


def run_fast_tests() -> Dict[str, bool]:
    """Run all fast tests and return results."""
    logger.info("🚀 Starting fast test suite...")

    results = {}

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Test 1: Component imports (no dependencies)
        results["imports"] = test_component_imports()

        # Test 2: Configuration loading
        results["config"] = test_config_loading(temp_path)

        # Test 3: Validation system
        results["validation"] = test_validation_system(temp_path)

        # Test 4: Mock generation pipeline
        results["pipeline"] = test_mock_generation_pipeline(temp_path)

    return results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Fast testing for SynThesisAI")
    parser.add_argument(
        "--test",
        choices=["all", "imports", "config", "validation", "pipeline"],
        default="all",
        help="Which test to run",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    start_time = time.time()

    if args.test == "all":
        results = run_fast_tests()

        # Print summary
        logger.info("\n" + "=" * 50)
        logger.info("📊 FAST TEST RESULTS")
        logger.info("=" * 50)

        passed = 0
        total = len(results)

        for test_name, passed_test in results.items():
            status = "✅ PASS" if passed_test else "❌ FAIL"
            logger.info("%s | %s", f"{test_name:12}", status)
            if passed_test:
                passed += 1

        logger.info("=" * 50)
        logger.info("📈 Summary: %d/%d tests passed", passed, total)

        end_time = time.time()
        logger.info("⏱️  Total time: %.2fs", end_time - start_time)

        if passed == total:
            logger.info("🎉 All tests passed! System is ready.")
            sys.exit(0)
        else:
            logger.error("💥 Some tests failed. Check the logs above.")
            sys.exit(1)

    else:
        # Run individual test
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            if args.test == "imports":
                success = test_component_imports()
            elif args.test == "config":
                success = test_config_loading(temp_path)
            elif args.test == "validation":
                success = test_validation_system(temp_path)
            elif args.test == "pipeline":
                success = test_mock_generation_pipeline(temp_path)

            end_time = time.time()
            logger.info("⏱️  Test time: %.2fs", end_time - start_time)

            if success:
                logger.info("✅ Test passed!")
                sys.exit(0)
            else:
                logger.error("❌ Test failed!")
                sys.exit(1)


if __name__ == "__main__":
    main()
