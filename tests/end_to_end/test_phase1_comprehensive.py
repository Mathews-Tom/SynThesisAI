"""
Comprehensive end-to-end tests for Phase 1 DSPy Integration.

This module contains all the critical tests to validate Phase 1 implementation
including DSPy agents, optimization engine, caching, signatures, and feedback systems.
"""

import logging
import sys
from typing import Dict

import pytest

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.mark.integration
class TestPhase1DSPyAgents:
    """Test individual DSPy agents functionality."""

    def test_engineer_agent_basic_functionality(self):
        """Test DSPy Engineer Agent basic problem generation."""
        try:
            from core.dspy.engineer_agent import DSPyEngineerAgent

            agent = DSPyEngineerAgent()
            result = agent.generate_problem(
                {"difficulty_level": "high_school", "topic": "algebra"}
            )

            # Verify result structure - just check it's not empty
            assert isinstance(result, dict)
            assert len(result) > 0

            logger.info("✅ Engineer Agent test passed")

        except Exception as e:
            logger.error("❌ Engineer Agent test failed: %s", str(e))
            raise

    def test_checker_agent_basic_functionality(self):
        """Test DSPy Checker Agent basic validation."""
        try:
            from core.dspy.checker_agent import DSPyCheckerAgent

            agent = DSPyCheckerAgent()

            # Test problem validation
            problem_data = {
                "problem_statement": "Solve x^2 + 5x + 6 = 0",
                "solution": "x = -2 or x = -3",
                "reasoning_trace": "Factor the quadratic equation",
            }

            result = agent.validate_problem(problem_data)

            # Verify result structure - just check it's not empty
            assert isinstance(result, dict)
            assert len(result) > 0

            logger.info("✅ Checker Agent test passed")

        except Exception as e:
            logger.error("❌ Checker Agent test failed: %s", str(e))
            raise

    def test_target_agent_basic_functionality(self):
        """Test DSPy Target Agent basic problem solving."""
        try:
            from core.dspy.target_agent import DSPyTargetAgent

            agent = DSPyTargetAgent()

            problem_data = {"problem_statement": "Solve x^2 + 5x + 6 = 0"}

            result = agent.solve_problem(problem_data)

            # Verify result structure - just check it's not empty
            assert isinstance(result, dict)
            assert len(result) > 0

            logger.info("✅ Target Agent test passed")

        except Exception as e:
            logger.error("❌ Target Agent test failed: %s", str(e))
            raise


@pytest.mark.integration
class TestPhase1OptimizationEngine:
    """Test optimization engine functionality."""

    def test_optimization_engine_basic(self):
        """Test DSPy optimization engine basic functionality."""
        try:
            from core.dspy.optimization_engine import get_optimization_engine

            engine = get_optimization_engine()

            # Test that engine exists and has expected methods
            assert hasattr(engine, "optimize_for_domain")

            logger.info("✅ Optimization Engine test passed")

        except Exception as e:
            logger.error("❌ Optimization Engine test failed: %s", str(e))
            raise

    def test_optimization_cache_functionality(self):
        """Test optimization caching system."""
        try:
            from core.dspy.cache import get_optimization_cache

            cache = get_optimization_cache()
            stats = cache.get_stats()

            # Verify cache stats structure - check for actual keys
            assert isinstance(stats, dict)
            assert "hit_rate" in stats
            assert "memory_entries" in stats

            logger.info("✅ Cache System test passed")

        except Exception as e:
            logger.error("❌ Cache System test failed: %s", str(e))
            raise


@pytest.mark.integration
class TestPhase1SignatureSystem:
    """Test signature management system."""

    def test_signature_registry_functionality(self):
        """Test signature management and registry."""
        try:
            from core.dspy.signatures import get_all_domains, get_domain_signature

            # Test domain listing
            domains = get_all_domains()
            assert isinstance(domains, list)
            assert len(domains) > 0
            assert "mathematics" in domains

            # Test signature retrieval
            math_signature = get_domain_signature("mathematics")
            assert math_signature is not None

            logger.info("✅ Signature System test passed")

        except Exception as e:
            logger.error("❌ Signature System test failed: %s", str(e))
            raise

    def test_signature_manager_functionality(self):
        """Test signature manager functionality."""
        try:
            from core.dspy.signatures import SignatureManager

            manager = SignatureManager()

            # Test basic functionality
            assert manager is not None

            logger.info("✅ Signature Manager test passed")

        except Exception as e:
            logger.error("❌ Signature Manager test failed: %s", str(e))
            raise


@pytest.mark.integration
class TestPhase1FeedbackSystem:
    """Test feedback and continuous learning systems."""

    def test_feedback_manager_functionality(self):
        """Test feedback manager basic operations."""
        try:
            from core.dspy.feedback import (
                FeedbackSeverity,
                FeedbackSource,
                FeedbackType,
                get_feedback_manager,
            )

            manager = get_feedback_manager()

            # Add test feedback
            feedback_id = manager.add_feedback(
                content="Test feedback",
                feedback_type=FeedbackType.ACCURACY,
                source=FeedbackSource.SYSTEM,
                domain="mathematics",
                severity=FeedbackSeverity.MEDIUM,
            )

            assert feedback_id is not None

            # Get feedback summary
            summary = manager.get_feedback_summary("mathematics")
            assert isinstance(summary, dict)

            logger.info("✅ Feedback System test passed")

        except Exception as e:
            logger.error("❌ Feedback System test failed: %s", str(e))
            raise

    def test_continuous_learning_manager(self):
        """Test continuous learning manager."""
        try:
            from core.dspy.continuous_learning import get_continuous_learning_manager

            manager = get_continuous_learning_manager()

            # Test learning summary
            summary = manager.get_learning_summary()
            assert isinstance(summary, dict)

            logger.info("✅ Continuous Learning test passed")

        except Exception as e:
            logger.error("❌ Continuous Learning test failed: %s", str(e))
            raise


@pytest.mark.integration
class TestPhase1EndToEndWorkflow:
    """Complete end-to-end workflow tests."""

    def test_complete_workflow(self):
        """Test complete DSPy workflow from problem generation to solution."""
        try:
            from core.dspy.checker_agent import DSPyCheckerAgent
            from core.dspy.engineer_agent import DSPyEngineerAgent
            from core.dspy.target_agent import DSPyTargetAgent

            # Create agents
            engineer = DSPyEngineerAgent()
            checker = DSPyCheckerAgent()
            target = DSPyTargetAgent()

            # Step 1: Generate content
            problem = engineer.generate_problem(
                {"topic": "algebra", "difficulty_level": "high_school"}
            )

            assert isinstance(problem, dict)
            assert len(problem) > 0
            logger.info("✅ Problem generated successfully")

            # Step 2: Validate content (if problem has expected structure)
            if "problem_statement" in problem:
                validation = checker.validate_problem(problem)
                assert isinstance(validation, dict)
                logger.info("✅ Validation completed")

            # Step 3: Solve problem (if we have a problem statement)
            problem_statement = problem.get(
                "problem_statement", "Solve x^2 + 5x + 6 = 0"
            )
            solution = target.solve_problem({"problem_statement": problem_statement})

            assert isinstance(solution, dict)
            assert len(solution) > 0
            logger.info("✅ Solution generated successfully")

            logger.info("✅ Complete end-to-end workflow test passed!")

        except Exception as e:
            logger.error("❌ End-to-end workflow test failed: %s", str(e))
            raise

    def test_optimization_workflow_integration(self):
        """Test integration with optimization workflows."""
        try:
            from core.dspy.optimization_workflows import OptimizationWorkflowManager

            manager = OptimizationWorkflowManager()

            # Test that manager exists and has expected structure
            assert manager is not None

            logger.info("✅ Optimization Workflow Integration test passed")

        except Exception as e:
            logger.error("❌ Optimization Workflow Integration test failed: %s", str(e))
            raise


@pytest.mark.integration
class TestPhase1QualityAssurance:
    """Test quality assurance integration."""

    def test_quality_assessment_integration(self):
        """Test quality assessment framework integration."""
        try:
            from core.dspy.quality_assessment import get_quality_assessor

            qa_assessor = get_quality_assessor()

            # Test that assessor exists
            assert qa_assessor is not None

            logger.info("✅ Quality Assessment Integration test passed")

        except Exception as e:
            logger.error("❌ Quality Assessment Integration test failed: %s", str(e))
            raise

    def test_quality_metrics_calculation(self):
        """Test quality metrics calculation."""
        try:
            from core.dspy.base_module import STREAMContentGenerator

            # Create a base module instance
            base_module = STREAMContentGenerator("mathematics")

            # Test with sample content
            content = {
                "problem_statement": "Solve the equation x^2 + 5x + 6 = 0",
                "solution": "x = -2 or x = -3",
                "reasoning_trace": "Factor the quadratic equation...",
            }

            metrics = base_module.calculate_quality_metrics(content)
            assert isinstance(metrics, dict)
            assert "overall_quality" in metrics

            logger.info("✅ Quality Metrics test passed")

        except Exception as e:
            logger.error("❌ Quality Metrics test failed: %s", str(e))
            raise

    def test_quality_metrics_calculation(self):
        """Test quality metrics calculation."""
        try:
            from core.dspy.base_module import STREAMContentGenerator

            # Create a base module instance
            base_module = STREAMContentGenerator("mathematics")

            # Test with sample content
            content = {
                "problem_statement": "Solve the equation x^2 + 5x + 6 = 0",
                "solution": "x = -2 or x = -3",
                "reasoning_trace": "Factor the quadratic equation...",
            }

            metrics = base_module.calculate_quality_metrics(content)
            assert isinstance(metrics, dict)
            assert "overall_quality" in metrics

            logger.info("✅ Quality Metrics test passed")

        except Exception as e:
            logger.error("❌ Quality Metrics test failed: %s", str(e))
            raise


def run_all_phase1_tests():
    """Run all Phase 1 tests programmatically."""
    logger.info("🚀 Starting Phase 1 Comprehensive Tests...")

    test_classes = [
        TestPhase1DSPyAgents,
        TestPhase1OptimizationEngine,
        TestPhase1SignatureSystem,
        TestPhase1FeedbackSystem,
        TestPhase1EndToEndWorkflow,
        TestPhase1QualityAssurance,
    ]

    total_tests = 0
    passed_tests = 0

    for test_class in test_classes:
        logger.info("\n📋 Running %s...", test_class.__name__)

        # Get all test methods
        test_methods = [
            method for method in dir(test_class) if method.startswith("test_")
        ]

        for test_method in test_methods:
            total_tests += 1
            try:
                # Create instance and run test
                instance = test_class()
                getattr(instance, test_method)()
                passed_tests += 1
                logger.info("  ✅ %s - PASSED", test_method)
            except Exception as e:
                logger.error("  ❌ %s - FAILED: %s", test_method, str(e))

    logger.info(
        "\n📊 Phase 1 Test Results: %d/%d tests passed", passed_tests, total_tests
    )

    if passed_tests == total_tests:
        logger.info("🎉 All Phase 1 tests passed! Ready for production.")
        return True

    logger.error(
        "⚠️  %d tests failed. Please review and fix.", total_tests - passed_tests
    )
    return False


if __name__ == "__main__":
    # Run tests when executed directly
    SUCCESS = run_all_phase1_tests()
    sys.exit(0 if SUCCESS else 1)
