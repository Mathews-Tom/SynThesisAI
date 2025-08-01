"""
Comprehensive end-to-end tests for Phase 1 DSPy Integration.

This module contains all critical tests to validate Phase 1 implementation
including DSPy agents, optimization engine, caching, signatures, and feedback systems.
"""

# Standard Library
import logging
import sys

# Third-Party Library
import pytest  # pylint: disable=import-error,no-member

# SynThesisAI Modules
from core.dspy.base_module import STREAMContentGenerator
from core.dspy.cache import get_optimization_cache
from core.dspy.checker_agent import DSPyCheckerAgent
from core.dspy.continuous_learning import get_continuous_learning_manager
from core.dspy.engineer_agent import DSPyEngineerAgent
from core.dspy.feedback import (
    FeedbackSeverity,
    FeedbackSource,
    FeedbackType,
    get_feedback_manager,
)
from core.dspy.optimization_engine import get_optimization_engine
from core.dspy.optimization_workflows import OptimizationWorkflowManager
from core.dspy.quality_assessment import get_quality_assessor
from core.dspy.signatures import SignatureManager, get_all_domains, get_domain_signature
from core.dspy.target_agent import DSPyTargetAgent

# pylint: disable=import-error,no-member


# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.mark.integration
class TestPhase1DSPyAgents:
    """Test individual DSPy agents functionality."""

    def test_engineer_agent_basic_functionality(self) -> None:
        """
        Test DSPy Engineer Agent basic problem generation.

        Raises:
            Exception: If generation fails.
        """
        agent = DSPyEngineerAgent()
        result = agent.generate_problem(
            {"difficulty_level": "high_school", "topic": "algebra"}
        )
        assert isinstance(result, dict) and result
        logger.info("✅ Engineer Agent test passed")

    def test_checker_agent_basic_functionality(self) -> None:
        """
        Test DSPy Checker Agent basic validation.

        Raises:
            Exception: If validation fails.
        """
        agent = DSPyCheckerAgent()
        data = {
            "problem_statement": "Solve x^2 + 5x + 6 = 0",
            "solution": "x = -2 or x = -3",
            "reasoning_trace": "Factor the quadratic equation",
        }
        result = agent.validate_problem(data)
        assert isinstance(result, dict) and result
        logger.info("✅ Checker Agent test passed")

    def test_target_agent_basic_functionality(self) -> None:
        """
        Test DSPy Target Agent basic problem solving.

        Raises:
            Exception: If solving fails.
        """
        agent = DSPyTargetAgent()
        result = agent.solve_problem({"problem_statement": "Solve x^2 + 5x + 6 = 0"})
        assert isinstance(result, dict) and result
        logger.info("✅ Target Agent test passed")


@pytest.mark.integration
class TestPhase1OptimizationEngine:
    """Test optimization engine functionality."""

    def test_optimization_engine_basic(self) -> None:
        """
        Test optimization engine basic functionality.

        Raises:
            Exception: If engine is unavailable.
        """
        engine = get_optimization_engine()
        assert hasattr(engine, "optimize_for_domain")
        logger.info("✅ Optimization Engine test passed")

    def test_optimization_cache_functionality(self) -> None:
        """
        Test optimization caching system.

        Raises:
            Exception: If cache stats are invalid.
        """
        cache = get_optimization_cache()
        stats = cache.get_stats()
        assert isinstance(stats, dict)
        assert "hit_rate" in stats and "memory_entries" in stats
        logger.info("✅ Cache System test passed")


@pytest.mark.integration
class TestPhase1SignatureSystem:
    """Test signature management system."""

    def test_signature_registry_functionality(self) -> None:
        """
        Test domain listing and signature retrieval.

        Raises:
            Exception: If signature operations fail.
        """
        domains = get_all_domains()
        assert isinstance(domains, list) and "mathematics" in domains
        sig = get_domain_signature("mathematics")
        assert sig is not None
        logger.info("✅ Signature System test passed")

    def test_signature_manager_functionality(self) -> None:
        """
        Test signature manager basic operations.

        Raises:
            Exception: If manager fails.
        """
        manager = SignatureManager()
        assert manager is not None
        logger.info("✅ Signature Manager test passed")


@pytest.mark.integration
class TestPhase1FeedbackSystem:
    """Test feedback and continuous learning systems."""

    def test_feedback_manager_functionality(self) -> None:
        """
        Test feedback manager operations.

        Raises:
            Exception: If feedback operations fail.
        """
        mgr = get_feedback_manager()
        fid = mgr.add_feedback(
            content="Test feedback",
            feedback_type=FeedbackType.ACCURACY,
            source=FeedbackSource.SYSTEM,
            domain="mathematics",
            severity=FeedbackSeverity.MEDIUM,
        )
        assert fid is not None
        summary = mgr.get_feedback_summary("mathematics")
        assert isinstance(summary, dict)
        logger.info("✅ Feedback System test passed")

    def test_continuous_learning_manager(self) -> None:
        """
        Test continuous learning manager operations.

        Raises:
            Exception: If continuous learning fails.
        """
        mgr = get_continuous_learning_manager()
        summary = mgr.get_learning_summary()
        assert isinstance(summary, dict)
        logger.info("✅ Continuous Learning test passed")


@pytest.mark.integration
class TestPhase1EndToEndWorkflow:
    """Complete end-to-end workflow tests."""

    def test_complete_workflow(self) -> None:
        """
        Test complete DSPy workflow from generation to solution.

        Raises:
            Exception: If any step fails.
        """
        engineer = DSPyEngineerAgent()
        checker = DSPyCheckerAgent()
        target = DSPyTargetAgent()

        problem = engineer.generate_problem(
            {"topic": "algebra", "difficulty_level": "high_school"}
        )
        assert isinstance(problem, dict) and problem
        logger.info("✅ Problem generated successfully")

        if "problem_statement" in problem:
            val = checker.validate_problem(problem)
            assert isinstance(val, dict)
            logger.info("✅ Validation completed")

        sol = target.solve_problem(
            {"problem_statement": problem.get("problem_statement", "")}
        )
        assert isinstance(sol, dict) and sol
        logger.info("✅ Solution generated successfully")
        logger.info("✅ Complete end-to-end workflow test passed!")

    def test_optimization_workflow_integration(self) -> None:
        """
        Test integration with optimization workflows.

        Raises:
            Exception: If workflow manager fails.
        """
        mgr = OptimizationWorkflowManager()
        assert mgr is not None
        logger.info("✅ Optimization Workflow Integration test passed")


@pytest.mark.integration
class TestPhase1QualityAssurance:
    """Test quality assurance integration."""

    def test_quality_assessment_integration(self) -> None:
        """
        Test quality assessor integration.

        Raises:
            Exception: If assessor fails.
        """
        assessor = get_quality_assessor()
        assert assessor is not None
        logger.info("✅ Quality Assessment Integration test passed")

    def test_quality_metrics_calculation(self) -> None:
        """
        Test quality metrics calculation for sample content.

        Raises:
            Exception: If metrics calculation fails.
        """
        base = STREAMContentGenerator("mathematics")
        sample = {
            "problem_statement": "Solve the equation x^2 + 5x + 6 = 0",
            "solution": "x = -2 or x = -3",
            "reasoning_trace": "Factor the quadratic equation...",
        }
        metrics = base.calculate_quality_metrics(sample)
        assert isinstance(metrics, dict) and "overall_quality" in metrics
        logger.info("✅ Quality Metrics test passed")


def run_all_phase1_tests() -> bool:
    """
    Run all Phase 1 tests programmatically.

    Returns:
        bool: True if all tests passed, False otherwise.
    """
    logger.info("🚀 Starting Phase 1 Comprehensive Tests...")
    test_classes = [
        TestPhase1DSPyAgents,
        TestPhase1OptimizationEngine,
        TestPhase1SignatureSystem,
        TestPhase1FeedbackSystem,
        TestPhase1EndToEndWorkflow,
        TestPhase1QualityAssurance,
    ]
    total, passed = 0, 0
    for cls in test_classes:
        logger.info("📋 Running %s...", cls.__name__)
        for name in dir(cls):
            if name.startswith("test_"):
                total += 1
                try:
                    getattr(cls(), name)()
                    passed += 1
                    logger.info("  ✅ %s - PASSED", name)
                except Exception as e:
                    logger.error("  ❌ %s - FAILED: %s", name, e)

    logger.info("📊 Phase 1 Test Results: %d/%d", passed, total)
    if passed == total:
        logger.info("🎉 All Phase 1 tests passed! Ready for production.")
        return True
    logger.error("⚠️  %d tests failed.", total - passed)
    return False


if __name__ == "__main__":
    success = run_all_phase1_tests()
    sys.exit(0 if success else 1)
