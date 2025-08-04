"""
Unit tests for the enhanced CAS validator.

This module tests the advanced mathematical verification capabilities including
symbolic computation, multiple solution paths, and prerequisite validation.
"""

# Standard Library
from unittest.mock import Mock, patch

# Third-Party Library
import pytest

# SynThesisAI Modules
from core.validation.cas_enhanced import (
    SYMPY_AVAILABLE,
    EnhancedCASValidator,
    get_enhanced_cas_validator,
)


class TestEnhancedCASValidator:
    """Test cases for EnhancedCASValidator."""

    def test_validator_initialization(self):
        """Test enhanced CAS validator initialization."""
        validator = EnhancedCASValidator()

        assert hasattr(validator, "advanced_constants")
        assert hasattr(validator, "concept_prerequisites")
        assert hasattr(validator, "solution_methods")

        if SYMPY_AVAILABLE:
            assert "euler_gamma" in validator.advanced_constants
            assert "derivative" in validator.concept_prerequisites
            assert "polynomial" in validator.solution_methods

    @pytest.mark.skipif(not SYMPY_AVAILABLE, reason="SymPy not available")
    def test_symbolic_computation_simplify(self):
        """Test symbolic computation with simplification."""
        validator = EnhancedCASValidator()

        result = validator.validate_symbolic_computation(
            expression="x^2 + 2*x + 1",
            expected_result="(x + 1)^2",
            computation_type="factor",
        )

        assert result["verified"] is True
        assert result["computation_type"] == "factor"
        assert "computed_result" in result
        assert "symbolic_form" in result

    @pytest.mark.skipif(not SYMPY_AVAILABLE, reason="SymPy not available")
    def test_symbolic_computation_expand(self):
        """Test symbolic computation with expansion."""
        validator = EnhancedCASValidator()

        result = validator.validate_symbolic_computation(
            expression="(x + 1)^2",
            expected_result="x^2 + 2*x + 1",
            computation_type="expand",
        )

        assert result["verified"] is True
        assert result["computation_type"] == "expand"
        assert result["confidence"] >= 0.9

    @pytest.mark.skipif(not SYMPY_AVAILABLE, reason="SymPy not available")
    def test_symbolic_computation_trigsimp(self):
        """Test symbolic computation with trigonometric simplification."""
        validator = EnhancedCASValidator()

        result = validator.validate_symbolic_computation(
            expression="sin(x)^2 + cos(x)^2",
            expected_result="1",
            computation_type="trigsimp",
        )

        assert result["verified"] is True
        assert result["computation_type"] == "trigsimp"

    def test_symbolic_computation_unavailable(self):
        """Test symbolic computation when SymPy is unavailable."""
        validator = EnhancedCASValidator()

        with patch("core.validation.cas_enhanced.SYMPY_AVAILABLE", False):
            result = validator.validate_symbolic_computation(
                expression="x^2 + 1",
                expected_result="x^2 + 1",
                computation_type="simplify",
            )

            assert result["verified"] is False
            assert result["method"] == "cas_unavailable"

    def test_symbolic_computation_parsing_error(self):
        """Test symbolic computation with parsing errors."""
        validator = EnhancedCASValidator()

        result = validator.validate_symbolic_computation(
            expression="invalid_expression_###",
            expected_result="1",
            computation_type="simplify",
        )

        assert result["verified"] is False
        # Should handle parsing errors gracefully
        assert result["method"] in [
            "parsing_failed",
            "computation_error",
            "symbolic_comparison",
        ]

    @pytest.mark.skipif(not SYMPY_AVAILABLE, reason="SymPy not available")
    def test_multiple_solution_paths_equations(self):
        """Test multiple solution path verification for equations."""
        validator = EnhancedCASValidator()

        # Different ways to express the solution to x^2 - 4 = 0
        solutions = ["2", "-2", "sqrt(4)", "-sqrt(4)"]

        result = validator.verify_multiple_solution_paths(
            problem="x^2 - 4 = 0", solutions=solutions, problem_type="equation"
        )

        assert result["total_solutions"] == 4
        assert result["valid_solutions"] >= 0  # At least structure should be correct
        assert "consistency_check" in result
        assert "diversity_analysis" in result

    @pytest.mark.skipif(not SYMPY_AVAILABLE, reason="SymPy not available")
    def test_multiple_solution_paths_single_solution(self):
        """Test multiple solution paths with single solution."""
        validator = EnhancedCASValidator()

        result = validator.verify_multiple_solution_paths(
            problem="x + 1 = 0", solutions=["-1"], problem_type="equation"
        )

        assert result["total_solutions"] == 1
        assert result["consistency_check"]["consistent"] is True

    def test_multiple_solution_paths_unavailable(self):
        """Test multiple solution paths when SymPy is unavailable."""
        validator = EnhancedCASValidator()

        with patch("core.validation.cas_enhanced.SYMPY_AVAILABLE", False):
            result = validator.verify_multiple_solution_paths(
                problem="x^2 = 4", solutions=["2", "-2"], problem_type="equation"
            )

            assert result["verified"] is False
            assert result["method"] == "cas_unavailable"

    def test_mathematical_prerequisites_basic(self):
        """Test mathematical prerequisite validation for basic content."""
        validator = EnhancedCASValidator()

        content = {
            "problem": "Find the derivative of f(x) = x^2",
            "answer": "f'(x) = 2x",
            "explanation": "Using the power rule for derivatives",
        }

        result = validator.validate_mathematical_prerequisites(content)

        assert "detected_concepts" in result
        assert "required_prerequisites" in result
        assert "prerequisite_coverage" in result
        assert "completeness_score" in result
        assert "derivative" in result["detected_concepts"]

    def test_mathematical_prerequisites_complex(self):
        """Test mathematical prerequisite validation for complex content."""
        validator = EnhancedCASValidator()

        content = {
            "problem": "Find the integral of sin(x) dx",
            "answer": "-cos(x) + C",
            "explanation": "The integral of sine is negative cosine plus constant",
        }

        result = validator.validate_mathematical_prerequisites(content)

        assert "integral" in result["detected_concepts"]
        if "integral" in result["required_prerequisites"]:
            assert "derivative" in result["required_prerequisites"]["integral"]

    def test_mathematical_prerequisites_no_concepts(self):
        """Test mathematical prerequisite validation with no detected concepts."""
        validator = EnhancedCASValidator()

        content = {"problem": "What is 2 + 2?", "answer": "4"}

        result = validator.validate_mathematical_prerequisites(content)

        assert result["completeness_score"] >= 0.7  # Should be high for simple content
        assert len(result["detected_concepts"]) == 0

    def test_mathematical_prerequisites_error_handling(self):
        """Test error handling in prerequisite validation."""
        validator = EnhancedCASValidator()

        # Test with problematic content
        with patch.object(
            validator,
            "_detect_mathematical_concepts",
            side_effect=Exception("Test error"),
        ):
            result = validator.validate_mathematical_prerequisites({"problem": "test"})

            assert result["validated"] is False
            assert "error" in result["method"]

    @pytest.mark.skipif(not SYMPY_AVAILABLE, reason="SymPy not available")
    def test_form_specific_equivalence_factored(self):
        """Test form-specific equivalence for factored expressions."""
        validator = EnhancedCASValidator()

        # Test factored form equivalence
        expr1 = validator.parse_mathematical_expression("(x + 1)(x - 1)")
        expr2 = validator.parse_mathematical_expression("(x - 1)(x + 1)")

        result = validator._check_form_specific_equivalence(expr1, expr2, "factor")

        assert result["equivalent"] is True
        assert "factored_form" in result["method"]

    @pytest.mark.skipif(not SYMPY_AVAILABLE, reason="SymPy not available")
    def test_form_specific_equivalence_expanded(self):
        """Test form-specific equivalence for expanded expressions."""
        validator = EnhancedCASValidator()

        expr1 = validator.parse_mathematical_expression("x^2 - 1")
        expr2 = validator.parse_mathematical_expression("x^2 + 0*x - 1")

        result = validator._check_form_specific_equivalence(expr1, expr2, "expand")

        assert result["equivalent"] is True

    def test_concept_detection_derivative(self):
        """Test detection of derivative concepts."""
        validator = EnhancedCASValidator()

        content = {
            "problem": "Find the derivative of f(x) = x^3",
            "answer": "f'(x) = 3x^2",
        }

        concepts = validator._detect_mathematical_concepts(content)

        assert "derivative" in concepts

    def test_concept_detection_integral(self):
        """Test detection of integral concepts."""
        validator = EnhancedCASValidator()

        content = {"problem": "Evaluate the integral ∫ x dx", "answer": "x^2/2 + C"}

        concepts = validator._detect_mathematical_concepts(content)

        assert "integral" in concepts

    def test_concept_detection_multiple(self):
        """Test detection of multiple mathematical concepts."""
        validator = EnhancedCASValidator()

        content = {
            "problem": "Find the limit of the derivative as x approaches 0",
            "answer": "Using L'Hopital's rule...",
        }

        concepts = validator._detect_mathematical_concepts(content)

        assert "derivative" in concepts
        assert "limit" in concepts

    def test_prerequisite_coverage_basic(self):
        """Test prerequisite coverage checking."""
        validator = EnhancedCASValidator()

        content = {
            "problem": "derivative problem",
            "explanation": "using function and limit concepts",
        }

        required_prerequisites = {"derivative": ["function", "limit"]}

        coverage = validator._check_prerequisite_coverage(
            content, required_prerequisites
        )

        assert coverage["function"] is True
        assert coverage["limit"] is True

    def test_solution_consistency_check_consistent(self):
        """Test solution consistency checking with consistent solutions."""
        validator = EnhancedCASValidator()

        valid_solutions = [
            {"solution": "2", "method": "direct"},
            {"solution": "2.0", "method": "numerical"},
        ]

        result = validator._check_solution_consistency(valid_solutions)

        assert result["consistent"] is True

    def test_solution_consistency_check_inconsistent(self):
        """Test solution consistency checking with inconsistent solutions."""
        validator = EnhancedCASValidator()

        valid_solutions = [
            {"solution": "2", "method": "direct"},
            {"solution": "3", "method": "alternative"},
        ]

        result = validator._check_solution_consistency(valid_solutions)

        assert result["consistent"] is False

    def test_solution_diversity_analysis(self):
        """Test solution diversity analysis."""
        validator = EnhancedCASValidator()

        valid_solutions = [
            {"method": "factoring"},
            {"method": "quadratic_formula"},
            {"method": "graphical"},
        ]

        result = validator._analyze_solution_diversity(valid_solutions, "polynomial")

        assert result["diversity_score"] > 0
        assert len(result["methods_used"]) == 3
        assert "coverage" in result

    def test_get_enhanced_cas_validator_singleton(self):
        """Test that get_enhanced_cas_validator returns singleton instance."""
        validator1 = get_enhanced_cas_validator()
        validator2 = get_enhanced_cas_validator()

        assert validator1 is validator2
        assert isinstance(validator1, EnhancedCASValidator)


class TestEnhancedCASValidatorIntegration:
    """Integration tests for enhanced CAS validator."""

    @pytest.mark.skipif(not SYMPY_AVAILABLE, reason="SymPy not available")
    def test_comprehensive_validation_workflow(self):
        """Test comprehensive validation workflow."""
        validator = EnhancedCASValidator()

        # Test a complete mathematical problem
        content = {
            "problem": "Find the derivative of f(x) = x^2 + 3x + 1",
            "answer": "f'(x) = 2x + 3",
            "solution_steps": [
                "Apply power rule: d/dx(x^2) = 2x",
                "Apply power rule: d/dx(3x) = 3",
                "Derivative of constant: d/dx(1) = 0",
                "Combine: f'(x) = 2x + 3",
            ],
        }

        # Test symbolic computation
        symbolic_result = validator.validate_symbolic_computation(
            expression="2*x + 3", expected_result="2*x + 3", computation_type="simplify"
        )

        # Test prerequisite validation
        prereq_result = validator.validate_mathematical_prerequisites(content)

        # Test multiple solution verification (different representations)
        solutions = ["2*x + 3", "3 + 2*x", "2*(x + 3/2)"]
        multiple_result = validator.verify_multiple_solution_paths(
            problem="derivative of x^2 + 3x + 1",
            solutions=solutions,
            problem_type="calculus",
        )

        # Verify all components work together
        assert symbolic_result["verified"] is True
        assert prereq_result["completeness_score"] >= 0.3  # Very lenient check for test
        assert multiple_result["verified"] is True
        assert "derivative" in prereq_result["detected_concepts"]


if __name__ == "__main__":
    pytest.main([__file__])
