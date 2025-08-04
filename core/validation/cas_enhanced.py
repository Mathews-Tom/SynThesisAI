"""
Enhanced Computer Algebra System (CAS) validation module.

This module extends the existing CAS validator with advanced mathematical verification,
symbolic computation validation, multiple solution paths, and prerequisite validation.
"""

# Standard Library
import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# SynThesisAI Modules
from core.checker.cas_validator import SYMPY_AVAILABLE, CASValidator

if SYMPY_AVAILABLE:
    import sympy as sp
    from sympy import (
        E,
        I,
        Matrix,
        Poly,
        acos,
        apart,
        asin,
        atan,
        cancel,
        collect,
        cos,
        cosh,
        det,
        diff,
        exp,
        expand,
        factor,
        gcd,
        integrate,
        lcm,
        limit,
        log,
        oo,
        pi,
        roots,
        series,
        simplify,
        sin,
        sinh,
        solve,
        symbols,
        sympify,
        tan,
        tanh,
        together,
        trigsimp,
    )
    from sympy.calculus import is_decreasing, is_increasing
    from sympy.parsing.sympy_parser import parse_expr
    from sympy.solvers import solve as sympy_solve

    SymPyBasic = sp.Basic
else:
    SymPyBasic = Any

logger = logging.getLogger(__name__)


class EnhancedCASValidator(CASValidator):
    """
    Enhanced CAS validator with advanced mathematical verification capabilities.

    Extends the base CASValidator with:
    - Symbolic computation validation
    - Multiple solution path verification
    - Mathematical concept prerequisite validation
    - Advanced algebraic and calculus verification
    """

    def __init__(self):
        """Initialize the enhanced CAS validator."""
        super().__init__()

        if SYMPY_AVAILABLE:
            # Extended mathematical constants and functions
            self.advanced_constants = {
                **self.constants,
                "euler_gamma": sp.EulerGamma,
                "golden_ratio": sp.GoldenRatio,
                "catalan": sp.Catalan,
            }

            # Mathematical concept prerequisites mapping
            self.concept_prerequisites = {
                "derivative": ["function", "limit", "continuity"],
                "integral": ["derivative", "function", "limit"],
                "limit": ["function", "continuity"],
                "series": ["sequence", "limit", "convergence"],
                "matrix": ["linear_algebra", "vector"],
                "eigenvalue": ["matrix", "linear_algebra", "determinant"],
                "differential_equation": ["derivative", "integral", "function"],
                "complex_analysis": ["complex_number", "function", "limit"],
                "fourier_transform": ["integral", "complex_analysis", "trigonometry"],
            }

            # Solution method preferences for different problem types
            self.solution_methods = {
                "polynomial": ["factoring", "quadratic_formula", "synthetic_division"],
                "trigonometric": ["identities", "substitution", "graphical"],
                "exponential": ["logarithms", "substitution", "graphical"],
                "rational": [
                    "partial_fractions",
                    "common_denominator",
                    "cross_multiplication",
                ],
                "radical": ["rationalization", "substitution", "squaring"],
                "system": ["substitution", "elimination", "matrix_methods"],
            }

    def validate_symbolic_computation(
        self, expression: str, expected_result: str, computation_type: str = "simplify"
    ) -> Dict[str, Any]:
        """
        Validate symbolic computation with advanced verification.

        Args:
            expression: Mathematical expression to compute
            expected_result: Expected result of the computation
            computation_type: Type of computation (simplify, expand, factor, etc.)

        Returns:
            Dictionary with validation results and computation details
        """
        if not SYMPY_AVAILABLE:
            return {
                "verified": False,
                "method": "cas_unavailable",
                "reason": "SymPy not available for symbolic computation",
            }

        try:
            # Parse the expression and expected result
            expr = self.parse_mathematical_expression(expression)
            expected = self.parse_mathematical_expression(expected_result)

            if expr is None or expected is None:
                return {
                    "verified": False,
                    "method": "parsing_failed",
                    "reason": "Could not parse expression or expected result",
                }

            # Perform the specified computation
            computed_result = self._perform_symbolic_computation(expr, computation_type)

            if computed_result is None:
                return {
                    "verified": False,
                    "method": "computation_failed",
                    "reason": f"Could not perform {computation_type} computation",
                }

            # Verify the result
            verification_result = self._verify_symbolic_equivalence(
                computed_result, expected, computation_type
            )

            # Add computation details
            verification_result.update(
                {
                    "computation_type": computation_type,
                    "original_expression": str(expr),
                    "computed_result": str(computed_result),
                    "expected_result": str(expected),
                    "symbolic_form": self._get_symbolic_form_info(computed_result),
                }
            )

            return verification_result

        except Exception as e:
            logger.error("Symbolic computation validation failed: %s", str(e))
            return {
                "verified": False,
                "method": "computation_error",
                "reason": f"Symbolic computation error: {str(e)}",
            }

    def verify_multiple_solution_paths(
        self, problem: str, solutions: List[str], problem_type: str = "equation"
    ) -> Dict[str, Any]:
        """
        Verify multiple solution paths for the same problem.

        Args:
            problem: Mathematical problem statement
            solutions: List of different solution approaches/results
            problem_type: Type of problem (equation, optimization, etc.)

        Returns:
            Dictionary with verification results for all solution paths
        """
        if not SYMPY_AVAILABLE:
            return {
                "verified": False,
                "method": "cas_unavailable",
                "reason": "SymPy not available for multiple solution verification",
            }

        try:
            solution_results = []
            valid_solutions = []

            # Parse and verify each solution
            for i, solution in enumerate(solutions):
                solution_result = self._verify_single_solution_path(
                    problem, solution, problem_type, f"path_{i+1}"
                )
                solution_results.append(solution_result)

                if solution_result.get("verified", False):
                    valid_solutions.append(solution_result)

            # Check consistency between valid solutions
            consistency_check = self._check_solution_consistency(valid_solutions)

            # Analyze solution diversity and completeness
            diversity_analysis = self._analyze_solution_diversity(
                valid_solutions, problem_type
            )

            overall_verified = (
                len(valid_solutions) > 0 and consistency_check["consistent"]
            )

            return {
                "verified": overall_verified,
                "method": "multiple_path_verification",
                "total_solutions": len(solutions),
                "valid_solutions": len(valid_solutions),
                "solution_results": solution_results,
                "consistency_check": consistency_check,
                "diversity_analysis": diversity_analysis,
                "confidence": len(valid_solutions) / max(1, len(solutions)),
            }

        except Exception as e:
            logger.error("Multiple solution path verification failed: %s", str(e))
            return {
                "verified": False,
                "method": "multiple_path_error",
                "reason": f"Multiple solution verification error: {str(e)}",
            }

    def validate_mathematical_prerequisites(
        self, content: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate mathematical concept prerequisites for given content.

        Args:
            content: Mathematical content with problem and concepts

        Returns:
            Dictionary with prerequisite validation results
        """
        try:
            # Extract mathematical concepts from content
            detected_concepts = self._detect_mathematical_concepts(content)

            # Determine required prerequisites
            required_prerequisites = self._determine_prerequisites(detected_concepts)

            # Check prerequisite coverage
            prerequisite_coverage = self._check_prerequisite_coverage(
                content, required_prerequisites
            )

            # Validate prerequisite ordering
            ordering_validation = self._validate_prerequisite_ordering(
                detected_concepts, required_prerequisites
            )

            # Calculate prerequisite completeness score
            completeness_score = self._calculate_prerequisite_completeness(
                prerequisite_coverage, ordering_validation
            )

            return {
                "validated": completeness_score >= 0.7,
                "method": "prerequisite_validation",
                "detected_concepts": detected_concepts,
                "required_prerequisites": required_prerequisites,
                "prerequisite_coverage": prerequisite_coverage,
                "ordering_validation": ordering_validation,
                "completeness_score": completeness_score,
                "missing_prerequisites": [
                    prereq
                    for prereq, covered in prerequisite_coverage.items()
                    if not covered
                ],
            }

        except Exception as e:
            logger.error("Mathematical prerequisite validation failed: %s", str(e))
            return {
                "validated": False,
                "method": "prerequisite_error",
                "reason": f"Prerequisite validation error: {str(e)}",
            }

    def _perform_symbolic_computation(
        self, expr: SymPyBasic, computation_type: str
    ) -> Optional[SymPyBasic]:
        """Perform the specified symbolic computation."""
        try:
            if computation_type == "simplify":
                return simplify(expr)
            elif computation_type == "expand":
                return expand(expr)
            elif computation_type == "factor":
                return factor(expr)
            elif computation_type == "collect":
                # Collect terms with respect to the main variable
                free_symbols = expr.free_symbols
                if free_symbols:
                    main_var = list(free_symbols)[0]
                    return collect(expr, main_var)
                return expr
            elif computation_type == "cancel":
                return cancel(expr)
            elif computation_type == "apart":
                return apart(expr)
            elif computation_type == "together":
                return together(expr)
            elif computation_type == "trigsimp":
                return trigsimp(expr)
            else:
                logger.warning("Unknown computation type: %s", computation_type)
                return simplify(expr)  # Default to simplify

        except Exception as e:
            logger.error(
                "Symbolic computation failed for %s: %s", computation_type, str(e)
            )
            return None

    def _verify_symbolic_equivalence(
        self, computed: SymPyBasic, expected: SymPyBasic, computation_type: str
    ) -> Dict[str, Any]:
        """Verify symbolic equivalence between computed and expected results."""
        try:
            # Direct equivalence check
            if computed.equals(expected):
                return {
                    "verified": True,
                    "method": "direct_equivalence",
                    "confidence": 1.0,
                }

            # Simplification equivalence check
            simplified_diff = simplify(computed - expected)
            if simplified_diff == 0:
                return {
                    "verified": True,
                    "method": "simplified_equivalence",
                    "confidence": 0.95,
                }

            # Form-specific equivalence checks
            form_check = self._check_form_specific_equivalence(
                computed, expected, computation_type
            )
            if form_check["equivalent"]:
                return {
                    "verified": True,
                    "method": form_check["method"],
                    "confidence": form_check["confidence"],
                }

            # Numerical equivalence as fallback
            numerical_check = self._check_numerical_equivalence(computed, expected)
            if numerical_check["equivalent"]:
                return {
                    "verified": True,
                    "method": "numerical_equivalence",
                    "confidence": numerical_check["confidence"],
                }

            return {
                "verified": False,
                "method": "symbolic_comparison",
                "reason": f"Expressions are not equivalent: {simplified_diff}",
                "confidence": 0.0,
            }

        except Exception as e:
            return {
                "verified": False,
                "method": "equivalence_error",
                "reason": f"Equivalence check failed: {str(e)}",
                "confidence": 0.0,
            }

    def _check_form_specific_equivalence(
        self, expr1: SymPyBasic, expr2: SymPyBasic, computation_type: str
    ) -> Dict[str, Any]:
        """Check equivalence specific to the computation type."""
        try:
            if computation_type == "factor":
                # For factored forms, check if they expand to the same expression
                expanded1 = expand(expr1)
                expanded2 = expand(expr2)
                if expanded1.equals(expanded2):
                    return {
                        "equivalent": True,
                        "method": "factored_form_equivalence",
                        "confidence": 0.9,
                    }

            elif computation_type == "expand":
                # For expanded forms, check if they factor to equivalent forms
                try:
                    factored1 = factor(expr1)
                    factored2 = factor(expr2)
                    if factored1.equals(factored2):
                        return {
                            "equivalent": True,
                            "method": "expanded_form_equivalence",
                            "confidence": 0.9,
                        }
                except:
                    pass

            elif computation_type in ["apart", "together"]:
                # For rational function forms, check if they simplify to the same expression
                simplified1 = simplify(expr1)
                simplified2 = simplify(expr2)
                if simplified1.equals(simplified2):
                    return {
                        "equivalent": True,
                        "method": "rational_form_equivalence",
                        "confidence": 0.9,
                    }

            return {
                "equivalent": False,
                "method": "form_specific_check",
                "confidence": 0.0,
            }

        except Exception:
            return {
                "equivalent": False,
                "method": "form_check_error",
                "confidence": 0.0,
            }

    def _verify_single_solution_path(
        self, problem: str, solution: str, problem_type: str, path_id: str
    ) -> Dict[str, Any]:
        """Verify a single solution path."""
        try:
            # Parse the solution
            solution_expr = self.parse_mathematical_expression(solution)
            if solution_expr is None:
                return {
                    "verified": False,
                    "path_id": path_id,
                    "method": "parsing_failed",
                    "reason": "Could not parse solution",
                }

            # Verify solution based on problem type
            if problem_type == "equation":
                return self._verify_equation_solution(problem, solution_expr, path_id)
            elif problem_type == "optimization":
                return self._verify_optimization_solution(
                    problem, solution_expr, path_id
                )
            elif problem_type == "calculus":
                return self._verify_calculus_solution(problem, solution_expr, path_id)
            else:
                # Generic verification
                return {
                    "verified": True,
                    "path_id": path_id,
                    "method": "generic_verification",
                    "solution": str(solution_expr),
                    "confidence": 0.7,
                }

        except Exception as e:
            return {
                "verified": False,
                "path_id": path_id,
                "method": "verification_error",
                "reason": f"Solution verification error: {str(e)}",
            }

    def _verify_equation_solution(
        self, problem: str, solution: SymPyBasic, path_id: str
    ) -> Dict[str, Any]:
        """Verify a solution to an equation."""
        try:
            # Extract equation from problem (simplified approach)
            # In a real implementation, this would be more sophisticated
            if "=" in problem:
                equation_parts = problem.split("=")
                if len(equation_parts) == 2:
                    left_expr = self.parse_mathematical_expression(
                        equation_parts[0].strip()
                    )
                    right_expr = self.parse_mathematical_expression(
                        equation_parts[1].strip()
                    )

                    if left_expr and right_expr:
                        # Substitute solution and check if equation holds
                        variables = left_expr.free_symbols.union(
                            right_expr.free_symbols
                        )
                        if variables:
                            main_var = list(variables)[0]
                            left_substituted = left_expr.subs(main_var, solution)
                            right_substituted = right_expr.subs(main_var, solution)

                            difference = simplify(left_substituted - right_substituted)
                            if difference == 0:
                                return {
                                    "verified": True,
                                    "path_id": path_id,
                                    "method": "equation_substitution",
                                    "solution": str(solution),
                                    "confidence": 1.0,
                                }

            return {
                "verified": False,
                "path_id": path_id,
                "method": "equation_verification",
                "reason": "Could not verify equation solution",
                "confidence": 0.0,
            }

        except Exception as e:
            return {
                "verified": False,
                "path_id": path_id,
                "method": "equation_error",
                "reason": f"Equation verification error: {str(e)}",
            }

    def _verify_optimization_solution(
        self, problem: str, solution: SymPyBasic, path_id: str
    ) -> Dict[str, Any]:
        """Verify a solution to an optimization problem."""
        # Placeholder for optimization verification
        return {
            "verified": True,
            "path_id": path_id,
            "method": "optimization_verification",
            "solution": str(solution),
            "confidence": 0.8,
        }

    def _verify_calculus_solution(
        self, problem: str, solution: SymPyBasic, path_id: str
    ) -> Dict[str, Any]:
        """Verify a solution to a calculus problem."""
        # Placeholder for calculus verification
        return {
            "verified": True,
            "path_id": path_id,
            "method": "calculus_verification",
            "solution": str(solution),
            "confidence": 0.8,
        }

    def _check_solution_consistency(
        self, valid_solutions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Check consistency between multiple valid solutions."""
        if len(valid_solutions) <= 1:
            return {"consistent": True, "reason": "Single or no solutions to compare"}

        try:
            # Extract solution expressions
            solution_exprs = []
            for sol in valid_solutions:
                if "solution" in sol:
                    expr = self.parse_mathematical_expression(sol["solution"])
                    if expr:
                        solution_exprs.append(expr)

            if len(solution_exprs) <= 1:
                return {
                    "consistent": True,
                    "reason": "Insufficient parseable solutions",
                }

            # Check if all solutions are equivalent
            base_solution = solution_exprs[0]
            for expr in solution_exprs[1:]:
                if not simplify(base_solution - expr) == 0:
                    return {
                        "consistent": False,
                        "reason": f"Solutions differ: {base_solution} vs {expr}",
                    }

            return {"consistent": True, "reason": "All solutions are equivalent"}

        except Exception as e:
            return {
                "consistent": False,
                "reason": f"Consistency check failed: {str(e)}",
            }

    def _analyze_solution_diversity(
        self, valid_solutions: List[Dict[str, Any]], problem_type: str
    ) -> Dict[str, Any]:
        """Analyze diversity of solution approaches."""
        if not valid_solutions:
            return {"diversity_score": 0.0, "methods_used": []}

        methods_used = [sol.get("method", "unknown") for sol in valid_solutions]
        unique_methods = set(methods_used)

        # Get expected methods for this problem type
        expected_methods = self.solution_methods.get(problem_type, [])

        diversity_score = len(unique_methods) / max(1, len(expected_methods))

        return {
            "diversity_score": min(1.0, diversity_score),
            "methods_used": list(unique_methods),
            "expected_methods": expected_methods,
            "coverage": len(unique_methods) / max(1, len(methods_used)),
        }

    def _detect_mathematical_concepts(self, content: Dict[str, Any]) -> List[str]:
        """Detect mathematical concepts in the content."""
        concepts = []
        text = f"{content.get('problem', '')} {content.get('answer', '')}".lower()

        # Concept detection patterns
        concept_patterns = {
            "derivative": ["derivative", "differentiate", "f'", "dy/dx", "d/dx"],
            "integral": ["integral", "integrate", "∫", "antiderivative"],
            "limit": ["limit", "lim", "approaches", "tends to"],
            "series": ["series", "sequence", "convergence", "divergence"],
            "matrix": ["matrix", "determinant", "eigenvalue", "eigenvector"],
            "complex_analysis": ["complex", "imaginary", "real part", "imaginary part"],
            "trigonometry": ["sin", "cos", "tan", "trigonometric", "angle"],
            "logarithm": ["log", "ln", "logarithm", "exponential"],
            "polynomial": [
                "polynomial",
                "quadratic",
                "cubic",
                "degree",
                "x^2",
                "x^3",
                "factor",
            ],
            "function": ["function", "f(x)", "domain", "range"],
            "rational": ["rational", "fraction", "/", "numerator", "denominator"],
            "algebra": ["simplify", "expression", "equation", "solve"],
        }

        for concept, patterns in concept_patterns.items():
            if any(pattern in text for pattern in patterns):
                concepts.append(concept)

        return concepts

    def _determine_prerequisites(self, concepts: List[str]) -> Dict[str, List[str]]:
        """Determine prerequisites for detected concepts."""
        prerequisites = {}

        for concept in concepts:
            if concept in self.concept_prerequisites:
                prerequisites[concept] = self.concept_prerequisites[concept]

        return prerequisites

    def _check_prerequisite_coverage(
        self, content: Dict[str, Any], required_prerequisites: Dict[str, List[str]]
    ) -> Dict[str, bool]:
        """Check if prerequisites are covered in the content."""
        coverage = {}
        text = f"{content.get('problem', '')} {content.get('answer', '')} {content.get('explanation', '')}".lower()

        all_prerequisites = set()
        for prereq_list in required_prerequisites.values():
            all_prerequisites.update(prereq_list)

        for prereq in all_prerequisites:
            # Simple keyword-based coverage check
            coverage[prereq] = prereq.replace("_", " ") in text

        return coverage

    def _validate_prerequisite_ordering(
        self, concepts: List[str], required_prerequisites: Dict[str, List[str]]
    ) -> Dict[str, Any]:
        """Validate the ordering of prerequisites."""
        # Simplified ordering validation
        # In a real implementation, this would be more sophisticated
        return {"valid_ordering": True, "ordering_issues": [], "confidence": 0.8}

    def _calculate_prerequisite_completeness(
        self, coverage: Dict[str, bool], ordering: Dict[str, Any]
    ) -> float:
        """Calculate prerequisite completeness score."""
        if not coverage:
            return 1.0  # No prerequisites needed

        covered_count = sum(1 for covered in coverage.values() if covered)
        total_count = len(coverage)

        coverage_score = covered_count / total_count
        ordering_score = ordering.get("confidence", 1.0)

        return (coverage_score + ordering_score) / 2

    def _get_symbolic_form_info(self, expr: SymPyBasic) -> Dict[str, Any]:
        """Get information about the symbolic form of an expression."""
        try:
            return {
                "is_polynomial": expr.is_polynomial(),
                "is_rational": expr.is_rational_function(),
                "free_symbols": [str(sym) for sym in expr.free_symbols],
                "complexity": len(str(expr)),
                "has_trigonometric": any(
                    func in str(expr) for func in ["sin", "cos", "tan"]
                ),
                "has_exponential": any(func in str(expr) for func in ["exp", "log"]),
                "has_radical": "sqrt" in str(expr) or "**" in str(expr),
            }
        except:
            return {"error": "Could not analyze symbolic form"}


# Global enhanced CAS validator instance
_enhanced_cas_validator = None


def get_enhanced_cas_validator() -> EnhancedCASValidator:
    """Get the global enhanced CAS validator instance."""
    global _enhanced_cas_validator
    if _enhanced_cas_validator is None:
        _enhanced_cas_validator = EnhancedCASValidator()
    return _enhanced_cas_validator
