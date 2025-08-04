"""
Mathematics domain validator for STREAM content validation.

This module provides comprehensive validation for mathematical content including
CAS verification, notation validation, proof checking, and difficulty assessment.
"""

# Standard Library
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

# SynThesisAI Modules
from core.checker.cas_validator import CASValidator, verify_with_cas
from core.validation.cas_enhanced import (
    EnhancedCASValidator,
    get_enhanced_cas_validator,
)

from ..base import (
    DomainValidator,
    QualityMetrics,
    SubValidationResult,
    ValidationResult,
)
from ..config import ValidationConfig
from ..exceptions import DomainValidationError

logger = logging.getLogger(__name__)


class MathNotationValidator:
    """Validator for mathematical notation and formatting."""

    def __init__(self):
        """Initialize notation validator."""
        self.latex_patterns = {
            "fractions": r"\\frac\{[^}]+\}\{[^}]+\}",
            "exponents": r"\^?\{?[^}]*\}?",
            "subscripts": r"_\{?[^}]*\}?",
            "square_roots": r"\\sqrt\{[^}]+\}",
            "integrals": r"\\int[^\\]*",
            "summations": r"\\sum[^\\]*",
            "greek_letters": r"\\[a-zA-Z]+",
            "matrices": r"\\begin\{[^}]*matrix[^}]*\}.*?\\end\{[^}]*matrix[^}]*\}",
        }

        self.common_errors = {
            "missing_braces": r"[_^][a-zA-Z0-9]{2,}(?!\})",
            "unmatched_braces": r"\{[^}]*$|^[^{]*\}",
            "invalid_commands": r"\\[a-zA-Z]+(?![a-zA-Z])",
            "spacing_issues": r"\s+(?=[_^])|(?<=[_^])\s+",
        }

    def validate_notation(self, mathematical_content: str) -> SubValidationResult:
        """
        Validate mathematical notation in content.

        Args:
            mathematical_content: Content containing mathematical notation

        Returns:
            SubValidationResult with notation validation details
        """
        issues = []
        notation_score = 1.0

        try:
            # Check for common notation errors
            for error_type, pattern in self.common_errors.items():
                matches = re.findall(pattern, mathematical_content)
                if matches:
                    issues.append(f"{error_type}: {len(matches)} instances found")
                    notation_score -= 0.1 * len(matches)

            # Validate LaTeX formatting if present
            if "\\" in mathematical_content:
                latex_issues = self._validate_latex_formatting(mathematical_content)
                issues.extend(latex_issues)
                notation_score -= 0.05 * len(latex_issues)

            # Check for proper mathematical symbols
            symbol_issues = self._validate_mathematical_symbols(mathematical_content)
            issues.extend(symbol_issues)
            notation_score -= 0.03 * len(symbol_issues)

            notation_score = max(0.0, min(1.0, notation_score))

            return SubValidationResult(
                subdomain="notation",
                is_valid=notation_score >= 0.8,
                details={
                    "notation_score": notation_score,
                    "issues": issues,
                    "latex_detected": "\\" in mathematical_content,
                    "total_issues": len(issues),
                },
                confidence_score=0.9 if notation_score >= 0.8 else 0.6,
            )

        except Exception as e:
            logger.error("Notation validation failed: %s", str(e))
            return SubValidationResult(
                subdomain="notation",
                is_valid=False,
                details={"error": str(e)},
                confidence_score=0.0,
                error_message=f"Notation validation error: {str(e)}",
            )

    def _validate_latex_formatting(self, content: str) -> List[str]:
        """Validate LaTeX formatting in mathematical content."""
        issues = []

        # Check for unmatched braces
        brace_count = content.count("{") - content.count("}")
        if brace_count != 0:
            issues.append(
                f"Unmatched braces: {abs(brace_count)} {'opening' if brace_count > 0 else 'closing'}"
            )

        # Check for proper command formatting
        commands = re.findall(r"\\([a-zA-Z]+)", content)
        valid_commands = {
            "frac",
            "sqrt",
            "int",
            "sum",
            "prod",
            "lim",
            "sin",
            "cos",
            "tan",
            "log",
            "ln",
            "exp",
            "alpha",
            "beta",
            "gamma",
            "delta",
            "theta",
            "pi",
            "sigma",
            "omega",
            "infty",
            "partial",
            "nabla",
        }

        for command in commands:
            if command not in valid_commands and not command.startswith("math"):
                issues.append(f"Unknown LaTeX command: \\{command}")

        return issues

    def _validate_mathematical_symbols(self, content: str) -> List[str]:
        """Validate proper use of mathematical symbols."""
        issues = []

        # Check for proper equation formatting
        if "=" in content:
            equations = content.split("=")
            if len(equations) > 2:
                # Multiple equals signs should be properly formatted
                if not re.search(r"\\begin\{align\}|\\begin\{eqnarray\}", content):
                    issues.append("Multiple equations should use proper alignment")

        # Check for proper fraction notation
        if "/" in content and not re.search(r"\\frac", content):
            # Suggest using \frac for complex fractions
            fraction_matches = re.findall(r"[a-zA-Z0-9\(\)]+/[a-zA-Z0-9\(\)]+", content)
            if len(fraction_matches) > 2:
                issues.append("Consider using \\frac{}{} for complex fractions")

        return issues


class ProofValidator:
    """Validator for mathematical proofs and logical reasoning."""

    def __init__(self):
        """Initialize proof validator."""
        self.proof_keywords = {
            "direct": [
                "assume",
                "given",
                "let",
                "suppose",
                "therefore",
                "thus",
                "hence",
            ],
            "contradiction": ["contradiction", "assume not", "suppose not", "absurd"],
            "induction": [
                "induction",
                "base case",
                "inductive step",
                "inductive hypothesis",
            ],
            "construction": ["construct", "define", "build", "create"],
        }

        self.logical_connectors = [
            "if",
            "then",
            "and",
            "or",
            "not",
            "implies",
            "iff",
            "because",
            "since",
        ]

    def validate_proof(
        self, proof_content: str, theorem: str = ""
    ) -> SubValidationResult:
        """
        Validate mathematical proof structure and logic.

        Args:
            proof_content: The proof text to validate
            theorem: The theorem being proved (optional)

        Returns:
            SubValidationResult with proof validation details
        """
        try:
            proof_analysis = self._analyze_proof_structure(proof_content)
            logical_flow = self._check_logical_flow(proof_content)
            completeness = self._assess_proof_completeness(proof_content, theorem)

            # Calculate overall proof score
            structure_score = proof_analysis["score"]
            logic_score = logical_flow["score"]
            completeness_score = completeness["score"]

            overall_score = (
                0.5 * structure_score + 0.4 * logic_score + 0.1 * completeness_score
            )

            return SubValidationResult(
                subdomain="proof",
                is_valid=overall_score >= 0.8,
                details={
                    "overall_score": overall_score,
                    "structure_analysis": proof_analysis,
                    "logical_flow": logical_flow,
                    "completeness": completeness,
                    "proof_type": proof_analysis.get("type", "unknown"),
                },
                confidence_score=0.8 if overall_score >= 0.8 else 0.5,
            )

        except Exception as e:
            logger.error("Proof validation failed: %s", str(e))
            return SubValidationResult(
                subdomain="proof",
                is_valid=False,
                details={"error": str(e)},
                confidence_score=0.0,
                error_message=f"Proof validation error: {str(e)}",
            )

    def _analyze_proof_structure(self, proof_content: str) -> Dict[str, Any]:
        """Analyze the structure of a mathematical proof."""
        proof_lower = proof_content.lower()

        # Identify proof type - check for specific patterns
        proof_type = "direct"  # default

        # Check for contradiction proof
        if any(
            keyword in proof_lower
            for keyword in ["contradiction", "assume not", "suppose not", "absurd"]
        ):
            proof_type = "contradiction"
        # Check for induction proof
        elif any(
            keyword in proof_lower
            for keyword in [
                "induction",
                "base case",
                "inductive step",
                "inductive hypothesis",
            ]
        ):
            proof_type = "induction"
        # Check for construction proof
        elif any(
            keyword in proof_lower
            for keyword in ["construct", "define", "build", "create"]
        ):
            proof_type = "construction"

        # Check for proper proof structure
        has_assumption = any(
            word in proof_lower for word in ["assume", "given", "let", "suppose"]
        )
        has_conclusion = any(
            word in proof_lower for word in ["therefore", "thus", "hence", "qed"]
        )
        has_logical_flow = any(word in proof_lower for word in self.logical_connectors)

        # Calculate structure score
        structure_elements = [has_assumption, has_conclusion, has_logical_flow]
        structure_score = sum(structure_elements) / len(structure_elements)

        return {
            "type": proof_type,
            "has_assumption": has_assumption,
            "has_conclusion": has_conclusion,
            "has_logical_flow": has_logical_flow,
            "score": structure_score,
        }

    def _check_logical_flow(self, proof_content: str) -> Dict[str, Any]:
        """Check the logical flow and reasoning in the proof."""
        sentences = re.split(r"[.!?]+", proof_content)
        logical_connections = 0
        total_sentences = len([s for s in sentences if s.strip()])

        for sentence in sentences:
            if any(
                connector in sentence.lower() for connector in self.logical_connectors
            ):
                logical_connections += 1

        # Calculate logical flow score
        if total_sentences > 0:
            # More reasonable scoring - don't penalize longer proofs as much
            expected_connections = max(
                1, total_sentences // 3
            )  # Expect connection every 3 sentences
            logic_score = min(1.0, logical_connections / expected_connections)
        else:
            logic_score = 0.0

        return {
            "logical_connections": logical_connections,
            "total_sentences": total_sentences,
            "score": logic_score,
        }

    def _assess_proof_completeness(
        self, proof_content: str, theorem: str
    ) -> Dict[str, Any]:
        """Assess the completeness of the proof."""
        # Basic completeness checks
        has_clear_start = any(
            word in proof_content.lower()[:100]
            for word in ["proof:", "proof.", "to prove"]
        )
        has_clear_end = any(
            word in proof_content.lower()[-100:]
            for word in ["qed", "therefore", "thus"]
        )

        # Check if key theorem elements are addressed
        completeness_score = 0.5  # base score

        if has_clear_start:
            completeness_score += 0.2
        if has_clear_end:
            completeness_score += 0.2

        # If theorem is provided, check if its elements are addressed
        if theorem:
            theorem_words = set(re.findall(r"\b\w+\b", theorem.lower()))
            proof_words = set(re.findall(r"\b\w+\b", proof_content.lower()))

            # Calculate overlap
            overlap = len(theorem_words.intersection(proof_words))
            if len(theorem_words) > 0:
                overlap_ratio = overlap / len(theorem_words)
                completeness_score += 0.1 * overlap_ratio

        return {
            "has_clear_start": has_clear_start,
            "has_clear_end": has_clear_end,
            "score": min(1.0, completeness_score),
        }


class DifficultyAssessor:
    """Assessor for mathematical content difficulty level."""

    def __init__(self):
        """Initialize difficulty assessor."""
        self.difficulty_indicators = {
            "basic": {
                "operations": ["addition", "subtraction", "multiplication", "division"],
                "concepts": ["number", "counting", "basic", "simple"],
                "symbols": ["+", "-", "*", "/", "="],
            },
            "intermediate": {
                "operations": ["exponent", "root", "logarithm", "trigonometry"],
                "concepts": ["algebra", "equation", "function", "graph"],
                "symbols": ["^", "sqrt", "log", "sin", "cos", "tan"],
            },
            "advanced": {
                "operations": [
                    "derivative",
                    "integral",
                    "limit",
                    "series",
                    "differentiate",
                    "find the derivative",
                ],
                "concepts": [
                    "calculus",
                    "analysis",
                    "proof",
                    "theorem",
                    "differential",
                ],
                "symbols": ["∫", "∂", "∑", "∏", "∞", "∇", "f'", "dx", "dy"],
            },
            "expert": {
                "operations": ["topology", "manifold", "homomorphism", "isomorphism"],
                "concepts": ["abstract", "category", "field", "ring", "group"],
                "symbols": ["⊕", "⊗", "∈", "∀", "∃", "⟨", "⟩"],
            },
        }

    def assess_difficulty(self, content: Dict[str, Any]) -> SubValidationResult:
        """
        Assess the difficulty level of mathematical content.

        Args:
            content: Mathematical content to assess

        Returns:
            SubValidationResult with difficulty assessment
        """
        try:
            problem_text = content.get("problem", "")
            answer_text = content.get("answer", "")
            full_text = f"{problem_text} {answer_text}".lower()

            # Calculate difficulty scores for each level
            level_scores = {}
            for level, indicators in self.difficulty_indicators.items():
                score = self._calculate_level_score(full_text, indicators)
                level_scores[level] = score

            # Determine primary difficulty level
            primary_level = max(level_scores, key=level_scores.get)
            confidence = level_scores[primary_level]

            # Check for prerequisite alignment
            prerequisites = self._identify_prerequisites(full_text, primary_level)

            # Assess appropriateness
            appropriateness = self._assess_level_appropriateness(content, primary_level)

            return SubValidationResult(
                subdomain="difficulty",
                is_valid=appropriateness["is_appropriate"],
                details={
                    "primary_level": primary_level,
                    "level_scores": level_scores,
                    "prerequisites": prerequisites,
                    "appropriateness": appropriateness,
                    "confidence": confidence,
                },
                confidence_score=confidence,
            )

        except Exception as e:
            logger.error("Difficulty assessment failed: %s", str(e))
            return SubValidationResult(
                subdomain="difficulty",
                is_valid=False,
                details={"error": str(e)},
                confidence_score=0.0,
                error_message=f"Difficulty assessment error: {str(e)}",
            )

    def _calculate_level_score(
        self, text: str, indicators: Dict[str, List[str]]
    ) -> float:
        """Calculate difficulty score for a specific level."""
        total_matches = 0
        total_possible = 0

        for category, terms in indicators.items():
            matches = sum(1 for term in terms if term in text)
            # Give more weight to operations and concepts vs symbols
            weight = 2.0 if category in ["operations", "concepts"] else 1.0
            total_matches += matches * weight
            total_possible += len(terms) * weight

        return total_matches / max(1, total_possible)

    def _identify_prerequisites(self, text: str, level: str) -> List[str]:
        """Identify mathematical prerequisites for the content."""
        prerequisites = []

        # Define prerequisite chains
        prerequisite_chain = {
            "basic": [],
            "intermediate": ["basic arithmetic", "number operations"],
            "advanced": ["algebra", "functions", "trigonometry"],
            "expert": ["calculus", "linear algebra", "abstract algebra"],
        }

        # Add level-specific prerequisites
        if level in prerequisite_chain:
            prerequisites.extend(prerequisite_chain[level])

        # Add content-specific prerequisites based on detected concepts
        if "derivative" in text or "integral" in text:
            prerequisites.extend(["limits", "functions"])
        if "matrix" in text or "vector" in text:
            prerequisites.append("linear algebra")
        if "proof" in text or "theorem" in text:
            prerequisites.append("mathematical reasoning")

        return list(set(prerequisites))  # Remove duplicates

    def _assess_level_appropriateness(
        self, content: Dict[str, Any], level: str
    ) -> Dict[str, Any]:
        """Assess if the difficulty level is appropriate for the content."""
        # Check if stated difficulty matches assessed difficulty
        stated_difficulty = content.get("difficulty", "").lower()

        is_appropriate = True
        issues = []

        if stated_difficulty and stated_difficulty != level:
            is_appropriate = False
            issues.append(
                f"Stated difficulty '{stated_difficulty}' doesn't match assessed '{level}'"
            )

        # Check for complexity mismatches
        problem_complexity = len(content.get("problem", "").split())
        answer_complexity = len(str(content.get("answer", "")).split())

        if level == "basic" and (problem_complexity > 50 or answer_complexity > 10):
            issues.append("Content seems too complex for basic level")
        elif level == "expert" and (problem_complexity < 20 or answer_complexity < 5):
            issues.append("Content seems too simple for expert level")

        return {
            "is_appropriate": is_appropriate and len(issues) == 0,
            "issues": issues,
            "complexity_metrics": {
                "problem_length": problem_complexity,
                "answer_length": answer_complexity,
            },
        }


class MathematicsValidator(DomainValidator):
    """Enhanced mathematics validator with CAS integration and comprehensive validation."""

    def __init__(self, domain: str, config: ValidationConfig):
        """
        Initialize mathematics validator.

        Args:
            domain: Should be "mathematics"
            config: Validation configuration for mathematics domain
        """
        super().__init__(domain, config)

        # Initialize specialized validators
        self.cas_validator = CASValidator()
        self.enhanced_cas_validator = get_enhanced_cas_validator()
        self.notation_validator = MathNotationValidator()
        self.proof_validator = ProofValidator()
        self.difficulty_assessor = DifficultyAssessor()

        logger.info(
            "Initialized MathematicsValidator with CAS support: %s",
            self.cas_validator.is_available(),
        )

    def validate_content(self, content: Dict[str, Any]) -> ValidationResult:
        """
        Comprehensive mathematics content validation.

        Args:
            content: Mathematical content to validate

        Returns:
            ValidationResult with comprehensive validation details
        """
        validation_details = {}

        try:
            # CAS verification for mathematical accuracy
            if "problem" in content and "answer" in content:
                cas_result = self._perform_cas_validation(content)
                validation_details["cas_verification"] = cas_result

                # Enhanced CAS validation for advanced verification
                enhanced_cas_result = self._perform_enhanced_cas_validation(content)
                validation_details["enhanced_cas_verification"] = enhanced_cas_result

            # Mathematical notation validation
            if "problem" in content:
                notation_result = self.notation_validator.validate_notation(
                    content["problem"]
                )
                validation_details["notation_validation"] = notation_result

            # Proof validation for proof-based problems
            if "proof" in content:
                proof_result = self.proof_validator.validate_proof(
                    content["proof"], content.get("theorem", "")
                )
                validation_details["proof_validation"] = proof_result

            # Difficulty level validation and assessment
            difficulty_result = self.difficulty_assessor.assess_difficulty(content)
            validation_details["difficulty_assessment"] = difficulty_result

            # Calculate quality metrics
            quality_metrics = self.calculate_quality_metrics(
                content, validation_details
            )

            # Determine overall validity
            is_valid = self._determine_overall_validity(validation_details)

            # Calculate confidence score
            confidence_score = self.calculate_confidence(validation_details)

            # Generate feedback
            feedback = self.generate_feedback_from_details(validation_details)

            return ValidationResult(
                domain=self.domain,
                is_valid=is_valid,
                quality_score=quality_metrics.overall_score,
                validation_details=validation_details,
                confidence_score=confidence_score,
                feedback=feedback,
                quality_metrics=quality_metrics,
            )

        except Exception as e:
            logger.error("Mathematics validation failed: %s", str(e))
            raise DomainValidationError(
                self.domain, f"Mathematics validation failed: {str(e)}"
            ) from e

    def _perform_cas_validation(self, content: Dict[str, Any]) -> SubValidationResult:
        """Perform CAS validation using the existing CAS validator."""
        try:
            problem = content["problem"]
            answer = content["answer"]

            # Use the existing CAS verification function
            cas_result = verify_with_cas(problem, answer, answer, "auto")

            return SubValidationResult(
                subdomain="cas",
                is_valid=cas_result.get("verified", False),
                details=cas_result,
                confidence_score=cas_result.get("confidence", 0.0),
            )

        except Exception as e:
            logger.error("CAS validation failed: %s", str(e))
            return SubValidationResult(
                subdomain="cas",
                is_valid=False,
                details={"error": str(e)},
                confidence_score=0.0,
                error_message=f"CAS validation error: {str(e)}",
            )

    def _perform_enhanced_cas_validation(
        self, content: Dict[str, Any]
    ) -> SubValidationResult:
        """Perform enhanced CAS validation with advanced mathematical verification."""
        try:
            problem = content["problem"]
            answer = content["answer"]

            # Perform symbolic computation validation if applicable
            symbolic_result = None
            if "computation_type" in content:
                symbolic_result = (
                    self.enhanced_cas_validator.validate_symbolic_computation(
                        expression=answer,
                        expected_result=answer,
                        computation_type=content["computation_type"],
                    )
                )

            # Perform multiple solution path verification if alternative solutions provided
            multiple_paths_result = None
            if "alternative_solutions" in content:
                solutions = [answer] + content["alternative_solutions"]
                multiple_paths_result = (
                    self.enhanced_cas_validator.verify_multiple_solution_paths(
                        problem=problem,
                        solutions=solutions,
                        problem_type=content.get("problem_type", "equation"),
                    )
                )

            # Perform prerequisite validation
            prerequisite_result = (
                self.enhanced_cas_validator.validate_mathematical_prerequisites(content)
            )

            # Aggregate results
            enhanced_details = {
                "symbolic_computation": symbolic_result,
                "multiple_paths": multiple_paths_result,
                "prerequisites": prerequisite_result,
            }

            # Determine overall validity
            is_valid = True
            confidence_scores = []

            if symbolic_result:
                is_valid = is_valid and symbolic_result.get("verified", True)
                confidence_scores.append(symbolic_result.get("confidence", 0.8))

            if multiple_paths_result:
                is_valid = is_valid and multiple_paths_result.get("verified", True)
                confidence_scores.append(multiple_paths_result.get("confidence", 0.8))

            if prerequisite_result:
                is_valid = is_valid and prerequisite_result.get("validated", True)
                confidence_scores.append(
                    prerequisite_result.get("completeness_score", 0.8)
                )

            overall_confidence = sum(confidence_scores) / max(1, len(confidence_scores))

            return SubValidationResult(
                subdomain="enhanced_cas",
                is_valid=is_valid,
                details=enhanced_details,
                confidence_score=overall_confidence,
            )

        except Exception as e:
            logger.error("Enhanced CAS validation failed: %s", str(e))
            return SubValidationResult(
                subdomain="enhanced_cas",
                is_valid=False,
                details={"error": str(e)},
                confidence_score=0.0,
                error_message=f"Enhanced CAS validation error: {str(e)}",
            )

    def _determine_overall_validity(self, validation_details: Dict[str, Any]) -> bool:
        """Determine overall validity based on validation results."""
        # Check critical validations
        cas_result = validation_details.get("cas_verification")
        cas_valid = cas_result.is_valid if cas_result else True

        notation_result = validation_details.get("notation_validation")
        notation_valid = notation_result.is_valid if notation_result else True

        # CAS validation is critical for mathematical accuracy
        if not cas_valid and self.cas_validator.is_available():
            return False

        # Notation validation is important but not critical
        if not notation_valid and notation_result:
            # Allow some tolerance for notation issues
            notation_score = notation_result.details.get("notation_score", 0.0)
            if notation_score < 0.6:
                return False

        # Check other validations
        proof_result = validation_details.get("proof_validation")
        proof_valid = proof_result.is_valid if proof_result else True

        difficulty_result = validation_details.get("difficulty_assessment")
        difficulty_valid = difficulty_result.is_valid if difficulty_result else True

        # Proof validation is critical if proof is present
        if proof_result and not proof_valid:
            return False

        return True

    def calculate_quality_score(self, content: Dict[str, Any]) -> float:
        """
        Calculate domain-specific quality score for mathematics.

        Args:
            content: Mathematical content to assess

        Returns:
            Quality score between 0.0 and 1.0
        """
        # Base score from parent class
        base_score = 0.7

        # Adjust based on mathematical complexity and correctness
        if "answer" in content and content["answer"]:
            base_score += 0.1  # Has answer

        if "problem" in content and len(content["problem"]) > 20:
            base_score += 0.1  # Substantial problem

        if "proof" in content:
            base_score += 0.1  # Includes proof

        return min(1.0, base_score)

    def generate_feedback(self, validation_result: ValidationResult) -> List[str]:
        """
        Generate mathematics-specific improvement feedback.

        Args:
            validation_result: Result of validation to generate feedback for

        Returns:
            List of feedback messages for content improvement
        """
        return self.generate_feedback_from_details(validation_result.validation_details)

    def generate_feedback_from_details(
        self, validation_details: Dict[str, Any]
    ) -> List[str]:
        """Generate feedback from validation details."""
        feedback = []

        # CAS validation feedback
        cas_result = validation_details.get("cas_verification")
        if cas_result and not cas_result.is_valid:
            cas_reason = cas_result.details.get("reason", "Unknown CAS error")
            feedback.append(f"Mathematical accuracy issue: {cas_reason}")

        # Notation validation feedback
        notation_result = validation_details.get("notation_validation")
        if notation_result and not notation_result.is_valid:
            notation_issues = notation_result.details.get("issues", [])
            if notation_issues:
                feedback.append(f"Notation issues: {'; '.join(notation_issues[:3])}")

        # Proof validation feedback
        proof_result = validation_details.get("proof_validation")
        if proof_result and not proof_result.is_valid:
            feedback.append("Proof structure or logic needs improvement")

        # Difficulty assessment feedback
        difficulty_result = validation_details.get("difficulty_assessment")
        if difficulty_result and not difficulty_result.is_valid:
            difficulty_issues = difficulty_result.details.get(
                "appropriateness", {}
            ).get("issues", [])
            if difficulty_issues:
                feedback.append(
                    f"Difficulty level issues: {'; '.join(difficulty_issues[:2])}"
                )

        return feedback
