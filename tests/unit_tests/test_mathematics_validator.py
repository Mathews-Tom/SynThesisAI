"""
Unit tests for the mathematics domain validator.

This module tests the enhanced mathematics validator including CAS integration,
notation validation, proof checking, and difficulty assessment.
"""

# Standard Library
from unittest.mock import Mock, patch

# Third-Party Library
import pytest

# SynThesisAI Modules
from core.validation.base import SubValidationResult, ValidationResult
from core.validation.config import ValidationConfig
from core.validation.domains.mathematics import (
    DifficultyAssessor,
    MathematicsValidator,
    MathNotationValidator,
    ProofValidator,
)
from core.validation.exceptions import DomainValidationError


class TestMathNotationValidator:
    """Test cases for MathNotationValidator."""

    def test_valid_notation(self):
        """Test validation of correct mathematical notation."""
        validator = MathNotationValidator()

        content = "Solve for x: 2x + 5 = 15"
        result = validator.validate_notation(content)

        assert isinstance(result, SubValidationResult)
        assert result.subdomain == "notation"
        assert result.is_valid is True
        assert result.details["notation_score"] >= 0.8

    def test_latex_notation(self):
        """Test validation of LaTeX mathematical notation."""
        validator = MathNotationValidator()

        content = r"Find the derivative: \frac{d}{dx}(x^2 + 3x + 1)"
        result = validator.validate_notation(content)

        assert result.subdomain == "notation"
        assert result.details["latex_detected"] is True
        # Should be valid LaTeX
        assert result.is_valid is True

    def test_invalid_latex_notation(self):
        """Test validation of invalid LaTeX notation."""
        validator = MathNotationValidator()

        content = r"Invalid LaTeX: \frac{x^2 + 3x + 1"  # Missing closing brace
        result = validator.validate_notation(content)

        assert result.subdomain == "notation"
        assert result.details["latex_detected"] is True
        assert len(result.details["issues"]) > 0
        assert result.is_valid is False

    def test_notation_error_handling(self):
        """Test error handling in notation validation."""
        validator = MathNotationValidator()

        # Test with error in validation (should be handled gracefully)
        with patch.object(
            validator, "_validate_latex_formatting", side_effect=Exception("Test error")
        ):
            result = validator.validate_notation(
                "\\test content"
            )  # LaTeX content to trigger the error

            assert result.is_valid is False
            assert "error" in result.details
            assert result.confidence_score == 0.0


class TestProofValidator:
    """Test cases for ProofValidator."""

    def test_direct_proof_validation(self):
        """Test validation of a direct proof."""
        validator = ProofValidator()

        proof_content = """
        Proof: Let x be an even integer. Then x = 2k for some integer k.
        We need to show that x^2 is even.
        Since x = 2k, we have x^2 = (2k)^2 = 4k^2 = 2(2k^2).
        Since 2k^2 is an integer, x^2 = 2(2k^2) is even.
        Therefore, if x is even, then x^2 is even. QED
        """

        result = validator.validate_proof(proof_content)

        assert isinstance(result, SubValidationResult)
        assert result.subdomain == "proof"
        assert result.is_valid is True
        assert result.details["proof_type"] == "direct"
        assert result.details["structure_analysis"]["has_assumption"] is True
        assert result.details["structure_analysis"]["has_conclusion"] is True

    def test_proof_by_contradiction(self):
        """Test validation of a proof by contradiction."""
        validator = ProofValidator()

        proof_content = """
        Proof by contradiction: Assume that sqrt(2) is rational.
        Then sqrt(2) = p/q where p and q are integers with no common factors.
        Squaring both sides: 2 = p^2/q^2, so 2q^2 = p^2.
        This means p^2 is even, so p is even. Let p = 2r.
        Then 2q^2 = (2r)^2 = 4r^2, so q^2 = 2r^2.
        This means q^2 is even, so q is even.
        But this contradicts our assumption that p and q have no common factors.
        Therefore, sqrt(2) is irrational. QED
        """

        result = validator.validate_proof(proof_content)

        assert result.subdomain == "proof"
        assert result.details["proof_type"] == "contradiction"
        assert result.is_valid is True

    def test_incomplete_proof(self):
        """Test validation of an incomplete proof."""
        validator = ProofValidator()

        proof_content = "Let x be a number. Then x + 1 is bigger."

        result = validator.validate_proof(proof_content)

        assert result.subdomain == "proof"
        assert result.is_valid is False  # Should be invalid due to incompleteness
        assert result.details["overall_score"] < 0.8

    def test_proof_error_handling(self):
        """Test error handling in proof validation."""
        validator = ProofValidator()

        with patch.object(
            validator, "_analyze_proof_structure", side_effect=Exception("Test error")
        ):
            result = validator.validate_proof("test proof")

            assert result.is_valid is False
            assert "error" in result.details
            assert result.confidence_score == 0.0


class TestDifficultyAssessor:
    """Test cases for DifficultyAssessor."""

    def test_basic_difficulty_assessment(self):
        """Test assessment of basic difficulty content."""
        assessor = DifficultyAssessor()

        content = {"problem": "What is 5 + 3?", "answer": "8", "difficulty": "basic"}

        result = assessor.assess_difficulty(content)

        assert isinstance(result, SubValidationResult)
        assert result.subdomain == "difficulty"
        assert result.details["primary_level"] == "basic"
        assert result.is_valid is True

    def test_advanced_difficulty_assessment(self):
        """Test assessment of advanced difficulty content."""
        assessor = DifficultyAssessor()

        content = {
            "problem": "Find the derivative of f(x) = x^3 + 2x^2 - 5x + 1",
            "answer": "f'(x) = 3x^2 + 4x - 5",
            "difficulty": "advanced",
        }

        result = assessor.assess_difficulty(content)

        assert result.subdomain == "difficulty"
        assert result.details["primary_level"] in ["intermediate", "advanced"]
        assert len(result.details["prerequisites"]) > 0

    def test_difficulty_mismatch(self):
        """Test detection of difficulty level mismatch."""
        assessor = DifficultyAssessor()

        content = {
            "problem": "What is 2 + 2?",
            "answer": "4",
            "difficulty": "expert",  # Mismatch: simple problem labeled as expert
        }

        result = assessor.assess_difficulty(content)

        assert result.subdomain == "difficulty"
        assert result.is_valid is False  # Should detect mismatch
        assert len(result.details["appropriateness"]["issues"]) > 0

    def test_prerequisite_identification(self):
        """Test identification of mathematical prerequisites."""
        assessor = DifficultyAssessor()

        content = {
            "problem": "Solve the integral of sin(x) dx",
            "answer": "-cos(x) + C",
        }

        result = assessor.assess_difficulty(content)

        prerequisites = result.details["prerequisites"]
        assert "limits" in prerequisites or "functions" in prerequisites
        assert len(prerequisites) > 0

    def test_difficulty_error_handling(self):
        """Test error handling in difficulty assessment."""
        assessor = DifficultyAssessor()

        with patch.object(
            assessor, "_calculate_level_score", side_effect=Exception("Test error")
        ):
            result = assessor.assess_difficulty({"problem": "test"})

            assert result.is_valid is False
            assert "error" in result.details
            assert result.confidence_score == 0.0


class TestMathematicsValidator:
    """Test cases for MathematicsValidator."""

    @pytest.fixture
    def math_config(self):
        """Create mathematics validation configuration."""
        return ValidationConfig(
            domain="mathematics",
            quality_thresholds={
                "fidelity_score": 0.85,
                "cas_verification": 0.95,
                "notation_accuracy": 0.90,
            },
            timeout_seconds=30,
        )

    @pytest.fixture
    def mock_cas_validator(self):
        """Create mock CAS validator."""
        mock_cas = Mock()
        mock_cas.is_available.return_value = True
        return mock_cas

    def test_validator_initialization(self, math_config):
        """Test mathematics validator initialization."""
        validator = MathematicsValidator("mathematics", math_config)

        assert validator.domain == "mathematics"
        assert validator.config == math_config
        assert hasattr(validator, "cas_validator")
        assert hasattr(validator, "notation_validator")
        assert hasattr(validator, "proof_validator")
        assert hasattr(validator, "difficulty_assessor")

    @patch("core.validation.domains.mathematics.verify_with_cas")
    def test_basic_content_validation(self, mock_verify_cas, math_config):
        """Test basic mathematical content validation."""
        # Mock CAS validation
        mock_verify_cas.return_value = {
            "verified": True,
            "method": "algebraic_equivalence",
            "confidence": 1.0,
        }

        validator = MathematicsValidator("mathematics", math_config)

        content = {"problem": "Solve for x: 2x + 5 = 15", "answer": "x = 5"}

        result = validator.validate_content(content)

        assert isinstance(result, ValidationResult)
        assert result.domain == "mathematics"
        assert result.is_valid is True
        assert "cas_verification" in result.validation_details
        assert "notation_validation" in result.validation_details
        assert "difficulty_assessment" in result.validation_details

    @patch("core.validation.domains.mathematics.verify_with_cas")
    def test_proof_content_validation(self, mock_verify_cas, math_config):
        """Test validation of content with mathematical proof."""
        mock_verify_cas.return_value = {
            "verified": True,
            "method": "algebraic_equivalence",
            "confidence": 1.0,
        }

        validator = MathematicsValidator("mathematics", math_config)

        content = {
            "problem": "Prove that the sum of two even integers is even",
            "answer": "The sum is 2m + 2n = 2(m + n), which is even",
            "proof": "Proof: Let a = 2m and b = 2n be two even integers where m and n are integers. Then a + b = 2m + 2n = 2(m + n). Since m + n is an integer, 2(m + n) is even. Therefore, the sum of two even integers is even. QED",
            "theorem": "Sum of even integers is even",
        }

        result = validator.validate_content(content)

        assert result.domain == "mathematics"
        assert "proof_validation" in result.validation_details
        assert result.is_valid is True

    @patch("core.validation.domains.mathematics.verify_with_cas")
    def test_cas_validation_failure(self, mock_verify_cas, math_config):
        """Test handling of CAS validation failure."""
        mock_verify_cas.return_value = {
            "verified": False,
            "method": "algebraic_comparison",
            "reason": "Expressions are not equivalent",
            "confidence": 0.0,
        }

        validator = MathematicsValidator("mathematics", math_config)

        content = {"problem": "What is 2 + 2?", "answer": "5"}  # Incorrect answer

        result = validator.validate_content(content)

        assert result.domain == "mathematics"
        assert result.is_valid is False  # Should fail due to CAS validation
        assert len(result.feedback) > 0
        assert any("accuracy" in feedback.lower() for feedback in result.feedback)

    def test_quality_score_calculation(self, math_config):
        """Test mathematics-specific quality score calculation."""
        validator = MathematicsValidator("mathematics", math_config)

        # Test with comprehensive content
        content = {
            "problem": "Find the derivative of f(x) = x^3 + 2x^2 - 5x + 1",
            "answer": "f'(x) = 3x^2 + 4x - 5",
            "proof": "Using the power rule for derivatives...",
        }

        quality_score = validator.calculate_quality_score(content)

        assert 0.0 <= quality_score <= 1.0
        assert quality_score > 0.8  # Should be high for comprehensive content

    def test_feedback_generation(self, math_config):
        """Test generation of improvement feedback."""
        validator = MathematicsValidator("mathematics", math_config)

        # Create validation details with issues
        validation_details = {
            "cas_verification": SubValidationResult(
                subdomain="cas",
                is_valid=False,
                details={"reason": "Mathematical error detected"},
                confidence_score=0.0,
            ),
            "notation_validation": SubValidationResult(
                subdomain="notation",
                is_valid=False,
                details={"issues": ["Invalid LaTeX syntax", "Missing braces"]},
                confidence_score=0.5,
            ),
        }

        feedback = validator.generate_feedback_from_details(validation_details)

        assert len(feedback) >= 2
        assert any("accuracy" in fb.lower() for fb in feedback)
        assert any("notation" in fb.lower() for fb in feedback)

    def test_validation_error_handling(self, math_config):
        """Test error handling in mathematics validation."""
        validator = MathematicsValidator("mathematics", math_config)

        # Mock an error in CAS validation
        with patch.object(
            validator, "_perform_cas_validation", side_effect=Exception("CAS error")
        ):
            content = {"problem": "test", "answer": "test"}

            with pytest.raises(DomainValidationError):
                validator.validate_content(content)

    @patch("core.validation.domains.mathematics.verify_with_cas")
    def test_quality_metrics_calculation(self, mock_verify_cas, math_config):
        """Test calculation of comprehensive quality metrics."""
        mock_verify_cas.return_value = {
            "verified": True,
            "method": "algebraic_equivalence",
            "confidence": 1.0,
        }

        validator = MathematicsValidator("mathematics", math_config)

        content = {
            "problem": "Solve the quadratic equation: x^2 - 5x + 6 = 0",
            "answer": "x = 2 or x = 3",
        }

        result = validator.validate_content(content)

        assert result.quality_metrics is not None
        assert hasattr(result.quality_metrics, "fidelity_score")
        assert hasattr(result.quality_metrics, "utility_score")
        assert hasattr(result.quality_metrics, "safety_score")
        assert hasattr(result.quality_metrics, "pedagogical_score")
        assert hasattr(result.quality_metrics, "overall_score")

        # All scores should be between 0 and 1
        metrics = result.quality_metrics
        assert 0 <= metrics.fidelity_score <= 1
        assert 0 <= metrics.utility_score <= 1
        assert 0 <= metrics.safety_score <= 1
        assert 0 <= metrics.pedagogical_score <= 1
        assert 0 <= metrics.overall_score <= 1


if __name__ == "__main__":
    pytest.main([__file__])
