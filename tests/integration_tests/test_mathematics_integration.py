"""
Integration tests for the mathematics domain validator.

This module tests the complete mathematics validation workflow including
CAS integration, notation validation, proof checking, and difficulty assessment.
"""

# Standard Library
from unittest.mock import patch

# Third-Party Library
import pytest

# SynThesisAI Modules
from core.validation import UniversalValidator, ValidationConfigManager
from core.validation.config import ValidationConfig
from core.validation.domains.mathematics import MathematicsValidator


class TestMathematicsValidatorIntegration:
    """Integration tests for mathematics validator with the universal system."""

    @pytest.mark.asyncio
    async def test_mathematics_validation_through_universal_validator(self):
        """Test mathematics validation through the universal validator."""
        # Create universal validator
        validator = UniversalValidator()

        # Test mathematical content
        content = {
            "problem": "Solve the equation: 3x + 7 = 22",
            "answer": "x = 5",
            "steps": ["3x + 7 = 22", "3x = 15", "x = 5"],
        }

        # Validate through universal validator
        result = await validator.validate_content(content, "mathematics")

        # Verify result structure
        assert result.domain == "mathematics"
        assert isinstance(result.is_valid, bool)
        assert 0 <= result.quality_score <= 1
        assert 0 <= result.confidence_score <= 1
        assert result.quality_metrics is not None

        # Verify mathematics-specific validation details
        assert "cas_verification" in result.validation_details
        assert "notation_validation" in result.validation_details
        assert "difficulty_assessment" in result.validation_details

    @pytest.mark.asyncio
    async def test_mathematics_proof_validation_integration(self):
        """Test mathematics proof validation integration."""
        validator = UniversalValidator()

        # Test content with mathematical proof
        content = {
            "problem": "Prove that the square of an even number is even",
            "answer": "If n is even, then n² is even",
            "proof": "Proof: Let n be an even integer. Then n = 2k for some integer k. We have n² = (2k)² = 4k² = 2(2k²). Since 2k² is an integer, n² is even. Therefore, the square of an even number is even. QED",
            "theorem": "Square of even number is even",
        }

        result = await validator.validate_content(content, "mathematics")

        # Should include proof validation
        assert "proof_validation" in result.validation_details
        proof_result = result.validation_details["proof_validation"]
        assert proof_result.subdomain == "proof"
        assert "proof_type" in proof_result.details

    @pytest.mark.asyncio
    async def test_mathematics_advanced_content_validation(self):
        """Test validation of advanced mathematical content."""
        validator = UniversalValidator()

        # Test calculus content
        content = {
            "problem": "Find the derivative of f(x) = x³ + 2x² - 5x + 1",
            "answer": "f'(x) = 3x² + 4x - 5",
            "difficulty": "advanced",
            "domain": "calculus",
        }

        result = await validator.validate_content(content, "mathematics")

        # Verify advanced content handling
        difficulty_result = result.validation_details["difficulty_assessment"]
        assert difficulty_result.details["primary_level"] in [
            "intermediate",
            "advanced",
        ]
        assert len(difficulty_result.details["prerequisites"]) > 0

    @pytest.mark.asyncio
    async def test_mathematics_error_handling_integration(self):
        """Test error handling in mathematics validation integration."""
        validator = UniversalValidator()

        # Test with problematic content
        content = {
            "problem": "What is 2 + 2?",
            "answer": "5",  # Incorrect answer
            "difficulty": "basic",
        }

        result = await validator.validate_content(content, "mathematics")

        # Should handle the error gracefully
        assert result.domain == "mathematics"
        # May be invalid due to incorrect answer, but shouldn't crash
        assert isinstance(result.is_valid, bool)
        assert len(result.feedback) >= 0  # May have feedback about the error

    def test_mathematics_validator_configuration_integration(self):
        """Test mathematics validator configuration integration."""
        # Create configuration manager
        config_manager = ValidationConfigManager()

        # Load mathematics configuration
        math_config = config_manager.load_domain_config("mathematics")

        # Verify configuration structure
        assert math_config.domain == "mathematics"
        assert "fidelity_score" in math_config.quality_thresholds
        assert math_config.timeout_seconds > 0

        # Create validator with configuration
        validator = MathematicsValidator("mathematics", math_config)

        # Verify validator initialization
        assert validator.domain == "mathematics"
        assert validator.config == math_config
        assert hasattr(validator, "cas_validator")
        assert hasattr(validator, "notation_validator")
        assert hasattr(validator, "proof_validator")
        assert hasattr(validator, "difficulty_assessor")

    @patch("core.validation.domains.mathematics.verify_with_cas")
    def test_mathematics_cas_integration(self, mock_cas):
        """Test CAS integration in mathematics validator."""
        # Mock CAS response
        mock_cas.return_value = {
            "verified": True,
            "method": "algebraic_equivalence",
            "confidence": 1.0,
            "reason": "Expressions are algebraically equivalent",
        }

        # Create validator
        config = ValidationConfig(domain="mathematics")
        validator = MathematicsValidator("mathematics", config)

        # Test content
        content = {"problem": "Simplify: 2x + 3x", "answer": "5x"}

        result = validator.validate_content(content)

        # Verify CAS integration
        assert "cas_verification" in result.validation_details
        cas_result = result.validation_details["cas_verification"]
        assert cas_result.subdomain == "cas"
        assert cas_result.is_valid is True

        # Verify CAS was called
        mock_cas.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
