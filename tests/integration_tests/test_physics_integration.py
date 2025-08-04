"""
Integration tests for the physics subdomain validator.

This module tests the complete physics validation workflow including
integration with the science domain validator.
"""

# Standard Library
from unittest.mock import patch

# Third-Party Library
import pytest

# SynThesisAI Modules
from core.validation.config import ValidationConfig
from core.validation.domains.physics import PhysicsValidator
from core.validation.domains.science import ScienceValidator


class TestPhysicsValidatorIntegration:
    """Integration tests for physics validator with the science system."""

    def test_physics_validator_with_science_validator(self):
        """Test physics validator integration with science validator."""
        science_config = ValidationConfig(
            domain="science",
            quality_thresholds={"fidelity_score": 0.8},
            timeout_seconds=30,
        )
        science_validator = ScienceValidator("science", science_config)

        # Test physics content
        content = {
            "problem": "Calculate the force when mass is 10 kg and acceleration is 5 m/s²",
            "answer": "F = 50 N",
            "explanation": "Using Newton's second law: F = ma = 10 kg × 5 m/s² = 50 N",
        }

        result = science_validator.validate_content(content)

        assert result.domain == "science"
        assert "physics_validation" in result.validation_details

        # Check that physics validation was performed
        physics_result = result.validation_details["physics_validation"]
        assert physics_result.subdomain == "physics"
        assert physics_result.is_valid is True

        # The overall science result may fail due to scientific method requirements,
        # but the physics validation should be successful
        assert physics_result.details["physics_score"] > 0.8

    def test_mechanics_problem_integration(self):
        """Test integration with mechanics problems."""
        physics_validator = PhysicsValidator()

        content = {
            "problem": "A car accelerates from rest at 2 m/s² for 10 seconds. Find final velocity.",
            "answer": "v = 20 m/s",
            "explanation": "Using v = u + at: v = 0 + 2×10 = 20 m/s",
        }

        result = physics_validator.validate(content)

        assert result.subdomain == "physics"
        assert result.is_valid is True
        assert result.details["physics_score"] > 0.6

        # Check unit validation
        unit_validation = result.details["unit_validation"]
        assert unit_validation["recognized_units"] > 0

    def test_electricity_problem_integration(self):
        """Test integration with electricity problems."""
        physics_validator = PhysicsValidator()

        content = {
            "problem": "Calculate power when voltage is 12 V and current is 2 A",
            "answer": "P = 24 W",
            "explanation": "Using P = VI: P = 12 V × 2 A = 24 W",
        }

        result = physics_validator.validate(content)

        assert result.subdomain == "physics"
        assert result.is_valid is True

        # Check that electrical units were recognized
        unit_validation = result.details["unit_validation"]
        units_found = unit_validation["units_found"]
        unit_strings = [u["unit_string"] for u in units_found]
        assert any("V" in unit_str for unit_str in unit_strings)
        assert any("A" in unit_str for unit_str in unit_strings)
        assert any("W" in unit_str for unit_str in unit_strings)

    def test_energy_conservation_integration(self):
        """Test integration with energy conservation problems."""
        physics_validator = PhysicsValidator()

        content = {
            "problem": "A ball falls from height 10 m. Find velocity just before impact.",
            "answer": "v = 14 m/s",
            "explanation": "By conservation of energy: mgh = ½mv², so v = √(2gh) = √(2×9.8×10) = 14 m/s",
        }

        result = physics_validator.validate(content)

        assert result.subdomain == "physics"
        assert result.details["physics_score"] > 0.3

        # Check that conservation law was identified
        law_validation = result.details["law_validation"]
        laws_identified = law_validation["laws_identified"]
        assert any("conservation" in law["name"].lower() for law in laws_identified)

    def test_physics_validation_with_errors(self):
        """Test physics validation with errors and feedback."""
        physics_validator = PhysicsValidator()

        content = {
            "problem": "Calculate something with wrong units",
            "answer": "The result is 42 xyz",
            "explanation": "Using some random formula",
        }

        result = physics_validator.validate(content)

        assert result.subdomain == "physics"
        # Should have lower score due to unrecognized units
        assert result.details["physics_score"] < 0.8

        # Should generate feedback
        feedback = physics_validator.generate_feedback(result)
        assert len(feedback) > 0

    def test_physics_constants_integration(self):
        """Test integration with physics constants."""
        physics_validator = PhysicsValidator()

        content = {
            "problem": "What is the speed of light?",
            "answer": "c = 3×10⁸ m/s",
            "explanation": "The speed of light in vacuum is approximately 299,792,458 m/s",
        }

        result = physics_validator.validate(content)

        assert result.subdomain == "physics"
        assert result.is_valid is True

        # Check that speed of light constant was identified
        law_validation = result.details["law_validation"]
        constants_found = law_validation["constants_check"]["constants_found"]
        assert any(const["symbol"] == "c" for const in constants_found)

    def test_dimensional_analysis_integration(self):
        """Test dimensional analysis integration."""
        physics_validator = PhysicsValidator()

        content = {
            "problem": "Verify that F = ma is dimensionally consistent",
            "answer": "Force has dimensions [MLT⁻²], mass [M], acceleration [LT⁻²]",
            "explanation": "F = ma gives [MLT⁻²] = [M][LT⁻²] = [MLT⁻²] ✓",
        }

        result = physics_validator.validate(content)

        assert result.subdomain == "physics"
        assert result.is_valid is True

        # Check dimensional analysis
        unit_validation = result.details["unit_validation"]
        dimensional_analysis = unit_validation["dimensional_analysis"]
        assert dimensional_analysis["score"] > 0.5

    def test_physics_validation_performance(self):
        """Test physics validation performance with multiple problems."""
        physics_validator = PhysicsValidator()

        # Create multiple physics problems
        problems = [
            {
                "problem": f"Calculate force when mass is {i} kg and acceleration is 2 m/s²",
                "answer": f"F = {i * 2} N",
                "explanation": f"Using F = ma: F = {i} kg × 2 m/s² = {i * 2} N",
            }
            for i in range(1, 6)
        ]

        results = []
        for problem in problems:
            result = physics_validator.validate(problem)
            results.append(result)

        # Check all validations completed successfully
        assert len(results) == 5
        assert all(result.subdomain == "physics" for result in results)
        assert all(result.is_valid for result in results)
        assert all(result.details["physics_score"] > 0.6 for result in results)

    def test_physics_validation_error_recovery(self):
        """Test physics validation error recovery."""
        physics_validator = PhysicsValidator()

        # Test with malformed content
        malformed_content = {
            "problem": None,
            "answer": "",
            "explanation": 123,  # Invalid type
        }

        # Should handle gracefully without crashing
        try:
            result = physics_validator.validate(malformed_content)
            # If it doesn't raise an exception, check that it handled the error
            assert result.subdomain == "physics"
        except Exception as e:
            # If it raises an exception, it should be handled gracefully
            assert "validation" in str(e).lower() or "error" in str(e).lower()

    def test_physics_feedback_quality(self):
        """Test quality of feedback generated by physics validator."""
        physics_validator = PhysicsValidator()

        # Content with multiple issues
        problematic_content = {
            "problem": "Calculate something with mixed units and wrong equations",
            "answer": "F = 10 pounds + 5 meters",
            "explanation": "Using some made-up formula: x = y + z",
        }

        result = physics_validator.validate(problematic_content)

        assert result.subdomain == "physics"

        # Generate feedback
        feedback = physics_validator.generate_feedback(result)

        if not result.is_valid:
            assert len(feedback) > 0

            # Check feedback quality
            feedback_text = " ".join(feedback).lower()

            # Feedback should be actionable
            assert any(
                word in feedback_text
                for word in ["check", "verify", "ensure", "review"]
            )

    def test_physics_subdomain_routing(self):
        """Test physics subdomain routing from science validator."""
        science_config = ValidationConfig(
            domain="science",
            quality_thresholds={"fidelity_score": 0.8},
            timeout_seconds=30,
        )
        science_validator = ScienceValidator("science", science_config)

        # Test that physics content is routed to physics validator
        physics_content = {
            "problem": "Calculate momentum of a 5 kg object moving at 10 m/s",
            "answer": "p = 50 kg⋅m/s",
            "explanation": "Using p = mv: p = 5 kg × 10 m/s = 50 kg⋅m/s",
        }

        # Check subdomain detection
        subdomain = science_validator._detect_subdomain(physics_content)
        assert subdomain == "physics"

        # Check full validation
        result = science_validator.validate_content(physics_content)
        assert result.domain == "science"
        assert "physics_validation" in result.validation_details


if __name__ == "__main__":
    pytest.main([__file__])
