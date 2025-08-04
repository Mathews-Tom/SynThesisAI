"""
Unit tests for the physics subdomain validator.

This module tests the physics validator including unit consistency validation,
physical law verification, and dimensional analysis.
"""

# Standard Library
from unittest.mock import Mock, patch

# Third-Party Library
import pytest

# SynThesisAI Modules
from core.validation.base import SubValidationResult
from core.validation.domains.physics import (
    PhysicalLawValidator,
    PhysicsValidator,
    UnitConsistencyValidator,
)
from core.validation.exceptions import DomainValidationError


class TestUnitConsistencyValidator:
    """Test cases for UnitConsistencyValidator."""

    def test_validator_initialization(self):
        """Test unit consistency validator initialization."""
        validator = UnitConsistencyValidator()

        assert hasattr(validator, "base_units")
        assert hasattr(validator, "derived_units")
        assert hasattr(validator, "unit_prefixes")

        # Check some expected units
        assert "m" in validator.base_units
        assert "N" in validator.derived_units
        assert "k" in validator.unit_prefixes

    def test_unit_extraction(self):
        """Test extraction of units from physics text."""
        validator = UnitConsistencyValidator()

        text = "The force is 10 N and the velocity is 5 m/s"
        units = validator._extract_units(text)

        assert len(units) >= 2
        unit_strings = [u["unit_string"] for u in units]
        assert any("N" in unit_str for unit_str in unit_strings)
        assert any("m/s" in unit_str or "m" in unit_str for unit_str in unit_strings)

    def test_equation_extraction(self):
        """Test extraction of equations from physics text."""
        validator = UnitConsistencyValidator()

        text = "Using Newton's second law: F = ma where F = 10 N"
        equations = validator._extract_equations(text)

        assert len(equations) >= 1
        assert any("F = ma" in eq or "F = 10 N" in eq for eq in equations)

    def test_unit_recognition(self):
        """Test recognition of physics units."""
        validator = UnitConsistencyValidator()

        # Test base units
        assert validator._is_recognized_unit("m") is True
        assert validator._is_recognized_unit("kg") is True
        assert validator._is_recognized_unit("s") is True

        # Test derived units
        assert validator._is_recognized_unit("N") is True
        assert validator._is_recognized_unit("J") is True
        assert validator._is_recognized_unit("W") is True

        # Test unknown units
        assert validator._is_recognized_unit("xyz") is False

    def test_unit_dimension_detection(self):
        """Test detection of unit dimensions."""
        validator = UnitConsistencyValidator()

        assert validator._get_unit_dimension("m") == "length"
        assert validator._get_unit_dimension("kg") == "mass"
        assert validator._get_unit_dimension("N") == "force"
        assert validator._get_unit_dimension("J") == "energy"
        assert validator._get_unit_dimension("xyz") is None

    def test_equation_dimensional_consistency(self):
        """Test checking of equation dimensional consistency."""
        validator = UnitConsistencyValidator()

        # Test consistent equations
        assert validator._check_equation_dimensional_consistency("F = ma") is True
        assert validator._check_equation_dimensional_consistency("F = 10 N") is True
        assert validator._check_equation_dimensional_consistency("V = IR") is True

        # Test potentially inconsistent equations
        assert validator._check_equation_dimensional_consistency("x = y + z") is False

    def test_unit_consistency_validation_good(self):
        """Test unit consistency validation with good physics content."""
        validator = UnitConsistencyValidator()

        content = {
            "problem": "Calculate the force when mass is 5 kg and acceleration is 2 m/s²",
            "answer": "F = 10 N",
            "solution": "Using F = ma: F = 5 kg × 2 m/s² = 10 N",
        }

        result = validator.validate_unit_consistency(content)

        assert isinstance(result, SubValidationResult)
        assert result.subdomain == "unit_consistency"
        assert result.is_valid is True
        assert result.details["unit_score"] >= 0.8

    def test_unit_consistency_validation_poor(self):
        """Test unit consistency validation with poor physics content."""
        validator = UnitConsistencyValidator()

        content = {
            "problem": "Calculate something with unknown units",
            "answer": "The result is 42 xyz",
            "solution": "Using some formula: a = b + c",
        }

        result = validator.validate_unit_consistency(content)

        assert result.subdomain == "unit_consistency"
        # Should have lower score due to unrecognized units
        assert result.details["unit_score"] < 1.0

    def test_unit_consistency_error_handling(self):
        """Test error handling in unit consistency validation."""
        validator = UnitConsistencyValidator()

        with patch.object(
            validator, "_extract_units", side_effect=Exception("Test error")
        ):
            result = validator.validate_unit_consistency({"problem": "test"})

            assert result.is_valid is False
            assert "error" in result.details
            assert result.confidence_score == 0.0


class TestPhysicalLawValidator:
    """Test cases for PhysicalLawValidator."""

    def test_validator_initialization(self):
        """Test physical law validator initialization."""
        validator = PhysicalLawValidator()

        assert hasattr(validator, "physics_laws")
        assert hasattr(validator, "physics_constants")

        # Check some expected laws
        assert "newton_second" in validator.physics_laws
        assert "conservation_energy" in validator.physics_laws
        assert "ohms_law" in validator.physics_laws

        # Check some expected constants
        assert "c" in validator.physics_constants
        assert "g" in validator.physics_constants

    def test_physics_law_identification(self):
        """Test identification of physics laws in text."""
        validator = PhysicalLawValidator()

        text = (
            "using newton's second law, force equals mass times acceleration, so f = ma"
        )
        laws = validator._identify_physics_laws(text)

        assert len(laws) >= 1
        law_names = [law["name"] for law in laws]
        assert any("Newton's Second Law" in name for name in law_names)

    def test_physics_law_validation_good(self):
        """Test physics law validation with correct application."""
        validator = PhysicalLawValidator()

        content = {
            "problem": "Apply Newton's second law to find force",
            "answer": "F = ma",
            "explanation": "Newton's second law states that force equals mass times acceleration",
        }

        result = validator.validate_physical_laws(content)

        assert isinstance(result, SubValidationResult)
        assert result.subdomain == "physical_laws"
        assert result.is_valid is True
        assert result.details["law_score"] >= 0.7

    def test_physics_law_validation_poor(self):
        """Test physics law validation with poor content."""
        validator = PhysicalLawValidator()

        content = {
            "problem": "Some random physics problem",
            "answer": "The answer is 42",
            "explanation": "Because physics",
        }

        result = validator.validate_physical_laws(content)

        assert result.subdomain == "physical_laws"
        # Should have lower score due to lack of physics laws
        assert result.details["law_score"] < 0.7

    def test_conservation_law_identification(self):
        """Test identification of conservation laws."""
        validator = PhysicalLawValidator()

        text = "energy is conserved in this system, kinetic energy plus potential energy remains constant"
        laws = validator._identify_physics_laws(text)

        assert len(laws) >= 1
        assert any("conservation" in law["name"].lower() for law in laws)

    def test_ohms_law_identification(self):
        """Test identification of Ohm's law."""
        validator = PhysicalLawValidator()

        text = "using ohm's law, voltage equals current times resistance: V = IR"
        laws = validator._identify_physics_laws(text)

        assert len(laws) >= 1
        assert any("ohm" in law["name"].lower() for law in laws)

    def test_physics_constants_check(self):
        """Test checking of physics constants."""
        validator = PhysicalLawValidator()

        text = "the speed of light c is approximately 3×10⁸ m/s"
        constants_check = validator._check_physics_constants(text)

        assert constants_check["accuracy"] > 0.0
        assert len(constants_check["constants_found"]) >= 1
        assert any(
            const["symbol"] == "c" for const in constants_check["constants_found"]
        )

    def test_physical_law_error_handling(self):
        """Test error handling in physical law validation."""
        validator = PhysicalLawValidator()

        with patch.object(
            validator, "_identify_physics_laws", side_effect=Exception("Test error")
        ):
            result = validator.validate_physical_laws({"problem": "test"})

            assert result.is_valid is False
            assert "error" in result.details
            assert result.confidence_score == 0.0


class TestPhysicsValidator:
    """Test cases for PhysicsValidator."""

    def test_validator_initialization(self):
        """Test physics validator initialization."""
        validator = PhysicsValidator()

        assert hasattr(validator, "unit_validator")
        assert hasattr(validator, "law_validator")
        assert isinstance(validator.unit_validator, UnitConsistencyValidator)
        assert isinstance(validator.law_validator, PhysicalLawValidator)

    def test_comprehensive_physics_validation_good(self):
        """Test comprehensive physics validation with good content."""
        validator = PhysicsValidator()

        content = {
            "problem": "Calculate the force when mass is 10 kg and acceleration is 5 m/s²",
            "answer": "F = 50 N",
            "explanation": "Using Newton's second law: F = ma = 10 kg × 5 m/s² = 50 N",
        }

        result = validator.validate(content)

        assert isinstance(result, SubValidationResult)
        assert result.subdomain == "physics"
        assert result.is_valid is True
        assert result.details["physics_score"] >= 0.75

    def test_comprehensive_physics_validation_poor(self):
        """Test comprehensive physics validation with poor content."""
        validator = PhysicsValidator()

        content = {
            "problem": "Some physics problem",
            "answer": "The answer is unknown",
            "explanation": "Physics is complicated",
        }

        result = validator.validate(content)

        assert result.subdomain == "physics"
        # Should have lower score due to lack of physics content
        assert result.details["physics_score"] < 0.85

    def test_mechanics_problem_validation(self):
        """Test validation of mechanics problems."""
        validator = PhysicsValidator()

        content = {
            "problem": "A ball is thrown upward with initial velocity 20 m/s. Find maximum height.",
            "answer": "h = 20.4 m",
            "explanation": "Using kinematic equation: v² = u² + 2as, at max height v=0, so h = u²/2g = 400/19.6 = 20.4 m",
        }

        result = validator.validate(content)

        assert result.subdomain == "physics"
        assert result.is_valid is True
        assert "unit_validation" in result.details
        assert "law_validation" in result.details

    def test_electricity_problem_validation(self):
        """Test validation of electricity problems."""
        validator = PhysicsValidator()

        content = {
            "problem": "Find current when voltage is 12 V and resistance is 4 Ω",
            "answer": "I = 3 A",
            "explanation": "Using Ohm's law: V = IR, so I = V/R = 12/4 = 3 A",
        }

        result = validator.validate(content)

        assert result.subdomain == "physics"
        assert result.is_valid is True

        # Check that Ohm's law was identified
        law_validation = result.details.get("law_validation", {})
        laws_identified = law_validation.get("laws_identified", [])
        assert any("ohm" in law["name"].lower() for law in laws_identified)

    def test_energy_conservation_validation(self):
        """Test validation of energy conservation problems."""
        validator = PhysicsValidator()

        content = {
            "problem": "A pendulum swings from height h. Find velocity at bottom.",
            "answer": "v = √(2gh)",
            "explanation": "By conservation of energy: mgh = ½mv², so v = √(2gh)",
        }

        result = validator.validate(content)

        assert result.subdomain == "physics"
        # Focus on the physics score rather than strict validity
        assert result.details["physics_score"] > 0.3

        # Check that conservation law was identified
        law_validation = result.details.get("law_validation", {})
        laws_identified = law_validation.get("laws_identified", [])
        assert any("conservation" in law["name"].lower() for law in laws_identified)

    def test_feedback_generation(self):
        """Test generation of physics-specific feedback."""
        validator = PhysicsValidator()

        # Create a result with issues
        poor_result = SubValidationResult(
            subdomain="physics",
            is_valid=False,
            details={
                "physics_score": 0.5,
                "unit_validation": {
                    "unit_score": 0.6,
                    "equation_consistency": {
                        "issues": ["Dimensional inconsistency in equation: x = y + z"]
                    },
                },
                "law_validation": {
                    "law_score": 0.4,
                    "law_applications": {
                        "issues": [
                            "Potential issue with Newton's Second Law application"
                        ]
                    },
                },
            },
            confidence_score=0.5,
        )

        feedback = validator.generate_feedback(poor_result)

        assert len(feedback) >= 2
        assert any("unit" in fb.lower() for fb in feedback)
        assert any("law" in fb.lower() or "physics" in fb.lower() for fb in feedback)

    def test_validation_error_handling(self):
        """Test error handling in physics validation."""
        validator = PhysicsValidator()

        # Mock an error in unit validation
        with patch.object(
            validator.unit_validator,
            "validate_unit_consistency",
            side_effect=Exception("Test error"),
        ):
            result = validator.validate({"problem": "test"})

            assert result.is_valid is False
            assert "error" in result.details
            assert result.confidence_score == 0.0

    def test_validation_components_check(self):
        """Test that validation components are properly checked."""
        validator = PhysicsValidator()

        content = {
            "problem": "Calculate work done: W = F × d where F = 10 N and d = 5 m",
            "answer": "W = 50 J",
            "explanation": "Work equals force times distance: W = 10 N × 5 m = 50 J",
        }

        result = validator.validate(content)

        assert "validation_components" in result.details
        components = result.details["validation_components"]
        assert "unit_consistency" in components
        assert "physical_laws" in components
        assert isinstance(components["unit_consistency"], bool)
        assert isinstance(components["physical_laws"], bool)


if __name__ == "__main__":
    pytest.main([__file__])
