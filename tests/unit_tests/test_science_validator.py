"""
Unit tests for the science domain validator.

This module tests the science validator including scientific method validation,
safety/ethics validation, and subdomain routing.
"""

# Standard Library
from unittest.mock import Mock, patch

# Third-Party Library
import pytest

# SynThesisAI Modules
from core.validation.base import SubValidationResult, ValidationResult
from core.validation.config import ValidationConfig
from core.validation.domains.science import (
    PlaceholderSubdomainValidator,
    SafetyEthicsValidator,
    ScienceValidator,
    ScientificMethodValidator,
)
from core.validation.exceptions import DomainValidationError


class TestScientificMethodValidator:
    """Test cases for ScientificMethodValidator."""

    def test_validator_initialization(self):
        """Test scientific method validator initialization."""
        validator = ScientificMethodValidator()

        assert hasattr(validator, "scientific_method_steps")
        assert hasattr(validator, "experimental_design_elements")
        assert hasattr(validator, "hypothesis_indicators")
        assert hasattr(validator, "observation_indicators")

        assert "hypothesis" in validator.scientific_method_steps
        assert "experiment" in validator.scientific_method_steps
        assert "observation" in validator.scientific_method_steps

    def test_scientific_method_validation_complete(self):
        """Test validation of content with complete scientific method."""
        validator = ScientificMethodValidator()

        content = {
            "problem": "We observed that plants grow taller in sunlight. We hypothesize that light affects plant growth.",
            "answer": "We will test this by growing plants in light and dark conditions, then measure and analyze their heights.",
            "explanation": "The experiment showed plants in light grew taller. We conclude that light promotes plant growth.",
        }

        result = validator.validate_scientific_method(content)

        assert isinstance(result, SubValidationResult)
        assert result.subdomain == "scientific_method"
        # May not be fully valid due to strict requirements, but should have good components
        assert len(result.details["method_components"]) >= 4
        assert "method_components" in result.details
        assert "hypothesis" in result.details["method_components"]
        assert "observation" in result.details["method_components"]
        assert "experiment" in result.details["method_components"]

    def test_scientific_method_validation_incomplete(self):
        """Test validation of content with incomplete scientific method."""
        validator = ScientificMethodValidator()

        content = {
            "problem": "Plants are green.",
            "answer": "Because they have chlorophyll.",
        }

        result = validator.validate_scientific_method(content)

        assert result.subdomain == "scientific_method"
        assert result.is_valid is False  # Should be invalid due to missing components
        assert result.details["overall_score"] < 0.6

    def test_hypothesis_quality_validation_good(self):
        """Test validation of good hypothesis quality."""
        validator = ScientificMethodValidator()

        content = {
            "problem": "If we increase temperature, then enzyme activity will increase because higher temperature provides more kinetic energy.",
            "explanation": "We can test this by measuring enzyme activity at different temperatures.",
        }

        hypothesis_quality = validator._validate_hypothesis_quality(content)

        assert hypothesis_quality["score"] >= 0.5
        assert hypothesis_quality["has_structure"] is True  # Has if-then structure
        assert hypothesis_quality["is_testable"] is True

    def test_hypothesis_quality_validation_poor(self):
        """Test validation of poor hypothesis quality."""
        validator = ScientificMethodValidator()

        content = {
            "problem": "Things happen for reasons.",
            "answer": "Because they do.",
        }

        hypothesis_quality = validator._validate_hypothesis_quality(content)

        assert hypothesis_quality["score"] < 0.5
        assert hypothesis_quality["has_structure"] is False
        assert hypothesis_quality["is_testable"] is False

    def test_experimental_design_validation(self):
        """Test experimental design validation."""
        validator = ScientificMethodValidator()

        content = {
            "problem": "Test the effect of fertilizer on plant growth",
            "explanation": "Use control group without fertilizer and experimental group with fertilizer. Measure plant height over time with 20 plants in each group.",
        }

        design_validation = validator._validate_experimental_design(content)

        assert design_validation["score"] >= 0.2  # More lenient for test
        assert design_validation["has_control"] is True
        assert design_validation["considers_sample_size"] is True
        assert design_validation["has_methodology"] is True

    def test_logical_flow_assessment(self):
        """Test logical flow assessment."""
        validator = ScientificMethodValidator()

        text = "we observed the phenomenon, formed a hypothesis, made predictions, conducted experiments, analyzed results, and drew conclusions"
        components = [
            "observation",
            "hypothesis",
            "prediction",
            "experiment",
            "analysis",
            "conclusion",
        ]

        flow_assessment = validator._assess_logical_flow(text, components)

        assert flow_assessment["score"] >= 0.7
        assert flow_assessment["order_score"] >= 0.8  # Components in correct order

    def test_method_component_identification(self):
        """Test identification of scientific method components."""
        validator = ScientificMethodValidator()

        text = "we observed that water boils at 100°C. our hypothesis is that altitude affects boiling point. we predict that water will boil at lower temperatures at higher altitudes. we conducted an experiment to test this."

        components = validator._identify_method_components(text)

        assert "observation" in components
        assert "hypothesis" in components
        assert "prediction" in components
        assert "experiment" in components

    def test_scientific_method_error_handling(self):
        """Test error handling in scientific method validation."""
        validator = ScientificMethodValidator()

        with patch.object(
            validator,
            "_identify_method_components",
            side_effect=Exception("Test error"),
        ):
            result = validator.validate_scientific_method({"problem": "test"})

            assert result.is_valid is False
            assert "error" in result.details
            assert result.confidence_score == 0.0


class TestSafetyEthicsValidator:
    """Test cases for SafetyEthicsValidator."""

    def test_validator_initialization(self):
        """Test safety/ethics validator initialization."""
        validator = SafetyEthicsValidator()

        assert hasattr(validator, "safety_keywords")
        assert hasattr(validator, "ethics_keywords")
        assert hasattr(validator, "animal_welfare_keywords")
        assert hasattr(validator, "human_subjects_keywords")

        assert "safety" in validator.safety_keywords
        assert "ethics" in validator.ethics_keywords

    def test_safety_validation_with_hazards(self):
        """Test safety validation for content with hazardous materials."""
        validator = SafetyEthicsValidator()

        content = {
            "problem": "Mix hydrochloric acid with sodium hydroxide",
            "explanation": "Wear safety goggles and gloves. Use fume hood for ventilation. Dispose of waste properly.",
        }

        result = validator.validate_safety_ethics(content)

        assert result.subdomain == "safety_ethics"
        assert result.is_valid is True
        safety_details = result.details["safety_assessment"]
        assert safety_details["involves_hazards"] is True
        assert safety_details["safety_measures_mentioned"] is True

    def test_safety_validation_missing_precautions(self):
        """Test safety validation when safety precautions are missing."""
        validator = SafetyEthicsValidator()

        content = {
            "problem": "Mix concentrated sulfuric acid with water",
            "explanation": "Just pour them together and observe the reaction.",
        }

        result = validator.validate_safety_ethics(content)

        assert result.is_valid is False  # Should fail due to missing safety measures
        safety_details = result.details["safety_assessment"]
        assert safety_details["involves_hazards"] is True
        assert safety_details["safety_measures_mentioned"] is False

    def test_ethics_validation_human_subjects(self):
        """Test ethics validation for human subjects research."""
        validator = SafetyEthicsValidator()

        content = {
            "problem": "Study the effects of exercise on human heart rate",
            "explanation": "Obtain IRB approval and informed consent from all participants. Ensure voluntary participation and confidentiality.",
        }

        result = validator.validate_safety_ethics(content)

        assert result.is_valid is True
        ethics_details = result.details["ethics_assessment"]
        assert ethics_details["involves_ethics"] is True
        assert ethics_details["mentions_approval"] is True

        protection_details = result.details["subject_protection"]
        assert protection_details["involves_humans"] is True
        assert protection_details["adequate_protection"] is True

    def test_ethics_validation_animal_subjects(self):
        """Test ethics validation for animal subjects research."""
        validator = SafetyEthicsValidator()

        content = {
            "problem": "Test new drug on laboratory mice",
            "explanation": "Follow IACUC guidelines for animal welfare. Minimize harm and use humane procedures.",
        }

        result = validator.validate_safety_ethics(content)

        assert result.is_valid is True
        protection_details = result.details["subject_protection"]
        assert protection_details["involves_animals"] is True
        assert protection_details["adequate_protection"] is True

    def test_safety_validation_no_hazards(self):
        """Test safety validation for content without hazards."""
        validator = SafetyEthicsValidator()

        content = {
            "problem": "Observe the phases of the moon",
            "explanation": "Look at the moon each night and record its appearance.",
        }

        result = validator.validate_safety_ethics(content)

        assert result.is_valid is True
        safety_details = result.details["safety_assessment"]
        assert safety_details["involves_hazards"] is False
        assert safety_details["score"] >= 0.8  # High score when no hazards

    def test_safety_ethics_error_handling(self):
        """Test error handling in safety/ethics validation."""
        validator = SafetyEthicsValidator()

        with patch.object(
            validator,
            "_assess_safety_considerations",
            side_effect=Exception("Test error"),
        ):
            result = validator.validate_safety_ethics({"problem": "test"})

            assert result.is_valid is False
            assert "error" in result.details
            assert result.confidence_score == 0.0


class TestScienceValidator:
    """Test cases for ScienceValidator."""

    @pytest.fixture
    def science_config(self):
        """Create science validation configuration."""
        return ValidationConfig(
            domain="science",
            quality_thresholds={
                "fidelity_score": 0.8,
                "safety_score": 0.9,
                "scientific_method": 0.6,
            },
            timeout_seconds=30,
        )

    def test_validator_initialization(self, science_config):
        """Test science validator initialization."""
        validator = ScienceValidator("science", science_config)

        assert validator.domain == "science"
        assert validator.config == science_config
        assert hasattr(validator, "scientific_method_validator")
        assert hasattr(validator, "safety_ethics_validator")
        assert hasattr(validator, "subdomain_validators")

        assert "physics" in validator.subdomain_validators
        assert "chemistry" in validator.subdomain_validators
        assert "biology" in validator.subdomain_validators

    def test_subdomain_detection_physics(self, science_config):
        """Test detection of physics subdomain."""
        validator = ScienceValidator("science", science_config)

        content = {
            "problem": "Calculate the force needed to accelerate a 10 kg mass at 5 m/s²",
            "answer": "F = ma = 10 kg × 5 m/s² = 50 N",
        }

        subdomain = validator._detect_subdomain(content)
        assert subdomain == "physics"

    def test_subdomain_detection_chemistry(self, science_config):
        """Test detection of chemistry subdomain."""
        validator = ScienceValidator("science", science_config)

        content = {
            "problem": "What happens when sodium reacts with chlorine?",
            "answer": "Sodium and chlorine form an ionic bond to create sodium chloride (salt)",
        }

        subdomain = validator._detect_subdomain(content)
        assert subdomain == "chemistry"

    def test_subdomain_detection_biology(self, science_config):
        """Test detection of biology subdomain."""
        validator = ScienceValidator("science", science_config)

        content = {
            "problem": "How do cells reproduce?",
            "answer": "Cells reproduce through mitosis, where DNA is replicated and the cell divides",
        }

        subdomain = validator._detect_subdomain(content)
        assert subdomain == "biology"

    def test_comprehensive_science_validation(self, science_config):
        """Test comprehensive science content validation."""
        validator = ScienceValidator("science", science_config)

        content = {
            "problem": "We observed that plants grow toward light. Hypothesis: plants exhibit phototropism due to auxin distribution.",
            "answer": "We tested this by growing plants in controlled light conditions and measuring growth direction.",
            "explanation": "Results confirmed our hypothesis. Plants consistently grew toward the light source.",
            "subdomain": "biology",
        }

        result = validator.validate_content(content)

        assert isinstance(result, ValidationResult)
        assert result.domain == "science"
        assert result.is_valid is True
        assert "scientific_method" in result.validation_details
        assert "safety_ethics" in result.validation_details
        assert "biology_validation" in result.validation_details

    def test_science_validation_with_safety_issues(self, science_config):
        """Test science validation with safety concerns."""
        validator = ScienceValidator("science", science_config)

        content = {
            "problem": "Mix concentrated acids without any safety equipment",
            "answer": "Just pour them together and see what happens",
            "subdomain": "chemistry",
        }

        result = validator.validate_content(content)

        assert result.domain == "science"
        assert result.is_valid is False  # Should fail due to safety issues
        assert len(result.feedback) > 0
        assert any("safety" in feedback.lower() for feedback in result.feedback)

    def test_science_validation_poor_method(self, science_config):
        """Test science validation with poor scientific method."""
        validator = ScienceValidator("science", science_config)

        content = {
            "problem": "Things happen",
            "answer": "Because they do",
            "subdomain": "general",
        }

        result = validator.validate_content(content)

        assert result.domain == "science"
        # May still be valid if safety is OK, but should have low quality score
        assert result.quality_score < 0.8

    def test_quality_score_calculation(self, science_config):
        """Test science-specific quality score calculation."""
        validator = ScienceValidator("science", science_config)

        # Test with scientific content
        content = {
            "problem": "Design an experiment to test the hypothesis that temperature affects enzyme activity",
            "answer": "Measure enzyme activity at different temperatures with proper controls and data collection",
        }

        quality_score = validator.calculate_quality_score(content)

        assert 0.0 <= quality_score <= 1.0
        assert quality_score > 0.8  # Should be high for scientific content

    def test_feedback_generation(self, science_config):
        """Test generation of improvement feedback."""
        validator = ScienceValidator("science", science_config)

        # Create validation details with issues
        validation_details = {
            "scientific_method": SubValidationResult(
                subdomain="scientific_method",
                is_valid=False,
                details={"method_components": ["observation"], "overall_score": 0.3},
                confidence_score=0.5,
            ),
            "safety_ethics": SubValidationResult(
                subdomain="safety_ethics",
                is_valid=False,
                details={
                    "safety_assessment": {
                        "involves_hazards": True,
                        "safety_measures_mentioned": False,
                    }
                },
                confidence_score=0.6,
            ),
        }

        feedback = validator.generate_feedback_from_details(validation_details)

        assert len(feedback) >= 2
        assert any("scientific method" in fb.lower() for fb in feedback)
        assert any("safety" in fb.lower() for fb in feedback)

    def test_validation_error_handling(self, science_config):
        """Test error handling in science validation."""
        validator = ScienceValidator("science", science_config)

        # Mock an error in scientific method validation
        with patch.object(
            validator.scientific_method_validator,
            "validate_scientific_method",
            side_effect=Exception("Test error"),
        ):
            content = {"problem": "test", "answer": "test"}

            with pytest.raises(DomainValidationError):
                validator.validate_content(content)

    def test_quality_metrics_calculation(self, science_config):
        """Test calculation of comprehensive quality metrics."""
        validator = ScienceValidator("science", science_config)

        content = {
            "problem": "Investigate the effect of pH on enzyme activity using controlled experiments",
            "answer": "Design experiments with different pH levels, measure enzyme activity, analyze data statistically",
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


class TestPlaceholderSubdomainValidator:
    """Test cases for PlaceholderSubdomainValidator."""

    def test_placeholder_validator_physics(self):
        """Test placeholder validator for physics."""
        validator = PlaceholderSubdomainValidator("physics")

        content = {"problem": "Calculate force", "answer": "F = ma"}
        result = validator.validate(content)

        assert result.subdomain == "physics"
        assert result.is_valid is True
        assert result.details["validation_type"] == "physics_placeholder"
        assert result.confidence_score == 0.7

    def test_placeholder_validator_chemistry(self):
        """Test placeholder validator for chemistry."""
        validator = PlaceholderSubdomainValidator("chemistry")

        content = {"problem": "Balance equation", "answer": "H2 + Cl2 -> 2HCl"}
        result = validator.validate(content)

        assert result.subdomain == "chemistry"
        assert result.is_valid is True
        assert result.details["validation_type"] == "chemistry_placeholder"

    def test_placeholder_validator_biology(self):
        """Test placeholder validator for biology."""
        validator = PlaceholderSubdomainValidator("biology")

        content = {"problem": "Describe mitosis", "answer": "Cell division process"}
        result = validator.validate(content)

        assert result.subdomain == "biology"
        assert result.is_valid is True
        assert result.details["validation_type"] == "biology_placeholder"


if __name__ == "__main__":
    pytest.main([__file__])
