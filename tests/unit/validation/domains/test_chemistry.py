"""
Unit tests for the Chemistry domain validator.

This module tests all aspects of chemistry content validation including
chemical equations, reaction mechanisms, safety protocols, and molecular structures.
"""

from unittest.mock import Mock, patch

# Standard Library
import pytest

# SynThesisAI Modules
from core.validation.config import ValidationConfig
from core.validation.domains.chemistry import ChemistryValidator


class TestChemistryValidator:
    """Test suite for ChemistryValidator class."""

    @pytest.fixture
    def validator(self):
        """Create a chemistry validator instance for testing."""
        config = ValidationConfig(
            domain="science", quality_thresholds={"chemistry_score": 0.7}
        )
        return ChemistryValidator("chemistry", config)

    @pytest.fixture
    def organic_validator(self):
        """Create an organic chemistry validator instance."""
        config = ValidationConfig(
            domain="science", quality_thresholds={"chemistry_score": 0.7}
        )
        return ChemistryValidator("organic_chemistry", config)

    def test_validator_initialization(self, validator):
        """Test chemistry validator initialization."""
        assert validator.domain == "science"
        assert validator.subdomain == "chemistry"
        assert validator.config is not None
        assert len(validator.elements) > 0
        assert len(validator.polyatomic_ions) > 0

    def test_invalid_subdomain_raises_error(self):
        """Test that invalid subdomain raises ValueError."""
        config = ValidationConfig(domain="science")

        with pytest.raises(ValueError, match="Invalid chemistry subdomain"):
            ChemistryValidator("invalid_subdomain", config)

    def test_valid_subdomains(self):
        """Test that all valid chemistry subdomains work."""
        config = ValidationConfig(domain="science")
        valid_subdomains = [
            "chemistry",
            "organic_chemistry",
            "inorganic_chemistry",
            "physical_chemistry",
            "analytical_chemistry",
            "biochemistry",
        ]

        for subdomain in valid_subdomains:
            validator = ChemistryValidator(subdomain, config)
            assert validator.subdomain == subdomain

    def test_validate_balanced_chemical_equation(self, validator):
        """Test validation of balanced chemical equations."""
        content = {
            "problem": "Balance the equation: H2 + O2 → H2O",
            "answer": "2H2 + O2 → 2H2O",
            "explanation": "The balanced equation shows 2 hydrogen molecules react with 1 oxygen molecule to form 2 water molecules.",
        }

        result = validator.validate_content(content)

        assert result.domain == "science"
        assert result.validation_details["subdomain"] == "chemistry"
        assert result.is_valid is True
        assert result.quality_score > 0.7
        assert "validation_scores" in result.validation_details

    def test_validate_unbalanced_equation_penalty(self, validator):
        """Test that unbalanced equations receive score penalty."""
        content = {
            "problem": "Balance the equation: H2 + O2 → H2O",
            "answer": "H2 + O2 → H2O",  # Unbalanced
            "explanation": "This equation is not balanced.",
        }

        result = validator.validate_content(content)

        # Should still validate but with lower score
        assert result.quality_score < 0.9
        feedback_text = " ".join(result.feedback).lower()
        assert "unbalanced" in feedback_text or "equation" in feedback_text

    def test_validate_reaction_mechanism(self, validator):
        """Test validation of reaction mechanisms."""
        content = {
            "problem": "Describe the mechanism for the SN2 reaction.",
            "answer": "The SN2 mechanism involves a single step with a transition state.",
            "explanation": "The nucleophile attacks the substrate, forming an intermediate transition state, then the leaving group departs. This is the rate determining step.",
        }

        result = validator.validate_content(content)

        assert result.is_valid is True
        assert result.quality_score > 0.6
        assert "mechanism_validation" in result.validation_details["validation_scores"]

    def test_validate_chemical_safety_with_dangerous_chemicals(self, validator):
        """Test safety validation when dangerous chemicals are mentioned."""
        content = {
            "problem": "How do you handle hydrofluoric acid safely?",
            "answer": "Use protective equipment and proper ventilation.",
            "explanation": "HF is extremely dangerous and requires special safety precautions including protective equipment, ventilation, and proper disposal methods.",
        }

        result = validator.validate_content(content)

        assert result.is_valid is True
        assert "safety_validation" in result.validation_details["validation_scores"]
        assert result.validation_details["validation_scores"]["safety_validation"] > 0.7

    def test_validate_safety_missing_for_dangerous_chemicals(self, validator):
        """Test penalty when dangerous chemicals mentioned without safety."""
        content = {
            "problem": "What is the structure of benzene?",
            "answer": "Benzene has a ring structure with alternating double bonds.",
            "explanation": "Benzene is an aromatic compound with the formula C6H6.",
        }

        result = validator.validate_content(content)

        # Should receive penalty for mentioning benzene without safety considerations
        safety_score = result.validation_details["validation_scores"][
            "safety_validation"
        ]
        assert safety_score < 1.0

    def test_validate_molecular_structures(self, validator):
        """Test validation of molecular structures."""
        content = {
            "problem": "What is the molecular formula of methane?",
            "answer": "CH4",
            "explanation": "Methane consists of one carbon atom bonded to four hydrogen atoms.",
        }

        result = validator.validate_content(content)

        assert result.is_valid is True
        assert "structure_validation" in result.validation_details["validation_scores"]
        assert (
            result.validation_details["validation_scores"]["structure_validation"] > 0.8
        )

    def test_validate_stereochemistry(self, validator):
        """Test validation of stereochemistry concepts."""
        content = {
            "problem": "Explain chirality in organic molecules.",
            "answer": "A chiral molecule has a non-superimposable mirror image.",
            "explanation": "Chirality occurs when a molecule has a chiral center, typically a carbon with four different substituents, leading to optical activity.",
        }

        result = validator.validate_content(content)

        assert result.is_valid is True
        assert result.quality_score > 0.7

    def test_validate_chemical_nomenclature(self, validator):
        """Test validation of chemical nomenclature."""
        content = {
            "problem": "Name the compound CH3CH2OH.",
            "answer": "ethanol",
            "explanation": "This is a two-carbon alcohol, so it is named ethanol according to IUPAC nomenclature.",
        }

        result = validator.validate_content(content)

        assert result.is_valid is True
        assert (
            "nomenclature_validation" in result.validation_details["validation_scores"]
        )

    def test_validate_invalid_chemical_formula(self, validator):
        """Test validation with invalid chemical formulas."""
        content = {
            "problem": "What is the formula of water?",
            "answer": "XYZ123",  # Invalid formula
            "explanation": "This is not a valid chemical formula.",
        }

        result = validator.validate_content(content)

        # Should receive penalty for invalid formula
        assert result.quality_score < 0.85

    def test_validate_empty_content(self, validator):
        """Test validation with empty content."""
        content = {"problem": "", "answer": "", "explanation": ""}

        result = validator.validate_content(content)

        assert result.is_valid is False
        assert result.quality_score == 0.0
        assert "Missing problem statement" in result.feedback

    def test_validate_content_without_problem(self, validator):
        """Test validation when problem is missing."""
        content = {"answer": "Some answer", "explanation": "Some explanation"}

        result = validator.validate_content(content)

        assert result.is_valid is False
        assert "Missing problem statement" in result.feedback

    def test_validation_scoring_weights(self, validator):
        """Test that validation scoring uses correct weights."""
        content = {
            "problem": "Test chemistry problem with equation: H2 + Cl2 → 2HCl",
            "answer": "Balanced equation with proper stoichiometry",
            "explanation": "This reaction shows hydrogen and chlorine forming hydrogen chloride",
        }

        result = validator.validate_content(content)

        # Check that weights are applied correctly
        expected_weights = {
            "equation_validation": 0.25,
            "mechanism_validation": 0.20,
            "safety_validation": 0.20,
            "structure_validation": 0.20,
            "nomenclature_validation": 0.15,
        }

        assert result.validation_details["weights"] == expected_weights

    def test_quality_threshold_application(self, validator):
        """Test that quality threshold is applied correctly."""
        # Set a high threshold
        validator.config.quality_thresholds["chemistry_score"] = 0.95

        content = {
            "problem": "Simple chemistry question",
            "answer": "Simple answer",
            "explanation": "Basic explanation",
        }

        result = validator.validate_content(content)

        # Should fail with high threshold
        assert result.is_valid is False
        assert result.validation_details["threshold"] == 0.95

    def test_organic_chemistry_subdomain(self, organic_validator):
        """Test organic chemistry specific validation."""
        content = {
            "problem": "Draw the structure of butane.",
            "answer": "CH3-CH2-CH2-CH3",
            "explanation": "Butane is a four-carbon alkane with the molecular formula C4H10.",
        }

        result = organic_validator.validate_content(content)

        assert result.validation_details["subdomain"] == "organic_chemistry"
        assert result.is_valid is True

    def test_validation_error_handling(self, validator):
        """Test error handling during validation."""
        # Test with invalid content type
        with patch.object(
            validator,
            "_validate_chemical_equations",
            side_effect=Exception("Test error"),
        ):
            content = {
                "problem": "Test problem",
                "answer": "Test answer",
                "explanation": "Test explanation",
            }

            result = validator.validate_content(content)

            assert result.is_valid is False
            assert result.quality_score == 0.0
            assert "validation error" in str(result.feedback).lower()
            assert "error" in result.validation_details

    def test_element_counting(self, validator):
        """Test element counting functionality."""
        # Test the helper method
        elements = validator._count_elements("H2O")

        assert elements["H"] == 2
        assert elements["O"] == 1

    def test_formula_validation(self, validator):
        """Test chemical formula validation."""
        # Valid formulas
        assert validator._is_valid_formula("H2O") is True
        assert validator._is_valid_formula("NaCl") is True
        assert validator._is_valid_formula("C6H12O6") is True

        # Invalid formulas (elements not in our limited set)
        assert validator._is_valid_formula("XYZ") is False

    def test_chemical_name_extraction(self, validator):
        """Test chemical name extraction."""
        content = (
            "The reaction produces methane and ethanol from the starting materials."
        )

        names = validator._extract_chemical_names(content)

        assert "methane" in names
        assert "ethanol" in names

    def test_safety_protocol_validation(self, validator):
        """Test safety protocol validation."""
        # Content with good safety protocols
        content_good = "Use protective equipment, ensure proper ventilation, and have emergency procedures ready"
        assert validator._validate_safety_protocols(content_good) is True

        # Content with insufficient safety protocols
        content_poor = "Just be careful"
        assert validator._validate_safety_protocols(content_poor) is False

    def test_catalyst_role_validation(self, validator):
        """Test catalyst role validation."""
        # Good catalyst description
        content_good = (
            "The catalyst lowers activation energy and is not consumed in the reaction"
        )
        assert validator._validate_catalyst_role(content_good) is True

        # Poor catalyst description
        content_poor = "The catalyst does something"
        assert validator._validate_catalyst_role(content_poor) is False

    def test_validate_alias_method(self, validator):
        """Test that validate method is an alias for validate_content."""
        content = {
            "problem": "Test problem",
            "answer": "Test answer",
            "explanation": "Test explanation",
        }

        result1 = validator.validate_content(content)
        result2 = validator.validate(content)

        # Results should be identical
        assert result1.is_valid == result2.is_valid
        assert result1.quality_score == result2.quality_score
        assert result1.feedback == result2.feedback


class TestChemistryValidatorIntegration:
    """Integration tests for chemistry validator."""

    def test_comprehensive_chemistry_problem(self):
        """Test validation of a comprehensive chemistry problem."""
        config = ValidationConfig(
            domain="science", quality_thresholds={"chemistry_score": 0.6}
        )
        validator = ChemistryValidator("chemistry", config)

        content = {
            "problem": "Balance the combustion reaction of methane and explain the mechanism. Consider safety precautions.",
            "answer": "CH4 + 2O2 → CO2 + 2H2O. The reaction proceeds through radical intermediates.",
            "explanation": "Methane combustion is exothermic. Safety requires proper ventilation due to toxic CO formation if incomplete. The mechanism involves radical chain reactions with intermediates. Proper protective equipment is essential.",
        }

        result = validator.validate_content(content)

        assert result.is_valid is True
        assert result.quality_score > 0.6
        assert all(
            score > 0
            for score in result.validation_details["validation_scores"].values()
        )

    def test_chemistry_problem_with_all_components(self):
        """Test chemistry problem covering all validation aspects."""
        config = ValidationConfig(
            domain="science", quality_thresholds={"chemistry_score": 0.5}
        )
        validator = ChemistryValidator("organic_chemistry", config)

        content = {
            "problem": "Synthesize ethanol from ethene. Show the mechanism, name all compounds, and discuss safety.",
            "answer": "C2H4 + H2O → C2H5OH (ethanol) via acid-catalyzed hydration",
            "explanation": "The mechanism involves protonation of the alkene, forming a carbocation intermediate, followed by water addition. Ethene is flammable, requiring proper ventilation and no ignition sources. The product ethanol is also flammable and toxic in large quantities.",
        }

        result = validator.validate_content(content)

        assert result.is_valid is True
        assert result.validation_details["subdomain"] == "organic_chemistry"

        # Check all validation components were evaluated
        validation_scores = result.validation_details["validation_scores"]
        assert "equation_validation" in validation_scores
        assert "mechanism_validation" in validation_scores
        assert "safety_validation" in validation_scores
        assert "structure_validation" in validation_scores
        assert "nomenclature_validation" in validation_scores
