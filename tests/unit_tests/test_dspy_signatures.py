"""
Unit tests for DSPy signatures.

These tests verify the enhanced signature functionality for STREAM domains,
including signature validation, parsing, and compatibility checking.
"""

import pytest

from core.dspy.signatures import (
    SignatureManager,
    SignatureValidationError,
    create_custom_signature,
    get_all_domains,
    get_domain_signature,
    get_signature_types,
    validate_signature,
)


class TestDSPySignatures:
    """Test DSPy signature functionality."""

    def test_get_domain_signature(self):
        """Test retrieving domain signatures."""
        # Test mathematics domain
        math_sig = get_domain_signature("mathematics", "generation")
        assert "mathematical_concept" in math_sig
        assert "problem_statement" in math_sig
        assert "proof" in math_sig
        assert "reasoning_trace" in math_sig
        assert "pedagogical_hints" in math_sig
        assert "misconception_analysis" in math_sig

        # Test science domain
        science_sig = get_domain_signature("science", "generation")
        assert "scientific_concept" in science_sig
        assert "experimental_design" in science_sig
        assert "evidence_evaluation" in science_sig
        assert "scientific_principles" in science_sig

        # Test technology domain
        tech_sig = get_domain_signature("technology", "generation")
        assert "technical_concept" in tech_sig
        assert "algorithm_explanation" in tech_sig
        assert "system_design" in tech_sig
        assert "implementation_considerations" in tech_sig

        # Test reading domain
        reading_sig = get_domain_signature("reading", "generation")
        assert "literary_concept" in reading_sig
        assert "comprehension_question" in reading_sig
        assert "analysis_prompt" in reading_sig
        assert "critical_thinking_exercise" in reading_sig

        # Test engineering domain
        engineering_sig = get_domain_signature("engineering", "generation")
        assert "engineering_concept" in engineering_sig
        assert "design_challenge" in engineering_sig
        assert "optimization_problem" in engineering_sig
        assert "constraint_analysis" in engineering_sig

        # Test arts domain
        arts_sig = get_domain_signature("arts", "generation")
        assert "artistic_concept" in arts_sig
        assert "creative_prompt" in arts_sig
        assert "aesthetic_analysis" in arts_sig
        assert "cultural_context" in arts_sig

        # Test interdisciplinary domain
        interdisciplinary_sig = get_domain_signature("interdisciplinary", "generation")
        assert "primary_domain" in interdisciplinary_sig
        assert "secondary_domain" in interdisciplinary_sig
        assert "cross_domain_connections" in interdisciplinary_sig
        assert "interdisciplinary_principles" in interdisciplinary_sig

    def test_validate_signature(self):
        """Test signature validation."""
        # Test valid signatures
        valid_signatures = [
            "input1, input2 -> output1, output2",
            "concept -> solution",
            "a, b, c -> x, y, z",
            "input_with_underscore -> output_with_underscore",
        ]
        for sig in valid_signatures:
            assert validate_signature(sig) is True

        # Test invalid signatures
        invalid_signatures = [
            "",  # Empty
            "input1, input2",  # Missing separator
            "-> output1, output2",  # Missing inputs
            "input1, input2 ->",  # Missing outputs
            "input1, input2 -> output1 -> extra",  # Too many separators
            "input@, input2 -> output1",  # Invalid field name
            "input1, -> output1",  # Empty field
        ]
        for sig in invalid_signatures:
            with pytest.raises(SignatureValidationError):
                validate_signature(sig)

    def test_create_custom_signature(self):
        """Test creating custom signatures."""
        # Test valid inputs
        inputs = ["concept", "difficulty"]
        outputs = ["problem", "solution"]
        sig = create_custom_signature(inputs, outputs)
        assert sig == "concept, difficulty -> problem, solution"

        # Test with more fields
        inputs = ["a", "b", "c", "d"]
        outputs = ["w", "x", "y", "z"]
        sig = create_custom_signature(inputs, outputs)
        assert sig == "a, b, c, d -> w, x, y, z"

        # Test invalid inputs
        with pytest.raises(SignatureValidationError):
            create_custom_signature([], outputs)

        with pytest.raises(SignatureValidationError):
            create_custom_signature(inputs, [])

        with pytest.raises(SignatureValidationError):
            create_custom_signature(["invalid@name"], outputs)

    def test_get_all_domains(self):
        """Test getting all domains."""
        domains = get_all_domains()
        assert "mathematics" in domains
        assert "science" in domains
        assert "technology" in domains
        assert "reading" in domains
        assert "engineering" in domains
        assert "arts" in domains
        assert "interdisciplinary" in domains
        assert len(domains) >= 7  # At least 7 domains

    def test_get_signature_types(self):
        """Test getting signature types."""
        types = get_signature_types("mathematics")
        assert "generation" in types
        assert "validation" in types
        assert "equivalence" in types
        assert "refinement" in types
        assert "assessment" in types
        assert len(types) >= 5  # At least 5 types

        # Test invalid domain
        with pytest.raises(SignatureValidationError):
            get_signature_types("invalid_domain")


class TestSignatureManager:
    """Test SignatureManager functionality."""

    def test_initialization(self):
        """Test SignatureManager initialization."""
        manager = SignatureManager()
        assert manager.signatures is not None
        assert "mathematics" in manager.signatures
        assert "science" in manager.signatures
        assert len(manager.signatures) >= 7  # At least 7 domains

    def test_get_signature(self):
        """Test getting signatures."""
        manager = SignatureManager()

        # Test getting built-in signatures
        math_sig = manager.get_signature("mathematics", "generation")
        assert "mathematical_concept" in math_sig
        assert "problem_statement" in math_sig

        # Test case insensitivity
        math_sig2 = manager.get_signature("MATHEMATICS", "generation")
        assert math_sig == math_sig2

    def test_register_custom_signature(self):
        """Test registering custom signatures."""
        manager = SignatureManager()

        # Register custom signature
        custom_sig = "custom_input1, custom_input2 -> custom_output1, custom_output2"
        result = manager.register_custom_signature(
            "custom_domain", "custom_type", custom_sig, "2.0.0"
        )
        assert result is True

        # Verify custom signature is registered
        retrieved_sig = manager.get_signature("custom_domain", "custom_type")
        assert retrieved_sig == custom_sig

        # Verify version is registered
        version = manager.get_signature_version("custom_domain", "custom_type")
        assert version == "2.0.0"

    def test_parse_signature(self):
        """Test parsing signatures."""
        manager = SignatureManager()

        # Parse simple signature
        inputs, outputs = manager.parse_signature("a, b -> x, y")
        assert inputs == ["a", "b"]
        assert outputs == ["x", "y"]

        # Parse complex signature
        inputs, outputs = manager.parse_signature(
            "input1, input2, input3 -> output1, output2, output3, output4"
        )
        assert inputs == ["input1", "input2", "input3"]
        assert outputs == ["output1", "output2", "output3", "output4"]

    def test_signature_compatibility(self):
        """Test signature compatibility checking."""
        manager = SignatureManager()

        # Test compatible signatures (non-strict)
        assert (
            manager.is_signature_compatible("a, b, c -> x, y, z", "a, b -> x, y")
            is True
        )

        # Test incompatible signatures (missing required input)
        assert (
            manager.is_signature_compatible("a, b -> x, y, z", "a, b, c -> x, y")
            is False
        )

        # Test incompatible signatures (missing required output)
        assert (
            manager.is_signature_compatible("a, b, c -> x, y", "a, b -> x, y, z")
            is False
        )

        # Test strict compatibility
        assert (
            manager.is_signature_compatible(
                "a, b, c -> x, y, z", "a, b, c -> x, y, z", strict=True
            )
            is True
        )

        assert (
            manager.is_signature_compatible(
                "a, b, c -> x, y, z", "a, b, c -> x, y", strict=True
            )
            is False
        )

    def test_extend_signature(self):
        """Test extending signatures."""
        manager = SignatureManager()

        # Extend with additional inputs and outputs
        extended_sig = manager.extend_signature(
            "a, b -> x, y", additional_inputs=["c", "d"], additional_outputs=["z"]
        )
        assert extended_sig == "a, b, c, d -> x, y, z"

        # Extend with duplicate fields (should be ignored)
        extended_sig = manager.extend_signature(
            "a, b -> x, y", additional_inputs=["b", "c"], additional_outputs=["y", "z"]
        )
        assert extended_sig == "a, b, c -> x, y, z"

    def test_create_composite_signature(self):
        """Test creating composite signatures."""
        manager = SignatureManager()

        # Create composite from mathematics and science
        composite_sig = manager.create_composite_signature(
            "mathematics", "science", "generation"
        )

        # Verify composite contains fields from both domains
        assert "mathematical_concept" in composite_sig
        assert "scientific_concept" in composite_sig
        assert "problem_statement" in composite_sig
        assert "experimental_design" in composite_sig
        assert "reasoning_trace" in composite_sig

    def test_simplify_signature(self):
        """Test simplifying signatures."""
        manager = SignatureManager()

        # Simplify by keeping only specific inputs and outputs
        simplified_sig = manager.simplify_signature(
            "a, b, c, d -> w, x, y, z",
            required_inputs=["a", "c"],
            required_outputs=["w", "z"],
        )
        assert simplified_sig == "a, c -> w, z"

        # Test with invalid required fields
        with pytest.raises(SignatureValidationError):
            manager.simplify_signature(
                "a, b -> x, y",
                required_inputs=["a", "c"],  # 'c' not in signature
                required_outputs=["x"],
            )

    def test_get_domain_signatures_by_type(self):
        """Test getting domain signatures by type."""
        manager = SignatureManager()

        # Get all generation signatures
        generation_sigs = manager.get_domain_signatures_by_type("generation")
        assert "mathematics" in generation_sigs
        assert "science" in generation_sigs
        assert "technology" in generation_sigs
        assert len(generation_sigs) >= 7  # At least 7 domains

        # Get all validation signatures
        validation_sigs = manager.get_domain_signatures_by_type("validation")
        assert "mathematics" in validation_sigs
        assert "science" in validation_sigs
        assert "technology" in validation_sigs
        assert len(validation_sigs) >= 7  # At least 7 domains


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
