"""
Unit tests for DSPy integration.

These tests verify the core functionality of the DSPy integration module,
including configuration, signatures, and basic module initialization.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from core.dspy import (
    DSPY_AVAILABLE,
    SignatureManager,
    SignatureValidationError,
    STREAMContentGenerator,
    create_custom_signature,
    get_all_domains,
    get_domain_signature,
    get_dspy_config,
    validate_signature,
)


class TestDSPySignatures:
    """Test DSPy signature functionality."""

    def test_get_domain_signature(self):
        """Test retrieving domain signatures."""
        math_sig = get_domain_signature("mathematics", "generation")
        assert "mathematical_concept" in math_sig
        assert "problem_statement" in math_sig
        assert "->" in math_sig

        science_sig = get_domain_signature("science", "generation")
        assert "scientific_concept" in science_sig
        assert "experimental_design" in science_sig

    def test_validate_signature(self):
        """Test signature validation."""
        valid_sig = "input1, input2 -> output1, output2"
        assert validate_signature(valid_sig) is True

        with pytest.raises(SignatureValidationError):
            validate_signature("invalid signature")

        with pytest.raises(SignatureValidationError):
            validate_signature("input1, input2")

        with pytest.raises(SignatureValidationError):
            validate_signature(" -> output1, output2")

    def test_create_custom_signature(self):
        """Test creating custom signatures."""
        inputs = ["concept", "difficulty"]
        outputs = ["problem", "solution"]

        signature = create_custom_signature(inputs, outputs)
        assert signature == "concept, difficulty -> problem, solution"

        with pytest.raises(SignatureValidationError):
            create_custom_signature([], outputs)

        with pytest.raises(SignatureValidationError):
            create_custom_signature(inputs, [])

    def test_signature_manager(self):
        """Test SignatureManager functionality."""
        manager = SignatureManager()

        # Test getting signatures
        math_sig = manager.get_signature("mathematics", "generation")
        assert "mathematical_concept" in math_sig

        # Test registering custom signature
        custom_sig = "concept, difficulty -> problem, solution"
        result = manager.register_custom_signature(
            "custom_domain", "generation", custom_sig
        )
        assert result is True

        # Test signature compatibility
        sig1 = "a, b, c -> x, y, z"
        sig2 = "a, b -> x, y"
        assert manager.is_signature_compatible(sig1, sig2) is True
        assert manager.is_signature_compatible(sig2, sig1) is False
        assert manager.is_signature_compatible(sig1, sig1, strict=True) is True


class TestDSPyConfig:
    """Test DSPy configuration functionality."""

    def test_get_config(self):
        """Test retrieving DSPy configuration."""
        config = get_dspy_config()
        assert config is not None

        # Test basic configuration methods
        assert isinstance(config.is_enabled(), bool)
        assert isinstance(config.get_cache_config(), dict)
        assert isinstance(config.get_optimization_config(), dict)

        # Test module configuration
        module_config = config.get_module_config("mathematics", "test_signature")
        assert module_config.domain == "mathematics"
        assert module_config.signature == "test_signature"
        assert isinstance(module_config.quality_requirements, dict)


@pytest.mark.skipif(not DSPY_AVAILABLE, reason="DSPy not available")
class TestSTREAMContentGenerator:
    """Test STREAMContentGenerator functionality."""

    def test_initialization(self):
        """Test initializing content generator."""
        generator = STREAMContentGenerator("mathematics")
        assert generator.domain == "mathematics"
        assert hasattr(generator, "generate")
        assert hasattr(generator, "refine")

    @patch("core.dspy.base_module.dspy.ChainOfThought")
    def test_forward_method(self, mock_chain):
        """Test forward method with mocked DSPy."""
        # Setup mock
        mock_instance = MagicMock()
        mock_chain.return_value = mock_instance

        # Create a mock result with sufficient content to avoid refinement
        mock_result = MagicMock()
        mock_result.problem_statement = (
            "Test problem with sufficient length to avoid refinement"
        )
        mock_result.solution = (
            "Test solution with sufficient length to avoid refinement"
        )
        mock_result.proof = "Test proof"
        mock_result.reasoning_trace = "Test reasoning trace"
        mock_result.pedagogical_hints = "Test hints"

        mock_instance.return_value = mock_result

        # Create generator and test forward method
        generator = STREAMContentGenerator("mathematics")

        # Patch the needs_refinement method to always return False
        with patch.object(generator, "needs_refinement", return_value=False):
            result = generator(
                mathematical_concept="algebra",
                difficulty_level="high_school",
                learning_objectives=["solve equations"],
            )

        # Verify result
        assert hasattr(result, "problem_statement")
        assert hasattr(result, "solution")

        # Verify mock was called at least once (we don't care about exact count)
        assert mock_instance.call_count >= 1


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
