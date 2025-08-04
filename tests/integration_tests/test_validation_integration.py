"""
Integration tests for the universal validation framework.

This module tests the complete validation workflow from configuration
loading through domain validation and result aggregation.
"""

# Standard Library
import asyncio
import tempfile
from pathlib import Path

# Third-Party Library
import pytest

# SynThesisAI Modules
from core.validation import (
    UniversalValidator,
    ValidationConfigManager,
    get_universal_validator,
)
from core.validation.base import ValidationResult


class TestValidationIntegration:
    """Integration tests for the complete validation system."""

    @pytest.mark.asyncio
    async def test_end_to_end_validation_workflow(self):
        """Test complete validation workflow from configuration to result."""
        # Create temporary configuration directory
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)

            # Create configuration manager with temporary directory
            config_manager = ValidationConfigManager(config_dir)

            # Create universal validator
            validator = UniversalValidator(config_manager)

            # Test content for validation
            test_content = {
                "problem": "What is 2 + 2?",
                "answer": "4",
                "domain": "mathematics",
                "difficulty": "basic",
            }

            # Perform validation
            result = await validator.validate_content(test_content, "mathematics")

            # Verify result structure
            assert isinstance(result, ValidationResult)
            assert result.domain == "mathematics"
            assert isinstance(result.is_valid, bool)
            assert 0 <= result.quality_score <= 1
            assert 0 <= result.confidence_score <= 1
            assert isinstance(result.feedback, list)
            assert isinstance(result.validation_details, dict)

    @pytest.mark.asyncio
    async def test_multi_domain_validation_integration(self):
        """Test validation across multiple domains."""
        # Create temporary configuration directory
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)
            config_manager = ValidationConfigManager(config_dir)
            validator = UniversalValidator(config_manager)

            # Test content that could apply to multiple domains
            test_content = {
                "content": "Calculate the area of a circle with radius 5",
                "type": "problem",
                "subject": "geometry",
            }

            # Validate across multiple domains
            domains = ["mathematics", "science"]
            results = await validator.validate_multiple_domains(test_content, domains)

            # Verify results
            assert len(results) == 2
            assert "mathematics" in results
            assert "science" in results

            for domain, result in results.items():
                assert isinstance(result, ValidationResult)
                assert result.domain == domain
                assert isinstance(result.is_valid, bool)

    def test_configuration_persistence_integration(self):
        """Test that configurations are properly loaded and persisted."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)
            config_manager = ValidationConfigManager(config_dir)

            # Load domain configuration (should create default)
            config1 = config_manager.load_domain_config("mathematics")

            # Verify configuration file was created
            config_file = config_dir / "mathematics_validation.json"
            assert config_file.exists()

            # Load configuration again (should read from file)
            config2 = config_manager.load_domain_config("mathematics")

            # Verify configurations are equivalent
            assert config1.domain == config2.domain
            assert config1.timeout_seconds == config2.timeout_seconds
            assert config1.quality_thresholds == config2.quality_thresholds

    def test_supported_domains_integration(self):
        """Test that all expected domains are supported."""
        validator = get_universal_validator()

        supported_domains = validator.get_supported_domains()
        expected_domains = [
            "mathematics",
            "science",
            "technology",
            "reading",
            "engineering",
            "arts",
        ]

        # All expected domains should be supported (even if using placeholder validators)
        for domain in expected_domains:
            assert validator.is_domain_supported(
                domain
            ), f"Domain {domain} should be supported"

    @pytest.mark.asyncio
    async def test_error_handling_integration(self):
        """Test error handling in the complete validation workflow."""
        validator = get_universal_validator()

        # Test with unsupported domain
        test_content = {"test": "content"}

        with pytest.raises(Exception):  # Should raise ValidationError
            await validator.validate_content(test_content, "unsupported_domain")

    @pytest.mark.asyncio
    async def test_validation_performance_integration(self):
        """Test validation performance meets requirements."""
        validator = get_universal_validator()

        test_content = {
            "problem": "Solve for x: 2x + 5 = 15",
            "answer": "x = 5",
            "steps": ["2x + 5 = 15", "2x = 10", "x = 5"],
        }

        # Measure validation time
        import time

        start_time = time.time()

        result = await validator.validate_content(test_content, "mathematics")

        validation_time = time.time() - start_time

        # Validation should complete quickly (placeholder validators are fast)
        assert (
            validation_time < 1.0
        ), f"Validation took {validation_time}s, should be < 1s"

        # Result should have timing information
        assert hasattr(result, "validation_time")
        assert result.validation_time >= 0


if __name__ == "__main__":
    pytest.main([__file__])
