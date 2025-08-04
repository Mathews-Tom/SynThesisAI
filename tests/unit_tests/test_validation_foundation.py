"""
Unit tests for the universal validation framework foundation.

This module tests the base classes, configuration management, and
orchestration components of the STREAM domain validation system.
"""

# Standard Library
import asyncio
import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import Mock, patch

# Third-Party Library
import pytest

# SynThesisAI Modules
from core.validation.base import (
    DomainValidator,
    QualityMetrics,
    SubValidationResult,
    ValidationResult,
)
from core.validation.config import (
    UniversalValidationConfig,
    ValidationConfig,
    ValidationConfigManager,
)
from core.validation.exceptions import (
    DomainValidationError,
    ValidationConfigError,
    ValidationError,
    ValidationTimeoutError,
)
from core.validation.orchestrator import PlaceholderValidator, UniversalValidator


class TestQualityMetrics:
    """Test cases for QualityMetrics data class."""

    def test_valid_quality_metrics(self):
        """Test creation of valid quality metrics."""
        metrics = QualityMetrics(
            fidelity_score=0.9,
            utility_score=0.8,
            safety_score=0.95,
            pedagogical_score=0.85,
            domain_specific_score=0.88,
            overall_score=0.87,
        )

        assert metrics.fidelity_score == 0.9
        assert metrics.utility_score == 0.8
        assert metrics.safety_score == 0.95
        assert metrics.pedagogical_score == 0.85
        assert metrics.domain_specific_score == 0.88
        assert metrics.overall_score == 0.87

    def test_invalid_quality_metrics_range(self):
        """Test that quality metrics must be between 0 and 1."""
        with pytest.raises(ValueError, match="fidelity_score must be between 0 and 1"):
            QualityMetrics(
                fidelity_score=1.5,  # Invalid: > 1
                utility_score=0.8,
                safety_score=0.95,
                pedagogical_score=0.85,
                domain_specific_score=0.88,
                overall_score=0.87,
            )

        with pytest.raises(ValueError, match="utility_score must be between 0 and 1"):
            QualityMetrics(
                fidelity_score=0.9,
                utility_score=-0.1,  # Invalid: < 0
                safety_score=0.95,
                pedagogical_score=0.85,
                domain_specific_score=0.88,
                overall_score=0.87,
            )


class TestValidationResult:
    """Test cases for ValidationResult data class."""

    def test_valid_validation_result(self):
        """Test creation of valid validation result."""
        result = ValidationResult(
            domain="mathematics",
            is_valid=True,
            quality_score=0.85,
            validation_details={"test": "passed"},
            confidence_score=0.9,
            feedback=["Good work"],
            quality_metrics=QualityMetrics(0.9, 0.8, 0.95, 0.85, 0.88, 0.87),
        )

        assert result.domain == "mathematics"
        assert result.is_valid is True
        assert result.quality_score == 0.85
        assert result.confidence_score == 0.9
        assert len(result.feedback) == 1
        assert result.quality_metrics is not None

    def test_invalid_quality_score_range(self):
        """Test that quality score must be between 0 and 1."""
        with pytest.raises(ValueError, match="Quality score must be between 0 and 1"):
            ValidationResult(
                domain="mathematics",
                is_valid=True,
                quality_score=1.5,  # Invalid: > 1
                validation_details={},
                confidence_score=0.9,
            )

    def test_invalid_confidence_score_range(self):
        """Test that confidence score must be between 0 and 1."""
        with pytest.raises(
            ValueError, match="Confidence score must be between 0 and 1"
        ):
            ValidationResult(
                domain="mathematics",
                is_valid=True,
                quality_score=0.85,
                validation_details={},
                confidence_score=-0.1,  # Invalid: < 0
            )


class TestValidationConfig:
    """Test cases for ValidationConfig data class."""

    def test_valid_validation_config(self):
        """Test creation of valid validation configuration."""
        config = ValidationConfig(
            domain="mathematics",
            quality_thresholds={"accuracy": 0.8, "completeness": 0.7},
            validation_rules={"rule1": "value1"},
            timeout_seconds=30,
            max_retries=3,
        )

        assert config.domain == "mathematics"
        assert config.quality_thresholds["accuracy"] == 0.8
        assert config.timeout_seconds == 30
        assert config.max_retries == 3

    def test_invalid_domain_empty(self):
        """Test that domain cannot be empty."""
        with pytest.raises(ValidationConfigError, match="Domain cannot be empty"):
            ValidationConfig(domain="")

    def test_invalid_timeout_negative(self):
        """Test that timeout must be positive."""
        with pytest.raises(ValidationConfigError, match="Timeout must be positive"):
            ValidationConfig(domain="test", timeout_seconds=-1)

    def test_invalid_quality_threshold_range(self):
        """Test that quality thresholds must be between 0 and 1."""
        with pytest.raises(
            ValidationConfigError, match="Threshold 1.5 must be between 0 and 1"
        ):
            ValidationConfig(
                domain="test", quality_thresholds={"accuracy": 1.5}  # Invalid: > 1
            )


class TestUniversalValidationConfig:
    """Test cases for UniversalValidationConfig data class."""

    def test_valid_universal_config(self):
        """Test creation of valid universal configuration."""
        config = UniversalValidationConfig(
            enabled_domains=["mathematics", "science"],
            global_timeout_seconds=60,
            parallel_validation=True,
            max_concurrent_validations=3,
        )

        assert len(config.enabled_domains) == 2
        assert config.global_timeout_seconds == 60
        assert config.parallel_validation is True
        assert config.max_concurrent_validations == 3

    def test_invalid_domains(self):
        """Test that only valid domains are accepted."""
        with pytest.raises(ValidationConfigError, match="Invalid domains"):
            UniversalValidationConfig(enabled_domains=["mathematics", "invalid_domain"])

    def test_invalid_aggregation_method(self):
        """Test that only valid aggregation methods are accepted."""
        with pytest.raises(ValidationConfigError, match="Invalid aggregation method"):
            UniversalValidationConfig(quality_aggregation_method="invalid_method")


class MockValidator(DomainValidator):
    """Mock validator for testing purposes."""

    def validate_content(self, content: Dict[str, Any]) -> ValidationResult:
        """Mock validation that returns success."""
        return ValidationResult(
            domain=self.domain,
            is_valid=True,
            quality_score=0.8,
            validation_details={"mock": True},
            confidence_score=0.9,
        )

    def calculate_quality_score(self, content: Dict[str, Any]) -> float:
        """Mock quality score calculation."""
        return 0.8

    def generate_feedback(self, validation_result: ValidationResult) -> List[str]:
        """Mock feedback generation."""
        return ["Mock feedback"]


class TestDomainValidator:
    """Test cases for DomainValidator abstract base class."""

    def test_validator_initialization(self):
        """Test validator initialization with configuration."""
        config = ValidationConfig(
            domain="test", quality_thresholds={"accuracy": 0.8}, timeout_seconds=30
        )

        validator = MockValidator("test", config)

        assert validator.domain == "test"
        assert validator.config == config
        assert validator.quality_thresholds["accuracy"] == 0.8

    def test_calculate_confidence_empty_details(self):
        """Test confidence calculation with empty validation details."""
        config = ValidationConfig(domain="test")
        validator = MockValidator("test", config)

        confidence = validator.calculate_confidence({})
        assert confidence == 0.0

    def test_calculate_confidence_mixed_results(self):
        """Test confidence calculation with mixed validation results."""
        config = ValidationConfig(domain="test")
        validator = MockValidator("test", config)

        validation_details = {
            "test1": True,
            "test2": False,
            "test3": SubValidationResult("sub", True, {}, 0.9),
            "test4": SubValidationResult("sub", False, {}, 0.5),
        }

        confidence = validator.calculate_confidence(validation_details)
        assert confidence == 0.5  # 2 out of 4 successful

    def test_quality_metrics_calculation(self):
        """Test quality metrics calculation."""
        config = ValidationConfig(domain="test")
        validator = MockValidator("test", config)

        content = {"test": "content"}
        validation_details = {"test": True}

        metrics = validator.calculate_quality_metrics(content, validation_details)

        assert isinstance(metrics, QualityMetrics)
        assert 0 <= metrics.fidelity_score <= 1
        assert 0 <= metrics.utility_score <= 1
        assert 0 <= metrics.safety_score <= 1
        assert 0 <= metrics.pedagogical_score <= 1
        assert 0 <= metrics.domain_specific_score <= 1
        assert 0 <= metrics.overall_score <= 1


class TestValidationConfigManager:
    """Test cases for ValidationConfigManager."""

    def test_config_manager_initialization(self):
        """Test configuration manager initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)
            manager = ValidationConfigManager(config_dir)

            assert manager.config_dir == config_dir

    def test_load_default_domain_config(self):
        """Test loading default domain configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)
            manager = ValidationConfigManager(config_dir)

            config = manager.load_domain_config("mathematics")

            assert config.domain == "mathematics"
            assert "fidelity_score" in config.quality_thresholds
            assert config.timeout_seconds > 0

    def test_load_existing_domain_config(self):
        """Test loading existing domain configuration from file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)
            config_dir.mkdir(exist_ok=True)

            # Create test configuration file
            config_data = {
                "quality_thresholds": {"accuracy": 0.9},
                "timeout_seconds": 45,
                "max_retries": 5,
            }

            config_file = config_dir / "mathematics_validation.json"
            with open(config_file, "w", encoding="utf-8") as f:
                json.dump(config_data, f)

            manager = ValidationConfigManager(config_dir)
            config = manager.load_domain_config("mathematics")

            assert config.domain == "mathematics"
            assert config.quality_thresholds["accuracy"] == 0.9
            assert config.timeout_seconds == 45
            assert config.max_retries == 5

    def test_load_universal_config(self):
        """Test loading universal validation configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)
            manager = ValidationConfigManager(config_dir)

            config = manager.load_universal_config()

            assert isinstance(config, UniversalValidationConfig)
            assert len(config.enabled_domains) > 0
            assert config.global_timeout_seconds > 0


class TestUniversalValidator:
    """Test cases for UniversalValidator orchestrator."""

    @pytest.fixture
    def mock_config_manager(self):
        """Create mock configuration manager."""
        manager = Mock(spec=ValidationConfigManager)

        # Mock universal config
        universal_config = UniversalValidationConfig(
            enabled_domains=["mathematics", "science"],
            parallel_validation=False,  # Use sequential for simpler testing
            fallback_on_error=True,
        )
        manager.load_universal_config.return_value = universal_config

        # Mock domain configs
        def mock_load_domain_config(domain):
            return ValidationConfig(domain=domain)

        manager.load_domain_config.side_effect = mock_load_domain_config

        return manager

    def test_universal_validator_initialization(self, mock_config_manager):
        """Test universal validator initialization."""
        with patch(
            "core.validation.orchestrator.get_config_manager",
            return_value=mock_config_manager,
        ):
            validator = UniversalValidator()

            assert len(validator.domain_validators) == 2
            assert "mathematics" in validator.domain_validators
            assert "science" in validator.domain_validators

    @pytest.mark.asyncio
    async def test_validate_content_success(self, mock_config_manager):
        """Test successful content validation."""
        with patch(
            "core.validation.orchestrator.get_config_manager",
            return_value=mock_config_manager,
        ):
            validator = UniversalValidator()

            content = {"problem": "2 + 2 = ?", "answer": "4"}
            result = await validator.validate_content(content, "mathematics")

            assert isinstance(result, ValidationResult)
            assert result.domain == "mathematics"
            assert result.is_valid is True

    @pytest.mark.asyncio
    async def test_validate_content_unsupported_domain(self, mock_config_manager):
        """Test validation with unsupported domain."""
        with patch(
            "core.validation.orchestrator.get_config_manager",
            return_value=mock_config_manager,
        ):
            validator = UniversalValidator()

            content = {"test": "content"}

            with pytest.raises(ValidationError, match="Unsupported domain"):
                await validator.validate_content(content, "unsupported_domain")

    @pytest.mark.asyncio
    async def test_validate_multiple_domains_sequential(self, mock_config_manager):
        """Test validation across multiple domains sequentially."""
        with patch(
            "core.validation.orchestrator.get_config_manager",
            return_value=mock_config_manager,
        ):
            validator = UniversalValidator()

            content = {"test": "content"}
            domains = ["mathematics", "science"]

            results = await validator.validate_multiple_domains(content, domains)

            assert len(results) == 2
            assert "mathematics" in results
            assert "science" in results
            assert all(
                isinstance(result, ValidationResult) for result in results.values()
            )

    def test_get_supported_domains(self, mock_config_manager):
        """Test getting list of supported domains."""
        with patch(
            "core.validation.orchestrator.get_config_manager",
            return_value=mock_config_manager,
        ):
            validator = UniversalValidator()

            domains = validator.get_supported_domains()

            assert "mathematics" in domains
            assert "science" in domains

    def test_is_domain_supported(self, mock_config_manager):
        """Test checking if domain is supported."""
        with patch(
            "core.validation.orchestrator.get_config_manager",
            return_value=mock_config_manager,
        ):
            validator = UniversalValidator()

            assert validator.is_domain_supported("mathematics") is True
            assert validator.is_domain_supported("unsupported") is False


class TestPlaceholderValidator:
    """Test cases for PlaceholderValidator."""

    def test_placeholder_validator_validation(self):
        """Test placeholder validator always passes validation."""
        config = ValidationConfig(domain="test")
        validator = PlaceholderValidator("test", config)

        content = {"test": "content"}
        result = validator.validate_content(content)

        assert result.domain == "test"
        assert result.is_valid is True
        assert result.quality_score == 0.8
        assert "placeholder" in result.validation_details

    def test_placeholder_validator_quality_score(self):
        """Test placeholder validator quality score calculation."""
        config = ValidationConfig(domain="test")
        validator = PlaceholderValidator("test", config)

        content = {"test": "content"}
        score = validator.calculate_quality_score(content)

        assert score == 0.8

    def test_placeholder_validator_feedback(self):
        """Test placeholder validator feedback generation."""
        config = ValidationConfig(domain="test")
        validator = PlaceholderValidator("test", config)

        result = ValidationResult(
            domain="test",
            is_valid=True,
            quality_score=0.8,
            validation_details={},
            confidence_score=0.5,
        )

        feedback = validator.generate_feedback(result)

        assert len(feedback) == 1
        assert "test" in feedback[0]


class TestValidationExceptions:
    """Test cases for validation exceptions."""

    def test_validation_error(self):
        """Test basic validation error."""
        error = ValidationError("Test error", {"key": "value"})

        assert str(error) == "Test error"
        assert error.details["key"] == "value"

    def test_domain_validation_error(self):
        """Test domain-specific validation error."""
        error = DomainValidationError(
            "mathematics", "Math error", {"equation": "invalid"}
        )

        assert "mathematics" in str(error)
        assert "Math error" in str(error)
        assert error.domain == "mathematics"
        assert error.details["equation"] == "invalid"

    def test_validation_timeout_error(self):
        """Test validation timeout error."""
        error = ValidationTimeoutError(30.0, "test validation")

        assert "30.0 seconds" in str(error)
        assert "test validation" in str(error)
        assert error.timeout_seconds == 30.0
        assert error.operation == "test validation"

    def test_validation_config_error(self):
        """Test validation configuration error."""
        error = ValidationConfigError("timeout", "Invalid timeout value")

        assert "timeout" in str(error)
        assert "Invalid timeout value" in str(error)
        assert error.config_key == "timeout"


if __name__ == "__main__":
    pytest.main([__file__])
