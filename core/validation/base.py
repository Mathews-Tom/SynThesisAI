"""
Base classes and data models for the STREAM domain validation system.

This module provides the abstract base classes and data models that all
domain-specific validators must implement.
"""

# Standard Library
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

# SynThesisAI Modules
from .config import ValidationConfig
from .exceptions import ValidationError


@dataclass
class QualityMetrics:
    """Quality metrics for validation results."""

    fidelity_score: float
    utility_score: float
    safety_score: float
    pedagogical_score: float
    domain_specific_score: float
    overall_score: float

    def __post_init__(self):
        """Validate quality metrics after initialization."""
        metrics = [
            self.fidelity_score,
            self.utility_score,
            self.safety_score,
            self.pedagogical_score,
            self.domain_specific_score,
            self.overall_score,
        ]

        for i, metric in enumerate(metrics):
            if not 0 <= metric <= 1:
                metric_names = [
                    "fidelity_score",
                    "utility_score",
                    "safety_score",
                    "pedagogical_score",
                    "domain_specific_score",
                    "overall_score",
                ]
                raise ValueError(
                    f"{metric_names[i]} must be between 0 and 1, got {metric}"
                )


@dataclass
class ValidationResult:
    """Result of domain validation with quality metrics and feedback."""

    domain: str
    is_valid: bool
    quality_score: float
    validation_details: Dict[str, Any]
    confidence_score: float
    feedback: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    quality_metrics: Optional[QualityMetrics] = None
    validation_time: float = 0.0

    def __post_init__(self):
        """Validate result after initialization."""
        if not 0 <= self.quality_score <= 1:
            raise ValueError(
                f"Quality score must be between 0 and 1, got {self.quality_score}"
            )

        if not 0 <= self.confidence_score <= 1:
            raise ValueError(
                f"Confidence score must be between 0 and 1, got {self.confidence_score}"
            )


@dataclass
class SubValidationResult:
    """Result of a sub-component validation within a domain."""

    subdomain: str
    is_valid: bool
    details: Dict[str, Any]
    confidence_score: float = 0.0
    error_message: Optional[str] = None

    def __post_init__(self):
        """Validate sub-result after initialization."""
        if not 0 <= self.confidence_score <= 1:
            raise ValueError(
                f"Confidence score must be between 0 and 1, got {self.confidence_score}"
            )


class DomainValidator(ABC):
    """Abstract base class for domain-specific validators."""

    def __init__(self, domain: str, config: ValidationConfig):
        """
        Initialize domain validator.

        Args:
            domain: The STREAM domain this validator handles
            config: Configuration for this domain validator
        """
        self.domain = domain
        self.config = config
        self.validation_rules = self._load_validation_rules()
        self.quality_thresholds = config.quality_thresholds

    @abstractmethod
    def validate_content(self, content: Dict[str, Any]) -> ValidationResult:
        """
        Validate domain-specific content.

        Args:
            content: Content to validate with domain-specific structure

        Returns:
            ValidationResult with validation outcome and quality metrics

        Raises:
            ValidationError: If validation cannot be performed
        """
        pass

    @abstractmethod
    def calculate_quality_score(self, content: Dict[str, Any]) -> float:
        """
        Calculate domain-specific quality score.

        Args:
            content: Content to assess for quality

        Returns:
            Quality score between 0.0 and 1.0
        """
        pass

    @abstractmethod
    def generate_feedback(self, validation_result: ValidationResult) -> List[str]:
        """
        Generate domain-specific improvement feedback.

        Args:
            validation_result: Result of validation to generate feedback for

        Returns:
            List of feedback messages for content improvement
        """
        pass

    def _load_validation_rules(self) -> Dict[str, Any]:
        """
        Load domain-specific validation rules.

        Returns:
            Dictionary of validation rules for this domain
        """
        # Default implementation returns rules from config
        return self.config.validation_rules.copy()

    def calculate_confidence(self, validation_details: Dict[str, Any]) -> float:
        """
        Calculate confidence score for validation result.

        Args:
            validation_details: Details from validation process

        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Default implementation based on number of successful validations
        if not validation_details:
            return 0.0

        successful_validations = sum(
            1
            for result in validation_details.values()
            if isinstance(result, (bool, SubValidationResult))
            and (result if isinstance(result, bool) else result.is_valid)
        )

        total_validations = len(validation_details)
        return (
            successful_validations / total_validations if total_validations > 0 else 0.0
        )

    def calculate_quality_metrics(
        self, content: Dict[str, Any], validation_details: Dict[str, Any]
    ) -> QualityMetrics:
        """
        Calculate comprehensive quality metrics.

        Args:
            content: Content that was validated
            validation_details: Details from validation process

        Returns:
            QualityMetrics with all quality scores
        """
        # Default implementation - subclasses should override for domain-specific logic
        base_score = self.calculate_quality_score(content)

        # Calculate individual metrics based on validation details
        fidelity_score = self._calculate_fidelity_score(content, validation_details)
        utility_score = self._calculate_utility_score(content, validation_details)
        safety_score = self._calculate_safety_score(content, validation_details)
        pedagogical_score = self._calculate_pedagogical_score(
            content, validation_details
        )

        # Domain-specific score is the base quality score
        domain_specific_score = base_score

        # Overall score is weighted average
        overall_score = (
            0.25 * fidelity_score
            + 0.25 * utility_score
            + 0.25 * safety_score
            + 0.25 * pedagogical_score
        )

        return QualityMetrics(
            fidelity_score=fidelity_score,
            utility_score=utility_score,
            safety_score=safety_score,
            pedagogical_score=pedagogical_score,
            domain_specific_score=domain_specific_score,
            overall_score=overall_score,
        )

    def _calculate_fidelity_score(
        self, content: Dict[str, Any], validation_details: Dict[str, Any]
    ) -> float:
        """Calculate fidelity score (accuracy and correctness)."""
        # Default implementation based on validation success rate
        return self.calculate_confidence(validation_details)

    def _calculate_utility_score(
        self, content: Dict[str, Any], validation_details: Dict[str, Any]
    ) -> float:
        """Calculate utility score (usefulness and relevance)."""
        # Default implementation - can be overridden by subclasses
        return 0.8  # Placeholder value

    def _calculate_safety_score(
        self, content: Dict[str, Any], validation_details: Dict[str, Any]
    ) -> float:
        """Calculate safety score (appropriateness and harm prevention)."""
        # Default implementation - assumes content is safe unless proven otherwise
        safety_issues = validation_details.get("safety_issues", [])
        if isinstance(safety_issues, list):
            return max(0.0, 1.0 - len(safety_issues) * 0.2)
        return 0.9  # Default safe score

    def _calculate_pedagogical_score(
        self, content: Dict[str, Any], validation_details: Dict[str, Any]
    ) -> float:
        """Calculate pedagogical score (educational value and effectiveness)."""
        # Default implementation - can be overridden by subclasses
        return 0.75  # Placeholder value

    def validate_with_timeout(self, content: Dict[str, Any]) -> ValidationResult:
        """
        Validate content with timeout protection.

        Args:
            content: Content to validate

        Returns:
            ValidationResult with validation outcome

        Raises:
            ValidationTimeoutError: If validation exceeds timeout
            ValidationError: If validation fails
        """
        start_time = time.time()

        try:
            result = self.validate_content(content)
            result.validation_time = time.time() - start_time
            return result

        except Exception as e:
            validation_time = time.time() - start_time

            if validation_time > self.config.timeout_seconds:
                from .exceptions import ValidationTimeoutError

                raise ValidationTimeoutError(
                    self.config.timeout_seconds, f"{self.domain} validation"
                ) from e

            # Re-raise the original exception
            raise ValidationError(
                f"Validation failed for domain {self.domain}: {str(e)}"
            ) from e
