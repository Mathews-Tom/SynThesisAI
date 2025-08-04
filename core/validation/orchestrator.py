"""
Universal validation orchestrator for coordinating domain-specific validators.

This module provides the UniversalValidator class that coordinates validation
across all STREAM domains and aggregates results.
"""

# Standard Library
import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

# SynThesisAI Modules
from .base import DomainValidator, QualityMetrics, ValidationResult
from .config import (
    UniversalValidationConfig,
    ValidationConfigManager,
    get_config_manager,
)
from .exceptions import ValidationError, ValidationTimeoutError

logger = logging.getLogger(__name__)


class UniversalValidator:
    """
    Universal validator that coordinates domain-specific validation.

    This class manages validation across all STREAM domains, providing
    unified interfaces and result aggregation.
    """

    def __init__(self, config_manager: Optional[ValidationConfigManager] = None):
        """
        Initialize universal validator.

        Args:
            config_manager: Configuration manager for validation settings
        """
        self.config_manager = config_manager or get_config_manager()
        self.universal_config = self.config_manager.load_universal_config()
        self.domain_validators: Dict[str, DomainValidator] = {}
        self.logger = logging.getLogger(__name__ + ".UniversalValidator")

        # Initialize domain validators
        self._initialize_domain_validators()

    def _initialize_domain_validators(self):
        """Initialize validators for all enabled domains."""
        for domain in self.universal_config.enabled_domains:
            try:
                # Import domain-specific validator
                validator_class = self._get_validator_class(domain)
                domain_config = self.config_manager.load_domain_config(domain)

                # Create validator instance
                self.domain_validators[domain] = validator_class(domain, domain_config)

                self.logger.info("Initialized validator for domain %s", domain)

            except Exception as e:
                self.logger.error(
                    "Failed to initialize validator for domain %s: %s", domain, str(e)
                )
                if not self.universal_config.fallback_on_error:
                    raise ValidationError(
                        f"Failed to initialize validator for domain {domain}"
                    ) from e

    def _get_validator_class(self, domain: str) -> type:
        """
        Get validator class for a specific domain.

        Args:
            domain: The STREAM domain to get validator for

        Returns:
            Validator class for the domain

        Raises:
            ValidationError: If validator class cannot be found
        """
        # Import domain-specific validators
        try:
            if domain == "mathematics":
                from .domains.mathematics import MathematicsValidator

                return MathematicsValidator
            elif domain == "science":
                from .domains.science import ScienceValidator

                return ScienceValidator
            elif domain == "technology":
                from .domains.technology import TechnologyValidator

                return TechnologyValidator
            elif domain == "reading":
                from .domains.reading import ReadingValidator

                return ReadingValidator
            elif domain == "engineering":
                from .domains.engineering import EngineeringValidator

                return EngineeringValidator
            elif domain == "arts":
                from .domains.arts import ArtsValidator

                return ArtsValidator
            else:
                raise ValidationError(f"Unknown domain: {domain}")

        except ImportError as e:
            # For now, return a placeholder validator
            self.logger.warning(
                "Domain validator for %s not yet implemented, using placeholder", domain
            )
            return PlaceholderValidator

    async def validate_content(
        self, content: Dict[str, Any], domain: str
    ) -> ValidationResult:
        """
        Validate content for a specific domain.

        Args:
            content: Content to validate
            domain: STREAM domain for validation

        Returns:
            ValidationResult with validation outcome

        Raises:
            ValidationError: If domain is not supported or validation fails
        """
        if domain not in self.domain_validators:
            raise ValidationError(f"Unsupported domain: {domain}")

        validator = self.domain_validators[domain]

        try:
            # Perform domain-specific validation with timeout
            result = await asyncio.to_thread(validator.validate_with_timeout, content)

            # Add universal quality checks
            result = await self._add_universal_checks(result, content)

            self.logger.info(
                "Validation completed for domain %s: valid=%s, score=%.2f",
                domain,
                result.is_valid,
                result.quality_score,
            )

            return result

        except asyncio.TimeoutError as e:
            raise ValidationTimeoutError(
                self.universal_config.global_timeout_seconds, f"{domain} validation"
            ) from e
        except Exception as e:
            error_msg = f"Validation failed for domain {domain}: {str(e)}"
            self.logger.error(error_msg)

            if self.universal_config.fallback_on_error:
                # Return a failed validation result instead of raising
                return ValidationResult(
                    domain=domain,
                    is_valid=False,
                    quality_score=0.0,
                    validation_details={"error": str(e)},
                    confidence_score=0.0,
                    feedback=[f"Validation failed: {str(e)}"],
                )
            else:
                raise ValidationError(error_msg) from e

    async def validate_multiple_domains(
        self, content: Dict[str, Any], domains: List[str]
    ) -> Dict[str, ValidationResult]:
        """
        Validate content across multiple domains.

        Args:
            content: Content to validate
            domains: List of STREAM domains for validation

        Returns:
            Dictionary mapping domains to their validation results
        """
        if self.universal_config.parallel_validation:
            return await self._validate_parallel(content, domains)
        else:
            return await self._validate_sequential(content, domains)

    async def _validate_parallel(
        self, content: Dict[str, Any], domains: List[str]
    ) -> Dict[str, ValidationResult]:
        """Validate content across domains in parallel."""
        # Limit concurrent validations
        semaphore = asyncio.Semaphore(self.universal_config.max_concurrent_validations)

        async def validate_with_semaphore(domain: str) -> tuple[str, ValidationResult]:
            async with semaphore:
                result = await self.validate_content(content, domain)
                return domain, result

        # Create validation tasks
        tasks = [validate_with_semaphore(domain) for domain in domains]

        # Execute with global timeout
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=self.universal_config.global_timeout_seconds,
            )

            # Process results
            validation_results = {}
            for result in results:
                if isinstance(result, Exception):
                    self.logger.error("Validation task failed: %s", str(result))
                    continue

                domain, validation_result = result
                validation_results[domain] = validation_result

            return validation_results

        except asyncio.TimeoutError as e:
            raise ValidationTimeoutError(
                self.universal_config.global_timeout_seconds, "multi-domain validation"
            ) from e

    async def _validate_sequential(
        self, content: Dict[str, Any], domains: List[str]
    ) -> Dict[str, ValidationResult]:
        """Validate content across domains sequentially."""
        validation_results = {}

        for domain in domains:
            try:
                result = await self.validate_content(content, domain)
                validation_results[domain] = result
            except Exception as e:
                self.logger.error(
                    "Sequential validation failed for domain %s: %s", domain, str(e)
                )
                if not self.universal_config.fallback_on_error:
                    raise

        return validation_results

    async def _add_universal_checks(
        self, result: ValidationResult, content: Dict[str, Any]
    ) -> ValidationResult:
        """
        Add universal quality checks to domain-specific validation.

        Args:
            result: Domain-specific validation result
            content: Original content that was validated

        Returns:
            Enhanced validation result with universal checks
        """
        # Add universal safety checks
        safety_score = await self._check_content_safety(content)

        # Add universal pedagogical value assessment
        pedagogical_score = await self._assess_pedagogical_value(content)

        # Update validation details
        result.validation_details["universal_safety"] = safety_score
        result.validation_details["universal_pedagogical"] = pedagogical_score

        # Recalculate quality score with universal factors
        result.quality_score = self._calculate_universal_quality_score(
            result.quality_score, safety_score, pedagogical_score
        )

        return result

    async def _check_content_safety(self, content: Dict[str, Any]) -> float:
        """
        Perform universal content safety checks.

        Args:
            content: Content to check for safety

        Returns:
            Safety score between 0.0 and 1.0
        """
        # Placeholder implementation - should include:
        # - Inappropriate content detection
        # - Bias detection
        # - Age-appropriateness assessment
        # - Cultural sensitivity checks

        # For now, return a high safety score
        return 0.95

    async def _assess_pedagogical_value(self, content: Dict[str, Any]) -> float:
        """
        Assess universal pedagogical value of content.

        Args:
            content: Content to assess

        Returns:
            Pedagogical value score between 0.0 and 1.0
        """
        # Placeholder implementation - should include:
        # - Learning objective alignment
        # - Cognitive load assessment
        # - Engagement potential
        # - Educational effectiveness

        # For now, return a moderate pedagogical score
        return 0.8

    def _calculate_universal_quality_score(
        self, domain_score: float, safety_score: float, pedagogical_score: float
    ) -> float:
        """
        Calculate universal quality score combining domain and universal factors.

        Args:
            domain_score: Domain-specific quality score
            safety_score: Universal safety score
            pedagogical_score: Universal pedagogical score

        Returns:
            Combined quality score between 0.0 and 1.0
        """
        # Weighted combination of scores
        weights = {"domain": 0.6, "safety": 0.2, "pedagogical": 0.2}

        return (
            weights["domain"] * domain_score
            + weights["safety"] * safety_score
            + weights["pedagogical"] * pedagogical_score
        )

    def get_supported_domains(self) -> List[str]:
        """
        Get list of supported validation domains.

        Returns:
            List of supported STREAM domains
        """
        return list(self.domain_validators.keys())

    def is_domain_supported(self, domain: str) -> bool:
        """
        Check if a domain is supported for validation.

        Args:
            domain: Domain to check

        Returns:
            True if domain is supported
        """
        return domain in self.domain_validators


class PlaceholderValidator(DomainValidator):
    """Placeholder validator for domains not yet implemented."""

    def validate_content(self, content: Dict[str, Any]) -> ValidationResult:
        """Placeholder validation that always passes."""
        return ValidationResult(
            domain=self.domain,
            is_valid=True,
            quality_score=0.8,
            validation_details={"placeholder": True},
            confidence_score=0.5,
            feedback=[f"Using placeholder validator for {self.domain} domain"],
        )

    def calculate_quality_score(self, content: Dict[str, Any]) -> float:
        """Placeholder quality score calculation."""
        return 0.8

    def generate_feedback(self, validation_result: ValidationResult) -> List[str]:
        """Placeholder feedback generation."""
        return [f"Placeholder feedback for {self.domain} domain"]


# Global universal validator instance
_universal_validator = None


def get_universal_validator() -> UniversalValidator:
    """Get the global universal validator instance."""
    global _universal_validator
    if _universal_validator is None:
        _universal_validator = UniversalValidator()
    return _universal_validator
