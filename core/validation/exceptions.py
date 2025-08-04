"""
Validation-specific exceptions for the STREAM domain validation system.

This module defines custom exceptions for validation errors, timeouts,
and domain-specific validation failures.
"""

# Standard Library
from typing import Any, Dict, Optional


class ValidationError(Exception):
    """Base validation error for all validation-related exceptions."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """
        Initialize validation error.

        Args:
            message: Error message describing the validation failure
            details: Optional dictionary with additional error context
        """
        super().__init__(message)
        self.details = details or {}


class DomainValidationError(ValidationError):
    """Domain-specific validation error with domain context."""

    def __init__(
        self, domain: str, message: str, details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize domain validation error.

        Args:
            domain: The STREAM domain where validation failed
            message: Error message describing the validation failure
            details: Optional dictionary with additional error context
        """
        self.domain = domain
        super().__init__(f"Validation error in {domain}: {message}", details)


class ValidationTimeoutError(ValidationError):
    """Validation timeout error when validation exceeds time limits."""

    def __init__(self, timeout_seconds: float, operation: str = "validation"):
        """
        Initialize validation timeout error.

        Args:
            timeout_seconds: The timeout limit that was exceeded
            operation: The validation operation that timed out
        """
        self.timeout_seconds = timeout_seconds
        self.operation = operation
        super().__init__(
            f"{operation} timed out after {timeout_seconds} seconds",
            {"timeout_seconds": timeout_seconds, "operation": operation},
        )


class ValidationConfigError(ValidationError):
    """Configuration error for validation system setup."""

    def __init__(self, config_key: str, message: str):
        """
        Initialize validation configuration error.

        Args:
            config_key: The configuration key that caused the error
            message: Error message describing the configuration issue
        """
        self.config_key = config_key
        super().__init__(
            f"Configuration error for '{config_key}': {message}",
            {"config_key": config_key},
        )


class ValidationCacheError(ValidationError):
    """Cache-related validation error."""

    def __init__(self, cache_operation: str, message: str):
        """
        Initialize validation cache error.

        Args:
            cache_operation: The cache operation that failed (get, store, invalidate)
            message: Error message describing the cache failure
        """
        self.cache_operation = cache_operation
        super().__init__(
            f"Cache {cache_operation} failed: {message}",
            {"cache_operation": cache_operation},
        )
