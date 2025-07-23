"""
DSPy-specific exceptions for error handling.
"""


class DSPyIntegrationError(Exception):
    """Base exception for DSPy integration issues."""

    def __init__(self, message: str, details: dict = None):
        """
        Initialize DSPy integration error.

        Args:
            message: Error message
            details: Optional dictionary with error details
        """
        super().__init__(message)
        self.details = details or {}


class OptimizationFailureError(DSPyIntegrationError):
    """Raised when DSPy optimization fails."""

    def __init__(
        self, message: str, optimizer_type: str = "unknown", details: dict = None
    ):
        """
        Initialize optimization failure error.

        Args:
            message: Error message
            optimizer_type: Type of optimizer that failed
            details: Optional dictionary with error details
        """
        super().__init__(f"[{optimizer_type}] {message}", details)
        self.optimizer_type = optimizer_type


class SignatureValidationError(DSPyIntegrationError):
    """Raised when DSPy signature validation fails."""

    def __init__(self, message: str, signature: str = "unknown", details: dict = None):
        """
        Initialize signature validation error.

        Args:
            message: Error message
            signature: The invalid signature
            details: Optional dictionary with error details
        """
        super().__init__(f"[{signature}] {message}", details)
        self.signature = signature


class CacheCorruptionError(DSPyIntegrationError):
    """Raised when optimization cache is corrupted."""

    def __init__(self, message: str, cache_key: str = "unknown", details: dict = None):
        """
        Initialize cache corruption error.

        Args:
            message: Error message
            cache_key: The corrupted cache key
            details: Optional dictionary with error details
        """
        super().__init__(f"[{cache_key}] {message}", details)
        self.cache_key = cache_key


class TrainingDataError(DSPyIntegrationError):
    """Raised when training data is invalid or insufficient."""

    def __init__(self, message: str, domain: str = "unknown", details: dict = None):
        """
        Initialize training data error.

        Args:
            message: Error message
            domain: The domain with invalid training data
            details: Optional dictionary with error details
        """
        super().__init__(f"[{domain}] {message}", details)
        self.domain = domain


class ModuleInitializationError(DSPyIntegrationError):
    """Raised when DSPy module initialization fails."""

    def __init__(
        self, message: str, module_type: str = "unknown", details: dict = None
    ):
        """
        Initialize module initialization error.

        Args:
            message: Error message
            module_type: Type of module that failed to initialize
            details: Optional dictionary with error details
        """
        super().__init__(f"[{module_type}] {message}", details)
        self.module_type = module_type


class QualityAssessmentError(DSPyIntegrationError):
    """Raised when quality assessment fails."""

    def __init__(self, message: str, domain: str = "unknown", details: dict = None):
        """
        Initialize quality assessment error.

        Args:
            message: Error message
            domain: The domain with quality assessment failure
            details: Optional dictionary with error details
        """
        super().__init__(f"[{domain}] {message}", details)
        self.domain = domain
