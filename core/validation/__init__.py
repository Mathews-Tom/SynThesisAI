"""
Universal validation framework for STREAM domain content validation.

This module provides the foundation for comprehensive validation across
Science, Technology, Reading, Engineering, Arts, and Mathematics domains.
"""

# SynThesisAI Modules
from .base import DomainValidator, QualityMetrics, ValidationResult
from .config import ValidationConfig, ValidationConfigManager, get_config_manager
from .exceptions import DomainValidationError, ValidationError, ValidationTimeoutError
from .orchestrator import UniversalValidator, get_universal_validator

__all__ = [
    "DomainValidator",
    "ValidationResult",
    "QualityMetrics",
    "UniversalValidator",
    "get_universal_validator",
    "ValidationConfig",
    "ValidationConfigManager",
    "get_config_manager",
    "ValidationError",
    "DomainValidationError",
    "ValidationTimeoutError",
]
