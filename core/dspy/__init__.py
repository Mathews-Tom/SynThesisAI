"""
DSPy Integration for SynThesisAI

This module provides DSPy (Declarative Self-improving Python) integration for the
SynThesisAI platform, enabling automated prompt optimization and self-improving
content generation pipelines.
"""

import logging

# Configure module logger
logger = logging.getLogger(__name__)

# Check if DSPy is available
try:
    import dspy

    DSPY_AVAILABLE = True
    logger.info("DSPy version %s is available", getattr(dspy, "__version__", "unknown"))
except ImportError:
    DSPY_AVAILABLE = False
    logger.warning("DSPy is not available, using mock implementations")

# Import core components
from .base_module import STREAMContentGenerator
from .cache import OptimizationCache, get_optimization_cache
from .cache_invalidation import (
    CacheInvalidationManager,
    get_invalidation_manager,
    invalidate_by_config_change,
    invalidate_by_domain,
    run_cache_maintenance,
)
from .cache_monitoring import (
    CachePerformanceMonitor,
    collect_cache_metrics,
    get_cache_monitor,
    get_cache_performance_report,
)
from .config import (
    DSPyConfig,
    DSPyModuleConfig,
    OptimizationResult,
    TrainingExample,
    get_dspy_config,
)
from .exceptions import (
    CacheCorruptionError,
    DSPyIntegrationError,
    ModuleInitializationError,
    OptimizationFailureError,
    QualityAssessmentError,
    SignatureValidationError,
    TrainingDataError,
)
from .optimization_engine import DSPyOptimizationEngine, TrainingDataManager
from .signature_registry import SignatureRegistry, get_signature_registry
from .signatures import (
    SignatureManager,
    create_custom_signature,
    get_all_domains,
    get_domain_signature,
    get_signature_types,
    validate_signature,
)

# Define version
__version__ = "0.1.0"

# Define public API
__all__ = [
    # Core components
    "STREAMContentGenerator",
    "DSPyOptimizationEngine",
    "TrainingDataManager",
    "SignatureManager",
    "OptimizationCache",
    "SignatureRegistry",
    "CacheInvalidationManager",
    "CachePerformanceMonitor",
    # Configuration
    "get_dspy_config",
    "DSPyConfig",
    "DSPyModuleConfig",
    "OptimizationResult",
    "TrainingExample",
    # Exceptions
    "DSPyIntegrationError",
    "OptimizationFailureError",
    "SignatureValidationError",
    "CacheCorruptionError",
    "TrainingDataError",
    "ModuleInitializationError",
    "QualityAssessmentError",
    # Utility functions
    "get_domain_signature",
    "validate_signature",
    "create_custom_signature",
    "get_all_domains",
    "get_signature_types",
    "get_optimization_cache",
    "get_signature_registry",
    "get_invalidation_manager",
    "invalidate_by_config_change",
    "invalidate_by_domain",
    "run_cache_maintenance",
    "get_cache_monitor",
    "collect_cache_metrics",
    "get_cache_performance_report",
    # Status
    "DSPY_AVAILABLE",
    "__version__",
]
