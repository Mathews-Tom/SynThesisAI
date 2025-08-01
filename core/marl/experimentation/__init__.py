"""
MARL Experimentation Framework.

This package provides comprehensive experimentation capabilities for MARL systems,
including A/B testing, algorithm comparison, and research data collection.
"""

from .ab_testing import ABTestManager, ABTestManagerFactory
from .experiment_manager import ExperimentManager, ExperimentManagerFactory
from .experiment_runner import ExperimentRunner, ExperimentRunnerFactory
from .research_logger import ResearchLogger, ResearchLoggerFactory

__all__ = [
    "ExperimentManager",
    "ExperimentManagerFactory",
    "ExperimentRunner",
    "ExperimentRunnerFactory",
    "ABTestManager",
    "ABTestManagerFactory",
    "ResearchLogger",
    "ResearchLoggerFactory",
]
