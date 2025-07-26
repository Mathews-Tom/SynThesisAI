"""
MARL Error Handling Module.

This module provides comprehensive error handling and recovery mechanisms
for the multi-agent reinforcement learning coordination system.
"""

from .error_analyzer import ErrorAnalyzer
from .error_handler import MARLErrorHandler
from .error_types import (
    AgentError,
    CommunicationError,
    ConfigurationError,
    ConsensusError,
    CoordinationError,
    LearningError,
    MARLError,
    PerformanceError,
)
from .recovery_strategies import RecoveryStrategyManager

__all__ = [
    "MARLError",
    "AgentError",
    "CoordinationError",
    "LearningError",
    "CommunicationError",
    "ConsensusError",
    "PerformanceError",
    "ConfigurationError",
    "MARLErrorHandler",
    "ErrorAnalyzer",
    "RecoveryStrategyManager",
]
