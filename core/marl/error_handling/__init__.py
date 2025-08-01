"""
MARL Error Handling Module.

This module provides comprehensive error handling and recovery mechanisms
for the multi-agent reinforcement learning coordination system.
"""

# SynThesisAI Modules
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
    "AgentError",
    "CommunicationError",
    "ConfigurationError",
    "ConsensusError",
    "CoordinationError",
    "ErrorAnalyzer",
    "LearningError",
    "MARLError",
    "MARLErrorHandler",
    "PerformanceError",
    "RecoveryStrategyManager",
]
