"""MARL Integration Module.

This module provides integration adapters and compatibility layers
for connecting the MARL coordination system with the existing
SynThesisAI architecture.
"""

# SynThesisAI Modules
from .marl_adapter import (
    MARLOrchestrationAdapter,
    MARLPipelineIntegration,
    create_marl_integration,
)

__all__ = [
    "MARLOrchestrationAdapter",
    "MARLPipelineIntegration",
    "create_marl_integration",
]
