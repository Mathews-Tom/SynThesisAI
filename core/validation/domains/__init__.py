"""
Domain-specific validators for STREAM content validation.

This module contains validators for each STREAM domain:
- Mathematics
- Science  
- Technology
- Reading
- Engineering
- Arts
"""

# SynThesisAI Modules (Domain validators)
from .mathematics import MathematicsValidator
from .physics import PhysicsValidator
from .science import ScienceValidator

# Validators to be implemented
# from .technology import TechnologyValidator
# from .reading import ReadingValidator
# from .engineering import EngineeringValidator
# from .arts import ArtsValidator

__all__ = [
    "MathematicsValidator",
    "ScienceValidator",
    "PhysicsValidator",
    # Will be populated as more validators are implemented
]
