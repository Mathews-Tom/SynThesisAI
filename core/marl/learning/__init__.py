"""
MARL Learning Module

This module provides learning infrastructure for multi-agent reinforcement learning,
including shared experience management and continuous learning systems.
"""

from .shared_experience import (
    ExperienceConfig,
    ExperienceFilter,
    ExperienceMetadata,
    ExperienceSharing,
    SharedExperienceManager,
    StateNoveltyTracker,
    create_shared_experience_manager,
)

__all__ = [
    "ExperienceConfig",
    "ExperienceMetadata",
    "ExperienceFilter",
    "StateNoveltyTracker",
    "ExperienceSharing",
    "SharedExperienceManager",
    "create_shared_experience_manager",
]
