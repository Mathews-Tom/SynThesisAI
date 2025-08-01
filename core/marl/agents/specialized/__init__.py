"""
Specialized MARL Agents.

This module provides specialized reinforcement learning agents for multi-agent
coordination in the SynThesisAI platform, including Generator, Validator, and
Curriculum agents with domain-specific capabilities.
"""

# SynThesisAI Modules
from .curriculum_agent import CurriculumRLAgent
from .generator_agent import GeneratorRLAgent
from .validator_agent import ValidatorRLAgent

__all__ = ["GeneratorRLAgent", "ValidatorRLAgent", "CurriculumRLAgent"]
