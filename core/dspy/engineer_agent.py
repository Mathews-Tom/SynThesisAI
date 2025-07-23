"""
DSPy Engineer Agent

This module implements the DSPyEngineerAgent class, which extends the base Agent
class and uses the STREAMContentGenerator for DSPy-powered content generation.
"""

import logging
from typing import Any, Dict, Optional

from core.agents import Agent, EngineerAgent
from core.dspy.base_module import STREAMContentGenerator
from core.dspy.config import get_dspy_config
from core.dspy.exceptions import DSPyIntegrationError
from core.dspy.optimization_engine import DSPyOptimizationEngine
from utils.exceptions import ValidationError
from utils.json_utils import safe_json_parse

logger = logging.getLogger(__name__)


class DSPyEngineerAgent(Agent):
    """
    DSPy-powered Engineer Agent for content generation.

    This class extends the base Agent class and uses the STREAMContentGenerator
    for DSPy-powered content generation with ChainOfThought reasoning.
    """

    def __init__(self):
        """Initialize the DSPyEngineerAgent."""
        super().__init__("engineer", "engineer_model")
        self.dspy_module = None
        self.optimization_engine = DSPyOptimizationEngine()
        self.dspy_config = get_dspy_config()
        self.logger = logging.getLogger(__name__ + ".DSPyEngineerAgent")
        self.logger.info("Initialized DSPyEngineerAgent")

    def get_system_prompt(self, difficulty_level: Optional[str] = None) -> str:
        """
        Get the system prompt for problem generation.

        Args:
            difficulty_level: Optional difficulty level specification

        Returns:
            System prompt string
        """
        # Use the same system prompt as the original EngineerAgent
        # This maintains backward compatibility
        engineer_agent = EngineerAgent()
        return engineer_agent.get_system_prompt(difficulty_level)

    def initialize_dspy_module(self, domain: str) -> None:
        """
        Initialize DSPy module for specific domain.

        Args:
            domain: The domain for content generation (e.g., 'mathematics')

        Raises:
            DSPyIntegrationError: If initialization fails
        """
        try:
            # Get domain signature
            signature = self._get_domain_signature(domain)

            # Create STREAMContentGenerator for the domain
            self.dspy_module = STREAMContentGenerator(domain, signature)

            # Optimize the module if enabled
            if self.dspy_config.is_enabled():
                self.dspy_module = self.optimization_engine.optimize_for_domain(
                    self.dspy_module, self._get_quality_requirements(domain)
                )

            self.logger.info("Initialized DSPy module for domain: %s", domain)

        except Exception as e:
            error_msg = (
                f"Failed to initialize DSPy module for domain {domain}: {str(e)}"
            )
            self.logger.error(error_msg)
            raise DSPyIntegrationError(
                error_msg, details={"domain": domain, "error": str(e)}
            )

    def _get_domain_signature(self, domain: str) -> str:
        """
        Get the appropriate signature for a domain.

        Args:
            domain: The domain for content generation

        Returns:
            Signature string for the domain
        """
        # Map subjects to STREAM domains
        domain_mapping = {
            "mathematics": "mathematics",
            "algebra": "mathematics",
            "calculus": "mathematics",
            "geometry": "mathematics",
            "statistics": "mathematics",
            "probability": "mathematics",
            "number theory": "mathematics",
            "discrete mathematics": "mathematics",
            "linear algebra": "mathematics",
            "physics": "science",
            "chemistry": "science",
            "biology": "science",
            "computer science": "technology",
            "programming": "technology",
            "algorithms": "technology",
            "data structures": "technology",
            "literature": "reading",
            "language": "reading",
            "mechanical engineering": "engineering",
            "electrical engineering": "engineering",
            "civil engineering": "engineering",
            "chemical engineering": "engineering",
            "art": "arts",
            "music": "arts",
            "design": "arts",
        }

        # Default to mathematics if domain not found
        stream_domain = domain_mapping.get(domain.lower(), "mathematics")

        # For mathematics domain, use a custom signature that matches the expected output format
        if stream_domain == "mathematics":
            return (
                "subject, topic, difficulty_level, learning_objectives -> "
                "problem_statement, solution, proof, reasoning_trace, hints"
            )

        return None  # Let STREAMContentGenerator use the default signature

    def _get_quality_requirements(self, domain: str) -> Dict[str, Any]:
        """
        Get quality requirements for a domain.

        Args:
            domain: The domain for content generation

        Returns:
            Dictionary of quality requirements
        """
        # Default quality requirements
        quality_requirements = {
            "min_problem_length": 100,
            "min_solution_length": 50,
            "min_reasoning_length": 200,
            "min_hints_count": 3,
        }

        # Add domain-specific requirements
        if domain.lower() == "mathematics":
            quality_requirements.update(
                {
                    "require_proof": True,
                    "min_proof_length": 100,
                    "require_hints": True,
                }
            )

        return quality_requirements

    def generate(
        self,
        subject: str,
        topic: str,
        seed_prompt: Optional[str] = None,
        difficulty_level: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate a math problem with hints using DSPy ChainOfThought reasoning.

        Args:
            subject: The math subject (e.g., 'Algebra', 'Calculus')
            topic: The specific topic within the subject
            seed_prompt: Optional seed/inspiration for the problem
            difficulty_level: Optional difficulty level specification
            **kwargs: Additional generation parameters

        Returns:
            Dict containing:
                - subject: The math subject
                - topic: The topic
                - problem: The problem statement
                - answer: The correct answer
                - hints: Dictionary of step-by-step hints
                - difficulty_level: The difficulty level used
                - topic_description: Description of the topic
                - tokens_prompt: Input tokens used
                - tokens_completion: Output tokens used
                - raw_output: Raw model output
                - raw_prompt: Raw prompt sent to model

        Raises:
            ValidationError: If the generated problem is invalid
            DSPyIntegrationError: If DSPy generation fails
        """
        self.logger.info(
            f"Generating problem for {subject} - {topic} (level: {difficulty_level})"
        )

        try:
            # Check if we need to initialize or reinitialize the DSPy module
            if not self.dspy_module or self.dspy_module.domain != subject.lower():
                self.initialize_dspy_module(subject.lower())

            # Get topic information from enhanced taxonomy
            taxonomy_data = self.config_manager.get("taxonomy", {})
            topic_info = self._get_topic_info(taxonomy_data, subject, topic)

            # Use topic info to determine difficulty level if not provided
            if not difficulty_level and topic_info:
                difficulty_level = topic_info.get("level")

            topic_description = topic_info.get("description", "")

            # Prepare learning objectives
            learning_objectives = [
                f"Understand {topic} concepts",
                f"Apply {topic} techniques to solve problems",
                f"Develop critical thinking skills in {subject}",
            ]

            if seed_prompt:
                learning_objectives.append(
                    f"Apply concepts to real-world scenario: {seed_prompt}"
                )

            # Use DSPy module for generation
            dspy_result = self.dspy_module(
                subject=subject,
                topic=topic,
                difficulty_level=difficulty_level or "Undergraduate",
                learning_objectives=learning_objectives,
            )

            # Convert DSPy result to expected format
            result = self._convert_dspy_result(
                dspy_result, subject, topic, difficulty_level, topic_description
            )

            self.logger.info(
                f"Successfully generated {difficulty_level or 'standard'} level problem with DSPy"
            )

            return result

        except DSPyIntegrationError as e:
            self.logger.warning(
                f"DSPy generation failed, falling back to legacy: {str(e)}"
            )
            # Fall back to legacy implementation
            return self._fallback_generate(
                subject, topic, seed_prompt, difficulty_level, **kwargs
            )
        except Exception as e:
            self.logger.error(
                "DSPy generation failed with unexpected error: %s", str(e)
            )
            # Fall back to legacy implementation
            return self._fallback_generate(
                subject, topic, seed_prompt, difficulty_level, **kwargs
            )

    def _convert_dspy_result(
        self,
        dspy_result: Any,
        subject: str,
        topic: str,
        difficulty_level: Optional[str],
        topic_description: Optional[str],
    ) -> Dict[str, Any]:
        """
        Convert DSPy result to expected format.

        Args:
            dspy_result: Result from DSPy module
            subject: The math subject
            topic: The specific topic
            difficulty_level: The difficulty level
            topic_description: Description of the topic

        Returns:
            Dictionary in the expected format

        Raises:
            ValidationError: If the result cannot be converted
        """
        try:
            # Extract fields from DSPy result
            problem_statement = getattr(dspy_result, "problem_statement", "")
            solution = getattr(dspy_result, "solution", "")
            proof = getattr(dspy_result, "proof", "")
            reasoning_trace = getattr(dspy_result, "reasoning_trace", "")

            # Extract hints or create from reasoning trace
            hints = {}
            if hasattr(dspy_result, "hints") and isinstance(dspy_result.hints, dict):
                hints = dspy_result.hints
            else:
                # Create hints from reasoning trace if not provided
                reasoning_parts = reasoning_trace.split("\n\n")
                for i, part in enumerate(reasoning_parts[:5]):  # Use up to 5 parts
                    if part.strip():
                        hints[str(i)] = part.strip()

                # Ensure we have at least 3 hints
                if len(hints) < 3:
                    solution_parts = solution.split(".")
                    for i, part in enumerate(solution_parts[: 3 - len(hints)]):
                        if part.strip() and str(i + len(hints)) not in hints:
                            hints[str(i + len(hints))] = part.strip() + "."

            # Validate hints
            if not hints or len(hints) < 3:
                raise ValidationError(
                    "Generated content has insufficient hints", field="hints"
                )

            # Construct result in expected format
            result = {
                "subject": subject,
                "topic": topic,
                "problem": problem_statement,
                "answer": solution,
                "hints": hints,
                "difficulty_level": difficulty_level,
                "topic_description": topic_description,
                "dspy_generated": True,
                "reasoning_trace": reasoning_trace,
            }

            return result

        except Exception as e:
            raise ValidationError(
                f"Failed to convert DSPy result: {str(e)}", field="dspy_result"
            )

    def _fallback_generate(
        self,
        subject: str,
        topic: str,
        seed_prompt: Optional[str] = None,
        difficulty_level: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Fallback to legacy implementation if DSPy generation fails.

        Args:
            subject: The math subject
            topic: The specific topic
            seed_prompt: Optional seed/inspiration
            difficulty_level: Optional difficulty level
            **kwargs: Additional parameters

        Returns:
            Dictionary in the expected format
        """
        self.logger.info("Falling back to legacy EngineerAgent implementation")

        # Create legacy EngineerAgent and use it
        legacy_agent = EngineerAgent()
        return legacy_agent.generate(
            subject=subject,
            topic=topic,
            seed_prompt=seed_prompt,
            difficulty_level=difficulty_level,
            **kwargs,
        )

    def _get_topic_info(
        self, taxonomy_data: Dict[str, Any], subject: str, topic: str
    ) -> Dict[str, Any]:
        """
        Get topic information from taxonomy.

        Args:
            taxonomy_data: Taxonomy data
            subject: The math subject
            topic: The specific topic

        Returns:
            Dictionary with topic information
        """
        # Import here to avoid circular imports
        from utils.taxonomy import get_topic_info

        return get_topic_info(taxonomy_data, subject, topic)


# Factory function for easy agent creation
def create_dspy_engineer_agent() -> DSPyEngineerAgent:
    """Create and return a DSPyEngineerAgent instance."""
    return DSPyEngineerAgent()
