"""
DSPy Checker Agent

This module implements the DSPyCheckerAgent class, which extends the base Agent
class and uses DSPy for validation and equivalence checking.
"""

import logging
from typing import Any, Dict, Optional

from core.agents import Agent, CheckerAgent
from core.dspy.base_module import STREAMContentGenerator
from core.dspy.config import get_dspy_config
from core.dspy.exceptions import DSPyIntegrationError
from core.dspy.optimization_engine import DSPyOptimizationEngine
from utils.exceptions import ValidationError
from utils.json_utils import safe_json_parse

logger = logging.getLogger(__name__)


class DSPyCheckerAgent(Agent):
    """
    DSPy-powered Checker Agent for validation and equivalence checking.

    This class extends the base Agent class and uses DSPy for validation
    and equivalence checking with structured feedback for optimization loops.
    """

    def __init__(self):
        """Initialize the DSPyCheckerAgent."""
        super().__init__("checker", "checker_model")
        self.dspy_module = None
        self.optimization_engine = DSPyOptimizationEngine()
        self.dspy_config = get_dspy_config()
        self.logger = logging.getLogger(__name__ + ".DSPyCheckerAgent")
        self.logger.info("Initialized DSPyCheckerAgent")

    def get_system_prompt(self) -> str:
        """
        Get the system prompt for validation.

        Returns:
            System prompt string
        """
        # Use the same system prompt as the original CheckerAgent
        # This maintains backward compatibility
        checker_agent = CheckerAgent()
        return checker_agent.get_system_prompt()

    def initialize_dspy_module(
        self, domain: str, validation_type: str = "validation"
    ) -> None:
        """
        Initialize DSPy module for specific domain and validation type.

        Args:
            domain: The domain for validation (e.g., 'mathematics')
            validation_type: Type of validation ('validation' or 'equivalence_check')

        Raises:
            DSPyIntegrationError: If initialization fails
        """
        try:
            # Get domain signature based on validation type
            signature = self._get_domain_signature(domain, validation_type)

            # Create STREAMContentGenerator for the domain
            self.dspy_module = STREAMContentGenerator(domain, signature)

            # Optimize the module if enabled
            if self.dspy_config.is_enabled():
                self.dspy_module = self.optimization_engine.optimize_for_domain(
                    self.dspy_module,
                    self._get_quality_requirements(domain, validation_type),
                )

            self.logger.info(
                "Initialized DSPy module for %s %s", domain, validation_type
            )

        except Exception as e:
            error_msg = f"Failed to initialize DSPy module for {domain} {validation_type}: {str(e)}"
            self.logger.error(error_msg)
            raise DSPyIntegrationError(
                error_msg,
                details={
                    "domain": domain,
                    "validation_type": validation_type,
                    "error": str(e),
                },
            )

    def _get_domain_signature(self, domain: str, validation_type: str) -> str:
        """
        Get the appropriate signature for a domain and validation type.

        Args:
            domain: The domain for validation
            validation_type: Type of validation

        Returns:
            Signature string for the domain and validation type
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

        # For mathematics domain, use custom signatures based on validation type
        if stream_domain == "mathematics":
            if validation_type == "validation":
                return (
                    "problem_statement, solution, hints, expected_difficulty -> "
                    "valid, reason, corrected_hints, quality_score, mathematical_accuracy, pedagogical_value"
                )
            elif validation_type == "equivalence_check":
                return (
                    "problem_statement, true_answer, model_answer, solution_context -> "
                    "equivalent, confidence_score, explanation, mathematical_justification"
                )

        return None  # Let STREAMContentGenerator use the default signature

    def _get_quality_requirements(
        self, domain: str, validation_type: str
    ) -> Dict[str, Any]:
        """
        Get quality requirements for a domain and validation type.

        Args:
            domain: The domain for validation
            validation_type: Type of validation

        Returns:
            Dictionary of quality requirements
        """
        # Default quality requirements
        quality_requirements = {
            "min_reason_length": 50,
            "min_explanation_length": 100,
        }

        # Add validation-type-specific requirements
        if validation_type == "validation":
            quality_requirements.update(
                {
                    "require_corrected_hints": True,
                    "min_quality_score": 0.7,
                }
            )
        elif validation_type == "equivalence_check":
            quality_requirements.update(
                {
                    "min_confidence_score": 0.8,
                    "require_justification": True,
                }
            )

        return quality_requirements

    def validate(
        self, problem_data: Dict[str, Any], mode: str = "initial", **kwargs
    ) -> Dict[str, Any]:
        """
        Validate a problem or perform enhanced answer equivalence check using DSPy.

        Args:
            problem_data: Dictionary containing problem information
            mode: Validation mode ('initial' or 'equivalence_check')
            **kwargs: Additional validation parameters

        Returns:
            Dict containing:
                - valid: Boolean indicating if problem is valid
                - reason: Explanation of validation result
                - corrected_hints: Optional corrected hints if needed
                - equivalence_confidence: Confidence score for equivalence (0-1)
                - tokens_prompt: Input tokens used
                - tokens_completion: Output tokens used
                - raw_output: Raw model output
                - raw_prompt: Raw prompt sent to model
                - dspy_validated: Boolean indicating if DSPy was used

        Raises:
            ValidationError: If mode is invalid
            DSPyIntegrationError: If DSPy validation fails
        """
        self.logger.info("Validating problem in %s mode using DSPy", mode)

        try:
            # Determine domain from problem data
            domain = problem_data.get("subject", "mathematics").lower()

            # Initialize or reinitialize DSPy module if needed
            if not self.dspy_module or self.dspy_module.domain != domain:
                self.initialize_dspy_module(domain, mode)

            # Prepare inputs for DSPy module based on mode
            dspy_inputs = self._prepare_dspy_inputs(problem_data, mode)

            # Use DSPy module for validation
            dspy_result = self.dspy_module(**dspy_inputs)

            # Convert DSPy result to expected format
            result = self._convert_dspy_result(dspy_result, problem_data, mode)

            self.logger.info(
                f"DSPy validation complete: {'VALID' if result['valid'] else 'INVALID'} "
                f"(confidence: {result.get('equivalence_confidence', 0):.2f})"
            )

            return result

        except DSPyIntegrationError as e:
            self.logger.warning(
                f"DSPy validation failed, falling back to legacy: {str(e)}"
            )
            # Fall back to legacy implementation
            return self._fallback_validate(problem_data, mode, **kwargs)
        except Exception as e:
            self.logger.error(
                "DSPy validation failed with unexpected error: %s", str(e)
            )
            # Fall back to legacy implementation
            return self._fallback_validate(problem_data, mode, **kwargs)

    def _prepare_dspy_inputs(
        self, problem_data: Dict[str, Any], mode: str
    ) -> Dict[str, Any]:
        """
        Prepare inputs for DSPy module based on validation mode.

        Args:
            problem_data: Dictionary containing problem information
            mode: Validation mode

        Returns:
            Dictionary of inputs for DSPy module

        Raises:
            ValidationError: If mode is invalid
        """
        if mode == "initial":
            return {
                "problem_statement": problem_data.get("problem", ""),
                "solution": problem_data.get("answer", ""),
                "hints": problem_data.get("hints", {}),
                "expected_difficulty": problem_data.get(
                    "difficulty_level", "Undergraduate"
                ),
            }
        elif mode == "equivalence_check":
            return {
                "problem_statement": problem_data.get("problem", ""),
                "true_answer": problem_data.get("answer", ""),
                "model_answer": problem_data.get("target_model_answer", ""),
                "solution_context": problem_data.get("topic_description", ""),
            }
        else:
            raise ValidationError(f"Unknown validation mode: {mode}", field="mode")

    def _convert_dspy_result(
        self, dspy_result: Any, problem_data: Dict[str, Any], mode: str
    ) -> Dict[str, Any]:
        """
        Convert DSPy result to expected format.

        Args:
            dspy_result: Result from DSPy module
            problem_data: Original problem data
            mode: Validation mode

        Returns:
            Dictionary in the expected format

        Raises:
            ValidationError: If the result cannot be converted
        """
        try:
            # Common fields
            result = {
                "dspy_validated": True,
            }

            # Mode-specific fields
            if mode == "initial":
                result.update(
                    {
                        "valid": getattr(dspy_result, "valid", False),
                        "reason": getattr(dspy_result, "reason", "No reason provided"),
                        "corrected_hints": getattr(dspy_result, "corrected_hints", {}),
                        "quality_score": getattr(dspy_result, "quality_score", 0.0),
                    }
                )

                # Add domain-specific metrics
                if hasattr(dspy_result, "mathematical_accuracy"):
                    result["mathematical_accuracy"] = dspy_result.mathematical_accuracy

                if hasattr(dspy_result, "pedagogical_value"):
                    result["pedagogical_value"] = dspy_result.pedagogical_value

            elif mode == "equivalence_check":
                result.update(
                    {
                        "valid": getattr(dspy_result, "equivalent", False),
                        "reason": getattr(
                            dspy_result, "explanation", "No explanation provided"
                        ),
                        "equivalence_confidence": getattr(
                            dspy_result, "confidence_score", 0.0
                        ),
                    }
                )

                # Add domain-specific metrics
                if hasattr(dspy_result, "mathematical_justification"):
                    result["mathematical_justification"] = (
                        dspy_result.mathematical_justification
                    )

            return result

        except Exception as e:
            raise ValidationError(
                f"Failed to convert DSPy result: {str(e)}", field="dspy_result"
            )

    def _fallback_validate(
        self, problem_data: Dict[str, Any], mode: str = "initial", **kwargs
    ) -> Dict[str, Any]:
        """
        Fallback to legacy implementation if DSPy validation fails.

        Args:
            problem_data: Dictionary containing problem information
            mode: Validation mode
            **kwargs: Additional parameters

        Returns:
            Dictionary in the expected format
        """
        self.logger.info("Falling back to legacy CheckerAgent implementation")

        # Create legacy CheckerAgent and use it
        legacy_agent = CheckerAgent()
        result = legacy_agent.validate(problem_data, mode, **kwargs)

        # Mark as not using DSPy
        result["dspy_validated"] = False

        return result


# Factory function for easy agent creation
def create_dspy_checker_agent() -> DSPyCheckerAgent:
    """Create and return a DSPyCheckerAgent instance."""
    return DSPyCheckerAgent()
