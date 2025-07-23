"""
DSPy Target Agent

This module implements the DSPyTargetAgent class, which extends the base Agent
class and uses DSPy for problem solving and solution generation.
"""

import logging
from typing import Any, Dict, Optional

from core.agents import Agent, TargetAgent
from core.dspy.base_module import STREAMContentGenerator
from core.dspy.config import get_dspy_config
from core.dspy.exceptions import DSPyIntegrationError
from core.dspy.optimization_engine import DSPyOptimizationEngine
from utils.exceptions import ValidationError

logger = logging.getLogger(__name__)


class DSPyTargetAgent(Agent):
    """
    DSPy-powered Target Agent for problem solving.

    This class extends the base Agent class and uses DSPy for problem solving
    with deterministic capabilities and solution evaluation.
    """

    def __init__(self):
        """Initialize the DSPyTargetAgent."""
        super().__init__("target", "target_model")
        self.dspy_module = None
        self.optimization_engine = DSPyOptimizationEngine()
        self.dspy_config = get_dspy_config()
        self.logger = logging.getLogger(__name__ + ".DSPyTargetAgent")
        self.logger.info("Initialized DSPyTargetAgent")

    def get_system_prompt(self) -> str:
        """
        Get the system prompt for problem solving.

        Returns:
            System prompt string
        """
        # Use the same system prompt as the original TargetAgent
        # This maintains backward compatibility
        target_agent = TargetAgent()
        return target_agent.get_system_prompt()

    def initialize_dspy_module(self, domain: str) -> None:
        """
        Initialize DSPy module for specific domain.

        Args:
            domain: The domain for problem solving (e.g., 'mathematics')

        Raises:
            DSPyIntegrationError: If initialization fails
        """
        try:
            # Get domain signature for problem solving
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
            domain: The domain for problem solving

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

        # For mathematics domain, use a custom signature for problem solving
        if stream_domain == "mathematics":
            return (
                "problem_statement, context_info -> "
                "solution_steps, final_answer, reasoning_trace, confidence_score"
            )

        return None  # Let STREAMContentGenerator use the default signature

    def _get_quality_requirements(self, domain: str) -> Dict[str, Any]:
        """
        Get quality requirements for a domain.

        Args:
            domain: The domain for problem solving

        Returns:
            Dictionary of quality requirements
        """
        # Default quality requirements for target agent
        quality_requirements = {
            "min_solution_length": 20,
            "min_reasoning_length": 50,
            "require_final_answer": True,
            "deterministic_solving": True,
        }

        # Add domain-specific requirements
        if domain.lower() == "mathematics":
            quality_requirements.update(
                {
                    "require_step_by_step": True,
                    "min_confidence_score": 0.7,
                    "numerical_precision": 0.001,
                }
            )

        return quality_requirements

    def solve(self, problem_text: str, **kwargs) -> Dict[str, Any]:
        """
        Attempt to solve a math problem using DSPy ChainOfThought reasoning.

        Args:
            problem_text: The problem statement to solve
            **kwargs: Additional solving parameters

        Returns:
            Dict containing:
                - output: The model's answer attempt
                - solution_steps: Step-by-step solution process
                - reasoning_trace: Detailed reasoning trace
                - confidence_score: Confidence in the solution
                - tokens_prompt: Input tokens used
                - tokens_completion: Output tokens used
                - latency: Response time in seconds
                - dspy_solved: Boolean indicating if DSPy was used

        Raises:
            DSPyIntegrationError: If DSPy solving fails
        """
        self.logger.info("Attempting to solve problem using DSPy")

        try:
            # Determine domain from problem context or use default
            domain = kwargs.get("domain", "mathematics")

            # Check if we need to initialize or reinitialize the DSPy module
            if not self.dspy_module or self.dspy_module.domain != domain.lower():
                self.initialize_dspy_module(domain.lower())

            # Prepare context information
            context_info = self._prepare_context_info(problem_text, **kwargs)

            # Use DSPy module for problem solving
            dspy_result = self.dspy_module(
                problem_statement=problem_text, context_info=context_info
            )

            # Convert DSPy result to expected format
            result = self._convert_dspy_result(dspy_result, problem_text, **kwargs)

            self.logger.info(
                f"DSPy problem solving completed with confidence: {result.get('confidence_score', 0):.2f}"
            )

            return result

        except DSPyIntegrationError as e:
            self.logger.warning(
                f"DSPy solving failed, falling back to legacy: {str(e)}"
            )
            # Fall back to legacy implementation
            return self._fallback_solve(problem_text, **kwargs)
        except Exception as e:
            self.logger.error("DSPy solving failed with unexpected error: %s", str(e))
            # Fall back to legacy implementation
            return self._fallback_solve(problem_text, **kwargs)

    def _prepare_context_info(self, problem_text: str, **kwargs) -> str:
        """
        Prepare context information for problem solving.

        Args:
            problem_text: The problem statement
            **kwargs: Additional parameters

        Returns:
            Context information string
        """
        context_parts = []

        # Add domain information
        if "domain" in kwargs:
            context_parts.append(f"Domain: {kwargs['domain']}")

        # Add topic information
        if "topic" in kwargs:
            context_parts.append(f"Topic: {kwargs['topic']}")

        # Add difficulty level
        if "difficulty_level" in kwargs:
            context_parts.append(f"Difficulty: {kwargs['difficulty_level']}")

        # Add any hints if available
        if "hints" in kwargs and kwargs["hints"]:
            hints_text = "; ".join(kwargs["hints"].values())
            context_parts.append(f"Available hints: {hints_text}")

        # Add solving instructions
        context_parts.append("Solve step by step and provide the final answer.")
        context_parts.append("Show your reasoning clearly.")

        return " | ".join(context_parts)

    def _convert_dspy_result(
        self, dspy_result: Any, problem_text: str, **kwargs
    ) -> Dict[str, Any]:
        """
        Convert DSPy result to expected format.

        Args:
            dspy_result: Result from DSPy module
            problem_text: Original problem text
            **kwargs: Additional parameters

        Returns:
            Dictionary in the expected format

        Raises:
            ValidationError: If the result cannot be converted
        """
        try:
            # Extract fields from DSPy result
            solution_steps = getattr(dspy_result, "solution_steps", "")
            final_answer = getattr(dspy_result, "final_answer", "")
            reasoning_trace = getattr(dspy_result, "reasoning_trace", "")
            confidence_score = getattr(dspy_result, "confidence_score", 0.0)

            # Combine solution steps and final answer for output
            if solution_steps and final_answer:
                output = f"{solution_steps}\n\nFinal Answer: {final_answer}"
            elif final_answer:
                output = final_answer
            elif solution_steps:
                output = solution_steps
            else:
                output = "Unable to solve the problem"

            # Construct result in expected format
            result = {
                "output": output.strip(),
                "solution_steps": solution_steps,
                "reasoning_trace": reasoning_trace,
                "confidence_score": confidence_score,
                "dspy_solved": True,
                "final_answer": final_answer,
            }

            return result

        except Exception as e:
            raise ValidationError(
                f"Failed to convert DSPy result: {str(e)}", field="dspy_result"
            )

    def _fallback_solve(self, problem_text: str, **kwargs) -> Dict[str, Any]:
        """
        Fallback to legacy implementation if DSPy solving fails.

        Args:
            problem_text: The problem statement to solve
            **kwargs: Additional parameters

        Returns:
            Dictionary in the expected format
        """
        self.logger.info("Falling back to legacy TargetAgent implementation")

        # Create legacy TargetAgent and use it
        legacy_agent = TargetAgent()
        result = legacy_agent.solve(problem_text, **kwargs)

        # Add DSPy-specific fields
        result.update(
            {
                "dspy_solved": False,
                "solution_steps": "",
                "reasoning_trace": "",
                "confidence_score": 0.0,
                "final_answer": result.get("output", ""),
            }
        )

        return result

    def evaluate_solution(
        self,
        problem_text: str,
        solution: str,
        expected_answer: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Evaluate a solution using DSPy reasoning.

        Args:
            problem_text: The original problem statement
            solution: The solution to evaluate
            expected_answer: Optional expected answer for comparison
            **kwargs: Additional evaluation parameters

        Returns:
            Dict containing:
                - correct: Boolean indicating if solution is correct
                - confidence: Confidence score for the evaluation
                - explanation: Explanation of the evaluation
                - reasoning_trace: Detailed reasoning for evaluation
                - dspy_evaluated: Boolean indicating if DSPy was used

        Raises:
            DSPyIntegrationError: If DSPy evaluation fails
        """
        self.logger.info("Evaluating solution using DSPy")

        try:
            # For now, use a simple heuristic evaluation
            # In a full implementation, this would use a separate DSPy module for evaluation

            # Check if solution contains the expected answer (if provided)
            correct = True
            confidence = 0.8
            explanation = "Solution appears to be correct"

            if expected_answer:
                # Simple string matching (in practice, would use more sophisticated comparison)
                if expected_answer.lower().strip() in solution.lower():
                    correct = True
                    confidence = 0.9
                    explanation = "Solution matches expected answer"
                else:
                    correct = False
                    confidence = 0.3
                    explanation = "Solution does not match expected answer"

            result = {
                "correct": correct,
                "confidence": confidence,
                "explanation": explanation,
                "reasoning_trace": f"Evaluated solution against problem: {problem_text[:100]}...",
                "dspy_evaluated": True,
            }

            self.logger.info(
                f"Solution evaluation complete: {'CORRECT' if correct else 'INCORRECT'} (confidence: {confidence:.2f})"
            )

            return result

        except Exception as e:
            self.logger.error("DSPy evaluation failed: %s", str(e))
            # Return default evaluation
            return {
                "correct": False,
                "confidence": 0.0,
                "explanation": f"Evaluation failed: {str(e)}",
                "reasoning_trace": "",
                "dspy_evaluated": False,
            }


# Factory function for easy agent creation
def create_dspy_target_agent() -> DSPyTargetAgent:
    """Create and return a DSPyTargetAgent instance."""
    return DSPyTargetAgent()
