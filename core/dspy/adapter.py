"""
Adapter pattern for backward compatibility with existing agents.

This module provides adapters that allow existing agent code to work with
the new DSPy-based modules without requiring immediate refactoring.
"""

import logging
from typing import Any, Dict, List, Optional, Union

from .base_module import STREAMContentGenerator
from .checker_agent import DSPyCheckerAgent
from .config import get_dspy_config
from .engineer_agent import DSPyEngineerAgent
from .exceptions import DSPyIntegrationError
from .target_agent import DSPyTargetAgent

logger = logging.getLogger(__name__)


from abc import ABC, abstractmethod


class AgentAdapter(ABC):
    """Base adapter for DSPy modules to legacy agent interfaces."""

    def __init__(self, domain: str, use_dspy: bool = True):
        """
        Initialize agent adapter.

        Args:
            domain: Domain for the agent
            use_dspy: Whether to use DSPy (True) or legacy implementation (False)
        """
        self.domain = domain
        self.use_dspy = use_dspy
        self.config = get_dspy_config()
        self.logger = logging.getLogger(__name__ + ".AgentAdapter")

        # Will be initialized on first use
        self._dspy_module = None
        self._legacy_agent = None

    def _get_dspy_module(self) -> STREAMContentGenerator:
        """
        Get or initialize DSPy module.

        Returns:
            Initialized DSPy module
        """
        if self._dspy_module is None:
            self._initialize_dspy_module()
        return self._dspy_module

    def _get_legacy_agent(self) -> Any:
        """
        Get or initialize legacy agent.

        Returns:
            Initialized legacy agent
        """
        if self._legacy_agent is None:
            self._initialize_legacy_agent()
        return self._legacy_agent

    @abstractmethod
    def _initialize_dspy_module(self) -> None:
        """
        Initialize DSPy module.
        """
        pass

    @abstractmethod
    def _initialize_legacy_agent(self) -> None:
        """
        Initialize legacy agent.
        """
        pass

    @abstractmethod
    def _convert_to_legacy_format(self, dspy_result: Dict[str, Any]) -> Any:
        """
        Convert DSPy result to legacy format.

        Args:
            dspy_result: Result from DSPy module

        Returns:
            Result in legacy format
        """
        pass

    @abstractmethod
    def _convert_to_dspy_format(self, legacy_input: Any) -> Dict[str, Any]:
        """
        Convert legacy input to DSPy format.

        Args:
            legacy_input: Input in legacy format

        Returns:
            Input in DSPy format
        """
        pass


class EngineerAgentAdapter(AgentAdapter):
    """Adapter for EngineerAgent to DSPyEngineerAgent."""

    def _initialize_dspy_module(self) -> None:
        """
        Initialize DSPy engineer module.
        """
        try:
            self._dspy_module = DSPyEngineerAgent(self.domain)
            self.logger.info(
                "Initialized DSPy engineer module for domain: %s", self.domain
            )
        except Exception as e:
            self.logger.error("Failed to initialize DSPy engineer module: %s", str(e))
            self._dspy_module = None
            # Force fallback to legacy
            self.use_dspy = False

    def _initialize_legacy_agent(self) -> None:
        """
        Initialize legacy engineer agent.
        """
        try:
            # Import here to avoid circular imports
            from legacy.agents import EngineerAgent

            self._legacy_agent = EngineerAgent(self.domain)
            self.logger.info(
                "Initialized legacy engineer agent for domain: %s", self.domain
            )
        except Exception as e:
            self.logger.error("Failed to initialize legacy engineer agent: %s", str(e))
            if not self.use_dspy:
                raise DSPyIntegrationError(
                    f"Failed to initialize legacy engineer agent: {str(e)}"
                ) from e

    def _convert_to_legacy_format(self, dspy_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert DSPy result to legacy format.

        Args:
            dspy_result: Result from DSPy module

        Returns:
            Result in legacy format
        """
        # For engineer agent, the formats are similar enough for direct conversion
        legacy_result = {
            "problem": dspy_result.get("problem_statement", ""),
            "solution": dspy_result.get("solution", ""),
            "explanation": dspy_result.get("reasoning_trace", ""),
            "difficulty": dspy_result.get("difficulty_level", "medium"),
        }

        # Add any additional fields that might be present
        for key, value in dspy_result.items():
            if key not in [
                "problem_statement",
                "solution",
                "reasoning_trace",
                "difficulty_level",
            ]:
                legacy_result[key] = value

        return legacy_result

    def _convert_to_dspy_format(self, legacy_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert legacy input to DSPy format.

        Args:
            legacy_input: Input in legacy format

        Returns:
            Input in DSPy format
        """
        dspy_input = {
            "difficulty_level": legacy_input.get("difficulty", "medium"),
            "learning_objectives": legacy_input.get("objectives", []),
        }

        # Add topic if present
        if "topic" in legacy_input:
            dspy_input["topic"] = legacy_input["topic"]

        # Add any additional fields that might be present
        for key, value in legacy_input.items():
            if key not in ["difficulty", "objectives", "topic"]:
                dspy_input[key] = value

        return dspy_input

    def generate_problem(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Generate a problem using either DSPy or legacy implementation.

        Args:
            input_data: Input data for problem generation
            **kwargs: Additional arguments

        Returns:
            Generated problem
        """
        try:
            if self.use_dspy and self.config.is_enabled():
                # Use DSPy implementation
                dspy_module = self._get_dspy_module()
                if dspy_module is not None:
                    dspy_input = self._convert_to_dspy_format(input_data)
                    dspy_result = dspy_module.generate(dspy_input, **kwargs)
                    return self._convert_to_legacy_format(dspy_result)

            # Fallback to legacy implementation
            legacy_agent = self._get_legacy_agent()
            return legacy_agent.generate_problem(input_data, **kwargs)

        except Exception as e:
            self.logger.error("Error in generate_problem: %s", str(e))
            # Always try to fall back to legacy implementation
            try:
                legacy_agent = self._get_legacy_agent()
                return legacy_agent.generate_problem(input_data, **kwargs)
            except Exception as fallback_error:
                self.logger.error(
                    "Fallback to legacy implementation failed: %s", str(fallback_error)
                )
                raise DSPyIntegrationError(
                    f"Failed to generate problem: {str(e)}"
                ) from e


class CheckerAgentAdapter(AgentAdapter):
    """Adapter for CheckerAgent to DSPyCheckerAgent."""

    def _initialize_dspy_module(self) -> None:
        """
        Initialize DSPy checker module.
        """
        try:
            self._dspy_module = DSPyCheckerAgent(self.domain)
            self.logger.info(
                "Initialized DSPy checker module for domain: %s", self.domain
            )
        except Exception as e:
            self.logger.error("Failed to initialize DSPy checker module: %s", str(e))
            self._dspy_module = None
            # Force fallback to legacy
            self.use_dspy = False

    def _initialize_legacy_agent(self) -> None:
        """
        Initialize legacy checker agent.
        """
        try:
            # Import here to avoid circular imports
            from legacy.agents import CheckerAgent

            self._legacy_agent = CheckerAgent(self.domain)
            self.logger.info(
                "Initialized legacy checker agent for domain: %s", self.domain
            )
        except Exception as e:
            self.logger.error("Failed to initialize legacy checker agent: %s", str(e))
            if not self.use_dspy:
                raise DSPyIntegrationError(
                    f"Failed to initialize legacy checker agent: {str(e)}"
                ) from e

    def _convert_to_legacy_format(self, dspy_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert DSPy result to legacy format.

        Args:
            dspy_result: Result from DSPy module

        Returns:
            Result in legacy format
        """
        legacy_result = {
            "is_valid": dspy_result.get("is_valid", False),
            "feedback": dspy_result.get("feedback", ""),
            "score": dspy_result.get("score", 0.0),
        }

        # Add detailed scores if present
        if "detailed_scores" in dspy_result:
            legacy_result["detailed_scores"] = dspy_result["detailed_scores"]

        # Add any additional fields
        for key, value in dspy_result.items():
            if key not in ["is_valid", "feedback", "score", "detailed_scores"]:
                legacy_result[key] = value

        return legacy_result

    def _convert_to_dspy_format(self, problem_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert legacy input to DSPy format.

        Args:
            problem_data: Problem data in legacy format

        Returns:
            Problem data in DSPy format
        """
        dspy_input = {}

        # Map common fields
        if "problem" in problem_data:
            dspy_input["problem_statement"] = problem_data["problem"]
        if "solution" in problem_data:
            dspy_input["solution"] = problem_data["solution"]
        if "explanation" in problem_data:
            dspy_input["reasoning_trace"] = problem_data["explanation"]

        # Add any additional fields
        for key, value in problem_data.items():
            if key not in ["problem", "solution", "explanation"]:
                dspy_input[key] = value

        return dspy_input

    def validate_problem(
        self, problem_data: Dict[str, Any], mode: str = "validation", **kwargs
    ) -> Dict[str, Any]:
        """
        Validate a problem using either DSPy or legacy implementation.

        Args:
            problem_data: Problem data to validate
            mode: Validation mode
            **kwargs: Additional arguments

        Returns:
            Validation result
        """
        try:
            if self.use_dspy and self.config.is_enabled():
                # Use DSPy implementation
                dspy_module = self._get_dspy_module()
                if dspy_module is not None:
                    dspy_input = self._convert_to_dspy_format(problem_data)
                    dspy_result = dspy_module.validate(dspy_input, mode, **kwargs)
                    return self._convert_to_legacy_format(dspy_result)

            # Fallback to legacy implementation
            legacy_agent = self._get_legacy_agent()
            return legacy_agent.validate_problem(problem_data, mode, **kwargs)

        except Exception as e:
            self.logger.error("Error in validate_problem: %s", str(e))
            # Always try to fall back to legacy implementation
            try:
                legacy_agent = self._get_legacy_agent()
                return legacy_agent.validate_problem(problem_data, mode, **kwargs)
            except Exception as fallback_error:
                self.logger.error(
                    "Fallback to legacy implementation failed: %s", str(fallback_error)
                )
                raise DSPyIntegrationError(
                    f"Failed to validate problem: {str(e)}"
                ) from e


class TargetAgentAdapter(AgentAdapter):
    """Adapter for TargetAgent to DSPyTargetAgent."""

    def _initialize_dspy_module(self) -> None:
        """
        Initialize DSPy target module.
        """
        try:
            self._dspy_module = DSPyTargetAgent(self.domain)
            self.logger.info(
                "Initialized DSPy target module for domain: %s", self.domain
            )
        except Exception as e:
            self.logger.error("Failed to initialize DSPy target module: %s", str(e))
            self._dspy_module = None
            # Force fallback to legacy
            self.use_dspy = False

    def _initialize_legacy_agent(self) -> None:
        """
        Initialize legacy target agent.
        """
        try:
            # Import here to avoid circular imports
            from legacy.agents import TargetAgent

            self._legacy_agent = TargetAgent(self.domain)
            self.logger.info(
                "Initialized legacy target agent for domain: %s", self.domain
            )
        except Exception as e:
            self.logger.error("Failed to initialize legacy target agent: %s", str(e))
            if not self.use_dspy:
                raise DSPyIntegrationError(
                    f"Failed to initialize legacy target agent: {str(e)}"
                ) from e

    def _convert_to_legacy_format(self, dspy_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert DSPy result to legacy format.

        Args:
            dspy_result: Result from DSPy module

        Returns:
            Result in legacy format
        """
        legacy_result = {
            "solution": dspy_result.get("solution", ""),
            "explanation": dspy_result.get("reasoning_trace", ""),
        }

        # Add any additional fields
        for key, value in dspy_result.items():
            if key not in ["solution", "reasoning_trace"]:
                legacy_result[key] = value

        return legacy_result

    def _convert_to_dspy_format(
        self, problem_text: Union[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Convert legacy input to DSPy format.

        Args:
            problem_text: Problem text in legacy format

        Returns:
            Problem data in DSPy format
        """
        # For target agent, the input might be just a string or a dict
        if isinstance(problem_text, str):
            return {"problem_statement": problem_text}
        elif isinstance(problem_text, dict):
            dspy_input = {}
            if "problem" in problem_text:
                dspy_input["problem_statement"] = problem_text["problem"]
            else:
                # Try to find any field that might contain the problem
                for key, value in problem_text.items():
                    if isinstance(value, str) and len(value) > 50:
                        dspy_input["problem_statement"] = value
                        break

            # Add any additional fields
            for key, value in problem_text.items():
                if key != "problem":
                    dspy_input[key] = value

            return dspy_input
        else:
            raise ValueError(f"Unsupported problem_text type: {type(problem_text)}")

    def solve_problem(
        self, problem_text: Union[str, Dict[str, Any]], **kwargs
    ) -> Dict[str, Any]:
        """
        Solve a problem using either DSPy or legacy implementation.

        Args:
            problem_text: Problem text or data
            **kwargs: Additional arguments

        Returns:
            Solution result
        """
        try:
            if self.use_dspy and self.config.is_enabled():
                # Use DSPy implementation
                dspy_module = self._get_dspy_module()
                if dspy_module is not None:
                    dspy_input = self._convert_to_dspy_format(problem_text)
                    dspy_result = dspy_module.solve(dspy_input, **kwargs)
                    return self._convert_to_legacy_format(dspy_result)

            # Fallback to legacy implementation
            legacy_agent = self._get_legacy_agent()
            return legacy_agent.solve_problem(problem_text, **kwargs)

        except Exception as e:
            self.logger.error("Error in solve_problem: %s", str(e))
            # Always try to fall back to legacy implementation
            try:
                legacy_agent = self._get_legacy_agent()
                return legacy_agent.solve_problem(problem_text, **kwargs)
            except Exception as fallback_error:
                self.logger.error(
                    "Fallback to legacy implementation failed: %s", str(fallback_error)
                )
                raise DSPyIntegrationError(f"Failed to solve problem: {str(e)}") from e

    def evaluate_solution(
        self, problem_data: Dict[str, Any], solution_data: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        """
        Evaluate a solution using either DSPy or legacy implementation.

        Args:
            problem_data: Problem data
            solution_data: Solution data
            **kwargs: Additional arguments

        Returns:
            Evaluation result
        """
        try:
            if self.use_dspy and self.config.is_enabled():
                # Use DSPy implementation
                dspy_module = self._get_dspy_module()
                if dspy_module is not None:
                    dspy_problem = self._convert_to_dspy_format(problem_data)
                    dspy_solution = self._convert_to_dspy_format(solution_data)
                    dspy_result = dspy_module.evaluate_solution(
                        dspy_problem, dspy_solution, **kwargs
                    )
                    return dspy_result  # Evaluation results are already compatible

            # Fallback to legacy implementation
            legacy_agent = self._get_legacy_agent()
            return legacy_agent.evaluate_solution(problem_data, solution_data, **kwargs)

        except Exception as e:
            self.logger.error("Error in evaluate_solution: %s", str(e))
            # Always try to fall back to legacy implementation
            try:
                legacy_agent = self._get_legacy_agent()
                return legacy_agent.evaluate_solution(
                    problem_data, solution_data, **kwargs
                )
            except Exception as fallback_error:
                self.logger.error(
                    "Fallback to legacy implementation failed: %s", str(fallback_error)
                )
                raise DSPyIntegrationError(
                    f"Failed to evaluate solution: {str(e)}"
                ) from e


# Factory functions to create adapters
def create_engineer_agent(domain: str, use_dspy: bool = True) -> EngineerAgentAdapter:
    """Create an engineer agent adapter."""
    return EngineerAgentAdapter(domain, use_dspy)


def create_checker_agent(domain: str, use_dspy: bool = True) -> CheckerAgentAdapter:
    """Create a checker agent adapter."""
    return CheckerAgentAdapter(domain, use_dspy)


def create_target_agent(domain: str, use_dspy: bool = True) -> TargetAgentAdapter:
    """Create a target agent adapter."""
    return TargetAgentAdapter(domain, use_dspy)
