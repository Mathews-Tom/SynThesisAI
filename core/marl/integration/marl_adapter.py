"""
MARL Integration Adapter

This module provides integration adapters to connect the MARL coordinator
with the existing SynThesisAI architecture and orchestration system.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

from core.marl.coordination.marl_coordinator import MultiAgentRLCoordinator
from utils.costs import CostTracker
from utils.exceptions import CoordinationError, PipelineError
from utils.logging_config import get_logger


class MARLOrchestrationAdapter:
    """
    Adapter to integrate MARL coordinator with existing orchestration system.

    This adapter provides compatibility between the MARL coordination system
    and the existing generation pipeline, allowing for gradual migration.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the MARL orchestration adapter.

        Args:
            config: Configuration dictionary for MARL integration
        """
        self.config = config or {}
        self.logger = get_logger(__name__ + ".MARLOrchestrationAdapter")

        # Initialize MARL coordinator
        marl_config = self.config.get("marl", {})
        self.marl_coordinator = MultiAgentRLCoordinator(marl_config)

        # Integration settings
        self.enabled = self.config.get("marl_enabled", True)
        self.fallback_enabled = self.config.get("fallback_enabled", True)
        self.success_threshold = self.config.get("success_threshold", 0.7)

        # Performance tracking
        self.integration_metrics = {
            "marl_requests": 0,
            "marl_successes": 0,
            "fallback_requests": 0,
            "total_coordination_time": 0.0,
        }

        self.logger.info(
            "Initialized MARL orchestration adapter (enabled=%s, fallback=%s)",
            self.enabled,
            self.fallback_enabled,
        )

    async def generate_content_with_marl(
        self, request: Dict[str, Any], cost_tracker: Optional[CostTracker] = None
    ) -> Dict[str, Any]:
        """
        Generate content using MARL coordination.

        Args:
            request: Content generation request
            cost_tracker: Optional cost tracking instance

        Returns:
            Generated content with MARL coordination metadata
        """
        if not self.enabled:
            raise PipelineError(
                "MARL coordination is disabled", stage="marl_generation"
            )

        self.integration_metrics["marl_requests"] += 1

        try:
            # Convert legacy request format to MARL format
            marl_request = self._convert_to_marl_request(request)

            # Coordinate generation through MARL
            result = await self.marl_coordinator.coordinate_generation(marl_request)

            # Convert MARL result back to legacy format
            legacy_result = self._convert_to_legacy_result(result, request)

            # Track costs if cost_tracker provided
            if cost_tracker:
                self._track_marl_costs(result, cost_tracker)

            self.integration_metrics["marl_successes"] += 1
            self.integration_metrics["total_coordination_time"] += result.get(
                "coordination_metadata", {}
            ).get("coordination_time", 0.0)

            self.logger.info("MARL content generation completed successfully")
            return legacy_result

        except Exception as e:
            self.logger.error("MARL content generation failed: %s", str(e))
            raise CoordinationError(
                f"MARL generation failed: {e}",
                request_summary=self._summarize_request(request),
            ) from e

    def _convert_to_marl_request(
        self, legacy_request: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Convert legacy request format to MARL request format.

        Args:
            legacy_request: Request in legacy format

        Returns:
            Request in MARL format
        """
        # Extract domain information
        domain = legacy_request.get("domain", "general")
        if not domain or domain == "general":
            # Try to infer domain from other fields
            subject = legacy_request.get("subject", "").lower()
            topic = legacy_request.get("topic", "").lower()

            # Mathematics domain keywords
            math_keywords = [
                "math",
                "algebra",
                "calculus",
                "geometry",
                "statistics",
                "arithmetic",
            ]
            # Science domain keywords
            science_keywords = [
                "science",
                "physics",
                "chemistry",
                "biology",
                "photosynthesis",
            ]
            # Technology domain keywords
            tech_keywords = ["technology", "programming", "computer", "software"]

            if any(keyword in subject or keyword in topic for keyword in math_keywords):
                domain = "mathematics"
            elif any(
                keyword in subject or keyword in topic for keyword in science_keywords
            ):
                domain = "science"
            elif any(
                keyword in subject or keyword in topic for keyword in tech_keywords
            ):
                domain = "technology"
            else:
                domain = "general"

        # Extract difficulty level
        difficulty_level = legacy_request.get("difficulty_level", "medium")
        if not difficulty_level:
            difficulty_level = "medium"

        # Extract learning objectives
        learning_objectives = legacy_request.get("learning_objectives", [])
        if not learning_objectives:
            # Try to create objectives from topic/subject
            topic = legacy_request.get("topic", "")
            subject = legacy_request.get("subject", "")
            if topic:
                learning_objectives = [f"Understand {topic}"]
            elif subject:
                learning_objectives = [f"Learn {subject}"]
            else:
                learning_objectives = ["General learning objective"]

        # Extract target audience
        target_audience = legacy_request.get("target_audience", "students")

        marl_request = {
            "domain": domain,
            "difficulty_level": difficulty_level,
            "learning_objectives": learning_objectives,
            "target_audience": target_audience,
            "topic": legacy_request.get("topic", ""),
            "subject": legacy_request.get("subject", ""),
            "original_request": legacy_request,  # Keep original for reference
        }

        self.logger.debug(
            "Converted legacy request to MARL format: domain=%s, difficulty=%s",
            domain,
            difficulty_level,
        )

        return marl_request

    def _convert_to_legacy_result(
        self, marl_result: Dict[str, Any], original_request: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Convert MARL result back to legacy format.

        Args:
            marl_result: Result from MARL coordination
            original_request: Original legacy request

        Returns:
            Result in legacy format
        """
        content = marl_result.get("content", {})
        validation = marl_result.get("validation", {})
        curriculum_guidance = marl_result.get("curriculum_guidance", {})
        coordination_metadata = marl_result.get("coordination_metadata", {})

        # Build legacy result structure
        legacy_result = {
            # Core content
            "problem_statement": content.get(
                "text", content.get("problem_statement", "")
            ),
            "solution": content.get("solution", ""),
            "quality_score": validation.get("quality_prediction", 0.5),
            # Validation information
            "passes_validation": validation.get("passes_threshold", False),
            "validation_feedback": validation.get("feedback", {}),
            "validation_confidence": validation.get("confidence", 0.5),
            # Curriculum information
            "curriculum_improvements": curriculum_guidance,
            "pedagogical_hints": curriculum_guidance.get("pedagogical_hints", []),
            "difficulty_adjustments": curriculum_guidance.get(
                "difficulty_adjustments", {}
            ),
            # MARL metadata
            "marl_coordination": {
                "strategy": coordination_metadata.get("strategy", "unknown"),
                "confidence": coordination_metadata.get("confidence", 0.5),
                "quality_score": coordination_metadata.get("quality_score", 0.5),
                "conflict_resolution_applied": coordination_metadata.get(
                    "conflict_resolution_applied", False
                ),
            },
            # Learning updates
            "learning_updates": marl_result.get("learning_updates", {}),
            # Original request context
            "request_context": {
                "domain": original_request.get("domain", "general"),
                "difficulty_level": original_request.get("difficulty_level", "medium"),
                "topic": original_request.get("topic", ""),
                "subject": original_request.get("subject", ""),
            },
        }

        # Add any additional content fields
        for key, value in content.items():
            if key not in ["text", "problem_statement", "solution"]:
                legacy_result[key] = value

        return legacy_result

    def _track_marl_costs(self, marl_result: Dict[str, Any], cost_tracker: CostTracker):
        """
        Track costs associated with MARL coordination.

        Args:
            marl_result: Result from MARL coordination
            cost_tracker: Cost tracking instance
        """
        # Estimate costs based on coordination complexity
        coordination_metadata = marl_result.get("coordination_metadata", {})

        # Base cost for coordination overhead
        base_cost = 0.001  # Small base cost

        # Additional cost based on strategy complexity
        strategy = coordination_metadata.get("strategy", "simple")
        strategy_multiplier = {
            "consensus": 1.2,
            "expert_priority": 1.1,
            "confidence_weighted": 1.0,
            "performance_based": 1.3,
            "adaptive_consensus": 1.5,
        }.get(strategy, 1.0)

        # Additional cost if conflict resolution was applied
        conflict_cost = (
            0.0005
            if coordination_metadata.get("conflict_resolution_applied", False)
            else 0.0
        )

        total_cost = base_cost * strategy_multiplier + conflict_cost

        # Log the cost
        cost_tracker.log_cost(
            "marl_coordination",
            total_cost,
            {
                "strategy": strategy,
                "conflict_resolution": coordination_metadata.get(
                    "conflict_resolution_applied", False
                ),
                "confidence": coordination_metadata.get("confidence", 0.5),
            },
        )

    def _summarize_request(self, request: Dict[str, Any]) -> str:
        """Create a summary of the request for logging."""
        return (
            f"domain={request.get('domain', 'unknown')}, "
            f"subject={request.get('subject', 'unknown')}, "
            f"topic={request.get('topic', 'unknown')}"
        )

    def get_integration_success_rate(self) -> float:
        """Get the MARL integration success rate."""
        if self.integration_metrics["marl_requests"] == 0:
            return 0.0
        return (
            self.integration_metrics["marl_successes"]
            / self.integration_metrics["marl_requests"]
        )

    def get_integration_metrics(self) -> Dict[str, Any]:
        """Get comprehensive integration metrics."""
        success_rate = self.get_integration_success_rate()
        avg_coordination_time = 0.0

        if self.integration_metrics["marl_successes"] > 0:
            avg_coordination_time = (
                self.integration_metrics["total_coordination_time"]
                / self.integration_metrics["marl_successes"]
            )

        return {
            "integration_metrics": self.integration_metrics.copy(),
            "success_rate": success_rate,
            "average_coordination_time": avg_coordination_time,
            "marl_coordinator_performance": self.marl_coordinator.get_performance_summary(),
        }

    async def shutdown(self):
        """Gracefully shutdown the adapter."""
        self.logger.info("Shutting down MARL orchestration adapter")
        await self.marl_coordinator.shutdown()


class MARLPipelineIntegration:
    """
    Integration layer for MARL with the existing generation pipeline.

    This class provides methods to integrate MARL coordination into the
    existing pipeline workflow with fallback mechanisms.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the MARL pipeline integration.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = get_logger(__name__ + ".MARLPipelineIntegration")

        # Initialize MARL adapter
        self.marl_adapter = MARLOrchestrationAdapter(config)

        # Integration settings
        self.use_marl = self.config.get("use_marl", False)
        self.marl_probability = self.config.get("marl_probability", 1.0)
        self.fallback_on_failure = self.config.get("fallback_on_failure", True)

        self.logger.info(
            "Initialized MARL pipeline integration (use_marl=%s, probability=%.2f)",
            self.use_marl,
            self.marl_probability,
        )

    async def generate_with_marl_integration(
        self,
        request: Dict[str, Any],
        cost_tracker: CostTracker,
        fallback_generator: Optional[callable] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate content with MARL integration and fallback support.

        Args:
            request: Content generation request
            cost_tracker: Cost tracking instance
            fallback_generator: Fallback generation function

        Returns:
            Tuple of (result_type, result_data)
        """
        # Check if MARL should be used
        if not self.use_marl:
            if fallback_generator:
                return await self._run_fallback_generation(
                    request, cost_tracker, fallback_generator
                )
            else:
                raise PipelineError(
                    "MARL disabled and no fallback generator provided",
                    stage="marl_integration",
                )

        # Determine if this request should use MARL (for A/B testing)
        import random

        use_marl_for_request = random.random() < self.marl_probability

        if not use_marl_for_request:
            self.logger.debug("Request selected for non-MARL generation (A/B testing)")
            if fallback_generator:
                return await self._run_fallback_generation(
                    request, cost_tracker, fallback_generator
                )

        # Try MARL generation
        try:
            self.logger.info("Attempting MARL-coordinated generation")
            result = await self.marl_adapter.generate_content_with_marl(
                request, cost_tracker
            )

            return ("marl_success", result)

        except Exception as e:
            self.logger.warning("MARL generation failed: %s", str(e))

            if self.fallback_on_failure and fallback_generator:
                self.logger.info("Falling back to legacy generation")
                return await self._run_fallback_generation(
                    request, cost_tracker, fallback_generator
                )
            else:
                # Re-raise the error if no fallback
                raise PipelineError(
                    f"MARL generation failed and fallback disabled: {e}",
                    stage="marl_generation",
                ) from e

    async def _run_fallback_generation(
        self,
        request: Dict[str, Any],
        cost_tracker: CostTracker,
        fallback_generator: callable,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Run fallback generation using legacy system.

        Args:
            request: Content generation request
            cost_tracker: Cost tracking instance
            fallback_generator: Fallback generation function

        Returns:
            Tuple of (result_type, result_data)
        """
        try:
            # Run fallback generation
            if asyncio.iscoroutinefunction(fallback_generator):
                result = await fallback_generator(request, cost_tracker)
            else:
                result = fallback_generator(request, cost_tracker)

            # Add fallback metadata
            if isinstance(result, tuple) and len(result) == 2:
                result_type, result_data = result
                if isinstance(result_data, dict):
                    result_data["generation_method"] = "fallback"
                return (result_type, result_data)
            else:
                return ("fallback_success", result)

        except Exception as e:
            self.logger.error("Fallback generation also failed: %s", str(e))
            raise PipelineError(
                f"Both MARL and fallback generation failed: {e}",
                stage="fallback_generation",
            ) from e

    def get_integration_status(self) -> Dict[str, Any]:
        """Get current integration status and metrics."""
        return {
            "marl_enabled": self.use_marl,
            "marl_probability": self.marl_probability,
            "fallback_enabled": self.fallback_on_failure,
            "adapter_metrics": self.marl_adapter.get_integration_metrics(),
        }

    async def shutdown(self):
        """Gracefully shutdown the integration."""
        self.logger.info("Shutting down MARL pipeline integration")
        await self.marl_adapter.shutdown()


def create_marl_integration(config: Dict[str, Any]) -> MARLPipelineIntegration:
    """
    Factory function to create MARL pipeline integration.

    Args:
        config: Configuration dictionary

    Returns:
        Configured MARL pipeline integration instance
    """
    return MARLPipelineIntegration(config)
