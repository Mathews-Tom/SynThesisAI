"""
Multi-Agent Reinforcement Learning Coordinator

This module implements the main MARL coordination system that orchestrates
the interaction between specialized RL agents for content generation.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.marl.agents.specialized.curriculum_agent import CurriculumRLAgent
from core.marl.agents.specialized.generator_agent import GeneratorRLAgent
from core.marl.agents.specialized.validator_agent import ValidatorRLAgent
from core.marl.coordination.communication_protocol import AgentCommunicationProtocol
from core.marl.coordination.coordination_policy import CoordinationPolicy
from utils.exceptions import AgentFailureError, CoordinationError
from utils.logging_config import get_logger


class MultiAgentRLCoordinator:
    """
    Main orchestration system for multi-agent reinforcement learning coordination.

    This class coordinates the interaction between Generator, Validator, and Curriculum
    agents to produce high-quality educational content through reinforcement learning.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the MARL coordinator.

        Args:
            config: Configuration dictionary for MARL coordination
        """
        self.config = config or {}
        self.logger = get_logger(__name__ + ".MultiAgentRLCoordinator")

        # Initialize agents
        self.generator_agent = GeneratorRLAgent(
            agent_id="generator", config=self.config.get("generator", {})
        )
        self.validator_agent = ValidatorRLAgent(
            agent_id="validator", config=self.config.get("validator", {})
        )
        self.curriculum_agent = CurriculumRLAgent(
            agent_id="curriculum", config=self.config.get("curriculum", {})
        )

        # Initialize coordination infrastructure
        self.communication_protocol = AgentCommunicationProtocol()
        self.coordination_policy = CoordinationPolicy(
            config=self.config.get("coordination", {})
        )

        # Register agents with communication protocol
        self._register_agents()

        # Coordination metrics
        self.coordination_metrics = {
            "total_requests": 0,
            "successful_coordinations": 0,
            "failed_coordinations": 0,
            "average_coordination_time": 0.0,
            "agent_utilization": {
                "generator": 0.0,
                "validator": 0.0,
                "curriculum": 0.0,
            },
        }

        self.logger.info("Initialized MultiAgentRLCoordinator with %d agents", 3)

    def _register_agents(self):
        """Register all agents with the communication protocol."""
        agents = [
            ("generator", self.generator_agent),
            ("validator", self.validator_agent),
            ("curriculum", self.curriculum_agent),
        ]

        for agent_id, agent in agents:
            self.communication_protocol.register_agent(agent_id, agent)
            self.logger.debug("Registered agent: %s", agent_id)

    async def coordinate_generation(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main coordination workflow for content generation.

        This method orchestrates the multi-agent process:
        1. Collect agent actions/proposals
        2. Coordinate actions through policy
        3. Execute coordinated actions
        4. Process and return results

        Args:
            request: Content generation request with parameters

        Returns:
            Dictionary containing generated content and coordination metadata
        """
        start_time = time.time()
        self.coordination_metrics["total_requests"] += 1

        try:
            self.logger.info(
                "Starting MARL coordination for request: %s",
                request.get("domain", "unknown"),
            )

            # Step 1: Collect agent actions
            agent_actions = await self._collect_agent_actions(request)

            # Step 2: Coordinate actions
            coordinated_action = await self._coordinate_actions(agent_actions, request)

            # Step 3: Execute coordinated actions
            result = await self._execute_coordinated_action(coordinated_action, request)

            # Step 4: Process results and update agents
            final_result = await self._process_results(
                result, coordinated_action, request
            )

            # Update metrics
            coordination_time = time.time() - start_time
            self._update_coordination_metrics(coordination_time, success=True)

            self.logger.info(
                "MARL coordination completed successfully in %.2fs", coordination_time
            )

            return final_result

        except Exception as e:
            coordination_time = time.time() - start_time
            self._update_coordination_metrics(coordination_time, success=False)

            self.logger.error(
                "MARL coordination failed after %.2fs: %s", coordination_time, str(e)
            )
            raise CoordinationError(
                f"MARL coordination failed: {e}",
                coordination_time=coordination_time,
                request_summary=self._summarize_request(request),
            ) from e

    async def _collect_agent_actions(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Collect actions/proposals from all agents.

        Args:
            request: Content generation request

        Returns:
            Dictionary mapping agent IDs to their proposed actions
        """
        self.logger.debug("Collecting agent actions for coordination")

        # Collect actions concurrently
        tasks = {
            "generator": self._get_generator_action(request),
            "validator": self._get_validator_action(request),
            "curriculum": self._get_curriculum_action(request),
        }

        # Wait for all agents to respond
        agent_actions = {}
        for agent_id, task in tasks.items():
            try:
                action = await task
                agent_actions[agent_id] = action
                self.logger.debug("Collected action from %s agent", agent_id)
            except Exception as e:
                self.logger.warning(
                    "Failed to collect action from %s agent: %s", agent_id, str(e)
                )
                # Use fallback action
                agent_actions[agent_id] = self._get_fallback_action(agent_id, request)

        return agent_actions

    async def _get_generator_action(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Get action proposal from generator agent."""
        # Get generator's strategy selection
        state = self.generator_agent.get_state_representation(request)
        action_index = self.generator_agent.select_action(state, training=True)
        selected_strategy = self.generator_agent.strategies[action_index]

        return {
            "agent_id": "generator",
            "action_type": "generation_strategy",
            "strategy": selected_strategy.name,
            "parameters": selected_strategy.parameters.copy(),
            "confidence": self.generator_agent.get_action_confidence(
                state, action_index
            ),
            "state_summary": self.generator_agent.summarize_state(state),
        }

    async def _get_validator_action(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Get action proposal from validator agent."""
        # Get validator's strategy selection
        state = self.validator_agent.get_state_representation(request)
        action_index = self.validator_agent.select_action(state, training=True)
        selected_strategy = self.validator_agent.strategies[action_index]

        return {
            "agent_id": "validator",
            "action_type": "validation_strategy",
            "strategy": selected_strategy.name,
            "parameters": selected_strategy.parameters.copy(),
            "confidence": self.validator_agent.get_action_confidence(
                state, action_index
            ),
            "threshold": self.validator_agent._get_threshold_for_strategy(
                selected_strategy
            ),
        }

    async def _get_curriculum_action(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Get action proposal from curriculum agent."""
        # Get curriculum's strategy selection
        state = self.curriculum_agent.get_state_representation(request)
        action_index = self.curriculum_agent.select_action(state, training=True)
        selected_strategy = self.curriculum_agent.strategies[action_index]

        return {
            "agent_id": "curriculum",
            "action_type": "curriculum_strategy",
            "strategy": selected_strategy.name,
            "parameters": selected_strategy.parameters.copy(),
            "confidence": self.curriculum_agent.get_action_confidence(
                state, action_index
            ),
            "pedagogical_focus": selected_strategy.parameters.get("focus", "general"),
        }

    def _get_fallback_action(
        self, agent_id: str, request: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate fallback action when agent fails to respond."""
        fallback_strategies = {
            "generator": {
                "strategy": "structured_reasoning",
                "parameters": {"structure_weight": 0.7},
                "confidence": 0.5,
            },
            "validator": {
                "strategy": "standard_validation_medium_threshold",
                "parameters": {"threshold": 0.7},
                "confidence": 0.5,
            },
            "curriculum": {
                "strategy": "linear_progression",
                "parameters": {"sequence_strength": 0.8},
                "confidence": 0.5,
            },
        }

        fallback = fallback_strategies.get(agent_id, {})
        return {
            "agent_id": agent_id,
            "action_type": f"{agent_id}_strategy",
            "strategy": fallback.get("strategy", "default"),
            "parameters": fallback.get("parameters", {}),
            "confidence": fallback.get("confidence", 0.3),
            "is_fallback": True,
        }

    async def _coordinate_actions(
        self, agent_actions: Dict[str, Any], request: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Coordinate agent actions through the coordination policy.

        Args:
            agent_actions: Actions proposed by each agent
            request: Original content generation request

        Returns:
            Coordinated action plan
        """
        self.logger.debug("Coordinating agent actions through policy")

        # Build state context for coordination
        state = {
            "domain": request.get("domain", "general"),
            "difficulty_level": request.get("difficulty_level", "medium"),
            "learning_objectives": request.get("learning_objectives", []),
            "target_audience": request.get("target_audience", "students"),
        }

        # Build agent context
        agent_context = {
            "agent_performance": {
                "generator": self.generator_agent.get_performance_summary(),
                "validator": self.validator_agent.get_performance_summary(),
                "curriculum": self.curriculum_agent.get_performance_summary(),
            }
        }

        # Use coordination policy to coordinate actions
        coordinated_action = await self.coordination_policy.coordinate(
            agent_actions, state, agent_context
        )

        self.logger.debug(
            "Actions coordinated with strategy: %s",
            coordinated_action.coordination_strategy,
        )

        return coordinated_action

    async def _execute_coordinated_action(
        self, coordinated_action: Any, request: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute the coordinated action plan.

        Args:
            coordinated_action: Coordinated action from policy
            request: Original content generation request

        Returns:
            Execution results
        """
        self.logger.debug("Executing coordinated action plan")

        try:
            # Extract coordinated strategies
            generator_strategy = coordinated_action.generator_strategy
            validator_strategy = coordinated_action.validation_criteria
            curriculum_strategy = coordinated_action.curriculum_guidance

            # Step 1: Generate content using coordinated generator strategy
            content = await self._execute_generation(generator_strategy, request)

            # Step 2: Get curriculum improvements
            curriculum_improvements = await self._execute_curriculum_guidance(
                curriculum_strategy, request, content
            )

            # Step 3: Validate content using coordinated validator strategy
            validation_result = await self._execute_validation(
                validator_strategy, content, request
            )

            # Combine results
            execution_result = {
                "content": content,
                "curriculum_improvements": curriculum_improvements,
                "validation_result": validation_result,
                "coordination_metadata": {
                    "strategy": coordinated_action.coordination_strategy,
                    "confidence": coordinated_action.confidence,
                    "quality_score": coordinated_action.quality,
                    "conflict_resolution_applied": coordinated_action.conflict_resolution_applied,
                },
            }

            return execution_result

        except Exception as e:
            self.logger.error("Failed to execute coordinated action: %s", str(e))
            raise CoordinationError(
                f"Action execution failed: {e}", action_summary=str(coordinated_action)
            ) from e

    async def _execute_generation(
        self, generator_strategy: Dict[str, Any], request: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute content generation with coordinated strategy."""
        # Use the generator agent's strategy selection
        strategy_name = generator_strategy.get("strategy", "structured_reasoning")

        # Find the strategy in the generator agent
        selected_strategy = None
        for strategy in self.generator_agent.strategies:
            if strategy.name == strategy_name:
                selected_strategy = strategy
                break

        if not selected_strategy:
            # Fallback to first strategy
            selected_strategy = self.generator_agent.strategies[0]
            self.logger.warning(
                "Strategy %s not found, using fallback: %s",
                strategy_name,
                selected_strategy.name,
            )

        # Generate content using the selected strategy
        content = self.generator_agent.generate_content_with_strategy(
            request, selected_strategy
        )

        return content

    async def _execute_curriculum_guidance(
        self,
        curriculum_strategy: Dict[str, Any],
        request: Dict[str, Any],
        content: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute curriculum guidance with coordinated strategy."""
        # Enhance request with generated content context
        enhanced_request = request.copy()
        enhanced_request["generated_content"] = content

        # Get curriculum improvements
        improvements = self.curriculum_agent.suggest_curriculum_improvements(
            enhanced_request
        )

        return improvements

    async def _execute_validation(
        self,
        validator_strategy: Dict[str, Any],
        content: Dict[str, Any],
        request: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute content validation with coordinated strategy."""
        # Build environment state for validation
        environment_state = {
            "domain": request.get("domain", "general"),
            "difficulty_level": request.get("difficulty_level", "medium"),
            "learning_objectives": request.get("learning_objectives", []),
        }

        # Validate content
        validation_result = self.validator_agent.predict_quality_and_provide_feedback(
            content, environment_state
        )

        return validation_result

    async def _process_results(
        self, result: Dict[str, Any], coordinated_action: Any, request: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process execution results and update agent learning.

        Args:
            result: Execution results
            coordinated_action: The coordinated action that was executed
            request: Original request

        Returns:
            Final processed results
        """
        self.logger.debug("Processing results and updating agent learning")

        # Calculate rewards based on results
        rewards = self._calculate_rewards(result, request)

        # Update agent policies with rewards
        await self._update_agent_policies(rewards, coordinated_action, request)

        # Build final result
        final_result = {
            "content": result["content"],
            "validation": result["validation_result"],
            "curriculum_guidance": result["curriculum_improvements"],
            "coordination_metadata": result["coordination_metadata"],
            "learning_updates": {"rewards": rewards, "coordination_success": True},
        }

        return final_result

    def _calculate_rewards(
        self, result: Dict[str, Any], request: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate rewards for each agent based on results."""
        validation_result = result.get("validation_result", {})
        curriculum_improvements = result.get("curriculum_improvements", {})

        # Base rewards
        rewards = {"generator": 0.0, "validator": 0.0, "curriculum": 0.0}

        # Generator reward based on validation quality
        quality_prediction = validation_result.get("quality_prediction", 0.5)
        passes_threshold = validation_result.get("passes_threshold", False)

        rewards["generator"] = quality_prediction
        if passes_threshold:
            rewards["generator"] += 0.2  # Bonus for passing validation

        # Validator reward based on confidence and accuracy
        validator_confidence = validation_result.get("confidence", 0.5)
        rewards["validator"] = validator_confidence

        # Curriculum reward based on alignment and coherence
        curriculum_confidence = curriculum_improvements.get("confidence", 0.5)
        objective_alignment = curriculum_improvements.get("objective_alignment", {})
        alignment_score = objective_alignment.get("alignment_score", 0.5)

        rewards["curriculum"] = (curriculum_confidence + alignment_score) / 2

        return rewards

    async def _update_agent_policies(
        self,
        rewards: Dict[str, float],
        coordinated_action: Any,
        request: Dict[str, Any],
    ):
        """Update agent policies with calculated rewards."""
        # Update generator agent
        generator_state = self.generator_agent.get_state_representation(request)
        generator_action = self._find_strategy_index(
            self.generator_agent.strategies,
            coordinated_action.generator_strategy.get("strategy", ""),
        )
        next_generator_state = np.random.random(len(generator_state)).astype(np.float32)

        self.generator_agent.update_policy(
            generator_state,
            generator_action,
            rewards["generator"],
            next_generator_state,
            False,  # Not done
        )

        # Update validator agent
        validator_state = self.validator_agent.get_state_representation(request)
        validator_action = self._find_strategy_index(
            self.validator_agent.strategies,
            coordinated_action.validation_criteria.get("strategy", ""),
        )
        next_validator_state = np.random.random(len(validator_state)).astype(np.float32)

        self.validator_agent.update_policy(
            validator_state,
            validator_action,
            rewards["validator"],
            next_validator_state,
            False,
        )

        # Update curriculum agent
        curriculum_state = self.curriculum_agent.get_state_representation(request)
        curriculum_action = self._find_strategy_index(
            self.curriculum_agent.strategies,
            coordinated_action.curriculum_guidance.get("strategy", ""),
        )
        next_curriculum_state = np.random.random(len(curriculum_state)).astype(
            np.float32
        )

        self.curriculum_agent.update_policy(
            curriculum_state,
            curriculum_action,
            rewards["curriculum"],
            next_curriculum_state,
            False,
        )

    def _find_strategy_index(self, strategies: List[Any], strategy_name: str) -> int:
        """Find the index of a strategy by name."""
        for i, strategy in enumerate(strategies):
            if strategy.name == strategy_name:
                return i
        return 0  # Default to first strategy if not found

    def _update_coordination_metrics(self, coordination_time: float, success: bool):
        """Update coordination performance metrics."""
        if success:
            self.coordination_metrics["successful_coordinations"] += 1
        else:
            self.coordination_metrics["failed_coordinations"] += 1

        # Update average coordination time
        total_coordinations = (
            self.coordination_metrics["successful_coordinations"]
            + self.coordination_metrics["failed_coordinations"]
        )

        current_avg = self.coordination_metrics["average_coordination_time"]
        self.coordination_metrics["average_coordination_time"] = (
            current_avg * (total_coordinations - 1) + coordination_time
        ) / total_coordinations

    def _summarize_request(self, request: Dict[str, Any]) -> str:
        """Create a summary of the request for logging."""
        return (
            f"domain={request.get('domain', 'unknown')}, "
            f"difficulty={request.get('difficulty_level', 'unknown')}, "
            f"objectives_count={len(request.get('learning_objectives', []))}"
        )

    def get_coordination_success_rate(self) -> float:
        """Get the current coordination success rate."""
        total = (
            self.coordination_metrics["successful_coordinations"]
            + self.coordination_metrics["failed_coordinations"]
        )
        if total == 0:
            return 0.0
        return self.coordination_metrics["successful_coordinations"] / total

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        return {
            "coordination_metrics": self.coordination_metrics.copy(),
            "success_rate": self.get_coordination_success_rate(),
            "agent_performance": {
                "generator": self.generator_agent.get_performance_summary(),
                "validator": self.validator_agent.get_performance_summary(),
                "curriculum": self.curriculum_agent.get_performance_summary(),
            },
        }

    async def shutdown(self):
        """Gracefully shutdown the coordinator."""
        self.logger.info("Shutting down MARL coordinator")

        # Shutdown communication protocol
        await self.communication_protocol.shutdown()

        # Save agent checkpoints
        try:
            self.generator_agent.save_checkpoint("generator_final")
            self.validator_agent.save_checkpoint("validator_final")
            self.curriculum_agent.save_checkpoint("curriculum_final")
            self.logger.info("Agent checkpoints saved successfully")
        except Exception as e:
            self.logger.warning("Failed to save agent checkpoints: %s", str(e))
