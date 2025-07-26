"""
Conflict Resolver

This module implements conflict resolution mechanisms for multi-agent RL coordination,
including negotiation strategies, priority-based resolution, and compromise generation.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..config import CoordinationConfig
from ..exceptions import ConflictResolutionError
from ..logging_config import get_marl_logger

logger = logging.getLogger(__name__)


class ConflictResolver:
    """
    Conflict resolution system for multi-agent coordination.

    This class implements various strategies for resolving conflicts between
    agent proposals, including negotiation, priority-based resolution, and compromise.
    """

    def __init__(self, config: CoordinationConfig):
        """
        Initialize conflict resolver.

        Args:
            config: Coordination configuration parameters
        """
        self.config = config
        self.logger = get_marl_logger("conflict_resolver")

        # Resolution strategies
        self.resolution_strategies = {
            "weighted_priority": self._weighted_priority_resolution,
            "negotiation_based": self._negotiation_based_resolution,
            "compromise_generation": self._compromise_generation,
            "expert_override": self._expert_override_resolution,
            "performance_based": self._performance_based_resolution,
        }

        # Resolution history and metrics
        self.resolution_history = []
        self.resolution_metrics = {
            "total_conflicts": 0,
            "resolved_conflicts": 0,
            "resolution_strategy_usage": {
                strategy: 0 for strategy in self.resolution_strategies
            },
            "average_resolution_time": 0.0,
            "resolution_success_rate": 0.0,
        }

        # Agent priority weights (can be adjusted based on performance)
        self.agent_priorities = {
            "generator": 1.0,
            "validator": 1.2,  # Slightly higher priority for quality control
            "curriculum": 1.1,  # Slightly higher priority for pedagogical coherence
        }

        self.logger.log_agent_action(
            "conflict_resolver",
            "initialized",
            1.0,
            "Strategies: %d" % len(self.resolution_strategies),
        )

    async def resolve(
        self,
        conflicts: Dict[str, List[str]],
        proposals: List[Any],
        state: Dict[str, Any],
        agent_context: Dict[str, Any],
    ) -> Optional[List[Any]]:
        """
        Resolve conflicts between agent proposals.

        Args:
            conflicts: Dictionary mapping conflict types to involved agents
            proposals: List of agent proposals
            state: Current environment state
            agent_context: Additional context about agents

        Returns:
            List of resolved proposals, or None if resolution failed

        Raises:
            ConflictResolutionError: If resolution fails critically
        """
        resolution_start_time = time.time()

        try:
            self.resolution_metrics["total_conflicts"] += 1

            # Select resolution strategy based on conflict types and context
            strategy = self._select_resolution_strategy(conflicts, proposals, state)

            self.logger.log_coordination_event(
                "conflict_resolution_started",
                {
                    "strategy": strategy,
                    "conflict_types": list(conflicts.keys()),
                    "agents_involved": self._get_all_involved_agents(conflicts),
                },
            )

            # Apply resolution strategy
            resolution_func = self.resolution_strategies[strategy]
            resolved_proposals = await resolution_func(
                conflicts, proposals, state, agent_context
            )

            # Validate resolution
            if resolved_proposals and self._validate_resolution(
                resolved_proposals, conflicts
            ):
                resolution_time = time.time() - resolution_start_time
                self._record_resolution_success(strategy, resolution_time)

                self.logger.log_coordination_event(
                    "conflict_resolution_success",
                    {
                        "strategy": strategy,
                        "resolution_time": resolution_time,
                        "resolved_proposals_count": len(resolved_proposals),
                    },
                )

                return resolved_proposals
            else:
                # Resolution failed, try fallback strategy
                fallback_strategy = self._get_fallback_strategy(strategy)
                if fallback_strategy and fallback_strategy != strategy:
                    self.logger.log_coordination_event(
                        "conflict_resolution_fallback",
                        {
                            "original_strategy": strategy,
                            "fallback_strategy": fallback_strategy,
                        },
                    )

                    fallback_func = self.resolution_strategies[fallback_strategy]
                    resolved_proposals = await fallback_func(
                        conflicts, proposals, state, agent_context
                    )

                    if resolved_proposals and self._validate_resolution(
                        resolved_proposals, conflicts
                    ):
                        resolution_time = time.time() - resolution_start_time
                        self._record_resolution_success(
                            fallback_strategy, resolution_time
                        )
                        return resolved_proposals

                # All resolution attempts failed
                resolution_time = time.time() - resolution_start_time
                self._record_resolution_failure(strategy, resolution_time)
                return None

        except Exception as e:
            resolution_time = time.time() - resolution_start_time
            self._record_resolution_failure("error", resolution_time)

            self.logger.log_error_with_context(
                e,
                {
                    "conflicts": conflicts,
                    "proposals_count": len(proposals),
                    "resolution_time": resolution_time,
                },
            )

            raise ConflictResolutionError(
                "Critical failure during conflict resolution",
                conflict_types=list(conflicts.keys()),
                agents_involved=self._get_all_involved_agents(conflicts),
                resolution_context={
                    "resolution_time": resolution_time,
                    "error_type": type(e).__name__,
                },
            ) from e

    def _select_resolution_strategy(
        self,
        conflicts: Dict[str, List[str]],
        proposals: List[Any],
        state: Dict[str, Any],
    ) -> str:
        """Select appropriate resolution strategy based on conflict characteristics."""
        conflict_types = set(conflicts.keys())

        # Strategy selection logic based on conflict types
        if "strategy_incompatibility" in conflict_types:
            return "compromise_generation"
        elif "low_confidence" in conflict_types:
            return "performance_based"
        elif "parameter_mismatch" in conflict_types:
            return "negotiation_based"
        elif "quality_requirements" in conflict_types:
            return "expert_override"
        else:
            return "weighted_priority"

    async def _weighted_priority_resolution(
        self,
        conflicts: Dict[str, List[str]],
        proposals: List[Any],
        state: Dict[str, Any],
        agent_context: Dict[str, Any],
    ) -> List[Any]:
        """Resolve conflicts using weighted priority system."""
        resolved_proposals = []

        # Calculate weighted scores for each proposal
        proposal_scores = []
        for proposal in proposals:
            agent_priority = self.agent_priorities.get(proposal.agent_id, 1.0)
            confidence_weight = proposal.confidence
            performance_weight = self._get_agent_performance_weight(
                proposal.agent_id, agent_context
            )

            total_score = agent_priority * confidence_weight * performance_weight
            proposal_scores.append((proposal, total_score))

        # Sort by score and select top proposals
        proposal_scores.sort(key=lambda x: x[1], reverse=True)

        # Select proposals that don't conflict with the highest-scoring one
        primary_proposal = proposal_scores[0][0]
        resolved_proposals.append(primary_proposal)

        for proposal, score in proposal_scores[1:]:
            if not self._proposals_conflict(primary_proposal, proposal):
                resolved_proposals.append(proposal)

        return resolved_proposals

    async def _negotiation_based_resolution(
        self,
        conflicts: Dict[str, List[str]],
        proposals: List[Any],
        state: Dict[str, Any],
        agent_context: Dict[str, Any],
    ) -> List[Any]:
        """Resolve conflicts through iterative negotiation."""
        resolved_proposals = proposals.copy()
        max_rounds = self.config.max_negotiation_rounds

        for round_num in range(max_rounds):
            # Identify conflicting proposals
            conflicting_pairs = self._identify_conflicting_pairs(resolved_proposals)

            if not conflicting_pairs:
                break  # No more conflicts

            # Negotiate each conflicting pair
            for prop1, prop2 in conflicting_pairs:
                negotiated_proposals = self._negotiate_proposal_pair(
                    prop1, prop2, state
                )

                # Replace original proposals with negotiated ones
                resolved_proposals = [
                    p for p in resolved_proposals if p not in [prop1, prop2]
                ]
                resolved_proposals.extend(negotiated_proposals)

        return resolved_proposals

    async def _compromise_generation(
        self,
        conflicts: Dict[str, List[str]],
        proposals: List[Any],
        state: Dict[str, Any],
        agent_context: Dict[str, Any],
    ) -> List[Any]:
        """Generate compromise proposals that satisfy multiple agents."""
        compromise_proposals = []

        # Group proposals by agent
        agent_proposals = {}
        for proposal in proposals:
            agent_proposals[proposal.agent_id] = proposal

        # Generate compromise for each agent type
        if "generator" in agent_proposals and "validator" in agent_proposals:
            gen_val_compromise = self._create_generator_validator_compromise(
                agent_proposals["generator"], agent_proposals["validator"]
            )
            compromise_proposals.append(gen_val_compromise)

        if "curriculum" in agent_proposals:
            curriculum_proposal = agent_proposals["curriculum"]
            # Adjust curriculum proposal to be compatible with compromises
            adjusted_curriculum = self._adjust_curriculum_for_compromise(
                curriculum_proposal, compromise_proposals
            )
            compromise_proposals.append(adjusted_curriculum)

        return compromise_proposals

    async def _expert_override_resolution(
        self,
        conflicts: Dict[str, List[str]],
        proposals: List[Any],
        state: Dict[str, Any],
        agent_context: Dict[str, Any],
    ) -> List[Any]:
        """Resolve conflicts by giving priority to expert agents."""
        # Define expert priority order for different domains
        domain = state.get("domain", "general")

        expert_priority = {
            "mathematics": ["curriculum", "validator", "generator"],
            "science": ["curriculum", "validator", "generator"],
            "general": ["validator", "curriculum", "generator"],
        }

        priority_order = expert_priority.get(domain, expert_priority["general"])

        # Select proposals in priority order
        resolved_proposals = []
        used_agents = set()

        for agent_type in priority_order:
            for proposal in proposals:
                if (
                    proposal.agent_id == agent_type
                    and proposal.agent_id not in used_agents
                ):
                    resolved_proposals.append(proposal)
                    used_agents.add(proposal.agent_id)
                    break

        return resolved_proposals

    async def _performance_based_resolution(
        self,
        conflicts: Dict[str, List[str]],
        proposals: List[Any],
        state: Dict[str, Any],
        agent_context: Dict[str, Any],
    ) -> List[Any]:
        """Resolve conflicts based on historical agent performance."""
        # Calculate performance scores for each agent
        performance_scores = {}
        for proposal in proposals:
            performance_scores[proposal.agent_id] = (
                self._calculate_agent_performance_score(
                    proposal.agent_id, agent_context
                )
            )

        # Filter out low-confidence proposals from low-performing agents
        resolved_proposals = []
        for proposal in proposals:
            agent_performance = performance_scores.get(proposal.agent_id, 0.5)

            # Accept proposal if either high confidence or high performance
            if proposal.confidence > 0.7 or agent_performance > 0.8:
                resolved_proposals.append(proposal)
            elif proposal.confidence > 0.5 and agent_performance > 0.6:
                # Reduce confidence for marginal proposals
                adjusted_proposal = self._create_adjusted_proposal(
                    proposal, confidence_factor=0.8
                )
                resolved_proposals.append(adjusted_proposal)

        return resolved_proposals

    def _get_fallback_strategy(self, failed_strategy: str) -> Optional[str]:
        """Get fallback strategy when primary resolution fails."""
        fallback_map = {
            "negotiation_based": "weighted_priority",
            "compromise_generation": "expert_override",
            "expert_override": "performance_based",
            "performance_based": "weighted_priority",
            "weighted_priority": "compromise_generation",
        }

        return fallback_map.get(failed_strategy)

    def _validate_resolution(
        self, resolved_proposals: List[Any], original_conflicts: Dict[str, List[str]]
    ) -> bool:
        """Validate that resolution actually resolves the conflicts."""
        # Check if resolved proposals still have the same conflicts
        remaining_conflicts = self._detect_conflicts_in_proposals(resolved_proposals)

        # Resolution is valid if major conflicts are resolved
        for conflict_type in original_conflicts:
            if conflict_type in remaining_conflicts:
                # Check if conflict severity is reduced
                original_agents = set(original_conflicts[conflict_type])
                remaining_agents = set(remaining_conflicts[conflict_type])

                if len(remaining_agents) >= len(original_agents):
                    return False  # Conflict not resolved

        return True

    def _detect_conflicts_in_proposals(
        self, proposals: List[Any]
    ) -> Dict[str, List[str]]:
        """Detect conflicts in a list of proposals."""
        conflicts = {}

        # Strategy compatibility conflicts
        strategy_conflicts = self._detect_strategy_conflicts_in_list(proposals)
        if strategy_conflicts:
            conflicts["strategy_incompatibility"] = strategy_conflicts

        # Confidence conflicts
        confidence_conflicts = self._detect_confidence_conflicts_in_list(proposals)
        if confidence_conflicts:
            conflicts["low_confidence"] = confidence_conflicts

        return conflicts

    def _detect_strategy_conflicts_in_list(self, proposals: List[Any]) -> List[str]:
        """Detect strategy conflicts in proposal list."""
        conflicts = []

        # Check for incompatible strategy combinations
        incompatible_combinations = {
            ("strict_validation_high_threshold", "creative_exploration"),
            ("lenient_validation_low_threshold", "structured_reasoning"),
        }

        for i, prop1 in enumerate(proposals):
            for prop2 in proposals[i + 1 :]:
                strategy_pair = (prop1.strategy_name, prop2.strategy_name)
                reverse_pair = (prop2.strategy_name, prop1.strategy_name)

                if (
                    strategy_pair in incompatible_combinations
                    or reverse_pair in incompatible_combinations
                ):
                    conflicts.extend([prop1.agent_id, prop2.agent_id])

        return list(set(conflicts))

    def _detect_confidence_conflicts_in_list(self, proposals: List[Any]) -> List[str]:
        """Detect confidence conflicts in proposal list."""
        conflicts = []
        min_confidence_threshold = 0.3

        for proposal in proposals:
            if proposal.confidence < min_confidence_threshold:
                conflicts.append(proposal.agent_id)

        return conflicts

    def _get_agent_performance_weight(
        self, agent_id: str, agent_context: Dict[str, Any]
    ) -> float:
        """Get performance weight for an agent."""
        agent_performance = agent_context.get("agent_performance", {})
        if agent_id in agent_performance:
            perf_data = agent_performance[agent_id]
            success_rate = perf_data.get("success_rate", 0.5)
            avg_confidence = perf_data.get("average_confidence", 0.5)
            return (success_rate + avg_confidence) / 2

        return 0.5  # Default neutral weight

    def _proposals_conflict(self, prop1: Any, prop2: Any) -> bool:
        """Check if two proposals conflict with each other."""
        # Strategy incompatibility
        incompatible_pairs = {
            ("strict_validation_high_threshold", "creative_exploration"),
            ("lenient_validation_low_threshold", "structured_reasoning"),
        }

        strategy_pair = (prop1.strategy_name, prop2.strategy_name)
        reverse_pair = (prop2.strategy_name, prop1.strategy_name)

        return strategy_pair in incompatible_pairs or reverse_pair in incompatible_pairs

    def _identify_conflicting_pairs(
        self, proposals: List[Any]
    ) -> List[Tuple[Any, Any]]:
        """Identify pairs of proposals that conflict."""
        conflicting_pairs = []

        for i, prop1 in enumerate(proposals):
            for prop2 in proposals[i + 1 :]:
                if self._proposals_conflict(prop1, prop2):
                    conflicting_pairs.append((prop1, prop2))

        return conflicting_pairs

    def _negotiate_proposal_pair(
        self, prop1: Any, prop2: Any, state: Dict[str, Any]
    ) -> List[Any]:
        """Negotiate between two conflicting proposals."""
        # Create compromise proposals
        compromise1 = self._create_adjusted_proposal(prop1, confidence_factor=0.9)
        compromise2 = self._create_adjusted_proposal(prop2, confidence_factor=0.9)

        # Adjust strategies to be more compatible
        if prop1.agent_id == "generator" and prop2.agent_id == "validator":
            compromise1 = self._adjust_generator_for_validator(compromise1, prop2)
            compromise2 = self._adjust_validator_for_generator(compromise2, prop1)

        return [compromise1, compromise2]

    def _create_generator_validator_compromise(
        self, gen_proposal: Any, val_proposal: Any
    ) -> Any:
        """Create compromise between generator and validator proposals."""
        # Blend strategy parameters
        compromise_params = {}

        # Take conservative approach - use stricter validation with structured generation
        if "creative" in gen_proposal.strategy_name:
            compromise_strategy = "structured_reasoning"
        else:
            compromise_strategy = gen_proposal.strategy_name

        # Adjust validation threshold to be moderate
        if "strict" in val_proposal.strategy_name:
            validation_threshold = 0.75  # Moderate threshold
        elif "lenient" in val_proposal.strategy_name:
            validation_threshold = 0.65  # Slightly higher than lenient
        else:
            validation_threshold = 0.7  # Standard threshold

        compromise_params.update(
            {
                "generation_strategy": compromise_strategy,
                "validation_threshold": validation_threshold,
                "quality_focus": 0.8,  # High quality focus
            }
        )

        # Create compromise proposal
        compromise_proposal = type(gen_proposal)(
            agent_id="generator_validator_compromise",
            strategy_name=f"{compromise_strategy}_with_validation",
            strategy_parameters=compromise_params,
            confidence=(gen_proposal.confidence + val_proposal.confidence) / 2,
            metadata={
                "compromise": True,
                "original_agents": ["generator", "validator"],
            },
        )

        return compromise_proposal

    def _adjust_curriculum_for_compromise(
        self, curriculum_proposal: Any, existing_compromises: List[Any]
    ) -> Any:
        """Adjust curriculum proposal to be compatible with existing compromises."""
        adjusted_params = curriculum_proposal.strategy_parameters.copy()

        # Make curriculum more flexible to accommodate compromises
        adjusted_params["adaptivity"] = min(
            adjusted_params.get("adaptivity", 0.7) + 0.1, 1.0
        )
        adjusted_params["flexibility"] = 0.8

        adjusted_proposal = type(curriculum_proposal)(
            agent_id=curriculum_proposal.agent_id,
            strategy_name=curriculum_proposal.strategy_name,
            strategy_parameters=adjusted_params,
            confidence=curriculum_proposal.confidence
            * 0.95,  # Slight confidence reduction
            metadata={**curriculum_proposal.metadata, "adjusted_for_compromise": True},
        )

        return adjusted_proposal

    def _calculate_agent_performance_score(
        self, agent_id: str, agent_context: Dict[str, Any]
    ) -> float:
        """Calculate performance score for an agent."""
        agent_performance = agent_context.get("agent_performance", {})
        if agent_id not in agent_performance:
            return 0.5  # Default neutral score

        perf_data = agent_performance[agent_id]
        success_rate = perf_data.get("success_rate", 0.5)
        avg_confidence = perf_data.get("average_confidence", 0.5)
        consistency = perf_data.get("consistency", 0.5)

        # Weighted performance score
        performance_score = (
            0.5 * success_rate + 0.3 * avg_confidence + 0.2 * consistency
        )

        return performance_score

    def _create_adjusted_proposal(
        self, original_proposal: Any, confidence_factor: float = 1.0
    ) -> Any:
        """Create adjusted version of a proposal."""
        adjusted_proposal = type(original_proposal)(
            agent_id=original_proposal.agent_id,
            strategy_name=original_proposal.strategy_name,
            strategy_parameters=original_proposal.strategy_parameters.copy(),
            confidence=original_proposal.confidence * confidence_factor,
            performance_history=original_proposal.performance_history,
            metadata={**original_proposal.metadata, "adjusted": True},
        )

        return adjusted_proposal

    def _adjust_generator_for_validator(
        self, gen_proposal: Any, val_proposal: Any
    ) -> Any:
        """Adjust generator proposal to be more compatible with validator."""
        adjusted_params = gen_proposal.strategy_parameters.copy()

        # Make generation more structured if validator is strict
        if "strict" in val_proposal.strategy_name:
            adjusted_params["structure_emphasis"] = 0.8
            adjusted_params["quality_focus"] = 0.9

        return self._create_adjusted_proposal(gen_proposal)

    def _adjust_validator_for_generator(
        self, val_proposal: Any, gen_proposal: Any
    ) -> Any:
        """Adjust validator proposal to be more compatible with generator."""
        adjusted_params = val_proposal.strategy_parameters.copy()

        # Make validation more flexible if generator is creative
        if "creative" in gen_proposal.strategy_name:
            adjusted_params["flexibility"] = 0.7
            adjusted_params["creativity_tolerance"] = 0.8

        return self._create_adjusted_proposal(val_proposal)

    def _get_all_involved_agents(self, conflicts: Dict[str, List[str]]) -> List[str]:
        """Get all agents involved in conflicts."""
        involved_agents = set()
        for agent_list in conflicts.values():
            involved_agents.update(agent_list)
        return list(involved_agents)

    def _record_resolution_success(self, strategy: str, resolution_time: float) -> None:
        """Record successful conflict resolution."""
        self.resolution_metrics["resolved_conflicts"] += 1
        self.resolution_metrics["resolution_strategy_usage"][strategy] += 1

        # Update average resolution time
        total_conflicts = self.resolution_metrics["total_conflicts"]
        current_avg = self.resolution_metrics["average_resolution_time"]
        self.resolution_metrics["average_resolution_time"] = (
            current_avg * (total_conflicts - 1) + resolution_time
        ) / total_conflicts

        # Update success rate
        self.resolution_metrics["resolution_success_rate"] = (
            self.resolution_metrics["resolved_conflicts"]
            / self.resolution_metrics["total_conflicts"]
        )

        # Record in history
        resolution_record = {
            "timestamp": time.time(),
            "strategy": strategy,
            "resolution_time": resolution_time,
            "success": True,
        }

        self.resolution_history.append(resolution_record)

        # Keep history manageable
        if len(self.resolution_history) > 1000:
            self.resolution_history = self.resolution_history[-1000:]

    def _record_resolution_failure(self, strategy: str, resolution_time: float) -> None:
        """Record failed conflict resolution."""
        if strategy in self.resolution_metrics["resolution_strategy_usage"]:
            self.resolution_metrics["resolution_strategy_usage"][strategy] += 1

        # Update success rate
        self.resolution_metrics["resolution_success_rate"] = (
            self.resolution_metrics["resolved_conflicts"]
            / self.resolution_metrics["total_conflicts"]
        )

        # Record in history
        resolution_record = {
            "timestamp": time.time(),
            "strategy": strategy,
            "resolution_time": resolution_time,
            "success": False,
        }

        self.resolution_history.append(resolution_record)

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive conflict resolution performance summary."""
        return {
            "resolution_metrics": self.resolution_metrics.copy(),
            "agent_priorities": self.agent_priorities.copy(),
            "recent_resolution_history": self.resolution_history[-20:],
            "most_used_strategy": max(
                self.resolution_metrics["resolution_strategy_usage"].items(),
                key=lambda x: x[1],
                default=("none", 0),
            )[0],
            "performance_status": "excellent"
            if self.resolution_metrics["resolution_success_rate"] > 0.9
            else "good"
            if self.resolution_metrics["resolution_success_rate"] > 0.7
            else "needs_improvement",
        }
