"""
Coordination Policy Framework

This module implements the coordination policy framework for multi-agent RL coordination,
including coordinated action generation, conflict detection, and policy execution.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..config import CoordinationConfig
from ..exceptions import CoordinationFailureError
from ..logging_config import get_marl_logger
from .conflict_resolver import ConflictResolver
from .consensus_mechanism import ConsensusMechanism

logger = logging.getLogger(__name__)


@dataclass
class AgentProposal:
    """Represents a proposal from an individual agent."""

    agent_id: str
    strategy_name: str
    strategy_parameters: Dict[str, Any]
    confidence: float
    performance_history: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class CoordinatedAction:
    """Represents a coordinated action from multiple agents."""

    generator_strategy: Dict[str, Any]
    validation_criteria: Dict[str, Any]
    curriculum_guidance: Dict[str, Any]
    coordination_confidence: float
    coordination_metadata: Dict[str, Any] = field(default_factory=dict)
    consensus_quality: float = 0.0
    conflict_resolution_applied: bool = False
    timestamp: float = field(default_factory=time.time)


class CoordinationPolicy:
    """
    Coordination policy framework for multi-agent RL coordination.

    This class manages the coordination of actions from multiple RL agents,
    including conflict detection, resolution, and consensus building.
    """

    def __init__(self, config: CoordinationConfig):
        """
        Initialize coordination policy.

        Args:
            config: Coordination configuration parameters
        """
        self.config = config
        self.logger = get_marl_logger("coordination_policy")

        # Coordination components
        self.consensus_mechanism = ConsensusMechanism(config)
        self.conflict_resolver = ConflictResolver(config)

        # Coordination history and metrics
        self.coordination_history = []
        self.coordination_metrics = {
            "total_coordinations": 0,
            "successful_coordinations": 0,
            "conflicts_detected": 0,
            "conflicts_resolved": 0,
            "average_coordination_time": 0.0,
            "average_consensus_quality": 0.0,
        }

        # Agent performance tracking
        self.agent_performance = {
            "generator": {"proposals": 0, "accepted": 0, "confidence_sum": 0.0},
            "validator": {"proposals": 0, "accepted": 0, "confidence_sum": 0.0},
            "curriculum": {"proposals": 0, "accepted": 0, "confidence_sum": 0.0},
        }

        self.logger.log_agent_action(
            "coordination_policy",
            "initialized",
            1.0,
            "Config: %s" % type(config).__name__,
        )

    async def coordinate(
        self,
        agent_actions: Dict[str, Any],
        state: Dict[str, Any],
        agent_context: Dict[str, Any],
    ) -> CoordinatedAction:
        """
        Coordinate agent actions using consensus mechanisms.

        Args:
            agent_actions: Actions proposed by each agent
            state: Current environment state
            agent_context: Additional context about agents

        Returns:
            CoordinatedAction representing the coordinated decision

        Raises:
            CoordinationFailureError: If coordination fails
        """
        coordination_start_time = time.time()

        try:
            # Update coordination metrics
            self.coordination_metrics["total_coordinations"] += 1

            # Convert agent actions to proposals
            proposals = self._convert_to_proposals(agent_actions, agent_context)

            # Update agent performance tracking
            self._update_agent_performance(proposals)

            # Check for conflicts
            conflicts = self.detect_conflicts(proposals)

            if conflicts:
                self.coordination_metrics["conflicts_detected"] += 1
                self.logger.log_coordination_event(
                    "conflict_detected",
                    {
                        "conflict_count": len(conflicts),
                        "agents_involved": list(conflicts.keys()),
                    },
                )

                # Resolve conflicts through negotiation
                resolved_proposals = await self.conflict_resolver.resolve(
                    conflicts, proposals, state, agent_context
                )

                if resolved_proposals:
                    self.coordination_metrics["conflicts_resolved"] += 1
                    proposals = resolved_proposals
                else:
                    # Conflict resolution failed, use fallback
                    proposals = self._apply_fallback_coordination(proposals, state)

            # Build consensus
            consensus = await self.consensus_mechanism.build_consensus(
                proposals, state, agent_context
            )

            # Validate consensus quality
            if consensus["quality"] < self.config.min_consensus_quality:
                self.logger.log_coordination_event(
                    "low_consensus_quality",
                    {
                        "quality": consensus["quality"],
                        "threshold": self.config.min_consensus_quality,
                    },
                )

                # Apply consensus improvement strategies
                consensus = await self._improve_consensus(consensus, proposals, state)

            # Create coordinated action
            coordinated_action = self._create_coordinated_action(
                consensus, proposals, conflicts is not None
            )

            # Record coordination success
            coordination_time = time.time() - coordination_start_time
            self._record_coordination_success(coordinated_action, coordination_time)

            self.logger.log_coordination_event(
                "coordination_success",
                {
                    "coordination_time": coordination_time,
                    "consensus_quality": coordinated_action.consensus_quality,
                    "conflicts_resolved": coordinated_action.conflict_resolution_applied,
                },
            )

            return coordinated_action

        except Exception as e:
            coordination_time = time.time() - coordination_start_time
            self._record_coordination_failure(e, coordination_time)

            self.logger.log_error_with_context(
                e,
                {
                    "coordination_time": coordination_time,
                    "agent_actions_keys": list(agent_actions.keys()),
                    "state_keys": list(state.keys()),
                },
            )

            raise CoordinationFailureError(
                "Coordination failed during policy execution",
                coordination_type="policy_coordination",
                agents_involved=list(agent_actions.keys()),
                failure_context={
                    "coordination_time": coordination_time,
                    "error_type": type(e).__name__,
                },
            ) from e

    def detect_conflicts(
        self, proposals: List[AgentProposal]
    ) -> Optional[Dict[str, List[str]]]:
        """
        Detect conflicts between agent proposals.

        Args:
            proposals: List of agent proposals

        Returns:
            Dictionary mapping conflict types to involved agents, or None if no conflicts
        """
        conflicts = {}

        # Strategy compatibility conflicts
        strategy_conflicts = self._detect_strategy_conflicts(proposals)
        if strategy_conflicts:
            conflicts["strategy_incompatibility"] = strategy_conflicts

        # Confidence threshold conflicts
        confidence_conflicts = self._detect_confidence_conflicts(proposals)
        if confidence_conflicts:
            conflicts["low_confidence"] = confidence_conflicts

        # Parameter conflicts
        parameter_conflicts = self._detect_parameter_conflicts(proposals)
        if parameter_conflicts:
            conflicts["parameter_mismatch"] = parameter_conflicts

        # Quality requirement conflicts
        quality_conflicts = self._detect_quality_conflicts(proposals)
        if quality_conflicts:
            conflicts["quality_requirements"] = quality_conflicts

        return conflicts if conflicts else None

    def _convert_to_proposals(
        self, agent_actions: Dict[str, Any], agent_context: Dict[str, Any]
    ) -> List[AgentProposal]:
        """Convert agent actions to standardized proposals."""
        proposals = []

        for agent_id, action in agent_actions.items():
            if isinstance(action, dict):
                proposal = AgentProposal(
                    agent_id=agent_id,
                    strategy_name=action.get("strategy", "unknown"),
                    strategy_parameters=action.get("parameters", {}),
                    confidence=action.get("confidence", 0.5),
                    performance_history=action.get("performance_history", {}),
                    metadata=action.get("metadata", {}),
                )
                proposals.append(proposal)

        return proposals

    def _detect_strategy_conflicts(self, proposals: List[AgentProposal]) -> List[str]:
        """Detect strategy compatibility conflicts."""
        conflicts = []

        # Define incompatible strategy combinations
        incompatible_combinations = {
            ("strict_validation_high_threshold", "creative_exploration"),
            ("lenient_validation_low_threshold", "structured_reasoning"),
            ("linear_progression", "adaptive_difficulty_adjustment"),
        }

        # Check for incompatible combinations
        strategy_pairs = []
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

    def _detect_confidence_conflicts(self, proposals: List[AgentProposal]) -> List[str]:
        """Detect confidence threshold conflicts."""
        conflicts = []
        min_confidence_threshold = 0.3

        for proposal in proposals:
            if proposal.confidence < min_confidence_threshold:
                conflicts.append(proposal.agent_id)

        return conflicts

    def _detect_parameter_conflicts(self, proposals: List[AgentProposal]) -> List[str]:
        """Detect parameter mismatch conflicts."""
        conflicts = []

        # Check for conflicting difficulty levels
        difficulty_levels = []
        for proposal in proposals:
            if "difficulty" in proposal.strategy_parameters:
                difficulty_levels.append(
                    (proposal.agent_id, proposal.strategy_parameters["difficulty"])
                )

        if len(set(level for _, level in difficulty_levels)) > 1:
            conflicts.extend([agent_id for agent_id, _ in difficulty_levels])

        return list(set(conflicts))

    def _detect_quality_conflicts(self, proposals: List[AgentProposal]) -> List[str]:
        """Detect quality requirement conflicts."""
        conflicts = []

        # Check for conflicting quality thresholds
        quality_thresholds = []
        for proposal in proposals:
            if "quality_threshold" in proposal.strategy_parameters:
                quality_thresholds.append(
                    (
                        proposal.agent_id,
                        proposal.strategy_parameters["quality_threshold"],
                    )
                )

        if quality_thresholds:
            threshold_values = [threshold for _, threshold in quality_thresholds]
            if (
                max(threshold_values) - min(threshold_values) > 0.3
            ):  # Significant difference
                conflicts.extend([agent_id for agent_id, _ in quality_thresholds])

        return list(set(conflicts))

    def _apply_fallback_coordination(
        self, proposals: List[AgentProposal], state: Dict[str, Any]
    ) -> List[AgentProposal]:
        """Apply fallback coordination when conflict resolution fails."""
        # Use the proposal with highest confidence as primary
        primary_proposal = max(proposals, key=lambda p: p.confidence)

        # Adjust other proposals to be compatible
        fallback_proposals = [primary_proposal]

        for proposal in proposals:
            if proposal.agent_id != primary_proposal.agent_id:
                # Create compatible version of the proposal
                compatible_proposal = AgentProposal(
                    agent_id=proposal.agent_id,
                    strategy_name=self._get_compatible_strategy(
                        proposal.strategy_name, primary_proposal.strategy_name
                    ),
                    strategy_parameters=proposal.strategy_parameters.copy(),
                    confidence=proposal.confidence
                    * 0.8,  # Reduce confidence due to fallback
                    performance_history=proposal.performance_history,
                    metadata={**proposal.metadata, "fallback_applied": True},
                )
                fallback_proposals.append(compatible_proposal)

        return fallback_proposals

    def _get_compatible_strategy(self, strategy: str, primary_strategy: str) -> str:
        """Get a compatible strategy for the given primary strategy."""
        compatibility_map = {
            "strict_validation_high_threshold": {
                "creative_exploration": "structured_reasoning",
                "adaptive_difficulty_adjustment": "linear_progression",
            },
            "lenient_validation_low_threshold": {
                "structured_reasoning": "concept_based_generation",
            },
            "linear_progression": {
                "adaptive_difficulty_adjustment": "mastery_based_progression",
            },
        }

        if primary_strategy in compatibility_map:
            return compatibility_map[primary_strategy].get(strategy, strategy)

        return strategy

    async def _improve_consensus(
        self,
        consensus: Dict[str, Any],
        proposals: List[AgentProposal],
        state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Improve consensus quality through additional negotiation."""
        # Re-weight proposals based on performance history
        weighted_proposals = []
        for proposal in proposals:
            weight = self._calculate_proposal_weight(proposal)
            weighted_proposals.append((proposal, weight))

        # Rebuild consensus with weights
        improved_consensus = await self.consensus_mechanism.build_weighted_consensus(
            weighted_proposals, state
        )

        return improved_consensus

    def _calculate_proposal_weight(self, proposal: AgentProposal) -> float:
        """Calculate weight for a proposal based on agent performance."""
        base_weight = proposal.confidence

        # Adjust based on agent performance history
        agent_perf = self.agent_performance.get(proposal.agent_id, {})
        if agent_perf.get("proposals", 0) > 0:
            success_rate = agent_perf.get("accepted", 0) / agent_perf["proposals"]
            avg_confidence = (
                agent_perf.get("confidence_sum", 0) / agent_perf["proposals"]
            )

            performance_multiplier = (success_rate + avg_confidence) / 2
            base_weight *= 1 + performance_multiplier

        return min(base_weight, 2.0)  # Cap at 2x weight

    def _create_coordinated_action(
        self,
        consensus: Dict[str, Any],
        proposals: List[AgentProposal],
        conflicts_resolved: bool,
    ) -> CoordinatedAction:
        """Create coordinated action from consensus and proposals."""
        # Extract strategy information from consensus
        generator_strategy = consensus.get("generator_strategy", {})
        validation_criteria = consensus.get("validation_criteria", {})
        curriculum_guidance = consensus.get("curriculum_guidance", {})

        # Create coordinated action
        coordinated_action = CoordinatedAction(
            generator_strategy=generator_strategy,
            validation_criteria=validation_criteria,
            curriculum_guidance=curriculum_guidance,
            coordination_confidence=consensus.get("confidence", 0.5),
            consensus_quality=consensus.get("quality", 0.5),
            conflict_resolution_applied=conflicts_resolved,
            coordination_metadata={
                "consensus_strategy": consensus.get("strategy", "unknown"),
                "participating_agents": [p.agent_id for p in proposals],
                "proposal_count": len(proposals),
                "coordination_timestamp": time.time(),
            },
        )

        return coordinated_action

    def _update_agent_performance(self, proposals: List[AgentProposal]) -> None:
        """Update agent performance tracking."""
        for proposal in proposals:
            agent_id = proposal.agent_id
            if agent_id in self.agent_performance:
                self.agent_performance[agent_id]["proposals"] += 1
                self.agent_performance[agent_id]["confidence_sum"] += (
                    proposal.confidence
                )

    def _record_coordination_success(
        self, coordinated_action: CoordinatedAction, coordination_time: float
    ) -> None:
        """Record successful coordination."""
        self.coordination_metrics["successful_coordinations"] += 1

        # Update average coordination time
        total_coords = self.coordination_metrics["total_coordinations"]
        current_avg = self.coordination_metrics["average_coordination_time"]
        self.coordination_metrics["average_coordination_time"] = (
            current_avg * (total_coords - 1) + coordination_time
        ) / total_coords

        # Update average consensus quality
        current_quality_avg = self.coordination_metrics["average_consensus_quality"]
        self.coordination_metrics["average_consensus_quality"] = (
            current_quality_avg * (total_coords - 1)
            + coordinated_action.consensus_quality
        ) / total_coords

        # Record in history
        coordination_record = {
            "timestamp": time.time(),
            "coordination_time": coordination_time,
            "consensus_quality": coordinated_action.consensus_quality,
            "conflicts_resolved": coordinated_action.conflict_resolution_applied,
            "participating_agents": coordinated_action.coordination_metadata.get(
                "participating_agents", []
            ),
        }

        self.coordination_history.append(coordination_record)

        # Keep history manageable
        if len(self.coordination_history) > 1000:
            self.coordination_history = self.coordination_history[-1000:]

    def _record_coordination_failure(
        self, error: Exception, coordination_time: float
    ) -> None:
        """Record coordination failure."""
        failure_record = {
            "timestamp": time.time(),
            "coordination_time": coordination_time,
            "error_type": type(error).__name__,
            "error_message": str(error),
        }

        self.coordination_history.append(failure_record)

    def get_coordination_success_rate(self) -> float:
        """Get current coordination success rate."""
        if self.coordination_metrics["total_coordinations"] == 0:
            return 0.0

        return (
            self.coordination_metrics["successful_coordinations"]
            / self.coordination_metrics["total_coordinations"]
        )

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive coordination performance summary."""
        success_rate = self.get_coordination_success_rate()

        return {
            "coordination_metrics": self.coordination_metrics.copy(),
            "coordination_success_rate": success_rate,
            "agent_performance": {
                agent_id: {
                    **perf,
                    "acceptance_rate": perf["accepted"] / max(perf["proposals"], 1),
                    "average_confidence": perf["confidence_sum"]
                    / max(perf["proposals"], 1),
                }
                for agent_id, perf in self.agent_performance.items()
            },
            "recent_coordination_history": self.coordination_history[-20:],
            "performance_status": "excellent"
            if success_rate > 0.9
            else "good"
            if success_rate > 0.7
            else "needs_improvement",
        }
