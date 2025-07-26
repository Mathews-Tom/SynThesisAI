"""
Consensus Mechanism

This module implements consensus building mechanisms for multi-agent RL coordination,
including various voting strategies, weighted consensus, and quality validation.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..config import CoordinationConfig
from ..exceptions import ConsensusFailureError
from ..logging_config import get_marl_logger

logger = logging.getLogger(__name__)


class ConsensusMechanism:
    """
    Consensus building system for multi-agent coordination.

    This class implements various consensus strategies including weighted voting,
    majority consensus, expert priority, and adaptive consensus selection.
    """

    def __init__(self, config: CoordinationConfig):
        """
        Initialize consensus mechanism.

        Args:
            config: Coordination configuration parameters
        """
        self.config = config
        self.logger = get_marl_logger("consensus_mechanism")

        # Consensus strategies
        self.consensus_strategies = {
            "weighted_average": self._weighted_average_consensus,
            "majority_vote": self._majority_vote_consensus,
            "expert_priority": self._expert_priority_consensus,
            "adaptive_consensus": self._adaptive_consensus,
            "confidence_weighted": self._confidence_weighted_consensus,
            "performance_based": self._performance_based_consensus,
        }

        # Consensus history and metrics
        self.consensus_history = []
        self.consensus_metrics = {
            "total_consensus_attempts": 0,
            "successful_consensus": 0,
            "strategy_usage": {strategy: 0 for strategy in self.consensus_strategies},
            "average_consensus_quality": 0.0,
            "average_consensus_time": 0.0,
        }

        # Strategy effectiveness tracking
        self.strategy_effectiveness = {
            strategy: {"successes": 0, "attempts": 0, "avg_quality": 0.0}
            for strategy in self.consensus_strategies
        }

        self.logger.log_agent_action(
            "consensus_mechanism",
            "initialized",
            1.0,
            "Strategies: %d" % len(self.consensus_strategies),
        )

    async def build_consensus(
        self,
        proposals: List[Any],
        state: Dict[str, Any],
        agent_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Build consensus from agent proposals.

        Args:
            proposals: List of agent proposals
            state: Current environment state
            agent_context: Additional context about agents

        Returns:
            Dictionary containing consensus information

        Raises:
            ConsensusFailureError: If consensus building fails
        """
        consensus_start_time = time.time()

        try:
            self.consensus_metrics["total_consensus_attempts"] += 1

            # Select consensus strategy based on context
            strategy = self._select_consensus_strategy(proposals, state, agent_context)

            self.logger.log_coordination_event(
                "consensus_building_started",
                {
                    "strategy": strategy,
                    "proposal_count": len(proposals),
                    "agents": [p.agent_id for p in proposals],
                },
            )

            # Build consensus using selected strategy
            consensus_func = self.consensus_strategies[strategy]
            consensus = await consensus_func(proposals, state, agent_context)

            # Validate consensus quality
            consensus_quality = self._validate_consensus_quality(consensus, proposals)
            consensus["quality"] = consensus_quality
            consensus["strategy"] = strategy
            consensus["timestamp"] = time.time()

            # Check if consensus meets minimum quality threshold
            if consensus_quality >= self.config.min_consensus_quality:
                consensus_time = time.time() - consensus_start_time
                self._record_consensus_success(
                    strategy, consensus_quality, consensus_time
                )

                self.logger.log_coordination_event(
                    "consensus_building_success",
                    {
                        "strategy": strategy,
                        "quality": consensus_quality,
                        "consensus_time": consensus_time,
                    },
                )

                return consensus
            else:
                # Try to improve consensus
                improved_consensus = await self._improve_consensus(
                    consensus, proposals, state, agent_context
                )

                if improved_consensus["quality"] >= self.config.min_consensus_quality:
                    consensus_time = time.time() - consensus_start_time
                    self._record_consensus_success(
                        strategy, improved_consensus["quality"], consensus_time
                    )
                    return improved_consensus
                else:
                    # Consensus quality still too low
                    consensus_time = time.time() - consensus_start_time
                    self._record_consensus_failure(
                        strategy, consensus_quality, consensus_time
                    )

                    # Return best effort consensus with warning
                    improved_consensus["quality_warning"] = True
                    return improved_consensus

        except Exception as e:
            consensus_time = time.time() - consensus_start_time
            self._record_consensus_failure("error", 0.0, consensus_time)

            self.logger.log_error_with_context(
                e,
                {
                    "proposals_count": len(proposals),
                    "consensus_time": consensus_time,
                },
            )

            raise ConsensusFailureError(
                "Critical failure during consensus building",
                consensus_strategy=strategy if "strategy" in locals() else "unknown",
                proposals_count=len(proposals),
                failure_context={
                    "consensus_time": consensus_time,
                    "error_type": type(e).__name__,
                },
            ) from e

    async def build_weighted_consensus(
        self,
        weighted_proposals: List[Tuple[Any, float]],
        state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Build consensus from weighted proposals.

        Args:
            weighted_proposals: List of (proposal, weight) tuples
            state: Current environment state

        Returns:
            Dictionary containing weighted consensus information
        """
        try:
            # Extract proposals and weights
            proposals = [prop for prop, weight in weighted_proposals]
            weights = np.array([weight for prop, weight in weighted_proposals])

            # Normalize weights
            weights = weights / np.sum(weights)

            # Build weighted consensus
            consensus = {
                "generator_strategy": self._weighted_strategy_selection(
                    proposals, weights, "generator"
                ),
                "validation_criteria": self._weighted_strategy_selection(
                    proposals, weights, "validator"
                ),
                "curriculum_guidance": self._weighted_strategy_selection(
                    proposals, weights, "curriculum"
                ),
                "confidence": np.average(
                    [p.confidence for p in proposals], weights=weights
                ),
                "weights_used": weights.tolist(),
                "weighted_consensus": True,
            }

            return consensus

        except Exception as e:
            self.logger.log_error_with_context(
                e,
                {
                    "weighted_proposals_count": len(weighted_proposals),
                },
            )
            raise ConsensusFailureError(
                "Failed to build weighted consensus",
                consensus_strategy="weighted_consensus",
                proposals_count=len(weighted_proposals),
            ) from e

    def _select_consensus_strategy(
        self, proposals: List[Any], state: Dict[str, Any], agent_context: Dict[str, Any]
    ) -> str:
        """Select appropriate consensus strategy based on context."""
        # Strategy selection logic
        proposal_count = len(proposals)
        confidence_variance = np.var([p.confidence for p in proposals])

        # Use adaptive consensus for complex scenarios
        if proposal_count > 2 and confidence_variance > 0.1:
            return "adaptive_consensus"

        # Use expert priority for domain-specific content
        domain = state.get("domain", "")
        if domain in ["mathematics", "science"]:
            return "expert_priority"

        # Use confidence weighted for high-confidence scenarios
        avg_confidence = np.mean([p.confidence for p in proposals])
        if avg_confidence > 0.8:
            return "confidence_weighted"

        # Use performance based if we have performance data
        if agent_context.get("agent_performance"):
            return "performance_based"

        # Default to weighted average
        return "weighted_average"

    async def _weighted_average_consensus(
        self, proposals: List[Any], state: Dict[str, Any], agent_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build consensus using weighted average of proposals."""
        if not proposals:
            return {"confidence": 0.0, "error": "No proposals provided"}

        # Calculate weights based on confidence and agent type
        weights = []
        for proposal in proposals:
            base_weight = proposal.confidence

            # Adjust weight based on agent type
            if proposal.agent_id == "validator":
                base_weight *= 1.2  # Validators get slightly higher weight
            elif proposal.agent_id == "curriculum":
                base_weight *= 1.1  # Curriculum gets slightly higher weight

            weights.append(base_weight)

        weights = np.array(weights)
        weights = weights / np.sum(weights)  # Normalize

        # Build consensus
        consensus = {
            "generator_strategy": self._select_strategy_by_weight(
                proposals, weights, "generator"
            ),
            "validation_criteria": self._select_strategy_by_weight(
                proposals, weights, "validator"
            ),
            "curriculum_guidance": self._select_strategy_by_weight(
                proposals, weights, "curriculum"
            ),
            "confidence": np.average(
                [p.confidence for p in proposals], weights=weights
            ),
            "participating_agents": [p.agent_id for p in proposals],
        }

        return consensus

    async def _majority_vote_consensus(
        self, proposals: List[Any], state: Dict[str, Any], agent_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build consensus using majority voting."""
        if not proposals:
            return {"confidence": 0.0, "error": "No proposals provided"}

        # Group proposals by strategy
        strategy_votes = {}
        for proposal in proposals:
            strategy_key = proposal.strategy_name
            if strategy_key not in strategy_votes:
                strategy_votes[strategy_key] = []
            strategy_votes[strategy_key].append(proposal)

        # Find majority strategy
        majority_strategy = max(strategy_votes.items(), key=lambda x: len(x[1]))
        majority_proposals = majority_strategy[1]

        # Build consensus from majority
        consensus = {
            "generator_strategy": self._extract_strategy_info(
                majority_proposals, "generator"
            ),
            "validation_criteria": self._extract_strategy_info(
                majority_proposals, "validator"
            ),
            "curriculum_guidance": self._extract_strategy_info(
                majority_proposals, "curriculum"
            ),
            "confidence": np.mean([p.confidence for p in majority_proposals]),
            "majority_strategy": majority_strategy[0],
            "vote_count": len(majority_proposals),
            "total_votes": len(proposals),
        }

        return consensus

    async def _expert_priority_consensus(
        self, proposals: List[Any], state: Dict[str, Any], agent_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build consensus using expert priority system."""
        if not proposals:
            return {"confidence": 0.0, "error": "No proposals provided"}

        # Define expert priority based on domain
        domain = state.get("domain", "general")

        expert_priorities = {
            "mathematics": {"curriculum": 3, "validator": 2, "generator": 1},
            "science": {"curriculum": 3, "validator": 2, "generator": 1},
            "general": {"validator": 3, "curriculum": 2, "generator": 1},
        }

        priorities = expert_priorities.get(domain, expert_priorities["general"])

        # Calculate expert-weighted consensus
        expert_weights = []
        for proposal in proposals:
            expert_weight = priorities.get(proposal.agent_id, 1) * proposal.confidence
            expert_weights.append(expert_weight)

        expert_weights = np.array(expert_weights)
        expert_weights = expert_weights / np.sum(expert_weights)

        # Build expert consensus
        consensus = {
            "generator_strategy": self._select_strategy_by_weight(
                proposals, expert_weights, "generator"
            ),
            "validation_criteria": self._select_strategy_by_weight(
                proposals, expert_weights, "validator"
            ),
            "curriculum_guidance": self._select_strategy_by_weight(
                proposals, expert_weights, "curriculum"
            ),
            "confidence": np.average(
                [p.confidence for p in proposals], weights=expert_weights
            ),
            "expert_domain": domain,
            "expert_weights": expert_weights.tolist(),
        }

        return consensus

    async def _adaptive_consensus(
        self, proposals: List[Any], state: Dict[str, Any], agent_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build consensus using adaptive strategy selection."""
        if not proposals:
            return {"confidence": 0.0, "error": "No proposals provided"}

        # Analyze proposal characteristics
        confidence_variance = np.var([p.confidence for p in proposals])
        avg_confidence = np.mean([p.confidence for p in proposals])

        # Select sub-strategy based on characteristics
        if confidence_variance < 0.05:  # Low variance - use simple average
            return await self._weighted_average_consensus(
                proposals, state, agent_context
            )
        elif avg_confidence > 0.8:  # High confidence - use confidence weighting
            return await self._confidence_weighted_consensus(
                proposals, state, agent_context
            )
        else:  # Mixed scenario - use expert priority
            return await self._expert_priority_consensus(
                proposals, state, agent_context
            )

    async def _confidence_weighted_consensus(
        self, proposals: List[Any], state: Dict[str, Any], agent_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build consensus weighted by proposal confidence."""
        if not proposals:
            return {"confidence": 0.0, "error": "No proposals provided"}

        # Use confidence as weights
        confidences = np.array([p.confidence for p in proposals])

        # Apply exponential weighting to emphasize high confidence
        exp_weights = np.exp(confidences * 2)  # Amplify confidence differences
        exp_weights = exp_weights / np.sum(exp_weights)

        # Build confidence-weighted consensus
        consensus = {
            "generator_strategy": self._select_strategy_by_weight(
                proposals, exp_weights, "generator"
            ),
            "validation_criteria": self._select_strategy_by_weight(
                proposals, exp_weights, "validator"
            ),
            "curriculum_guidance": self._select_strategy_by_weight(
                proposals, exp_weights, "curriculum"
            ),
            "confidence": np.average(confidences, weights=exp_weights),
            "confidence_weights": exp_weights.tolist(),
            "confidence_emphasis": True,
        }

        return consensus

    async def _performance_based_consensus(
        self, proposals: List[Any], state: Dict[str, Any], agent_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build consensus based on agent performance history."""
        if not proposals:
            return {"confidence": 0.0, "error": "No proposals provided"}

        # Get performance weights
        performance_weights = []
        agent_performance = agent_context.get("agent_performance", {})

        for proposal in proposals:
            if proposal.agent_id in agent_performance:
                perf_data = agent_performance[proposal.agent_id]
                success_rate = perf_data.get("success_rate", 0.5)
                avg_confidence = perf_data.get("average_confidence", 0.5)
                performance_weight = (success_rate + avg_confidence) / 2
            else:
                performance_weight = 0.5  # Default neutral weight

            # Combine with proposal confidence
            combined_weight = performance_weight * proposal.confidence
            performance_weights.append(combined_weight)

        performance_weights = np.array(performance_weights)
        performance_weights = performance_weights / np.sum(performance_weights)

        # Build performance-based consensus
        consensus = {
            "generator_strategy": self._select_strategy_by_weight(
                proposals, performance_weights, "generator"
            ),
            "validation_criteria": self._select_strategy_by_weight(
                proposals, performance_weights, "validator"
            ),
            "curriculum_guidance": self._select_strategy_by_weight(
                proposals, performance_weights, "curriculum"
            ),
            "confidence": np.average(
                [p.confidence for p in proposals], weights=performance_weights
            ),
            "performance_weights": performance_weights.tolist(),
            "performance_based": True,
        }

        return consensus

    def _select_strategy_by_weight(
        self, proposals: List[Any], weights: np.ndarray, agent_type: str
    ) -> Dict[str, Any]:
        """Select strategy for specific agent type using weights."""
        # Find proposals from the specified agent type
        agent_proposals = [p for p in proposals if p.agent_id == agent_type]

        if not agent_proposals:
            return {"strategy": "default", "parameters": {}, "confidence": 0.0}

        if len(agent_proposals) == 1:
            proposal = agent_proposals[0]
            return {
                "strategy": proposal.strategy_name,
                "parameters": proposal.strategy_parameters,
                "confidence": proposal.confidence,
            }

        # Multiple proposals from same agent type - use highest weighted
        agent_indices = [i for i, p in enumerate(proposals) if p.agent_id == agent_type]
        agent_weights = weights[agent_indices]

        best_index = agent_indices[np.argmax(agent_weights)]
        best_proposal = proposals[best_index]

        return {
            "strategy": best_proposal.strategy_name,
            "parameters": best_proposal.strategy_parameters,
            "confidence": best_proposal.confidence,
            "weight": weights[best_index],
        }

    def _weighted_strategy_selection(
        self, proposals: List[Any], weights: np.ndarray, agent_type: str
    ) -> Dict[str, Any]:
        """Select strategy using weighted selection for specific agent type."""
        return self._select_strategy_by_weight(proposals, weights, agent_type)

    def _extract_strategy_info(
        self, proposals: List[Any], agent_type: str
    ) -> Dict[str, Any]:
        """Extract strategy information for specific agent type."""
        agent_proposals = [p for p in proposals if p.agent_id == agent_type]

        if not agent_proposals:
            return {"strategy": "default", "parameters": {}, "confidence": 0.0}

        # Use the proposal with highest confidence
        best_proposal = max(agent_proposals, key=lambda p: p.confidence)

        return {
            "strategy": best_proposal.strategy_name,
            "parameters": best_proposal.strategy_parameters,
            "confidence": best_proposal.confidence,
        }

    def _validate_consensus_quality(
        self, consensus: Dict[str, Any], proposals: List[Any]
    ) -> float:
        """Validate the quality of the consensus."""
        quality_factors = []

        # Confidence factor
        consensus_confidence = consensus.get("confidence", 0.0)
        quality_factors.append(consensus_confidence)

        # Participation factor (how many agents contributed)
        participating_agents = consensus.get("participating_agents", [])
        participation_rate = len(participating_agents) / max(len(proposals), 1)
        quality_factors.append(participation_rate)

        # Strategy coherence factor
        strategies = [
            consensus.get("generator_strategy", {}).get("strategy", ""),
            consensus.get("validation_criteria", {}).get("strategy", ""),
            consensus.get("curriculum_guidance", {}).get("strategy", ""),
        ]

        # Check for strategy compatibility
        coherence_score = self._calculate_strategy_coherence(strategies)
        quality_factors.append(coherence_score)

        # Weight factors and calculate overall quality
        weights = [0.4, 0.3, 0.3]  # Confidence, participation, coherence
        overall_quality = np.average(quality_factors, weights=weights)

        return float(overall_quality)

    def _calculate_strategy_coherence(self, strategies: List[str]) -> float:
        """Calculate coherence score for strategy combination."""
        # Define coherent strategy combinations
        coherent_combinations = {
            (
                "step_by_step_approach",
                "standard_validation_medium_threshold",
                "linear_progression",
            ),
            (
                "creative_exploration",
                "lenient_validation_low_threshold",
                "multi_modal_learning",
            ),
            (
                "structured_reasoning",
                "strict_validation_high_threshold",
                "mastery_based_progression",
            ),
        }

        # Check if current combination is coherent
        strategy_tuple = tuple(strategies)

        # Exact match
        if strategy_tuple in coherent_combinations:
            return 1.0

        # Partial match (2 out of 3 strategies match known combinations)
        partial_matches = 0
        for coherent_combo in coherent_combinations:
            matches = sum(1 for s in strategies if s in coherent_combo)
            partial_matches = max(partial_matches, matches)

        return partial_matches / 3.0

    async def _improve_consensus(
        self,
        consensus: Dict[str, Any],
        proposals: List[Any],
        state: Dict[str, Any],
        agent_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Improve consensus quality through additional processing."""
        # Try different consensus strategy
        current_strategy = consensus.get("strategy", "unknown")

        # Select alternative strategy
        alternative_strategies = [
            s for s in self.consensus_strategies.keys() if s != current_strategy
        ]

        best_consensus = consensus
        best_quality = consensus.get("quality", 0.0)

        for alt_strategy in alternative_strategies[:2]:  # Try top 2 alternatives
            try:
                alt_consensus_func = self.consensus_strategies[alt_strategy]
                alt_consensus = await alt_consensus_func(
                    proposals, state, agent_context
                )
                alt_quality = self._validate_consensus_quality(alt_consensus, proposals)

                if alt_quality > best_quality:
                    alt_consensus["quality"] = alt_quality
                    alt_consensus["strategy"] = alt_strategy
                    alt_consensus["improved"] = True
                    best_consensus = alt_consensus
                    best_quality = alt_quality

            except Exception as e:
                self.logger.log_error_with_context(
                    e, {"alternative_strategy": alt_strategy}
                )
                continue

        return best_consensus

    def _record_consensus_success(
        self, strategy: str, quality: float, consensus_time: float
    ) -> None:
        """Record successful consensus building."""
        self.consensus_metrics["successful_consensus"] += 1
        self.consensus_metrics["strategy_usage"][strategy] += 1

        # Update strategy effectiveness
        self.strategy_effectiveness[strategy]["successes"] += 1
        self.strategy_effectiveness[strategy]["attempts"] += 1

        current_avg = self.strategy_effectiveness[strategy]["avg_quality"]
        attempts = self.strategy_effectiveness[strategy]["attempts"]
        self.strategy_effectiveness[strategy]["avg_quality"] = (
            current_avg * (attempts - 1) + quality
        ) / attempts

        # Update overall metrics
        total_attempts = self.consensus_metrics["total_consensus_attempts"]
        current_quality_avg = self.consensus_metrics["average_consensus_quality"]
        self.consensus_metrics["average_consensus_quality"] = (
            current_quality_avg * (total_attempts - 1) + quality
        ) / total_attempts

        current_time_avg = self.consensus_metrics["average_consensus_time"]
        self.consensus_metrics["average_consensus_time"] = (
            current_time_avg * (total_attempts - 1) + consensus_time
        ) / total_attempts

        # Record in history
        consensus_record = {
            "timestamp": time.time(),
            "strategy": strategy,
            "quality": quality,
            "consensus_time": consensus_time,
            "success": True,
        }

        self.consensus_history.append(consensus_record)

        # Keep history manageable
        if len(self.consensus_history) > 1000:
            self.consensus_history = self.consensus_history[-1000:]

    def _record_consensus_failure(
        self, strategy: str, quality: float, consensus_time: float
    ) -> None:
        """Record failed consensus building."""
        if strategy in self.consensus_metrics["strategy_usage"]:
            self.consensus_metrics["strategy_usage"][strategy] += 1

        if strategy in self.strategy_effectiveness:
            self.strategy_effectiveness[strategy]["attempts"] += 1

        # Record in history
        consensus_record = {
            "timestamp": time.time(),
            "strategy": strategy,
            "quality": quality,
            "consensus_time": consensus_time,
            "success": False,
        }

        self.consensus_history.append(consensus_record)

    def get_consensus_success_rate(self) -> float:
        """Get current consensus success rate."""
        if self.consensus_metrics["total_consensus_attempts"] == 0:
            return 0.0

        return (
            self.consensus_metrics["successful_consensus"]
            / self.consensus_metrics["total_consensus_attempts"]
        )

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive consensus performance summary."""
        success_rate = self.get_consensus_success_rate()

        # Find best performing strategy
        best_strategy = max(
            self.strategy_effectiveness.items(),
            key=lambda x: x[1]["avg_quality"] if x[1]["attempts"] > 0 else 0,
            default=("none", {"avg_quality": 0}),
        )

        return {
            "consensus_metrics": self.consensus_metrics.copy(),
            "consensus_success_rate": success_rate,
            "strategy_effectiveness": self.strategy_effectiveness.copy(),
            "best_performing_strategy": {
                "name": best_strategy[0],
                "avg_quality": best_strategy[1]["avg_quality"],
                "success_rate": best_strategy[1]["successes"]
                / max(best_strategy[1]["attempts"], 1),
            },
            "recent_consensus_history": self.consensus_history[-20:],
            "performance_status": "excellent"
            if success_rate > 0.9
            else "good"
            if success_rate > 0.7
            else "needs_improvement",
        }
