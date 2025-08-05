"""
Unit tests for Coordination Policy Framework

Tests the coordination policy, conflict resolution, and consensus mechanisms
for multi-agent RL coordination.
"""

# Standard Library
from unittest.mock import AsyncMock, patch

# Third-Party Library
import pytest

# SynThesisAI Modules
from core.marl.config import CoordinationConfig
from core.marl.coordination.conflict_resolver import ConflictResolver
from core.marl.coordination.consensus_mechanism import ConsensusMechanism
from core.marl.coordination.coordination_policy import (
    AgentProposal,
    CoordinatedAction,
    CoordinationPolicy,
)
from core.marl.exceptions import CoordinationFailureError


class TestAgentProposal:
    """Test AgentProposal dataclass."""

    def test_initialization(self):
        """Test AgentProposal initialization."""
        proposal = AgentProposal(
            agent_id="generator",
            strategy_name="step_by_step_approach",
            strategy_parameters={"structure_weight": 0.8},
            confidence=0.9,
        )

        assert proposal.agent_id == "generator"
        assert proposal.strategy_name == "step_by_step_approach"
        assert proposal.confidence == 0.9
        assert "structure_weight" in proposal.strategy_parameters
        assert isinstance(proposal.timestamp, float)

    def test_default_values(self):
        """Test AgentProposal default values."""
        proposal = AgentProposal(
            agent_id="validator",
            strategy_name="strict_validation",
            strategy_parameters={},
            confidence=0.7,
        )

        assert proposal.performance_history == {}
        assert proposal.metadata == {}
        assert proposal.timestamp > 0


class TestCoordinatedAction:
    """Test CoordinatedAction dataclass."""

    def test_initialization(self):
        """Test CoordinatedAction initialization."""
        action = CoordinatedAction(
            generator_strategy={"strategy": "step_by_step", "confidence": 0.8},
            validation_criteria={"threshold": 0.7, "strategy": "standard"},
            curriculum_guidance={"progression": "linear", "difficulty": "medium"},
            coordination_confidence=0.85,
        )

        assert action.coordination_confidence == 0.85
        assert action.generator_strategy["strategy"] == "step_by_step"
        assert action.validation_criteria["threshold"] == 0.7
        assert action.curriculum_guidance["progression"] == "linear"
        assert isinstance(action.timestamp, float)

    def test_default_values(self):
        """Test CoordinatedAction default values."""
        action = CoordinatedAction(
            generator_strategy={},
            validation_criteria={},
            curriculum_guidance={},
            coordination_confidence=0.5,
        )

        assert action.coordination_metadata == {}
        assert action.consensus_quality == 0.0
        assert not action.conflict_resolution_applied


class TestCoordinationPolicy:
    """Test CoordinationPolicy class."""

    @pytest.fixture
    def config(self):
        """Create test coordination configuration."""
        return CoordinationConfig(
            consensus_strategy="adaptive_consensus",
            min_consensus_quality=0.7,
            consensus_timeout=30.0,
            max_negotiation_rounds=3,
            conflict_resolution_strategy="weighted_priority",
            message_queue_size=100,
            communication_timeout=10.0,
            coordination_success_threshold=0.85,
            coordination_failure_threshold=0.3,
        )

    @pytest.fixture
    def coordination_policy(self, config):
        """Create test CoordinationPolicy."""
        with patch("core.marl.coordination.coordination_policy.get_marl_logger"):
            return CoordinationPolicy(config)

    @pytest.fixture
    def sample_proposals(self):
        """Create sample agent proposals."""
        return [
            AgentProposal(
                agent_id="generator",
                strategy_name="step_by_step_approach",
                strategy_parameters={"structure_weight": 0.8},
                confidence=0.9,
            ),
            AgentProposal(
                agent_id="validator",
                strategy_name="standard_validation_medium_threshold",
                strategy_parameters={"threshold": 0.7},
                confidence=0.8,
            ),
            AgentProposal(
                agent_id="curriculum",
                strategy_name="linear_progression",
                strategy_parameters={"sequence_strength": 0.9},
                confidence=0.85,
            ),
        ]

    def test_initialization(self, coordination_policy, config):
        """Test CoordinationPolicy initialization."""
        assert coordination_policy.config == config
        assert isinstance(coordination_policy.consensus_mechanism, ConsensusMechanism)
        assert isinstance(coordination_policy.conflict_resolver, ConflictResolver)
        assert coordination_policy.coordination_metrics["total_coordinations"] == 0
        assert "generator" in coordination_policy.agent_performance
        assert "validator" in coordination_policy.agent_performance
        assert "curriculum" in coordination_policy.agent_performance

    @pytest.mark.asyncio
    async def test_coordinate_success(self, coordination_policy, sample_proposals):
        """Test successful coordination."""
        agent_actions = {
            "generator": {
                "strategy": "step_by_step_approach",
                "parameters": {"structure_weight": 0.8},
                "confidence": 0.9,
            },
            "validator": {
                "strategy": "standard_validation_medium_threshold",
                "parameters": {"threshold": 0.7},
                "confidence": 0.8,
            },
            "curriculum": {
                "strategy": "linear_progression",
                "parameters": {"sequence_strength": 0.9},
                "confidence": 0.85,
            },
        }

        state = {"domain": "mathematics", "difficulty_level": "high_school"}
        agent_context = {"agent_performance": {}}

        # Mock consensus mechanism to return successful consensus
        with patch.object(
            coordination_policy.consensus_mechanism,
            "build_consensus",
            new_callable=AsyncMock,
        ) as mock_consensus:
            mock_consensus.return_value = {
                "generator_strategy": {"strategy": "step_by_step_approach"},
                "validation_criteria": {"strategy": "standard_validation"},
                "curriculum_guidance": {"strategy": "linear_progression"},
                "confidence": 0.85,
                "quality": 0.8,
            }

            result = await coordination_policy.coordinate(
                agent_actions, state, agent_context
            )

            assert isinstance(result, CoordinatedAction)
            assert result.coordination_confidence == 0.85
            assert result.consensus_quality == 0.8
            assert coordination_policy.coordination_metrics["total_coordinations"] == 1
            assert (
                coordination_policy.coordination_metrics["successful_coordinations"]
                == 1
            )

    @pytest.mark.asyncio
    async def test_coordinate_with_conflicts(self, coordination_policy):
        """Test coordination with conflicts."""
        # Create conflicting agent actions
        agent_actions = {
            "generator": {
                "strategy": "creative_exploration",
                "parameters": {"creativity_weight": 0.9},
                "confidence": 0.8,
            },
            "validator": {
                "strategy": "strict_validation_high_threshold",
                "parameters": {"threshold": 0.9},
                "confidence": 0.9,
            },
        }

        state = {"domain": "mathematics"}
        agent_context = {}

        # Mock conflict resolver to return resolved proposals
        with patch.object(
            coordination_policy.conflict_resolver, "resolve", new_callable=AsyncMock
        ) as mock_resolver:
            mock_resolver.return_value = [
                AgentProposal(
                    agent_id="generator",
                    strategy_name="structured_reasoning",
                    strategy_parameters={"structure_emphasis": 0.8},
                    confidence=0.7,
                ),
                AgentProposal(
                    agent_id="validator",
                    strategy_name="standard_validation_medium_threshold",
                    strategy_parameters={"threshold": 0.75},
                    confidence=0.8,
                ),
            ]

            # Mock consensus mechanism
            with patch.object(
                coordination_policy.consensus_mechanism,
                "build_consensus",
                new_callable=AsyncMock,
            ) as mock_consensus:
                mock_consensus.return_value = {
                    "generator_strategy": {"strategy": "structured_reasoning"},
                    "validation_criteria": {"strategy": "standard_validation"},
                    "curriculum_guidance": {"strategy": "default"},
                    "confidence": 0.75,
                    "quality": 0.8,
                }

                result = await coordination_policy.coordinate(
                    agent_actions, state, agent_context
                )

                assert isinstance(result, CoordinatedAction)
                assert result.conflict_resolution_applied
                assert (
                    coordination_policy.coordination_metrics["conflicts_detected"] == 1
                )
                assert (
                    coordination_policy.coordination_metrics["conflicts_resolved"] == 1
                )

    def test_detect_conflicts_strategy_incompatibility(
        self, coordination_policy, sample_proposals
    ):
        """Test conflict detection for strategy incompatibility."""
        # Create conflicting proposals
        conflicting_proposals = [
            AgentProposal(
                agent_id="generator",
                strategy_name="creative_exploration",
                strategy_parameters={},
                confidence=0.8,
            ),
            AgentProposal(
                agent_id="validator",
                strategy_name="strict_validation_high_threshold",
                strategy_parameters={},
                confidence=0.9,
            ),
        ]

        conflicts = coordination_policy.detect_conflicts(conflicting_proposals)

        assert conflicts is not None
        assert "strategy_incompatibility" in conflicts
        assert "generator" in conflicts["strategy_incompatibility"]
        assert "validator" in conflicts["strategy_incompatibility"]

    def test_detect_conflicts_low_confidence(self, coordination_policy):
        """Test conflict detection for low confidence."""
        low_confidence_proposals = [
            AgentProposal(
                agent_id="generator",
                strategy_name="step_by_step_approach",
                strategy_parameters={},
                confidence=0.2,  # Very low confidence
            ),
            AgentProposal(
                agent_id="validator",
                strategy_name="standard_validation",
                strategy_parameters={},
                confidence=0.8,
            ),
        ]

        conflicts = coordination_policy.detect_conflicts(low_confidence_proposals)

        assert conflicts is not None
        assert "low_confidence" in conflicts
        assert "generator" in conflicts["low_confidence"]

    def test_detect_no_conflicts(self, coordination_policy, sample_proposals):
        """Test no conflict detection with compatible proposals."""
        conflicts = coordination_policy.detect_conflicts(sample_proposals)

        # Should be None or empty since proposals are compatible
        assert conflicts is None or len(conflicts) == 0

    def test_convert_to_proposals(self, coordination_policy):
        """Test conversion of agent actions to proposals."""
        agent_actions = {
            "generator": {
                "strategy": "step_by_step_approach",
                "parameters": {"structure_weight": 0.8},
                "confidence": 0.9,
                "metadata": {"test": True},
            },
            "validator": {
                "strategy": "standard_validation",
                "parameters": {"threshold": 0.7},
                "confidence": 0.8,
            },
        }

        agent_context = {}
        proposals = coordination_policy._convert_to_proposals(
            agent_actions, agent_context
        )

        assert len(proposals) == 2
        assert proposals[0].agent_id == "generator"
        assert proposals[0].strategy_name == "step_by_step_approach"
        assert proposals[0].confidence == 0.9
        assert proposals[0].metadata["test"]

        assert proposals[1].agent_id == "validator"
        assert proposals[1].strategy_name == "standard_validation"
        assert proposals[1].confidence == 0.8

    def test_apply_fallback_coordination(self, coordination_policy, sample_proposals):
        """Test fallback coordination mechanism."""
        state = {"domain": "mathematics"}
        fallback_proposals = coordination_policy._apply_fallback_coordination(
            sample_proposals, state
        )

        assert len(fallback_proposals) == len(sample_proposals)

        # Primary proposal should be the one with highest confidence
        primary_proposal = max(sample_proposals, key=lambda p: p.confidence)
        assert fallback_proposals[0].agent_id == primary_proposal.agent_id

        # Other proposals should have reduced confidence
        for proposal in fallback_proposals[1:]:
            original_proposal = next(
                p for p in sample_proposals if p.agent_id == proposal.agent_id
            )
            assert proposal.confidence <= original_proposal.confidence

    def test_get_coordination_success_rate(self, coordination_policy):
        """Test coordination success rate calculation."""
        # Initially should be 0
        assert coordination_policy.get_coordination_success_rate() == 0.0

        # Simulate some coordinations
        coordination_policy.coordination_metrics["total_coordinations"] = 10
        coordination_policy.coordination_metrics["successful_coordinations"] = 8

        success_rate = coordination_policy.get_coordination_success_rate()
        assert success_rate == 0.8

    def test_performance_summary(self, coordination_policy):
        """Test performance summary generation."""
        # Add some test data
        coordination_policy.coordination_metrics["total_coordinations"] = 5
        coordination_policy.coordination_metrics["successful_coordinations"] = 4
        coordination_policy.agent_performance["generator"]["proposals"] = 3
        coordination_policy.agent_performance["generator"]["accepted"] = 2
        coordination_policy.agent_performance["generator"]["confidence_sum"] = 2.4

        summary = coordination_policy.get_performance_summary()

        assert "coordination_metrics" in summary
        assert "coordination_success_rate" in summary
        assert "agent_performance" in summary
        assert "performance_status" in summary

        assert summary["coordination_success_rate"] == 0.8
        assert summary["agent_performance"]["generator"]["acceptance_rate"] == 2 / 3
        assert (
            abs(summary["agent_performance"]["generator"]["average_confidence"] - 0.8)
            < 0.001
        )

    @pytest.mark.asyncio
    async def test_coordinate_failure_handling(self, coordination_policy):
        """Test coordination failure handling."""
        agent_actions = {"generator": {"strategy": "test", "confidence": 0.5}}
        state = {}
        agent_context = {}

        # Mock consensus mechanism to raise exception
        with patch.object(
            coordination_policy.consensus_mechanism,
            "build_consensus",
            new_callable=AsyncMock,
        ) as mock_consensus:
            mock_consensus.side_effect = Exception("Test error")

            with pytest.raises(CoordinationFailureError):
                await coordination_policy.coordinate(
                    agent_actions, state, agent_context
                )

            # Should record the failure
            assert len(coordination_policy.coordination_history) > 0
            assert (
                coordination_policy.coordination_history[-1]["error_type"]
                == "Exception"
            )


class TestConflictResolver:
    """Test ConflictResolver class."""

    @pytest.fixture
    def config(self):
        """Create test coordination configuration."""
        return CoordinationConfig(max_negotiation_rounds=3)

    @pytest.fixture
    def conflict_resolver(self, config):
        """Create test ConflictResolver."""
        with patch("core.marl.coordination.conflict_resolver.get_marl_logger"):
            return ConflictResolver(config)

    @pytest.fixture
    def conflicting_proposals(self):
        """Create conflicting proposals."""
        return [
            AgentProposal(
                agent_id="generator",
                strategy_name="creative_exploration",
                strategy_parameters={},
                confidence=0.8,
            ),
            AgentProposal(
                agent_id="validator",
                strategy_name="strict_validation_high_threshold",
                strategy_parameters={},
                confidence=0.9,
            ),
        ]

    def test_initialization(self, conflict_resolver, config):
        """Test ConflictResolver initialization."""
        assert conflict_resolver.config == config
        assert len(conflict_resolver.resolution_strategies) > 0
        assert "weighted_priority" in conflict_resolver.resolution_strategies
        assert "negotiation_based" in conflict_resolver.resolution_strategies
        assert (
            conflict_resolver.agent_priorities["validator"]
            > conflict_resolver.agent_priorities["generator"]
        )

    @pytest.mark.asyncio
    async def test_weighted_priority_resolution(
        self, conflict_resolver, conflicting_proposals
    ):
        """Test weighted priority resolution strategy."""
        conflicts = {"strategy_incompatibility": ["generator", "validator"]}
        state = {}
        agent_context = {}

        resolved = await conflict_resolver._weighted_priority_resolution(
            conflicts, conflicting_proposals, state, agent_context
        )

        assert len(resolved) > 0
        # Validator should be prioritized due to higher priority weight
        primary_agent = resolved[0].agent_id
        assert primary_agent == "validator"  # Higher priority

    @pytest.mark.asyncio
    async def test_compromise_generation(
        self, conflict_resolver, conflicting_proposals
    ):
        """Test compromise generation strategy."""
        conflicts = {"strategy_incompatibility": ["generator", "validator"]}
        state = {}
        agent_context = {}

        resolved = await conflict_resolver._compromise_generation(
            conflicts, conflicting_proposals, state, agent_context
        )

        assert len(resolved) > 0
        # Should create compromise proposals
        compromise_found = any("compromise" in p.agent_id for p in resolved)
        assert compromise_found

    def test_select_resolution_strategy(self, conflict_resolver):
        """Test resolution strategy selection."""
        # Test different conflict types
        strategy_conflicts = {"strategy_incompatibility": ["generator", "validator"]}
        strategy = conflict_resolver._select_resolution_strategy(
            strategy_conflicts, [], {}
        )
        assert strategy == "compromise_generation"

        confidence_conflicts = {"low_confidence": ["generator"]}
        strategy = conflict_resolver._select_resolution_strategy(
            confidence_conflicts, [], {}
        )
        assert strategy == "performance_based"

        parameter_conflicts = {"parameter_mismatch": ["validator", "curriculum"]}
        strategy = conflict_resolver._select_resolution_strategy(
            parameter_conflicts, [], {}
        )
        assert strategy == "negotiation_based"

    def test_get_fallback_strategy(self, conflict_resolver):
        """Test fallback strategy selection."""
        fallback = conflict_resolver._get_fallback_strategy("negotiation_based")
        assert fallback == "weighted_priority"

        fallback = conflict_resolver._get_fallback_strategy("compromise_generation")
        assert fallback == "expert_override"

        fallback = conflict_resolver._get_fallback_strategy("unknown_strategy")
        assert fallback is None

    def test_proposals_conflict_detection(self, conflict_resolver):
        """Test conflict detection between two proposals."""
        prop1 = AgentProposal(
            agent_id="generator",
            strategy_name="creative_exploration",
            strategy_parameters={},
            confidence=0.8,
        )
        prop2 = AgentProposal(
            agent_id="validator",
            strategy_name="strict_validation_high_threshold",
            strategy_parameters={},
            confidence=0.9,
        )

        assert conflict_resolver._proposals_conflict(prop1, prop2)

        # Test compatible proposals
        prop3 = AgentProposal(
            agent_id="validator",
            strategy_name="standard_validation_medium_threshold",
            strategy_parameters={},
            confidence=0.8,
        )

        assert not conflict_resolver._proposals_conflict(prop1, prop3)

    def test_performance_summary(self, conflict_resolver):
        """Test conflict resolver performance summary."""
        # Add some test data
        conflict_resolver.resolution_metrics["total_conflicts"] = 10
        conflict_resolver.resolution_metrics["resolved_conflicts"] = 8
        conflict_resolver.resolution_metrics["resolution_success_rate"] = 0.8  # 8/10
        conflict_resolver.resolution_metrics["resolution_strategy_usage"][
            "weighted_priority"
        ] = 5

        summary = conflict_resolver.get_performance_summary()

        assert "resolution_metrics" in summary
        assert "agent_priorities" in summary
        assert "most_used_strategy" in summary
        assert "performance_status" in summary

        assert summary["most_used_strategy"] == "weighted_priority"
        assert summary["performance_status"] == "good"  # 80% success rate


class TestConsensusMechanism:
    """Test ConsensusMechanism class."""

    @pytest.fixture
    def config(self):
        """Create test coordination configuration."""
        return CoordinationConfig(min_consensus_quality=0.7)

    @pytest.fixture
    def consensus_mechanism(self, config):
        """Create test ConsensusMechanism."""
        with patch("core.marl.coordination.consensus_mechanism.get_marl_logger"):
            return ConsensusMechanism(config)

    @pytest.fixture
    def sample_proposals(self):
        """Create sample proposals for consensus."""
        return [
            AgentProposal(
                agent_id="generator",
                strategy_name="step_by_step_approach",
                strategy_parameters={"structure_weight": 0.8},
                confidence=0.9,
            ),
            AgentProposal(
                agent_id="validator",
                strategy_name="standard_validation_medium_threshold",
                strategy_parameters={"threshold": 0.7},
                confidence=0.8,
            ),
            AgentProposal(
                agent_id="curriculum",
                strategy_name="linear_progression",
                strategy_parameters={"sequence_strength": 0.9},
                confidence=0.85,
            ),
        ]

    def test_initialization(self, consensus_mechanism, config):
        """Test ConsensusMechanism initialization."""
        assert consensus_mechanism.config == config
        assert len(consensus_mechanism.consensus_strategies) > 0
        assert "weighted_average" in consensus_mechanism.consensus_strategies
        assert "majority_vote" in consensus_mechanism.consensus_strategies
        assert "expert_priority" in consensus_mechanism.consensus_strategies

    @pytest.mark.asyncio
    async def test_weighted_average_consensus(
        self, consensus_mechanism, sample_proposals
    ):
        """Test weighted average consensus strategy."""
        state = {}
        agent_context = {}

        consensus = await consensus_mechanism._weighted_average_consensus(
            sample_proposals, state, agent_context
        )

        assert "generator_strategy" in consensus
        assert "validation_criteria" in consensus
        assert "curriculum_guidance" in consensus
        assert "confidence" in consensus
        assert "participating_agents" in consensus

        assert len(consensus["participating_agents"]) == 3
        assert 0.0 <= consensus["confidence"] <= 1.0

    @pytest.mark.asyncio
    async def test_expert_priority_consensus(
        self, consensus_mechanism, sample_proposals
    ):
        """Test expert priority consensus strategy."""
        state = {"domain": "mathematics"}
        agent_context = {}

        consensus = await consensus_mechanism._expert_priority_consensus(
            sample_proposals, state, agent_context
        )

        assert "expert_domain" in consensus
        assert "expert_weights" in consensus
        assert consensus["expert_domain"] == "mathematics"
        assert len(consensus["expert_weights"]) == len(sample_proposals)

    @pytest.mark.asyncio
    async def test_confidence_weighted_consensus(
        self, consensus_mechanism, sample_proposals
    ):
        """Test confidence weighted consensus strategy."""
        state = {}
        agent_context = {}

        consensus = await consensus_mechanism._confidence_weighted_consensus(
            sample_proposals, state, agent_context
        )

        assert "confidence_weights" in consensus
        assert "confidence_emphasis" in consensus
        assert consensus["confidence_emphasis"]
        assert len(consensus["confidence_weights"]) == len(sample_proposals)

    def test_select_consensus_strategy(self, consensus_mechanism, sample_proposals):
        """Test consensus strategy selection."""
        # Test high confidence scenario
        state = {}
        agent_context = {}

        strategy = consensus_mechanism._select_consensus_strategy(
            sample_proposals, state, agent_context
        )
        assert strategy in consensus_mechanism.consensus_strategies

        # Test domain-specific scenario
        state = {"domain": "mathematics"}
        strategy = consensus_mechanism._select_consensus_strategy(
            sample_proposals, state, agent_context
        )
        assert strategy == "expert_priority"

        # Test performance-based scenario (with low confidence and performance data)
        low_confidence_proposals = [
            AgentProposal(
                agent_id="generator",
                strategy_name="step_by_step_approach",
                strategy_parameters={"structure_weight": 0.8},
                confidence=0.6,  # Lower confidence
            ),
            AgentProposal(
                agent_id="curriculum",
                strategy_name="linear_progression",
                strategy_parameters={"sequence_strength": 0.9},
                confidence=0.5,  # Lower confidence
            ),
        ]
        state_no_domain = {}
        agent_context = {"agent_performance": {"generator": {"success_rate": 0.8}}}
        strategy = consensus_mechanism._select_consensus_strategy(
            low_confidence_proposals, state_no_domain, agent_context
        )
        assert strategy == "performance_based"

    def test_validate_consensus_quality(self, consensus_mechanism, sample_proposals):
        """Test consensus quality validation."""
        consensus = {
            "confidence": 0.8,
            "participating_agents": ["generator", "validator", "curriculum"],
            "generator_strategy": {"strategy": "step_by_step_approach"},
            "validation_criteria": {"strategy": "standard_validation_medium_threshold"},
            "curriculum_guidance": {"strategy": "linear_progression"},
        }

        quality = consensus_mechanism._validate_consensus_quality(
            consensus, sample_proposals
        )

        assert 0.0 <= quality <= 1.0
        assert quality > 0.5  # Should be reasonably high for good consensus

    def test_calculate_strategy_coherence(self, consensus_mechanism):
        """Test strategy coherence calculation."""
        # Test coherent combination
        coherent_strategies = [
            "step_by_step_approach",
            "standard_validation_medium_threshold",
            "linear_progression",
        ]
        coherence = consensus_mechanism._calculate_strategy_coherence(
            coherent_strategies
        )
        assert coherence == 1.0

        # Test partially coherent combination
        partial_strategies = [
            "step_by_step_approach",
            "standard_validation_medium_threshold",
            "unknown_strategy",
        ]
        coherence = consensus_mechanism._calculate_strategy_coherence(
            partial_strategies
        )
        assert 0.0 < coherence < 1.0

        # Test incoherent combination
        incoherent_strategies = ["unknown1", "unknown2", "unknown3"]
        coherence = consensus_mechanism._calculate_strategy_coherence(
            incoherent_strategies
        )
        assert coherence == 0.0

    @pytest.mark.asyncio
    async def test_build_consensus_success(self, consensus_mechanism, sample_proposals):
        """Test successful consensus building."""
        state = {}
        agent_context = {}

        consensus = await consensus_mechanism.build_consensus(
            sample_proposals, state, agent_context
        )

        assert "confidence" in consensus
        assert (
            "generator_strategy" in consensus
            or "validation_criteria" in consensus
            or "curriculum_guidance" in consensus
        )
        assert consensus["confidence"] >= 0.0
        assert consensus_mechanism.consensus_metrics["total_consensus_attempts"] == 1

    @pytest.mark.asyncio
    async def test_build_weighted_consensus(
        self, consensus_mechanism, sample_proposals
    ):
        """Test weighted consensus building."""
        weights = [0.5, 0.3, 0.2]
        weighted_proposals = list(zip(sample_proposals, weights))
        state = {}

        consensus = await consensus_mechanism.build_weighted_consensus(
            weighted_proposals, state
        )

        assert "weighted_consensus" in consensus
        assert "weights_used" in consensus
        assert consensus["weighted_consensus"]
        assert len(consensus["weights_used"]) == len(sample_proposals)

    def test_get_consensus_success_rate(self, consensus_mechanism):
        """Test consensus success rate calculation."""
        # Initially should be 0
        assert consensus_mechanism.get_consensus_success_rate() == 0.0

        # Simulate some consensus attempts
        consensus_mechanism.consensus_metrics["total_consensus_attempts"] = 10
        consensus_mechanism.consensus_metrics["successful_consensus"] = 8

        success_rate = consensus_mechanism.get_consensus_success_rate()
        assert success_rate == 0.8

    def test_performance_summary(self, consensus_mechanism):
        """Test consensus mechanism performance summary."""
        # Add some test data
        consensus_mechanism.consensus_metrics["total_consensus_attempts"] = 5
        consensus_mechanism.consensus_metrics["successful_consensus"] = 4
        consensus_mechanism.strategy_effectiveness["weighted_average"]["attempts"] = 3
        consensus_mechanism.strategy_effectiveness["weighted_average"]["successes"] = 2
        consensus_mechanism.strategy_effectiveness["weighted_average"][
            "avg_quality"
        ] = 0.8

        summary = consensus_mechanism.get_performance_summary()

        assert "consensus_metrics" in summary
        assert "consensus_success_rate" in summary
        assert "strategy_effectiveness" in summary
        assert "best_performing_strategy" in summary
        assert "performance_status" in summary

        assert summary["consensus_success_rate"] == 0.8
        assert summary["performance_status"] == "good"
