"""
Unit tests for MultiAgentRLCoordinator
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.marl.coordination.marl_coordinator import MultiAgentRLCoordinator
from utils.exceptions import CoordinationError


class TestMultiAgentRLCoordinator:
    """Test cases for MultiAgentRLCoordinator."""

    @pytest.fixture
    def coordinator_config(self):
        """Create test configuration for coordinator."""
        return {
            "generator": {"learning_rate": 0.001},
            "validator": {"learning_rate": 0.001},
            "curriculum": {"learning_rate": 0.001},
            "coordination": {"consensus_threshold": 0.7},
        }

    @pytest.fixture
    def coordinator(self, coordinator_config):
        """Create test coordinator instance."""
        with (
            patch("core.marl.coordination.marl_coordinator.GeneratorRLAgent"),
            patch("core.marl.coordination.marl_coordinator.ValidatorRLAgent"),
            patch("core.marl.coordination.marl_coordinator.CurriculumRLAgent"),
            patch("core.marl.coordination.marl_coordinator.AgentCommunicationProtocol"),
            patch("core.marl.coordination.marl_coordinator.CoordinationPolicy"),
        ):
            coordinator = MultiAgentRLCoordinator(coordinator_config)
            return coordinator

    @pytest.fixture
    def sample_request(self):
        """Create sample content generation request."""
        return {
            "domain": "mathematics",
            "difficulty_level": "high_school",
            "learning_objectives": ["Solve quadratic equations", "Graph functions"],
            "target_audience": "students",
            "topic": "algebra",
        }

    def test_initialization(self, coordinator_config):
        """Test coordinator initialization."""
        with (
            patch(
                "core.marl.coordination.marl_coordinator.GeneratorRLAgent"
            ) as mock_gen,
            patch(
                "core.marl.coordination.marl_coordinator.ValidatorRLAgent"
            ) as mock_val,
            patch(
                "core.marl.coordination.marl_coordinator.CurriculumRLAgent"
            ) as mock_cur,
            patch(
                "core.marl.coordination.marl_coordinator.AgentCommunicationProtocol"
            ) as mock_comm,
            patch(
                "core.marl.coordination.marl_coordinator.CoordinationPolicy"
            ) as mock_policy,
        ):
            coordinator = MultiAgentRLCoordinator(coordinator_config)

            # Verify agents were created
            mock_gen.assert_called_once_with(
                agent_id="generator", config=coordinator_config["generator"]
            )
            mock_val.assert_called_once_with(
                agent_id="validator", config=coordinator_config["validator"]
            )
            mock_cur.assert_called_once_with(
                agent_id="curriculum", config=coordinator_config["curriculum"]
            )

            # Verify coordination infrastructure
            mock_comm.assert_called_once()
            mock_policy.assert_called_once_with(
                config=coordinator_config["coordination"]
            )

            # Verify initial metrics
            assert coordinator.coordination_metrics["total_requests"] == 0
            assert coordinator.coordination_metrics["successful_coordinations"] == 0
            assert coordinator.coordination_metrics["failed_coordinations"] == 0

    def test_agent_registration(self, coordinator):
        """Test agent registration with communication protocol."""
        # Verify register_agent was called for each agent
        expected_calls = [
            (("generator", coordinator.generator_agent),),
            (("validator", coordinator.validator_agent),),
            (("curriculum", coordinator.curriculum_agent),),
        ]

        actual_calls = coordinator.communication_protocol.register_agent.call_args_list
        assert len(actual_calls) == 3

        # Check that all agent IDs were registered
        registered_ids = [call[0][0] for call in actual_calls]
        assert "generator" in registered_ids
        assert "validator" in registered_ids
        assert "curriculum" in registered_ids

    @pytest.mark.asyncio
    async def test_coordinate_generation_success(self, coordinator, sample_request):
        """Test successful coordination workflow."""
        # Mock agent actions
        mock_agent_actions = {
            "generator": {
                "agent_id": "generator",
                "action_type": "generation_strategy",
                "strategy": "structured_reasoning",
                "confidence": 0.8,
            },
            "validator": {
                "agent_id": "validator",
                "action_type": "validation_strategy",
                "strategy": "standard_validation",
                "confidence": 0.7,
            },
            "curriculum": {
                "agent_id": "curriculum",
                "action_type": "curriculum_strategy",
                "strategy": "linear_progression",
                "confidence": 0.9,
            },
        }

        # Mock coordinated action
        mock_coordinated_action = MagicMock()
        mock_coordinated_action.coordination_strategy = "consensus"
        mock_coordinated_action.confidence = 0.8
        mock_coordinated_action.quality = 0.85
        mock_coordinated_action.conflict_resolution_applied = False
        mock_coordinated_action.generator_strategy = {
            "strategy": "structured_reasoning"
        }
        mock_coordinated_action.validation_criteria = {
            "strategy": "standard_validation"
        }
        mock_coordinated_action.curriculum_guidance = {"strategy": "linear_progression"}

        # Mock execution result
        mock_execution_result = {
            "content": {"text": "Generated content", "quality_score": 0.8},
            "curriculum_improvements": {"confidence": 0.9},
            "validation_result": {
                "quality_prediction": 0.8,
                "passes_threshold": True,
                "confidence": 0.7,
            },
            "coordination_metadata": {
                "strategy": "consensus",
                "confidence": 0.8,
                "quality_score": 0.85,
                "conflict_resolution_applied": False,
            },
        }

        # Mock methods
        coordinator._collect_agent_actions = AsyncMock(return_value=mock_agent_actions)
        coordinator._coordinate_actions = AsyncMock(
            return_value=mock_coordinated_action
        )
        coordinator._execute_coordinated_action = AsyncMock(
            return_value=mock_execution_result
        )
        coordinator._process_results = AsyncMock(
            return_value={
                "content": mock_execution_result["content"],
                "validation": mock_execution_result["validation_result"],
                "curriculum_guidance": mock_execution_result["curriculum_improvements"],
                "coordination_metadata": mock_execution_result["coordination_metadata"],
                "learning_updates": {
                    "rewards": {"generator": 0.8, "validator": 0.7, "curriculum": 0.9}
                },
            }
        )

        # Execute coordination
        result = await coordinator.coordinate_generation(sample_request)

        # Verify workflow steps were called
        coordinator._collect_agent_actions.assert_called_once_with(sample_request)
        coordinator._coordinate_actions.assert_called_once_with(
            mock_agent_actions, sample_request
        )
        coordinator._execute_coordinated_action.assert_called_once_with(
            mock_coordinated_action, sample_request
        )
        coordinator._process_results.assert_called_once()

        # Verify result structure
        assert "content" in result
        assert "validation" in result
        assert "curriculum_guidance" in result
        assert "coordination_metadata" in result
        assert "learning_updates" in result

        # Verify metrics updated
        assert coordinator.coordination_metrics["total_requests"] == 1
        assert coordinator.coordination_metrics["successful_coordinations"] == 1

    @pytest.mark.asyncio
    async def test_coordinate_generation_failure(self, coordinator, sample_request):
        """Test coordination failure handling."""
        # Mock failure in action collection
        coordinator._collect_agent_actions = AsyncMock(
            side_effect=Exception("Agent communication failed")
        )

        # Execute coordination and expect failure
        with pytest.raises(CoordinationError) as exc_info:
            await coordinator.coordinate_generation(sample_request)

        assert "MARL coordination failed" in str(exc_info.value)

        # Verify failure metrics
        assert coordinator.coordination_metrics["total_requests"] == 1
        assert coordinator.coordination_metrics["failed_coordinations"] == 1
        assert coordinator.coordination_metrics["successful_coordinations"] == 0

    @pytest.mark.asyncio
    async def test_collect_agent_actions(self, coordinator, sample_request):
        """Test agent action collection."""
        # Mock agent responses
        mock_generator_action = {
            "agent_id": "generator",
            "strategy": "structured_reasoning",
            "confidence": 0.8,
        }
        mock_validator_action = {
            "agent_id": "validator",
            "strategy": "standard_validation",
            "confidence": 0.7,
        }
        mock_curriculum_action = {
            "agent_id": "curriculum",
            "strategy": "linear_progression",
            "confidence": 0.9,
        }

        coordinator._get_generator_action = AsyncMock(
            return_value=mock_generator_action
        )
        coordinator._get_validator_action = AsyncMock(
            return_value=mock_validator_action
        )
        coordinator._get_curriculum_action = AsyncMock(
            return_value=mock_curriculum_action
        )

        # Collect actions
        actions = await coordinator._collect_agent_actions(sample_request)

        # Verify all agents were called
        coordinator._get_generator_action.assert_called_once_with(sample_request)
        coordinator._get_validator_action.assert_called_once_with(sample_request)
        coordinator._get_curriculum_action.assert_called_once_with(sample_request)

        # Verify action structure
        assert "generator" in actions
        assert "validator" in actions
        assert "curriculum" in actions
        assert actions["generator"] == mock_generator_action
        assert actions["validator"] == mock_validator_action
        assert actions["curriculum"] == mock_curriculum_action

    @pytest.mark.asyncio
    async def test_collect_agent_actions_with_failure(
        self, coordinator, sample_request
    ):
        """Test agent action collection with agent failure."""
        # Mock one agent failing
        coordinator._get_generator_action = AsyncMock(
            return_value={"agent_id": "generator"}
        )
        coordinator._get_validator_action = AsyncMock(
            side_effect=Exception("Validator failed")
        )
        coordinator._get_curriculum_action = AsyncMock(
            return_value={"agent_id": "curriculum"}
        )
        coordinator._get_fallback_action = MagicMock(
            return_value={"agent_id": "validator", "is_fallback": True}
        )

        # Collect actions
        actions = await coordinator._collect_agent_actions(sample_request)

        # Verify fallback was used
        coordinator._get_fallback_action.assert_called_once_with(
            "validator", sample_request
        )
        assert actions["validator"]["is_fallback"] is True

    @pytest.mark.asyncio
    async def test_get_generator_action(self, coordinator, sample_request):
        """Test generator action retrieval."""
        # Mock generator agent methods
        mock_state = [0.1, 0.2, 0.3]
        mock_strategy = MagicMock()
        mock_strategy.name = "structured_reasoning"
        mock_strategy.parameters = {"structure_weight": 0.8}

        coordinator.generator_agent.get_state_representation.return_value = mock_state
        coordinator.generator_agent.select_action.return_value = 0
        coordinator.generator_agent.strategies = [mock_strategy]
        coordinator.generator_agent.get_action_confidence.return_value = 0.8
        coordinator.generator_agent.summarize_state.return_value = "test_summary"

        # Get action
        action = await coordinator._get_generator_action(sample_request)

        # Verify action structure
        assert action["agent_id"] == "generator"
        assert action["action_type"] == "generation_strategy"
        assert action["strategy"] == "structured_reasoning"
        assert action["parameters"] == {"structure_weight": 0.8}
        assert action["confidence"] == 0.8
        assert action["state_summary"] == "test_summary"

    @pytest.mark.asyncio
    async def test_get_validator_action(self, coordinator, sample_request):
        """Test validator action retrieval."""
        # Mock validator agent methods
        mock_state = [0.4, 0.5, 0.6]
        mock_strategy = MagicMock()
        mock_strategy.name = "standard_validation"
        mock_strategy.parameters = {"threshold": 0.7}

        coordinator.validator_agent.get_state_representation.return_value = mock_state
        coordinator.validator_agent.select_action.return_value = 0
        coordinator.validator_agent.strategies = [mock_strategy]
        coordinator.validator_agent.get_action_confidence.return_value = 0.7
        coordinator.validator_agent._get_threshold_for_strategy.return_value = 0.7

        # Get action
        action = await coordinator._get_validator_action(sample_request)

        # Verify action structure
        assert action["agent_id"] == "validator"
        assert action["action_type"] == "validation_strategy"
        assert action["strategy"] == "standard_validation"
        assert action["parameters"] == {"threshold": 0.7}
        assert action["confidence"] == 0.7
        assert action["threshold"] == 0.7

    @pytest.mark.asyncio
    async def test_get_curriculum_action(self, coordinator, sample_request):
        """Test curriculum action retrieval."""
        # Mock curriculum agent methods
        mock_state = [0.7, 0.8, 0.9]
        mock_strategy = MagicMock()
        mock_strategy.name = "linear_progression"
        mock_strategy.parameters = {"sequence_strength": 0.9, "focus": "conceptual"}

        coordinator.curriculum_agent.get_state_representation.return_value = mock_state
        coordinator.curriculum_agent.select_action.return_value = 0
        coordinator.curriculum_agent.strategies = [mock_strategy]
        coordinator.curriculum_agent.get_action_confidence.return_value = 0.9

        # Get action
        action = await coordinator._get_curriculum_action(sample_request)

        # Verify action structure
        assert action["agent_id"] == "curriculum"
        assert action["action_type"] == "curriculum_strategy"
        assert action["strategy"] == "linear_progression"
        assert action["parameters"] == {"sequence_strength": 0.9, "focus": "conceptual"}
        assert action["confidence"] == 0.9
        assert action["pedagogical_focus"] == "conceptual"

    def test_get_fallback_action(self, coordinator, sample_request):
        """Test fallback action generation."""
        # Test generator fallback
        fallback = coordinator._get_fallback_action("generator", sample_request)
        assert fallback["agent_id"] == "generator"
        assert fallback["strategy"] == "structured_reasoning"
        assert fallback["is_fallback"] is True

        # Test validator fallback
        fallback = coordinator._get_fallback_action("validator", sample_request)
        assert fallback["agent_id"] == "validator"
        assert fallback["strategy"] == "standard_validation_medium_threshold"
        assert fallback["is_fallback"] is True

        # Test curriculum fallback
        fallback = coordinator._get_fallback_action("curriculum", sample_request)
        assert fallback["agent_id"] == "curriculum"
        assert fallback["strategy"] == "linear_progression"
        assert fallback["is_fallback"] is True

        # Test unknown agent fallback
        fallback = coordinator._get_fallback_action("unknown", sample_request)
        assert fallback["agent_id"] == "unknown"
        assert fallback["strategy"] == "default"
        assert fallback["confidence"] == 0.3

    @pytest.mark.asyncio
    async def test_coordinate_actions(self, coordinator, sample_request):
        """Test action coordination through policy."""
        mock_agent_actions = {
            "generator": {"strategy": "structured_reasoning"},
            "validator": {"strategy": "standard_validation"},
            "curriculum": {"strategy": "linear_progression"},
        }

        mock_coordinated_action = MagicMock()
        mock_coordinated_action.coordination_strategy = "consensus"

        # Mock agent performance summaries
        coordinator.generator_agent.get_performance_summary.return_value = {
            "success_rate": 0.8
        }
        coordinator.validator_agent.get_performance_summary.return_value = {
            "success_rate": 0.7
        }
        coordinator.curriculum_agent.get_performance_summary.return_value = {
            "success_rate": 0.9
        }

        # Mock coordination policy
        coordinator.coordination_policy.coordinate = AsyncMock(
            return_value=mock_coordinated_action
        )

        # Coordinate actions
        result = await coordinator._coordinate_actions(
            mock_agent_actions, sample_request
        )

        # Verify coordination policy was called with correct parameters
        coordinator.coordination_policy.coordinate.assert_called_once()
        call_args = coordinator.coordination_policy.coordinate.call_args

        assert call_args[0][0] == mock_agent_actions  # agent_actions

        # Verify state context
        state = call_args[0][1]
        assert state["domain"] == "mathematics"
        assert state["difficulty_level"] == "high_school"
        assert state["learning_objectives"] == [
            "Solve quadratic equations",
            "Graph functions",
        ]

        # Verify agent context
        agent_context = call_args[0][2]
        assert "agent_performance" in agent_context
        assert "generator" in agent_context["agent_performance"]
        assert "validator" in agent_context["agent_performance"]
        assert "curriculum" in agent_context["agent_performance"]

        assert result == mock_coordinated_action

    def test_calculate_rewards(self, coordinator, sample_request):
        """Test reward calculation based on results."""
        result = {
            "validation_result": {
                "quality_prediction": 0.8,
                "passes_threshold": True,
                "confidence": 0.7,
            },
            "curriculum_improvements": {
                "confidence": 0.9,
                "objective_alignment": {"alignment_score": 0.85},
            },
        }

        rewards = coordinator._calculate_rewards(result, sample_request)

        # Verify reward structure
        assert "generator" in rewards
        assert "validator" in rewards
        assert "curriculum" in rewards

        # Verify generator reward (quality + bonus for passing)
        assert rewards["generator"] == 1.0  # 0.8 + 0.2 bonus

        # Verify validator reward (confidence)
        assert rewards["validator"] == 0.7

        # Verify curriculum reward (average of confidence and alignment)
        expected_curriculum = (0.9 + 0.85) / 2
        assert rewards["curriculum"] == expected_curriculum

    def test_find_strategy_index(self, coordinator):
        """Test strategy index finding."""
        # Mock strategies
        mock_strategies = [
            MagicMock(name="strategy1"),
            MagicMock(name="strategy2"),
            MagicMock(name="strategy3"),
        ]
        mock_strategies[0].name = "first_strategy"
        mock_strategies[1].name = "second_strategy"
        mock_strategies[2].name = "third_strategy"

        # Test finding existing strategy
        index = coordinator._find_strategy_index(mock_strategies, "second_strategy")
        assert index == 1

        # Test finding non-existent strategy (should return 0)
        index = coordinator._find_strategy_index(mock_strategies, "nonexistent")
        assert index == 0

    def test_update_coordination_metrics(self, coordinator):
        """Test coordination metrics updates."""
        # Test successful coordination
        coordinator._update_coordination_metrics(1.5, success=True)
        assert coordinator.coordination_metrics["successful_coordinations"] == 1
        assert coordinator.coordination_metrics["failed_coordinations"] == 0
        assert coordinator.coordination_metrics["average_coordination_time"] == 1.5

        # Test failed coordination
        coordinator._update_coordination_metrics(2.0, success=False)
        assert coordinator.coordination_metrics["successful_coordinations"] == 1
        assert coordinator.coordination_metrics["failed_coordinations"] == 1
        assert (
            coordinator.coordination_metrics["average_coordination_time"] == 1.75
        )  # (1.5 + 2.0) / 2

    def test_get_coordination_success_rate(self, coordinator):
        """Test coordination success rate calculation."""
        # Test with no coordinations
        assert coordinator.get_coordination_success_rate() == 0.0

        # Test with some coordinations
        coordinator.coordination_metrics["successful_coordinations"] = 3
        coordinator.coordination_metrics["failed_coordinations"] = 1
        assert coordinator.get_coordination_success_rate() == 0.75

    def test_get_performance_summary(self, coordinator):
        """Test performance summary generation."""
        # Mock agent performance summaries
        coordinator.generator_agent.get_performance_summary.return_value = {
            "metric": "generator"
        }
        coordinator.validator_agent.get_performance_summary.return_value = {
            "metric": "validator"
        }
        coordinator.curriculum_agent.get_performance_summary.return_value = {
            "metric": "curriculum"
        }

        # Set some coordination metrics
        coordinator.coordination_metrics["successful_coordinations"] = 5
        coordinator.coordination_metrics["failed_coordinations"] = 1

        summary = coordinator.get_performance_summary()

        # Verify summary structure
        assert "coordination_metrics" in summary
        assert "success_rate" in summary
        assert "agent_performance" in summary

        assert summary["success_rate"] == 5 / 6  # 5 successful out of 6 total
        assert summary["agent_performance"]["generator"]["metric"] == "generator"
        assert summary["agent_performance"]["validator"]["metric"] == "validator"
        assert summary["agent_performance"]["curriculum"]["metric"] == "curriculum"

    def test_summarize_request(self, coordinator, sample_request):
        """Test request summarization."""
        summary = coordinator._summarize_request(sample_request)

        expected = "domain=mathematics, difficulty=high_school, objectives_count=2"
        assert summary == expected

        # Test with minimal request
        minimal_request = {}
        summary = coordinator._summarize_request(minimal_request)
        expected = "domain=unknown, difficulty=unknown, objectives_count=0"
        assert summary == expected

    @pytest.mark.asyncio
    async def test_shutdown(self, coordinator):
        """Test coordinator shutdown."""
        # Mock agent save_checkpoint methods
        coordinator.generator_agent.save_checkpoint = MagicMock()
        coordinator.validator_agent.save_checkpoint = MagicMock()
        coordinator.curriculum_agent.save_checkpoint = MagicMock()

        # Mock communication protocol shutdown
        coordinator.communication_protocol.shutdown = AsyncMock()

        # Shutdown coordinator
        await coordinator.shutdown()

        # Verify shutdown steps
        coordinator.communication_protocol.shutdown.assert_called_once()
        coordinator.generator_agent.save_checkpoint.assert_called_once_with(
            "generator_final"
        )
        coordinator.validator_agent.save_checkpoint.assert_called_once_with(
            "validator_final"
        )
        coordinator.curriculum_agent.save_checkpoint.assert_called_once_with(
            "curriculum_final"
        )

    @pytest.mark.asyncio
    async def test_shutdown_with_checkpoint_failure(self, coordinator):
        """Test coordinator shutdown with checkpoint save failure."""
        # Mock agent save_checkpoint methods with failure
        coordinator.generator_agent.save_checkpoint = MagicMock(
            side_effect=Exception("Save failed")
        )
        coordinator.validator_agent.save_checkpoint = MagicMock()
        coordinator.curriculum_agent.save_checkpoint = MagicMock()

        # Mock communication protocol shutdown
        coordinator.communication_protocol.shutdown = AsyncMock()

        # Shutdown should not raise exception even if checkpoint save fails
        await coordinator.shutdown()

        # Verify shutdown still completed
        coordinator.communication_protocol.shutdown.assert_called_once()


class TestMARLCoordinatorIntegration:
    """Integration tests for MARL coordinator."""

    @pytest.fixture
    def integration_coordinator(self):
        """Create coordinator for integration testing."""
        config = {
            "generator": {"learning_rate": 0.001, "epsilon": 0.1},
            "validator": {"learning_rate": 0.001, "epsilon": 0.1},
            "curriculum": {"learning_rate": 0.001, "epsilon": 0.1},
            "coordination": {"consensus_threshold": 0.7},
        }

        with (
            patch("core.marl.coordination.marl_coordinator.GeneratorRLAgent"),
            patch("core.marl.coordination.marl_coordinator.ValidatorRLAgent"),
            patch("core.marl.coordination.marl_coordinator.CurriculumRLAgent"),
            patch("core.marl.coordination.marl_coordinator.AgentCommunicationProtocol"),
            patch("core.marl.coordination.marl_coordinator.CoordinationPolicy"),
        ):
            return MultiAgentRLCoordinator(config)

    @pytest.mark.asyncio
    async def test_full_coordination_workflow(self, integration_coordinator):
        """Test complete coordination workflow integration."""
        request = {
            "domain": "science",
            "difficulty_level": "college",
            "learning_objectives": ["Understand photosynthesis"],
            "target_audience": "biology_students",
        }

        # Mock the entire workflow
        integration_coordinator._collect_agent_actions = AsyncMock(
            return_value={
                "generator": {"strategy": "scientific_reasoning", "confidence": 0.8},
                "validator": {
                    "strategy": "domain_expert_validation",
                    "confidence": 0.9,
                },
                "curriculum": {
                    "strategy": "conceptual_progression",
                    "confidence": 0.85,
                },
            }
        )

        mock_coordinated_action = MagicMock()
        mock_coordinated_action.coordination_strategy = "expert_priority"
        mock_coordinated_action.confidence = 0.85
        mock_coordinated_action.quality = 0.9
        mock_coordinated_action.conflict_resolution_applied = True
        mock_coordinated_action.generator_strategy = {
            "strategy": "scientific_reasoning"
        }
        mock_coordinated_action.validation_criteria = {
            "strategy": "domain_expert_validation"
        }
        mock_coordinated_action.curriculum_guidance = {
            "strategy": "conceptual_progression"
        }

        integration_coordinator._coordinate_actions = AsyncMock(
            return_value=mock_coordinated_action
        )
        integration_coordinator._execute_coordinated_action = AsyncMock(
            return_value={
                "content": {"text": "Photosynthesis explanation", "quality_score": 0.9},
                "curriculum_improvements": {
                    "confidence": 0.85,
                    "objective_alignment": {"alignment_score": 0.9},
                },
                "validation_result": {
                    "quality_prediction": 0.9,
                    "passes_threshold": True,
                    "confidence": 0.9,
                },
                "coordination_metadata": {
                    "strategy": "expert_priority",
                    "confidence": 0.85,
                },
            }
        )
        integration_coordinator._process_results = AsyncMock(
            return_value={
                "content": {"text": "Photosynthesis explanation", "quality_score": 0.9},
                "validation": {"quality_prediction": 0.9, "passes_threshold": True},
                "curriculum_guidance": {"confidence": 0.85},
                "coordination_metadata": {"strategy": "expert_priority"},
                "learning_updates": {
                    "rewards": {"generator": 1.1, "validator": 0.9, "curriculum": 0.875}
                },
            }
        )

        # Execute full workflow
        result = await integration_coordinator.coordinate_generation(request)

        # Verify complete result structure
        assert "content" in result
        assert "validation" in result
        assert "curriculum_guidance" in result
        assert "coordination_metadata" in result
        assert "learning_updates" in result

        # Verify metrics were updated
        assert integration_coordinator.coordination_metrics["total_requests"] == 1
        assert (
            integration_coordinator.coordination_metrics["successful_coordinations"]
            == 1
        )
        assert integration_coordinator.get_coordination_success_rate() == 1.0

    @pytest.mark.asyncio
    async def test_error_recovery_workflow(self, integration_coordinator):
        """Test error recovery in coordination workflow."""
        request = {"domain": "mathematics", "difficulty_level": "elementary"}

        # Mock partial failure and recovery
        integration_coordinator._collect_agent_actions = AsyncMock(
            return_value={
                "generator": {"strategy": "basic_explanation", "confidence": 0.6},
                "validator": {
                    "strategy": "basic_validation",
                    "confidence": 0.5,
                    "is_fallback": True,
                },
                "curriculum": {"strategy": "elementary_progression", "confidence": 0.7},
            }
        )

        mock_coordinated_action = MagicMock()
        mock_coordinated_action.coordination_strategy = "fallback_consensus"
        mock_coordinated_action.confidence = 0.6
        mock_coordinated_action.quality = 0.65
        mock_coordinated_action.conflict_resolution_applied = True
        mock_coordinated_action.generator_strategy = {"strategy": "basic_explanation"}
        mock_coordinated_action.validation_criteria = {"strategy": "basic_validation"}
        mock_coordinated_action.curriculum_guidance = {
            "strategy": "elementary_progression"
        }

        integration_coordinator._coordinate_actions = AsyncMock(
            return_value=mock_coordinated_action
        )
        integration_coordinator._execute_coordinated_action = AsyncMock(
            return_value={
                "content": {"text": "Basic math explanation", "quality_score": 0.65},
                "curriculum_improvements": {
                    "confidence": 0.7,
                    "objective_alignment": {"alignment_score": 0.6},
                },
                "validation_result": {
                    "quality_prediction": 0.65,
                    "passes_threshold": True,
                    "confidence": 0.5,
                },
                "coordination_metadata": {
                    "strategy": "fallback_consensus",
                    "confidence": 0.6,
                },
            }
        )
        integration_coordinator._process_results = AsyncMock(
            return_value={
                "content": {"text": "Basic math explanation", "quality_score": 0.65},
                "validation": {"quality_prediction": 0.65, "passes_threshold": True},
                "curriculum_guidance": {"confidence": 0.7},
                "coordination_metadata": {"strategy": "fallback_consensus"},
                "learning_updates": {
                    "rewards": {"generator": 0.85, "validator": 0.5, "curriculum": 0.65}
                },
            }
        )

        # Execute workflow with recovery
        result = await integration_coordinator.coordinate_generation(request)

        # Verify recovery was successful
        assert result["coordination_metadata"]["strategy"] == "fallback_consensus"
        assert (
            integration_coordinator.coordination_metrics["successful_coordinations"]
            == 1
        )
        assert integration_coordinator.coordination_metrics["failed_coordinations"] == 0
