"""
Unit tests for MARL Integration Adapters
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.marl.integration.marl_adapter import (
    MARLOrchestrationAdapter,
    MARLPipelineIntegration,
    create_marl_integration,
)
from utils.costs import CostTracker
from utils.exceptions import CoordinationError, PipelineError


class TestMARLOrchestrationAdapter:
    """Test cases for MARLOrchestrationAdapter."""

    @pytest.fixture
    def adapter_config(self):
        """Create test configuration for adapter."""
        return {
            "marl_enabled": True,
            "fallback_enabled": True,
            "success_threshold": 0.7,
            "marl": {
                "generator": {"learning_rate": 0.001},
                "validator": {"learning_rate": 0.001},
                "curriculum": {"learning_rate": 0.001},
            },
        }

    @pytest.fixture
    def adapter(self, adapter_config):
        """Create test adapter instance."""
        with patch("core.marl.integration.marl_adapter.MultiAgentRLCoordinator"):
            return MARLOrchestrationAdapter(adapter_config)

    @pytest.fixture
    def legacy_request(self):
        """Create sample legacy request."""
        return {
            "subject": "Mathematics",
            "topic": "algebra",
            "difficulty_level": "high_school",
            "domain": "mathematics",
        }

    @pytest.fixture
    def marl_result(self):
        """Create sample MARL result."""
        return {
            "content": {
                "text": "Generated problem statement",
                "solution": "Step-by-step solution",
                "quality_score": 0.85,
            },
            "validation": {
                "quality_prediction": 0.85,
                "passes_threshold": True,
                "confidence": 0.8,
                "feedback": {"strengths": ["Clear explanation"], "improvements": []},
            },
            "curriculum_guidance": {
                "confidence": 0.9,
                "pedagogical_hints": ["Start with basics", "Use visual aids"],
                "difficulty_adjustments": {"current_level": "appropriate"},
            },
            "coordination_metadata": {
                "strategy": "consensus",
                "confidence": 0.8,
                "quality_score": 0.85,
                "conflict_resolution_applied": False,
                "coordination_time": 1.5,
            },
            "learning_updates": {
                "rewards": {"generator": 0.85, "validator": 0.8, "curriculum": 0.9}
            },
        }

    def test_initialization(self, adapter_config):
        """Test adapter initialization."""
        with patch(
            "core.marl.integration.marl_adapter.MultiAgentRLCoordinator"
        ) as mock_coordinator:
            adapter = MARLOrchestrationAdapter(adapter_config)

            # Verify coordinator was created with correct config
            mock_coordinator.assert_called_once_with(adapter_config["marl"])

            # Verify settings
            assert adapter.enabled is True
            assert adapter.fallback_enabled is True
            assert adapter.success_threshold == 0.7

            # Verify initial metrics
            assert adapter.integration_metrics["marl_requests"] == 0
            assert adapter.integration_metrics["marl_successes"] == 0

    def test_initialization_with_defaults(self):
        """Test adapter initialization with default config."""
        with patch("core.marl.integration.marl_adapter.MultiAgentRLCoordinator"):
            adapter = MARLOrchestrationAdapter()

            # Verify default settings
            assert adapter.enabled is True
            assert adapter.fallback_enabled is True
            assert adapter.success_threshold == 0.7

    @pytest.mark.asyncio
    async def test_generate_content_with_marl_success(
        self, adapter, legacy_request, marl_result
    ):
        """Test successful MARL content generation."""
        # Mock MARL coordinator
        adapter.marl_coordinator.coordinate_generation = AsyncMock(
            return_value=marl_result
        )

        # Mock cost tracker
        cost_tracker = MagicMock(spec=CostTracker)
        cost_tracker.log_cost = MagicMock()

        # Generate content
        result = await adapter.generate_content_with_marl(legacy_request, cost_tracker)

        # Verify MARL coordinator was called
        adapter.marl_coordinator.coordinate_generation.assert_called_once()

        # Verify result structure
        assert "problem_statement" in result
        assert "solution" in result
        assert "quality_score" in result
        assert "passes_validation" in result
        assert "marl_coordination" in result

        # Verify content
        assert result["problem_statement"] == "Generated problem statement"
        assert result["solution"] == "Step-by-step solution"
        assert result["quality_score"] == 0.85
        assert result["passes_validation"] is True

        # Verify MARL metadata
        assert result["marl_coordination"]["strategy"] == "consensus"
        assert result["marl_coordination"]["confidence"] == 0.8

        # Verify metrics updated
        assert adapter.integration_metrics["marl_requests"] == 1
        assert adapter.integration_metrics["marl_successes"] == 1

    @pytest.mark.asyncio
    async def test_generate_content_with_marl_disabled(self, adapter, legacy_request):
        """Test content generation with MARL disabled."""
        adapter.enabled = False

        with pytest.raises(PipelineError) as exc_info:
            await adapter.generate_content_with_marl(legacy_request)

        assert "MARL coordination is disabled" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_generate_content_with_marl_failure(self, adapter, legacy_request):
        """Test MARL content generation failure."""
        # Mock MARL coordinator failure
        adapter.marl_coordinator.coordinate_generation = AsyncMock(
            side_effect=Exception("Coordination failed")
        )

        with pytest.raises(CoordinationError) as exc_info:
            await adapter.generate_content_with_marl(legacy_request)

        assert "MARL generation failed" in str(exc_info.value)

        # Verify metrics
        assert adapter.integration_metrics["marl_requests"] == 1
        assert adapter.integration_metrics["marl_successes"] == 0

    def test_convert_to_marl_request(self, adapter, legacy_request):
        """Test conversion from legacy to MARL request format."""
        marl_request = adapter._convert_to_marl_request(legacy_request)

        # Verify required fields
        assert marl_request["domain"] == "mathematics"
        assert marl_request["difficulty_level"] == "high_school"
        assert marl_request["target_audience"] == "students"
        assert marl_request["topic"] == "algebra"
        assert marl_request["subject"] == "Mathematics"

        # Verify learning objectives were created
        assert len(marl_request["learning_objectives"]) > 0

        # Verify original request is preserved
        assert marl_request["original_request"] == legacy_request

    def test_convert_to_marl_request_minimal(self, adapter):
        """Test conversion with minimal legacy request."""
        minimal_request = {"topic": "calculus"}

        marl_request = adapter._convert_to_marl_request(minimal_request)

        # Verify defaults
        assert marl_request["domain"] == "mathematics"  # Calculus is a math keyword
        assert marl_request["difficulty_level"] == "medium"
        assert marl_request["target_audience"] == "students"
        assert marl_request["learning_objectives"] == ["Understand calculus"]

    def test_convert_to_marl_request_empty(self, adapter):
        """Test conversion with empty legacy request."""
        empty_request = {}

        marl_request = adapter._convert_to_marl_request(empty_request)

        # Verify defaults
        assert marl_request["domain"] == "general"
        assert marl_request["difficulty_level"] == "medium"
        assert marl_request["target_audience"] == "students"
        assert marl_request["learning_objectives"] == ["General learning objective"]

    def test_convert_to_legacy_result(self, adapter, marl_result, legacy_request):
        """Test conversion from MARL to legacy result format."""
        legacy_result = adapter._convert_to_legacy_result(marl_result, legacy_request)

        # Verify core content
        assert legacy_result["problem_statement"] == "Generated problem statement"
        assert legacy_result["solution"] == "Step-by-step solution"
        assert legacy_result["quality_score"] == 0.85

        # Verify validation information
        assert legacy_result["passes_validation"] is True
        assert legacy_result["validation_confidence"] == 0.8
        assert "validation_feedback" in legacy_result

        # Verify curriculum information
        assert "curriculum_improvements" in legacy_result
        assert legacy_result["pedagogical_hints"] == [
            "Start with basics",
            "Use visual aids",
        ]

        # Verify MARL metadata
        assert "marl_coordination" in legacy_result
        assert legacy_result["marl_coordination"]["strategy"] == "consensus"
        assert legacy_result["marl_coordination"]["confidence"] == 0.8

        # Verify request context
        assert "request_context" in legacy_result
        assert legacy_result["request_context"]["domain"] == "mathematics"

    def test_track_marl_costs(self, adapter, marl_result):
        """Test MARL cost tracking."""
        cost_tracker = MagicMock(spec=CostTracker)
        cost_tracker.log_cost = MagicMock()

        adapter._track_marl_costs(marl_result, cost_tracker)

        # Verify cost was logged
        cost_tracker.log_cost.assert_called_once()
        call_args = cost_tracker.log_cost.call_args

        assert call_args[0][0] == "marl_coordination"  # cost_type
        assert isinstance(call_args[0][1], float)  # cost amount
        assert isinstance(call_args[0][2], dict)  # metadata

    def test_track_marl_costs_with_conflict_resolution(self, adapter):
        """Test cost tracking with conflict resolution."""
        marl_result_with_conflict = {
            "coordination_metadata": {
                "strategy": "adaptive_consensus",
                "conflict_resolution_applied": True,
                "confidence": 0.9,
            }
        }

        cost_tracker = MagicMock(spec=CostTracker)
        cost_tracker.log_cost = MagicMock()
        adapter._track_marl_costs(marl_result_with_conflict, cost_tracker)

        # Verify higher cost due to conflict resolution and complex strategy
        cost_tracker.log_cost.assert_called_once()
        call_args = cost_tracker.log_cost.call_args

        # Should have higher cost due to adaptive_consensus (1.5x) + conflict resolution
        cost_amount = call_args[0][1]
        assert cost_amount > 0.001  # Base cost

    def test_get_integration_success_rate(self, adapter):
        """Test integration success rate calculation."""
        # Test with no requests
        assert adapter.get_integration_success_rate() == 0.0

        # Test with some requests
        adapter.integration_metrics["marl_requests"] = 10
        adapter.integration_metrics["marl_successes"] = 8
        assert adapter.get_integration_success_rate() == 0.8

    def test_get_integration_metrics(self, adapter):
        """Test integration metrics retrieval."""
        # Set some metrics
        adapter.integration_metrics["marl_requests"] = 5
        adapter.integration_metrics["marl_successes"] = 4
        adapter.integration_metrics["total_coordination_time"] = 10.0

        # Mock coordinator performance
        adapter.marl_coordinator.get_performance_summary = MagicMock(
            return_value={"coordination_success_rate": 0.8}
        )

        metrics = adapter.get_integration_metrics()

        # Verify metrics structure
        assert "integration_metrics" in metrics
        assert "success_rate" in metrics
        assert "average_coordination_time" in metrics
        assert "marl_coordinator_performance" in metrics

        # Verify values
        assert metrics["success_rate"] == 0.8
        assert metrics["average_coordination_time"] == 2.5  # 10.0 / 4

    @pytest.mark.asyncio
    async def test_shutdown(self, adapter):
        """Test adapter shutdown."""
        adapter.marl_coordinator.shutdown = AsyncMock()

        await adapter.shutdown()

        adapter.marl_coordinator.shutdown.assert_called_once()


class TestMARLPipelineIntegration:
    """Test cases for MARLPipelineIntegration."""

    @pytest.fixture
    def integration_config(self):
        """Create test configuration for integration."""
        return {
            "use_marl": True,
            "marl_probability": 1.0,
            "fallback_on_failure": True,
            "marl_enabled": True,
        }

    @pytest.fixture
    def integration(self, integration_config):
        """Create test integration instance."""
        with patch("core.marl.integration.marl_adapter.MARLOrchestrationAdapter"):
            return MARLPipelineIntegration(integration_config)

    @pytest.fixture
    def sample_request(self):
        """Create sample request."""
        return {
            "subject": "Science",
            "topic": "photosynthesis",
            "difficulty_level": "college",
        }

    @pytest.fixture
    def cost_tracker(self):
        """Create mock cost tracker."""
        return MagicMock(spec=CostTracker)

    def test_initialization(self, integration_config):
        """Test integration initialization."""
        with patch(
            "core.marl.integration.marl_adapter.MARLOrchestrationAdapter"
        ) as mock_adapter:
            integration = MARLPipelineIntegration(integration_config)

            # Verify adapter was created
            mock_adapter.assert_called_once_with(integration_config)

            # Verify settings
            assert integration.use_marl is True
            assert integration.marl_probability == 1.0
            assert integration.fallback_on_failure is True

    def test_initialization_with_defaults(self):
        """Test integration initialization with defaults."""
        with patch("core.marl.integration.marl_adapter.MARLOrchestrationAdapter"):
            integration = MARLPipelineIntegration()

            # Verify default settings
            assert integration.use_marl is False
            assert integration.marl_probability == 1.0
            assert integration.fallback_on_failure is True

    @pytest.mark.asyncio
    async def test_generate_with_marl_integration_success(
        self, integration, sample_request, cost_tracker
    ):
        """Test successful MARL integration generation."""
        # Mock successful MARL generation
        mock_result = {"problem_statement": "Generated content", "quality_score": 0.8}
        integration.marl_adapter.generate_content_with_marl = AsyncMock(
            return_value=mock_result
        )

        result_type, result_data = await integration.generate_with_marl_integration(
            sample_request, cost_tracker
        )

        # Verify MARL was used
        integration.marl_adapter.generate_content_with_marl.assert_called_once_with(
            sample_request, cost_tracker
        )

        # Verify result
        assert result_type == "marl_success"
        assert result_data == mock_result

    @pytest.mark.asyncio
    async def test_generate_with_marl_disabled(
        self, integration, sample_request, cost_tracker
    ):
        """Test generation with MARL disabled."""
        integration.use_marl = False

        # Mock fallback generator
        async def mock_fallback(request, tracker):
            return ("fallback_success", {"content": "fallback result"})

        result_type, result_data = await integration.generate_with_marl_integration(
            sample_request, cost_tracker, mock_fallback
        )

        # Verify fallback was used
        assert result_type == "fallback_success"
        assert result_data["content"] == "fallback result"
        assert result_data["generation_method"] == "fallback"

    @pytest.mark.asyncio
    async def test_generate_with_marl_disabled_no_fallback(
        self, integration, sample_request, cost_tracker
    ):
        """Test generation with MARL disabled and no fallback."""
        integration.use_marl = False

        with pytest.raises(PipelineError) as exc_info:
            await integration.generate_with_marl_integration(
                sample_request, cost_tracker
            )

        assert "MARL disabled and no fallback generator provided" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_generate_with_marl_failure_and_fallback(
        self, integration, sample_request, cost_tracker
    ):
        """Test MARL failure with successful fallback."""
        # Mock MARL failure
        integration.marl_adapter.generate_content_with_marl = AsyncMock(
            side_effect=CoordinationError("MARL failed")
        )

        # Mock successful fallback
        async def mock_fallback(request, tracker):
            return ("fallback_success", {"content": "fallback result"})

        result_type, result_data = await integration.generate_with_marl_integration(
            sample_request, cost_tracker, mock_fallback
        )

        # Verify fallback was used
        assert result_type == "fallback_success"
        assert result_data["content"] == "fallback result"

    @pytest.mark.asyncio
    async def test_generate_with_marl_failure_no_fallback(
        self, integration, sample_request, cost_tracker
    ):
        """Test MARL failure without fallback."""
        integration.fallback_on_failure = False

        # Mock MARL failure
        integration.marl_adapter.generate_content_with_marl = AsyncMock(
            side_effect=CoordinationError("MARL failed")
        )

        with pytest.raises(PipelineError) as exc_info:
            await integration.generate_with_marl_integration(
                sample_request, cost_tracker
            )

        assert "MARL generation failed and fallback disabled" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_generate_with_ab_testing(
        self, integration, sample_request, cost_tracker
    ):
        """Test A/B testing functionality."""
        integration.marl_probability = 0.0  # Never use MARL

        # Mock fallback generator
        async def mock_fallback(request, tracker):
            return ("fallback_success", {"content": "fallback result"})

        result_type, result_data = await integration.generate_with_marl_integration(
            sample_request, cost_tracker, mock_fallback
        )

        # Verify fallback was used due to A/B testing
        assert result_type == "fallback_success"
        assert result_data["content"] == "fallback result"

    @pytest.mark.asyncio
    async def test_run_fallback_generation_async(
        self, integration, sample_request, cost_tracker
    ):
        """Test fallback generation with async function."""

        async def async_fallback(request, tracker):
            return ("async_success", {"content": "async result"})

        result_type, result_data = await integration._run_fallback_generation(
            sample_request, cost_tracker, async_fallback
        )

        assert result_type == "async_success"
        assert result_data["content"] == "async result"
        assert result_data["generation_method"] == "fallback"

    @pytest.mark.asyncio
    async def test_run_fallback_generation_sync(
        self, integration, sample_request, cost_tracker
    ):
        """Test fallback generation with sync function."""

        def sync_fallback(request, tracker):
            return ("sync_success", {"content": "sync result"})

        result_type, result_data = await integration._run_fallback_generation(
            sample_request, cost_tracker, sync_fallback
        )

        assert result_type == "sync_success"
        assert result_data["content"] == "sync result"
        assert result_data["generation_method"] == "fallback"

    @pytest.mark.asyncio
    async def test_run_fallback_generation_failure(
        self, integration, sample_request, cost_tracker
    ):
        """Test fallback generation failure."""

        def failing_fallback(request, tracker):
            raise Exception("Fallback failed")

        with pytest.raises(PipelineError) as exc_info:
            await integration._run_fallback_generation(
                sample_request, cost_tracker, failing_fallback
            )

        assert "Both MARL and fallback generation failed" in str(exc_info.value)

    def test_get_integration_status(self, integration):
        """Test integration status retrieval."""
        # Mock adapter metrics
        integration.marl_adapter.get_integration_metrics = MagicMock(
            return_value={"success_rate": 0.8}
        )

        status = integration.get_integration_status()

        # Verify status structure
        assert "marl_enabled" in status
        assert "marl_probability" in status
        assert "fallback_enabled" in status
        assert "adapter_metrics" in status

        # Verify values
        assert status["marl_enabled"] is True
        assert status["marl_probability"] == 1.0
        assert status["fallback_enabled"] is True
        assert status["adapter_metrics"]["success_rate"] == 0.8

    @pytest.mark.asyncio
    async def test_shutdown(self, integration):
        """Test integration shutdown."""
        integration.marl_adapter.shutdown = AsyncMock()

        await integration.shutdown()

        integration.marl_adapter.shutdown.assert_called_once()


class TestMARLIntegrationFactory:
    """Test cases for MARL integration factory function."""

    def test_create_marl_integration(self):
        """Test MARL integration factory function."""
        config = {
            "use_marl": True,
            "marl_probability": 0.8,
            "fallback_on_failure": True,
        }

        with patch("core.marl.integration.marl_adapter.MARLOrchestrationAdapter"):
            integration = create_marl_integration(config)

            assert isinstance(integration, MARLPipelineIntegration)
            assert integration.use_marl is True
            assert integration.marl_probability == 0.8
            assert integration.fallback_on_failure is True


class TestMARLIntegrationIntegration:
    """Integration tests for MARL integration components."""

    @pytest.fixture
    def full_integration_config(self):
        """Create comprehensive integration configuration."""
        return {
            "use_marl": True,
            "marl_probability": 1.0,
            "fallback_on_failure": True,
            "marl_enabled": True,
            "success_threshold": 0.8,
            "marl": {
                "generator": {"learning_rate": 0.001, "epsilon": 0.1},
                "validator": {"learning_rate": 0.001, "epsilon": 0.1},
                "curriculum": {"learning_rate": 0.001, "epsilon": 0.1},
                "coordination": {"consensus_threshold": 0.7},
            },
        }

    @pytest.mark.asyncio
    async def test_end_to_end_integration_workflow(self, full_integration_config):
        """Test complete end-to-end integration workflow."""
        with patch(
            "core.marl.integration.marl_adapter.MultiAgentRLCoordinator"
        ) as mock_coordinator_class:
            # Mock coordinator instance
            mock_coordinator = MagicMock()
            mock_coordinator.coordinate_generation = AsyncMock(
                return_value={
                    "content": {
                        "text": "Integrated content",
                        "solution": "Integrated solution",
                    },
                    "validation": {
                        "quality_prediction": 0.9,
                        "passes_threshold": True,
                        "confidence": 0.85,
                    },
                    "curriculum_guidance": {
                        "confidence": 0.9,
                        "pedagogical_hints": ["Use examples"],
                    },
                    "coordination_metadata": {
                        "strategy": "expert_priority",
                        "confidence": 0.85,
                    },
                    "learning_updates": {
                        "rewards": {
                            "generator": 0.9,
                            "validator": 0.85,
                            "curriculum": 0.9,
                        }
                    },
                }
            )
            mock_coordinator.get_performance_summary = MagicMock(
                return_value={"success_rate": 0.9}
            )
            mock_coordinator.shutdown = AsyncMock()
            mock_coordinator_class.return_value = mock_coordinator

            # Create integration
            integration = MARLPipelineIntegration(full_integration_config)

            # Test request
            request = {
                "subject": "Physics",
                "topic": "quantum mechanics",
                "difficulty_level": "graduate",
                "domain": "science",
            }

            cost_tracker = MagicMock(spec=CostTracker)
            cost_tracker.log_cost = MagicMock()

            # Execute integration
            result_type, result_data = await integration.generate_with_marl_integration(
                request, cost_tracker
            )

            # Verify successful integration
            assert result_type == "marl_success"
            assert "problem_statement" in result_data
            assert "solution" in result_data
            assert "marl_coordination" in result_data
            assert result_data["quality_score"] == 0.9

            # Verify coordinator was called
            mock_coordinator.coordinate_generation.assert_called_once()

            # Test shutdown
            await integration.shutdown()
            mock_coordinator.shutdown.assert_called_once()
