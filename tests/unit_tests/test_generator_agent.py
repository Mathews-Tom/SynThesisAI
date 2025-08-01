"""
Unit tests for Generator RL Agent.

This module tests the Generator RL Agent implementation including strategy
selection, reward calculation, state representation, and performance tracking.
"""

# Standard Library
import time
from unittest.mock import patch

# Third-Party Library
import numpy as np
import pytest

# SynThesisAI Modules
from core.marl.agents.base_agent import ActionSpace
from core.marl.agents.specialized.generator_agent import (
    GenerationContextEncoder,
    GenerationStrategy,
    GeneratorRLAgent,
)
from core.marl.config import GeneratorAgentConfig
from core.marl.exceptions import AgentFailureError


class TestGenerationStrategy:
    """Test GenerationStrategy class."""

    def test_strategy_initialization(self):
        """Test strategy initialization."""
        params = {"param1": 0.8, "param2": "value"}
        strategy = GenerationStrategy("test_strategy", "Test strategy description", params)

        assert strategy.name == "test_strategy"
        assert strategy.description == "Test strategy description"
        assert strategy.parameters == params
        assert strategy.usage_count == 0
        assert strategy.success_rate == 0.0
        assert strategy.average_quality == 0.0

    def test_strategy_performance_update(self):
        """Test strategy performance tracking."""
        strategy = GenerationStrategy("test", "desc", {})

        # Update with successful performance
        strategy.update_performance(0.8, 0.7, 0.9, True)

        assert strategy.usage_count == 1
        assert strategy.average_quality > 0.0
        assert strategy.average_novelty > 0.0
        assert strategy.average_efficiency > 0.0
        assert strategy.success_rate > 0.0

        # Update with poor performance
        strategy.update_performance(0.3, 0.2, 0.4, False)

        assert strategy.usage_count == 2
        # Averages should be influenced by both values (running average with alpha=0.1)
        assert 0.05 < strategy.average_quality < 0.15  # Should be around 0.102
        assert 0.05 < strategy.average_novelty < 0.12  # Should be around 0.083
        assert 0.08 < strategy.average_efficiency < 0.15  # Should be around 0.121

    def test_strategy_performance_summary(self):
        """Test strategy performance summary."""
        strategy = GenerationStrategy("test", "desc", {})

        # Update performance
        strategy.update_performance(0.8, 0.7, 0.9, True)

        summary = strategy.get_performance_summary()

        assert summary["name"] == "test"
        assert summary["usage_count"] == 1
        assert summary["success_rate"] > 0.0
        assert "overall_score" in summary
        assert summary["overall_score"] > 0.0


class TestGenerationContextEncoder:
    """Test GenerationContextEncoder class."""

    def test_encoder_initialization(self):
        """Test encoder initialization."""
        encoder = GenerationContextEncoder()

        assert len(encoder.domain_mapping) == 6
        assert len(encoder.difficulty_mapping) == 4
        assert "mathematics" in encoder.domain_mapping
        assert "elementary" in encoder.difficulty_mapping

    def test_domain_encoding(self):
        """Test domain encoding."""
        encoder = GenerationContextEncoder()

        # Test known domain
        math_encoding = encoder.encode_domain("mathematics")
        assert len(math_encoding) == 6
        assert math_encoding[0] == 1.0  # Mathematics should be first
        assert sum(math_encoding) == 1.0  # One-hot encoding

        # Test unknown domain
        unknown_encoding = encoder.encode_domain("unknown")
        assert len(unknown_encoding) == 6
        assert sum(unknown_encoding) == 0.0  # All zeros for unknown

    def test_difficulty_encoding(self):
        """Test difficulty level encoding."""
        encoder = GenerationContextEncoder()

        # Test known difficulty
        elementary_encoding = encoder.encode_difficulty("elementary")
        assert len(elementary_encoding) == 4
        assert elementary_encoding[0] == 1.0
        assert sum(elementary_encoding) == 1.0

        # Test with spaces
        high_school_encoding = encoder.encode_difficulty("high school")
        assert len(high_school_encoding) == 4
        assert high_school_encoding[1] == 1.0

    def test_topic_encoding(self):
        """Test topic encoding."""
        encoder = GenerationContextEncoder()

        # Test simple topic
        simple_encoding = encoder.encode_topic("algebra")
        assert len(simple_encoding) == 4
        assert all(isinstance(x, float) for x in simple_encoding)

        # Test complex topic
        complex_encoding = encoder.encode_topic("Advanced Calculus with Applications 123")
        assert len(complex_encoding) == 4
        assert complex_encoding[0] > simple_encoding[0]  # Longer topic
        assert complex_encoding[1] > simple_encoding[1]  # More words

    def test_quality_requirements_encoding(self):
        """Test quality requirements encoding."""
        encoder = GenerationContextEncoder()

        requirements = {
            "accuracy_threshold": 0.9,
            "clarity_threshold": 0.8,
            "completeness_threshold": 0.85,
            "engagement_threshold": 0.7,
        }

        encoding = encoder.encode_quality_requirements(requirements)
        assert len(encoding) == 4
        assert encoding[0] == 0.9
        assert encoding[1] == 0.8
        assert encoding[2] == 0.85
        assert encoding[3] == 0.7

    def test_audience_encoding(self):
        """Test audience encoding."""
        encoder = GenerationContextEncoder()

        # Test student audience
        student_encoding = encoder.encode_audience("undergraduate student")
        assert len(student_encoding) == 4
        assert student_encoding[0] == 1.0  # Contains 'student'

        # Test teacher audience
        teacher_encoding = encoder.encode_audience("high school teacher")
        assert teacher_encoding[1] == 1.0  # Contains 'teacher'

    def test_objectives_encoding(self):
        """Test learning objectives encoding."""
        encoder = GenerationContextEncoder()

        # Test empty objectives
        empty_encoding = encoder.encode_objectives([])
        assert len(empty_encoding) == 4
        assert all(x == 0.0 for x in empty_encoding)

        # Test with objectives
        objectives = [
            "understand basic concepts",
            "apply knowledge to problems",
            "analyze complex scenarios",
        ]

        encoding = encoder.encode_objectives(objectives)
        assert len(encoding) == 4
        assert encoding[0] > 0.0  # Number of objectives
        assert encoding[1] > 0.0  # Average length
        assert encoding[2] > 0.0  # Contains 'understand'
        assert encoding[3] > 0.0  # Contains 'apply'

    def test_coordination_context_encoding(self):
        """Test coordination context encoding."""
        encoder = GenerationContextEncoder()

        context = {
            "coordination_quality": 0.8,
            "consensus_level": 0.7,
            "coordination_success": True,
            "agent_agreement": 0.9,
        }

        encoding = encoder.encode_coordination_context(context)
        assert len(encoding) == 4
        assert encoding[0] == 0.8
        assert encoding[1] == 0.7
        assert encoding[2] == 1.0  # True converted to 1.0
        assert encoding[3] == 0.9


class TestGeneratorRLAgent:
    """Test GeneratorRLAgent class."""

    def test_agent_initialization(self):
        """Test agent initialization."""
        config = GeneratorAgentConfig()
        agent = GeneratorRLAgent(config)

        assert agent.agent_id == "generator"
        assert agent.config == config
        assert len(agent.strategies) == 8  # Default strategies
        assert len(agent.strategy_history) == 0
        assert agent.generation_metrics["total_generations"] == 0
        assert isinstance(agent.context_encoder, GenerationContextEncoder)

    def test_strategy_initialization(self):
        """Test strategy initialization."""
        config = GeneratorAgentConfig()
        agent = GeneratorRLAgent(config)

        strategy_names = [s.name for s in agent.strategies]
        expected_strategies = [
            "step_by_step_approach",
            "concept_based_generation",
            "problem_solving_focus",
            "creative_exploration",
            "structured_reasoning",
            "adaptive_difficulty",
            "multi_perspective",
            "real_world_application",
        ]

        for expected in expected_strategies:
            assert expected in strategy_names

        # Check strategy parameters
        for strategy in agent.strategies:
            assert isinstance(strategy.parameters, dict)
            assert len(strategy.parameters) > 0

    def test_action_space(self):
        """Test action space definition."""
        config = GeneratorAgentConfig()
        agent = GeneratorRLAgent(config)

        action_space = agent.get_action_space()

        assert isinstance(action_space, ActionSpace)
        assert len(action_space) == 8
        assert "step_by_step_approach" in action_space.actions
        assert "creative_exploration" in action_space.actions

    def test_state_representation(self):
        """Test state representation encoding."""
        config = GeneratorAgentConfig()
        agent = GeneratorRLAgent(config)

        environment_state = {
            "domain": "mathematics",
            "difficulty_level": "high school",
            "topic": "algebra basics",
            "quality_requirements": {
                "accuracy_threshold": 0.9,
                "clarity_threshold": 0.8,
            },
            "target_audience": "high school student",
            "learning_objectives": ["understand variables", "solve equations"],
            "coordination_context": {
                "coordination_quality": 0.8,
                "coordination_success": True,
            },
        }

        state = agent.get_state_representation(environment_state)

        assert isinstance(state, np.ndarray)
        assert state.dtype == np.float32
        assert len(state) > 0
        assert not np.isnan(state).any()

    def test_state_representation_error_handling(self):
        """Test state representation with invalid input."""
        config = GeneratorAgentConfig()
        agent = GeneratorRLAgent(config)

        # Test with invalid environment state
        invalid_state = {"invalid_key": "invalid_value"}

        state = agent.get_state_representation(invalid_state)

        # Should return default state without crashing
        assert isinstance(state, np.ndarray)
        assert len(state) == 42  # Default size
        assert state.dtype == np.float32

    def test_reward_calculation(self):
        """Test reward calculation."""
        config = GeneratorAgentConfig()
        config.quality_weight = 0.5
        config.novelty_weight = 0.3
        config.efficiency_weight = 0.2
        config.coordination_bonus = 0.1
        config.validation_penalty = 0.2

        agent = GeneratorRLAgent(config)

        state = np.array([1.0, 2.0, 3.0])
        action = 0
        result = {
            "quality_metrics": {"overall_score": 0.8},
            "novelty_score": 0.7,
            "efficiency_metrics": {"generation_time_score": 0.9},
            "coordination_success": True,
            "validation_passed": True,
        }

        reward = agent.calculate_reward(state, action, result)

        expected_base = 0.5 * 0.8 + 0.3 * 0.7 + 0.2 * 0.9  # 0.79
        expected_with_bonus = expected_base + 0.1  # 0.89

        assert abs(reward - expected_with_bonus) < 0.01

        # Test with validation failure
        result["validation_passed"] = False
        reward_with_penalty = agent.calculate_reward(state, action, result)

        expected_with_penalty = expected_base + 0.1 - 0.2  # 0.69
        assert abs(reward_with_penalty - expected_with_penalty) < 0.01

    def test_reward_calculation_error_handling(self):
        """Test reward calculation with invalid input."""
        config = GeneratorAgentConfig()
        agent = GeneratorRLAgent(config)

        state = np.array([1.0, 2.0, 3.0])
        action = 0
        invalid_result = {}  # Missing required fields

        reward = agent.calculate_reward(state, action, invalid_result)

        # Should return 0.0 without crashing
        assert reward == 0.0

    def test_strategy_selection(self):
        """Test generation strategy selection."""
        config = GeneratorAgentConfig()
        agent = GeneratorRLAgent(config)

        environment_state = {
            "domain": "mathematics",
            "difficulty_level": "high school",
            "topic": "algebra",
            "quality_requirements": {"accuracy_threshold": 0.8},
        }

        strategy_info = agent.select_generation_strategy(environment_state)

        assert "strategy" in strategy_info
        assert "description" in strategy_info
        assert "parameters" in strategy_info
        assert "confidence" in strategy_info
        assert "performance_history" in strategy_info

        # Check that strategy is valid
        strategy_names = [s.name for s in agent.strategies]
        assert strategy_info["strategy"] in strategy_names

        # Check that history was updated
        assert len(agent.strategy_history) == 1
        assert agent.strategy_history[0]["strategy"] == strategy_info["strategy"]

    def test_strategy_selection_error_handling(self):
        """Test strategy selection with invalid input."""
        config = GeneratorAgentConfig()
        agent = GeneratorRLAgent(config)

        # Mock select_action to raise an exception
        with patch.object(agent, "select_action", side_effect=Exception("Test error")):
            with pytest.raises(AgentFailureError):
                agent.select_generation_strategy({})

    def test_performance_history_encoding(self):
        """Test performance history encoding."""
        config = GeneratorAgentConfig()
        agent = GeneratorRLAgent(config)

        # Update some metrics
        agent.generation_metrics["average_quality"] = 0.8
        agent.generation_metrics["average_novelty"] = 0.7
        agent.generation_metrics["average_efficiency"] = 0.9
        agent.generation_metrics["total_generations"] = 10
        agent.generation_metrics["successful_generations"] = 8

        # Add some strategy history
        agent.strategy_history = [
            {"strategy": "step_by_step_approach", "timestamp": time.time()},
            {"strategy": "creative_exploration", "timestamp": time.time()},
            {"strategy": "step_by_step_approach", "timestamp": time.time()},
        ]

        features = agent._encode_performance_history()

        assert len(features) > 0
        assert features[0] == 0.8  # Average quality
        assert features[1] == 0.7  # Average novelty
        assert features[2] == 0.9  # Average efficiency
        assert features[3] == 0.8  # Success rate (8/10)

    def test_generation_metrics_update(self):
        """Test generation metrics update."""
        config = GeneratorAgentConfig()
        agent = GeneratorRLAgent(config)

        # Update metrics using a real strategy name
        strategy_name = "step_by_step_approach"
        agent._update_generation_metrics(0.8, 0.7, 0.9, True, strategy_name)

        assert agent.generation_metrics["total_generations"] == 1
        assert agent.generation_metrics["successful_generations"] == 1
        assert agent.generation_metrics["average_quality"] > 0.0
        assert agent.generation_metrics["average_novelty"] > 0.0
        assert agent.generation_metrics["average_efficiency"] > 0.0
        assert agent.generation_metrics["strategy_usage"][strategy_name] == 1

        # Update with failure
        agent._update_generation_metrics(0.3, 0.2, 0.4, False, strategy_name)

        assert agent.generation_metrics["total_generations"] == 2
        assert agent.generation_metrics["successful_generations"] == 1  # Still 1
        assert agent.generation_metrics["strategy_usage"][strategy_name] == 2

    def test_performance_summary(self):
        """Test agent performance summary."""
        config = GeneratorAgentConfig()
        agent = GeneratorRLAgent(config)

        # Add some performance data
        agent.generation_metrics["total_generations"] = 5
        agent.generation_metrics["successful_generations"] = 4
        agent.strategy_history = [
            {"strategy": "test_strategy", "timestamp": time.time(), "confidence": 0.8}
        ]

        # Update strategy performance
        agent.strategies[0].update_performance(0.8, 0.7, 0.9, True)

        summary = agent.get_agent_performance_summary()

        assert summary["agent_id"] == "generator"
        assert "generation_metrics" in summary
        assert "strategy_performance" in summary
        assert "recent_strategy_history" in summary
        assert "learning_progress" in summary

        # Check strategy performance is sorted
        strategy_perf = summary["strategy_performance"]
        assert len(strategy_perf) == 8  # All strategies

        # First strategy should have the highest overall score
        if len(strategy_perf) > 1:
            assert strategy_perf[0]["overall_score"] >= strategy_perf[1]["overall_score"]

    def test_state_summarization(self):
        """Test state summarization for logging."""
        config = GeneratorAgentConfig()
        agent = GeneratorRLAgent(config)

        environment_state = {
            "domain": "mathematics",
            "difficulty_level": "high school",
            "topic": "algebra",
            "other_field": "ignored",
        }

        summary = agent._summarize_state(environment_state)

        assert "domain=mathematics" in summary
        assert "difficulty=high school" in summary
        assert "topic=algebra" in summary

    def test_strategy_history_management(self):
        """Test strategy history size management."""
        config = GeneratorAgentConfig()
        agent = GeneratorRLAgent(config)

        # Add more than 1000 entries
        for i in range(1100):
            agent.strategy_history.append(
                {
                    "strategy": f"strategy_{i}",
                    "timestamp": time.time(),
                    "confidence": 0.5,
                }
            )

        # Trigger history management by selecting a strategy
        environment_state = {"domain": "test"}
        agent.select_generation_strategy(environment_state)

        # History should be trimmed to 1000
        assert len(agent.strategy_history) == 1000
