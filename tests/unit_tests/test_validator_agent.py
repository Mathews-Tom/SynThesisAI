"""
Unit tests for Validator RL Agent

Tests the Validator RL Agent implementation including validation strategies,
quality prediction, feedback generation, and learning mechanisms.
"""

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import torch

from core.marl.agents.specialized.validator_agent import (
    ContentAnalyzer,
    FeedbackGenerator,
    ValidationStrategy,
    ValidatorRLAgent,
)
from core.marl.config import ValidatorAgentConfig
from core.marl.exceptions import AgentFailureError


class TestValidationStrategy:
    """Test ValidationStrategy class."""

    def test_initialization(self):
        """Test ValidationStrategy initialization."""
        strategy = ValidationStrategy(
            "test_strategy",
            "Test strategy description",
            {"threshold": 0.8, "strictness": 0.7},
        )

        assert strategy.name == "test_strategy"
        assert strategy.description == "Test strategy description"
        assert strategy.parameters["threshold"] == 0.8
        assert strategy.usage_count == 0
        assert strategy.accuracy_rate == 0.0

    def test_update_performance(self):
        """Test performance update mechanism."""
        strategy = ValidationStrategy("test", "desc", {})

        # Update performance
        strategy.update_performance(0.8, 0.7, 0.9, 1, 0)

        assert strategy.usage_count == 1
        assert strategy.accuracy_rate > 0
        assert strategy.feedback_quality > 0
        assert strategy.efficiency_score > 0

    def test_performance_summary(self):
        """Test performance summary generation."""
        strategy = ValidationStrategy("test", "desc", {})
        strategy.update_performance(0.8, 0.7, 0.9, 0, 1)

        summary = strategy.get_performance_summary()

        assert "name" in summary
        assert "usage_count" in summary
        assert "accuracy_rate" in summary
        assert "overall_score" in summary
        assert summary["usage_count"] == 1


class TestValidatorRLAgent:
    """Test ValidatorRLAgent class."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return ValidatorAgentConfig(
            buffer_size=1000,
            batch_size=32,
            learning_rate=0.001,
            epsilon_initial=1.0,
            epsilon_decay=0.995,
            epsilon_min=0.01,
            gamma=0.99,
            target_update_freq=100,
            hidden_layers=[64, 32],
            accuracy_weight=0.7,
            efficiency_weight=0.3,
            feedback_quality_weight=0.2,
            false_positive_penalty=0.1,
            false_negative_penalty=0.15,
            coordination_bonus=0.1,
        )

    @pytest.fixture
    def agent(self, config):
        """Create test ValidatorRLAgent."""
        with patch("core.marl.agents.specialized.validator_agent.get_marl_logger"):
            return ValidatorRLAgent(config)

    def test_initialization(self, agent):
        """Test ValidatorRLAgent initialization."""
        assert agent.agent_id == "validator"
        assert len(agent.strategies) == 8
        assert len(agent.validation_thresholds) == 10
        assert "total_validations" in agent.validation_metrics
        assert isinstance(agent.content_analyzer, ContentAnalyzer)
        assert isinstance(agent.feedback_generator, FeedbackGenerator)

    def test_strategy_initialization(self, agent):
        """Test validation strategies initialization."""
        strategy_names = [s.name for s in agent.strategies]

        expected_strategies = [
            "strict_validation_high_threshold",
            "standard_validation_medium_threshold",
            "lenient_validation_low_threshold",
            "adaptive_threshold_based_on_content",
            "domain_specific_threshold",
            "quality_focused_validation",
            "efficiency_focused_validation",
            "comprehensive_validation",
        ]

        for expected in expected_strategies:
            assert expected in strategy_names

    def test_get_action_space(self, agent):
        """Test action space definition."""
        action_space = agent.get_action_space()

        assert len(action_space) == 8
        assert "strict_validation_high_threshold" in action_space.actions
        assert "comprehensive_validation" in action_space.actions

    def test_state_representation(self, agent):
        """Test state representation encoding."""
        environment_state = {
            "content": {
                "text": "This is test content for validation",
                "type": "problem",
                "domain": "mathematics",
                "accuracy_score": 0.8,
            },
            "domain": "mathematics",
            "difficulty_level": "high_school",
            "generator_strategy": {
                "strategy": "step_by_step_approach",
                "confidence": 0.8,
            },
            "quality_requirements": {
                "accuracy_threshold": 0.8,
                "clarity_threshold": 0.7,
            },
            "coordination_context": {
                "coordination_quality": 0.7,
                "consensus_level": 0.8,
            },
        }

        state = agent.get_state_representation(environment_state)

        assert isinstance(state, np.ndarray)
        assert state.dtype == np.float32
        assert len(state) > 0
        assert not np.isnan(state).any()

    def test_state_representation_error_handling(self, agent):
        """Test state representation with invalid input."""
        # Test with empty environment state
        state = agent.get_state_representation({})

        assert isinstance(state, np.ndarray)
        assert len(state) == 53  # Default state size
        assert state.dtype == np.float32

    def test_calculate_reward(self, agent):
        """Test reward calculation."""
        state = np.random.random(50).astype(np.float32)
        action = 0
        result = {
            "validation_accuracy": 0.8,
            "feedback_quality_score": 0.7,
            "validation_time_score": 0.9,
            "false_positive_count": 1,
            "false_negative_count": 0,
            "coordination_success": True,
        }

        reward = agent.calculate_reward(state, action, result)

        assert isinstance(reward, float)
        assert -1.0 <= reward <= 2.0

    def test_calculate_reward_with_penalties(self, agent):
        """Test reward calculation with penalties."""
        state = np.random.random(50).astype(np.float32)
        action = 0
        result = {
            "validation_accuracy": 0.6,
            "feedback_quality_score": 0.5,
            "validation_time_score": 0.7,
            "false_positive_count": 3,
            "false_negative_count": 2,
            "coordination_success": False,
        }

        reward = agent.calculate_reward(state, action, result)

        assert isinstance(reward, float)
        assert reward < 0.5  # Should be low due to penalties

    def test_predict_quality_and_provide_feedback(self, agent):
        """Test quality prediction and feedback generation."""
        content = {
            "text": "Solve the equation 2x + 3 = 7. First, subtract 3 from both sides.",
            "type": "problem",
            "domain": "mathematics",
            "accuracy_score": 0.8,
            "clarity_score": 0.7,
        }

        environment_state = {
            "domain": "mathematics",
            "difficulty_level": "high_school",
            "quality_requirements": {"accuracy_threshold": 0.8},
        }

        with patch.object(agent, "select_action", return_value=0):
            result = agent.predict_quality_and_provide_feedback(
                content, environment_state
            )

        assert "quality_prediction" in result
        assert "validation_threshold" in result
        assert "passes_threshold" in result
        assert "feedback" in result
        assert "confidence" in result
        assert "strategy_used" in result

        assert isinstance(result["quality_prediction"], float)
        assert 0.0 <= result["quality_prediction"] <= 1.0
        assert isinstance(result["passes_threshold"], bool)

    def test_threshold_for_strategy(self, agent):
        """Test threshold calculation for different strategies."""
        strict_strategy = agent.strategies[0]  # strict_validation_high_threshold
        lenient_strategy = agent.strategies[2]  # lenient_validation_low_threshold

        strict_threshold = agent._get_threshold_for_strategy(strict_strategy)
        lenient_threshold = agent._get_threshold_for_strategy(lenient_strategy)

        assert strict_threshold >= 0.8
        assert lenient_threshold <= 0.6
        assert strict_threshold > lenient_threshold

    def test_validation_history_encoding(self, agent):
        """Test validation history encoding."""
        # Add some validation history
        agent.validation_history = [
            {
                "strategy": "strict_validation_high_threshold",
                "quality_prediction": 0.8,
                "confidence": 0.7,
            },
            {
                "strategy": "standard_validation_medium_threshold",
                "quality_prediction": 0.7,
                "confidence": 0.6,
            },
        ]

        features = agent._encode_validation_history()

        assert isinstance(features, list)
        assert len(features) > 0
        assert all(isinstance(f, float) for f in features)

    def test_generator_strategy_encoding(self, agent):
        """Test generator strategy encoding."""
        generator_strategy = {
            "strategy": "step_by_step_approach",
            "confidence": 0.8,
        }

        features = agent._encode_generator_strategy(generator_strategy)

        assert isinstance(features, list)
        assert len(features) == 9  # 8 strategy types + 1 confidence
        assert features[-1] == 0.8  # Confidence should be last

    def test_domain_context_encoding(self, agent):
        """Test domain context encoding."""
        math_features = agent._encode_domain_context("mathematics")
        science_features = agent._encode_domain_context("science")
        unknown_features = agent._encode_domain_context("unknown")

        assert len(math_features) == 6
        assert len(science_features) == 6
        assert len(unknown_features) == 6

        assert math_features[0] == 1.0  # Mathematics should be first
        assert science_features[1] == 1.0  # Science should be second
        assert sum(unknown_features) == 0.0  # Unknown should be all zeros

    def test_update_validation_metrics(self, agent):
        """Test validation metrics update."""
        initial_total = agent.validation_metrics["total_validations"]

        agent._update_validation_metrics(
            0.8, 0.7, 1, 0, "strict_validation_high_threshold"
        )

        assert agent.validation_metrics["total_validations"] == initial_total + 1
        assert agent.validation_metrics["average_accuracy"] > 0
        assert agent.validation_metrics["false_positives"] == 1

    def test_performance_summary(self, agent):
        """Test agent performance summary."""
        # Add some performance data
        agent.validation_metrics["total_validations"] = 10
        agent.validation_metrics["accurate_validations"] = 8

        summary = agent.get_agent_performance_summary()

        assert "agent_id" in summary
        assert "validation_metrics" in summary
        assert "strategy_performance" in summary
        assert "learning_progress" in summary

        assert summary["agent_id"] == "validator"
        assert len(summary["strategy_performance"]) == 8

    def test_error_handling_in_quality_prediction(self, agent):
        """Test error handling in quality prediction."""
        with patch.object(
            agent, "get_state_representation", side_effect=Exception("Test error")
        ):
            with pytest.raises(AgentFailureError):
                agent.predict_quality_and_provide_feedback({}, {})


class TestContentAnalyzer:
    """Test ContentAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create test ContentAnalyzer."""
        return ContentAnalyzer()

    def test_analyze_content_complexity(self, analyzer):
        """Test content complexity analysis."""
        content = {
            "text": "This is a complex mathematical problem involving advanced calculus concepts.",
            "type": "problem",
        }

        features = analyzer.analyze_content_complexity(content)

        assert isinstance(features, list)
        assert len(features) == 4
        assert all(isinstance(f, float) for f in features)
        assert all(f >= 0.0 for f in features)

    def test_analyze_content_domain(self, analyzer):
        """Test content domain analysis."""
        math_content = {
            "text": "Solve this equation using the quadratic formula and theorem.",
            "domain": "mathematics",
        }

        features = analyzer.analyze_content_domain(math_content)

        assert isinstance(features, list)
        assert len(features) == 4
        assert features[0] > 0  # Should detect math keywords
        assert features[2] == 1.0  # Mathematics domain flag

    def test_analyze_quality_indicators(self, analyzer):
        """Test quality indicators analysis."""
        content = {
            "text": "Here's an example: What is 2+2? The answer is 4.",
            "accuracy_score": 0.9,
            "clarity_score": 0.8,
            "completeness_score": 0.7,
            "engagement_score": 0.6,
        }

        features = analyzer.analyze_quality_indicators(content)

        assert isinstance(features, list)
        assert len(features) == 6
        assert features[0] == 0.9  # Accuracy score
        assert features[4] == 1.0  # Has questions
        assert features[5] == 1.0  # Has examples

    def test_predict_quality_score(self, analyzer):
        """Test quality score prediction."""
        content = {
            "text": "This is high-quality content with clear examples.",
            "accuracy_score": 0.9,
            "clarity_score": 0.8,
            "completeness_score": 0.8,
            "engagement_score": 0.7,
        }

        strategy = ValidationStrategy("test", "desc", {"strictness_level": 0.5})
        score = analyzer.predict_quality_score(content, strategy)

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_predict_quality_score_with_strict_strategy(self, analyzer):
        """Test quality score prediction with strict strategy."""
        content = {
            "text": "Basic content",
            "accuracy_score": 0.8,
            "clarity_score": 0.7,
            "completeness_score": 0.7,
            "engagement_score": 0.6,
        }

        strict_strategy = ValidationStrategy(
            "strict", "desc", {"strictness_level": 0.9}
        )
        lenient_strategy = ValidationStrategy(
            "lenient", "desc", {"strictness_level": 0.2}
        )

        strict_score = analyzer.predict_quality_score(content, strict_strategy)
        lenient_score = analyzer.predict_quality_score(content, lenient_strategy)

        assert strict_score < lenient_score


class TestFeedbackGenerator:
    """Test FeedbackGenerator class."""

    @pytest.fixture
    def generator(self):
        """Create test FeedbackGenerator."""
        return FeedbackGenerator()

    def test_initialization(self, generator):
        """Test FeedbackGenerator initialization."""
        assert hasattr(generator, "feedback_templates")
        assert "accuracy" in generator.feedback_templates
        assert "clarity" in generator.feedback_templates
        assert "completeness" in generator.feedback_templates
        assert "engagement" in generator.feedback_templates

    def test_generate_structured_feedback(self, generator):
        """Test structured feedback generation."""
        content = {
            "text": "Basic mathematical content",
            "accuracy_score": 0.6,
            "clarity_score": 0.5,
        }

        strategy = ValidationStrategy("test", "desc", {"feedback_detail": 0.8})
        environment_state = {"domain": "mathematics"}

        feedback = generator.generate_structured_feedback(
            content, 0.6, strategy, environment_state
        )

        assert "overall_score" in feedback
        assert "strategy_used" in feedback
        assert "areas_for_improvement" in feedback
        assert "strengths" in feedback
        assert "specific_suggestions" in feedback
        assert "priority_level" in feedback

        assert feedback["overall_score"] == 0.6
        assert (
            feedback["priority_level"] == "high"
        )  # Low score should trigger high priority

    def test_identify_improvement_areas(self, generator):
        """Test improvement areas identification."""
        low_quality_content = {
            "accuracy_score": 0.5,
            "clarity_score": 0.6,
            "completeness_score": 0.5,
            "engagement_score": 0.4,
        }

        areas = generator._identify_improvement_areas(low_quality_content, 0.5)

        assert "accuracy" in areas
        assert "completeness" in areas
        assert "engagement" in areas

    def test_identify_strengths(self, generator):
        """Test strengths identification."""
        good_content = {
            "text": "Here's a step-by-step example: What is the solution? Step 1: analyze the problem.",
        }

        strengths = generator._identify_strengths(good_content)

        assert len(strengths) > 0
        assert any("example" in strength.lower() for strength in strengths)
        assert any("step" in strength.lower() for strength in strengths)

    def test_generate_specific_suggestions(self, generator):
        """Test specific suggestions generation."""
        content = {
            "text": "Basic content",
            "accuracy_score": 0.5,
            "clarity_score": 0.6,
        }

        strategy = ValidationStrategy("test", "desc", {})

        suggestions = generator._generate_specific_suggestions(content, 0.5, strategy)

        assert isinstance(suggestions, list)
        # Should have suggestions for low-scoring areas

    def test_generate_domain_feedback(self, generator):
        """Test domain-specific feedback generation."""
        math_feedback = generator._generate_domain_feedback({}, "mathematics")
        science_feedback = generator._generate_domain_feedback({}, "science")
        unknown_feedback = generator._generate_domain_feedback({}, "unknown")

        assert "mathematical" in math_feedback.lower()
        assert "scientific" in science_feedback.lower()
        assert "domain-specific" in unknown_feedback.lower()


@pytest.mark.integration
class TestValidatorAgentIntegration:
    """Integration tests for ValidatorRLAgent."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return ValidatorAgentConfig(
            buffer_size=100,
            batch_size=16,
            learning_rate=0.01,
            epsilon_initial=0.5,
            epsilon_decay=0.99,
            epsilon_min=0.1,
            gamma=0.95,
            target_update_freq=50,
            hidden_layers=[32, 16],
            accuracy_weight=0.7,
            efficiency_weight=0.3,
            feedback_quality_weight=0.2,
            false_positive_penalty=0.1,
            false_negative_penalty=0.15,
            coordination_bonus=0.1,
        )

    @pytest.fixture
    def agent(self, config):
        """Create test ValidatorRLAgent."""
        with patch("core.marl.agents.specialized.validator_agent.get_marl_logger"):
            return ValidatorRLAgent(config)

    def test_full_validation_workflow(self, agent):
        """Test complete validation workflow."""
        content = {
            "text": "Solve 2x + 3 = 7. Step 1: Subtract 3 from both sides to get 2x = 4. Step 2: Divide by 2 to get x = 2.",
            "type": "solution",
            "domain": "mathematics",
            "accuracy_score": 0.9,
            "clarity_score": 0.8,
            "completeness_score": 0.8,
            "engagement_score": 0.7,
        }

        environment_state = {
            "domain": "mathematics",
            "difficulty_level": "high_school",
            "generator_strategy": {
                "strategy": "step_by_step_approach",
                "confidence": 0.8,
            },
            "quality_requirements": {
                "accuracy_threshold": 0.8,
                "clarity_threshold": 0.7,
            },
            "coordination_context": {
                "coordination_quality": 0.8,
                "consensus_level": 0.7,
                "coordination_success": True,
            },
        }

        # Predict quality and get feedback
        result = agent.predict_quality_and_provide_feedback(content, environment_state)

        # Verify result structure
        assert "quality_prediction" in result
        assert "validation_threshold" in result
        assert "passes_threshold" in result
        assert "feedback" in result
        assert "confidence" in result

        # Simulate learning update
        state = agent.get_state_representation(
            {**environment_state, "content": content}
        )
        action = 0  # Use first strategy
        reward_result = {
            "validation_accuracy": 0.9,
            "feedback_quality_score": 0.8,
            "validation_time_score": 0.7,
            "false_positive_count": 0,
            "false_negative_count": 0,
            "coordination_success": True,
        }

        reward = agent.calculate_reward(state, action, reward_result)
        assert reward > 0  # Should be positive for good performance

        # Update policy
        next_state = np.random.random(len(state)).astype(np.float32)
        agent.update_policy(state, action, reward, next_state, False)

        # Verify metrics were updated
        assert agent.validation_metrics["total_validations"] > 0

    def test_learning_progression(self, agent):
        """Test agent learning progression over multiple episodes."""
        initial_epsilon = agent.epsilon

        # Simulate multiple validation episodes
        for episode in range(10):
            content = {
                "text": f"Test content for episode {episode}",
                "accuracy_score": 0.7 + (episode * 0.02),  # Gradually improving content
            }

            environment_state = {
                "domain": "mathematics",
                "difficulty_level": "high_school",
            }

            # Get validation result
            result = agent.predict_quality_and_provide_feedback(
                content, environment_state
            )

            # Simulate learning update
            state = agent.get_state_representation(
                {**environment_state, "content": content}
            )
            action = episode % len(agent.strategies)  # Vary strategies
            reward = 0.5 + (episode * 0.05)  # Gradually improving rewards

            next_state = np.random.random(len(state)).astype(np.float32)
            agent.update_policy(state, action, reward, next_state, False)

        # Verify learning progression
        assert agent.epsilon < initial_epsilon  # Epsilon should decay
        assert agent.training_step > 0
        assert agent.validation_metrics["total_validations"] == 10

    def test_strategy_performance_tracking(self, agent):
        """Test strategy performance tracking."""
        # Use specific strategy multiple times
        strategy_index = 0
        strategy = agent.strategies[strategy_index]

        initial_usage = strategy.usage_count

        # Simulate multiple uses of the same strategy
        for _ in range(5):
            strategy.update_performance(0.8, 0.7, 0.9, 0, 1)

        assert strategy.usage_count == initial_usage + 5
        assert strategy.accuracy_rate > 0
        assert strategy.feedback_quality > 0

        # Get performance summary
        summary = strategy.get_performance_summary()
        assert summary["usage_count"] == initial_usage + 5
        assert summary["overall_score"] > 0


if __name__ == "__main__":
    pytest.main([__file__])
