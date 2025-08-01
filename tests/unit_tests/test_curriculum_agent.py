"""
Unit tests for Curriculum RL Agent

Tests the Curriculum RL Agent implementation including curriculum strategies,
learning progression, pedagogical guidance, and learning mechanisms.
"""

# Standard Library
from unittest.mock import patch

# Third-Party Library
import numpy as np
import pytest

# SynThesisAI Modules
from core.marl.agents.specialized.curriculum_agent import (
    CurriculumRLAgent,
    CurriculumStrategy,
    LearningAnalyzer,
    ObjectiveAligner,
    ProgressionModeler,
)
from core.marl.config import CurriculumAgentConfig
from core.marl.exceptions import AgentFailureError


class TestCurriculumStrategy:
    """Test CurriculumStrategy class."""

    def test_initialization(self):
        """Test CurriculumStrategy initialization."""
        strategy = CurriculumStrategy(
            "test_strategy",
            "Test strategy description",
            {"mastery_threshold": 0.8, "adaptivity": 0.7},
        )

        assert strategy.name == "test_strategy"
        assert strategy.description == "Test strategy description"
        assert strategy.parameters["mastery_threshold"] == 0.8
        assert strategy.usage_count == 0
        assert strategy.pedagogical_coherence == 0.0

    def test_update_performance(self):
        """Test performance update mechanism."""
        strategy = CurriculumStrategy("test", "desc", {})

        # Update performance
        strategy.update_performance(0.8, 0.7, 0.9, True)

        assert strategy.usage_count == 1
        assert strategy.pedagogical_coherence > 0
        assert strategy.learning_progression > 0
        assert strategy.objective_alignment > 0
        assert strategy.integration_success_rate > 0

    def test_performance_summary(self):
        """Test performance summary generation."""
        strategy = CurriculumStrategy("test", "desc", {})
        strategy.update_performance(0.8, 0.7, 0.9, True)

        summary = strategy.get_performance_summary()

        assert "name" in summary
        assert "usage_count" in summary
        assert "pedagogical_coherence" in summary
        assert "overall_score" in summary
        assert summary["usage_count"] == 1


class TestCurriculumRLAgent:
    """Test CurriculumRLAgent class."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return CurriculumAgentConfig(
            buffer_size=1000,
            batch_size=32,
            learning_rate=0.001,
            epsilon_initial=1.0,
            epsilon_decay=0.995,
            epsilon_min=0.01,
            gamma=0.99,
            target_update_freq=100,
            hidden_layers=[64, 32],
            pedagogical_coherence_weight=0.4,
            learning_progression_weight=0.4,
            objective_alignment_weight=0.2,
            coherence_threshold=0.7,
            progression_threshold=0.7,
            integration_bonus=0.15,
        )

    @pytest.fixture
    def agent(self, config):
        """Create test CurriculumRLAgent."""
        with patch("core.marl.agents.specialized.curriculum_agent.get_marl_logger"):
            return CurriculumRLAgent(config)

    def test_initialization(self, agent):
        """Test CurriculumRLAgent initialization."""
        assert agent.agent_id == "curriculum"
        assert len(agent.strategies) == 8
        assert "total_recommendations" in agent.curriculum_metrics
        assert isinstance(agent.learning_analyzer, LearningAnalyzer)
        assert isinstance(agent.progression_modeler, ProgressionModeler)
        assert isinstance(agent.objective_aligner, ObjectiveAligner)

    def test_strategy_initialization(self, agent):
        """Test curriculum strategies initialization."""
        strategy_names = [s.name for s in agent.strategies]

        expected_strategies = [
            "linear_progression",
            "spiral_curriculum",
            "mastery_based_progression",
            "adaptive_difficulty_adjustment",
            "prerequisite_reinforcement",
            "concept_scaffolding",
            "multi_modal_learning",
            "personalized_pathway",
        ]

        for expected in expected_strategies:
            assert expected in strategy_names

    def test_get_action_space(self, agent):
        """Test action space definition."""
        action_space = agent.get_action_space()

        assert len(action_space) == 8
        assert "linear_progression" in action_space.actions
        assert "personalized_pathway" in action_space.actions

    def test_state_representation(self, agent):
        """Test state representation encoding."""
        environment_state = {
            "learning_objectives": [
                "Understand basic algebra concepts",
                "Apply algebraic methods to solve problems",
            ],
            "target_audience": "high school students",
            "domain": "mathematics",
            "difficulty_level": "high_school",
            "content": {
                "text": "This lesson covers algebraic equations and their solutions.",
                "complexity_score": 0.7,
                "engagement_score": 0.6,
            },
            "generator_strategy": {
                "strategy": "step_by_step_approach",
                "confidence": 0.8,
            },
            "validator_feedback": {
                "quality_prediction": 0.8,
                "confidence": 0.7,
                "passes_threshold": True,
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
        assert len(state) == 68  # Default state size
        assert state.dtype == np.float32

    def test_calculate_reward(self, agent):
        """Test reward calculation."""
        state = np.random.random(50).astype(np.float32)
        action = 0
        result = {
            "pedagogical_coherence_score": 0.8,
            "learning_progression_score": 0.7,
            "objective_alignment_score": 0.9,
            "curriculum_integration_success": True,
            "coordination_success": True,
        }

        reward = agent.calculate_reward(state, action, result)

        assert isinstance(reward, float)
        assert -1.0 <= reward <= 2.0

    def test_calculate_reward_with_bonuses(self, agent):
        """Test reward calculation with bonuses."""
        state = np.random.random(50).astype(np.float32)
        action = 0
        result = {
            "pedagogical_coherence_score": 0.9,  # Above threshold
            "learning_progression_score": 0.8,  # Above threshold
            "objective_alignment_score": 0.8,
            "curriculum_integration_success": True,
            "coordination_success": True,
        }

        reward = agent.calculate_reward(state, action, result)

        assert isinstance(reward, float)
        assert reward > 1.0  # Should be high due to bonuses

    def test_suggest_curriculum_improvements(self, agent):
        """Test curriculum improvement suggestions."""
        request = {
            "learning_objectives": [
                "Understand quadratic equations",
                "Solve quadratic equations using various methods",
            ],
            "target_audience": "high school students",
            "domain": "mathematics",
            "difficulty_level": "high_school",
            "topic": "quadratic equations",
            "content": {
                "text": "Quadratic equations are polynomial equations of degree 2.",
                "type": "lesson",
            },
        }

        with patch.object(agent, "select_action", return_value=0):
            result = agent.suggest_curriculum_improvements(request)

        assert "curriculum_strategy" in result
        assert "strategy_description" in result
        assert "difficulty_adjustments" in result
        assert "prerequisite_recommendations" in result
        assert "learning_pathway" in result
        assert "pedagogical_hints" in result
        assert "progression_guidance" in result
        assert "objective_alignment" in result
        assert "confidence" in result

        assert isinstance(result["prerequisite_recommendations"], list)
        assert isinstance(result["pedagogical_hints"], list)

    def test_difficulty_adjustments(self, agent):
        """Test difficulty adjustment suggestions."""
        # Test with beginner audience and high difficulty
        request = {
            "difficulty_level": "advanced",
            "target_audience": "beginner students",
            "learning_objectives": ["Complex objective with many components"],
        }

        adjustments = agent._suggest_difficulty_adjustments(request)

        assert adjustments["current_difficulty"] == "advanced"
        assert adjustments["recommended_difficulty"] == "medium"
        assert adjustments["scaffolding_needed"]
        assert "beginner" in adjustments["adjustment_rationale"]

    def test_prerequisite_identification(self, agent):
        """Test prerequisite identification."""
        request = {
            "domain": "mathematics",
            "topic": "calculus derivatives",
            "difficulty_level": "undergraduate",
        }

        prerequisites = agent._identify_prerequisites(request)

        assert isinstance(prerequisites, list)
        assert len(prerequisites) > 0

        # Should identify algebra as prerequisite for calculus
        algebra_found = any(
            "algebra" in prereq["concept"].lower() for prereq in prerequisites
        )
        assert algebra_found

    def test_learning_pathway_generation(self, agent):
        """Test learning pathway generation."""
        request = {
            "learning_objectives": [
                "Understand basic concepts",
                "Apply concepts to problems",
                "Analyze complex scenarios",
            ],
            "domain": "mathematics",
            "difficulty_level": "high_school",
        }

        pathway = agent._generate_learning_pathway(request)

        assert "pathway_type" in pathway
        assert "total_steps" in pathway
        assert "steps" in pathway
        assert "milestones" in pathway

        assert len(pathway["steps"]) == 3  # One per objective
        assert len(pathway["milestones"]) > 0

        # Check step structure
        for step in pathway["steps"]:
            assert "step_number" in step
            assert "objective" in step
            assert "activities" in step
            assert "resources" in step
            assert "assessment" in step

    def test_pedagogical_hints_generation(self, agent):
        """Test pedagogical hints generation."""
        request = {
            "domain": "mathematics",
            "target_audience": "beginner students",
            "learning_objectives": ["Analyze mathematical patterns"],
        }

        hints = agent._generate_pedagogical_hints(request)

        assert isinstance(hints, list)
        assert len(hints) > 0

        # Should include domain-specific and audience-specific hints
        math_hint_found = any(
            "mathematical" in hint.lower() or "visual" in hint.lower() for hint in hints
        )
        beginner_hint_found = any(
            "concrete" in hint.lower() or "simple" in hint.lower() for hint in hints
        )

        assert math_hint_found or beginner_hint_found

    def test_objective_alignment_assessment(self, agent):
        """Test objective alignment assessment."""
        request = {
            "learning_objectives": [
                "Understand quadratic equations",
                "Solve quadratic problems",
            ],
            "content": {
                "text": "Quadratic equations are equations with degree 2. To solve quadratic problems, we use various methods.",
            },
        }

        alignment = agent._assess_objective_alignment(request)

        assert "overall_alignment" in alignment
        assert "objective_coverage" in alignment
        assert "gaps_identified" in alignment
        assert "recommendations" in alignment

        assert isinstance(alignment["overall_alignment"], float)
        assert 0.0 <= alignment["overall_alignment"] <= 1.0

    def test_objective_coverage_calculation(self, agent):
        """Test objective coverage calculation."""
        objective = "Understand quadratic equations and their properties"
        content = {
            "text": "Quadratic equations are polynomial equations with degree 2. They have specific properties that make them unique.",
        }

        coverage = agent._calculate_objective_coverage(objective, content)

        assert isinstance(coverage, float)
        assert 0.0 <= coverage <= 1.0
        assert coverage > 0.5  # Should have good coverage due to matching terms

    def test_learning_activities_suggestions(self, agent):
        """Test learning activities suggestions."""
        # Test different objective types
        understand_objective = "Understand basic concepts"
        apply_objective = "Apply methods to solve problems"
        create_objective = "Create original solutions"

        understand_activities = agent._suggest_learning_activities(
            understand_objective, "mathematics"
        )
        apply_activities = agent._suggest_learning_activities(
            apply_objective, "science"
        )
        create_activities = agent._suggest_learning_activities(create_objective, "arts")

        assert isinstance(understand_activities, list)
        assert isinstance(apply_activities, list)
        assert isinstance(create_activities, list)

        # Should suggest appropriate activities for each objective type
        assert any(
            "reading" in activity.lower() or "concept" in activity.lower()
            for activity in understand_activities
        )
        assert any(
            "practice" in activity.lower() or "problem" in activity.lower()
            for activity in apply_activities
        )
        assert any(
            "project" in activity.lower() or "creative" in activity.lower()
            for activity in create_activities
        )

    def test_assessment_method_suggestions(self, agent):
        """Test assessment method suggestions."""
        understand_method = agent._suggest_assessment_method(
            "Understand basic concepts"
        )
        apply_method = agent._suggest_assessment_method(
            "Apply methods to solve problems"
        )
        analyze_method = agent._suggest_assessment_method("Analyze complex scenarios")
        create_method = agent._suggest_assessment_method("Create original designs")

        assert (
            "quiz" in understand_method.lower()
            or "explanation" in understand_method.lower()
        )
        assert "problem" in apply_method.lower() or "assignment" in apply_method.lower()
        assert (
            "analysis" in analyze_method.lower()
            or "evaluation" in analyze_method.lower()
        )
        assert (
            "project" in create_method.lower() or "portfolio" in create_method.lower()
        )

    def test_curriculum_metrics_update(self, agent):
        """Test curriculum metrics update."""
        initial_total = agent.curriculum_metrics["total_recommendations"]

        agent._update_curriculum_metrics(0.8, 0.7, 0.9, True, "linear_progression")

        assert agent.curriculum_metrics["total_recommendations"] == initial_total + 1
        assert agent.curriculum_metrics["successful_integrations"] == 1
        assert agent.curriculum_metrics["average_pedagogical_coherence"] > 0
        assert agent.curriculum_metrics["average_learning_progression"] > 0
        assert agent.curriculum_metrics["average_objective_alignment"] > 0

    def test_performance_summary(self, agent):
        """Test agent performance summary."""
        # Add some performance data
        agent.curriculum_metrics["total_recommendations"] = 10
        agent.curriculum_metrics["successful_integrations"] = 8

        summary = agent.get_agent_performance_summary()

        assert "agent_id" in summary
        assert "curriculum_metrics" in summary
        assert "strategy_performance" in summary
        assert "learning_progress" in summary

        assert summary["agent_id"] == "curriculum"
        assert len(summary["strategy_performance"]) == 8

    def test_error_handling_in_curriculum_suggestions(self, agent):
        """Test error handling in curriculum suggestions."""
        with patch.object(
            agent, "get_state_representation", side_effect=Exception("Test error")
        ):
            with pytest.raises(AgentFailureError):
                agent.suggest_curriculum_improvements({})


class TestLearningAnalyzer:
    """Test LearningAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create test LearningAnalyzer."""
        return LearningAnalyzer()

    def test_encode_learning_objectives(self, analyzer):
        """Test learning objectives encoding."""
        objectives = [
            "Understand basic algebra concepts",
            "Apply algebraic methods to solve problems",
            "Analyze complex algebraic expressions",
            "Create original algebraic solutions",
        ]

        features = analyzer.encode_learning_objectives(objectives)

        assert isinstance(features, list)
        assert len(features) == 8  # 2 basic features + 6 Bloom's taxonomy levels
        assert all(isinstance(f, float) for f in features)

        # Should detect different Bloom's taxonomy levels
        assert features[2] == 0  # Remember level (no remember keywords in objectives)
        assert features[3] > 0  # Understand level
        assert features[4] > 0  # Apply level
        assert features[5] > 0  # Analyze level
        assert features[6] == 0  # Evaluate level (no evaluate keywords in objectives)
        assert features[7] > 0  # Create level

    def test_encode_learning_objectives_empty(self, analyzer):
        """Test learning objectives encoding with empty list."""
        features = analyzer.encode_learning_objectives([])

        assert isinstance(features, list)
        assert len(features) == 8
        assert all(f == 0.0 for f in features)

    def test_encode_target_audience(self, analyzer):
        """Test target audience encoding."""
        student_features = analyzer.encode_target_audience("high school students")
        teacher_features = analyzer.encode_target_audience("professional teachers")
        beginner_features = analyzer.encode_target_audience("beginner learners")

        assert len(student_features) == 6
        assert len(teacher_features) == 6
        assert len(beginner_features) == 6

        assert student_features[0] == 1.0  # Student flag
        assert teacher_features[1] == 1.0  # Teacher flag
        assert beginner_features[2] == 1.0  # Beginner flag

    def test_analyze_content_characteristics(self, analyzer):
        """Test content characteristics analysis."""
        content = {
            "text": "This is an example lesson with step-by-step instructions. What do you think?",
            "complexity_score": 0.7,
            "engagement_score": 0.8,
        }

        features = analyzer.analyze_content_characteristics(content)

        assert isinstance(features, list)
        assert len(features) == 6
        assert all(isinstance(f, float) for f in features)

        # Should detect examples and questions
        assert features[2] == 1.0  # Has examples
        assert features[3] == 1.0  # Has steps
        assert features[4] == 0.7  # Complexity score
        assert features[5] == 0.8  # Engagement score


class TestProgressionModeler:
    """Test ProgressionModeler class."""

    @pytest.fixture
    def modeler(self):
        """Create test ProgressionModeler."""
        return ProgressionModeler()

    def test_model_difficulty_progression(self, modeler):
        """Test difficulty progression modeling."""
        # Test progression from elementary to graduate
        progression = modeler.model_difficulty_progression("elementary", "graduate")

        assert "progression_steps" in progression
        assert "intermediate_levels" in progression
        assert "estimated_duration" in progression

        assert progression["progression_steps"] == 3
        assert "high_school" in progression["intermediate_levels"]
        assert "undergraduate" in progression["intermediate_levels"]
        assert progression["estimated_duration"] == "long"

    def test_model_difficulty_progression_same_level(self, modeler):
        """Test difficulty progression modeling for same level."""
        progression = modeler.model_difficulty_progression("high_school", "high_school")

        assert progression["progression_steps"] == 0
        assert len(progression["intermediate_levels"]) == 0
        assert progression["estimated_duration"] == "short"

    def test_model_difficulty_progression_invalid_levels(self, modeler):
        """Test difficulty progression modeling with invalid levels."""
        progression = modeler.model_difficulty_progression("invalid", "unknown")

        assert progression["progression_steps"] == 1
        assert len(progression["intermediate_levels"]) == 0
        assert progression["estimated_duration"] == "medium"


class TestObjectiveAligner:
    """Test ObjectiveAligner class."""

    @pytest.fixture
    def aligner(self):
        """Create test ObjectiveAligner."""
        return ObjectiveAligner()

    def test_assess_alignment(self, aligner):
        """Test objective alignment assessment."""
        content = {
            "text": "This lesson covers quadratic equations and their solutions using various mathematical methods.",
        }

        objectives = [
            "Understand quadratic equations",
            "Learn solution methods for equations",
        ]

        alignment = aligner.assess_alignment(content, objectives)

        assert isinstance(alignment, float)
        assert 0.0 <= alignment <= 1.0
        assert alignment > 0.5  # Should have good alignment

    def test_assess_alignment_no_objectives(self, aligner):
        """Test objective alignment assessment with no objectives."""
        content = {"text": "Some content"}
        alignment = aligner.assess_alignment(content, [])

        assert alignment == 0.5  # Default moderate alignment

    def test_assess_alignment_poor_match(self, aligner):
        """Test objective alignment assessment with poor content match."""
        content = {
            "text": "This lesson is about biology and cellular structures.",
        }

        objectives = [
            "Understand quadratic equations",
            "Solve mathematical problems",
        ]

        alignment = aligner.assess_alignment(content, objectives)

        assert isinstance(alignment, float)
        assert alignment < 0.3  # Should have poor alignment


@pytest.mark.integration
class TestCurriculumAgentIntegration:
    """Integration tests for CurriculumRLAgent."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return CurriculumAgentConfig(
            buffer_size=100,
            batch_size=16,
            learning_rate=0.01,
            epsilon_initial=0.5,
            epsilon_decay=0.99,
            epsilon_min=0.1,
            gamma=0.95,
            target_update_freq=50,
            hidden_layers=[32, 16],
            pedagogical_coherence_weight=0.4,
            learning_progression_weight=0.4,
            objective_alignment_weight=0.2,
            coherence_threshold=0.7,
            progression_threshold=0.7,
            integration_bonus=0.15,
        )

    @pytest.fixture
    def agent(self, config):
        """Create test CurriculumRLAgent."""
        with patch("core.marl.agents.specialized.curriculum_agent.get_marl_logger"):
            return CurriculumRLAgent(config)

    def test_full_curriculum_workflow(self, agent):
        """Test complete curriculum workflow."""
        request = {
            "learning_objectives": [
                "Understand linear equations",
                "Solve linear equations using various methods",
                "Apply linear equations to real-world problems",
            ],
            "target_audience": "high school students",
            "domain": "mathematics",
            "difficulty_level": "high_school",
            "topic": "linear equations",
            "content": {
                "text": "Linear equations are first-degree polynomial equations. They can be solved using algebraic methods.",
                "complexity_score": 0.6,
                "engagement_score": 0.7,
            },
            "generator_strategy": {
                "strategy": "step_by_step_approach",
                "confidence": 0.8,
            },
            "validator_feedback": {
                "quality_prediction": 0.8,
                "confidence": 0.7,
                "passes_threshold": True,
                "areas_for_improvement": ["clarity"],
            },
            "coordination_context": {
                "coordination_quality": 0.8,
                "consensus_level": 0.7,
                "coordination_success": True,
                "agent_agreement": 0.8,
            },
        }

        # Get curriculum improvements
        result = agent.suggest_curriculum_improvements(request)

        # Verify result structure
        assert "curriculum_strategy" in result
        assert "difficulty_adjustments" in result
        assert "prerequisite_recommendations" in result
        assert "learning_pathway" in result
        assert "pedagogical_hints" in result
        assert "confidence" in result

        # Verify learning pathway structure
        pathway = result["learning_pathway"]
        assert len(pathway["steps"]) == 3  # One per objective
        assert len(pathway["milestones"]) > 0

        # Simulate learning update
        state = agent.get_state_representation(request)
        action = 0  # Use first strategy
        reward_result = {
            "pedagogical_coherence_score": 0.8,
            "learning_progression_score": 0.7,
            "objective_alignment_score": 0.9,
            "curriculum_integration_success": True,
            "coordination_success": True,
        }

        reward = agent.calculate_reward(state, action, reward_result)
        assert reward > 0  # Should be positive for good performance

        # Update policy
        next_state = np.random.random(len(state)).astype(np.float32)
        agent.update_policy(state, action, reward, next_state, False)

        # Verify metrics were updated
        assert agent.curriculum_metrics["total_recommendations"] > 0

    def test_learning_progression(self, agent):
        """Test agent learning progression over multiple episodes."""
        initial_epsilon = agent.epsilon

        # Simulate multiple curriculum episodes
        for episode in range(10):
            request = {
                "learning_objectives": [f"Objective {episode + 1}"],
                "target_audience": "students",
                "domain": "mathematics",
                "difficulty_level": "high_school",
            }

            # Get curriculum suggestions
            agent.suggest_curriculum_improvements(request)

            # Simulate learning update
            state = agent.get_state_representation(request)
            action = episode % len(agent.strategies)  # Vary strategies
            reward = 0.5 + (episode * 0.05)  # Gradually improving rewards

            next_state = np.random.random(len(state)).astype(np.float32)
            agent.update_policy(state, action, reward, next_state, False)

        # Verify learning progression
        assert agent.epsilon < initial_epsilon  # Epsilon should decay
        assert agent.training_step > 0
        assert agent.curriculum_metrics["total_recommendations"] == 10

    def test_strategy_performance_tracking(self, agent):
        """Test strategy performance tracking."""
        # Use specific strategy multiple times
        strategy_index = 0
        strategy = agent.strategies[strategy_index]

        initial_usage = strategy.usage_count

        # Simulate multiple uses of the same strategy
        for _ in range(5):
            strategy.update_performance(0.8, 0.7, 0.9, True)

        assert strategy.usage_count == initial_usage + 5
        assert strategy.pedagogical_coherence > 0
        assert strategy.learning_progression > 0
        assert strategy.integration_success_rate > 0

        # Get performance summary
        summary = strategy.get_performance_summary()
        assert summary["usage_count"] == initial_usage + 5
        assert summary["overall_score"] > 0
