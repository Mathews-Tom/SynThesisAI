"""
Validator RL Agent

This module implements the Validator RL Agent for multi-agent reinforcement learning
coordination. The Validator Agent is responsible for predicting content quality,
providing structured feedback, and learning optimal validation strategies.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from ...config import ValidatorAgentConfig
from ...exceptions import AgentFailureError
from ...logging_config import get_marl_logger
from ..base_agent import ActionSpace, BaseRLAgent
from ..experience import Experience

logger = logging.getLogger(__name__)


class ValidationStrategy:
    """Represents a content validation strategy with parameters."""

    def __init__(self, name: str, description: str, parameters: Dict[str, Any]):
        """
        Initialize validation strategy.

        Args:
            name: Strategy name
            description: Strategy description
            parameters: Strategy-specific parameters
        """
        self.name = name
        self.description = description
        self.parameters = parameters
        self.usage_count = 0
        self.accuracy_rate = 0.0
        self.feedback_quality = 0.0
        self.efficiency_score = 0.0
        self.false_positive_rate = 0.0
        self.false_negative_rate = 0.0

    def update_performance(
        self,
        accuracy: float,
        feedback_quality: float,
        efficiency: float,
        false_positives: int,
        false_negatives: int,
    ) -> None:
        """Update strategy performance metrics."""
        self.usage_count += 1

        # Update running averages
        alpha = 0.1  # Learning rate for running average
        self.accuracy_rate = (1 - alpha) * self.accuracy_rate + alpha * accuracy
        self.feedback_quality = (
            1 - alpha
        ) * self.feedback_quality + alpha * feedback_quality
        self.efficiency_score = (1 - alpha) * self.efficiency_score + alpha * efficiency

        # Update error rates
        total_validations = max(self.usage_count, 1)
        self.false_positive_rate = (1 - alpha) * self.false_positive_rate + alpha * (
            false_positives / total_validations
        )
        self.false_negative_rate = (1 - alpha) * self.false_negative_rate + alpha * (
            false_negatives / total_validations
        )

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for this strategy."""
        return {
            "name": self.name,
            "usage_count": self.usage_count,
            "accuracy_rate": self.accuracy_rate,
            "feedback_quality": self.feedback_quality,
            "efficiency_score": self.efficiency_score,
            "false_positive_rate": self.false_positive_rate,
            "false_negative_rate": self.false_negative_rate,
            "overall_score": (
                self.accuracy_rate + self.feedback_quality + self.efficiency_score
            )
            / 3,
        }


class ValidatorRLAgent(BaseRLAgent):
    """
    Validator RL Agent for content quality prediction and feedback generation.

    This agent learns to predict content quality, set appropriate validation thresholds,
    and provide structured feedback to improve content generation through reinforcement learning.
    """

    def __init__(self, config: ValidatorAgentConfig):
        """
        Initialize Validator RL Agent.

        Args:
            config: Validator agent configuration
        """
        # Initialize strategies first (needed for action space)
        self.strategies = self._initialize_strategies()

        super().__init__("validator", config)
        self.config = config
        self.logger = get_marl_logger("validator_agent")

        # Additional initialization
        self.validation_history = []

        # Validation thresholds for different strategies
        self.validation_thresholds = np.linspace(0.5, 0.95, 10)

        # Performance tracking
        self.validation_metrics = {
            "total_validations": 0,
            "accurate_validations": 0,
            "average_accuracy": 0.0,
            "average_feedback_quality": 0.0,
            "false_positives": 0,
            "false_negatives": 0,
            "strategy_usage": {strategy.name: 0 for strategy in self.strategies},
        }

        # Content analysis components
        self.content_analyzer = ContentAnalyzer()
        self.feedback_generator = FeedbackGenerator()

        self.logger.log_agent_action(
            self.agent_id,
            "validator_initialized",
            1.0,
            f"Strategies: {len(self.strategies)}, Thresholds: {len(self.validation_thresholds)}",
        )

    def _initialize_strategies(self) -> List[ValidationStrategy]:
        """Initialize available validation strategies."""
        strategies = [
            ValidationStrategy(
                "strict_validation_high_threshold",
                "Apply strict validation with high quality threshold",
                {
                    "threshold": 0.85,
                    "strictness_level": 0.9,
                    "feedback_detail": 0.8,
                    "error_tolerance": 0.1,
                },
            ),
            ValidationStrategy(
                "standard_validation_medium_threshold",
                "Apply standard validation with medium quality threshold",
                {
                    "threshold": 0.7,
                    "strictness_level": 0.6,
                    "feedback_detail": 0.6,
                    "error_tolerance": 0.2,
                },
            ),
            ValidationStrategy(
                "lenient_validation_low_threshold",
                "Apply lenient validation with low quality threshold",
                {
                    "threshold": 0.55,
                    "strictness_level": 0.3,
                    "feedback_detail": 0.4,
                    "error_tolerance": 0.3,
                },
            ),
            ValidationStrategy(
                "adaptive_threshold_based_on_content",
                "Adapt validation threshold based on content characteristics",
                {
                    "adaptivity": 0.8,
                    "content_sensitivity": 0.9,
                    "dynamic_adjustment": 0.7,
                    "context_awareness": 0.8,
                },
            ),
            ValidationStrategy(
                "domain_specific_threshold",
                "Use domain-specific validation criteria and thresholds",
                {
                    "domain_specialization": 0.9,
                    "criteria_specificity": 0.8,
                    "domain_knowledge": 0.8,
                    "specialized_feedback": 0.7,
                },
            ),
            ValidationStrategy(
                "quality_focused_validation",
                "Focus primarily on content quality metrics",
                {
                    "quality_weight": 0.9,
                    "accuracy_emphasis": 0.8,
                    "completeness_check": 0.8,
                    "clarity_assessment": 0.7,
                },
            ),
            ValidationStrategy(
                "efficiency_focused_validation",
                "Balance quality with validation efficiency",
                {
                    "efficiency_weight": 0.8,
                    "speed_optimization": 0.7,
                    "resource_awareness": 0.8,
                    "quick_assessment": 0.6,
                },
            ),
            ValidationStrategy(
                "comprehensive_validation",
                "Perform comprehensive multi-dimensional validation",
                {
                    "comprehensiveness": 0.9,
                    "multi_criteria": 0.8,
                    "detailed_analysis": 0.9,
                    "holistic_assessment": 0.8,
                },
            ),
        ]

        return strategies

    def get_state_representation(self, environment_state: Dict[str, Any]) -> np.ndarray:
        """
        Convert environment state to validator-specific representation.

        Args:
            environment_state: Raw environment state

        Returns:
            Numpy array representing the state for the validator agent
        """
        try:
            features = []

            # Content features
            content = environment_state.get("content", {})
            features.extend(self.content_analyzer.analyze_content_complexity(content))
            features.extend(self.content_analyzer.analyze_content_domain(content))
            features.extend(self.content_analyzer.analyze_quality_indicators(content))

            # Validation history features
            features.extend(self._encode_validation_history())

            # Generator strategy features
            generator_strategy = environment_state.get("generator_strategy", {})
            features.extend(self._encode_generator_strategy(generator_strategy))

            # Context features
            domain = environment_state.get("domain", "")
            features.extend(self._encode_domain_context(domain))

            difficulty_level = environment_state.get("difficulty_level", "")
            features.extend(self._encode_difficulty_context(difficulty_level))

            # Quality requirements
            quality_requirements = environment_state.get("quality_requirements", {})
            features.extend(self._encode_quality_requirements(quality_requirements))

            # Coordination context
            coordination_context = environment_state.get("coordination_context", {})
            features.extend(self._encode_coordination_context(coordination_context))

            return np.array(features, dtype=np.float32)

        except Exception as e:
            error_msg = f"Failed to encode state for validator agent"
            self.logger.log_error_with_context(
                e,
                {
                    "environment_state_keys": list(environment_state.keys()),
                    "agent_id": self.agent_id,
                },
            )
            # Return default state representation
            return np.zeros(80, dtype=np.float32)

    def get_action_space(self) -> ActionSpace:
        """
        Define validation threshold and feedback action space.

        Returns:
            ActionSpace defining available validation strategies
        """
        strategy_names = [strategy.name for strategy in self.strategies]
        return ActionSpace(strategy_names)

    def calculate_reward(
        self, state: np.ndarray, action: int, result: Dict[str, Any]
    ) -> float:
        """
        Calculate validator-specific reward for the given state-action-result.

        Args:
            state: State representation
            action: Action taken (strategy index)
            result: Result of the validation

        Returns:
            Reward value for the validator agent
        """
        try:
            # Extract performance metrics from result
            validation_accuracy = result.get("validation_accuracy", 0.0)
            feedback_quality = result.get("feedback_quality_score", 0.0)
            validation_efficiency = result.get("validation_time_score", 0.0)

            # Base reward from accuracy and efficiency
            base_reward = (
                self.config.accuracy_weight * validation_accuracy
                + self.config.efficiency_weight * validation_efficiency
            )

            # Bonus for high-quality feedback
            feedback_bonus = self.config.feedback_quality_weight * feedback_quality
            base_reward += feedback_bonus

            # Penalties for false positives/negatives
            false_positive_count = result.get("false_positive_count", 0)
            false_negative_count = result.get("false_negative_count", 0)

            false_positive_penalty = (
                false_positive_count * self.config.false_positive_penalty
            )
            false_negative_penalty = (
                false_negative_count * self.config.false_negative_penalty
            )

            base_reward -= false_positive_penalty + false_negative_penalty

            # Bonus for coordination success
            coordination_success = result.get("coordination_success", False)
            if coordination_success:
                base_reward += self.config.coordination_bonus

            # Update strategy performance
            strategy = self.strategies[action]
            strategy.update_performance(
                validation_accuracy,
                feedback_quality,
                validation_efficiency,
                false_positive_count,
                false_negative_count,
            )

            # Update agent metrics
            self._update_validation_metrics(
                validation_accuracy,
                feedback_quality,
                false_positive_count,
                false_negative_count,
                strategy.name,
            )

            return float(
                np.clip(base_reward, -1.0, 2.0)
            )  # Clip reward to reasonable range

        except Exception as e:
            error_msg = f"Failed to calculate reward for validator agent"
            self.logger.log_error_with_context(
                e,
                {
                    "action": action,
                    "result_keys": list(result.keys()),
                    "agent_id": self.agent_id,
                },
            )
            return 0.0

    def predict_quality_and_provide_feedback(
        self, content: Dict[str, Any], environment_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Predict content quality and provide structured feedback.

        Args:
            content: Content to validate
            environment_state: Current environment state

        Returns:
            Dictionary with quality prediction and feedback
        """
        try:
            # Get state representation
            state = self.get_state_representation(
                {**environment_state, "content": content}
            )

            # Select validation strategy using RL policy
            action_index = self.select_action(state, training=True)
            selected_strategy = self.strategies[action_index]

            # Get validation threshold for this strategy
            threshold = self._get_threshold_for_strategy(selected_strategy)

            # Predict quality score using content analysis
            quality_prediction = self.content_analyzer.predict_quality_score(
                content, selected_strategy
            )

            # Generate structured feedback
            feedback = self.feedback_generator.generate_structured_feedback(
                content, quality_prediction, selected_strategy, environment_state
            )

            # Calculate confidence
            confidence = self.get_action_confidence(state, action_index)

            # Record validation
            validation_record = {
                "strategy": selected_strategy.name,
                "timestamp": time.time(),
                "quality_prediction": quality_prediction,
                "threshold": threshold,
                "confidence": confidence,
                "content_summary": self._summarize_content(content),
            }

            self.validation_history.append(validation_record)

            # Update metrics
            self.validation_metrics["total_validations"] += 1

            # Keep history manageable
            if len(self.validation_history) > 1000:
                self.validation_history = self.validation_history[-1000:]

            self.logger.log_agent_action(
                self.agent_id,
                selected_strategy.name,
                confidence,
                f"Quality: {quality_prediction:.3f}, Threshold: {threshold:.3f}",
            )

            return {
                "quality_prediction": quality_prediction,
                "validation_threshold": threshold,
                "passes_threshold": quality_prediction >= threshold,
                "feedback": feedback,
                "confidence": confidence,
                "strategy_used": selected_strategy.name,
                "performance_history": selected_strategy.get_performance_summary(),
            }

        except Exception as e:
            error_msg = f"Failed to predict quality and provide feedback"
            self.logger.log_error_with_context(
                e,
                {
                    "content_keys": list(content.keys()) if content else [],
                    "agent_id": self.agent_id,
                },
            )
            raise AgentFailureError(
                error_msg, agent_id=self.agent_id, failure_type="quality_prediction"
            ) from e

    def _get_threshold_for_strategy(self, strategy: ValidationStrategy) -> float:
        """Get validation threshold for the given strategy."""
        base_threshold = strategy.parameters.get("threshold", 0.7)

        # Adjust threshold based on strategy characteristics
        if "strict" in strategy.name:
            return max(base_threshold, 0.8)
        elif "lenient" in strategy.name:
            return min(base_threshold, 0.6)
        elif "adaptive" in strategy.name:
            # Use recent performance to adapt threshold
            recent_accuracy = strategy.accuracy_rate
            if recent_accuracy > 0.8:
                return base_threshold + 0.05
            elif recent_accuracy < 0.6:
                return base_threshold - 0.05

        return base_threshold

    def _encode_validation_history(self) -> List[float]:
        """Encode validation history into features."""
        features = []

        # Overall validation metrics
        features.append(self.validation_metrics["average_accuracy"])
        features.append(self.validation_metrics["average_feedback_quality"])

        # Error rates
        total_validations = max(self.validation_metrics["total_validations"], 1)
        false_positive_rate = (
            self.validation_metrics["false_positives"] / total_validations
        )
        false_negative_rate = (
            self.validation_metrics["false_negatives"] / total_validations
        )

        features.append(false_positive_rate)
        features.append(false_negative_rate)

        # Recent validation performance (last 10 validations)
        recent_validations = (
            self.validation_history[-10:] if self.validation_history else []
        )
        strategy_performance = np.zeros(len(self.strategies))

        for record in recent_validations:
            strategy_name = record["strategy"]
            for i, strategy in enumerate(self.strategies):
                if strategy.name == strategy_name:
                    strategy_performance[i] += 1
                    break

        # Normalize by number of recent validations
        if recent_validations:
            strategy_performance /= len(recent_validations)

        features.extend(strategy_performance.tolist())

        return features

    def _encode_generator_strategy(
        self, generator_strategy: Dict[str, Any]
    ) -> List[float]:
        """Encode generator strategy information into features."""
        features = []

        strategy_name = generator_strategy.get("strategy", "")
        confidence = generator_strategy.get("confidence", 0.0)

        # Strategy type encoding
        strategy_types = [
            "step_by_step",
            "concept_based",
            "problem_solving",
            "creative",
            "structured",
            "adaptive",
            "multi_perspective",
            "real_world",
        ]

        strategy_encoding = [
            1.0 if stype in strategy_name else 0.0 for stype in strategy_types
        ]
        features.extend(strategy_encoding)

        # Strategy confidence
        features.append(confidence)

        return features

    def _encode_domain_context(self, domain: str) -> List[float]:
        """Encode domain context into features."""
        domain_mapping = {
            "mathematics": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "science": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            "technology": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            "reading": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            "engineering": [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            "arts": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        }

        domain_lower = domain.lower()
        return domain_mapping.get(domain_lower, [0.0] * 6)

    def _encode_difficulty_context(self, difficulty: str) -> List[float]:
        """Encode difficulty level into features."""
        difficulty_mapping = {
            "elementary": [1.0, 0.0, 0.0, 0.0],
            "high_school": [0.0, 1.0, 0.0, 0.0],
            "undergraduate": [0.0, 0.0, 1.0, 0.0],
            "graduate": [0.0, 0.0, 0.0, 1.0],
        }

        difficulty_lower = difficulty.lower().replace(" ", "_")
        return difficulty_mapping.get(difficulty_lower, [0.0] * 4)

    def _encode_quality_requirements(self, requirements: Dict[str, Any]) -> List[float]:
        """Encode quality requirements into features."""
        features = [
            requirements.get("accuracy_threshold", 0.8),
            requirements.get("clarity_threshold", 0.7),
            requirements.get("completeness_threshold", 0.8),
            requirements.get("engagement_threshold", 0.6),
        ]
        return features

    def _encode_coordination_context(self, context: Dict[str, Any]) -> List[float]:
        """Encode coordination context into features."""
        features = [
            context.get("coordination_quality", 0.5),
            context.get("consensus_level", 0.5),
            1.0 if context.get("coordination_success", False) else 0.0,
            context.get("agent_agreement", 0.5),
        ]
        return features

    def _update_validation_metrics(
        self,
        accuracy: float,
        feedback_quality: float,
        false_positives: int,
        false_negatives: int,
        strategy_name: str,
    ) -> None:
        """Update validation performance metrics."""
        self.validation_metrics["total_validations"] += 1

        if accuracy > 0.7:  # Consider validation accurate if above threshold
            self.validation_metrics["accurate_validations"] += 1

        # Update running averages
        alpha = 0.1
        self.validation_metrics["average_accuracy"] = (
            1 - alpha
        ) * self.validation_metrics["average_accuracy"] + alpha * accuracy

        self.validation_metrics["average_feedback_quality"] = (
            1 - alpha
        ) * self.validation_metrics[
            "average_feedback_quality"
        ] + alpha * feedback_quality

        # Update error counts
        self.validation_metrics["false_positives"] += false_positives
        self.validation_metrics["false_negatives"] += false_negatives

        # Update strategy usage
        self.validation_metrics["strategy_usage"][strategy_name] += 1

    def _summarize_content(self, content: Dict[str, Any]) -> str:
        """Create a summary of the content for logging."""
        content_type = content.get("type", "unknown")
        content_length = len(str(content.get("text", "")))
        domain = content.get("domain", "unknown")

        return f"type={content_type}, length={content_length}, domain={domain}"

    def get_agent_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary for the validator agent."""
        strategy_summaries = [
            strategy.get_performance_summary() for strategy in self.strategies
        ]

        # Sort strategies by overall performance
        strategy_summaries.sort(key=lambda x: x["overall_score"], reverse=True)

        return {
            "agent_id": self.agent_id,
            "validation_metrics": self.validation_metrics.copy(),
            "strategy_performance": strategy_summaries,
            "recent_validation_history": self.validation_history[-20:]
            if self.validation_history
            else [],
            "learning_progress": {
                "total_episodes": self.episode_count,
                "training_steps": self.training_step,
                "current_epsilon": self.epsilon,
                "learning_metrics": self.learning_metrics.get_summary(),
            },
        }


class ContentAnalyzer:
    """Analyzes content characteristics for validation purposes."""

    def analyze_content_complexity(self, content: Dict[str, Any]) -> List[float]:
        """Analyze content complexity and return features."""
        text = str(content.get("text", ""))

        features = [
            len(text) / 1000.0,  # Text length (normalized)
            text.count(".") / max(len(text.split()), 1),  # Sentence complexity
            len(set(text.lower().split()))
            / max(len(text.split()), 1),  # Vocabulary diversity
            sum(1 for word in text.split() if len(word) > 6)
            / max(len(text.split()), 1),  # Complex words ratio
        ]

        return features

    def analyze_content_domain(self, content: Dict[str, Any]) -> List[float]:
        """Analyze content domain characteristics."""
        text = str(content.get("text", "")).lower()
        domain = content.get("domain", "").lower()

        # Domain-specific keyword presence
        math_keywords = [
            "equation",
            "formula",
            "calculate",
            "solve",
            "theorem",
            "proof",
        ]
        science_keywords = [
            "experiment",
            "hypothesis",
            "theory",
            "observation",
            "analysis",
        ]

        features = [
            sum(1 for keyword in math_keywords if keyword in text) / len(math_keywords),
            sum(1 for keyword in science_keywords if keyword in text)
            / len(science_keywords),
            1.0 if "mathematics" in domain else 0.0,
            1.0 if "science" in domain else 0.0,
        ]

        return features

    def analyze_quality_indicators(self, content: Dict[str, Any]) -> List[float]:
        """Analyze content quality indicators."""
        text = str(content.get("text", ""))

        # Quality indicators
        features = [
            content.get("accuracy_score", 0.5),
            content.get("clarity_score", 0.5),
            content.get("completeness_score", 0.5),
            content.get("engagement_score", 0.5),
            1.0 if text.count("?") > 0 else 0.0,  # Has questions
            1.0 if "example" in text.lower() else 0.0,  # Has examples
        ]

        return features

    def predict_quality_score(
        self, content: Dict[str, Any], strategy: ValidationStrategy
    ) -> float:
        """Predict quality score for content using the given strategy."""
        # Extract content features
        complexity_features = self.analyze_content_complexity(content)
        domain_features = self.analyze_content_domain(content)
        quality_features = self.analyze_quality_indicators(content)

        # Combine features
        all_features = complexity_features + domain_features + quality_features

        # Simple quality prediction based on features and strategy
        base_score = np.mean(quality_features[:4])  # Average of quality indicators

        # Adjust based on strategy parameters
        strictness = strategy.parameters.get("strictness_level", 0.5)
        if strictness > 0.7:
            base_score *= 0.9  # Stricter evaluation
        elif strictness < 0.4:
            base_score *= 1.1  # More lenient evaluation

        # Ensure score is in valid range
        return float(np.clip(base_score, 0.0, 1.0))


class FeedbackGenerator:
    """Generates structured feedback for content improvement."""

    def __init__(self):
        """Initialize feedback generator with templates."""
        self.feedback_templates = {
            "accuracy": [
                "Consider verifying the mathematical calculations in step {step}.",
                "The formula used in {section} may need review for accuracy.",
                "Double-check the factual claims made in {area}.",
            ],
            "clarity": [
                "The explanation in {section} could be clearer with more examples.",
                "Consider breaking down the complex concept in {area} into smaller parts.",
                "Add transitional phrases to improve flow between {section1} and {section2}.",
            ],
            "completeness": [
                "The solution is missing the final verification step.",
                "Consider adding more context about {concept} for completeness.",
                "The explanation would benefit from addressing potential edge cases.",
            ],
            "engagement": [
                "Adding real-world applications could make this more engaging.",
                "Consider including interactive elements or questions.",
                "Visual aids could help illustrate the concept better.",
            ],
        }

    def generate_structured_feedback(
        self,
        content: Dict[str, Any],
        quality_score: float,
        strategy: ValidationStrategy,
        environment_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate structured feedback for content improvement."""
        feedback = {
            "overall_score": quality_score,
            "strategy_used": strategy.name,
            "areas_for_improvement": [],
            "strengths": [],
            "specific_suggestions": [],
            "priority_level": "medium",
        }

        # Determine feedback detail level based on strategy
        detail_level = strategy.parameters.get("feedback_detail", 0.6)

        # Analyze content weaknesses
        if quality_score < 0.7:
            feedback["priority_level"] = "high"
            feedback["areas_for_improvement"] = self._identify_improvement_areas(
                content, quality_score
            )

        if quality_score > 0.8:
            feedback["strengths"] = self._identify_strengths(content)

        # Generate specific suggestions based on detail level
        if detail_level > 0.5:
            feedback["specific_suggestions"] = self._generate_specific_suggestions(
                content, quality_score, strategy
            )

        # Add domain-specific feedback if applicable
        domain = environment_state.get("domain", "")
        if domain and "domain_specific" in strategy.name:
            feedback["domain_specific_notes"] = self._generate_domain_feedback(
                content, domain
            )

        return feedback

    def _identify_improvement_areas(
        self, content: Dict[str, Any], quality_score: float
    ) -> List[str]:
        """Identify areas that need improvement."""
        areas = []

        # Check specific quality aspects
        accuracy_score = content.get("accuracy_score", quality_score)
        clarity_score = content.get("clarity_score", quality_score)
        completeness_score = content.get("completeness_score", quality_score)
        engagement_score = content.get("engagement_score", quality_score)

        if accuracy_score < 0.7:
            areas.append("accuracy")
        if clarity_score < 0.7:
            areas.append("clarity")
        if completeness_score < 0.7:
            areas.append("completeness")
        if engagement_score < 0.6:
            areas.append("engagement")

        return areas

    def _identify_strengths(self, content: Dict[str, Any]) -> List[str]:
        """Identify content strengths."""
        strengths = []

        text = str(content.get("text", "")).lower()

        if "example" in text:
            strengths.append("Good use of examples")
        if text.count("?") > 0:
            strengths.append("Includes engaging questions")
        if "step" in text:
            strengths.append("Clear step-by-step approach")
        if len(text.split()) > 100:
            strengths.append("Comprehensive coverage")

        return strengths

    def _generate_specific_suggestions(
        self,
        content: Dict[str, Any],
        quality_score: float,
        strategy: ValidationStrategy,
    ) -> List[str]:
        """Generate specific improvement suggestions."""
        suggestions = []

        # Get improvement areas
        areas = self._identify_improvement_areas(content, quality_score)

        for area in areas:
            if area in self.feedback_templates:
                # Select appropriate template
                templates = self.feedback_templates[area]
                suggestion = np.random.choice(templates)
                suggestions.append(suggestion)

        return suggestions

    def _generate_domain_feedback(self, content: Dict[str, Any], domain: str) -> str:
        """Generate domain-specific feedback."""
        domain_feedback = {
            "mathematics": "Ensure all mathematical notation is correct and calculations are verified.",
            "science": "Verify that scientific concepts are accurately represented and up-to-date.",
            "technology": "Check that technical information is current and properly explained.",
            "reading": "Focus on reading comprehension and literary analysis techniques.",
            "engineering": "Ensure practical applications and design principles are sound.",
            "arts": "Consider creative expression and artistic techniques in the content.",
        }

        return domain_feedback.get(
            domain.lower(), "Apply domain-specific best practices."
        )
