"""
Validator RL Agent.

This module implements the Validator RL Agent for multi-agent reinforcement learning
coordination. The Validator Agent is responsible for predicting content quality,
providing structured feedback, and learning optimal validation strategies.
"""

# Standard Library
import logging
import time
from typing import Any, Dict, List

# Third-Party Library
import numpy as np

# SynThesisAI Modules
from ...config import ValidatorAgentConfig
from ...exceptions import AgentFailureError
from ...logging_config import get_marl_logger
from ..base_agent import ActionSpace, BaseRLAgent

logger = logging.getLogger(__name__)


class ValidationStrategy:
    """Represents a content validation strategy with parameters."""

    def __init__(self, name: str, description: str, parameters: Dict[str, Any]) -> None:
        """
        Initialize validation strategy.

        Args:
            name: Strategy name.
            description: Strategy description.
            parameters: Strategy-specific parameters.
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
        alpha = 0.1
        self.accuracy_rate = (1 - alpha) * self.accuracy_rate + alpha * accuracy
        self.feedback_quality = (
            1 - alpha
        ) * self.feedback_quality + alpha * feedback_quality
        self.efficiency_score = (1 - alpha) * self.efficiency_score + alpha * efficiency
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

    def __init__(self, config: ValidatorAgentConfig) -> None:
        """
        Initialize Validator RL Agent.

        Args:
            config: Validator agent configuration.
        """
        self.strategies = self._initialize_strategies()
        super().__init__("validator", config)
        self.config: ValidatorAgentConfig = config
        self.logger = get_marl_logger("validator_agent")
        self.validation_history: List[Dict[str, Any]] = []
        self.validation_thresholds = np.linspace(0.5, 0.95, 10)
        self.validation_metrics = {
            "total_validations": 0,
            "accurate_validations": 0,
            "average_accuracy": 0.0,
            "average_feedback_quality": 0.0,
            "false_positives": 0,
            "false_negatives": 0,
            "strategy_usage": {strategy.name: 0 for strategy in self.strategies},
        }
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
        return [
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

    def get_state_representation(self, environment_state: Dict[str, Any]) -> np.ndarray:
        """
        Convert environment state to validator-specific representation.

        Args:
            environment_state: Raw environment state.

        Returns:
            A NumPy array representing the state for the validator agent.
        """
        try:
            features = []
            content = environment_state.get("content", {})
            features.extend(self.content_analyzer.analyze_content_complexity(content))
            features.extend(self.content_analyzer.analyze_content_domain(content))
            features.extend(self.content_analyzer.analyze_quality_indicators(content))
            features.extend(self._encode_validation_history())
            generator_strategy = environment_state.get("generator_strategy", {})
            features.extend(self._encode_generator_strategy(generator_strategy))
            domain = environment_state.get("domain", "")
            features.extend(self._encode_domain_context(domain))
            difficulty_level = environment_state.get("difficulty_level", "")
            features.extend(self._encode_difficulty_context(difficulty_level))
            quality_requirements = environment_state.get("quality_requirements", {})
            features.extend(self._encode_quality_requirements(quality_requirements))
            coordination_context = environment_state.get("coordination_context", {})
            features.extend(self._encode_coordination_context(coordination_context))
            return np.array(features, dtype=np.float32)
        except Exception as e:
            error_msg = "Failed to encode state for validator agent"
            self.logger.log_error_with_context(
                e,
                {
                    "environment_state_keys": list(environment_state.keys()),
                    "agent_id": self.agent_id,
                },
            )
            return np.zeros(80, dtype=np.float32)

    def get_action_space(self) -> ActionSpace:
        """
        Define validation threshold and feedback action space.

        Returns:
            An ActionSpace defining available validation strategies.
        """
        return ActionSpace([strategy.name for strategy in self.strategies])

    def calculate_reward(
        self, state: np.ndarray, action: int, result: Dict[str, Any]
    ) -> float:
        """
        Calculate validator-specific reward.

        Args:
            state: The state representation.
            action: The action taken (strategy index).
            result: The result of the validation.

        Returns:
            The reward value for the validator agent.
        """
        try:
            validation_accuracy = result.get("validation_accuracy", 0.0)
            feedback_quality = result.get("feedback_quality_score", 0.0)
            validation_efficiency = result.get("validation_time_score", 0.0)
            base_reward = (
                self.config.accuracy_weight * validation_accuracy
                + self.config.efficiency_weight * validation_efficiency
            )
            feedback_bonus = self.config.feedback_quality_weight * feedback_quality
            base_reward += feedback_bonus
            false_positive_count = result.get("false_positive_count", 0)
            false_negative_count = result.get("false_negative_count", 0)
            false_positive_penalty = (
                false_positive_count * self.config.false_positive_penalty
            )
            false_negative_penalty = (
                false_negative_count * self.config.false_negative_penalty
            )
            base_reward -= false_positive_penalty + false_negative_penalty
            if result.get("coordination_success", False):
                base_reward += self.config.coordination_bonus
            strategy = self.strategies[action]
            strategy.update_performance(
                validation_accuracy,
                feedback_quality,
                validation_efficiency,
                false_positive_count,
                false_negative_count,
            )
            self._update_validation_metrics(
                validation_accuracy,
                feedback_quality,
                false_positive_count,
                false_negative_count,
                strategy.name,
            )
            return float(np.clip(base_reward, -1.0, 2.0))
        except Exception as e:
            error_msg = "Failed to calculate reward for validator agent"
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
            content: The content to validate.
            environment_state: The current environment state.

        Returns:
            A dictionary with quality prediction and feedback.
        """
        try:
            state = self.get_state_representation(
                {**environment_state, "content": content}
            )
            action_index = self.select_action(state, training=True)
            selected_strategy = self.strategies[action_index]
            threshold = self._get_threshold_for_strategy(selected_strategy)
            quality_prediction = self.content_analyzer.predict_quality_score(
                content, selected_strategy
            )
            feedback = self.feedback_generator.generate_structured_feedback(
                content, quality_prediction, selected_strategy, environment_state
            )
            confidence = self.get_action_confidence(state, action_index)
            validation_record = {
                "strategy": selected_strategy.name,
                "timestamp": time.time(),
                "quality_prediction": quality_prediction,
                "threshold": threshold,
                "confidence": confidence,
                "content_summary": self._summarize_content(content),
            }
            self.validation_history.append(validation_record)
            self.validation_metrics["total_validations"] += 1
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
            error_msg = "Failed to predict quality and provide feedback"
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
        if "strict" in strategy.name:
            return max(base_threshold, 0.8)
        if "lenient" in strategy.name:
            return min(base_threshold, 0.6)
        if "adaptive" in strategy.name:
            recent_accuracy = strategy.accuracy_rate
            if recent_accuracy > 0.8:
                return base_threshold + 0.05
            if recent_accuracy < 0.6:
                return base_threshold - 0.05
        return base_threshold

    def _encode_validation_history(self) -> List[float]:
        """Encode validation history into features."""
        features = [
            self.validation_metrics["average_accuracy"],
            self.validation_metrics["average_feedback_quality"],
        ]
        total_validations = max(self.validation_metrics["total_validations"], 1)
        false_positive_rate = (
            self.validation_metrics["false_positives"] / total_validations
        )
        false_negative_rate = (
            self.validation_metrics["false_negatives"] / total_validations
        )
        features.extend([false_positive_rate, false_negative_rate])
        recent_validations = self.validation_history[-10:]
        strategy_performance = np.zeros(len(self.strategies))
        for record in recent_validations:
            strategy_name = record["strategy"]
            for i, strategy in enumerate(self.strategies):
                if strategy.name == strategy_name:
                    strategy_performance[i] += 1
                    break
        if recent_validations:
            strategy_performance /= len(recent_validations)
        features.extend(strategy_performance.tolist())
        return features

    def _encode_generator_strategy(
        self, generator_strategy: Dict[str, Any]
    ) -> List[float]:
        """Encode generator strategy information into features."""
        strategy_name = generator_strategy.get("strategy", "")
        confidence = generator_strategy.get("confidence", 0.0)
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
        return strategy_encoding + [confidence]

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
        return domain_mapping.get(domain.lower(), [0.0] * 6)

    def _encode_difficulty_context(self, difficulty: str) -> List[float]:
        """Encode difficulty level into features."""
        difficulty_mapping = {
            "elementary": [1.0, 0.0, 0.0, 0.0],
            "high_school": [0.0, 1.0, 0.0, 0.0],
            "undergraduate": [0.0, 0.0, 1.0, 0.0],
            "graduate": [0.0, 0.0, 0.0, 1.0],
        }
        return difficulty_mapping.get(difficulty.lower().replace(" ", "_"), [0.0] * 4)

    def _encode_quality_requirements(self, requirements: Dict[str, Any]) -> List[float]:
        """Encode quality requirements into features."""
        return [
            requirements.get("accuracy_threshold", 0.8),
            requirements.get("clarity_threshold", 0.7),
            requirements.get("completeness_threshold", 0.8),
            requirements.get("engagement_threshold", 0.6),
        ]

    def _encode_coordination_context(self, context: Dict[str, Any]) -> List[float]:
        """Encode coordination context into features."""
        return [
            context.get("coordination_quality", 0.5),
            context.get("consensus_level", 0.5),
            1.0 if context.get("coordination_success") else 0.0,
            context.get("agent_agreement", 0.5),
        ]

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
        if accuracy > 0.7:
            self.validation_metrics["accurate_validations"] += 1
        alpha = 0.1
        self.validation_metrics["average_accuracy"] = (
            1 - alpha
        ) * self.validation_metrics["average_accuracy"] + alpha * accuracy
        self.validation_metrics["average_feedback_quality"] = (
            1 - alpha
        ) * self.validation_metrics[
            "average_feedback_quality"
        ] + alpha * feedback_quality
        self.validation_metrics["false_positives"] += false_positives
        self.validation_metrics["false_negatives"] += false_negatives
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
        strategy_summaries.sort(key=lambda x: x["overall_score"], reverse=True)
        return {
            "agent_id": self.agent_id,
            "validation_metrics": self.validation_metrics.copy(),
            "strategy_performance": strategy_summaries,
            "recent_validation_history": self.validation_history[-20:],
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
        return [
            len(text) / 1000.0,
            text.count(".") / max(len(text.split()), 1),
            len(set(text.lower().split())) / max(len(text.split()), 1),
            sum(1 for word in text.split() if len(word) > 6)
            / max(len(text.split()), 1),
        ]

    def analyze_content_domain(self, content: Dict[str, Any]) -> List[float]:
        """Analyze content domain characteristics."""
        text = str(content.get("text", "")).lower()
        domain = content.get("domain", "").lower()
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
        return [
            sum(1 for keyword in math_keywords if keyword in text) / len(math_keywords),
            sum(1 for keyword in science_keywords if keyword in text)
            / len(science_keywords),
            1.0 if "mathematics" in domain else 0.0,
            1.0 if "science" in domain else 0.0,
        ]

    def analyze_quality_indicators(self, content: Dict[str, Any]) -> List[float]:
        """Analyze content quality indicators."""
        text = str(content.get("text", ""))
        return [
            content.get("accuracy_score", 0.5),
            content.get("clarity_score", 0.5),
            content.get("completeness_score", 0.5),
            content.get("engagement_score", 0.5),
            1.0 if text.count("?") > 0 else 0.0,
            1.0 if "example" in text.lower() else 0.0,
        ]

    def predict_quality_score(
        self, content: Dict[str, Any], strategy: ValidationStrategy
    ) -> float:
        """Predict quality score for content using the given strategy."""
        complexity_features = self.analyze_content_complexity(content)
        domain_features = self.analyze_content_domain(content)
        quality_features = self.analyze_quality_indicators(content)
        all_features = complexity_features + domain_features + quality_features
        base_score = np.mean(quality_features[:4])
        strictness = strategy.parameters.get("strictness_level", 0.5)
        if strictness > 0.7:
            base_score *= 0.9
        elif strictness < 0.4:
            base_score *= 1.1
        return float(np.clip(base_score, 0.0, 1.0))


class FeedbackGenerator:
    """Generates structured feedback for content improvement."""

    def __init__(self) -> None:
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
        feedback: Dict[str, Any] = {
            "overall_score": quality_score,
            "strategy_used": strategy.name,
            "areas_for_improvement": [],
            "strengths": [],
            "specific_suggestions": [],
            "priority_level": "medium",
        }
        detail_level = strategy.parameters.get("feedback_detail", 0.6)
        if quality_score < 0.7:
            feedback["priority_level"] = "high"
            feedback["areas_for_improvement"] = self._identify_improvement_areas(
                content, quality_score
            )
        if quality_score > 0.8:
            feedback["strengths"] = self._identify_strengths(content)
        if detail_level > 0.5:
            feedback["specific_suggestions"] = self._generate_specific_suggestions(
                content, quality_score, strategy
            )
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
        areas = self._identify_improvement_areas(content, quality_score)
        for area in areas:
            if area in self.feedback_templates:
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
