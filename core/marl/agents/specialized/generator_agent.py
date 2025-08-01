"""
Generator RL Agent.

This module implements the Generator RL Agent for multi-agent reinforcement learning
coordination. The Generator Agent is responsible for selecting optimal content
generation strategies based on environmental state and learning from feedback.
"""

# Standard Library
import logging
import time
from typing import Any, Dict, List

# Third-Party Library
import numpy as np

# SynThesisAI Modules
from ...config import GeneratorAgentConfig
from ...exceptions import AgentFailureError
from ...logging_config import get_marl_logger
from ..base_agent import ActionSpace, BaseRLAgent

logger = logging.getLogger(__name__)


class GenerationStrategy:
    """Represents a content generation strategy with parameters."""

    def __init__(self, name: str, description: str, parameters: Dict[str, Any]) -> None:
        """
        Initialize generation strategy.

        Args:
            name: Strategy name.
            description: Strategy description.
            parameters: Strategy-specific parameters.
        """
        self.name = name
        self.description = description
        self.parameters = parameters
        self.usage_count = 0
        self.success_rate = 0.0
        self.average_quality = 0.0
        self.average_novelty = 0.0
        self.average_efficiency = 0.0

    def update_performance(
        self, quality: float, novelty: float, efficiency: float, success: bool
    ) -> None:
        """Update strategy performance metrics."""
        self.usage_count += 1
        # Update running averages
        alpha = 0.1  # Learning rate for running average
        self.average_quality = (1 - alpha) * self.average_quality + alpha * quality
        self.average_novelty = (1 - alpha) * self.average_novelty + alpha * novelty
        self.average_efficiency = (
            1 - alpha
        ) * self.average_efficiency + alpha * efficiency

        # Update success rate
        self.success_rate = (1 - alpha) * self.success_rate + alpha * (
            1.0 if success else 0.0
        )

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for this strategy."""
        return {
            "name": self.name,
            "usage_count": self.usage_count,
            "success_rate": self.success_rate,
            "average_quality": self.average_quality,
            "average_novelty": self.average_novelty,
            "average_efficiency": self.average_efficiency,
            "overall_score": (
                self.average_quality + self.average_novelty + self.average_efficiency
            )
            / 3,
        }


class GeneratorRLAgent(BaseRLAgent):
    """
    Generator RL Agent for content generation strategy selection.

    This agent learns to select optimal content generation strategies based on
    the current context and requirements, optimizing for quality, novelty, and
    efficiency through reinforcement learning.
    """

    def __init__(self, config: GeneratorAgentConfig) -> None:
        """
        Initialize Generator RL Agent.

        Args:
            config: Generator agent configuration.
        """
        self.config: GeneratorAgentConfig = config
        self.logger = get_marl_logger("generator_agent")

        # Generation strategies - initialize before calling super()
        self.strategies = self._initialize_strategies()
        super().__init__("generator", config)
        self.strategy_history: List[Dict[str, Any]] = []

        # Performance tracking
        self.generation_metrics = {
            "total_generations": 0,
            "successful_generations": 0,
            "average_quality": 0.0,
            "average_novelty": 0.0,
            "average_efficiency": 0.0,
            "strategy_usage": {strategy.name: 0 for strategy in self.strategies},
        }

        # Context encoding
        self.context_encoder = GenerationContextEncoder()
        self.logger.log_agent_action(
            self.agent_id,
            "generator_initialized",
            1.0,
            f"Strategies: {len(self.strategies)}, Config: {type(config).__name__}",
        )

    def _initialize_strategies(self) -> List[GenerationStrategy]:
        """Initialize available generation strategies."""
        return [
            GenerationStrategy(
                "step_by_step_approach",
                "Generate content with clear step-by-step progression",
                {
                    "structure_weight": 0.8,
                    "clarity_emphasis": 0.9,
                    "progression_type": "linear",
                },
            ),
            GenerationStrategy(
                "concept_based_generation",
                "Focus on core concepts and build understanding",
                {
                    "concept_depth": 0.7,
                    "connection_emphasis": 0.8,
                    "abstraction_level": "medium",
                },
            ),
            GenerationStrategy(
                "problem_solving_focus",
                "Emphasize problem-solving techniques and methods",
                {
                    "technique_variety": 0.8,
                    "solution_depth": 0.7,
                    "method_explanation": 0.9,
                },
            ),
            GenerationStrategy(
                "creative_exploration",
                "Encourage creative thinking and exploration",
                {
                    "creativity_weight": 0.9,
                    "exploration_depth": 0.8,
                    "unconventional_approaches": 0.7,
                },
            ),
            GenerationStrategy(
                "structured_reasoning",
                "Provide structured logical reasoning",
                {
                    "logic_emphasis": 0.9,
                    "reasoning_depth": 0.8,
                    "structure_clarity": 0.8,
                },
            ),
            GenerationStrategy(
                "adaptive_difficulty",
                "Adapt difficulty based on context and requirements",
                {
                    "difficulty_sensitivity": 0.8,
                    "adaptation_speed": 0.7,
                    "context_awareness": 0.9,
                },
            ),
            GenerationStrategy(
                "multi_perspective",
                "Present multiple perspectives and approaches",
                {
                    "perspective_variety": 0.8,
                    "comparison_depth": 0.7,
                    "synthesis_quality": 0.8,
                },
            ),
            GenerationStrategy(
                "real_world_application",
                "Connect content to real-world applications",
                {
                    "application_relevance": 0.9,
                    "practical_emphasis": 0.8,
                    "context_connection": 0.8,
                },
            ),
        ]

    def get_state_representation(self, environment_state: Dict[str, Any]) -> np.ndarray:
        """
        Convert environment state to generator-specific representation.

        Args:
            environment_state: Raw environment state.

        Returns:
            A NumPy array representing the state for the generator agent.
        """
        try:
            # Extract key features from environment state
            features = []

            # Context request features
            domain = environment_state.get("domain", "")
            features.extend(self.context_encoder.encode_domain(domain))

            difficulty_level = environment_state.get("difficulty_level", "")
            features.extend(self.context_encoder.encode_difficulty(difficulty_level))

            topic = environment_state.get("topic", "")
            features.extend(self.context_encoder.encode_topic(topic))

            # Quality requirements
            quality_requirements = environment_state.get("quality_requirements", {})
            features.extend(
                self.context_encoder.encode_quality_requirements(quality_requirements)
            )

            # Historical performance features
            features.extend(self._encode_performance_history())

            # Context features
            target_audience = environment_state.get("target_audience", "")
            features.extend(self.context_encoder.encode_audience(target_audience))

            learning_objectives = environment_state.get("learning_objectives", [])
            features.extend(self.context_encoder.encode_objectives(learning_objectives))

            # Coordination context
            coordination_context = environment_state.get("coordination_context", {})
            features.extend(
                self.context_encoder.encode_coordination_context(coordination_context)
            )
            return np.array(features, dtype=np.float32)
        except Exception as e:
            self.logger.log_error_with_context(
                e,
                {
                    "environment_state_keys": list(environment_state.keys()),
                    "agent_id": self.agent_id,
                },
            )
            # Return default state representation
            return np.zeros(64, dtype=np.float32)

    def get_action_space(self) -> ActionSpace:
        """
        Define generation strategy action space.

        Returns:
            An ActionSpace defining available generation strategies.
        """
        return ActionSpace([strategy.name for strategy in self.strategies])

    def calculate_reward(
        self, state: np.ndarray, action: int, result: Dict[str, Any]
    ) -> float:
        """
        Calculate generator-specific reward.

        Args:
            state: The state representation.
            action: The action taken (strategy index).
            result: The result of the generation.

        Returns:
            The reward value for the generator agent.
        """
        try:
            # Extract performance metrics from result
            quality_score = result.get("quality_metrics", {}).get("overall_score", 0.0)
            novelty_score = result.get("novelty_score", 0.0)
            efficiency_metrics = result.get("efficiency_metrics", {})
            efficiency_score = efficiency_metrics.get("generation_time_score", 0.0)

            # Multi-objective reward function
            quality_weight = self.config.quality_weight
            novelty_weight = self.config.novelty_weight
            efficiency_weight = self.config.efficiency_weight

            base_reward = (
                quality_weight * quality_score
                + novelty_weight * novelty_score
                + efficiency_weight * efficiency_score
            )

            # Bonus for successful coordination
            if result.get("coordination_success", False):
                base_reward += self.config.coordination_bonus

            # Penalty for validation failures
            validation_passed = result.get("validation_passed", True)
            if not validation_passed:
                base_reward -= self.config.validation_penalty

            # Strategy-specific bonuses
            strategy = self.strategies[action]
            if quality_score > self.config.quality_threshold:
                base_reward += 0.1 * strategy.average_quality

            if novelty_score > self.config.novelty_threshold:
                base_reward += 0.1 * strategy.average_novelty

            # Update strategy performance
            strategy.update_performance(
                quality_score, novelty_score, efficiency_score, validation_passed
            )

            # Update agent metrics
            self._update_generation_metrics(
                quality_score,
                novelty_score,
                efficiency_score,
                validation_passed,
                strategy.name,
            )
            # Clip reward to reasonable range
            return float(np.clip(base_reward, -1.0, 2.0))
        except Exception as e:
            self.logger.log_error_with_context(
                e,
                {
                    "action": action,
                    "result_keys": list(result.keys()),
                    "agent_id": self.agent_id,
                },
            )
            return 0.0

    def select_generation_strategy(
        self, environment_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Select generation strategy based on current environment state.

        Args:
            environment_state: Current environment state.

        Returns:
            A dictionary with selected strategy information.
        """
        try:
            # Get state representation
            state = self.get_state_representation(environment_state)

            # Select action using RL policy
            action_index = self.select_action(state, training=True)

            # Get selected strategy
            selected_strategy = self.strategies[action_index]

            # Calculate confidence
            confidence = self.get_action_confidence(state, action_index)

            # Record strategy selection
            self.strategy_history.append(
                {
                    "strategy": selected_strategy.name,
                    "timestamp": time.time(),
                    "confidence": confidence,
                    "state_summary": self._summarize_state(environment_state),
                }
            )

            # Keep history manageable
            if len(self.strategy_history) > 1000:
                self.strategy_history = self.strategy_history[-1000:]
            self.logger.log_agent_action(
                self.agent_id,
                selected_strategy.name,
                confidence,
                (
                    f"Usage: {selected_strategy.usage_count}, "
                    f"Success: {selected_strategy.success_rate:.2f}"
                ),
            )

            return {
                "strategy": selected_strategy.name,
                "description": selected_strategy.description,
                "parameters": selected_strategy.parameters.copy(),
                "confidence": confidence,
                "performance_history": selected_strategy.get_performance_summary(),
            }
        except Exception as e:
            error_msg = "Failed to select generation strategy"
            self.logger.log_error_with_context(
                e,
                {
                    "environment_state_keys": list(environment_state.keys()),
                    "agent_id": self.agent_id,
                },
            )
            raise AgentFailureError(
                error_msg, agent_id=self.agent_id, failure_type="strategy_selection"
            ) from e

    def _encode_performance_history(self) -> List[float]:
        """Encode historical performance into features."""
        features = [
            # Overall performance metrics
            self.generation_metrics["average_quality"],
            self.generation_metrics["average_novelty"],
            self.generation_metrics["average_efficiency"],
            self.generation_metrics["successful_generations"]
            / max(self.generation_metrics["total_generations"], 1),
        ]

        # Recent strategy performance (last 10 strategies)
        recent_strategies = self.strategy_history[-10:]
        strategy_performance = np.zeros(len(self.strategies))
        for record in recent_strategies:
            strategy_name = record["strategy"]
            for i, strategy in enumerate(self.strategies):
                if strategy.name == strategy_name:
                    strategy_performance[i] += 1
                    break

        # Normalize by number of recent strategies
        if recent_strategies:
            strategy_performance /= len(recent_strategies)

        features.extend(strategy_performance.tolist())

        return features

    def _update_generation_metrics(
        self,
        quality: float,
        novelty: float,
        efficiency: float,
        success: bool,
        strategy_name: str,
    ) -> None:
        """Update generation performance metrics."""
        self.generation_metrics["total_generations"] += 1
        if success:
            self.generation_metrics["successful_generations"] += 1

        # Update running averages
        alpha = 0.1
        self.generation_metrics["average_quality"] = (
            1 - alpha
        ) * self.generation_metrics["average_quality"] + alpha * quality
        self.generation_metrics["average_novelty"] = (
            1 - alpha
        ) * self.generation_metrics["average_novelty"] + alpha * novelty
        self.generation_metrics["average_efficiency"] = (
            1 - alpha
        ) * self.generation_metrics["average_efficiency"] + alpha * efficiency

        # Update strategy usage
        self.generation_metrics["strategy_usage"][strategy_name] += 1

    def _summarize_state(self, environment_state: Dict[str, Any]) -> str:
        """Create a summary of the environment state for logging."""
        domain = environment_state.get("domain", "unknown")
        difficulty = environment_state.get("difficulty_level", "unknown")
        topic = environment_state.get("topic", "unknown")

        return f"domain={domain}, difficulty={difficulty}, topic={topic}"

    def get_agent_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary for the generator agent."""
        strategy_summaries = [
            strategy.get_performance_summary() for strategy in self.strategies
        ]

        # Sort strategies by overall performance
        strategy_summaries.sort(key=lambda x: x["overall_score"], reverse=True)

        return {
            "agent_id": self.agent_id,
            "generation_metrics": self.generation_metrics.copy(),
            "strategy_performance": strategy_summaries,
            "recent_strategy_history": self.strategy_history[-20:],
            "learning_progress": {
                "total_episodes": self.episode_count,
                "training_steps": self.training_step,
                "current_epsilon": self.epsilon,
                "learning_metrics": self.learning_metrics.get_summary(),
            },
        }


class GenerationContextEncoder:
    """Encodes generation context into numerical features for the RL agent."""

    def __init__(self) -> None:
        """Initialize context encoder with domain mappings."""
        self.domain_mapping = {
            "mathematics": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "science": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            "technology": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            "reading": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            "engineering": [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            "arts": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        }
        self.difficulty_mapping = {
            "elementary": [1.0, 0.0, 0.0, 0.0],
            "high_school": [0.0, 1.0, 0.0, 0.0],
            "undergraduate": [0.0, 0.0, 1.0, 0.0],
            "graduate": [0.0, 0.0, 0.0, 1.0],
        }

    def encode_domain(self, domain: str) -> List[float]:
        """Encode domain into numerical features."""
        return self.domain_mapping.get(domain.lower(), [0.0] * 6)

    def encode_difficulty(self, difficulty: str) -> List[float]:
        """Encode difficulty level into numerical features."""
        return self.difficulty_mapping.get(
            difficulty.lower().replace(" ", "_"), [0.0] * 4
        )

    def encode_topic(self, topic: str) -> List[float]:
        """Encode topic into numerical features."""
        # Simple topic encoding based on length and complexity
        return [
            len(topic) / 100.0,  # Topic length (normalized)
            topic.count(" ") / 10.0,  # Word count (normalized)
            sum(1 for c in topic if c.isupper()) / max(len(topic), 1),  # Capitalization ratio
            sum(1 for c in topic if c.isdigit()) / max(len(topic), 1),  # Digit ratio
        ]

    def encode_quality_requirements(self, requirements: Dict[str, Any]) -> List[float]:
        """Encode quality requirements into numerical features."""
        return [
            requirements.get("accuracy_threshold", 0.8),
            requirements.get("clarity_threshold", 0.7),
            requirements.get("completeness_threshold", 0.8),
            requirements.get("engagement_threshold", 0.6),
        ]

    def encode_audience(self, audience: str) -> List[float]:
        """Encode target audience into numerical features."""
        audience_lower = audience.lower()

        # Audience characteristics
        return [
            1.0 if "student" in audience_lower else 0.0,
            1.0 if "teacher" in audience_lower else 0.0,
            1.0 if "beginner" in audience_lower else 0.0,
            1.0 if "advanced" in audience_lower else 0.0,
        ]

    def encode_objectives(self, objectives: List[str]) -> List[float]:
        """Encode learning objectives into numerical features."""
        if not objectives:
            return [0.0] * 4

        # Objective characteristics
        total_length = sum(len(obj) for obj in objectives)
        avg_length = total_length / len(objectives)
        return [
            len(objectives) / 10.0,     # Number of objectives (normalized)
            avg_length / 100.0,         # Average objective length (normalized)
            sum(1 for obj in objectives if "understand" in obj.lower())
            / len(objectives),
            sum(1 for obj in objectives if "apply" in obj.lower()) / len(objectives),
        ]

    def encode_coordination_context(self, context: Dict[str, Any]) -> List[float]:
        """Encode coordination context into numerical features."""
        return [
            context.get("coordination_quality", 0.5),
            context.get("consensus_level", 0.5),
            1.0 if context.get("coordination_success") else 0.0,
            context.get("agent_agreement", 0.5),
        ]
