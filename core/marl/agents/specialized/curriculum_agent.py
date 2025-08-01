"""Curriculum RL Agent for Multi-Agent Reinforcement Learning.

This module implements the Curriculum RL Agent, which is responsible for ensuring
pedagogical coherence, learning progression, and curriculum-based improvements in
content generation within the SynThesisAI system.
"""

# Standard Library
import logging
import time
from typing import Any, Dict, List

# Third-Party Library
import numpy as np

# SynThesisAI Modules
from core.marl.agents.base_agent import ActionSpace, BaseRLAgent
from core.marl.config import CurriculumAgentConfig
from core.marl.exceptions import AgentFailureError
from core.marl.logging_config import get_marl_logger

# TODO: Move LearningAnalyzer, ProgressionModeler, and ObjectiveAligner to separate
# files in a dedicated 'analysis' or 'utils' module to improve modularity.
# For now, they remain here to adhere to the single-file refactoring constraint.

logger = logging.getLogger(__name__)


class CurriculumStrategy:
    """Represents a curriculum strategy with its parameters and performance metrics.

    Attributes:
        name (str): The name of the strategy.
        description (str): A brief description of the strategy.
        parameters (Dict[str, Any]): Strategy-specific configuration parameters.
        usage_count (int): The number of times the strategy has been used.
        pedagogical_coherence (float): The running average of the pedagogical
            coherence score.
        learning_progression (float): The running average of the learning
            progression score.
        objective_alignment (float): The running average of the objective
            alignment score.
        integration_success_rate (float): The running average of the integration
            success rate.
    """

    def __init__(self, name: str, description: str, parameters: Dict[str, Any]):
        """Initializes the CurriculumStrategy.

        Args:
            name: The name of the strategy.
            description: A brief description of the strategy.
            parameters: Strategy-specific configuration parameters.
        """
        self.name = name
        self.description = description
        self.parameters = parameters
        self.usage_count = 0
        self.pedagogical_coherence = 0.0
        self.learning_progression = 0.0
        self.objective_alignment = 0.0
        self.integration_success_rate = 0.0

    def update_performance(
        self,
        pedagogical_coherence: float,
        learning_progression: float,
        objective_alignment: float,
        integration_success: bool,
    ) -> None:
        """Updates the strategy's performance metrics using a running average.

        Args:
            pedagogical_coherence: The pedagogical coherence score from the latest use.
            learning_progression: The learning progression score from the latest use.
            objective_alignment: The objective alignment score from the latest use.
            integration_success: Whether the curriculum was successfully integrated.
        """
        self.usage_count += 1
        alpha = 0.1  # Learning rate for the running average

        self.pedagogical_coherence = (
            1 - alpha
        ) * self.pedagogical_coherence + alpha * pedagogical_coherence
        self.learning_progression = (
            1 - alpha
        ) * self.learning_progression + alpha * learning_progression
        self.objective_alignment = (
            1 - alpha
        ) * self.objective_alignment + alpha * objective_alignment
        self.integration_success_rate = (1 - alpha) * self.integration_success_rate + alpha * (
            1.0 if integration_success else 0.0
        )

    def get_performance_summary(self) -> Dict[str, Any]:
        """Returns a summary of the strategy's performance.

        Returns:
            A dictionary containing performance metrics and an overall score.
        """
        overall_score = (
            self.pedagogical_coherence + self.learning_progression + self.objective_alignment
        ) / 3
        return {
            "name": self.name,
            "usage_count": self.usage_count,
            "pedagogical_coherence": self.pedagogical_coherence,
            "learning_progression": self.learning_progression,
            "objective_alignment": self.objective_alignment,
            "integration_success_rate": self.integration_success_rate,
            "overall_score": overall_score,
        }


class CurriculumRLAgent(BaseRLAgent):
    """An RL agent that learns to optimize curriculum-based content generation.

    This agent ensures pedagogical coherence, optimizes learning progression, and
    aligns content with learning objectives through reinforcement learning.
    """

    def __init__(self, config: CurriculumAgentConfig):
        """Initializes the CurriculumRLAgent.

        Args:
            config: The configuration object for the curriculum agent.
        """
        self.strategies = self._initialize_strategies()
        super().__init__("curriculum", config)
        self.config = config
        self.logger = get_marl_logger("curriculum_agent")

        self.curriculum_history: List[Dict[str, Any]] = []
        self.curriculum_metrics: Dict[str, Any] = {
            "total_recommendations": 0,
            "successful_integrations": 0,
            "average_pedagogical_coherence": 0.0,
            "average_learning_progression": 0.0,
            "average_objective_alignment": 0.0,
            "strategy_usage": {strategy.name: 0 for strategy in self.strategies},
        }

        self.learning_analyzer = LearningAnalyzer()
        self.progression_modeler = ProgressionModeler()
        self.objective_aligner = ObjectiveAligner()

        self.logger.log_agent_action(
            self.agent_id,
            "curriculum_initialized",
            1.0,
            f"Strategies: {len(self.strategies)}, Config: {type(config).__name__}",
        )

    def _initialize_strategies(self) -> List[CurriculumStrategy]:
        """Initializes the list of available curriculum strategies.

        Returns:
            A list of `CurriculumStrategy` objects.
        """
        # In a production system, this would ideally be loaded from a configuration file.
        return [
            CurriculumStrategy(
                "linear_progression",
                "Organize content in linear, sequential progression.",
                {
                    "sequence_strength": 0.9,
                    "prerequisite_enforcement": 0.8,
                    "difficulty_gradient": 0.7,
                    "concept_ordering": "sequential",
                },
            ),
            CurriculumStrategy(
                "spiral_curriculum",
                "Revisit concepts with increasing complexity.",
                {
                    "spiral_depth": 0.8,
                    "concept_reinforcement": 0.9,
                    "complexity_layering": 0.8,
                    "revisit_frequency": 0.7,
                },
            ),
            CurriculumStrategy(
                "mastery_based_progression",
                "Ensure mastery before advancing to the next concepts.",
                {
                    "mastery_threshold": 0.85,
                    "competency_validation": 0.9,
                    "adaptive_pacing": 0.8,
                    "skill_verification": 0.8,
                },
            ),
            CurriculumStrategy(
                "adaptive_difficulty_adjustment",
                "Dynamically adjust difficulty based on learner progress.",
                {
                    "adaptivity_sensitivity": 0.8,
                    "difficulty_calibration": 0.9,
                    "progress_monitoring": 0.8,
                    "adjustment_speed": 0.7,
                },
            ),
            CurriculumStrategy(
                "prerequisite_reinforcement",
                "Strengthen prerequisite knowledge before introducing new concepts.",
                {
                    "prerequisite_depth": 0.8,
                    "knowledge_gap_detection": 0.9,
                    "reinforcement_intensity": 0.7,
                    "foundation_strength": 0.8,
                },
            ),
            CurriculumStrategy(
                "concept_scaffolding",
                "Provide structured support for complex concepts.",
                {
                    "scaffolding_levels": 0.8,
                    "support_gradation": 0.9,
                    "independence_building": 0.7,
                    "guidance_fading": 0.8,
                },
            ),
            CurriculumStrategy(
                "multi_modal_learning",
                "Integrate multiple learning modalities and approaches.",
                {
                    "modality_variety": 0.8,
                    "learning_style_accommodation": 0.9,
                    "engagement_diversity": 0.8,
                    "accessibility_support": 0.7,
                },
            ),
            CurriculumStrategy(
                "personalized_pathway",
                "Create individualized learning pathways based on learner profiles.",
                {
                    "personalization_depth": 0.9,
                    "learner_profile_integration": 0.8,
                    "pathway_flexibility": 0.8,
                    "individual_optimization": 0.9,
                },
            ),
        ]

    def get_state_representation(self, environment_state: Dict[str, Any]) -> np.ndarray:
        """Converts the environment state into a numerical representation.

        Args:
            environment_state: The raw state from the environment.

        Returns:
            A NumPy array representing the state for the curriculum agent.
        """
        try:
            features = []
            learning_objectives = environment_state.get("learning_objectives", [])
            features.extend(self.learning_analyzer.encode_learning_objectives(learning_objectives))

            target_audience = environment_state.get("target_audience", "")
            features.extend(self.learning_analyzer.encode_target_audience(target_audience))

            features.extend(self._encode_content_progression_history())
            features.extend(self._encode_pedagogical_context(environment_state))

            domain = environment_state.get("domain", "")
            features.extend(self._encode_domain_context(domain))

            difficulty_level = environment_state.get("difficulty_level", "")
            features.extend(self._encode_difficulty_context(difficulty_level))

            content = environment_state.get("content", {})
            features.extend(self.learning_analyzer.analyze_content_characteristics(content))

            generator_strategy = environment_state.get("generator_strategy", {})
            features.extend(self._encode_generator_context(generator_strategy))

            validator_feedback = environment_state.get("validator_feedback", {})
            features.extend(self._encode_validator_context(validator_feedback))

            coordination_context = environment_state.get("coordination_context", {})
            features.extend(self._encode_coordination_context(coordination_context))

            return np.array(features, dtype=np.float32)

        except Exception as e:
            self.logger.log_error_with_context(
                e,
                {
                    "environment_state_keys": list(environment_state.keys()),
                    "agent_id": self.agent_id,
                },
            )
            # Return a default state representation to prevent system failure.
            return np.zeros(100, dtype=np.float32)

    def get_action_space(self) -> ActionSpace:
        """Defines the action space for the curriculum agent.

        Returns:
            An `ActionSpace` object where each action corresponds to a curriculum
            strategy.
        """
        strategy_names = [strategy.name for strategy in self.strategies]
        return ActionSpace(strategy_names)

    def calculate_reward(self, state: np.ndarray, action: int, result: Dict[str, Any]) -> float:
        """Calculates the reward based on the outcome of a curriculum recommendation.

        Args:
            state: The state representation.
            action: The index of the action (strategy) taken.
            result: The result of the curriculum recommendation.

        Returns:
            The calculated reward value.
        """
        try:
            pedagogical_coherence = result.get("pedagogical_coherence_score", 0.0)
            learning_progression = result.get("learning_progression_score", 0.0)
            objective_alignment = result.get("objective_alignment_score", 0.0)

            reward = (
                self.config.pedagogical_coherence_weight * pedagogical_coherence
                + self.config.learning_progression_weight * learning_progression
                + self.config.objective_alignment_weight * objective_alignment
            )

            integration_success = result.get("curriculum_integration_success", False)
            if integration_success:
                reward += self.config.integration_bonus

            if pedagogical_coherence > self.config.coherence_threshold:
                reward += 0.1
            if learning_progression > self.config.progression_threshold:
                reward += 0.1
            if objective_alignment < 0.5:
                reward -= 0.2  # Penalty for poor alignment

            if result.get("coordination_success", False):
                reward += 0.1

            strategy = self.strategies[action]
            strategy.update_performance(
                pedagogical_coherence,
                learning_progression,
                objective_alignment,
                integration_success,
            )

            self._update_curriculum_metrics(
                pedagogical_coherence,
                learning_progression,
                objective_alignment,
                integration_success,
                strategy.name,
            )

            return float(np.clip(reward, -1.0, 2.0))

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

    def suggest_curriculum_improvements(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Suggests curriculum-based improvements for content generation.

        Args:
            request: The content generation request with contextual information.

        Returns:
            A dictionary containing curriculum improvement suggestions.

        Raises:
            AgentFailureError: If the agent fails to generate suggestions.
        """
        try:
            state = self.get_state_representation(request)
            action_index = self.select_action(state, training=True)
            selected_strategy = self.strategies[action_index]
            confidence = self.get_action_confidence(state, action_index)

            improvements = {
                "curriculum_strategy": selected_strategy.name,
                "strategy_description": selected_strategy.description,
                "strategy_parameters": selected_strategy.parameters.copy(),
                "difficulty_adjustments": self._suggest_difficulty_adjustments(request),
                "prerequisite_recommendations": self._identify_prerequisites(request),
                "learning_pathway": self._generate_learning_pathway(request),
                "pedagogical_hints": self._generate_pedagogical_hints(request),
                "progression_guidance": self._generate_progression_guidance(request),
                "objective_alignment": self._assess_objective_alignment(request),
                "confidence": confidence,
            }

            self._record_recommendation(selected_strategy.name, confidence, request)

            summary = (
                f"Usage: {selected_strategy.usage_count}, "
                f"Success: {selected_strategy.integration_success_rate:.2f}"
            )
            self.logger.log_agent_action(self.agent_id, selected_strategy.name, confidence, summary)

            return improvements

        except Exception as e:
            self.logger.log_error_with_context(
                e,
                {
                    "request_keys": list(request.keys()) if request else [],
                    "agent_id": self.agent_id,
                },
            )
            raise AgentFailureError(
                "Failed to suggest curriculum improvements",
                agent_id=self.agent_id,
                failure_type="curriculum_suggestion",
            ) from e

    def _record_recommendation(
        self, strategy_name: str, confidence: float, request: Dict[str, Any]
    ) -> None:
        """Records a curriculum recommendation and updates metrics."""
        curriculum_record = {
            "strategy": strategy_name,
            "timestamp": time.time(),
            "confidence": confidence,
            "request_summary": self._summarize_request(request),
        }
        self.curriculum_history.append(curriculum_record)
        self.curriculum_metrics["total_recommendations"] += 1

        # Keep history at a manageable size.
        if len(self.curriculum_history) > 1000:
            self.curriculum_history = self.curriculum_history[-1000:]

    def _suggest_difficulty_adjustments(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Suggests difficulty adjustments based on curriculum analysis."""
        current_difficulty = request.get("difficulty_level", "medium")
        target_audience = request.get("target_audience", "")
        learning_objectives = request.get("learning_objectives", [])

        adjustments = {
            "current_difficulty": current_difficulty,
            "recommended_difficulty": current_difficulty,
            "adjustment_rationale": "No adjustment needed.",
            "scaffolding_needed": False,
            "prerequisite_review": False,
        }

        if "beginner" in target_audience.lower() and current_difficulty in [
            "high",
            "advanced",
        ]:
            adjustments.update(
                recommended_difficulty="medium",
                adjustment_rationale="Reduce difficulty for beginner audience.",
                scaffolding_needed=True,
            )
        elif "advanced" in target_audience.lower() and current_difficulty in [
            "low",
            "elementary",
        ]:
            adjustments.update(
                recommended_difficulty="high",
                adjustment_rationale="Increase difficulty for advanced audience.",
            )

        complex_objectives = [obj for obj in learning_objectives if len(obj.split()) > 5]
        if len(complex_objectives) > len(learning_objectives) / 2:
            adjustments["prerequisite_review"] = True

        return adjustments

    def _identify_prerequisites(self, request: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identifies prerequisite knowledge and skills for a given request."""
        domain = request.get("domain", "").lower()
        topic = request.get("topic", "").lower()
        difficulty = request.get("difficulty_level", "")
        prerequisites = []

        if domain == "mathematics":
            if "algebra" in topic:
                prerequisites.extend(
                    [
                        {
                            "concept": "Basic arithmetic",
                            "importance": "high",
                            "description": "Operations on numbers.",
                        },
                        {
                            "concept": "Number properties",
                            "importance": "medium",
                            "description": "Integers, fractions, decimals.",
                        },
                    ]
                )
            if "calculus" in topic:
                prerequisites.extend(
                    [
                        {
                            "concept": "Algebra fundamentals",
                            "importance": "critical",
                            "description": "Solving equations, functions.",
                        },
                        {
                            "concept": "Trigonometry",
                            "importance": "high",
                            "description": "Functions and identities.",
                        },
                    ]
                )
        elif domain == "science" and "physics" in topic:
            prerequisites.extend(
                [
                    {
                        "concept": "Mathematical foundations",
                        "importance": "high",
                        "description": "Algebra and basic calculus.",
                    },
                    {
                        "concept": "Scientific method",
                        "importance": "medium",
                        "description": "Hypothesis and experimentation.",
                    },
                ]
            )

        if difficulty in ["high", "advanced"]:
            prerequisites.append(
                {
                    "concept": "Critical thinking",
                    "importance": "high",
                    "description": "Ability to analyze and synthesize information.",
                }
            )
        return prerequisites

    def _generate_learning_pathway(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Generates a structured learning pathway."""
        learning_objectives = request.get("learning_objectives", [])
        domain = request.get("domain", "")

        pathway = {
            "pathway_type": "sequential",
            "total_steps": len(learning_objectives) or 3,
            "estimated_duration": "medium",
            "steps": [],
            "milestones": [],
        }

        if not learning_objectives:
            # Provide a default pathway if no objectives are specified.
            pathway["steps"] = [
                {
                    "step_number": 1,
                    "objective": "Introduction",
                    "activities": ["Concept introduction"],
                },
                {
                    "step_number": 2,
                    "objective": "Practice",
                    "activities": ["Guided practice"],
                },
                {
                    "step_number": 3,
                    "objective": "Mastery",
                    "activities": ["Independent work"],
                },
            ]
            return pathway

        for i, objective in enumerate(learning_objectives):
            step = {
                "step_number": i + 1,
                "objective": objective,
                "activities": self._suggest_learning_activities(objective, domain),
                "resources": self._suggest_learning_resources(objective, domain),
                "assessment": self._suggest_assessment_method(objective),
            }
            pathway["steps"].append(step)

            if (i + 1) % 3 == 0 or (i + 1) == len(learning_objectives):
                pathway["milestones"].append(
                    {
                        "milestone_number": len(pathway["milestones"]) + 1,
                        "description": f"Complete objectives 1-{i + 1}",
                        "success_criteria": "Demonstrate understanding of concepts.",
                    }
                )
        return pathway

    def _generate_pedagogical_hints(self, request: Dict[str, Any]) -> List[str]:
        """Generates pedagogical hints for content improvement."""
        hints = []
        domain = request.get("domain", "").lower()
        target_audience = request.get("target_audience", "").lower()
        objectives = request.get("learning_objectives", [])

        if domain == "mathematics":
            hints.extend(
                [
                    "Use visual aids to illustrate abstract concepts.",
                    "Provide step-by-step examples.",
                    "Connect concepts to real-world applications.",
                ]
            )
        elif domain == "science":
            hints.extend(
                [
                    "Incorporate hands-on experiments or demonstrations.",
                    "Use analogies to explain complex phenomena.",
                    "Emphasize evidence-based reasoning.",
                ]
            )

        if "beginner" in target_audience:
            hints.extend(
                [
                    "Start with concrete examples before abstract theories.",
                    "Use simple, clear language and avoid jargon.",
                    "Provide frequent practice and feedback opportunities.",
                ]
            )
        elif "advanced" in target_audience:
            hints.extend(
                [
                    "Challenge learners with complex problems.",
                    "Encourage critical analysis and independent thinking.",
                    "Connect to current research and advanced topics.",
                ]
            )

        if any("analyze" in obj.lower() for obj in objectives):
            hints.append("Include activities that require comparison and evaluation.")
        if any("create" in obj.lower() or "design" in obj.lower() for obj in objectives):
            hints.append("Provide opportunities for creative problem-solving.")

        return hints

    def _generate_progression_guidance(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Generates guidance for learning progression."""
        target_audience = request.get("target_audience", "").lower()
        preferences = str(request.get("curriculum_preferences", ""))

        guidance = {
            "progression_type": "linear",
            "pacing_recommendations": "moderate",
            "checkpoint_frequency": "regular",
            "adaptation_triggers": [
                "Low performance on assessments",
                "Difficulty with prerequisite concepts",
                "Rapid mastery of the current level",
            ],
            "support_mechanisms": [
                "Additional practice problems",
                "Prerequisite review materials",
                "Peer collaboration opportunities",
            ],
        }

        if "spiral" in preferences:
            guidance["progression_type"] = "spiral"
        elif "mastery" in preferences:
            guidance["progression_type"] = "mastery_based"

        if "beginner" in target_audience:
            guidance["pacing_recommendations"] = "slow"
            guidance["checkpoint_frequency"] = "frequent"
        elif "advanced" in target_audience:
            guidance["pacing_recommendations"] = "fast"
            guidance["checkpoint_frequency"] = "moderate"

        return guidance

    def _assess_objective_alignment(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Assesses content alignment with learning objectives."""
        objectives = request.get("learning_objectives", [])
        content = request.get("content", {})
        alignment = {
            "overall_alignment": 0.5,
            "objective_coverage": {},
            "gaps_identified": [],
            "recommendations": [],
        }

        if not objectives:
            return alignment

        coverage_scores = []
        for i, objective in enumerate(objectives):
            score = self._calculate_objective_coverage(objective, content)
            coverage_scores.append(score)
            status = "covered" if score > 0.7 else "partial" if score > 0.4 else "missing"
            alignment["objective_coverage"][f"objective_{i+1}"] = {
                "objective": objective,
                "coverage_score": score,
                "status": status,
            }
            if score < 0.7:
                alignment["gaps_identified"].append(f"Insufficient coverage of: {objective}")

        if coverage_scores:
            alignment["overall_alignment"] = np.mean(coverage_scores)

        if alignment["overall_alignment"] < 0.7:
            alignment["recommendations"].append("Increase focus on learning objectives.")
        if alignment["gaps_identified"]:
            alignment["recommendations"].append("Address identified coverage gaps.")

        return alignment

    def _calculate_objective_coverage(self, objective: str, content: Dict[str, Any]) -> float:
        """Calculates how well content covers a specific objective."""
        content_text = str(content.get("text", "")).lower()
        objective_lower = objective.lower()
        key_terms = [word for word in objective_lower.split() if len(word) > 3]

        if not key_terms:
            return 0.0

        covered_terms = sum(1 for term in key_terms if term in content_text)
        return min(covered_terms / len(key_terms), 1.0)

    def _suggest_learning_activities(self, objective: str, domain: str) -> List[str]:
        """Suggests learning activities for a given objective."""
        activities = []
        objective_lower = objective.lower()

        if "understand" in objective_lower or "explain" in objective_lower:
            activities.extend(["Reading", "Concept mapping", "Discussion"])
        if "apply" in objective_lower or "solve" in objective_lower:
            activities.extend(["Practice problems", "Case studies", "Simulations"])
        if "analyze" in objective_lower or "evaluate" in objective_lower:
            activities.extend(["Critical analysis", "Peer review"])
        if "create" in objective_lower or "design" in objective_lower:
            activities.extend(["Project work", "Design challenges"])

        if domain.lower() == "mathematics":
            activities.extend(["Problem-solving exercises", "Mathematical proofs"])
        elif domain.lower() == "science":
            activities.extend(["Laboratory experiments", "Data analysis"])

        return activities[:3]

    def _suggest_learning_resources(self, objective: str, domain: str) -> List[str]:
        """Suggests learning resources for a given objective."""
        resources = ["Textbook chapters", "Online tutorials", "Practice worksheets"]
        if domain.lower() == "mathematics":
            resources.extend(["Mathematical software", "Problem banks"])
        elif domain.lower() == "science":
            resources.extend(["Lab manuals", "Scientific databases"])
        return resources[:4]

    def _suggest_assessment_method(self, objective: str) -> str:
        """Suggests an assessment method for a given objective."""
        objective_lower = objective.lower()
        if "understand" in objective_lower:
            return "Conceptual quiz"
        if "apply" in objective_lower:
            return "Problem-solving assignment"
        if "analyze" in objective_lower:
            return "Analysis paper"
        if "create" in objective_lower:
            return "Creative project"
        return "Comprehensive assessment"

    def _encode_content_progression_history(self) -> List[float]:
        """Encodes the history of content progression into features."""
        metrics = self.curriculum_metrics
        features = [
            metrics["average_pedagogical_coherence"],
            metrics["average_learning_progression"],
            metrics["average_objective_alignment"],
            metrics["successful_integrations"] / max(metrics["total_recommendations"], 1),
        ]

        recent_recommendations = self.curriculum_history[-10:]
        strategy_performance = np.zeros(len(self.strategies))
        if recent_recommendations:
            for record in recent_recommendations:
                for i, strategy in enumerate(self.strategies):
                    if strategy.name == record["strategy"]:
                        strategy_performance[i] += 1
                        break
            strategy_performance /= len(recent_recommendations)

        features.extend(strategy_performance.tolist())
        return features

    def _encode_pedagogical_context(self, state: Dict[str, Any]) -> List[float]:
        """Encodes pedagogical context from the environment state."""
        prefs = state.get("curriculum_preferences", {})
        assess = state.get("assessment_requirements", {})
        styles = state.get("learning_style_preferences", [])

        features = [
            1.0 if prefs.get("spiral_curriculum") else 0.0,
            1.0 if prefs.get("mastery_based") else 0.0,
            1.0 if prefs.get("adaptive_difficulty") else 0.0,
            assess.get("formative_weight", 0.5),
            assess.get("summative_weight", 0.5),
        ]

        style_encoding = [0.0] * 4  # visual, auditory, kinesthetic, reading
        for style in styles:
            style_lower = style.lower()
            if "visual" in style_lower:
                style_encoding[0] = 1.0
            elif "auditory" in style_lower:
                style_encoding[1] = 1.0
            elif "kinesthetic" in style_lower:
                style_encoding[2] = 1.0
            elif "reading" in style_lower:
                style_encoding[3] = 1.0
        features.extend(style_encoding)
        return features

    def _encode_domain_context(self, domain: str) -> List[float]:
        """Encodes the domain context using one-hot encoding."""
        domain_map = {
            "mathematics": 0,
            "science": 1,
            "technology": 2,
            "reading": 3,
            "engineering": 4,
            "arts": 5,
        }
        encoding = [0.0] * len(domain_map)
        if domain.lower() in domain_map:
            encoding[domain_map[domain.lower()]] = 1.0
        return encoding

    def _encode_difficulty_context(self, difficulty: str) -> List[float]:
        """Encodes the difficulty level using one-hot encoding."""
        difficulty_map = {
            "elementary": 0,
            "high_school": 1,
            "undergraduate": 2,
            "graduate": 3,
        }
        encoding = [0.0] * len(difficulty_map)
        key = difficulty.lower().replace(" ", "_")
        if key in difficulty_map:
            encoding[difficulty_map[key]] = 1.0
        return encoding

    def _encode_generator_context(self, strategy: Dict[str, Any]) -> List[float]:
        """Encodes the generator strategy context."""
        name = strategy.get("strategy", "")
        types = [
            "step_by_step",
            "concept_based",
            "problem_solving",
            "creative",
            "structured",
            "adaptive",
            "multi_perspective",
            "real_world",
        ]
        encoding = [1.0 if stype in name else 0.0 for stype in types]
        encoding.append(strategy.get("confidence", 0.0))
        return encoding

    def _encode_validator_context(self, feedback: Dict[str, Any]) -> List[float]:
        """Encodes the validator feedback context."""
        return [
            feedback.get("quality_prediction", 0.5),
            feedback.get("confidence", 0.5),
            1.0 if feedback.get("passes_threshold") else 0.0,
            len(feedback.get("areas_for_improvement", [])) / 5.0,
        ]

    def _encode_coordination_context(self, context: Dict[str, Any]) -> List[float]:
        """Encodes the coordination context."""
        return [
            context.get("coordination_quality", 0.5),
            context.get("consensus_level", 0.5),
            1.0 if context.get("coordination_success") else 0.0,
            context.get("agent_agreement", 0.5),
        ]

    def _update_curriculum_metrics(
        self,
        pedagogical_coherence: float,
        learning_progression: float,
        objective_alignment: float,
        integration_success: bool,
        strategy_name: str,
    ) -> None:
        """Updates the agent's overall curriculum performance metrics."""
        metrics = self.curriculum_metrics
        metrics["total_recommendations"] += 1
        if integration_success:
            metrics["successful_integrations"] += 1

        alpha = 0.1
        metrics["average_pedagogical_coherence"] = (1 - alpha) * metrics[
            "average_pedagogical_coherence"
        ] + alpha * pedagogical_coherence
        metrics["average_learning_progression"] = (1 - alpha) * metrics[
            "average_learning_progression"
        ] + alpha * learning_progression
        metrics["average_objective_alignment"] = (1 - alpha) * metrics[
            "average_objective_alignment"
        ] + alpha * objective_alignment

        metrics["strategy_usage"][strategy_name] += 1

    def _summarize_request(self, request: Dict[str, Any]) -> str:
        """Creates a concise summary of a request for logging purposes."""
        domain = request.get("domain", "N/A")
        difficulty = request.get("difficulty_level", "N/A")
        num_objectives = len(request.get("learning_objectives", []))
        return f"domain={domain}, difficulty={difficulty}, objectives={num_objectives}"

    def get_agent_performance_summary(self) -> Dict[str, Any]:
        """Returns a comprehensive performance summary for the agent."""
        strategy_summaries = [s.get_performance_summary() for s in self.strategies]
        strategy_summaries.sort(key=lambda x: x["overall_score"], reverse=True)

        return {
            "agent_id": self.agent_id,
            "curriculum_metrics": self.curriculum_metrics.copy(),
            "strategy_performance": strategy_summaries,
            "recent_curriculum_history": self.curriculum_history[-20:],
            "learning_progress": {
                "total_episodes": self.episode_count,
                "training_steps": self.training_step,
                "current_epsilon": self.epsilon,
                "learning_metrics": self.learning_metrics.get_summary(),
            },
        }


class LearningAnalyzer:
    """Analyzes learning characteristics for curriculum planning."""

    def encode_learning_objectives(self, objectives: List[str]) -> List[float]:
        """Encodes learning objectives into numerical features."""
        if not objectives:
            return [0.0] * 8

        avg_length = sum(len(obj) for obj in objectives) / len(objectives)
        bloom_levels = {
            "remember": ["remember", "recall", "identify", "list"],
            "understand": ["understand", "explain", "describe", "summarize"],
            "apply": ["apply", "use", "solve", "demonstrate"],
            "analyze": ["analyze", "compare", "examine", "investigate"],
            "evaluate": ["evaluate", "assess", "judge", "critique"],
            "create": ["create", "design", "develop", "construct"],
        }

        bloom_scores = [0.0] * 6
        for i, (level, keywords) in enumerate(bloom_levels.items()):
            for obj in objectives:
                obj_lower = obj.lower()
                if any(keyword in obj_lower for keyword in keywords):
                    bloom_scores[i] += 1

        bloom_scores = [score / len(objectives) for score in bloom_scores]
        features = [len(objectives) / 10.0, avg_length / 100.0]
        features.extend(bloom_scores)
        return features

    def encode_target_audience(self, audience: str) -> List[float]:
        """Encodes the target audience into numerical features."""
        audience_lower = audience.lower()
        return [
            1.0 if "student" in audience_lower else 0.0,
            1.0 if "teacher" in audience_lower else 0.0,
            1.0 if "beginner" in audience_lower else 0.0,
            1.0 if "intermediate" in audience_lower else 0.0,
            1.0 if "advanced" in audience_lower else 0.0,
            1.0 if "professional" in audience_lower else 0.0,
        ]

    def analyze_content_characteristics(self, content: Dict[str, Any]) -> List[float]:
        """Analyzes content characteristics for curriculum planning."""
        text = str(content.get("text", ""))
        return [
            len(text) / 1000.0,
            text.count("?") / max(len(text.split()), 1),
            1.0 if "example" in text.lower() else 0.0,
            1.0 if "step" in text.lower() else 0.0,
            content.get("complexity_score", 0.5),
            content.get("engagement_score", 0.5),
        ]


class ProgressionModeler:
    """Models learning progression for curriculum planning."""

    def model_difficulty_progression(self, current_level: str, target_level: str) -> Dict[str, Any]:
        """Models the progression from a current to a target difficulty level."""
        levels = ["elementary", "high_school", "undergraduate", "graduate"]
        try:
            current_idx = levels.index(current_level.lower().replace(" ", "_"))
            target_idx = levels.index(target_level.lower().replace(" ", "_"))
        except ValueError:
            return {
                "progression_steps": 1,
                "intermediate_levels": [],
                "estimated_duration": "medium",
            }

        progression_steps = abs(target_idx - current_idx)
        intermediate_levels = []
        if progression_steps > 1:
            if current_idx < target_idx:
                intermediate_levels = levels[current_idx + 1 : target_idx]
            else:
                intermediate_levels = levels[target_idx + 1 : current_idx][::-1]

        duration = (
            "short" if progression_steps <= 1 else "medium" if progression_steps <= 2 else "long"
        )
        return {
            "progression_steps": progression_steps,
            "intermediate_levels": intermediate_levels,
            "estimated_duration": duration,
        }


class ObjectiveAligner:
    """Aligns content with learning objectives."""

    def assess_alignment(self, content: Dict[str, Any], objectives: List[str]) -> float:
        """Assess how well content aligns with a list of learning objectives."""
        if not objectives:
            return 0.5

        content_text = str(content.get("text", "")).lower()
        alignment_scores = []
        for objective in objectives:
            objective_words = [word for word in objective.lower().split() if len(word) > 3]
            if not objective_words:
                continue
            matches = sum(1 for word in objective_words if word in content_text)
            alignment_scores.append(matches / len(objective_words))

        return np.mean(alignment_scores) if alignment_scores else 0.5
