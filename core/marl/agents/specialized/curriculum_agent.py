"""
Curriculum RL Agent

This module implements the Curriculum RL Agent for multi-agent reinforcement learning
coordination. The Curriculum Agent is responsible for ensuring pedagogical coherence,
learning progression, and curriculum-based improvements in content generation.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from ...config import CurriculumAgentConfig
from ...exceptions import AgentFailureError
from ...logging_config import get_marl_logger
from ..base_agent import ActionSpace, BaseRLAgent
from ..experience import Experience

logger = logging.getLogger(__name__)


class CurriculumStrategy:
    """Represents a curriculum strategy with parameters."""

    def __init__(self, name: str, description: str, parameters: Dict[str, Any]):
        """
        Initialize curriculum strategy.

        Args:
            name: Strategy name
            description: Strategy description
            parameters: Strategy-specific parameters
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
        """Update strategy performance metrics."""
        self.usage_count += 1

        # Update running averages
        alpha = 0.1  # Learning rate for running average
        self.pedagogical_coherence = (
            1 - alpha
        ) * self.pedagogical_coherence + alpha * pedagogical_coherence
        self.learning_progression = (
            1 - alpha
        ) * self.learning_progression + alpha * learning_progression
        self.objective_alignment = (
            1 - alpha
        ) * self.objective_alignment + alpha * objective_alignment

        # Update integration success rate
        self.integration_success_rate = (
            1 - alpha
        ) * self.integration_success_rate + alpha * (
            1.0 if integration_success else 0.0
        )

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for this strategy."""
        return {
            "name": self.name,
            "usage_count": self.usage_count,
            "pedagogical_coherence": self.pedagogical_coherence,
            "learning_progression": self.learning_progression,
            "objective_alignment": self.objective_alignment,
            "integration_success_rate": self.integration_success_rate,
            "overall_score": (
                self.pedagogical_coherence
                + self.learning_progression
                + self.objective_alignment
            )
            / 3,
        }


class CurriculumRLAgent(BaseRLAgent):
    """
    Curriculum RL Agent for pedagogical coherence and learning progression.

    This agent learns to ensure pedagogical coherence, optimize learning progression,
    and align content with learning objectives through reinforcement learning.
    """

    def __init__(self, config: CurriculumAgentConfig):
        """
        Initialize Curriculum RL Agent.

        Args:
            config: Curriculum agent configuration
        """
        # Initialize strategies first (needed for action space)
        self.strategies = self._initialize_strategies()

        super().__init__("curriculum", config)
        self.config = config
        self.logger = get_marl_logger("curriculum_agent")

        # Additional initialization
        self.curriculum_history = []

        # Performance tracking
        self.curriculum_metrics = {
            "total_recommendations": 0,
            "successful_integrations": 0,
            "average_pedagogical_coherence": 0.0,
            "average_learning_progression": 0.0,
            "average_objective_alignment": 0.0,
            "strategy_usage": {strategy.name: 0 for strategy in self.strategies},
        }

        # Curriculum analysis components
        self.learning_analyzer = LearningAnalyzer()
        self.progression_modeler = ProgressionModeler()
        self.objective_aligner = ObjectiveAligner()

        self.logger.log_agent_action(
            self.agent_id,
            "curriculum_initialized",
            1.0,
            "Strategies: %d, Config: %s"
            % (len(self.strategies), type(config).__name__),
        )

    def _initialize_strategies(self) -> List[CurriculumStrategy]:
        """Initialize available curriculum strategies."""
        strategies = [
            CurriculumStrategy(
                "linear_progression",
                "Organize content in linear, sequential progression",
                {
                    "sequence_strength": 0.9,
                    "prerequisite_enforcement": 0.8,
                    "difficulty_gradient": 0.7,
                    "concept_ordering": "sequential",
                },
            ),
            CurriculumStrategy(
                "spiral_curriculum",
                "Revisit concepts with increasing complexity",
                {
                    "spiral_depth": 0.8,
                    "concept_reinforcement": 0.9,
                    "complexity_layering": 0.8,
                    "revisit_frequency": 0.7,
                },
            ),
            CurriculumStrategy(
                "mastery_based_progression",
                "Ensure mastery before advancing to next concepts",
                {
                    "mastery_threshold": 0.85,
                    "competency_validation": 0.9,
                    "adaptive_pacing": 0.8,
                    "skill_verification": 0.8,
                },
            ),
            CurriculumStrategy(
                "adaptive_difficulty_adjustment",
                "Dynamically adjust difficulty based on learner progress",
                {
                    "adaptivity_sensitivity": 0.8,
                    "difficulty_calibration": 0.9,
                    "progress_monitoring": 0.8,
                    "adjustment_speed": 0.7,
                },
            ),
            CurriculumStrategy(
                "prerequisite_reinforcement",
                "Strengthen prerequisite knowledge before new concepts",
                {
                    "prerequisite_depth": 0.8,
                    "knowledge_gap_detection": 0.9,
                    "reinforcement_intensity": 0.7,
                    "foundation_strength": 0.8,
                },
            ),
            CurriculumStrategy(
                "concept_scaffolding",
                "Provide structured support for complex concepts",
                {
                    "scaffolding_levels": 0.8,
                    "support_gradation": 0.9,
                    "independence_building": 0.7,
                    "guidance_fading": 0.8,
                },
            ),
            CurriculumStrategy(
                "multi_modal_learning",
                "Integrate multiple learning modalities and approaches",
                {
                    "modality_variety": 0.8,
                    "learning_style_accommodation": 0.9,
                    "engagement_diversity": 0.8,
                    "accessibility_support": 0.7,
                },
            ),
            CurriculumStrategy(
                "personalized_pathway",
                "Create individualized learning pathways",
                {
                    "personalization_depth": 0.9,
                    "learner_profile_integration": 0.8,
                    "pathway_flexibility": 0.8,
                    "individual_optimization": 0.9,
                },
            ),
        ]

        return strategies

    def get_state_representation(self, environment_state: Dict[str, Any]) -> np.ndarray:
        """
        Convert environment state to curriculum-specific representation.

        Args:
            environment_state: Raw environment state

        Returns:
            Numpy array representing the state for the curriculum agent
        """
        try:
            features = []

            # Learning objective features
            learning_objectives = environment_state.get("learning_objectives", [])
            features.extend(
                self.learning_analyzer.encode_learning_objectives(learning_objectives)
            )

            # Student/audience features
            target_audience = environment_state.get("target_audience", "")
            features.extend(
                self.learning_analyzer.encode_target_audience(target_audience)
            )

            # Content progression features
            features.extend(self._encode_content_progression_history())

            # Pedagogical context features
            features.extend(self._encode_pedagogical_context(environment_state))

            # Domain and difficulty context
            domain = environment_state.get("domain", "")
            features.extend(self._encode_domain_context(domain))

            difficulty_level = environment_state.get("difficulty_level", "")
            features.extend(self._encode_difficulty_context(difficulty_level))

            # Content characteristics
            content = environment_state.get("content", {})
            features.extend(
                self.learning_analyzer.analyze_content_characteristics(content)
            )

            # Generator and validator context
            generator_strategy = environment_state.get("generator_strategy", {})
            features.extend(self._encode_generator_context(generator_strategy))

            validator_feedback = environment_state.get("validator_feedback", {})
            features.extend(self._encode_validator_context(validator_feedback))

            # Coordination context
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
            # Return default state representation
            return np.zeros(100, dtype=np.float32)

    def get_action_space(self) -> ActionSpace:
        """
        Define curriculum strategy action space.

        Returns:
            ActionSpace defining available curriculum strategies
        """
        strategy_names = [strategy.name for strategy in self.strategies]
        return ActionSpace(strategy_names)

    def calculate_reward(
        self, state: np.ndarray, action: int, result: Dict[str, Any]
    ) -> float:
        """
        Calculate curriculum-specific reward for the given state-action-result.

        Args:
            state: State representation
            action: Action taken (strategy index)
            result: Result of the curriculum recommendation

        Returns:
            Reward value for the curriculum agent
        """
        try:
            # Extract performance metrics from result
            pedagogical_coherence = result.get("pedagogical_coherence_score", 0.0)
            learning_progression = result.get("learning_progression_score", 0.0)
            objective_alignment = result.get("objective_alignment_score", 0.0)

            # Multi-objective curriculum reward
            coherence_weight = self.config.pedagogical_coherence_weight
            progression_weight = self.config.learning_progression_weight
            alignment_weight = self.config.objective_alignment_weight

            base_reward = (
                coherence_weight * pedagogical_coherence
                + progression_weight * learning_progression
                + alignment_weight * objective_alignment
            )

            # Bonus for successful curriculum integration
            integration_success = result.get("curriculum_integration_success", False)
            if integration_success:
                base_reward += self.config.integration_bonus

            # Bonus for exceeding thresholds
            if pedagogical_coherence > self.config.coherence_threshold:
                base_reward += 0.1

            if learning_progression > self.config.progression_threshold:
                base_reward += 0.1

            # Penalty for poor alignment
            if objective_alignment < 0.5:
                base_reward -= 0.2

            # Coordination success bonus
            coordination_success = result.get("coordination_success", False)
            if coordination_success:
                base_reward += 0.1

            # Update strategy performance
            strategy = self.strategies[action]
            strategy.update_performance(
                pedagogical_coherence,
                learning_progression,
                objective_alignment,
                integration_success,
            )

            # Update agent metrics
            self._update_curriculum_metrics(
                pedagogical_coherence,
                learning_progression,
                objective_alignment,
                integration_success,
                strategy.name,
            )

            return float(
                np.clip(base_reward, -1.0, 2.0)
            )  # Clip reward to reasonable range

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

    def suggest_curriculum_improvements(
        self, request: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Suggest curriculum-based improvements for content generation.

        Args:
            request: Content generation request with context

        Returns:
            Dictionary with curriculum improvement suggestions
        """
        try:
            # Get state representation
            state = self.get_state_representation(request)

            # Select curriculum strategy using RL policy
            action_index = self.select_action(state, training=True)
            selected_strategy = self.strategies[action_index]

            # Generate curriculum improvements
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
            }

            # Calculate confidence
            confidence = self.get_action_confidence(state, action_index)
            improvements["confidence"] = confidence

            # Record curriculum recommendation
            curriculum_record = {
                "strategy": selected_strategy.name,
                "timestamp": time.time(),
                "confidence": confidence,
                "request_summary": self._summarize_request(request),
            }

            self.curriculum_history.append(curriculum_record)

            # Update metrics
            self.curriculum_metrics["total_recommendations"] += 1

            # Keep history manageable
            if len(self.curriculum_history) > 1000:
                self.curriculum_history = self.curriculum_history[-1000:]

            self.logger.log_agent_action(
                self.agent_id,
                selected_strategy.name,
                confidence,
                "Usage: %d, Success: %.2f"
                % (
                    selected_strategy.usage_count,
                    selected_strategy.integration_success_rate,
                ),
            )

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

    def _suggest_difficulty_adjustments(
        self, request: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Suggest difficulty adjustments based on curriculum analysis."""
        current_difficulty = request.get("difficulty_level", "medium")
        target_audience = request.get("target_audience", "")
        learning_objectives = request.get("learning_objectives", [])

        adjustments = {
            "current_difficulty": current_difficulty,
            "recommended_difficulty": current_difficulty,
            "adjustment_rationale": "",
            "scaffolding_needed": False,
            "prerequisite_review": False,
        }

        # Analyze if difficulty adjustment is needed
        if "beginner" in target_audience.lower():
            if current_difficulty in ["high", "advanced"]:
                adjustments["recommended_difficulty"] = "medium"
                adjustments["adjustment_rationale"] = (
                    "Reduce difficulty for beginner audience"
                )
                adjustments["scaffolding_needed"] = True

        elif "advanced" in target_audience.lower():
            if current_difficulty in ["low", "elementary"]:
                adjustments["recommended_difficulty"] = "high"
                adjustments["adjustment_rationale"] = (
                    "Increase difficulty for advanced audience"
                )

        # Check if prerequisites are needed
        complex_objectives = [
            obj for obj in learning_objectives if len(obj.split()) > 5
        ]
        if len(complex_objectives) > len(learning_objectives) / 2:
            adjustments["prerequisite_review"] = True

        return adjustments

    def _identify_prerequisites(self, request: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify prerequisite knowledge and skills."""
        domain = request.get("domain", "")
        topic = request.get("topic", "")
        difficulty_level = request.get("difficulty_level", "")

        prerequisites = []

        # Domain-specific prerequisites
        if domain.lower() == "mathematics":
            if "algebra" in topic.lower():
                prerequisites.append(
                    {
                        "concept": "Basic arithmetic operations",
                        "importance": "high",
                        "description": "Addition, subtraction, multiplication, division",
                    }
                )
                prerequisites.append(
                    {
                        "concept": "Number properties",
                        "importance": "medium",
                        "description": "Understanding of integers, fractions, decimals",
                    }
                )

            if "calculus" in topic.lower():
                prerequisites.append(
                    {
                        "concept": "Algebra fundamentals",
                        "importance": "critical",
                        "description": "Solving equations, working with functions",
                    }
                )
                prerequisites.append(
                    {
                        "concept": "Trigonometry basics",
                        "importance": "high",
                        "description": "Trigonometric functions and identities",
                    }
                )

        elif domain.lower() == "science":
            if "physics" in topic.lower():
                prerequisites.append(
                    {
                        "concept": "Mathematical foundations",
                        "importance": "high",
                        "description": "Algebra and basic calculus",
                    }
                )
                prerequisites.append(
                    {
                        "concept": "Scientific method",
                        "importance": "medium",
                        "description": "Understanding of hypothesis, experimentation",
                    }
                )

        # Difficulty-based prerequisites
        if difficulty_level in ["high", "advanced"]:
            prerequisites.append(
                {
                    "concept": "Critical thinking skills",
                    "importance": "high",
                    "description": "Ability to analyze and synthesize information",
                }
            )

        return prerequisites

    def _generate_learning_pathway(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a structured learning pathway."""
        learning_objectives = request.get("learning_objectives", [])
        domain = request.get("domain", "")
        difficulty_level = request.get("difficulty_level", "")

        pathway = {
            "pathway_type": "sequential",
            "total_steps": len(learning_objectives) if learning_objectives else 3,
            "estimated_duration": "medium",
            "steps": [],
            "milestones": [],
            "assessment_points": [],
        }

        # Generate pathway steps
        if learning_objectives:
            for i, objective in enumerate(learning_objectives):
                step = {
                    "step_number": i + 1,
                    "objective": objective,
                    "activities": self._suggest_learning_activities(objective, domain),
                    "resources": self._suggest_learning_resources(objective, domain),
                    "assessment": self._suggest_assessment_method(objective),
                }
                pathway["steps"].append(step)

                # Add milestones at key points
                if (i + 1) % 3 == 0 or i == len(learning_objectives) - 1:
                    pathway["milestones"].append(
                        {
                            "milestone_number": len(pathway["milestones"]) + 1,
                            "description": "Complete objectives 1-%d" % (i + 1),
                            "success_criteria": "Demonstrate understanding of covered concepts",
                        }
                    )

        else:
            # Default pathway structure
            pathway["steps"] = [
                {
                    "step_number": 1,
                    "objective": "Introduction and foundation",
                    "activities": ["Concept introduction", "Basic examples"],
                    "resources": ["Textbook readings", "Video tutorials"],
                    "assessment": "Formative quiz",
                },
                {
                    "step_number": 2,
                    "objective": "Practice and application",
                    "activities": ["Guided practice", "Problem solving"],
                    "resources": ["Practice problems", "Interactive exercises"],
                    "assessment": "Practice assignments",
                },
                {
                    "step_number": 3,
                    "objective": "Mastery and extension",
                    "activities": ["Independent work", "Advanced applications"],
                    "resources": ["Challenge problems", "Real-world examples"],
                    "assessment": "Comprehensive evaluation",
                },
            ]

        return pathway

    def _generate_pedagogical_hints(self, request: Dict[str, Any]) -> List[str]:
        """Generate pedagogical hints for content improvement."""
        hints = []

        domain = request.get("domain", "")
        target_audience = request.get("target_audience", "")
        learning_objectives = request.get("learning_objectives", [])

        # Domain-specific hints
        if domain.lower() == "mathematics":
            hints.append(
                "Use visual representations and diagrams to illustrate abstract concepts"
            )
            hints.append(
                "Provide step-by-step worked examples before independent practice"
            )
            hints.append("Connect mathematical concepts to real-world applications")

        elif domain.lower() == "science":
            hints.append(
                "Incorporate hands-on experiments or demonstrations when possible"
            )
            hints.append("Use analogies to explain complex scientific phenomena")
            hints.append("Emphasize the scientific method and evidence-based reasoning")

        # Audience-specific hints
        if "beginner" in target_audience.lower():
            hints.append(
                "Start with concrete examples before introducing abstract concepts"
            )
            hints.append("Use simple, clear language and avoid jargon")
            hints.append("Provide frequent opportunities for practice and feedback")

        elif "advanced" in target_audience.lower():
            hints.append("Challenge learners with complex problems and scenarios")
            hints.append("Encourage critical analysis and independent thinking")
            hints.append("Connect to advanced topics and current research")

        # Objective-based hints
        if learning_objectives:
            if any("analyze" in obj.lower() for obj in learning_objectives):
                hints.append(
                    "Include activities that require comparison and evaluation"
                )

            if any(
                "create" in obj.lower() or "design" in obj.lower()
                for obj in learning_objectives
            ):
                hints.append(
                    "Provide opportunities for creative problem-solving and design"
                )

        return hints

    def _generate_progression_guidance(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Generate guidance for learning progression."""
        guidance = {
            "progression_type": "linear",
            "pacing_recommendations": "moderate",
            "checkpoint_frequency": "regular",
            "adaptation_triggers": [],
            "support_mechanisms": [],
        }

        difficulty_level = request.get("difficulty_level", "")
        target_audience = request.get("target_audience", "")

        # Determine progression type
        if "spiral" in str(request.get("curriculum_preferences", "")):
            guidance["progression_type"] = "spiral"
        elif "mastery" in str(request.get("curriculum_preferences", "")):
            guidance["progression_type"] = "mastery_based"

        # Pacing recommendations
        if "beginner" in target_audience.lower():
            guidance["pacing_recommendations"] = "slow"
            guidance["checkpoint_frequency"] = "frequent"
        elif "advanced" in target_audience.lower():
            guidance["pacing_recommendations"] = "fast"
            guidance["checkpoint_frequency"] = "moderate"

        # Adaptation triggers
        guidance["adaptation_triggers"] = [
            "Low performance on assessments",
            "Difficulty with prerequisite concepts",
            "Rapid mastery of current level",
            "Learner feedback indicating confusion or boredom",
        ]

        # Support mechanisms
        guidance["support_mechanisms"] = [
            "Additional practice problems",
            "Prerequisite review materials",
            "Peer collaboration opportunities",
            "Instructor feedback and guidance",
        ]

        return guidance

    def _assess_objective_alignment(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Assess alignment with learning objectives."""
        learning_objectives = request.get("learning_objectives", [])
        content = request.get("content", {})

        alignment = {
            "overall_alignment": 0.7,  # Default moderate alignment
            "objective_coverage": {},
            "gaps_identified": [],
            "recommendations": [],
        }

        if learning_objectives:
            # Analyze coverage of each objective
            for i, objective in enumerate(learning_objectives):
                coverage_score = self._calculate_objective_coverage(objective, content)
                alignment["objective_coverage"]["objective_%d" % (i + 1)] = {
                    "objective": objective,
                    "coverage_score": coverage_score,
                    "status": "covered"
                    if coverage_score > 0.7
                    else "partial"
                    if coverage_score > 0.4
                    else "missing",
                }

                if coverage_score < 0.7:
                    alignment["gaps_identified"].append(
                        "Insufficient coverage of: %s" % objective
                    )

            # Calculate overall alignment
            coverage_scores = [
                obj["coverage_score"]
                for obj in alignment["objective_coverage"].values()
            ]
            alignment["overall_alignment"] = (
                np.mean(coverage_scores) if coverage_scores else 0.5
            )

        # Generate recommendations
        if alignment["overall_alignment"] < 0.7:
            alignment["recommendations"].append(
                "Increase focus on stated learning objectives"
            )
        if len(alignment["gaps_identified"]) > 0:
            alignment["recommendations"].append("Address identified coverage gaps")

        return alignment

    def _calculate_objective_coverage(
        self, objective: str, content: Dict[str, Any]
    ) -> float:
        """Calculate how well content covers a specific objective."""
        # Simple keyword-based coverage analysis
        content_text = str(content.get("text", "")).lower()
        objective_lower = objective.lower()

        # Extract key terms from objective
        key_terms = [word for word in objective_lower.split() if len(word) > 3]

        # Calculate coverage based on term presence
        covered_terms = sum(1 for term in key_terms if term in content_text)
        coverage_score = covered_terms / max(len(key_terms), 1)

        return min(coverage_score, 1.0)

    def _suggest_learning_activities(self, objective: str, domain: str) -> List[str]:
        """Suggest learning activities for an objective."""
        activities = []

        objective_lower = objective.lower()

        # Activity suggestions based on objective verbs
        if "understand" in objective_lower or "explain" in objective_lower:
            activities.extend(
                ["Reading comprehension", "Concept mapping", "Discussion"]
            )

        if "apply" in objective_lower or "solve" in objective_lower:
            activities.extend(["Practice problems", "Case studies", "Simulations"])

        if "analyze" in objective_lower or "evaluate" in objective_lower:
            activities.extend(
                ["Critical analysis", "Comparison exercises", "Peer review"]
            )

        if "create" in objective_lower or "design" in objective_lower:
            activities.extend(
                ["Project work", "Creative assignments", "Design challenges"]
            )

        # Domain-specific activities
        if domain.lower() == "mathematics":
            activities.extend(
                [
                    "Problem-solving exercises",
                    "Mathematical proofs",
                    "Graphing activities",
                ]
            )
        elif domain.lower() == "science":
            activities.extend(
                ["Laboratory experiments", "Data analysis", "Scientific inquiry"]
            )

        return activities[:3]  # Limit to top 3 suggestions

    def _suggest_learning_resources(self, objective: str, domain: str) -> List[str]:
        """Suggest learning resources for an objective."""
        resources = ["Textbook chapters", "Online tutorials", "Practice worksheets"]

        # Domain-specific resources
        if domain.lower() == "mathematics":
            resources.extend(
                ["Mathematical software", "Graphing calculators", "Problem banks"]
            )
        elif domain.lower() == "science":
            resources.extend(
                ["Laboratory manuals", "Scientific databases", "Simulation software"]
            )

        return resources[:4]  # Limit to top 4 suggestions

    def _suggest_assessment_method(self, objective: str) -> str:
        """Suggest appropriate assessment method for an objective."""
        objective_lower = objective.lower()

        if "understand" in objective_lower or "explain" in objective_lower:
            return "Conceptual quiz or explanation task"
        elif "apply" in objective_lower or "solve" in objective_lower:
            return "Problem-solving assignment"
        elif "analyze" in objective_lower or "evaluate" in objective_lower:
            return "Analysis paper or critical evaluation"
        elif "create" in objective_lower or "design" in objective_lower:
            return "Creative project or design portfolio"
        else:
            return "Comprehensive assessment"

    def _encode_content_progression_history(self) -> List[float]:
        """Encode content progression history into features."""
        features = []

        # Overall curriculum metrics
        features.append(self.curriculum_metrics["average_pedagogical_coherence"])
        features.append(self.curriculum_metrics["average_learning_progression"])
        features.append(self.curriculum_metrics["average_objective_alignment"])

        # Success rate
        success_rate = self.curriculum_metrics["successful_integrations"] / max(
            self.curriculum_metrics["total_recommendations"], 1
        )
        features.append(success_rate)

        # Recent strategy performance (last 10 recommendations)
        recent_recommendations = (
            self.curriculum_history[-10:] if self.curriculum_history else []
        )
        strategy_performance = np.zeros(len(self.strategies))

        for record in recent_recommendations:
            strategy_name = record["strategy"]
            for i, strategy in enumerate(self.strategies):
                if strategy.name == strategy_name:
                    strategy_performance[i] += 1
                    break

        # Normalize by number of recent recommendations
        if recent_recommendations:
            strategy_performance /= len(recent_recommendations)

        features.extend(strategy_performance.tolist())

        return features

    def _encode_pedagogical_context(
        self, environment_state: Dict[str, Any]
    ) -> List[float]:
        """Encode pedagogical context into features."""
        features = []

        # Extract pedagogical information
        curriculum_preferences = environment_state.get("curriculum_preferences", {})
        assessment_requirements = environment_state.get("assessment_requirements", {})
        learning_style_preferences = environment_state.get(
            "learning_style_preferences", []
        )

        # Curriculum preference encoding
        features.append(
            1.0 if curriculum_preferences.get("spiral_curriculum", False) else 0.0
        )
        features.append(
            1.0 if curriculum_preferences.get("mastery_based", False) else 0.0
        )
        features.append(
            1.0 if curriculum_preferences.get("adaptive_difficulty", False) else 0.0
        )

        # Assessment requirement encoding
        features.append(assessment_requirements.get("formative_weight", 0.5))
        features.append(assessment_requirements.get("summative_weight", 0.5))

        # Learning style encoding
        style_encoding = [0.0] * 4  # visual, auditory, kinesthetic, reading
        for style in learning_style_preferences:
            if "visual" in style.lower():
                style_encoding[0] = 1.0
            elif "auditory" in style.lower():
                style_encoding[1] = 1.0
            elif "kinesthetic" in style.lower():
                style_encoding[2] = 1.0
            elif "reading" in style.lower():
                style_encoding[3] = 1.0

        features.extend(style_encoding)

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

    def _encode_generator_context(
        self, generator_strategy: Dict[str, Any]
    ) -> List[float]:
        """Encode generator strategy context into features."""
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

    def _encode_validator_context(
        self, validator_feedback: Dict[str, Any]
    ) -> List[float]:
        """Encode validator feedback context into features."""
        features = [
            validator_feedback.get("quality_prediction", 0.5),
            validator_feedback.get("confidence", 0.5),
            1.0 if validator_feedback.get("passes_threshold", False) else 0.0,
            len(validator_feedback.get("areas_for_improvement", []))
            / 5.0,  # Normalized
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

    def _update_curriculum_metrics(
        self,
        pedagogical_coherence: float,
        learning_progression: float,
        objective_alignment: float,
        integration_success: bool,
        strategy_name: str,
    ) -> None:
        """Update curriculum performance metrics."""
        self.curriculum_metrics["total_recommendations"] += 1

        if integration_success:
            self.curriculum_metrics["successful_integrations"] += 1

        # Update running averages
        alpha = 0.1
        self.curriculum_metrics["average_pedagogical_coherence"] = (
            1 - alpha
        ) * self.curriculum_metrics[
            "average_pedagogical_coherence"
        ] + alpha * pedagogical_coherence

        self.curriculum_metrics["average_learning_progression"] = (
            1 - alpha
        ) * self.curriculum_metrics[
            "average_learning_progression"
        ] + alpha * learning_progression

        self.curriculum_metrics["average_objective_alignment"] = (
            1 - alpha
        ) * self.curriculum_metrics[
            "average_objective_alignment"
        ] + alpha * objective_alignment

        # Update strategy usage
        self.curriculum_metrics["strategy_usage"][strategy_name] += 1

    def _summarize_request(self, request: Dict[str, Any]) -> str:
        """Create a summary of the request for logging."""
        domain = request.get("domain", "unknown")
        difficulty = request.get("difficulty_level", "unknown")
        objectives_count = len(request.get("learning_objectives", []))

        return "domain=%s, difficulty=%s, objectives=%d" % (
            domain,
            difficulty,
            objectives_count,
        )

    def get_agent_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary for the curriculum agent."""
        strategy_summaries = [
            strategy.get_performance_summary() for strategy in self.strategies
        ]

        # Sort strategies by overall performance
        strategy_summaries.sort(key=lambda x: x["overall_score"], reverse=True)

        return {
            "agent_id": self.agent_id,
            "curriculum_metrics": self.curriculum_metrics.copy(),
            "strategy_performance": strategy_summaries,
            "recent_curriculum_history": self.curriculum_history[-20:]
            if self.curriculum_history
            else [],
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
        """Encode learning objectives into numerical features."""
        if not objectives:
            return [0.0] * 8

        # Objective characteristics
        total_length = sum(len(obj) for obj in objectives)
        avg_length = total_length / len(objectives)

        # Bloom's taxonomy level analysis
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
                # Check if any keyword appears as a word (not just substring)
                obj_words = obj_lower.split()
                if any(
                    keyword in obj_words or any(keyword in word for word in obj_words)
                    for keyword in keywords
                ):
                    bloom_scores[i] += 1

        # Normalize by number of objectives
        bloom_scores = [score / len(objectives) for score in bloom_scores]

        features = [
            len(objectives) / 10.0,  # Number of objectives (normalized)
            avg_length / 100.0,  # Average objective length (normalized)
        ]
        features.extend(bloom_scores)

        return features

    def encode_target_audience(self, audience: str) -> List[float]:
        """Encode target audience into numerical features."""
        audience_lower = audience.lower()

        features = [
            1.0 if "student" in audience_lower else 0.0,
            1.0 if "teacher" in audience_lower else 0.0,
            1.0 if "beginner" in audience_lower else 0.0,
            1.0 if "intermediate" in audience_lower else 0.0,
            1.0 if "advanced" in audience_lower else 0.0,
            1.0 if "professional" in audience_lower else 0.0,
        ]

        return features

    def analyze_content_characteristics(self, content: Dict[str, Any]) -> List[float]:
        """Analyze content characteristics for curriculum planning."""
        text = str(content.get("text", ""))

        features = [
            len(text) / 1000.0,  # Content length (normalized)
            text.count("?") / max(len(text.split()), 1),  # Question density
            1.0 if "example" in text.lower() else 0.0,  # Has examples
            1.0 if "step" in text.lower() else 0.0,  # Has steps
            content.get("complexity_score", 0.5),  # Content complexity
            content.get("engagement_score", 0.5),  # Content engagement
        ]

        return features


class ProgressionModeler:
    """Models learning progression for curriculum planning."""

    def model_difficulty_progression(
        self, current_level: str, target_level: str
    ) -> Dict[str, Any]:
        """Model progression from current to target difficulty level."""
        levels = ["elementary", "high_school", "undergraduate", "graduate"]

        try:
            current_idx = levels.index(current_level.lower().replace(" ", "_"))
            target_idx = levels.index(target_level.lower().replace(" ", "_"))
        except ValueError:
            # Default progression
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

        return {
            "progression_steps": progression_steps,
            "intermediate_levels": intermediate_levels,
            "estimated_duration": "short"
            if progression_steps <= 1
            else "medium"
            if progression_steps <= 2
            else "long",
        }


class ObjectiveAligner:
    """Aligns content with learning objectives."""

    def assess_alignment(self, content: Dict[str, Any], objectives: List[str]) -> float:
        """Assess how well content aligns with learning objectives."""
        if not objectives:
            return 0.5  # Default moderate alignment

        content_text = str(content.get("text", "")).lower()
        alignment_scores = []

        for objective in objectives:
            # Simple keyword-based alignment assessment
            objective_words = [
                word for word in objective.lower().split() if len(word) > 3
            ]
            matches = sum(1 for word in objective_words if word in content_text)
            alignment_score = matches / max(len(objective_words), 1)
            alignment_scores.append(alignment_score)

        return np.mean(alignment_scores) if alignment_scores else 0.5
