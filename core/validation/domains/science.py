"""
Science domain validator for STREAM content validation.

This module provides comprehensive validation for scientific content including
physics, chemistry, biology, scientific method, safety, and ethics validation.
"""

# Standard Library
import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# SynThesisAI Modules
from ..base import (
    DomainValidator,
    QualityMetrics,
    SubValidationResult,
    ValidationResult,
)
from ..config import ValidationConfig
from ..exceptions import DomainValidationError

logger = logging.getLogger(__name__)


class ScientificMethodValidator:
    """Validator for scientific method and experimental design."""

    def __init__(self):
        """Initialize scientific method validator."""
        self.scientific_method_steps = [
            "observation",
            "hypothesis",
            "prediction",
            "experiment",
            "analysis",
            "conclusion",
            "peer_review",
        ]

        self.experimental_design_elements = [
            "control_group",
            "experimental_group",
            "variables",
            "sample_size",
            "methodology",
            "data_collection",
        ]

        self.hypothesis_indicators = [
            "hypothesis",
            "hypothesize",
            "predict",
            "expect",
            "propose",
            "if...then",
            "because",
            "due to",
            "caused by",
        ]

        self.observation_indicators = [
            "observe",
            "notice",
            "see",
            "measure",
            "record",
            "data",
            "evidence",
            "result",
            "finding",
        ]

    def validate_scientific_method(
        self, content: Dict[str, Any]
    ) -> SubValidationResult:
        """
        Validate scientific method application in content.

        Args:
            content: Scientific content to validate

        Returns:
            SubValidationResult with scientific method validation details
        """
        try:
            text = f"{content.get('problem', '')} {content.get('answer', '')} {content.get('explanation', '')}".lower()

            # Check for scientific method components
            method_components = self._identify_method_components(text)

            # Validate hypothesis formation
            hypothesis_quality = self._validate_hypothesis_quality(content)

            # Check experimental design if applicable
            experimental_design = self._validate_experimental_design(content)

            # Assess logical flow
            logical_flow = self._assess_logical_flow(text, method_components)

            # Calculate overall scientific method score
            component_score = len(method_components) / len(self.scientific_method_steps)
            hypothesis_score = hypothesis_quality.get("score", 0.5)
            design_score = experimental_design.get(
                "score", 0.7
            )  # Default if no experiment
            flow_score = logical_flow.get("score", 0.5)

            overall_score = (
                0.3 * component_score
                + 0.3 * hypothesis_score
                + 0.2 * design_score
                + 0.2 * flow_score
            )

            return SubValidationResult(
                subdomain="scientific_method",
                is_valid=overall_score >= 0.4,
                details={
                    "overall_score": overall_score,
                    "method_components": method_components,
                    "hypothesis_quality": hypothesis_quality,
                    "experimental_design": experimental_design,
                    "logical_flow": logical_flow,
                    "component_coverage": component_score,
                },
                confidence_score=0.8 if overall_score >= 0.6 else 0.5,
            )

        except Exception as e:
            logger.error("Scientific method validation failed: %s", str(e))
            return SubValidationResult(
                subdomain="scientific_method",
                is_valid=False,
                details={"error": str(e)},
                confidence_score=0.0,
                error_message=f"Scientific method validation error: {str(e)}",
            )

    def _identify_method_components(self, text: str) -> List[str]:
        """Identify scientific method components in text."""
        components = []

        # Check for observations
        if any(indicator in text for indicator in self.observation_indicators):
            components.append("observation")

        # Check for hypothesis
        if any(indicator in text for indicator in self.hypothesis_indicators):
            components.append("hypothesis")

        # Check for predictions
        if any(
            word in text for word in ["predict", "expect", "anticipate", "forecast"]
        ):
            components.append("prediction")

        # Check for experiments
        if any(
            word in text
            for word in ["experiment", "test", "trial", "study", "investigation"]
        ):
            components.append("experiment")

        # Check for analysis
        if any(
            word in text
            for word in ["analyze", "analysis", "examine", "evaluate", "interpret"]
        ):
            components.append("analysis")

        # Check for conclusions
        if any(
            word in text
            for word in ["conclude", "conclusion", "therefore", "thus", "result"]
        ):
            components.append("conclusion")

        return components

    def _validate_hypothesis_quality(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the quality of hypothesis formation."""
        text = f"{content.get('problem', '')} {content.get('answer', '')} {content.get('explanation', '')}".lower()

        # Check for testable hypothesis
        is_testable = any(
            word in text for word in ["test", "measure", "observe", "experiment"]
        )

        # Check for if-then structure
        has_structure = "if" in text and "then" in text

        # Check for variables identification
        has_variables = any(
            word in text for word in ["variable", "factor", "cause", "effect"]
        )

        # Check for falsifiability
        is_falsifiable = any(
            word in text for word in ["disprove", "refute", "contradict", "alternative"]
        )

        quality_elements = [is_testable, has_structure, has_variables, is_falsifiable]
        quality_score = sum(quality_elements) / len(quality_elements)

        return {
            "score": quality_score,
            "is_testable": is_testable,
            "has_structure": has_structure,
            "has_variables": has_variables,
            "is_falsifiable": is_falsifiable,
        }

    def _validate_experimental_design(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Validate experimental design elements."""
        text = f"{content.get('problem', '')} {content.get('answer', '')} {content.get('explanation', '')}".lower()

        design_elements = {}

        # Check for control group
        design_elements["has_control"] = any(
            word in text for word in ["control", "baseline", "comparison"]
        )

        # Check for variables
        design_elements["identifies_variables"] = any(
            word in text
            for word in [
                "independent variable",
                "dependent variable",
                "controlled variable",
                "constant",
            ]
        )

        # Check for sample size consideration
        design_elements["considers_sample_size"] = any(
            word in text
            for word in [
                "sample",
                "participants",
                "subjects",
                "trials",
                "repetition",
                "plants",
                "specimens",
            ]
        ) or any(
            f"{i} " in text for i in range(5, 101)
        )  # Numbers 5-100 indicate sample size

        # Check for methodology
        design_elements["has_methodology"] = any(
            word in text
            for word in [
                "method",
                "procedure",
                "protocol",
                "steps",
                "process",
                "measure",
                "use",
            ]
        )

        # Check for data collection
        design_elements["plans_data_collection"] = any(
            word in text
            for word in ["data", "measurement", "record", "collect", "gather"]
        )

        design_score = sum(design_elements.values()) / len(design_elements)

        return {"score": design_score, **design_elements}

    def _assess_logical_flow(self, text: str, components: List[str]) -> Dict[str, Any]:
        """Assess the logical flow of scientific reasoning."""
        # Check if components follow logical order
        expected_order = [
            "observation",
            "hypothesis",
            "prediction",
            "experiment",
            "analysis",
            "conclusion",
        ]

        present_components = [comp for comp in expected_order if comp in components]

        # Calculate order score
        order_score = 1.0
        for i in range(len(present_components) - 1):
            current_idx = expected_order.index(present_components[i])
            next_idx = expected_order.index(present_components[i + 1])
            if next_idx < current_idx:
                order_score -= 0.2

        order_score = max(0.0, order_score)

        # Check for logical connectors
        logical_connectors = [
            "because",
            "therefore",
            "thus",
            "since",
            "as a result",
            "consequently",
        ]
        has_connectors = any(connector in text for connector in logical_connectors)

        connector_score = 1.0 if has_connectors else 0.5

        overall_flow_score = (order_score + connector_score) / 2

        return {
            "score": overall_flow_score,
            "order_score": order_score,
            "has_logical_connectors": has_connectors,
            "component_order": present_components,
        }


class SafetyEthicsValidator:
    """Validator for scientific safety and ethics considerations."""

    def __init__(self):
        """Initialize safety and ethics validator."""
        self.safety_keywords = [
            "safety",
            "hazard",
            "risk",
            "danger",
            "precaution",
            "protection",
            "toxic",
            "harmful",
            "corrosive",
            "flammable",
            "explosive",
            "acid",
            "base",
            "chemical",
            "concentrated",
            "caustic",
        ]

        self.ethics_keywords = [
            "ethics",
            "ethical",
            "consent",
            "approval",
            "welfare",
            "rights",
            "harm",
            "benefit",
            "justice",
            "autonomy",
            "beneficence",
        ]

        self.animal_welfare_keywords = [
            "animal",
            "welfare",
            "humane",
            "iacuc",
            "institutional",
            "care",
        ]

        self.human_subjects_keywords = [
            "human",
            "participant",
            "subject",
            "irb",
            "consent",
            "volunteer",
        ]

    def validate_safety_ethics(self, content: Dict[str, Any]) -> SubValidationResult:
        """
        Validate safety and ethics considerations in scientific content.

        Args:
            content: Scientific content to validate

        Returns:
            SubValidationResult with safety and ethics validation details
        """
        try:
            text = f"{content.get('problem', '')} {content.get('answer', '')} {content.get('explanation', '')}".lower()

            # Check for safety considerations
            safety_assessment = self._assess_safety_considerations(text, content)

            # Check for ethical considerations
            ethics_assessment = self._assess_ethical_considerations(text, content)

            # Check for specific subject protections
            subject_protection = self._assess_subject_protection(text, content)

            # Calculate overall safety/ethics score
            safety_score = safety_assessment.get(
                "score", 0.8
            )  # Default high if no safety issues
            ethics_score = ethics_assessment.get(
                "score", 0.8
            )  # Default high if no ethics issues
            protection_score = subject_protection.get("score", 0.9)  # Default high

            overall_score = (
                0.4 * safety_score + 0.4 * ethics_score + 0.2 * protection_score
            )

            return SubValidationResult(
                subdomain="safety_ethics",
                is_valid=overall_score >= 0.7,
                details={
                    "overall_score": overall_score,
                    "safety_assessment": safety_assessment,
                    "ethics_assessment": ethics_assessment,
                    "subject_protection": subject_protection,
                },
                confidence_score=0.8 if overall_score >= 0.7 else 0.6,
            )

        except Exception as e:
            logger.error("Safety/ethics validation failed: %s", str(e))
            return SubValidationResult(
                subdomain="safety_ethics",
                is_valid=False,
                details={"error": str(e)},
                confidence_score=0.0,
                error_message=f"Safety/ethics validation error: {str(e)}",
            )

    def _assess_safety_considerations(
        self, text: str, content: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess safety considerations in scientific content."""
        # Check if content involves potentially hazardous materials or procedures
        involves_hazards = any(keyword in text for keyword in self.safety_keywords)

        # Check if safety measures are mentioned when hazards are present
        safety_measures_mentioned = False
        if involves_hazards:
            safety_measures = [
                " ppe ",
                "ventilation",
                "fume hood",
                "gloves",
                "goggles",
                "lab coat",
                "wear safety",
                "wear protective",
                "use protection",
                "use safety",
                "protective equipment",
                "safety precaution",
            ]
            safety_measures_mentioned = any(
                measure in text for measure in safety_measures
            )

        # Check for proper disposal mentions
        mentions_disposal = any(
            word in text for word in ["dispose", "disposal", "waste", "discard"]
        )

        # Calculate safety score
        if not involves_hazards:
            safety_score = 0.9  # High score if no hazards
        else:
            safety_elements = [safety_measures_mentioned, mentions_disposal]
            safety_score = sum(safety_elements) / len(safety_elements)
            # If hazards are present but no safety measures, score should be low
            if not safety_measures_mentioned:
                safety_score = min(safety_score, 0.3)

        return {
            "score": safety_score,
            "involves_hazards": involves_hazards,
            "safety_measures_mentioned": safety_measures_mentioned,
            "mentions_disposal": mentions_disposal,
        }

    def _assess_ethical_considerations(
        self, text: str, content: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess ethical considerations in scientific content."""
        # Check if content involves ethical considerations
        involves_ethics = any(keyword in text for keyword in self.ethics_keywords)

        # Check for research ethics mentions
        mentions_approval = any(
            word in text for word in ["approval", "committee", "review", "irb", "iacuc"]
        )

        # Check for informed consent
        mentions_consent = "consent" in text or "voluntary" in text

        # Check for risk-benefit analysis
        mentions_risk_benefit = any(
            word in text for word in ["risk", "benefit", "harm", "advantage"]
        )

        # Calculate ethics score
        if not involves_ethics:
            ethics_score = 0.9  # High score if no ethical issues
        else:
            ethics_elements = [
                mentions_approval,
                mentions_consent,
                mentions_risk_benefit,
            ]
            ethics_score = sum(ethics_elements) / len(ethics_elements)

        return {
            "score": ethics_score,
            "involves_ethics": involves_ethics,
            "mentions_approval": mentions_approval,
            "mentions_consent": mentions_consent,
            "mentions_risk_benefit": mentions_risk_benefit,
        }

    def _assess_subject_protection(
        self, text: str, content: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess protection of human/animal subjects."""
        # Check for human subjects
        involves_humans = any(
            keyword in text for keyword in self.human_subjects_keywords
        )

        # Check for animal subjects
        involves_animals = any(
            keyword in text for keyword in self.animal_welfare_keywords
        )

        # Assess protection measures
        protection_score = 0.9  # Default high

        if involves_humans:
            human_protections = ["consent", "confidentiality", "privacy", "withdrawal"]
            protection_mentioned = any(
                protection in text for protection in human_protections
            )
            if not protection_mentioned:
                protection_score -= 0.3

        if involves_animals:
            animal_protections = [
                "welfare",
                "humane",
                "minimize",
                "reduce",
                "replace",
                "iacuc",
                "guidelines",
            ]
            protection_mentioned = any(
                protection in text for protection in animal_protections
            )
            if not protection_mentioned:
                protection_score -= 0.3

        protection_score = max(0.0, protection_score)

        return {
            "score": protection_score,
            "involves_humans": involves_humans,
            "involves_animals": involves_animals,
            "adequate_protection": protection_score >= 0.6,
        }


class ScienceValidator(DomainValidator):
    """Science domain validator with subdomain routing and comprehensive validation."""

    def __init__(self, domain: str, config: ValidationConfig):
        """
        Initialize science validator.

        Args:
            domain: Should be "science"
            config: Validation configuration for science domain
        """
        super().__init__(domain, config)

        # Initialize specialized validators
        self.scientific_method_validator = ScientificMethodValidator()
        self.safety_ethics_validator = SafetyEthicsValidator()

        # Subdomain validators (placeholders for now)
        self.subdomain_validators = {
            "physics": self._create_physics_validator,
            "chemistry": self._create_chemistry_validator,
            "biology": self._create_biology_validator,
        }

        logger.info("Initialized ScienceValidator with subdomain routing")

    def validate_content(self, content: Dict[str, Any]) -> ValidationResult:
        """
        Comprehensive science content validation.

        Args:
            content: Scientific content to validate

        Returns:
            ValidationResult with comprehensive validation details
        """
        validation_details = {}

        try:
            # Determine subdomain
            subdomain = content.get("subdomain", self._detect_subdomain(content))

            # Subdomain-specific validation
            if subdomain in self.subdomain_validators:
                subdomain_result = self._validate_subdomain(content, subdomain)
                validation_details[f"{subdomain}_validation"] = subdomain_result

            # Scientific method validation
            method_result = self.scientific_method_validator.validate_scientific_method(
                content
            )
            validation_details["scientific_method"] = method_result

            # Safety and ethics validation
            safety_ethics_result = self.safety_ethics_validator.validate_safety_ethics(
                content
            )
            validation_details["safety_ethics"] = safety_ethics_result

            # Calculate quality metrics
            quality_metrics = self.calculate_quality_metrics(
                content, validation_details
            )

            # Determine overall validity
            is_valid = self._determine_overall_validity(validation_details)

            # Calculate confidence score
            confidence_score = self.calculate_confidence(validation_details)

            # Generate feedback
            feedback = self.generate_feedback_from_details(validation_details)

            return ValidationResult(
                domain=self.domain,
                is_valid=is_valid,
                quality_score=quality_metrics.overall_score,
                validation_details=validation_details,
                confidence_score=confidence_score,
                feedback=feedback,
                quality_metrics=quality_metrics,
            )

        except Exception as e:
            logger.error("Science validation failed: %s", str(e))
            raise DomainValidationError(
                self.domain, f"Science validation failed: {str(e)}"
            ) from e

    def _detect_subdomain(self, content: Dict[str, Any]) -> str:
        """Detect the science subdomain from content."""
        text = f"{content.get('problem', '')} {content.get('answer', '')}".lower()

        # Physics indicators
        physics_keywords = [
            "force",
            "energy",
            "motion",
            "velocity",
            "acceleration",
            "mass",
            "gravity",
            "momentum",
            "wave",
            "frequency",
            "electric",
            "magnetic",
        ]

        # Chemistry indicators
        chemistry_keywords = [
            "molecule",
            "atom",
            "reaction",
            "chemical",
            "bond",
            "element",
            "compound",
            "solution",
            "acid",
            "base",
            "ph",
            "catalyst",
        ]

        # Biology indicators
        biology_keywords = [
            "cell",
            "organism",
            "gene",
            "dna",
            "protein",
            "evolution",
            "ecosystem",
            "species",
            "reproduction",
            "metabolism",
            "photosynthesis",
            "chlorophyll",
            "plants",
            "glucose",
        ]

        # Count matches for each subdomain
        physics_score = sum(1 for keyword in physics_keywords if keyword in text)
        chemistry_score = sum(1 for keyword in chemistry_keywords if keyword in text)
        biology_score = sum(1 for keyword in biology_keywords if keyword in text)

        # Return subdomain with highest score
        scores = {
            "physics": physics_score,
            "chemistry": chemistry_score,
            "biology": biology_score,
        }
        max_subdomain = max(scores, key=scores.get)

        return max_subdomain if scores[max_subdomain] > 0 else "general"

    def _validate_subdomain(
        self, content: Dict[str, Any], subdomain: str
    ) -> SubValidationResult:
        """Validate content for specific science subdomain."""
        try:
            validator_creator = self.subdomain_validators.get(subdomain)
            if validator_creator:
                validator = validator_creator()
                return validator.validate(content)
            else:
                # Generic science validation
                return SubValidationResult(
                    subdomain=subdomain,
                    is_valid=True,
                    details={"validation_type": "generic_science"},
                    confidence_score=0.7,
                )

        except Exception as e:
            logger.error("Subdomain validation failed for %s: %s", subdomain, str(e))
            return SubValidationResult(
                subdomain=subdomain,
                is_valid=False,
                details={"error": str(e)},
                confidence_score=0.0,
                error_message=f"Subdomain validation error: {str(e)}",
            )

    def _create_physics_validator(self):
        """Create physics subdomain validator."""
        from .physics import PhysicsValidator

        return PhysicsValidator()

    def _create_chemistry_validator(self):
        """Create chemistry subdomain validator (placeholder)."""
        return PlaceholderSubdomainValidator("chemistry")

    def _create_biology_validator(self):
        """Create biology subdomain validator (placeholder)."""
        return PlaceholderSubdomainValidator("biology")

    def _determine_overall_validity(self, validation_details: Dict[str, Any]) -> bool:
        """Determine overall validity based on validation results."""
        # Scientific method validation is important
        method_result = validation_details.get("scientific_method")
        method_valid = method_result.is_valid if method_result else True

        # Safety/ethics validation is critical
        safety_result = validation_details.get("safety_ethics")
        safety_valid = safety_result.is_valid if safety_result else True

        # Safety/ethics is critical - must pass
        if not safety_valid:
            return False

        # Scientific method should generally be valid
        if not method_valid:
            # Allow some tolerance for basic content
            method_score = (
                method_result.details.get("overall_score", 0.0)
                if method_result
                else 0.5
            )
            if method_score < 0.4:
                return False

        # Check subdomain validations
        for key, result in validation_details.items():
            if key.endswith("_validation") and key not in [
                "scientific_method",
                "safety_ethics",
            ]:
                if hasattr(result, "is_valid") and not result.is_valid:
                    return False

        return True

    def calculate_quality_score(self, content: Dict[str, Any]) -> float:
        """
        Calculate domain-specific quality score for science.

        Args:
            content: Scientific content to assess

        Returns:
            Quality score between 0.0 and 1.0
        """
        base_score = 0.7

        # Adjust based on scientific rigor
        if "experiment" in content.get("problem", "").lower():
            base_score += 0.1

        if "hypothesis" in content.get("problem", "").lower():
            base_score += 0.1

        if "data" in content.get("answer", "").lower():
            base_score += 0.1

        return min(1.0, base_score)

    def generate_feedback(self, validation_result: ValidationResult) -> List[str]:
        """
        Generate science-specific improvement feedback.

        Args:
            validation_result: Result of validation to generate feedback for

        Returns:
            List of feedback messages for content improvement
        """
        return self.generate_feedback_from_details(validation_result.validation_details)

    def generate_feedback_from_details(
        self, validation_details: Dict[str, Any]
    ) -> List[str]:
        """Generate feedback from validation details."""
        feedback = []

        # Scientific method feedback
        method_result = validation_details.get("scientific_method")
        if method_result and not method_result.is_valid:
            method_details = method_result.details
            missing_components = set(
                self.scientific_method_validator.scientific_method_steps
            ) - set(method_details.get("method_components", []))
            if missing_components:
                feedback.append(
                    f"Consider including these scientific method components: {', '.join(list(missing_components)[:3])}"
                )

        # Safety/ethics feedback
        safety_result = validation_details.get("safety_ethics")
        if safety_result and not safety_result.is_valid:
            safety_details = safety_result.details.get("safety_assessment", {})
            if safety_details.get("involves_hazards") and not safety_details.get(
                "safety_measures_mentioned"
            ):
                feedback.append(
                    "Include safety precautions and protective measures for hazardous materials or procedures"
                )

            ethics_details = safety_result.details.get("ethics_assessment", {})
            if ethics_details.get("involves_ethics") and not ethics_details.get(
                "mentions_approval"
            ):
                feedback.append(
                    "Consider mentioning ethical approval or review processes for research involving subjects"
                )

        # Subdomain-specific feedback
        for key, result in validation_details.items():
            if key.endswith("_validation") and key not in [
                "scientific_method",
                "safety_ethics",
            ]:
                if hasattr(result, "is_valid") and not result.is_valid:
                    subdomain = key.replace("_validation", "")
                    feedback.append(
                        f"Review {subdomain}-specific content for accuracy and completeness"
                    )

        return feedback


class PlaceholderSubdomainValidator:
    """Placeholder validator for science subdomains."""

    def __init__(self, subdomain: str):
        """Initialize placeholder validator.

        Args:
            subdomain: The science subdomain (physics, chemistry, biology)
        """
        self.subdomain = subdomain

    def validate(self, content: Dict[str, Any]) -> SubValidationResult:
        """
        Placeholder validation for subdomain.

        Args:
            content: Content to validate

        Returns:
            SubValidationResult with placeholder validation
        """
        return SubValidationResult(
            subdomain=self.subdomain,
            is_valid=True,
            details={
                "validation_type": f"{self.subdomain}_placeholder",
                "message": f"Placeholder validation for {self.subdomain} subdomain",
            },
            confidence_score=0.7,
        )
        self.subdomain = subdomain

    def validate(self, content: Dict[str, Any]) -> SubValidationResult:
        """Placeholder validation that always passes."""
        return SubValidationResult(
            subdomain=self.subdomain,
            is_valid=True,
            details={
                "placeholder": True,
                "validation_type": f"{self.subdomain}_placeholder",
            },
            confidence_score=0.7,
        )
