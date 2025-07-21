"""
DSPy Signatures for STREAM Domains

This module defines domain-specific DSPy signatures for all STREAM fields
(Science, Technology, Reading, Engineering, Arts, Mathematics), providing
structured input/output specifications for content generation.
"""

import logging
from typing import Dict, List, Optional, Tuple

from .exceptions import SignatureValidationError

logger = logging.getLogger(__name__)


# Domain-specific DSPy signatures
STREAM_SIGNATURES = {
    "mathematics": {
        "generation": (
            "mathematical_concept, difficulty_level, learning_objectives, context_info -> "
            "problem_statement, solution, proof, reasoning_trace, pedagogical_hints, misconception_analysis"
        ),
        "validation": (
            "problem_statement, solution, hints, expected_difficulty -> "
            "valid, reason, corrected_hints, quality_score, mathematical_accuracy, pedagogical_value"
        ),
        "equivalence": (
            "problem_statement, true_answer, model_answer, solution_context -> "
            "equivalent, confidence_score, explanation, mathematical_justification"
        ),
        "refinement": (
            "content, feedback, quality_metrics, target_audience -> "
            "refined_content, improvements, confidence_score, pedagogical_enhancements"
        ),
        "assessment": (
            "student_solution, reference_solution, assessment_criteria -> "
            "score, feedback, misconceptions, improvement_suggestions, conceptual_understanding"
        ),
    },
    "science": {
        "generation": (
            "scientific_concept, difficulty_level, learning_objectives, science_domain -> "
            "problem_statement, solution, experimental_design, evidence_evaluation, reasoning_trace, scientific_principles"
        ),
        "validation": (
            "problem_statement, solution, experimental_design, domain_knowledge -> "
            "valid, reason, scientific_accuracy, methodology_score, factual_correctness, conceptual_soundness"
        ),
        "equivalence": (
            "problem_statement, true_answer, model_answer, scientific_context -> "
            "equivalent, confidence_score, scientific_reasoning, evidence_analysis"
        ),
        "refinement": (
            "content, feedback, quality_metrics, scientific_standards -> "
            "refined_content, improvements, confidence_score, scientific_rigor_enhancements"
        ),
        "assessment": (
            "student_experiment, reference_procedure, assessment_criteria -> "
            "score, feedback, methodological_issues, scientific_reasoning_quality, improvement_suggestions"
        ),
    },
    "technology": {
        "generation": (
            "technical_concept, difficulty_level, learning_objectives, tech_domain -> "
            "problem_statement, solution, algorithm_explanation, system_design, reasoning_trace, implementation_considerations"
        ),
        "validation": (
            "problem_statement, solution, algorithm_explanation, technical_constraints -> "
            "valid, reason, technical_accuracy, implementation_feasibility, efficiency_analysis, scalability_assessment"
        ),
        "equivalence": (
            "problem_statement, true_answer, model_answer, technical_context -> "
            "equivalent, confidence_score, technical_analysis, performance_comparison"
        ),
        "refinement": (
            "content, feedback, quality_metrics, technical_standards -> "
            "refined_content, improvements, confidence_score, technical_optimizations"
        ),
        "assessment": (
            "student_implementation, reference_implementation, assessment_criteria -> "
            "score, feedback, code_quality, algorithmic_efficiency, improvement_suggestions"
        ),
    },
    "reading": {
        "generation": (
            "literary_concept, difficulty_level, learning_objectives, text_genre -> "
            "comprehension_question, analysis_prompt, critical_thinking_exercise, reasoning_trace, literary_context"
        ),
        "validation": (
            "comprehension_question, analysis_prompt, critical_thinking_exercise, literary_standards -> "
            "valid, reason, literary_accuracy, cognitive_complexity, educational_value, engagement_level"
        ),
        "equivalence": (
            "question, expected_response, student_response, textual_evidence -> "
            "equivalent, confidence_score, literary_analysis, interpretation_validity"
        ),
        "refinement": (
            "content, feedback, quality_metrics, literary_standards -> "
            "refined_content, improvements, confidence_score, literary_enhancements"
        ),
        "assessment": (
            "student_analysis, reference_analysis, assessment_criteria -> "
            "score, feedback, analytical_depth, textual_evidence_usage, improvement_suggestions"
        ),
    },
    "engineering": {
        "generation": (
            "engineering_concept, difficulty_level, learning_objectives, engineering_domain -> "
            "design_challenge, optimization_problem, constraint_analysis, reasoning_trace, engineering_principles"
        ),
        "validation": (
            "design_challenge, optimization_problem, constraint_analysis, engineering_standards -> "
            "valid, reason, engineering_feasibility, design_quality, safety_assessment, efficiency_analysis"
        ),
        "equivalence": (
            "problem_statement, reference_solution, proposed_solution, engineering_context -> "
            "equivalent, confidence_score, engineering_assessment, performance_comparison"
        ),
        "refinement": (
            "content, feedback, quality_metrics, engineering_standards -> "
            "refined_content, improvements, confidence_score, engineering_optimizations"
        ),
        "assessment": (
            "student_design, reference_design, assessment_criteria -> "
            "score, feedback, design_quality, constraint_satisfaction, improvement_suggestions"
        ),
    },
    "arts": {
        "generation": (
            "artistic_concept, difficulty_level, learning_objectives, art_form -> "
            "creative_prompt, aesthetic_analysis, cultural_context, reasoning_trace, artistic_principles"
        ),
        "validation": (
            "creative_prompt, aesthetic_analysis, cultural_context, artistic_standards -> "
            "valid, reason, artistic_merit, cultural_sensitivity, creativity_assessment, educational_value"
        ),
        "equivalence": (
            "prompt, expected_interpretation, student_interpretation, artistic_context -> "
            "equivalent, confidence_score, artistic_analysis, creative_assessment"
        ),
        "refinement": (
            "content, feedback, quality_metrics, artistic_standards -> "
            "refined_content, improvements, confidence_score, artistic_enhancements"
        ),
        "assessment": (
            "student_creation, reference_criteria, assessment_rubric -> "
            "score, feedback, creative_expression, technical_execution, improvement_suggestions"
        ),
    },
    "interdisciplinary": {
        "generation": (
            "primary_domain, secondary_domain, difficulty_level, learning_objectives -> "
            "problem_statement, solution, cross_domain_connections, reasoning_trace, interdisciplinary_principles"
        ),
        "validation": (
            "problem_statement, solution, domain_connections, interdisciplinary_standards -> "
            "valid, reason, cross_domain_accuracy, integration_quality, educational_value"
        ),
        "equivalence": (
            "problem_statement, reference_solution, proposed_solution, interdisciplinary_context -> "
            "equivalent, confidence_score, cross_domain_assessment, integration_analysis"
        ),
        "refinement": (
            "content, feedback, quality_metrics, interdisciplinary_standards -> "
            "refined_content, improvements, confidence_score, integration_enhancements"
        ),
        "assessment": (
            "student_solution, reference_solution, interdisciplinary_criteria -> "
            "score, feedback, cross_domain_understanding, integration_quality, improvement_suggestions"
        ),
    },
}


def get_domain_signature(domain: str, signature_type: str = "generation") -> str:
    """
    Get DSPy signature for a specific domain and type.

    Args:
        domain: STREAM domain (e.g., 'mathematics', 'science')
        signature_type: Type of signature ('generation', 'validation', 'equivalence', 'refinement')

    Returns:
        DSPy signature string

    Raises:
        SignatureValidationError: If domain or signature type is invalid
    """
    domain = domain.lower()

    if domain not in STREAM_SIGNATURES:
        available_domains = list(STREAM_SIGNATURES.keys())
        raise SignatureValidationError(
            "Unknown domain '%s'. Available domains: %s" % (domain, available_domains),
            signature=f"{domain}:{signature_type}",
        )

    domain_signatures = STREAM_SIGNATURES[domain]

    if signature_type not in domain_signatures:
        available_types = list(domain_signatures.keys())
        raise SignatureValidationError(
            "Unknown signature type '%s' for domain '%s'. Available types: %s"
            % (signature_type, domain, available_types),
            signature=f"{domain}:{signature_type}",
        )

    signature = domain_signatures[signature_type]
    logger.debug("Retrieved signature for %s:%s: %s", domain, signature_type, signature)

    return signature


def validate_signature(signature: str) -> bool:
    """
    Validate a DSPy signature format.

    Args:
        signature: DSPy signature string to validate

    Returns:
        True if signature is valid

    Raises:
        SignatureValidationError: If signature format is invalid
    """
    if not signature or not isinstance(signature, str):
        raise SignatureValidationError(
            "Signature must be a non-empty string", signature=str(signature)
        )

    if " -> " not in signature:
        raise SignatureValidationError(
            "Signature must contain ' -> ' separator between inputs and outputs",
            signature=signature,
        )

    parts = signature.split(" -> ")
    if len(parts) != 2:
        raise SignatureValidationError(
            "Signature must have exactly one ' -> ' separator", signature=signature
        )

    inputs, outputs = parts

    # Validate inputs
    if not inputs.strip():
        raise SignatureValidationError(
            "Signature must have at least one input field", signature=signature
        )

    # Validate outputs
    if not outputs.strip():
        raise SignatureValidationError(
            "Signature must have at least one output field", signature=signature
        )

    # Check for valid field names (basic validation)
    input_fields = [field.strip() for field in inputs.split(",")]
    output_fields = [field.strip() for field in outputs.split(",")]

    for field in input_fields + output_fields:
        if not field or not field.replace("_", "").isalnum():
            raise SignatureValidationError(
                f"Invalid field name '{field}'. Fields must be alphanumeric with underscores",
                signature=signature,
            )

    logger.debug(
        "Signature validation passed: %d inputs, %d outputs",
        len(input_fields),
        len(output_fields),
    )
    return True


def get_all_domains() -> list:
    """Get list of all available STREAM domains."""
    return list(STREAM_SIGNATURES.keys())


def get_signature_types(domain: str) -> list:
    """
    Get list of available signature types for a domain.

    Args:
        domain: STREAM domain

    Returns:
        List of available signature types

    Raises:
        SignatureValidationError: If domain is invalid
    """
    domain = domain.lower()

    if domain not in STREAM_SIGNATURES:
        available_domains = list(STREAM_SIGNATURES.keys())
        raise SignatureValidationError(
            "Unknown domain '%s'. Available domains: %s" % (domain, available_domains),
            signature=domain,
        )

    return list(STREAM_SIGNATURES[domain].keys())


def create_custom_signature(inputs: list, outputs: list) -> str:
    """
    Create a custom DSPy signature from input and output field lists.

    Args:
        inputs: List of input field names
        outputs: List of output field names

    Returns:
        DSPy signature string

    Raises:
        SignatureValidationError: If inputs or outputs are invalid
    """
    if not inputs or not isinstance(inputs, list):
        raise SignatureValidationError(
            "Inputs must be a non-empty list", signature=f"inputs={inputs}"
        )

    if not outputs or not isinstance(outputs, list):
        raise SignatureValidationError(
            "Outputs must be a non-empty list", signature=f"outputs={outputs}"
        )

    # Validate field names
    for field in inputs + outputs:
        if not isinstance(field, str) or not field.strip():
            raise SignatureValidationError(
                "Field names must be non-empty strings: '%s'" % field,
                signature=f"field={field}",
            )

        if not field.replace("_", "").isalnum():
            raise SignatureValidationError(
                "Field names must be alphanumeric with underscores: '%s'" % field,
                signature=f"field={field}",
            )

    # Create signature
    inputs_str = ", ".join(inputs)
    outputs_str = ", ".join(outputs)
    signature = f"{inputs_str} -> {outputs_str}"

    # Validate the created signature
    validate_signature(signature)

    logger.debug("Created custom signature: %s", signature)
    return signature


class SignatureManager:
    """
    Manages DSPy signatures for STREAM domains.

    Provides functionality for loading, validating, and versioning signatures
    across different domains and signature types.
    """

    def __init__(self):
        """Initialize signature manager."""
        self.logger = logging.getLogger(f"{__name__}.SignatureManager")
        self.signatures = STREAM_SIGNATURES
        self.custom_signatures = {}
        self.signature_versions = {}

        # Initialize version tracking
        for domain in self.signatures:
            self.signature_versions[domain] = {
                signature_type: "1.0.0" for signature_type in self.signatures[domain]
            }

        self.logger.info(
            "Initialized signature manager with %d domains", len(self.signatures)
        )

    def get_signature(self, domain: str, signature_type: str = "generation") -> str:
        """
        Get signature for domain and type.

        Args:
            domain: STREAM domain
            signature_type: Type of signature

        Returns:
            DSPy signature string
        """
        # Check custom signatures first
        domain = domain.lower()
        if (
            domain in self.custom_signatures
            and signature_type in self.custom_signatures[domain]
        ):
            return self.custom_signatures[domain][signature_type]

        # Fall back to built-in signatures
        return get_domain_signature(domain, signature_type)

    def register_custom_signature(
        self, domain: str, signature_type: str, signature: str, version: str = "1.0.0"
    ) -> bool:
        """
        Register a custom signature for a domain.

        Args:
            domain: STREAM domain
            signature_type: Type of signature
            signature: DSPy signature string
            version: Signature version

        Returns:
            True if registered successfully
        """
        try:
            # Validate signature
            validate_signature(signature)

            # Create domain if needed
            domain = domain.lower()
            if domain not in self.custom_signatures:
                self.custom_signatures[domain] = {}

            # Store signature
            self.custom_signatures[domain][signature_type] = signature

            # Update version
            if domain not in self.signature_versions:
                self.signature_versions[domain] = {}
            self.signature_versions[domain][signature_type] = version

            self.logger.info(
                "Registered custom signature for %s:%s (v%s)",
                domain,
                signature_type,
                version,
            )
            return True

        except Exception as e:
            self.logger.error("Failed to register custom signature: %s", str(e))
            return False

    def parse_signature(self, signature: str) -> Tuple[List[str], List[str]]:
        """
        Parse a signature into input and output fields.

        Args:
            signature: DSPy signature string

        Returns:
            Tuple of (input_fields, output_fields)
        """
        validate_signature(signature)

        inputs_str, outputs_str = signature.split(" -> ")

        input_fields = [field.strip() for field in inputs_str.split(",")]
        output_fields = [field.strip() for field in outputs_str.split(",")]

        return input_fields, output_fields

    def get_signature_version(self, domain: str, signature_type: str) -> str:
        """
        Get version of a signature.

        Args:
            domain: STREAM domain
            signature_type: Type of signature

        Returns:
            Version string
        """
        domain = domain.lower()

        if domain in self.signature_versions:
            return self.signature_versions.get(domain, {}).get(
                signature_type, "unknown"
            )

        return "unknown"

    def is_signature_compatible(
        self, signature1: str, signature2: str, strict: bool = False
    ) -> bool:
        """
        Check if two signatures are compatible.

        Args:
            signature1: First signature
            signature2: Second signature
            strict: Whether to require exact match

        Returns:
            True if signatures are compatible
        """
        try:
            # Parse signatures
            inputs1, outputs1 = self.parse_signature(signature1)
            inputs2, outputs2 = self.parse_signature(signature2)

            if strict:
                # Strict mode requires exact match
                return inputs1 == inputs2 and outputs1 == outputs2

            # Compatible if all required inputs are present
            for input_field in inputs2:
                if input_field not in inputs1:
                    return False

            # Compatible if all required outputs are present
            for output_field in outputs2:
                if output_field not in outputs1:
                    return False

            return True

        except Exception as e:
            self.logger.error("Error checking signature compatibility: %s", str(e))
            return False

    def get_all_signatures(self) -> Dict[str, Dict[str, str]]:
        """
        Get all available signatures.

        Returns:
            Dictionary of all signatures by domain and type
        """
        # Combine built-in and custom signatures
        all_signatures = {}

        # Add built-in signatures
        for domain, domain_signatures in self.signatures.items():
            all_signatures[domain] = dict(domain_signatures)

        # Add custom signatures (overriding built-in if needed)
        for domain, domain_signatures in self.custom_signatures.items():
            if domain not in all_signatures:
                all_signatures[domain] = {}

            for sig_type, signature in domain_signatures.items():
                all_signatures[domain][sig_type] = signature

        return all_signatures

    def extend_signature(
        self,
        base_signature: str,
        additional_inputs: List[str] = None,
        additional_outputs: List[str] = None,
    ) -> str:
        """
        Extend an existing signature with additional inputs and outputs.

        Args:
            base_signature: Base signature to extend
            additional_inputs: Additional input fields to add
            additional_outputs: Additional output fields to add

        Returns:
            Extended signature string
        """
        inputs, outputs = self.parse_signature(base_signature)

        if additional_inputs:
            for input_field in additional_inputs:
                if input_field not in inputs:
                    inputs.append(input_field)

        if additional_outputs:
            for output_field in additional_outputs:
                if output_field not in outputs:
                    outputs.append(output_field)

        return create_custom_signature(inputs, outputs)

    def create_composite_signature(
        self, domain1: str, domain2: str, signature_type: str = "generation"
    ) -> str:
        """
        Create a composite signature from two domains.

        Args:
            domain1: First STREAM domain
            domain2: Second STREAM domain
            signature_type: Type of signature

        Returns:
            Composite signature string
        """
        try:
            # Get signatures for both domains
            sig1 = self.get_signature(domain1, signature_type)
            sig2 = self.get_signature(domain2, signature_type)

            # Parse signatures
            inputs1, outputs1 = self.parse_signature(sig1)
            inputs2, outputs2 = self.parse_signature(sig2)

            # Combine inputs and outputs (avoiding duplicates)
            combined_inputs = list(inputs1)
            for input_field in inputs2:
                if input_field not in combined_inputs:
                    combined_inputs.append(input_field)

            combined_outputs = list(outputs1)
            for output_field in outputs2:
                if output_field not in combined_outputs:
                    combined_outputs.append(output_field)

            # Create composite signature
            composite_sig = create_custom_signature(combined_inputs, combined_outputs)

            self.logger.info(
                "Created composite signature from %s and %s for %s",
                domain1,
                domain2,
                signature_type,
            )
            return composite_sig

        except Exception as e:
            self.logger.error("Error creating composite signature: %s", str(e))
            raise SignatureValidationError(
                "Failed to create composite signature: %s" % str(e),
                signature=f"{domain1}+{domain2}:{signature_type}",
            ) from e

    def simplify_signature(
        self,
        signature: str,
        required_inputs: List[str] = None,
        required_outputs: List[str] = None,
    ) -> str:
        """
        Simplify a signature by keeping only required inputs and outputs.

        Args:
            signature: Signature to simplify
            required_inputs: Required input fields to keep
            required_outputs: Required output fields to keep

        Returns:
            Simplified signature string
        """
        inputs, outputs = self.parse_signature(signature)

        if required_inputs:
            # Keep only required inputs (and validate they exist)
            for req_input in required_inputs:
                if req_input not in inputs:
                    raise SignatureValidationError(
                        "Required input '%s' not found in signature" % req_input,
                        signature=signature,
                    )
            inputs = [inp for inp in inputs if inp in required_inputs]

        if required_outputs:
            # Keep only required outputs (and validate they exist)
            for req_output in required_outputs:
                if req_output not in outputs:
                    raise SignatureValidationError(
                        "Required output '%s' not found in signature" % req_output,
                        signature=signature,
                    )
            outputs = [out for out in outputs if out in required_outputs]

        return create_custom_signature(inputs, outputs)

    def get_domain_signatures_by_type(self, signature_type: str) -> Dict[str, str]:
        """
        Get signatures for all domains of a specific type.

        Args:
            signature_type: Type of signature

        Returns:
            Dictionary mapping domains to signatures
        """
        result = {}

        # Get built-in signatures
        for domain, domain_signatures in self.signatures.items():
            if signature_type in domain_signatures:
                result[domain] = domain_signatures[signature_type]

        # Add custom signatures (overriding built-in if needed)
        for domain, domain_signatures in self.custom_signatures.items():
            if signature_type in domain_signatures:
                result[domain] = domain_signatures[signature_type]

        return result
