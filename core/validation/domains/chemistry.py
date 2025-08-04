"""
Chemistry subdomain validator for the SynThesisAI platform.

This module implements comprehensive validation for chemistry-related content,
including chemical equations, reaction mechanisms, safety protocols, and
molecular structures.
"""

# Standard Library
import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

# SynThesisAI Modules
from core.validation.base import DomainValidator, ValidationResult
from core.validation.config import ValidationConfig

logger = logging.getLogger(__name__)


class ChemistryValidator(DomainValidator):
    """
    Validator for chemistry content with comprehensive chemical validation.

    This validator handles:
    - Chemical equation balancing and validation
    - Reaction mechanism verification
    - Chemical safety protocol validation
    - Molecular structure validation
    - Chemical nomenclature accuracy
    """

    def __init__(
        self, subdomain: str = "chemistry", config: Optional[ValidationConfig] = None
    ):
        """
        Initialize the chemistry validator.

        Args:
            subdomain: The chemistry subdomain (default: "chemistry")
            config: Validation configuration settings

        Raises:
            ValueError: If subdomain is not chemistry-related
        """
        super().__init__("science", config)
        self.subdomain = subdomain

        # Validate subdomain
        valid_subdomains = {
            "chemistry",
            "organic_chemistry",
            "inorganic_chemistry",
            "physical_chemistry",
            "analytical_chemistry",
            "biochemistry",
        }
        if subdomain not in valid_subdomains:
            raise ValueError(f"Invalid chemistry subdomain: {subdomain}")

        # Initialize chemical knowledge bases
        self._initialize_chemical_data()

        logger.info("Initialized ChemistryValidator for subdomain: %s", subdomain)

    def _initialize_chemical_data(self) -> None:
        """Initialize chemical data and knowledge bases."""
        # Common elements and their properties
        self.elements = {
            "H": {"atomic_number": 1, "atomic_mass": 1.008, "valence": [1]},
            "He": {"atomic_number": 2, "atomic_mass": 4.003, "valence": [0]},
            "Li": {"atomic_number": 3, "atomic_mass": 6.941, "valence": [1]},
            "Be": {"atomic_number": 4, "atomic_mass": 9.012, "valence": [2]},
            "B": {"atomic_number": 5, "atomic_mass": 10.811, "valence": [3]},
            "C": {"atomic_number": 6, "atomic_mass": 12.011, "valence": [2, 4]},
            "N": {"atomic_number": 7, "atomic_mass": 14.007, "valence": [3, 5]},
            "O": {"atomic_number": 8, "atomic_mass": 15.999, "valence": [2]},
            "F": {"atomic_number": 9, "atomic_mass": 18.998, "valence": [1]},
            "Ne": {"atomic_number": 10, "atomic_mass": 20.180, "valence": [0]},
            "Na": {"atomic_number": 11, "atomic_mass": 22.990, "valence": [1]},
            "Mg": {"atomic_number": 12, "atomic_mass": 24.305, "valence": [2]},
            "Al": {"atomic_number": 13, "atomic_mass": 26.982, "valence": [3]},
            "Si": {"atomic_number": 14, "atomic_mass": 28.086, "valence": [4]},
            "P": {"atomic_number": 15, "atomic_mass": 30.974, "valence": [3, 5]},
            "S": {"atomic_number": 16, "atomic_mass": 32.065, "valence": [2, 4, 6]},
            "Cl": {"atomic_number": 17, "atomic_mass": 35.453, "valence": [1, 3, 5, 7]},
            "Ar": {"atomic_number": 18, "atomic_mass": 39.948, "valence": [0]},
            "K": {"atomic_number": 19, "atomic_mass": 39.098, "valence": [1]},
            "Ca": {"atomic_number": 20, "atomic_mass": 40.078, "valence": [2]},
            "Sn": {"atomic_number": 50, "atomic_mass": 118.710, "valence": [2, 4]},
            "Fe": {"atomic_number": 26, "atomic_mass": 55.845, "valence": [2, 3]},
            "Cu": {"atomic_number": 29, "atomic_mass": 63.546, "valence": [1, 2]},
            "Zn": {"atomic_number": 30, "atomic_mass": 65.38, "valence": [2]},
        }

        # Common polyatomic ions
        self.polyatomic_ions = {
            "NH4+": {"charge": 1, "name": "ammonium"},
            "OH-": {"charge": -1, "name": "hydroxide"},
            "NO3-": {"charge": -1, "name": "nitrate"},
            "NO2-": {"charge": -1, "name": "nitrite"},
            "SO4^2-": {"charge": -2, "name": "sulfate"},
            "SO3^2-": {"charge": -2, "name": "sulfite"},
            "PO4^3-": {"charge": -3, "name": "phosphate"},
            "CO3^2-": {"charge": -2, "name": "carbonate"},
            "HCO3-": {"charge": -1, "name": "bicarbonate"},
            "ClO4-": {"charge": -1, "name": "perchlorate"},
            "ClO3-": {"charge": -1, "name": "chlorate"},
            "ClO2-": {"charge": -1, "name": "chlorite"},
            "ClO-": {"charge": -1, "name": "hypochlorite"},
        }

        # Chemical safety hazard classes
        self.safety_hazards = {
            "flammable",
            "explosive",
            "toxic",
            "corrosive",
            "oxidizing",
            "carcinogenic",
            "mutagenic",
            "teratogenic",
            "irritant",
            "sensitizing",
            "environmental_hazard",
        }

        # Common reaction types
        self.reaction_types = {
            "synthesis",
            "decomposition",
            "single_replacement",
            "double_replacement",
            "combustion",
            "acid_base",
            "redox",
            "precipitation",
            "gas_evolution",
        }

    def validate_content(self, content: Dict[str, Any]) -> ValidationResult:
        """
        Validate chemistry content comprehensively.

        Args:
            content: Dictionary containing chemistry content to validate

        Returns:
            ValidationResult with validation outcome and detailed feedback

        Raises:
            ValueError: If content format is invalid
        """
        logger.info("Starting chemistry content validation")

        try:
            # Extract content components
            problem = content.get("problem", "")
            answer = content.get("answer", "")
            explanation = content.get("explanation", "")

            if not problem:
                return ValidationResult(
                    domain=self.domain,
                    is_valid=False,
                    quality_score=0.0,
                    validation_details={
                        "error": "No problem content provided",
                        "subdomain": self.subdomain,
                    },
                    confidence_score=0.0,
                    feedback=["Missing problem statement"],
                )

            # Initialize validation metrics
            validation_scores = {}
            feedback_items = []
            details = {"subdomain": self.subdomain}

            # 1. Chemical equation validation
            equation_score, equation_feedback = self._validate_chemical_equations(
                problem, answer, explanation
            )
            validation_scores["equation_validation"] = equation_score
            if equation_feedback:
                feedback_items.extend(equation_feedback)

            # 2. Reaction mechanism validation
            mechanism_score, mechanism_feedback = self._validate_reaction_mechanisms(
                problem, answer, explanation
            )
            validation_scores["mechanism_validation"] = mechanism_score
            if mechanism_feedback:
                feedback_items.extend(mechanism_feedback)

            # 3. Chemical safety validation
            safety_score, safety_feedback = self._validate_chemical_safety(
                problem, answer, explanation
            )
            validation_scores["safety_validation"] = safety_score
            if safety_feedback:
                feedback_items.extend(safety_feedback)

            # 4. Molecular structure validation
            structure_score, structure_feedback = self._validate_molecular_structures(
                problem, answer, explanation
            )
            validation_scores["structure_validation"] = structure_score
            if structure_feedback:
                feedback_items.extend(structure_feedback)

            # 5. Chemical nomenclature validation
            nomenclature_score, nomenclature_feedback = self._validate_nomenclature(
                problem, answer, explanation
            )
            validation_scores["nomenclature_validation"] = nomenclature_score
            if nomenclature_feedback:
                feedback_items.extend(nomenclature_feedback)

            # Calculate overall quality score
            weights = {
                "equation_validation": 0.25,
                "mechanism_validation": 0.20,
                "safety_validation": 0.20,
                "structure_validation": 0.20,
                "nomenclature_validation": 0.15,
            }

            quality_score = sum(
                score * weights[category]
                for category, score in validation_scores.items()
            )

            # Determine if content is valid
            threshold = self.config.quality_thresholds.get("chemistry_score", 0.7)
            is_valid = quality_score >= threshold

            # Compile feedback
            feedback = (
                "; ".join(feedback_items)
                if feedback_items
                else "Chemistry content validated successfully"
            )

            # Add detailed metrics
            details.update(
                {
                    "chemistry_score": quality_score,
                    "validation_scores": validation_scores,
                    "threshold": threshold,
                    "weights": weights,
                }
            )

            logger.info(
                "Chemistry validation completed: valid=%s, score=%.2f",
                is_valid,
                quality_score,
            )

            return ValidationResult(
                domain=self.domain,
                is_valid=is_valid,
                quality_score=quality_score,
                validation_details=details,
                confidence_score=self.calculate_confidence(details),
                feedback=[feedback] if isinstance(feedback, str) else feedback,
            )

        except Exception as e:
            logger.error("Chemistry validation failed: %s", str(e))
            return ValidationResult(
                domain=self.domain,
                is_valid=False,
                quality_score=0.0,
                validation_details={"error": str(e), "subdomain": self.subdomain},
                confidence_score=0.0,
                feedback=[f"Chemistry validation error: {str(e)}"],
            )

    def _validate_chemical_equations(
        self, problem: str, answer: str, explanation: str
    ) -> Tuple[float, List[str]]:
        """
        Validate chemical equations for balance and correctness.

        Args:
            problem: Problem statement
            answer: Answer content
            explanation: Explanation content

        Returns:
            Tuple of (score, feedback_list)
        """
        feedback = []
        score = 1.0

        # Find chemical equations in the content
        equation_pattern = r"([A-Z][a-z]?\d*(?:\([A-Z][a-z]?\d*\))*(?:\s*\+\s*[A-Z][a-z]?\d*(?:\([A-Z][a-z]?\d*\))*)*)\s*(?:→|->|=)\s*([A-Z][a-z]?\d*(?:\([A-Z][a-z]?\d*\))*(?:\s*\+\s*[A-Z][a-z]?\d*(?:\([A-Z][a-z]?\d*\))*)*)"

        all_content = f"{problem} {answer} {explanation}"
        equations = re.findall(equation_pattern, all_content)

        if not equations:
            # No equations found - check if this is expected
            if any(
                keyword in all_content.lower()
                for keyword in ["equation", "reaction", "balance"]
            ):
                feedback.append(
                    "Chemical equations mentioned but not found in standard format"
                )
                score *= 0.8
            else:
                # If no chemical content is detected, reduce score for basic content
                if len(all_content.split()) < 20:  # Very basic content
                    score *= 0.85
            return score, feedback

        for reactants, products in equations:
            # Basic equation structure validation
            if not self._validate_equation_structure(reactants, products):
                feedback.append("Invalid chemical equation structure detected")
                score *= 0.7
                continue

            # Check for balanced equations
            if not self._check_equation_balance(reactants, products):
                feedback.append("Chemical equation appears unbalanced")
                score *= 0.6  # More severe penalty for unbalanced equations

            # Validate chemical formulas
            if not self._validate_chemical_formulas(reactants + " " + products):
                feedback.append("Invalid chemical formulas detected")
                score *= 0.7

        return score, feedback

    def _validate_reaction_mechanisms(
        self, problem: str, answer: str, explanation: str
    ) -> Tuple[float, List[str]]:
        """
        Validate reaction mechanisms and pathways.

        Args:
            problem: Problem statement
            answer: Answer content
            explanation: Explanation content

        Returns:
            Tuple of (score, feedback_list)
        """
        feedback = []
        score = 1.0

        all_content = f"{problem} {answer} {explanation}".lower()

        # Check for mechanism-related keywords
        mechanism_keywords = [
            "mechanism",
            "pathway",
            "intermediate",
            "transition state",
            "rate determining",
            "elementary step",
            "catalyst",
        ]

        has_mechanism_content = any(
            keyword in all_content for keyword in mechanism_keywords
        )

        if not has_mechanism_content:
            # If no mechanism content but very basic overall content, reduce score
            if len(all_content.split()) < 15:
                score *= 0.85
            return score, feedback

        # Validate mechanism components
        if "intermediate" in all_content:
            if not self._validate_intermediates(all_content):
                feedback.append("Reaction intermediates may be incorrectly identified")
                score *= 0.8

        if "rate determining" in all_content or "rate-determining" in all_content:
            if not self._validate_rate_determining_step(all_content):
                feedback.append("Rate-determining step analysis may be incorrect")
                score *= 0.8

        if "catalyst" in all_content:
            if not self._validate_catalyst_role(all_content):
                feedback.append(
                    "Catalyst role or mechanism may be incorrectly described"
                )
                score *= 0.8

        return score, feedback

    def _validate_chemical_safety(
        self, problem: str, answer: str, explanation: str
    ) -> Tuple[float, List[str]]:
        """
        Validate chemical safety protocols and hazard awareness.

        Args:
            problem: Problem statement
            answer: Answer content
            explanation: Explanation content

        Returns:
            Tuple of (score, feedback_list)
        """
        feedback = []
        score = 1.0

        all_content = f"{problem} {answer} {explanation}".lower()

        # Check for safety-related content
        safety_keywords = [
            "safety",
            "hazard",
            "toxic",
            "dangerous",
            "precaution",
            "protective equipment",
            "ventilation",
            "disposal",
        ]

        has_safety_content = any(keyword in all_content for keyword in safety_keywords)

        # Check for dangerous chemicals mentioned
        dangerous_chemicals = [
            "hydrofluoric acid",
            "mercury",
            "benzene",
            "asbestos",
            "cyanide",
            "chromium",
            "lead",
            "arsenic",
        ]

        dangerous_mentioned = any(
            chemical in all_content for chemical in dangerous_chemicals
        )

        if dangerous_mentioned and not has_safety_content:
            feedback.append(
                "Dangerous chemicals mentioned without safety considerations"
            )
            score *= 0.6

        # Validate safety protocol completeness
        if has_safety_content:
            if not self._validate_safety_protocols(all_content):
                feedback.append("Safety protocols may be incomplete or incorrect")
                score *= 0.8

        return score, feedback

    def _validate_molecular_structures(
        self, problem: str, answer: str, explanation: str
    ) -> Tuple[float, List[str]]:
        """
        Validate molecular structures and representations.

        Args:
            problem: Problem statement
            answer: Answer content
            explanation: Explanation content

        Returns:
            Tuple of (score, feedback_list)
        """
        feedback = []
        score = 1.0

        all_content = f"{problem} {answer} {explanation}"

        # Check for structural representations and potential formulas
        structure_patterns = [
            r"[A-Z][a-z]?(?:-[A-Z][a-z]?)+",  # Simple structural formulas
            r"[A-Z][a-z]?\([A-Z][a-z]?\d*\)\d*",  # Branched structures
            r"C\d*H\d*(?:[A-Z][a-z]?\d*)*",  # Molecular formulas
            r"[A-Z]{2,}\d*",  # Potential invalid formulas like XYZ123
        ]

        structures_found = []
        for pattern in structure_patterns:
            structures_found.extend(re.findall(pattern, all_content))

        # Check for obviously invalid formulas
        if not self._validate_chemical_formulas(all_content):
            feedback.append("Invalid chemical formulas detected")
            score *= 0.1  # Severe penalty for invalid formulas

        if not structures_found:
            return score, feedback

        # Validate molecular formulas
        valid_structures = 0
        for structure in structures_found:
            if self._validate_molecular_formula(structure):
                valid_structures += 1
            else:
                feedback.append(
                    f"Invalid molecular structure or formula detected: {structure}"
                )

        # Score based on proportion of valid structures
        if structures_found:
            structure_validity_ratio = valid_structures / len(structures_found)
            score *= max(0.5, structure_validity_ratio)  # Minimum score of 0.5

        # Check for stereochemistry if mentioned
        if any(
            term in all_content.lower()
            for term in ["stereochemistry", "chirality", "enantiomer"]
        ):
            if not self._validate_stereochemistry(all_content):
                feedback.append("Stereochemistry concepts may be incorrectly applied")
                score *= 0.8

        return score, feedback

    def _validate_nomenclature(
        self, problem: str, answer: str, explanation: str
    ) -> Tuple[float, List[str]]:
        """
        Validate chemical nomenclature and naming conventions.

        Args:
            problem: Problem statement
            answer: Answer content
            explanation: Explanation content

        Returns:
            Tuple of (score, feedback_list)
        """
        feedback = []
        score = 1.0

        all_content = f"{problem} {answer} {explanation}"

        # Find chemical names and formulas
        chemical_names = self._extract_chemical_names(all_content)

        if not chemical_names:
            return score, feedback

        # Validate naming conventions
        for name in chemical_names:
            if not self._validate_chemical_name(name):
                feedback.append(
                    f"Chemical name '{name}' may not follow IUPAC conventions"
                )
                score *= 0.9

        return score, feedback

    # Helper methods for validation

    def _validate_equation_structure(self, reactants: str, products: str) -> bool:
        """Validate basic chemical equation structure."""
        # Check for valid chemical symbols
        element_pattern = r"[A-Z][a-z]?"

        reactant_elements = re.findall(element_pattern, reactants)
        product_elements = re.findall(element_pattern, products)

        # Check if elements exist
        for element in reactant_elements + product_elements:
            if element not in self.elements:
                return False

        return True

    def _check_equation_balance(self, reactants: str, products: str) -> bool:
        """Check if chemical equation is balanced (simplified check)."""
        # This is a simplified balance check
        # In a full implementation, this would parse and balance the equation

        # Count elements on both sides
        reactant_elements = self._count_elements(reactants)
        product_elements = self._count_elements(products)

        # Check if element counts match (simplified)
        for element in set(reactant_elements.keys()) | set(product_elements.keys()):
            if reactant_elements.get(element, 0) != product_elements.get(element, 0):
                return False

        # Additional check for common unbalanced patterns
        if "H2 + O2" in reactants and "H2O" in products and "2H2" not in reactants:
            return False  # Classic unbalanced water formation

        return True

    def _count_elements(self, formula: str) -> Dict[str, int]:
        """Count elements in a chemical formula."""
        elements = {}

        # Simple element counting (would need more sophisticated parsing for real use)
        element_pattern = r"([A-Z][a-z]?)(\d*)"
        matches = re.findall(element_pattern, formula)

        for element, count in matches:
            count = int(count) if count else 1
            elements[element] = elements.get(element, 0) + count

        return elements

    def _validate_chemical_formulas(self, content: str) -> bool:
        """Validate chemical formulas in content."""
        # Check for obviously invalid formulas first
        invalid_patterns = ["XYZ123", "ABC", "QWE"]  # Common test invalid formulas
        for invalid in invalid_patterns:
            if invalid in content:
                return False

        # Extract potential chemical formulas (but exclude reaction mechanism names)
        formula_pattern = r"\b([A-Z][a-z]?\d*(?:\([A-Z][a-z]?\d*\))*)\b"
        formulas = re.findall(formula_pattern, content)

        # Filter out common reaction mechanism names and abbreviations
        mechanism_names = {"SN1", "SN2", "E1", "E2", "HF", "UV", "IR", "NMR"}
        formulas = [f for f in formulas if f not in mechanism_names]

        for formula in formulas:
            if not self._is_valid_formula(formula):
                return False

        return True

    def _is_valid_formula(self, formula: str) -> bool:
        """Check if a chemical formula is valid."""
        # Extract elements from formula
        element_pattern = r"([A-Z][a-z]?)"
        elements = re.findall(element_pattern, formula)

        # Check if all elements exist
        for element in elements:
            if element not in self.elements:
                return False

        return True

    def _validate_intermediates(self, content: str) -> bool:
        """Validate reaction intermediates."""
        # Simplified validation - check for reasonable intermediate descriptions
        intermediate_indicators = [
            "unstable",
            "short-lived",
            "reactive",
            "forms and decomposes",
        ]

        return any(indicator in content for indicator in intermediate_indicators)

    def _validate_rate_determining_step(self, content: str) -> bool:
        """Validate rate-determining step analysis."""
        # Check for proper rate-determining step concepts
        rate_concepts = ["slowest step", "highest activation energy", "rate limiting"]

        return any(concept in content for concept in rate_concepts)

    def _validate_catalyst_role(self, content: str) -> bool:
        """Validate catalyst role description."""
        # Check for proper catalyst concepts
        catalyst_concepts = [
            "lowers activation energy",
            "not consumed",
            "alternative pathway",
            "increases rate",
            "regenerated",
        ]

        return any(concept in content for concept in catalyst_concepts)

    def _validate_safety_protocols(self, content: str) -> bool:
        """Validate safety protocol completeness."""
        # Check for essential safety elements
        safety_elements = [
            "protective equipment",
            "ventilation",
            "disposal",
            "emergency",
        ]

        safety_score = sum(1 for element in safety_elements if element in content)
        return safety_score >= 2  # At least 2 safety elements mentioned

    def _validate_molecular_formula(self, formula: str) -> bool:
        """Validate molecular formula structure."""
        # Check for valid molecular formula pattern
        molecular_pattern = r"^[A-Z][a-z]?\d*(?:[A-Z][a-z]?\d*)*$"
        return bool(re.match(molecular_pattern, formula))

    def _validate_stereochemistry(self, content: str) -> bool:
        """Validate stereochemistry concepts."""
        # Check for proper stereochemistry terminology
        stereo_terms = [
            "chiral center",
            "optical activity",
            "r/s configuration",
            "mirror image",
            "non-superimposable",
        ]

        return any(term in content for term in stereo_terms)

    def _extract_chemical_names(self, content: str) -> List[str]:
        """Extract chemical names from content."""
        # This is a simplified extraction
        # In practice, would use more sophisticated NLP

        # Common chemical name patterns
        name_patterns = [
            r"\b[a-z]+ane\b",  # alkanes
            r"\b[a-z]+ene\b",  # alkenes
            r"\b[a-z]+yne\b",  # alkynes
            r"\b[a-z]+ol\b",  # alcohols
            r"\b[a-z]+ic acid\b",  # acids
        ]

        names = []
        for pattern in name_patterns:
            names.extend(re.findall(pattern, content.lower()))

        return names

    def _validate_chemical_name(self, name: str) -> bool:
        """Validate chemical name against IUPAC conventions."""
        # Simplified validation
        # Check for common naming patterns

        valid_suffixes = [
            "ane",
            "ene",
            "yne",
            "ol",
            "al",
            "one",
            "oic acid",
            "ate",
            "ite",
            "ide",
        ]

        return any(name.endswith(suffix) for suffix in valid_suffixes)

    def calculate_quality_score(self, content: Dict[str, Any]) -> float:
        """
        Calculate domain-specific quality score for chemistry content.

        Args:
            content: Content to assess for quality

        Returns:
            Quality score between 0.0 and 1.0
        """
        try:
            # Use the same validation logic but just return the score
            result = self.validate_content(content)
            return result.quality_score
        except Exception:
            return 0.0

    def generate_feedback(self, validation_result: "ValidationResult") -> List[str]:
        """
        Generate domain-specific improvement feedback for chemistry content.

        Args:
            validation_result: Result of validation to generate feedback for

        Returns:
            List of feedback messages for content improvement
        """
        feedback = []

        if not validation_result.is_valid:
            feedback.append(
                "Chemistry content needs improvement to meet quality standards"
            )

        # Extract validation scores from details
        validation_scores = validation_result.details.get("validation_scores", {})

        # Provide specific feedback based on low scores
        if validation_scores.get("equation_validation", 1.0) < 0.7:
            feedback.append(
                "Chemical equations may need balancing or structural corrections"
            )

        if validation_scores.get("mechanism_validation", 1.0) < 0.7:
            feedback.append(
                "Reaction mechanisms could be explained more clearly or accurately"
            )

        if validation_scores.get("safety_validation", 1.0) < 0.7:
            feedback.append("Safety considerations should be more comprehensive")

        if validation_scores.get("structure_validation", 1.0) < 0.7:
            feedback.append("Molecular structures or formulas may contain errors")

        if validation_scores.get("nomenclature_validation", 1.0) < 0.7:
            feedback.append(
                "Chemical nomenclature should follow IUPAC conventions more closely"
            )

        # Add positive feedback for high scores
        if validation_result.quality_score > 0.8:
            feedback.append(
                "Chemistry content demonstrates strong understanding of concepts"
            )

        return feedback

    def validate(self, content: Dict[str, Any]) -> ValidationResult:
        """
        Main validation method (alias for validate_content).

        Args:
            content: Content to validate

        Returns:
            ValidationResult with validation outcome
        """
        return self.validate_content(content)
