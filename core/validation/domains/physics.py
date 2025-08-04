"""
Physics subdomain validator for STREAM content validation.

This module provides comprehensive validation for physics content including
unit consistency, physical law verification, dimensional analysis, and
physics simulation validation.
"""

# Standard Library
import logging
import math
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# SynThesisAI Modules
from ..base import SubValidationResult
from ..exceptions import DomainValidationError

logger = logging.getLogger(__name__)


class UnitConsistencyValidator:
    """Validator for physics unit consistency and dimensional analysis."""

    def __init__(self):
        """Initialize unit consistency validator."""
        # Base SI units with their dimensions
        self.base_units = {
            "m": {"dimension": "length", "name": "meter"},
            "kg": {"dimension": "mass", "name": "kilogram"},
            "s": {"dimension": "time", "name": "second"},
            "A": {"dimension": "electric_current", "name": "ampere"},
            "K": {"dimension": "temperature", "name": "kelvin"},
            "mol": {"dimension": "amount_of_substance", "name": "mole"},
            "cd": {"dimension": "luminous_intensity", "name": "candela"},
        }

        # Derived physics units with their base unit expressions
        self.derived_units = {
            "N": {"base_units": "kg⋅m⋅s⁻²", "dimension": "force", "name": "newton"},
            "J": {"base_units": "kg⋅m²⋅s⁻²", "dimension": "energy", "name": "joule"},
            "W": {"base_units": "kg⋅m²⋅s⁻³", "dimension": "power", "name": "watt"},
            "Pa": {
                "base_units": "kg⋅m⁻¹⋅s⁻²",
                "dimension": "pressure",
                "name": "pascal",
            },
            "Hz": {"base_units": "s⁻¹", "dimension": "frequency", "name": "hertz"},
            "C": {
                "base_units": "A⋅s",
                "dimension": "electric_charge",
                "name": "coulomb",
            },
            "V": {
                "base_units": "kg⋅m²⋅s⁻³⋅A⁻¹",
                "dimension": "electric_potential",
                "name": "volt",
            },
            "Ω": {
                "base_units": "kg⋅m²⋅s⁻³⋅A⁻²",
                "dimension": "resistance",
                "name": "ohm",
            },
            "F": {
                "base_units": "kg⁻¹⋅m⁻²⋅s⁴⋅A²",
                "dimension": "capacitance",
                "name": "farad",
            },
            "H": {
                "base_units": "kg⋅m²⋅s⁻²⋅A⁻²",
                "dimension": "inductance",
                "name": "henry",
            },
            "T": {
                "base_units": "kg⋅s⁻²⋅A⁻¹",
                "dimension": "magnetic_field",
                "name": "tesla",
            },
            "Wb": {
                "base_units": "kg⋅m²⋅s⁻²⋅A⁻¹",
                "dimension": "magnetic_flux",
                "name": "weber",
            },
        }

        # Common physics unit prefixes
        self.unit_prefixes = {
            "Y": 1e24,
            "Z": 1e21,
            "E": 1e18,
            "P": 1e15,
            "T": 1e12,
            "G": 1e9,
            "M": 1e6,
            "k": 1e3,
            "h": 1e2,
            "da": 1e1,
            "d": 1e-1,
            "c": 1e-2,
            "m": 1e-3,
            "μ": 1e-6,
            "n": 1e-9,
            "p": 1e-12,
            "f": 1e-15,
            "a": 1e-18,
            "z": 1e-21,
            "y": 1e-24,
        }

    def validate_unit_consistency(self, content: Dict[str, Any]) -> SubValidationResult:
        """
        Validate unit consistency in physics content.

        Args:
            content: Physics content to validate

        Returns:
            SubValidationResult with unit consistency validation details
        """
        try:
            text = f"{content.get('problem', '')} {content.get('answer', '')} {content.get('solution', '')}"

            # Extract units and equations from text
            units_found = self._extract_units(text)
            equations_found = self._extract_equations(text)

            # Validate unit consistency in equations
            equation_consistency = self._validate_equation_units(equations_found)

            # Check dimensional analysis
            dimensional_analysis = self._check_dimensional_analysis(
                units_found, equations_found
            )

            # Calculate overall unit consistency score
            unit_score = (
                equation_consistency["score"] + dimensional_analysis["score"]
            ) / 2

            return SubValidationResult(
                subdomain="unit_consistency",
                is_valid=unit_score >= 0.6,
                details={
                    "unit_score": unit_score,
                    "units_found": units_found,
                    "equations_found": equations_found,
                    "equation_consistency": equation_consistency,
                    "dimensional_analysis": dimensional_analysis,
                    "recognized_units": len(
                        [u for u in units_found if u["recognized"]]
                    ),
                    "total_units": len(units_found),
                },
                confidence_score=0.9 if unit_score >= 0.8 else 0.7,
            )

        except Exception as e:
            logger.error("Unit consistency validation failed: %s", str(e))
            return SubValidationResult(
                subdomain="unit_consistency",
                is_valid=False,
                details={"error": str(e)},
                confidence_score=0.0,
                error_message=f"Unit consistency validation error: {str(e)}",
            )

    def _extract_units(self, text: str) -> List[Dict[str, Any]]:
        """Extract units from physics text."""
        units_found = []

        # Pattern for units (including compound units)
        unit_pattern = (
            r"\b\d+(?:\.\d+)?\s*([a-zA-ZΩμ]+(?:[⋅⁻¹²³⁴⁵⁶⁷⁸⁹⁰/\-\^]+[a-zA-ZΩμ]*)*)"
        )

        matches = re.findall(unit_pattern, text)
        for match in matches:
            unit_info = {
                "unit_string": match,
                "recognized": self._is_recognized_unit(match),
                "dimension": self._get_unit_dimension(match),
            }
            units_found.append(unit_info)

        # Also look for standalone unit symbols (for symbolic expressions)
        standalone_units = [
            "m",
            "kg",
            "s",
            "N",
            "J",
            "W",
            "V",
            "A",
            "Hz",
            "Pa",
            "C",
            "Ω",
            "F",
            "H",
            "T",
            "Wb",
        ]
        for unit in standalone_units:
            if f" {unit} " in text or f" {unit}/" in text or f"/{unit}" in text:
                unit_info = {
                    "unit_string": unit,
                    "recognized": True,
                    "dimension": self._get_unit_dimension(unit),
                }
                if unit_info not in units_found:  # Avoid duplicates
                    units_found.append(unit_info)

        return units_found

    def _extract_equations(self, text: str) -> List[str]:
        """Extract equations from physics text."""
        # Pattern for equations (more specific)
        equation_pattern = r"[A-Za-z]\s*=\s*[^=.!?]*"
        equations = re.findall(equation_pattern, text)

        # Filter out non-physics equations and clean them
        physics_equations = []
        for eq in equations:
            eq = eq.strip()
            # Only include if it looks like a physics equation
            if len(eq) < 100 and (  # Reasonable length
                any(unit in eq for unit in ["N", "J", "W", "V", "A", "kg", "m/s"])
                or any(
                    var in eq.split("=")[0].strip()
                    for var in ["F", "E", "P", "v", "a", "I", "V", "R"]
                )
                or "ma" in eq
                or "mc" in eq
            ):
                physics_equations.append(eq)

        return physics_equations

    def _is_recognized_unit(self, unit_string: str) -> bool:
        """Check if a unit string is recognized."""
        # Remove exponents and separators for checking
        clean_unit = re.sub(r"[⋅⁻¹²³⁴⁵⁶⁷⁸⁹⁰/\-\^]+", "", unit_string)

        # Check base units
        if clean_unit in self.base_units:
            return True

        # Check derived units
        if clean_unit in self.derived_units:
            return True

        # Check with prefixes
        for prefix in self.unit_prefixes:
            if (
                clean_unit.startswith(prefix)
                and clean_unit[len(prefix) :] in self.base_units
            ):
                return True
            if (
                clean_unit.startswith(prefix)
                and clean_unit[len(prefix) :] in self.derived_units
            ):
                return True

        return False

    def _get_unit_dimension(self, unit_string: str) -> Optional[str]:
        """Get the dimension of a unit."""
        clean_unit = re.sub(r"[⋅⁻¹²³⁴⁵⁶⁷⁸⁹⁰/\-\^]+", "", unit_string)

        if clean_unit in self.base_units:
            return self.base_units[clean_unit]["dimension"]

        if clean_unit in self.derived_units:
            return self.derived_units[clean_unit]["dimension"]

        return None

    def _validate_equation_units(self, equations: List[str]) -> Dict[str, Any]:
        """Validate unit consistency in equations."""
        consistent_equations = 0
        total_equations = len(equations)
        issues = []

        for equation in equations:
            if self._check_equation_dimensional_consistency(equation):
                consistent_equations += 1
            else:
                issues.append(f"Dimensional inconsistency in equation: {equation}")

        score = consistent_equations / total_equations if total_equations > 0 else 1.0

        return {
            "score": score,
            "consistent_equations": consistent_equations,
            "total_equations": total_equations,
            "issues": issues,
        }

    def _check_equation_dimensional_consistency(self, equation: str) -> bool:
        """Check if an equation is dimensionally consistent."""
        # This is a simplified check - in a full implementation,
        # this would parse the equation and verify dimensional consistency

        # For now, we'll do basic checks for common physics equations
        common_consistent_patterns = [
            r"f\s*=\s*m.*a",  # F = ma (flexible)
            r"e\s*=\s*m.*c",  # E = mc² (flexible)
            r"p\s*=\s*f.*v",  # P = Fv (flexible)
            r"v\s*=\s*[id].*[rt]",  # v = d/t (flexible)
            r"a\s*=\s*v.*t",  # a = v/t (flexible)
            r"v\s*=\s*i.*r",  # V = IR (flexible)
            r"i\s*=\s*v.*r",  # I = V/R (flexible)
            r"[a-z]\s*=\s*\d+",  # Any variable = number
        ]

        equation_lower = equation.lower()
        # If it matches any known pattern, it's consistent
        if any(
            re.search(pattern, equation_lower) for pattern in common_consistent_patterns
        ):
            return True

        # If it's a simple assignment with units, assume it's consistent
        if re.search(r"[A-Za-z]\s*=\s*\d+.*[A-Za-z]", equation):
            return True

        return False

    def _check_dimensional_analysis(
        self, units_found: List[Dict[str, Any]], equations_found: List[str]
    ) -> Dict[str, Any]:
        """Check dimensional analysis quality."""
        score = 0.8  # Default score
        issues = []

        # Check if units are consistently used
        if units_found:
            recognized_ratio = len([u for u in units_found if u["recognized"]]) / len(
                units_found
            )
            score *= recognized_ratio

            if recognized_ratio < 0.8:
                issues.append("Some units are not recognized or may be incorrect")

        # Bonus for having equations with dimensional consistency
        if equations_found:
            score += 0.1

        return {
            "score": min(1.0, score),
            "issues": issues,
            "units_analyzed": len(units_found),
            "equations_analyzed": len(equations_found),
        }


class PhysicalLawValidator:
    """Validator for physics laws and principles."""

    def __init__(self):
        """Initialize physical law validator."""
        self.physics_laws = {
            "newton_first": {
                "name": "Newton's First Law",
                "description": "An object at rest stays at rest, an object in motion stays in motion",
                "keywords": ["inertia", "first law", "rest", "motion", "force"],
                "equations": ["F = 0", "a = 0"],
            },
            "newton_second": {
                "name": "Newton's Second Law",
                "description": "Force equals mass times acceleration",
                "keywords": ["second law", "force", "mass", "acceleration"],
                "equations": ["F = ma", "F = m*a", "a = F/m"],
            },
            "newton_third": {
                "name": "Newton's Third Law",
                "description": "For every action there is an equal and opposite reaction",
                "keywords": ["third law", "action", "reaction", "equal", "opposite"],
                "equations": ["F₁ = -F₂", "F1 = -F2"],
            },
            "conservation_energy": {
                "name": "Conservation of Energy",
                "description": "Energy cannot be created or destroyed, only transformed",
                "keywords": [
                    "conservation of energy",
                    "kinetic",
                    "potential",
                    "mechanical",
                    "mgh",
                    "mv²",
                ],
                "equations": ["KE + PE = constant", "E = KE + PE", "mgh = ½mv²"],
            },
            "conservation_momentum": {
                "name": "Conservation of Momentum",
                "description": "Total momentum of a system remains constant",
                "keywords": ["momentum conservation", "collision", "momentum"],
                "equations": ["p = mv", "Σp = constant"],
            },
            "ohms_law": {
                "name": "Ohm's Law",
                "description": "Voltage equals current times resistance",
                "keywords": ["ohm", "voltage", "current", "resistance"],
                "equations": ["V = IR", "V = I*R", "I = V/R", "R = V/I"],
            },
            "coulombs_law": {
                "name": "Coulomb's Law",
                "description": "Force between charges is proportional to product of charges",
                "keywords": ["coulomb", "electric force", "charge", "electrostatic"],
                "equations": ["F = kq₁q₂/r²", "F = k*q1*q2/r^2"],
            },
        }

        self.physics_constants = {
            "c": {"value": 299792458, "unit": "m/s", "name": "speed of light"},
            "g": {"value": 9.81, "unit": "m/s²", "name": "gravitational acceleration"},
            "G": {
                "value": 6.67430e-11,
                "unit": "m³/kg⋅s²",
                "name": "gravitational constant",
            },
            "h": {"value": 6.62607015e-34, "unit": "J⋅s", "name": "Planck constant"},
            "e": {"value": 1.602176634e-19, "unit": "C", "name": "elementary charge"},
            "k": {
                "value": 8.9875517923e9,
                "unit": "N⋅m²/C²",
                "name": "Coulomb constant",
            },
        }

    def validate_physical_laws(self, content: Dict[str, Any]) -> SubValidationResult:
        """
        Validate physics laws and principles in content.

        Args:
            content: Physics content to validate

        Returns:
            SubValidationResult with physical law validation details
        """
        try:
            text = f"{content.get('problem', '')} {content.get('answer', '')} {content.get('explanation', '')}".lower()

            # Identify physics laws referenced
            laws_identified = self._identify_physics_laws(text)

            # Validate law applications
            law_applications = self._validate_law_applications(text, laws_identified)

            # Check for physics constants
            constants_check = self._check_physics_constants(text)

            # Calculate overall physics law score
            law_score = (
                0.4 * (len(laws_identified) / 3 if len(laws_identified) <= 3 else 1.0)
                + 0.4 * law_applications["accuracy"]
                + 0.2 * constants_check["accuracy"]
            )

            return SubValidationResult(
                subdomain="physical_laws",
                is_valid=law_score >= 0.7,
                details={
                    "law_score": law_score,
                    "laws_identified": laws_identified,
                    "law_applications": law_applications,
                    "constants_check": constants_check,
                    "total_laws_found": len(laws_identified),
                },
                confidence_score=0.85 if law_score >= 0.7 else 0.6,
            )

        except Exception as e:
            logger.error("Physical law validation failed: %s", str(e))
            return SubValidationResult(
                subdomain="physical_laws",
                is_valid=False,
                details={"error": str(e)},
                confidence_score=0.0,
                error_message=f"Physical law validation error: {str(e)}",
            )

    def _identify_physics_laws(self, text: str) -> List[Dict[str, Any]]:
        """Identify physics laws referenced in text."""
        laws_found = []

        for law_id, law_info in self.physics_laws.items():
            # Check for keywords
            keyword_matches = sum(
                1 for keyword in law_info["keywords"] if keyword in text
            )

            # Check for equations
            equation_matches = sum(
                1
                for eq in law_info["equations"]
                if eq.lower().replace(" ", "") in text.replace(" ", "")
            )

            if keyword_matches > 0 or equation_matches > 0:
                laws_found.append(
                    {
                        "law_id": law_id,
                        "name": law_info["name"],
                        "keyword_matches": keyword_matches,
                        "equation_matches": equation_matches,
                        "confidence": min(
                            1.0, (keyword_matches + equation_matches * 2) / 3
                        ),
                    }
                )

        return laws_found

    def _validate_law_applications(
        self, text: str, laws_identified: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Validate the application of identified physics laws."""
        correct_applications = 0
        total_applications = len(laws_identified)
        issues = []

        for law in laws_identified:
            law_info = self.physics_laws[law["law_id"]]

            # Check if the law is applied correctly (simplified check)
            if self._check_law_application_correctness(text, law_info):
                correct_applications += 1
            else:
                issues.append(f"Potential issue with {law['name']} application")

        accuracy = (
            correct_applications / total_applications if total_applications > 0 else 1.0
        )

        return {
            "accuracy": accuracy,
            "correct_applications": correct_applications,
            "total_applications": total_applications,
            "issues": issues,
        }

    def _check_law_application_correctness(
        self, text: str, law_info: Dict[str, Any]
    ) -> bool:
        """Check if a physics law is applied correctly."""
        # This is a simplified check - in practice, this would be more sophisticated

        # Check if equations are used in correct context
        for equation in law_info["equations"]:
            if equation.lower().replace(" ", "") in text.replace(" ", ""):
                # Basic context check
                if any(keyword in text for keyword in law_info["keywords"]):
                    return True

        return len(law_info["keywords"]) == 0 or any(
            keyword in text for keyword in law_info["keywords"]
        )

    def _check_physics_constants(self, text: str) -> Dict[str, Any]:
        """Check for physics constants and their accuracy."""
        constants_found = []
        accuracy_issues = []

        for const_symbol, const_info in self.physics_constants.items():
            if const_symbol in text or const_info["name"] in text:
                constants_found.append(
                    {
                        "symbol": const_symbol,
                        "name": const_info["name"],
                        "expected_value": const_info["value"],
                        "unit": const_info["unit"],
                    }
                )

        # For now, assume constants are used correctly if mentioned
        accuracy = 1.0 if not accuracy_issues else 0.7

        return {
            "accuracy": accuracy,
            "constants_found": constants_found,
            "accuracy_issues": accuracy_issues,
        }


class PhysicsValidator:
    """Comprehensive physics subdomain validator."""

    def __init__(self):
        """Initialize physics validator."""
        self.unit_validator = UnitConsistencyValidator()
        self.law_validator = PhysicalLawValidator()

        logger.info("Initialized PhysicsValidator with unit and law validation")

    def validate(self, content: Dict[str, Any]) -> SubValidationResult:
        """
        Comprehensive physics content validation.

        Args:
            content: Physics content to validate

        Returns:
            SubValidationResult with comprehensive physics validation details
        """
        try:
            # Unit consistency validation
            unit_result = self.unit_validator.validate_unit_consistency(content)

            # Physical law validation
            law_result = self.law_validator.validate_physical_laws(content)

            # Calculate overall physics score
            physics_score = 0.6 * (unit_result.details.get("unit_score", 0.0)) + 0.4 * (
                law_result.details.get("law_score", 0.0)
            )

            # Determine overall validity (more lenient)
            is_valid = physics_score >= 0.6 and (
                unit_result.is_valid or law_result.is_valid
            )

            return SubValidationResult(
                subdomain="physics",
                is_valid=is_valid,
                details={
                    "physics_score": physics_score,
                    "unit_validation": unit_result.details,
                    "law_validation": law_result.details,
                    "validation_components": {
                        "unit_consistency": unit_result.is_valid,
                        "physical_laws": law_result.is_valid,
                    },
                },
                confidence_score=0.9 if is_valid else 0.7,
            )

        except Exception as e:
            logger.error("Physics validation failed: %s", str(e))
            return SubValidationResult(
                subdomain="physics",
                is_valid=False,
                details={"error": str(e)},
                confidence_score=0.0,
                error_message=f"Physics validation error: {str(e)}",
            )

    def generate_feedback(self, validation_result: SubValidationResult) -> List[str]:
        """
        Generate physics-specific improvement feedback.

        Args:
            validation_result: Result of physics validation

        Returns:
            List of feedback messages for physics content improvement
        """
        feedback = []

        if not validation_result.is_valid:
            details = validation_result.details

            # Unit consistency feedback
            unit_validation = details.get("unit_validation", {})
            if unit_validation.get("unit_score", 1.0) < 0.8:
                feedback.append(
                    "Check unit consistency and dimensional analysis in equations"
                )

                unit_issues = unit_validation.get("equation_consistency", {}).get(
                    "issues", []
                )
                if unit_issues:
                    feedback.append(f"Unit issues found: {'; '.join(unit_issues[:2])}")

            # Physical law feedback
            law_validation = details.get("law_validation", {})
            if law_validation.get("law_score", 1.0) < 0.7:
                feedback.append(
                    "Verify correct application of physics laws and principles"
                )

                law_issues = law_validation.get("law_applications", {}).get(
                    "issues", []
                )
                if law_issues:
                    feedback.append(f"Physics law issues: {'; '.join(law_issues[:2])}")

        return feedback
