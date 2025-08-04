"""
Biology subdomain validator for the SynThesisAI platform.

This module implements comprehensive validation for biology-related content,
including biological processes, taxonomic accuracy, biological ethics, and
biological system models.
"""

# Standard Library
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

# SynThesisAI Modules
from core.validation.base import DomainValidator, ValidationResult
from core.validation.config import ValidationConfig

logger = logging.getLogger(__name__)


class BiologyValidator(DomainValidator):
    """
    Validator for biology content with comprehensive biological validation.

    This validator handles:
    - Biological process validation
    - Taxonomic accuracy verification
    - Biological ethics validation
    - Biological system model validation
    - Scientific accuracy in biological contexts
    """

    def __init__(
        self, subdomain: str = "biology", config: Optional[ValidationConfig] = None
    ):
        """
        Initialize the biology validator.

        Args:
            subdomain: The biology subdomain (default: "biology")
            config: Validation configuration settings

        Raises:
            ValueError: If subdomain is not biology-related
        """
        super().__init__("science", config)
        self.subdomain = subdomain

        # Validate subdomain
        valid_subdomains = {
            "biology",
            "molecular_biology",
            "cell_biology",
            "genetics",
            "ecology",
            "evolution",
            "anatomy",
            "physiology",
            "microbiology",
            "botany",
            "zoology",
            "biochemistry",
        }
        if subdomain not in valid_subdomains:
            raise ValueError(f"Invalid biology subdomain: {subdomain}")

        # Initialize biological knowledge bases
        self._initialize_biological_data()

        logger.info("Initialized BiologyValidator for subdomain: %s", subdomain)

    def _initialize_biological_data(self) -> None:
        """Initialize biological data and knowledge bases."""
        # Biological processes and their key components
        self.biological_processes = {
            "photosynthesis": {
                "keywords": [
                    "chlorophyll",
                    "glucose",
                    "carbon dioxide",
                    "sunlight",
                    "ATP",
                    "NADPH",
                ],
                "equation": "6CO2 + 6H2O + light energy → C6H12O6 + 6O2",
                "location": ["chloroplast", "leaf", "plant cell"],
            },
            "cellular_respiration": {
                "keywords": [
                    "oxygen",
                    "glucose",
                    "ATP",
                    "mitochondria",
                    "pyruvate",
                    "citric acid cycle",
                ],
                "equation": "C6H12O6 + 6O2 → 6CO2 + 6H2O + ATP",
                "location": ["mitochondria", "cell", "cytoplasm"],
            },
            "dna_replication": {
                "keywords": [
                    "DNA",
                    "polymerase",
                    "nucleotides",
                    "double helix",
                    "primer",
                    "ligase",
                ],
                "components": ["helicase", "primase", "DNA polymerase", "ligase"],
                "location": ["nucleus", "cell cycle", "S phase"],
            },
            "protein_synthesis": {
                "keywords": [
                    "mRNA",
                    "tRNA",
                    "ribosome",
                    "amino acids",
                    "translation",
                    "transcription",
                ],
                "components": ["RNA polymerase", "ribosome", "codon", "anticodon"],
                "location": ["nucleus", "ribosome", "endoplasmic reticulum"],
            },
            "mitosis": {
                "keywords": [
                    "chromosome",
                    "spindle",
                    "centromere",
                    "cytokinesis",
                    "prophase",
                    "metaphase",
                ],
                "phases": ["prophase", "metaphase", "anaphase", "telophase"],
                "location": ["nucleus", "cell division", "somatic cells"],
            },
            "meiosis": {
                "keywords": [
                    "gametes",
                    "crossing over",
                    "independent assortment",
                    "haploid",
                    "diploid",
                ],
                "phases": ["prophase I", "metaphase I", "anaphase I", "telophase I"],
                "location": ["reproductive cells", "gonads", "germ cells"],
            },
        }

        # Taxonomic hierarchy and classification
        self.taxonomic_levels = [
            "domain",
            "kingdom",
            "phylum",
            "class",
            "order",
            "family",
            "genus",
            "species",
        ]

        self.taxonomic_examples = {
            "homo_sapiens": {
                "domain": "eukarya",
                "kingdom": "animalia",
                "phylum": "chordata",
                "class": "mammalia",
                "order": "primates",
                "family": "hominidae",
                "genus": "homo",
                "species": "sapiens",
            },
            "escherichia_coli": {
                "domain": "bacteria",
                "kingdom": "bacteria",
                "phylum": "proteobacteria",
                "class": "gammaproteobacteria",
                "order": "enterobacteriales",
                "family": "enterobacteriaceae",
                "genus": "escherichia",
                "species": "coli",
            },
        }

        # Biological ethics considerations
        self.ethics_keywords = {
            "animal_research": [
                "animal welfare",
                "3Rs",
                "replacement",
                "reduction",
                "refinement",
            ],
            "genetic_engineering": [
                "informed consent",
                "safety",
                "environmental impact",
                "regulation",
            ],
            "human_subjects": [
                "informed consent",
                "IRB",
                "privacy",
                "confidentiality",
                "risk assessment",
            ],
            "environmental": [
                "conservation",
                "biodiversity",
                "sustainability",
                "ecosystem protection",
            ],
        }

        # Biological system models and their components
        self.biological_systems = {
            "ecosystem": {
                "components": [
                    "producers",
                    "consumers",
                    "decomposers",
                    "abiotic factors",
                ],
                "interactions": [
                    "food chain",
                    "food web",
                    "energy flow",
                    "nutrient cycling",
                ],
                "levels": [
                    "individual",
                    "population",
                    "community",
                    "ecosystem",
                    "biosphere",
                ],
            },
            "cell": {
                "components": ["membrane", "nucleus", "cytoplasm", "organelles"],
                "organelles": [
                    "mitochondria",
                    "chloroplast",
                    "ribosome",
                    "endoplasmic reticulum",
                ],
                "processes": ["metabolism", "growth", "reproduction", "response"],
            },
            "organ_system": {
                "circulatory": ["heart", "blood vessels", "blood", "circulation"],
                "respiratory": ["lungs", "trachea", "bronchi", "gas exchange"],
                "digestive": ["stomach", "intestines", "liver", "digestion"],
                "nervous": ["brain", "spinal cord", "neurons", "synapses"],
            },
        }

        # Evolution concepts and evidence
        self.evolution_concepts = {
            "natural_selection": [
                "variation",
                "inheritance",
                "selection pressure",
                "fitness",
            ],
            "evidence": [
                "fossil record",
                "comparative anatomy",
                "molecular biology",
                "biogeography",
            ],
            "mechanisms": [
                "mutation",
                "gene flow",
                "genetic drift",
                "natural selection",
            ],
            "speciation": [
                "reproductive isolation",
                "geographic isolation",
                "adaptive radiation",
            ],
        }

    def validate_content(self, content: Dict[str, Any]) -> ValidationResult:
        """
        Validate biology content comprehensively.

        Args:
            content: Dictionary containing biology content to validate

        Returns:
            ValidationResult with validation outcome and detailed feedback

        Raises:
            ValueError: If content format is invalid
        """
        logger.info("Starting biology content validation")

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

            # 1. Biological process validation
            process_score, process_feedback = self._validate_biological_processes(
                problem, answer, explanation
            )
            validation_scores["process_validation"] = process_score
            if process_feedback:
                feedback_items.extend(process_feedback)

            # 2. Taxonomic accuracy validation
            taxonomy_score, taxonomy_feedback = self._validate_taxonomic_accuracy(
                problem, answer, explanation
            )
            validation_scores["taxonomy_validation"] = taxonomy_score
            if taxonomy_feedback:
                feedback_items.extend(taxonomy_feedback)

            # 3. Biological ethics validation
            ethics_score, ethics_feedback = self._validate_biological_ethics(
                problem, answer, explanation
            )
            validation_scores["ethics_validation"] = ethics_score
            if ethics_feedback:
                feedback_items.extend(ethics_feedback)

            # 4. Biological system model validation
            system_score, system_feedback = self._validate_biological_systems(
                problem, answer, explanation
            )
            validation_scores["system_validation"] = system_score
            if system_feedback:
                feedback_items.extend(system_feedback)

            # 5. Scientific accuracy validation
            accuracy_score, accuracy_feedback = self._validate_scientific_accuracy(
                problem, answer, explanation
            )
            validation_scores["accuracy_validation"] = accuracy_score
            if accuracy_feedback:
                feedback_items.extend(accuracy_feedback)

            # Calculate overall quality score
            weights = {
                "process_validation": 0.25,
                "taxonomy_validation": 0.20,
                "ethics_validation": 0.15,
                "system_validation": 0.20,
                "accuracy_validation": 0.20,
            }

            quality_score = sum(
                score * weights[category]
                for category, score in validation_scores.items()
            )

            # Determine if content is valid
            threshold = self.config.quality_thresholds.get("biology_score", 0.7)
            is_valid = quality_score >= threshold

            # Compile feedback
            feedback = (
                "; ".join(feedback_items)
                if feedback_items
                else "Biology content validated successfully"
            )

            # Add detailed metrics
            details.update(
                {
                    "biology_score": quality_score,
                    "validation_scores": validation_scores,
                    "threshold": threshold,
                    "weights": weights,
                }
            )

            logger.info(
                "Biology validation completed: valid=%s, score=%.2f",
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
            logger.error("Biology validation failed: %s", str(e))
            return ValidationResult(
                domain=self.domain,
                is_valid=False,
                quality_score=0.0,
                validation_details={"error": str(e), "subdomain": self.subdomain},
                confidence_score=0.0,
                feedback=[f"Biology validation error: {str(e)}"],
            )

    def _validate_biological_processes(
        self, problem: str, answer: str, explanation: str
    ) -> Tuple[float, List[str]]:
        """
        Validate biological processes and their accuracy.

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

        # Check for biological process keywords
        processes_found = []
        for process_name, process_data in self.biological_processes.items():
            keywords = process_data["keywords"]
            keyword_matches = sum(
                1 for keyword in keywords if keyword.lower() in all_content
            )

            if keyword_matches >= 2:  # At least 2 keywords must match
                processes_found.append((process_name, keyword_matches, len(keywords)))

        if not processes_found:
            # If no biological processes detected, check for basic content
            if len(all_content.split()) < 20:
                score *= 0.85
            return score, feedback

        # Validate each identified process
        for process_name, matches, total_keywords in processes_found:
            process_data = self.biological_processes[process_name]

            # Check keyword coverage
            keyword_coverage = matches / total_keywords
            if keyword_coverage < 0.5:
                feedback.append(
                    f"{process_name.replace('_', ' ').title()} process may be incomplete"
                )
                score *= 0.9

            # Check for process equation if applicable
            if "equation" in process_data:
                equation = process_data["equation"]
                # Simple check for key components of the equation
                equation_components = re.findall(r"[A-Z][A-Za-z0-9]*", equation)
                component_matches = sum(
                    1 for comp in equation_components if comp.lower() in all_content
                )

                if component_matches < len(equation_components) * 0.6:
                    feedback.append(
                        f"Chemical equation for {process_name.replace('_', ' ')} may be missing or incorrect"
                    )
                    score *= 0.9

            # Check for correct location/context
            if "location" in process_data:
                locations = process_data["location"]
                location_matches = sum(
                    1 for loc in locations if loc.lower() in all_content
                )

                if location_matches == 0:
                    feedback.append(
                        f"Cellular/tissue location for {process_name.replace('_', ' ')} not specified"
                    )
                    score *= 0.95

        return score, feedback

    def _validate_taxonomic_accuracy(
        self, problem: str, answer: str, explanation: str
    ) -> Tuple[float, List[str]]:
        """
        Validate taxonomic classification and accuracy.

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

        # Check for taxonomic hierarchy usage
        taxonomy_levels_found = [
            level for level in self.taxonomic_levels if level in all_content
        ]

        if not taxonomy_levels_found:
            # Check for species names or classification context
            species_indicators = [
                "species",
                "genus",
                "family",
                "classification",
                "taxonomy",
            ]
            if any(indicator in all_content for indicator in species_indicators):
                feedback.append(
                    "Taxonomic classification mentioned but hierarchy not clearly specified"
                )
                score *= 0.9
            return score, feedback

        # Validate taxonomic hierarchy order
        if len(taxonomy_levels_found) >= 2:
            # Check if levels are mentioned in logical order
            level_positions = [
                (level, all_content.find(level)) for level in taxonomy_levels_found
            ]
            level_positions.sort(key=lambda x: x[1])  # Sort by position in text

            expected_order = [
                level
                for level in self.taxonomic_levels
                if level in taxonomy_levels_found
            ]
            actual_order = [level for level, _ in level_positions]

            if actual_order != expected_order:
                feedback.append(
                    "Taxonomic hierarchy may not be presented in correct order"
                )
                score *= 0.9

        # Check for binomial nomenclature format
        binomial_pattern = r"\b[A-Z][a-z]+ [a-z]+\b"
        binomial_matches = re.findall(
            binomial_pattern, f"{problem} {answer} {explanation}"
        )

        if binomial_matches:
            # Validate binomial nomenclature format
            for match in binomial_matches:
                if not self._validate_binomial_nomenclature(match):
                    feedback.append(
                        f"Binomial nomenclature '{match}' may not follow correct format"
                    )
                    score *= 0.95

        return score, feedback

    def _validate_biological_ethics(
        self, problem: str, answer: str, explanation: str
    ) -> Tuple[float, List[str]]:
        """
        Validate biological ethics considerations.

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

        # Check for ethics-sensitive topics
        ethics_topics_found = []
        for topic, keywords in self.ethics_keywords.items():
            topic_indicators = [
                "animal",
                "experiment",
                "research",
                "genetic",
                "human",
                "subject",
                "environment",
                "conservation",
                "ethics",
                "welfare",
            ]

            if any(indicator in all_content for indicator in topic_indicators):
                keyword_matches = sum(
                    1 for keyword in keywords if keyword.lower() in all_content
                )
                if keyword_matches > 0:
                    ethics_topics_found.append((topic, keyword_matches, len(keywords)))

        if not ethics_topics_found:
            # Check if ethics should be considered
            sensitive_terms = [
                "experiment",
                "research",
                "genetic modification",
                "animal",
                "human",
            ]
            if any(term in all_content for term in sensitive_terms):
                feedback.append("Ethical considerations may need to be addressed")
                score *= 0.9
            return score, feedback

        # Validate ethics coverage for identified topics
        for topic, matches, total_keywords in ethics_topics_found:
            coverage = matches / total_keywords
            if coverage < 0.3:
                feedback.append(
                    f"Ethical considerations for {topic.replace('_', ' ')} may be insufficient"
                )
                score *= 0.85

        return score, feedback

    def _validate_biological_systems(
        self, problem: str, answer: str, explanation: str
    ) -> Tuple[float, List[str]]:
        """
        Validate biological system models and their components.

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

        # Check for biological system references
        systems_found = []
        for system_name, system_data in self.biological_systems.items():
            if system_name in all_content or any(
                comp.lower() in all_content
                for comp in system_data.get("components", [])
            ):
                systems_found.append(system_name)

        if not systems_found:
            # Check for basic biological content
            if len(all_content.split()) < 15:
                score *= 0.85
            return score, feedback

        # Validate each identified system
        for system_name in systems_found:
            system_data = self.biological_systems[system_name]

            # Check component coverage
            if "components" in system_data:
                components = system_data["components"]
                component_matches = sum(
                    1 for comp in components if comp.lower() in all_content
                )
                component_coverage = component_matches / len(components)

                if component_coverage < 0.4:
                    feedback.append(
                        f"{system_name.replace('_', ' ').title()} system components may be incomplete"
                    )
                    score *= 0.9

            # Check for system interactions or processes
            if "interactions" in system_data:
                interactions = system_data["interactions"]
                interaction_matches = sum(
                    1 for inter in interactions if inter.lower() in all_content
                )

                if interaction_matches == 0 and len(interactions) > 0:
                    feedback.append(
                        f"Interactions within {system_name.replace('_', ' ')} system not described"
                    )
                    score *= 0.95

        return score, feedback

    def _validate_scientific_accuracy(
        self, problem: str, answer: str, explanation: str
    ) -> Tuple[float, List[str]]:
        """
        Validate scientific accuracy in biological context.

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

        # Check for common biological misconceptions
        misconceptions = {
            "evolution": ["evolution is just a theory", "humans evolved from monkeys"],
            "genetics": ["one gene one trait", "genetic determinism"],
            "ecology": ["balance of nature", "survival of the fittest means strongest"],
        }

        for topic, misconception_list in misconceptions.items():
            if topic in all_content:
                for misconception in misconception_list:
                    if misconception.lower() in all_content:
                        feedback.append(
                            f"Potential misconception about {topic}: {misconception}"
                        )
                        score *= 0.8

        # Check for appropriate use of scientific terminology
        scientific_terms = [
            "hypothesis",
            "theory",
            "experiment",
            "control",
            "variable",
            "correlation",
            "causation",
            "evidence",
            "peer review",
        ]

        term_usage = sum(1 for term in scientific_terms if term in all_content)
        if term_usage == 0 and len(all_content.split()) > 30:
            feedback.append(
                "Scientific methodology terminology could be more prominent"
            )
            score *= 0.95

        # Check for quantitative vs qualitative descriptions
        quantitative_indicators = [
            "percent",
            "%",
            "ratio",
            "rate",
            "measurement",
            "data",
            "statistics",
        ]
        has_quantitative = any(
            indicator in all_content for indicator in quantitative_indicators
        )

        if not has_quantitative and any(
            word in all_content for word in ["study", "research", "experiment"]
        ):
            feedback.append(
                "Quantitative aspects of biological research could be emphasized"
            )
            score *= 0.95

        return score, feedback

    # Helper methods

    def _validate_binomial_nomenclature(self, name: str) -> bool:
        """Validate binomial nomenclature format."""
        parts = name.split()
        if len(parts) != 2:
            return False

        genus, species = parts
        # Genus should start with capital, species should be lowercase
        return genus[0].isupper() and genus[1:].islower() and species.islower()

    def calculate_quality_score(self, content: Dict[str, Any]) -> float:
        """
        Calculate domain-specific quality score for biology content.

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
        Generate domain-specific improvement feedback for biology content.

        Args:
            validation_result: Result of validation to generate feedback for

        Returns:
            List of feedback messages for content improvement
        """
        feedback = []

        if not validation_result.is_valid:
            feedback.append(
                "Biology content needs improvement to meet quality standards"
            )

        # Extract validation scores from details
        validation_scores = validation_result.details.get("validation_scores", {})

        # Provide specific feedback based on low scores
        if validation_scores.get("process_validation", 1.0) < 0.7:
            feedback.append(
                "Biological processes could be described more accurately or completely"
            )

        if validation_scores.get("taxonomy_validation", 1.0) < 0.7:
            feedback.append(
                "Taxonomic classification should follow proper hierarchy and nomenclature"
            )

        if validation_scores.get("ethics_validation", 1.0) < 0.7:
            feedback.append(
                "Biological ethics considerations should be more comprehensive"
            )

        if validation_scores.get("system_validation", 1.0) < 0.7:
            feedback.append(
                "Biological system models could include more components and interactions"
            )

        if validation_scores.get("accuracy_validation", 1.0) < 0.7:
            feedback.append("Scientific accuracy and methodology could be improved")

        # Add positive feedback for high scores
        if validation_result.quality_score > 0.8:
            feedback.append(
                "Biology content demonstrates strong understanding of biological concepts"
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
