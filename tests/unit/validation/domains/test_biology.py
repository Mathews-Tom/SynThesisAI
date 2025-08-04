"""
Unit tests for the Biology domain validator.

This module tests all aspects of biology content validation including
biological processes, taxonomic accuracy, biological ethics, and
biological system models.
"""

# Standard Library
from unittest.mock import patch

# Third-Party Library
import pytest

# SynThesisAI Modules
from core.validation.config import ValidationConfig
from core.validation.domains.biology import BiologyValidator


class TestBiologyValidator:
    """Test suite for BiologyValidator class."""

    @pytest.fixture
    def validator(self):
        """Create a biology validator instance for testing."""
        config = ValidationConfig(
            domain="science", quality_thresholds={"biology_score": 0.7}
        )
        return BiologyValidator("biology", config)

    @pytest.fixture
    def genetics_validator(self):
        """Create a genetics validator instance."""
        config = ValidationConfig(
            domain="science", quality_thresholds={"biology_score": 0.7}
        )
        return BiologyValidator("genetics", config)

    def test_validator_initialization(self, validator):
        """Test biology validator initialization."""
        assert validator.domain == "science"
        assert validator.subdomain == "biology"
        assert validator.config is not None
        assert len(validator.biological_processes) > 0
        assert len(validator.taxonomic_levels) > 0

    def test_invalid_subdomain_raises_error(self):
        """Test that invalid subdomain raises ValueError."""
        config = ValidationConfig(domain="science")

        with pytest.raises(ValueError, match="Invalid biology subdomain"):
            BiologyValidator("invalid_subdomain", config)

    def test_valid_subdomains(self):
        """Test that all valid biology subdomains work."""
        config = ValidationConfig(domain="science")
        valid_subdomains = [
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
        ]

        for subdomain in valid_subdomains:
            validator = BiologyValidator(subdomain, config)
            assert validator.subdomain == subdomain

    def test_validate_photosynthesis_process(self, validator):
        """Test validation of photosynthesis biological process."""
        content = {
            "problem": "Explain the process of photosynthesis in plants.",
            "answer": "Photosynthesis converts carbon dioxide and water into glucose using sunlight and chlorophyll.",
            "explanation": "The process occurs in chloroplasts where light energy is captured by chlorophyll. Carbon dioxide from the air and water from the roots combine to produce glucose and oxygen. ATP and NADPH are also produced during the light reactions.",
        }

        result = validator.validate_content(content)

        assert result.domain == "science"
        assert result.validation_details["subdomain"] == "biology"
        assert result.is_valid is True
        assert result.quality_score > 0.7
        assert "process_validation" in result.validation_details["validation_scores"]

    def test_validate_cellular_respiration(self, validator):
        """Test validation of cellular respiration process."""
        content = {
            "problem": "Describe cellular respiration and its importance.",
            "answer": "Cellular respiration breaks down glucose using oxygen to produce ATP energy.",
            "explanation": "This process occurs in mitochondria where glucose and oxygen react to produce carbon dioxide, water, and ATP. The citric acid cycle and electron transport chain are key components of this process.",
        }

        result = validator.validate_content(content)

        assert result.is_valid is True
        assert result.quality_score > 0.6
        assert "process_validation" in result.validation_details["validation_scores"]

    def test_validate_taxonomic_classification(self, validator):
        """Test validation of taxonomic classification."""
        content = {
            "problem": "Classify humans using taxonomic hierarchy.",
            "answer": "Humans belong to the species Homo sapiens.",
            "explanation": "The full classification is: Kingdom Animalia, Phylum Chordata, Class Mammalia, Order Primates, Family Hominidae, Genus Homo, Species sapiens. This follows the standard taxonomic hierarchy from broad to specific categories.",
        }

        result = validator.validate_content(content)

        assert result.is_valid is True
        assert "taxonomy_validation" in result.validation_details["validation_scores"]
        assert (
            result.validation_details["validation_scores"]["taxonomy_validation"] > 0.7
        )

    def test_validate_binomial_nomenclature(self, validator):
        """Test validation of binomial nomenclature format."""
        content = {
            "problem": "What is the scientific name for the domestic dog?",
            "answer": "Canis familiaris",
            "explanation": "The scientific name follows binomial nomenclature where Canis is the genus and familiaris is the species name.",
        }

        result = validator.validate_content(content)

        assert result.is_valid is True
        assert result.quality_score > 0.7

    def test_validate_biological_ethics_animal_research(self, validator):
        """Test validation when animal research ethics are mentioned."""
        content = {
            "problem": "What ethical considerations apply to animal research?",
            "answer": "Animal research must follow the 3Rs principle and ensure animal welfare.",
            "explanation": "The 3Rs stand for Replacement, Reduction, and Refinement. Researchers must minimize animal use, replace animals when possible, and refine procedures to reduce suffering. Animal welfare committees oversee research protocols.",
        }

        result = validator.validate_content(content)

        assert result.is_valid is True
        assert "ethics_validation" in result.validation_details["validation_scores"]
        assert result.validation_details["validation_scores"]["ethics_validation"] > 0.7

    def test_validate_ecosystem_components(self, validator):
        """Test validation of ecosystem biological system."""
        content = {
            "problem": "Describe the components of an ecosystem.",
            "answer": "Ecosystems contain producers, consumers, and decomposers interacting with abiotic factors.",
            "explanation": "Producers like plants convert sunlight to energy. Primary and secondary consumers form food chains. Decomposers recycle nutrients. Abiotic factors include temperature, water, and soil that influence all organisms.",
        }

        result = validator.validate_content(content)

        assert result.is_valid is True
        assert "system_validation" in result.validation_details["validation_scores"]
        assert result.validation_details["validation_scores"]["system_validation"] > 0.7

    def test_validate_cell_structure(self, validator):
        """Test validation of cell biological system."""
        content = {
            "problem": "What are the main components of a cell?",
            "answer": "Cells have a membrane, nucleus, and cytoplasm with various organelles.",
            "explanation": "The cell membrane controls what enters and exits. The nucleus contains DNA. Organelles like mitochondria produce energy, ribosomes make proteins, and the endoplasmic reticulum transports materials.",
        }

        result = validator.validate_content(content)

        assert result.is_valid is True
        assert result.quality_score > 0.7

    def test_validate_evolution_concepts(self, validator):
        """Test validation of evolution and natural selection."""
        content = {
            "problem": "Explain how natural selection leads to evolution.",
            "answer": "Natural selection acts on variation within populations to increase fitness over time.",
            "explanation": "Individuals with favorable traits have higher survival and reproduction rates. These traits are inherited by offspring. Over many generations, beneficial traits become more common in the population, leading to evolutionary change.",
        }

        result = validator.validate_content(content)

        assert result.is_valid is True
        assert result.quality_score > 0.7

    def test_validate_dna_replication(self, validator):
        """Test validation of DNA replication process."""
        content = {
            "problem": "Describe the process of DNA replication.",
            "answer": "DNA replication produces two identical copies of the double helix using DNA polymerase.",
            "explanation": "The process begins when helicase unwinds the double helix. Primase adds RNA primers, then DNA polymerase adds nucleotides to form new strands. Ligase joins DNA fragments to complete replication in the nucleus during S phase.",
        }

        result = validator.validate_content(content)

        assert result.is_valid is True
        assert "process_validation" in result.validation_details["validation_scores"]

    def test_validate_scientific_accuracy_misconceptions(self, validator):
        """Test detection of biological misconceptions."""
        content = {
            "problem": "Is evolution just a theory?",
            "answer": "Evolution is just a theory, so it's not proven.",
            "explanation": "Since evolution is called a theory, it means scientists aren't sure about it.",
        }

        result = validator.validate_content(content)

        # Should detect misconception and lower accuracy score
        accuracy_score = result.validation_details["validation_scores"][
            "accuracy_validation"
        ]
        assert accuracy_score < 0.9
        feedback_text = " ".join(result.feedback).lower()
        assert "misconception" in feedback_text or "theory" in feedback_text

    def test_validate_empty_content(self, validator):
        """Test validation with empty content."""
        content = {"problem": "", "answer": "", "explanation": ""}

        result = validator.validate_content(content)

        assert result.is_valid is False
        assert result.quality_score == 0.0
        assert "Missing problem statement" in result.feedback

    def test_validate_content_without_problem(self, validator):
        """Test validation when problem is missing."""
        content = {"answer": "Some answer", "explanation": "Some explanation"}

        result = validator.validate_content(content)

        assert result.is_valid is False
        assert "Missing problem statement" in result.feedback

    def test_validation_scoring_weights(self, validator):
        """Test that validation scoring uses correct weights."""
        content = {
            "problem": "Test biology problem about photosynthesis and taxonomy",
            "answer": "Plants use photosynthesis. Homo sapiens is the scientific name for humans.",
            "explanation": "This covers biological processes and taxonomic classification",
        }

        result = validator.validate_content(content)

        # Check that weights are applied correctly
        expected_weights = {
            "process_validation": 0.25,
            "taxonomy_validation": 0.20,
            "ethics_validation": 0.15,
            "system_validation": 0.20,
            "accuracy_validation": 0.20,
        }

        assert result.validation_details["weights"] == expected_weights

    def test_quality_threshold_application(self, validator):
        """Test that quality threshold is applied correctly."""
        # Set a high threshold
        validator.config.quality_thresholds["biology_score"] = 0.95

        content = {
            "problem": "Simple biology question",
            "answer": "Simple answer",
            "explanation": "Basic explanation",
        }

        result = validator.validate_content(content)

        # Should fail with high threshold
        assert result.is_valid is False
        assert result.validation_details["threshold"] == 0.95

    def test_genetics_subdomain(self, genetics_validator):
        """Test genetics-specific validation."""
        content = {
            "problem": "Explain DNA structure and function.",
            "answer": "DNA is a double helix containing genetic information in nucleotide sequences.",
            "explanation": "DNA consists of four nucleotides (A, T, G, C) that pair specifically. The sequence of nucleotides encodes genetic information that is transcribed to RNA and translated to proteins.",
        }

        result = genetics_validator.validate_content(content)

        assert result.validation_details["subdomain"] == "genetics"
        assert result.is_valid is True

    def test_validation_error_handling(self, validator):
        """Test error handling during validation."""
        # Test with invalid content type
        with patch.object(
            validator,
            "_validate_biological_processes",
            side_effect=Exception("Test error"),
        ):
            content = {
                "problem": "Test problem",
                "answer": "Test answer",
                "explanation": "Test explanation",
            }

            result = validator.validate_content(content)

            assert result.is_valid is False
            assert result.quality_score == 0.0
            assert "validation error" in str(result.feedback).lower()
            assert "error" in result.validation_details

    def test_binomial_nomenclature_validation(self, validator):
        """Test binomial nomenclature validation helper method."""
        # Valid binomial names
        assert validator._validate_binomial_nomenclature("Homo sapiens") is True
        assert validator._validate_binomial_nomenclature("Escherichia coli") is True

        # Invalid binomial names
        assert (
            validator._validate_binomial_nomenclature("homo sapiens") is False
        )  # genus not capitalized
        assert (
            validator._validate_binomial_nomenclature("Homo Sapiens") is False
        )  # species capitalized
        assert validator._validate_binomial_nomenclature("Homo") is False  # only genus
        assert (
            validator._validate_binomial_nomenclature("homo sapiens sapiens") is False
        )  # three parts

    def test_validate_alias_method(self, validator):
        """Test that validate method is an alias for validate_content."""
        content = {
            "problem": "Test problem",
            "answer": "Test answer",
            "explanation": "Test explanation",
        }

        result1 = validator.validate_content(content)
        result2 = validator.validate(content)

        # Results should be identical
        assert result1.is_valid == result2.is_valid
        assert result1.quality_score == result2.quality_score
        assert result1.feedback == result2.feedback

    def test_protein_synthesis_validation(self, validator):
        """Test validation of protein synthesis process."""
        content = {
            "problem": "Describe protein synthesis in cells.",
            "answer": "Protein synthesis involves transcription and translation using mRNA, tRNA, and ribosomes.",
            "explanation": "First, DNA is transcribed to mRNA in the nucleus by RNA polymerase. Then mRNA travels to ribosomes where tRNA brings amino acids. The ribosome reads codons and tRNA anticodons match to build the protein chain.",
        }

        result = validator.validate_content(content)

        assert result.is_valid is True
        assert result.quality_score > 0.7
        assert "process_validation" in result.validation_details["validation_scores"]

    def test_mitosis_vs_meiosis_validation(self, validator):
        """Test validation of cell division processes."""
        content = {
            "problem": "Compare mitosis and meiosis.",
            "answer": "Mitosis produces identical diploid cells, while meiosis produces genetically diverse haploid gametes.",
            "explanation": "Mitosis has phases like prophase, metaphase, anaphase, and telophase, producing two identical cells. Meiosis includes crossing over and independent assortment, creating four genetically different gametes for reproduction.",
        }

        result = validator.validate_content(content)

        assert result.is_valid is True
        assert result.quality_score > 0.7

    def test_ecological_interactions_validation(self, validator):
        """Test validation of ecological system interactions."""
        content = {
            "problem": "Explain food webs and energy flow in ecosystems.",
            "answer": "Food webs show complex feeding relationships with energy flowing from producers to consumers.",
            "explanation": "Energy flows from producers through primary and secondary consumers to decomposers. Food webs are more realistic than simple food chains because organisms often eat multiple food sources. Energy decreases at each trophic level.",
        }

        result = validator.validate_content(content)

        assert result.is_valid is True
        assert "system_validation" in result.validation_details["validation_scores"]
        assert result.validation_details["validation_scores"]["system_validation"] > 0.7


class TestBiologyValidatorIntegration:
    """Integration tests for biology validator."""

    def test_comprehensive_biology_problem(self):
        """Test validation of a comprehensive biology problem."""
        config = ValidationConfig(
            domain="science", quality_thresholds={"biology_score": 0.6}
        )
        validator = BiologyValidator("biology", config)

        content = {
            "problem": "Explain photosynthesis, its cellular location, and ecological importance. Include proper scientific terminology and consider any ethical aspects of plant research.",
            "answer": "Photosynthesis occurs in chloroplasts where chlorophyll captures light energy to convert CO2 and H2O into glucose and O2. This process is fundamental to ecosystems as it provides energy for food webs.",
            "explanation": "The light reactions occur in thylakoids producing ATP and NADPH, while the Calvin cycle in the stroma fixes carbon dioxide. Ecologically, photosynthesis supports all life by producing oxygen and organic compounds. Research on plants follows ethical guidelines for environmental protection and sustainable practices.",
        }

        result = validator.validate_content(content)

        assert result.is_valid is True
        assert result.quality_score > 0.6
        assert all(
            score > 0
            for score in result.validation_details["validation_scores"].values()
        )

    def test_biology_problem_with_all_components(self):
        """Test biology problem covering all validation aspects."""
        config = ValidationConfig(
            domain="science", quality_thresholds={"biology_score": 0.5}
        )
        validator = BiologyValidator("molecular_biology", config)

        content = {
            "problem": "Describe DNA replication in Escherichia coli, including enzyme functions, cellular location, and research ethics considerations.",
            "answer": "DNA replication in E. coli involves helicase, primase, DNA polymerase, and ligase working in the bacterial cell during chromosome replication.",
            "explanation": "The process begins at the origin of replication where helicase unwinds DNA. Primase adds RNA primers, DNA polymerase III synthesizes new strands, and ligase joins fragments. This occurs in the bacterial cytoplasm. Research with E. coli follows biosafety protocols and ethical guidelines for genetic manipulation studies.",
        }

        result = validator.validate_content(content)

        assert result.is_valid is True
        assert result.validation_details["subdomain"] == "molecular_biology"

        # Check all validation components were evaluated
        validation_scores = result.validation_details["validation_scores"]
        assert "process_validation" in validation_scores
        assert "taxonomy_validation" in validation_scores
        assert "ethics_validation" in validation_scores
        assert "system_validation" in validation_scores
        assert "accuracy_validation" in validation_scores
