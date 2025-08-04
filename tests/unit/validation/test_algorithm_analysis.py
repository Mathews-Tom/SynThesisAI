"""
Unit tests for the algorithm analysis validation module.

This module tests all aspects of algorithm analysis including
complexity analysis, pattern detection, correctness verification,
and optimization suggestions.
"""

# Third-Party Library
import pytest

# SynThesisAI Modules
from core.validation.algorithm_analysis import (
    AlgorithmAnalyzer,
    AlgorithmPattern,
    AlgorithmValidator,
    ComplexityAnalysis,
)


class TestComplexityAnalysis:
    """Test suite for ComplexityAnalysis dataclass."""

    def test_complexity_analysis_creation(self):
        """Test creation of ComplexityAnalysis object."""
        analysis = ComplexityAnalysis(
            time_complexity="O(n)",
            space_complexity="O(1)",
            confidence=0.8,
            reasoning=["Linear loop detected"],
            optimization_suggestions=["Consider using hash table"],
        )

        assert analysis.time_complexity == "O(n)"
        assert analysis.space_complexity == "O(1)"
        assert analysis.confidence == 0.8
        assert len(analysis.reasoning) == 1
        assert len(analysis.optimization_suggestions) == 1


class TestAlgorithmPattern:
    """Test suite for AlgorithmPattern dataclass."""

    def test_algorithm_pattern_creation(self):
        """Test creation of AlgorithmPattern object."""
        pattern = AlgorithmPattern(
            pattern_name="divide_and_conquer",
            confidence=0.9,
            characteristics=["recursive", "merge"],
            typical_complexity="O(n log n)",
        )

        assert pattern.pattern_name == "divide_and_conquer"
        assert pattern.confidence == 0.9
        assert "recursive" in pattern.characteristics
        assert pattern.typical_complexity == "O(n log n)"


class TestAlgorithmAnalyzer:
    """Test suite for AlgorithmAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create an algorithm analyzer instance for testing."""
        return AlgorithmAnalyzer()

    def test_analyzer_initialization(self, analyzer):
        """Test algorithm analyzer initialization."""
        assert len(analyzer.complexity_patterns) > 0
        assert len(analyzer.algorithm_patterns) > 0
        assert len(analyzer.known_algorithms) > 0
        assert "O(1)" in analyzer.complexity_patterns
        assert "O(n)" in analyzer.complexity_patterns
        assert "O(n^2)" in analyzer.complexity_patterns

    def test_analyze_linear_algorithm(self, analyzer):
        """Test analysis of linear time algorithm."""
        linear_code = """
def find_max(arr):
    max_val = arr[0]
    for item in arr:
        if item > max_val:
            max_val = item
    return max_val
"""

        result = analyzer.analyze_algorithm(linear_code, "python")

        assert "complexity_analysis" in result
        complexity = result["complexity_analysis"]
        assert complexity.time_complexity == "O(n)"
        assert complexity.confidence > 0.0

    def test_analyze_quadratic_algorithm(self, analyzer):
        """Test analysis of quadratic time algorithm."""
        quadratic_code = """
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr
"""

        result = analyzer.analyze_algorithm(quadratic_code, "python")

        assert "complexity_analysis" in result
        complexity = result["complexity_analysis"]
        assert complexity.time_complexity == "O(n^2)"
        assert complexity.confidence > 0.0

    def test_analyze_logarithmic_algorithm(self, analyzer):
        """Test analysis of logarithmic time algorithm."""
        binary_search_code = """
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
"""

        result = analyzer.analyze_algorithm(binary_search_code, "python")

        assert "complexity_analysis" in result
        complexity = result["complexity_analysis"]
        assert complexity.time_complexity == "O(log n)"
        assert complexity.confidence > 0.0

    def test_detect_divide_and_conquer_pattern(self, analyzer):
        """Test detection of divide and conquer pattern."""
        merge_sort_code = """
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    return merge(left, right)
"""

        result = analyzer.analyze_algorithm(merge_sort_code, "python")

        assert "algorithm_patterns" in result
        patterns = result["algorithm_patterns"]
        pattern_names = [p.pattern_name for p in patterns]
        assert "divide_and_conquer" in pattern_names

    def test_detect_dynamic_programming_pattern(self, analyzer):
        """Test detection of dynamic programming pattern."""
        dp_code = """
def fibonacci_dp(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fibonacci_dp(n-1, memo) + fibonacci_dp(n-2, memo)
    return memo[n]
"""

        result = analyzer.analyze_algorithm(dp_code, "python")

        assert "algorithm_patterns" in result
        patterns = result["algorithm_patterns"]
        pattern_names = [p.pattern_name for p in patterns]
        assert "dynamic_programming" in pattern_names

    def test_identify_known_algorithms(self, analyzer):
        """Test identification of known algorithms."""
        binary_search_code = """
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
"""

        result = analyzer.analyze_algorithm(binary_search_code, "python")

        assert "known_algorithms" in result
        algorithms = result["known_algorithms"]
        algorithm_names = [algo["name"] for algo in algorithms]
        assert "binary_search" in algorithm_names

    def test_analyze_space_complexity_recursive(self, analyzer):
        """Test space complexity analysis for recursive algorithms."""
        recursive_code = """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
"""

        space_complexity = analyzer._analyze_space_complexity(recursive_code, "python")
        assert space_complexity == "O(n)"

    def test_analyze_space_complexity_iterative(self, analyzer):
        """Test space complexity analysis for iterative algorithms."""
        iterative_code = """
def factorial(n):
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result
"""

        space_complexity = analyzer._analyze_space_complexity(iterative_code, "python")
        assert space_complexity == "O(1)"

    def test_analyze_correctness_with_base_case(self, analyzer):
        """Test correctness analysis for recursive function with base case."""
        recursive_code = """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
"""

        correctness = analyzer._analyze_correctness(recursive_code, "python")

        assert correctness["has_base_case"] is True
        assert len(correctness["potential_issues"]) == 0

    def test_analyze_correctness_missing_base_case(self, analyzer):
        """Test correctness analysis for recursive function missing base case."""
        bad_recursive_code = """
def factorial(n):
    return n * factorial(n - 1)
"""

        correctness = analyzer._analyze_correctness(bad_recursive_code, "python")

        assert correctness["has_base_case"] is False
        assert len(correctness["potential_issues"]) > 0
        assert any("base case" in issue for issue in correctness["potential_issues"])

    def test_generate_optimization_suggestions(self, analyzer):
        """Test generation of optimization suggestions."""
        inefficient_code = """
def find_duplicates(arr):
    duplicates = []
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            if arr[i] == arr[j] and arr[i] not in duplicates:
                duplicates.append(arr[i])
    return duplicates
"""

        suggestions = analyzer._generate_optimization_suggestions(
            inefficient_code, "python"
        )

        assert len(suggestions) > 0
        assert any("nested loops" in suggestion.lower() for suggestion in suggestions)

    def test_extract_complexity_from_description(self, analyzer):
        """Test extraction of complexity from algorithm description."""
        description1 = "This algorithm runs in O(n log n) time complexity"
        complexity1 = analyzer._extract_complexity_from_description(description1)
        assert complexity1 == "O(n log n)"

        description2 = "The time complexity is O(n^2) in the worst case"
        complexity2 = analyzer._extract_complexity_from_description(description2)
        assert complexity2 == "O(n^2)"

        description3 = "This is a linear algorithm"
        complexity3 = analyzer._extract_complexity_from_description(description3)
        assert complexity3 is None

    def test_validate_against_description(self, analyzer):
        """Test validation of algorithm against description."""
        code = """
def linear_search(arr, target):
    for i, item in enumerate(arr):
        if item == target:
            return i
    return -1
"""

        description = "This is a linear search algorithm with O(n) time complexity"

        # First analyze the algorithm
        analysis_result = analyzer.analyze_algorithm(code, "python")

        # Then validate against description
        validation = analyzer._validate_against_description(
            code, description, analysis_result
        )

        assert "complexity_matches" in validation
        assert "description_accuracy" in validation
        assert validation["description_accuracy"] > 0.0

    def test_analyze_performance_characteristics(self, analyzer):
        """Test analysis of performance characteristics."""
        merge_sort_code = """
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    return merge(left, right)
"""

        characteristics = analyzer._analyze_performance_characteristics(
            merge_sort_code, "python"
        )

        assert "is_stable" in characteristics
        assert "is_in_place" in characteristics
        assert "memory_access_pattern" in characteristics

    def test_get_pattern_complexity(self, analyzer):
        """Test getting typical complexity for algorithm patterns."""
        assert analyzer._get_pattern_complexity("divide_and_conquer") == "O(n log n)"
        assert analyzer._get_pattern_complexity("two_pointers") == "O(n)"
        assert analyzer._get_pattern_complexity("backtracking") == "O(2^n)"
        assert analyzer._get_pattern_complexity("unknown_pattern") == "O(n)"


class TestAlgorithmValidator:
    """Test suite for AlgorithmValidator class."""

    @pytest.fixture
    def validator(self):
        """Create an algorithm validator instance for testing."""
        return AlgorithmValidator()

    def test_validator_initialization(self, validator):
        """Test algorithm validator initialization."""
        assert validator.analyzer is not None
        assert isinstance(validator.analyzer, AlgorithmAnalyzer)

    def test_validate_algorithm_content_with_code(self, validator):
        """Test validation of algorithm content with code."""
        content = {
            "problem": "Implement binary search algorithm",
            "answer": "Binary search works by repeatedly dividing the search space in half",
            "explanation": "The algorithm has O(log n) time complexity due to the divide and conquer approach",
            "code": """
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
""",
        }

        result = validator.validate_algorithm_content(
            content,
            expected_complexity="O(log n)",
            expected_pattern="divide_and_conquer",
        )

        assert "validation_score" in result
        assert "algorithm_analysis" in result
        assert "complexity_validation" in result
        assert "pattern_validation" in result
        assert result["is_valid"] is True
        assert result["validation_score"] > 0.7

    def test_validate_algorithm_content_without_code(self, validator):
        """Test validation of algorithm content without code."""
        content = {
            "problem": "Explain the time complexity of bubble sort",
            "answer": "Bubble sort has O(n^2) time complexity",
            "explanation": "The nested loops cause quadratic time complexity in the worst case",
        }

        result = validator.validate_algorithm_content(content)

        assert "validation_score" in result
        assert "concept_analysis" in result
        assert result["concept_analysis"]["complexity_mentioned"] is True

    def test_validate_complexity_matching(self, validator):
        """Test complexity validation when expected matches detected."""
        analysis = {
            "complexity_analysis": ComplexityAnalysis(
                time_complexity="O(n)",
                space_complexity="O(1)",
                confidence=0.9,
                reasoning=["Linear loop detected"],
                optimization_suggestions=[],
            )
        }

        validation = validator._validate_complexity(
            analysis, "O(n)", "linear algorithm"
        )

        assert validation["matches"] is True
        assert validation["expected"] == "O(n)"
        assert validation["detected"] == "O(n)"
        assert validation["confidence"] == 0.9

    def test_validate_complexity_not_matching(self, validator):
        """Test complexity validation when expected doesn't match detected."""
        analysis = {
            "complexity_analysis": ComplexityAnalysis(
                time_complexity="O(n^2)",
                space_complexity="O(1)",
                confidence=0.8,
                reasoning=["Nested loops detected"],
                optimization_suggestions=[],
            )
        }

        validation = validator._validate_complexity(
            analysis, "O(n)", "quadratic algorithm"
        )

        assert validation["matches"] is False
        assert validation["expected"] == "O(n)"
        assert validation["detected"] == "O(n^2)"

    def test_validate_pattern_matching(self, validator):
        """Test pattern validation when expected matches detected."""
        analysis = {
            "algorithm_patterns": [
                AlgorithmPattern(
                    pattern_name="divide_and_conquer",
                    confidence=0.8,
                    characteristics=["recursive", "merge"],
                    typical_complexity="O(n log n)",
                )
            ]
        }

        validation = validator._validate_pattern(
            analysis, "divide_and_conquer", "divide and conquer approach"
        )

        assert validation["matches"] is True
        assert validation["expected"] == "divide_and_conquer"
        assert "divide_and_conquer" in validation["detected_patterns"]

    def test_analyze_algorithm_concepts(self, validator):
        """Test analysis of algorithm concepts in text."""
        text_content = """
        This algorithm has O(n log n) time complexity and is very efficient.
        We can optimize it further by using better data structures.
        The correctness proof relies on the loop invariant.
        There's a trade-off between time and space complexity.
        """

        concepts = validator._analyze_algorithm_concepts(text_content)

        assert concepts["complexity_mentioned"] is True
        assert concepts["optimization_discussed"] is True
        assert concepts["correctness_addressed"] is True
        assert concepts["trade_offs_mentioned"] is True
        assert concepts["concept_score"] == 1.0

    def test_calculate_validation_score(self, validator):
        """Test calculation of overall validation score."""
        algorithm_analysis = {
            "complexity_analysis": ComplexityAnalysis(
                time_complexity="O(n)",
                space_complexity="O(1)",
                confidence=0.8,
                reasoning=[],
                optimization_suggestions=[],
            )
        }

        complexity_validation = {"matches": True, "confidence": 0.9}

        pattern_validation = {"matches": False, "confidence": 0.5}

        concept_analysis = {"concept_score": 0.8}

        score = validator._calculate_validation_score(
            algorithm_analysis,
            complexity_validation,
            pattern_validation,
            concept_analysis,
        )

        assert 0.0 <= score <= 1.0
        assert score > 0.5  # Should be reasonably high given good inputs

    def test_generate_feedback_positive(self, validator):
        """Test feedback generation for good algorithm validation."""
        algorithm_analysis = {
            "complexity_analysis": ComplexityAnalysis(
                time_complexity="O(n)",
                space_complexity="O(1)",
                confidence=0.9,
                reasoning=[],
                optimization_suggestions=[],
            )
        }

        complexity_validation = {"matches": True}
        pattern_validation = {"matches": True}

        feedback = validator._generate_feedback(
            algorithm_analysis, complexity_validation, pattern_validation
        )

        assert len(feedback) > 0
        assert "looks good" in feedback[0].lower()

    def test_generate_feedback_with_issues(self, validator):
        """Test feedback generation when there are validation issues."""
        algorithm_analysis = {
            "complexity_analysis": ComplexityAnalysis(
                time_complexity="O(n^2)",
                space_complexity="O(1)",
                confidence=0.8,
                reasoning=[],
                optimization_suggestions=["Consider using hash table for O(1) lookup"],
            )
        }

        complexity_validation = {
            "matches": False,
            "expected": "O(n)",
            "detected": "O(n^2)",
        }

        pattern_validation = {
            "matches": False,
            "expected": "divide_and_conquer",
            "detected_patterns": ["brute_force"],
        }

        feedback = validator._generate_feedback(
            algorithm_analysis, complexity_validation, pattern_validation
        )

        assert len(feedback) > 0
        assert any("expected" in fb.lower() for fb in feedback)

    def test_validate_algorithm_content_error_handling(self, validator):
        """Test error handling in algorithm content validation."""
        # Test with invalid content
        invalid_content = None

        result = validator.validate_algorithm_content(invalid_content)

        assert result["validation_score"] == 0.0
        assert result["is_valid"] is False
        assert "error" in result
        assert len(result["feedback"]) > 0


class TestAlgorithmAnalysisIntegration:
    """Integration tests for algorithm analysis."""

    def test_comprehensive_algorithm_analysis(self):
        """Test comprehensive analysis of a complex algorithm."""
        analyzer = AlgorithmAnalyzer()

        quicksort_code = """
def quicksort(arr, low=0, high=None):
    if high is None:
        high = len(arr) - 1
    
    if low < high:
        # Partition the array
        pivot_index = partition(arr, low, high)
        
        # Recursively sort elements before and after partition
        quicksort(arr, low, pivot_index - 1)
        quicksort(arr, pivot_index + 1, high)
    
    return arr

def partition(arr, low, high):
    pivot = arr[high]
    i = low - 1
    
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1
"""

        result = analyzer.analyze_algorithm(quicksort_code, "python")

        # Should detect divide and conquer pattern
        assert "algorithm_patterns" in result
        patterns = [p.pattern_name for p in result["algorithm_patterns"]]
        assert "divide_and_conquer" in patterns

        # Should identify as quicksort
        assert "known_algorithms" in result
        algorithms = [algo["name"] for algo in result["known_algorithms"]]
        assert "quick_sort" in algorithms

        # Should have reasonable complexity
        assert "complexity_analysis" in result
        complexity = result["complexity_analysis"]
        assert complexity.time_complexity in [
            "O(n log n)",
            "O(n^2)",
        ]  # Average or worst case

    def test_algorithm_validator_end_to_end(self):
        """Test end-to-end algorithm validation workflow."""
        validator = AlgorithmValidator()

        content = {
            "problem": "Implement merge sort algorithm with O(n log n) complexity",
            "answer": "Merge sort uses divide and conquer approach",
            "explanation": "The algorithm divides the array recursively and merges sorted subarrays. Time complexity is O(n log n) due to log n levels and n work per level.",
            "code": """
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    return result
""",
        }

        result = validator.validate_algorithm_content(
            content,
            expected_complexity="O(n log n)",
            expected_pattern="divide_and_conquer",
        )

        # Should be valid with good score
        assert result["is_valid"] is True
        assert result["validation_score"] > 0.7

        # Should match expected complexity and pattern
        assert result["complexity_validation"]["matches"] is True
        assert result["pattern_validation"]["matches"] is True

        # Should have good concept analysis
        assert result["concept_analysis"]["complexity_mentioned"] is True
        assert result["concept_analysis"]["concept_score"] >= 0.5
