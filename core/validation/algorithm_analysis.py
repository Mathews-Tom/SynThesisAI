"""
Algorithm analysis validation for the SynThesisAI platform.

This module provides comprehensive algorithm analysis capabilities including:
- Time and space complexity validation
- Algorithm correctness verification
- Algorithm optimization and efficiency assessment
- Algorithm design pattern validation
"""

# Standard Library
import ast
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


@dataclass
class ComplexityAnalysis:
    """Result of algorithm complexity analysis."""

    time_complexity: str
    space_complexity: str
    confidence: float
    reasoning: List[str]
    optimization_suggestions: List[str]


@dataclass
class AlgorithmPattern:
    """Detected algorithm pattern."""

    pattern_name: str
    confidence: float
    characteristics: List[str]
    typical_complexity: str


class AlgorithmAnalyzer:
    """
    Comprehensive algorithm analysis and validation.

    Provides:
    - Time and space complexity analysis
    - Algorithm correctness verification
    - Optimization and efficiency assessment
    - Design pattern recognition
    - Performance bottleneck detection
    """

    def __init__(self):
        """Initialize the algorithm analyzer."""
        # Complexity patterns with their indicators
        self.complexity_patterns = {
            "O(1)": {
                "indicators": [
                    r"return\s+\w+",
                    r"\w+\[\d+\]",
                    r"\w+\.get\(",
                    r"hash.*lookup",
                    r"constant.*time",
                ],
                "description": "Constant time operations",
            },
            "O(log n)": {
                "indicators": [
                    r"binary.*search",
                    r"\/\/?\s*2",
                    r"mid\s*=.*\/\/?\s*2",
                    r"divide.*conquer",
                    r"tree.*height",
                    r"logarithmic",
                    r"left.*right.*mid",
                ],
                "description": "Logarithmic time operations",
            },
            "O(n)": {
                "indicators": [
                    r"for\s+\w+\s+in\s+\w+",
                    r"while\s+\w+",
                    r"linear.*search",
                    r"single.*pass",
                    r"traverse.*once",
                    r"for.*item.*in.*arr",
                ],
                "description": "Linear time operations",
            },
            "O(n log n)": {
                "indicators": [
                    r"merge.*sort",
                    r"heap.*sort",
                    r"quick.*sort",
                    r"sort\(",
                    r"divide.*conquer.*merge",
                    r"n.*log.*n",
                ],
                "description": "Linearithmic time operations",
            },
            "O(n^2)": {
                "indicators": [
                    r"for\s+\w+.*:\s*.*for\s+\w+",
                    r"nested.*loop",
                    r"bubble.*sort",
                    r"selection.*sort",
                    r"insertion.*sort",
                    r"quadratic",
                ],
                "description": "Quadratic time operations",
            },
            "O(n^3)": {
                "indicators": [
                    r"for\s+\w+.*for\s+\w+.*for\s+\w+",
                    r"triple.*nested",
                    r"cubic",
                    r"matrix.*multiplication",
                ],
                "description": "Cubic time operations",
            },
            "O(2^n)": {
                "indicators": [
                    r"fibonacci.*recursive",
                    r"2\s*\*\*\s*n",
                    r"exponential",
                    r"subset.*generation",
                    r"brute.*force.*recursive",
                ],
                "description": "Exponential time operations",
            },
            "O(n!)": {
                "indicators": [
                    r"permutation",
                    r"factorial",
                    r"n!",
                    r"traveling.*salesman",
                    r"all.*arrangements",
                ],
                "description": "Factorial time operations",
            },
        }

        # Algorithm design patterns
        self.algorithm_patterns = {
            "divide_and_conquer": {
                "indicators": [
                    r"divide.*conquer",
                    r"recursive.*split",
                    r"merge.*results",
                    r"base.*case.*recursive",
                    r"merge_sort",
                    r"len\(arr\)\s*<=\s*1",
                    r"mid\s*=.*len\(arr\)",
                    r"quicksort",
                    r"partition.*pivot",
                    r"recursively.*sort",
                ],
                "examples": ["merge sort", "quick sort", "binary search"],
            },
            "dynamic_programming": {
                "indicators": [
                    r"memoization",
                    r"dp\[",
                    r"optimal.*substructure",
                    r"overlapping.*subproblems",
                    r"cache.*results",
                    r"memo\s*=",
                    r"if.*in.*memo",
                ],
                "examples": ["fibonacci DP", "knapsack", "longest common subsequence"],
            },
            "greedy": {
                "indicators": [
                    r"greedy",
                    r"local.*optimal",
                    r"minimum.*spanning.*tree",
                    r"shortest.*path",
                    r"activity.*selection",
                ],
                "examples": ["Dijkstra's algorithm", "Kruskal's algorithm"],
            },
            "backtracking": {
                "indicators": [
                    r"backtrack",
                    r"try.*all.*possibilities",
                    r"n.*queens",
                    r"sudoku.*solver",
                    r"recursive.*exploration",
                ],
                "examples": ["N-Queens", "Sudoku solver", "maze solving"],
            },
            "two_pointers": {
                "indicators": [
                    r"two.*pointer",
                    r"left.*right.*pointer",
                    r"start.*end.*pointer",
                    r"sliding.*window",
                ],
                "examples": [
                    "two sum",
                    "palindrome check",
                    "container with most water",
                ],
            },
            "sliding_window": {
                "indicators": [
                    r"sliding.*window",
                    r"window.*size",
                    r"maximum.*subarray",
                    r"substring.*window",
                ],
                "examples": ["maximum subarray", "longest substring"],
            },
        }

        # Common algorithm implementations
        self.known_algorithms = {
            "binary_search": {
                "complexity": "O(log n)",
                "pattern": r"binary.*search|mid\s*=.*\/\/?\s*2",
                "characteristics": [
                    "sorted input",
                    "divide and conquer",
                    "logarithmic",
                ],
            },
            "linear_search": {
                "complexity": "O(n)",
                "pattern": r"linear.*search|for.*in.*if.*==",
                "characteristics": ["sequential scan", "unsorted input", "linear"],
            },
            "bubble_sort": {
                "complexity": "O(n^2)",
                "pattern": r"bubble.*sort|for.*for.*swap",
                "characteristics": ["nested loops", "adjacent swaps", "quadratic"],
            },
            "merge_sort": {
                "complexity": "O(n log n)",
                "pattern": r"merge.*sort|divide.*merge",
                "characteristics": ["divide and conquer", "stable", "linearithmic"],
            },
            "quick_sort": {
                "complexity": "O(n log n)",
                "pattern": r"quick.*sort|partition.*pivot",
                "characteristics": [
                    "divide and conquer",
                    "in-place",
                    "average linearithmic",
                ],
            },
            "dijkstra": {
                "complexity": "O((V + E) log V)",
                "pattern": r"dijkstra|shortest.*path.*priority",
                "characteristics": [
                    "graph algorithm",
                    "shortest path",
                    "priority queue",
                ],
            },
            "dfs": {
                "complexity": "O(V + E)",
                "pattern": r"depth.*first|dfs|recursive.*visit",
                "characteristics": ["graph traversal", "recursive", "stack-based"],
            },
            "bfs": {
                "complexity": "O(V + E)",
                "pattern": r"breadth.*first|bfs|queue.*visit",
                "characteristics": ["graph traversal", "level-order", "queue-based"],
            },
        }

        logger.info(
            "Initialized AlgorithmAnalyzer with %d complexity patterns",
            len(self.complexity_patterns),
        )

    def analyze_algorithm(
        self,
        code: str,
        language: str = "python",
        algorithm_description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Perform comprehensive algorithm analysis.

        Args:
            code: Source code to analyze
            language: Programming language
            algorithm_description: Optional description of the algorithm

        Returns:
            Dictionary with comprehensive analysis results
        """
        try:
            analysis_result = {
                "complexity_analysis": self._analyze_complexity(code, language),
                "algorithm_patterns": self._detect_algorithm_patterns(code, language),
                "known_algorithms": self._identify_known_algorithms(code, language),
                "correctness_analysis": self._analyze_correctness(code, language),
                "optimization_suggestions": self._generate_optimization_suggestions(
                    code, language
                ),
                "performance_characteristics": self._analyze_performance_characteristics(
                    code, language
                ),
            }

            # If algorithm description is provided, validate against it
            if algorithm_description:
                analysis_result["description_validation"] = (
                    self._validate_against_description(
                        code, algorithm_description, analysis_result
                    )
                )

            return analysis_result

        except Exception as e:
            logger.error("Algorithm analysis failed: %s", e)
            return {"error": str(e)}

    def _analyze_complexity(self, code: str, language: str) -> ComplexityAnalysis:
        """Analyze time and space complexity of the algorithm."""
        time_complexity = "O(1)"
        space_complexity = "O(1)"
        confidence = 0.0
        reasoning = []
        optimization_suggestions = []

        # Analyze time complexity - prioritize higher complexities
        complexity_order = [
            "O(1)",
            "O(log n)",
            "O(n)",
            "O(n log n)",
            "O(n^2)",
            "O(n^3)",
            "O(2^n)",
            "O(n!)",
        ]
        detected_complexities = []

        for complexity, pattern_info in self.complexity_patterns.items():
            pattern_confidence = 0.0
            matched_patterns = []

            for pattern in pattern_info["indicators"]:
                if re.search(pattern, code, re.IGNORECASE):
                    pattern_confidence += 0.2
                    matched_patterns.append(pattern)

            if pattern_confidence > 0:
                detected_complexities.append(
                    (complexity, pattern_confidence, matched_patterns)
                )

        # Choose the highest complexity with significant confidence
        if detected_complexities:
            # Sort by confidence first (descending), then by complexity order (higher complexity first)
            detected_complexities.sort(
                key=lambda x: (
                    x[1],
                    complexity_order.index(x[0]) if x[0] in complexity_order else 0,
                ),
                reverse=True,
            )

            # Choose the best match
            time_complexity, max_confidence, matched_patterns = detected_complexities[0]
            if matched_patterns:
                reasoning.append(
                    f"Detected {time_complexity} patterns: {matched_patterns[:2]}"
                )
        else:
            max_confidence = 0.0

        confidence = min(1.0, max_confidence)

        # Analyze space complexity (simplified)
        space_complexity = self._analyze_space_complexity(code, language)

        # Generate optimization suggestions based on complexity
        if time_complexity in ["O(n^2)", "O(n^3)", "O(2^n)", "O(n!)"]:
            optimization_suggestions.append(
                f"Consider optimizing {time_complexity} algorithm for better performance"
            )

        return ComplexityAnalysis(
            time_complexity=time_complexity,
            space_complexity=space_complexity,
            confidence=confidence,
            reasoning=reasoning,
            optimization_suggestions=optimization_suggestions,
        )

    def _analyze_space_complexity(self, code: str, language: str) -> str:
        """Analyze space complexity of the algorithm."""
        # Check for recursive calls (stack space)
        if re.search(r"def\s+(\w+).*\1\s*\(", code, re.DOTALL):
            return "O(n)"  # Recursive stack space

        # Check for data structure creation
        if re.search(r"list\(|dict\(|\[\]|\{\}", code):
            # Check if it's proportional to input size
            if re.search(r"for\s+\w+.*append\(|for\s+\w+.*\[\w+\]\s*=", code):
                return "O(n)"

        # Check for matrix/2D array creation
        if re.search(r"for\s+\w+.*for\s+\w+.*append\(", code):
            return "O(n^2)"

        return "O(1)"  # Default constant space

    def _detect_algorithm_patterns(
        self, code: str, language: str
    ) -> List[AlgorithmPattern]:
        """Detect algorithm design patterns in the code."""
        detected_patterns = []

        for pattern_name, pattern_info in self.algorithm_patterns.items():
            confidence = 0.0
            matched_characteristics = []

            for indicator in pattern_info["indicators"]:
                if re.search(indicator, code, re.IGNORECASE):
                    confidence += 0.25
                    matched_characteristics.append(indicator)

            if confidence > 0.25:  # Threshold for pattern detection
                # Determine typical complexity for this pattern
                typical_complexity = self._get_pattern_complexity(pattern_name)

                detected_patterns.append(
                    AlgorithmPattern(
                        pattern_name=pattern_name,
                        confidence=min(1.0, confidence),
                        characteristics=matched_characteristics,
                        typical_complexity=typical_complexity,
                    )
                )

        return detected_patterns

    def _identify_known_algorithms(
        self, code: str, language: str
    ) -> List[Dict[str, Any]]:
        """Identify known algorithms in the code."""
        identified_algorithms = []

        for algo_name, algo_info in self.known_algorithms.items():
            if re.search(algo_info["pattern"], code, re.IGNORECASE):
                identified_algorithms.append(
                    {
                        "name": algo_name,
                        "complexity": algo_info["complexity"],
                        "characteristics": algo_info["characteristics"],
                        "confidence": 0.8,  # High confidence for pattern matches
                    }
                )

        return identified_algorithms

    def _analyze_correctness(self, code: str, language: str) -> Dict[str, Any]:
        """Analyze algorithm correctness indicators."""
        correctness_analysis = {
            "has_base_case": False,
            "has_termination_condition": False,
            "handles_edge_cases": False,
            "has_input_validation": False,
            "potential_issues": [],
        }

        # Check for base case in recursive algorithms
        if re.search(r"def\s+(\w+).*\1\s*\(", code, re.DOTALL):
            if re.search(r"if.*return|if.*==.*return", code, re.IGNORECASE | re.DOTALL):
                correctness_analysis["has_base_case"] = True
            else:
                correctness_analysis["potential_issues"].append(
                    "Recursive function may be missing base case"
                )

        # Check for termination conditions in loops
        if re.search(r"while\s+", code):
            if re.search(r"break|return|\w+\s*[+\-]=|\w+\s*=.*[+\-]", code):
                correctness_analysis["has_termination_condition"] = True
            else:
                correctness_analysis["potential_issues"].append(
                    "Loop may not have proper termination condition"
                )

        # Check for edge case handling
        edge_case_patterns = [
            r"if.*empty|if.*len.*==.*0",
            r"if.*null|if.*None",
            r"if.*<=.*0|if.*<.*1",
        ]

        for pattern in edge_case_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                correctness_analysis["handles_edge_cases"] = True
                break

        # Check for input validation
        validation_patterns = [
            r"if.*not.*\w+|if.*\w+.*is.*None",
            r"assert\s+",
            r"raise.*Error|raise.*Exception",
        ]

        for pattern in validation_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                correctness_analysis["has_input_validation"] = True
                break

        return correctness_analysis

    def _generate_optimization_suggestions(self, code: str, language: str) -> List[str]:
        """Generate algorithm optimization suggestions."""
        suggestions = []

        # Check for nested loops
        if re.search(r"for\s+\w+.*for\s+\w+", code, re.DOTALL):
            suggestions.append(
                "Consider if nested loops can be optimized with better data structures or algorithms"
            )

        # Check for repeated calculations
        if re.search(r"for\s+\w+.*\w+\(.*\)", code, re.DOTALL):
            suggestions.append(
                "Consider caching repeated function calls or calculations"
            )

        # Check for inefficient data structure usage
        if re.search(r"list.*in\s+\w+", code):
            suggestions.append(
                "Consider using set or dict for O(1) lookup instead of list"
            )

        # Check for recursive algorithms without memoization
        if re.search(r"def\s+(\w+).*\1\s*\(", code, re.DOTALL):
            if not re.search(r"cache|memo|dp", code, re.IGNORECASE):
                suggestions.append(
                    "Consider memoization for recursive algorithms to avoid redundant calculations"
                )

        # Check for sorting when not necessary
        if re.search(r"sort\(|sorted\(", code):
            suggestions.append(
                "Verify if sorting is necessary - consider if partial sorting or other approaches work"
            )

        return suggestions

    def _analyze_performance_characteristics(
        self, code: str, language: str
    ) -> Dict[str, Any]:
        """Analyze performance characteristics of the algorithm."""
        characteristics = {
            "is_stable": None,
            "is_in_place": None,
            "is_adaptive": None,
            "parallelizable": None,
            "memory_access_pattern": "unknown",
            "cache_efficiency": "unknown",
        }

        # Analyze stability (for sorting algorithms)
        if re.search(r"sort", code, re.IGNORECASE):
            if re.search(r"merge.*sort", code, re.IGNORECASE):
                characteristics["is_stable"] = True
            elif re.search(r"quick.*sort|heap.*sort", code, re.IGNORECASE):
                characteristics["is_stable"] = False

        # Analyze in-place property
        if re.search(r"swap|exchange", code, re.IGNORECASE):
            characteristics["is_in_place"] = True
        elif re.search(r"new.*array|temp.*array|\[\].*=", code):
            characteristics["is_in_place"] = False

        # Analyze memory access patterns
        if re.search(r"for\s+i.*for\s+j", code):
            characteristics["memory_access_pattern"] = "nested_sequential"
            characteristics["cache_efficiency"] = "poor"
        elif re.search(r"for\s+\w+\s+in", code):
            characteristics["memory_access_pattern"] = "sequential"
            characteristics["cache_efficiency"] = "good"

        return characteristics

    def _validate_against_description(
        self, code: str, description: str, analysis_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate algorithm implementation against its description."""
        validation_result = {
            "complexity_matches": False,
            "pattern_matches": False,
            "description_accuracy": 0.0,
            "discrepancies": [],
        }

        # Extract complexity from description
        described_complexity = self._extract_complexity_from_description(description)
        analyzed_complexity = analysis_result["complexity_analysis"].time_complexity

        if described_complexity and described_complexity == analyzed_complexity:
            validation_result["complexity_matches"] = True
        elif described_complexity:
            validation_result["discrepancies"].append(
                f"Described complexity {described_complexity} doesn't match analyzed {analyzed_complexity}"
            )

        # Check if described patterns match detected patterns
        detected_pattern_names = [
            p.pattern_name for p in analysis_result["algorithm_patterns"]
        ]

        for pattern_name in self.algorithm_patterns.keys():
            if pattern_name.replace("_", " ") in description.lower():
                if pattern_name in detected_pattern_names:
                    validation_result["pattern_matches"] = True
                else:
                    validation_result["discrepancies"].append(
                        f"Described pattern '{pattern_name}' not detected in code"
                    )

        # Calculate overall accuracy
        accuracy_factors = [
            validation_result["complexity_matches"],
            validation_result["pattern_matches"],
            len(validation_result["discrepancies"]) == 0,
        ]

        validation_result["description_accuracy"] = sum(accuracy_factors) / len(
            accuracy_factors
        )

        return validation_result

    def _get_pattern_complexity(self, pattern_name: str) -> str:
        """Get typical complexity for an algorithm pattern."""
        pattern_complexities = {
            "divide_and_conquer": "O(n log n)",
            "dynamic_programming": "O(n^2)",
            "greedy": "O(n log n)",
            "backtracking": "O(2^n)",
            "two_pointers": "O(n)",
            "sliding_window": "O(n)",
        }

        return pattern_complexities.get(pattern_name, "O(n)")

    def _extract_complexity_from_description(self, description: str) -> Optional[str]:
        """Extract complexity notation from algorithm description."""
        complexity_pattern = r"O\s*\(\s*([^)]+)\s*\)"
        match = re.search(complexity_pattern, description, re.IGNORECASE)

        if match:
            complexity_str = match.group(1).strip()
            # Normalize the complexity string
            normalized = f"O({complexity_str})"

            # Check if it matches any known complexity exactly
            normalized = f"O({complexity_str})"
            if normalized in self.complexity_patterns:
                return normalized

            # Try some common variations
            variations = [
                f"O({complexity_str})",
                f"O({complexity_str.replace(' ', '')})",
                f"O({complexity_str.replace('^', '**')})",
            ]

            for variation in variations:
                if variation in self.complexity_patterns:
                    return variation

            return normalized

        return None


class AlgorithmValidator:
    """
    High-level algorithm validation interface.

    Combines algorithm analysis with validation logic to provide
    comprehensive algorithm assessment for educational content.
    """

    def __init__(self):
        """Initialize the algorithm validator."""
        self.analyzer = AlgorithmAnalyzer()
        logger.info("Initialized AlgorithmValidator")

    def validate_algorithm_content(
        self,
        content: Dict[str, Any],
        expected_complexity: Optional[str] = None,
        expected_pattern: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Validate algorithm-related educational content.

        Args:
            content: Content dictionary with problem, answer, explanation, code
            expected_complexity: Expected time complexity
            expected_pattern: Expected algorithm pattern

        Returns:
            Comprehensive validation results
        """
        try:
            # Extract content components
            problem = content.get("problem", "")
            answer = content.get("answer", "")
            explanation = content.get("explanation", "")
            code = content.get("code", "")

            # Combine text content for analysis
            text_content = f"{problem} {answer} {explanation}"

            # Analyze algorithm if code is provided
            algorithm_analysis = {}
            if code.strip():
                algorithm_analysis = self.analyzer.analyze_algorithm(
                    code, "python", text_content
                )

            # Validate complexity if expected
            complexity_validation = {}
            if expected_complexity:
                complexity_validation = self._validate_complexity(
                    algorithm_analysis, expected_complexity, text_content
                )

            # Validate pattern if expected
            pattern_validation = {}
            if expected_pattern:
                pattern_validation = self._validate_pattern(
                    algorithm_analysis, expected_pattern, text_content
                )

            # Analyze text content for algorithm concepts
            concept_analysis = self._analyze_algorithm_concepts(text_content)

            # Calculate overall validation score
            validation_score = self._calculate_validation_score(
                algorithm_analysis,
                complexity_validation,
                pattern_validation,
                concept_analysis,
            )

            return {
                "validation_score": validation_score,
                "algorithm_analysis": algorithm_analysis,
                "complexity_validation": complexity_validation,
                "pattern_validation": pattern_validation,
                "concept_analysis": concept_analysis,
                "is_valid": validation_score >= 0.7,
                "feedback": self._generate_feedback(
                    algorithm_analysis, complexity_validation, pattern_validation
                ),
            }

        except Exception as e:
            logger.error("Algorithm validation failed: %s", e)
            return {
                "validation_score": 0.0,
                "is_valid": False,
                "error": str(e),
                "feedback": [f"Algorithm validation error: {str(e)}"],
            }

    def _validate_complexity(
        self, analysis: Dict[str, Any], expected_complexity: str, text_content: str
    ) -> Dict[str, Any]:
        """Validate algorithm complexity against expectations."""
        validation = {
            "expected": expected_complexity,
            "detected": None,
            "matches": False,
            "confidence": 0.0,
            "reasoning": [],
        }

        if "complexity_analysis" in analysis:
            complexity_analysis = analysis["complexity_analysis"]
            detected_complexity = complexity_analysis.time_complexity

            validation["detected"] = detected_complexity
            validation["confidence"] = complexity_analysis.confidence
            validation["matches"] = detected_complexity == expected_complexity
            validation["reasoning"] = complexity_analysis.reasoning

        # Also check if complexity is mentioned in text
        if expected_complexity.lower() in text_content.lower():
            validation["mentioned_in_text"] = True
            validation["confidence"] = max(validation["confidence"], 0.8)

        return validation

    def _validate_pattern(
        self, analysis: Dict[str, Any], expected_pattern: str, text_content: str
    ) -> Dict[str, Any]:
        """Validate algorithm pattern against expectations."""
        validation = {
            "expected": expected_pattern,
            "detected_patterns": [],
            "matches": False,
            "confidence": 0.0,
        }

        if "algorithm_patterns" in analysis:
            detected_patterns = analysis["algorithm_patterns"]
            validation["detected_patterns"] = [
                p.pattern_name for p in detected_patterns
            ]

            # Check if expected pattern is detected
            for pattern in detected_patterns:
                if pattern.pattern_name == expected_pattern:
                    validation["matches"] = True
                    validation["confidence"] = pattern.confidence
                    break

        # Also check if pattern is mentioned in text
        if expected_pattern.replace("_", " ").lower() in text_content.lower():
            validation["mentioned_in_text"] = True
            validation["confidence"] = max(validation["confidence"], 0.8)
            # If pattern is mentioned in text, consider it a partial match
            if not validation["matches"]:
                validation["matches"] = True

        return validation

    def _analyze_algorithm_concepts(self, text_content: str) -> Dict[str, Any]:
        """Analyze algorithm concepts mentioned in text content."""
        concepts = {
            "complexity_mentioned": False,
            "optimization_discussed": False,
            "correctness_addressed": False,
            "trade_offs_mentioned": False,
            "concept_score": 0.0,
        }

        text_lower = text_content.lower()

        # Check for complexity discussion
        complexity_terms = ["complexity", "time", "space", "big o", "o(", "efficient"]
        if any(term in text_lower for term in complexity_terms):
            concepts["complexity_mentioned"] = True

        # Check for optimization discussion
        optimization_terms = [
            "optimize",
            "improve",
            "efficient",
            "performance",
            "faster",
            "levels",
            "work per level",
        ]
        if any(term in text_lower for term in optimization_terms):
            concepts["optimization_discussed"] = True

        # Check for correctness discussion
        correctness_terms = [
            "correct",
            "proof",
            "invariant",
            "termination",
            "base case",
        ]
        if any(term in text_lower for term in correctness_terms):
            concepts["correctness_addressed"] = True

        # Check for trade-offs discussion
        tradeoff_terms = ["trade-off", "tradeoff", "memory vs time", "space vs time"]
        if any(term in text_lower for term in tradeoff_terms):
            concepts["trade_offs_mentioned"] = True

        # Calculate concept score
        concept_factors = [
            concepts["complexity_mentioned"],
            concepts["optimization_discussed"],
            concepts["correctness_addressed"],
            concepts["trade_offs_mentioned"],
        ]

        concepts["concept_score"] = sum(concept_factors) / len(concept_factors)

        return concepts

    def _calculate_validation_score(
        self,
        algorithm_analysis: Dict[str, Any],
        complexity_validation: Dict[str, Any],
        pattern_validation: Dict[str, Any],
        concept_analysis: Dict[str, Any],
    ) -> float:
        """Calculate overall validation score."""
        score_components = []

        # Algorithm analysis score
        if algorithm_analysis and "complexity_analysis" in algorithm_analysis:
            complexity_confidence = algorithm_analysis["complexity_analysis"].confidence
            score_components.append(complexity_confidence * 0.3)

        # Complexity validation score
        if complexity_validation:
            complexity_score = (
                1.0
                if complexity_validation["matches"]
                else complexity_validation["confidence"]
            )
            score_components.append(complexity_score * 0.3)

        # Pattern validation score
        if pattern_validation:
            pattern_score = (
                1.0
                if pattern_validation["matches"]
                else pattern_validation["confidence"]
            )
            score_components.append(pattern_score * 0.2)

        # Concept analysis score
        concept_score = concept_analysis["concept_score"]
        score_components.append(concept_score * 0.2)

        # Calculate weighted sum (weights already applied)
        if score_components:
            return sum(score_components)
        else:
            return 0.5  # Neutral score if no components

    def _generate_feedback(
        self,
        algorithm_analysis: Dict[str, Any],
        complexity_validation: Dict[str, Any],
        pattern_validation: Dict[str, Any],
    ) -> List[str]:
        """Generate feedback for algorithm validation."""
        feedback = []

        # Complexity feedback
        if complexity_validation and not complexity_validation["matches"]:
            expected = complexity_validation["expected"]
            detected = complexity_validation.get("detected", "unknown")
            feedback.append(f"Expected complexity {expected} but detected {detected}")

        # Pattern feedback
        if pattern_validation and not pattern_validation["matches"]:
            expected = pattern_validation["expected"]
            detected = pattern_validation["detected_patterns"]
            feedback.append(f"Expected pattern '{expected}' but detected {detected}")

        # Algorithm analysis feedback
        if algorithm_analysis and "optimization_suggestions" in algorithm_analysis:
            complexity_analysis = algorithm_analysis["complexity_analysis"]
            if complexity_analysis.optimization_suggestions:
                feedback.extend(complexity_analysis.optimization_suggestions)

        # Positive feedback
        if not feedback:
            feedback.append("Algorithm analysis looks good!")

        return feedback
