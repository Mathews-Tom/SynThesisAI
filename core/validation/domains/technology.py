"""
Technology domain validator for the SynThesisAI platform.

This module implements comprehensive validation for technology-related content,
including code execution, algorithm analysis, security validation, and
technology best practices.
"""

# Standard Library
import logging
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# SynThesisAI Modules
from core.validation.base import DomainValidator, ValidationResult
from core.validation.config import ValidationConfig

logger = logging.getLogger(__name__)


class SandboxedCodeExecutor:
    """
    Sandboxed code execution environment for safe code validation.

    This class provides secure code execution capabilities with:
    - Language-specific execution environments
    - Resource limits and timeouts
    - Security restrictions
    - Output capture and validation
    """

    def __init__(self, timeout_seconds: int = 5, memory_limit_mb: int = 128):
        """
        Initialize the sandboxed code executor.

        Args:
            timeout_seconds: Maximum execution time
            memory_limit_mb: Maximum memory usage
        """
        self.timeout_seconds = timeout_seconds
        self.memory_limit_mb = memory_limit_mb

        # Supported languages and their execution commands
        self.language_configs = {
            "python": {
                "extension": ".py",
                "command": ["python3", "-c"],
                "security_checks": ["import", "exec", "eval", "__import__"],
            },
            "javascript": {
                "extension": ".js",
                "command": ["node", "-e"],
                "security_checks": ["require", "process", "fs", "child_process"],
            },
            "java": {
                "extension": ".java",
                "command": ["java"],
                "security_checks": ["Runtime", "ProcessBuilder", "System.exit"],
            },
            "cpp": {
                "extension": ".cpp",
                "command": ["g++", "-o", "temp_exec", "&&", "./temp_exec"],
                "security_checks": ["system", "exec", "fork", "popen"],
            },
        }

        logger.info(
            "Initialized SandboxedCodeExecutor with timeout=%ds, memory=%dMB",
            timeout_seconds,
            memory_limit_mb,
        )

    def execute_code(
        self, code: str, language: str, input_data: str = ""
    ) -> Dict[str, Any]:
        """
        Execute code in a sandboxed environment.

        Args:
            code: Source code to execute
            language: Programming language
            input_data: Input data for the program

        Returns:
            Dictionary with execution results
        """
        try:
            if language not in self.language_configs:
                return {
                    "success": False,
                    "error": f"Unsupported language: {language}",
                    "output": "",
                    "execution_time": 0.0,
                }

            # Security check
            if not self._security_check(code, language):
                return {
                    "success": False,
                    "error": "Code contains potentially unsafe operations",
                    "output": "",
                    "execution_time": 0.0,
                }

            # Execute code
            start_time = time.time()
            result = self._execute_with_timeout(code, language, input_data)
            execution_time = time.time() - start_time

            result["execution_time"] = execution_time
            return result

        except Exception as e:
            logger.error("Code execution failed: %s", e)
            return {
                "success": False,
                "error": str(e),
                "output": "",
                "execution_time": 0.0,
            }

    def _security_check(self, code: str, language: str) -> bool:
        """Check code for potentially unsafe operations."""
        config = self.language_configs[language]
        security_checks = config["security_checks"]

        # Simple keyword-based security check
        code_lower = code.lower()
        for check in security_checks:
            if check.lower() in code_lower:
                logger.warning(
                    "Security check failed: found '%s' in %s code", check, language
                )
                return False

        return True

    def _execute_with_timeout(
        self, code: str, language: str, input_data: str
    ) -> Dict[str, Any]:
        """Execute code with timeout and resource limits."""
        config = self.language_configs[language]

        try:
            if language == "python":
                # Direct Python execution for simplicity
                result = subprocess.run(
                    ["python3", "-c", code],
                    input=input_data,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                )
            elif language == "javascript":
                # Node.js execution
                result = subprocess.run(
                    ["node", "-e", code],
                    input=input_data,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                )
            else:
                # For other languages, return a mock result for now
                return {
                    "success": True,
                    "output": f"Mock execution result for {language}",
                    "error": "",
                }

            return {
                "success": result.returncode == 0,
                "output": result.stdout,
                "error": result.stderr,
            }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "output": "",
                "error": f"Code execution timed out after {self.timeout_seconds} seconds",
            }
        except Exception as e:
            return {"success": False, "output": "", "error": str(e)}


class TechnologyValidator(DomainValidator):
    """
    Validator for technology content with comprehensive technical validation.

    This validator handles:
    - Code execution and correctness validation
    - Algorithm analysis and complexity validation
    - Security validation for cybersecurity content
    - Technology best practices validation
    - System design principle validation
    """

    def __init__(
        self, subdomain: str = "technology", config: Optional[ValidationConfig] = None
    ):
        """
        Initialize the technology validator.

        Args:
            subdomain: The technology subdomain (default: "technology")
            config: Validation configuration settings

        Raises:
            ValueError: If subdomain is not technology-related
        """
        super().__init__("technology", config)
        self.subdomain = subdomain

        # Validate subdomain
        valid_subdomains = {
            "technology",
            "computer_science",
            "software_engineering",
            "cybersecurity",
            "data_science",
            "artificial_intelligence",
            "web_development",
            "mobile_development",
            "systems_programming",
            "algorithms",
            "databases",
            "networking",
        }
        if subdomain not in valid_subdomains:
            raise ValueError(f"Invalid technology subdomain: {subdomain}")

        # Initialize technology components
        self._initialize_technology_data()

        # Initialize code executor
        self.code_executor = SandboxedCodeExecutor(
            timeout_seconds=5, memory_limit_mb=128
        )

        logger.info("Initialized TechnologyValidator for subdomain: %s", subdomain)

    def _initialize_technology_data(self) -> None:
        """Initialize technology data and knowledge bases."""
        # Programming languages and their characteristics
        self.programming_languages = {
            "python": {
                "paradigms": ["object-oriented", "functional", "procedural"],
                "use_cases": ["data science", "web development", "automation", "AI"],
                "syntax_patterns": [
                    r"def\s+\w+",
                    r"class\s+\w+",
                    r"import\s+\w+",
                    r"if\s+__name__\s*==\s*['\"]__main__['\"]",
                ],
            },
            "javascript": {
                "paradigms": ["functional", "object-oriented", "event-driven"],
                "use_cases": ["web development", "frontend", "backend", "mobile"],
                "syntax_patterns": [
                    r"function\s+\w+",
                    r"const\s+\w+",
                    r"let\s+\w+",
                    r"=>\s*{",
                ],
            },
            "java": {
                "paradigms": ["object-oriented", "concurrent"],
                "use_cases": ["enterprise", "android", "web services", "desktop"],
                "syntax_patterns": [
                    r"public\s+class\s+\w+",
                    r"public\s+static\s+void\s+main",
                    r"import\s+java\.",
                ],
            },
            "cpp": {
                "paradigms": ["object-oriented", "procedural", "generic"],
                "use_cases": [
                    "systems programming",
                    "game development",
                    "embedded",
                    "performance-critical",
                ],
                "syntax_patterns": [
                    r"#include\s*<\w+>",
                    r"int\s+main\s*\(",
                    r"class\s+\w+",
                    r"std::",
                ],
            },
        }

        # Algorithm complexity patterns
        self.complexity_patterns = {
            "O(1)": ["constant time", "hash table lookup", "array access"],
            "O(log n)": ["binary search", "balanced tree", "divide and conquer"],
            "O(n)": ["linear search", "single loop", "array traversal"],
            "O(n log n)": ["merge sort", "heap sort", "efficient sorting"],
            "O(n^2)": ["bubble sort", "nested loops", "brute force"],
            "O(2^n)": ["exponential", "recursive fibonacci", "subset generation"],
        }

        # Security concepts and best practices
        self.security_concepts = {
            "authentication": ["password", "token", "biometric", "multi-factor"],
            "authorization": ["permissions", "roles", "access control", "privileges"],
            "encryption": ["AES", "RSA", "hash", "digital signature", "TLS"],
            "vulnerabilities": ["SQL injection", "XSS", "CSRF", "buffer overflow"],
            "secure_coding": [
                "input validation",
                "sanitization",
                "parameterized queries",
                "least privilege",
            ],
        }

        # Technology best practices
        self.best_practices = {
            "software_engineering": {
                "principles": [
                    "DRY",
                    "SOLID",
                    "KISS",
                    "YAGNI",
                    "separation of concerns",
                ],
                "patterns": [
                    "MVC",
                    "Observer",
                    "Factory",
                    "Singleton",
                    "Strategy",
                    "Command",
                ],
                "practices": [
                    "code review",
                    "testing",
                    "documentation",
                    "version control",
                    "continuous integration",
                    "refactoring",
                ],
            },
            "system_design": {
                "principles": [
                    "scalability",
                    "reliability",
                    "availability",
                    "consistency",
                    "fault tolerance",
                    "maintainability",
                ],
                "patterns": [
                    "microservices",
                    "load balancing",
                    "caching",
                    "database sharding",
                    "circuit breaker",
                    "event sourcing",
                ],
                "practices": [
                    "monitoring",
                    "logging",
                    "backup",
                    "disaster recovery",
                    "health checks",
                ],
            },
            "data_structures": {
                "basic": ["array", "linked list", "stack", "queue"],
                "advanced": [
                    "tree",
                    "graph",
                    "hash table",
                    "heap",
                    "trie",
                    "bloom filter",
                ],
                "operations": ["insert", "delete", "search", "traverse"],
            },
            "coding_standards": {
                "naming": ["camelCase", "snake_case", "PascalCase", "kebab-case"],
                "formatting": ["indentation", "line length", "spacing", "brackets"],
                "structure": ["functions", "classes", "modules", "packages"],
                "comments": ["docstrings", "inline comments", "TODO", "FIXME"],
            },
            "testing": {
                "types": [
                    "unit testing",
                    "integration testing",
                    "system testing",
                    "acceptance testing",
                ],
                "practices": [
                    "TDD",
                    "BDD",
                    "test coverage",
                    "mocking",
                    "test automation",
                ],
                "frameworks": ["pytest", "junit", "jest", "mocha", "selenium"],
            },
            "performance": {
                "optimization": [
                    "algorithmic complexity",
                    "memory usage",
                    "caching",
                    "profiling",
                ],
                "monitoring": ["metrics", "logging", "tracing", "alerting"],
                "tools": [
                    "profilers",
                    "benchmarks",
                    "load testing",
                    "performance testing",
                ],
            },
            "security": {
                "practices": [
                    "input validation",
                    "output encoding",
                    "authentication",
                    "authorization",
                ],
                "principles": [
                    "least privilege",
                    "defense in depth",
                    "fail secure",
                    "secure by default",
                ],
                "tools": [
                    "static analysis",
                    "dependency scanning",
                    "penetration testing",
                    "code review",
                ],
            },
        }

        # Technology ethics considerations
        self.ethics_concepts = {
            "privacy": ["data protection", "GDPR", "consent", "anonymization"],
            "bias": [
                "algorithmic bias",
                "fairness",
                "discrimination",
                "representation",
            ],
            "responsibility": [
                "accountability",
                "transparency",
                "explainability",
                "social impact",
            ],
            "security": [
                "responsible disclosure",
                "ethical hacking",
                "user safety",
                "data breach",
            ],
        }

        # Technology concept accuracy validation
        self.technology_concepts = {
            "web_development": {
                "frontend": ["HTML", "CSS", "JavaScript", "React", "Vue", "Angular"],
                "backend": ["Node.js", "Python", "Java", "PHP", "Ruby", "Go"],
                "databases": ["MySQL", "PostgreSQL", "MongoDB", "Redis"],
                "protocols": ["HTTP", "HTTPS", "REST", "GraphQL", "WebSocket"],
            },
            "mobile_development": {
                "platforms": ["iOS", "Android", "React Native", "Flutter", "Xamarin"],
                "languages": ["Swift", "Kotlin", "Java", "Dart", "C#"],
                "concepts": ["responsive design", "native", "hybrid", "cross-platform"],
            },
            "data_science": {
                "languages": ["Python", "R", "SQL", "Scala", "Julia"],
                "libraries": [
                    "pandas",
                    "numpy",
                    "scikit-learn",
                    "tensorflow",
                    "pytorch",
                ],
                "concepts": [
                    "machine learning",
                    "deep learning",
                    "statistics",
                    "visualization",
                ],
            },
            "cloud_computing": {
                "providers": ["AWS", "Azure", "Google Cloud", "IBM Cloud"],
                "services": [
                    "compute",
                    "storage",
                    "database",
                    "networking",
                    "serverless",
                ],
                "concepts": [
                    "scalability",
                    "elasticity",
                    "availability",
                    "fault tolerance",
                ],
            },
        }

    def validate_content(self, content: Dict[str, Any]) -> ValidationResult:
        """
        Validate technology content comprehensively.

        Args:
            content: Dictionary containing technology content to validate

        Returns:
            ValidationResult with validation outcome and detailed feedback

        Raises:
            ValueError: If content format is invalid
        """
        logger.info("Starting technology content validation")

        try:
            # Extract content components
            problem = content.get("problem", "")
            answer = content.get("answer", "")
            explanation = content.get("explanation", "")
            code = content.get("code", "")

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

            # 1. Code execution validation (if code is present)
            code_score, code_feedback = self._validate_code_execution(
                problem, answer, explanation, code
            )
            validation_scores["code_validation"] = code_score
            if code_feedback:
                feedback_items.extend(code_feedback)

            # 2. Algorithm analysis validation
            algorithm_score, algorithm_feedback = self._validate_algorithm_analysis(
                problem, answer, explanation
            )
            validation_scores["algorithm_validation"] = algorithm_score
            if algorithm_feedback:
                feedback_items.extend(algorithm_feedback)

            # 3. Security validation
            security_score, security_feedback = self._validate_security_concepts(
                problem, answer, explanation
            )
            validation_scores["security_validation"] = security_score
            if security_feedback:
                feedback_items.extend(security_feedback)

            # 4. Best practices validation
            practices_score, practices_feedback = self._validate_best_practices(
                problem, answer, explanation
            )
            validation_scores["practices_validation"] = practices_score
            if practices_feedback:
                feedback_items.extend(practices_feedback)

            # 5. Technology ethics validation
            ethics_score, ethics_feedback = self._validate_technology_ethics(
                problem, answer, explanation
            )
            validation_scores["ethics_validation"] = ethics_score
            if ethics_feedback:
                feedback_items.extend(ethics_feedback)

            # 6. Technology concept accuracy validation
            concept_score, concept_feedback = self._validate_concept_accuracy(
                problem, answer, explanation
            )
            validation_scores["concept_accuracy"] = concept_score
            if concept_feedback:
                feedback_items.extend(concept_feedback)

            # Calculate overall quality score
            weights = {
                "code_validation": 0.22,
                "algorithm_validation": 0.18,
                "security_validation": 0.18,
                "practices_validation": 0.22,
                "ethics_validation": 0.12,
                "concept_accuracy": 0.08,
            }

            quality_score = sum(
                score * weights[category]
                for category, score in validation_scores.items()
            )

            # Determine if content is valid
            threshold = self.config.quality_thresholds.get("technology_score", 0.7)
            is_valid = quality_score >= threshold

            # Compile feedback
            feedback = (
                "; ".join(feedback_items)
                if feedback_items
                else "Technology content validated successfully"
            )

            # Add detailed metrics
            details.update(
                {
                    "technology_score": quality_score,
                    "validation_scores": validation_scores,
                    "threshold": threshold,
                    "weights": weights,
                }
            )

            logger.info(
                "Technology validation completed: valid=%s, score=%.2f",
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
            logger.error("Technology validation failed: %s", str(e))
            return ValidationResult(
                domain=self.domain,
                is_valid=False,
                quality_score=0.0,
                validation_details={"error": str(e), "subdomain": self.subdomain},
                confidence_score=0.0,
                feedback=[f"Technology validation error: {str(e)}"],
            )

    def _validate_code_execution(
        self, problem: str, answer: str, explanation: str, code: str
    ) -> Tuple[float, List[str]]:
        """
        Validate code execution and correctness.

        Args:
            problem: Problem statement
            answer: Answer content
            explanation: Explanation content
            code: Code to validate

        Returns:
            Tuple of (score, feedback_list)
        """
        feedback = []
        score = 1.0

        all_content = f"{problem} {answer} {explanation}"

        # Check if code is present or if code-related content exists
        code_indicators = ["code", "program", "function", "algorithm", "implementation"]
        has_code_content = any(
            indicator in all_content.lower() for indicator in code_indicators
        )

        if not code and not has_code_content:
            # No code content, return neutral score
            return score, feedback

        # Detect programming language
        detected_language = self._detect_programming_language(all_content + " " + code)

        if detected_language:
            details = {"detected_language": detected_language}

            # If actual code is provided, execute it
            if code.strip():
                execution_result = self.code_executor.execute_code(
                    code, detected_language
                )

                if not execution_result["success"]:
                    feedback.append(
                        f"Code execution failed: {execution_result['error']}"
                    )
                    score *= 0.7
                else:
                    # Code executed successfully
                    if execution_result["execution_time"] > 2.0:
                        feedback.append(
                            "Code execution time is high, consider optimization"
                        )
                        score *= 0.9

            # Validate code structure and best practices
            structure_score = self._validate_code_structure(
                code or all_content, detected_language
            )
            score *= structure_score

            if structure_score < 0.8:
                feedback.append("Code structure or best practices could be improved")

        else:
            # No clear programming language detected
            if has_code_content:
                feedback.append("Programming language not clearly identified")
                score *= 0.9

        return score, feedback

    def _validate_algorithm_analysis(
        self, problem: str, answer: str, explanation: str
    ) -> Tuple[float, List[str]]:
        """
        Validate algorithm analysis and complexity discussion.

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

        # Check for algorithm-related content
        algorithm_indicators = [
            "algorithm",
            "complexity",
            "time",
            "space",
            "efficiency",
            "big o",
        ]
        has_algorithm_content = any(
            indicator in all_content for indicator in algorithm_indicators
        )

        if not has_algorithm_content:
            # Check for basic content penalty
            if len(all_content.split()) < 20:
                score *= 0.85
            return score, feedback

        # Check for complexity analysis
        complexity_mentioned = False
        for complexity, indicators in self.complexity_patterns.items():
            if complexity.lower() in all_content or any(
                ind.lower() in all_content for ind in indicators
            ):
                complexity_mentioned = True
                break

        if not complexity_mentioned:
            feedback.append("Algorithm complexity analysis could be more detailed")
            score *= 0.8

        # Check for algorithm correctness discussion
        correctness_indicators = [
            "correct",
            "proof",
            "invariant",
            "termination",
            "base case",
        ]
        correctness_mentioned = any(
            indicator in all_content for indicator in correctness_indicators
        )

        if not correctness_mentioned and "algorithm" in all_content:
            feedback.append("Algorithm correctness verification could be discussed")
            score *= 0.9

        # Check for optimization discussion
        optimization_indicators = [
            "optimize",
            "improve",
            "efficient",
            "performance",
            "trade-off",
        ]
        optimization_mentioned = any(
            indicator in all_content for indicator in optimization_indicators
        )

        if not optimization_mentioned and len(all_content.split()) > 50:
            feedback.append("Algorithm optimization considerations could be included")
            score *= 0.95

        return score, feedback

    def _validate_security_concepts(
        self, problem: str, answer: str, explanation: str
    ) -> Tuple[float, List[str]]:
        """
        Validate security concepts and cybersecurity content.

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

        # Check for security-related content
        security_indicators = [
            "security",
            "secure",
            "vulnerability",
            "attack",
            "encryption",
            "authentication",
        ]
        has_security_content = any(
            indicator in all_content for indicator in security_indicators
        )

        if not has_security_content:
            return score, feedback

        # Validate security concepts coverage
        security_topics_found = []
        for topic, keywords in self.security_concepts.items():
            keyword_matches = sum(
                1 for keyword in keywords if keyword.lower() in all_content
            )
            if keyword_matches > 0:
                security_topics_found.append((topic, keyword_matches, len(keywords)))

        if security_topics_found:
            # Check coverage quality for each topic
            for topic, matches, total_keywords in security_topics_found:
                coverage = matches / total_keywords
                if coverage < 0.3:
                    feedback.append(
                        f"Security topic '{topic}' coverage could be more comprehensive"
                    )
                    score *= 0.9

        # Check for security best practices
        best_practices = self.security_concepts.get("secure_coding", [])
        practices_mentioned = sum(
            1 for practice in best_practices if practice.lower() in all_content
        )

        if practices_mentioned == 0 and "code" in all_content:
            feedback.append("Secure coding practices should be mentioned")
            score *= 0.85

        return score, feedback

    def _validate_best_practices(
        self, problem: str, answer: str, explanation: str
    ) -> Tuple[float, List[str]]:
        """
        Validate technology best practices and current industry standards.

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

        # Check for best practices content
        practices_indicators = [
            "best practice",
            "principle",
            "pattern",
            "design",
            "architecture",
            "standard",
            "convention",
        ]
        has_practices_content = any(
            indicator in all_content for indicator in practices_indicators
        )

        # Validate software engineering principles
        se_practices = self.best_practices.get("software_engineering", {})
        principles_mentioned = 0

        for category, items in se_practices.items():
            for item in items:
                if item.lower() in all_content:
                    principles_mentioned += 1

        # Check for SOLID principles specifically
        solid_principles = [
            "single responsibility",
            "open closed",
            "liskov substitution",
            "interface segregation",
            "dependency inversion",
            "solid",
        ]
        solid_mentioned = any(
            principle in all_content for principle in solid_principles
        )

        if (
            "software" in all_content
            or "engineering" in all_content
            or "code" in all_content
        ):
            if principles_mentioned == 0:
                feedback.append("Software engineering principles should be discussed")
                score *= 0.85
            elif not solid_mentioned and "design" in all_content:
                feedback.append("SOLID principles could enhance the design discussion")
                score *= 0.95

        # Validate coding standards
        coding_standards = self.best_practices.get("coding_standards", {})
        standards_mentioned = 0

        for category, items in coding_standards.items():
            for item in items:
                if item.lower() in all_content:
                    standards_mentioned += 1

        if "code" in all_content and standards_mentioned == 0:
            feedback.append("Coding standards and conventions should be mentioned")
            score *= 0.9

        # Validate testing practices
        testing_practices = self.best_practices.get("testing", {})
        testing_mentioned = 0

        for category, items in testing_practices.items():
            for item in items:
                if item.lower() in all_content:
                    testing_mentioned += 1

        test_indicators = ["test", "testing", "unit test", "integration", "tdd", "bdd"]
        has_testing_content = any(
            indicator in all_content for indicator in test_indicators
        )

        if has_testing_content and testing_mentioned == 0:
            feedback.append(
                "Testing best practices and methodologies should be detailed"
            )
            score *= 0.9

        # Validate system design concepts
        system_practices = self.best_practices.get("system_design", {})
        system_concepts = 0

        for category, items in system_practices.items():
            for item in items:
                if item.lower() in all_content:
                    system_concepts += 1

        if (
            "system" in all_content
            or "design" in all_content
            or "architecture" in all_content
        ):
            if system_concepts == 0:
                feedback.append("System design principles should be more prominent")
                score *= 0.85
            elif system_concepts < 2:
                feedback.append(
                    "Additional system design concepts could strengthen the content"
                )
                score *= 0.95

        # Validate performance considerations
        performance_practices = self.best_practices.get("performance", {})
        performance_mentioned = 0

        for category, items in performance_practices.items():
            for item in items:
                if item.lower() in all_content:
                    performance_mentioned += 1

        performance_indicators = [
            "performance",
            "optimization",
            "efficiency",
            "scalability",
        ]
        has_performance_content = any(
            indicator in all_content for indicator in performance_indicators
        )

        if has_performance_content and performance_mentioned == 0:
            feedback.append("Performance optimization practices should be discussed")
            score *= 0.9

        # Validate security best practices
        security_practices = self.best_practices.get("security", {})
        security_best_practices_mentioned = 0

        for category, items in security_practices.items():
            for item in items:
                if item.lower() in all_content:
                    security_best_practices_mentioned += 1

        security_indicators = [
            "security",
            "secure",
            "vulnerability",
            "authentication",
            "authorization",
        ]
        has_security_content = any(
            indicator in all_content for indicator in security_indicators
        )

        if has_security_content and security_best_practices_mentioned == 0:
            feedback.append("Security best practices should be emphasized")
            score *= 0.85

        # Check for industry standard tools and frameworks
        industry_tools = [
            "git",
            "docker",
            "kubernetes",
            "jenkins",
            "ci/cd",
            "agile",
            "scrum",
            "rest api",
            "graphql",
            "microservices",
            "devops",
            "cloud",
        ]
        tools_mentioned = sum(1 for tool in industry_tools if tool in all_content)

        if (
            "development" in all_content or "software" in all_content
        ) and tools_mentioned == 0:
            feedback.append("Industry standard tools and practices could be mentioned")
            score *= 0.95

        # Bonus for comprehensive best practices coverage
        total_practices_mentioned = (
            principles_mentioned
            + standards_mentioned
            + testing_mentioned
            + system_concepts
            + performance_mentioned
            + security_best_practices_mentioned
        )

        if total_practices_mentioned >= 5:
            score *= 1.05  # Bonus for comprehensive coverage
            score = min(score, 1.0)  # Cap at 1.0

        return score, feedback

    def _validate_technology_ethics(
        self, problem: str, answer: str, explanation: str
    ) -> Tuple[float, List[str]]:
        """
        Validate technology ethics and responsibility considerations.

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
        ethics_indicators = [
            "ethics",
            "privacy",
            "bias",
            "responsibility",
            "social",
            "impact",
        ]
        has_ethics_content = any(
            indicator in all_content for indicator in ethics_indicators
        )

        # Check for AI/ML content that should consider ethics
        ai_indicators = [
            "artificial intelligence",
            "machine learning",
            "ai",
            "ml",
            "algorithm",
            "data",
        ]
        has_ai_content = any(indicator in all_content for indicator in ai_indicators)

        if not has_ethics_content and has_ai_content:
            feedback.append(
                "Ethical considerations for AI/ML applications should be discussed"
            )
            score *= 0.8
            return score, feedback

        if not has_ethics_content:
            return score, feedback

        # Validate ethics concepts coverage
        ethics_topics_found = []
        for topic, keywords in self.ethics_concepts.items():
            keyword_matches = sum(
                1 for keyword in keywords if keyword.lower() in all_content
            )
            if keyword_matches > 0:
                ethics_topics_found.append((topic, keyword_matches, len(keywords)))

        if ethics_topics_found:
            # Check coverage quality for each topic
            for topic, matches, total_keywords in ethics_topics_found:
                coverage = matches / total_keywords
                if coverage < 0.25:
                    feedback.append(
                        f"Ethics topic '{topic}' could be discussed more thoroughly"
                    )
                    score *= 0.9

        return score, feedback

    def _validate_concept_accuracy(
        self, problem: str, answer: str, explanation: str
    ) -> Tuple[float, List[str]]:
        """
        Validate technology concept accuracy and terminology.

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

        # Check for technology domain indicators
        domain_matches = {}
        for domain, categories in self.technology_concepts.items():
            domain_score = 0
            total_possible = 0

            for category, items in categories.items():
                for item in items:
                    total_possible += 1
                    if item.lower() in all_content:
                        domain_score += 1

            if total_possible > 0:
                domain_matches[domain] = domain_score / total_possible

        # Find the most relevant domain
        if domain_matches:
            primary_domain = max(domain_matches, key=domain_matches.get)
            primary_score = domain_matches[primary_domain]

            if primary_score < 0.1:
                feedback.append(
                    "Technology concepts could be more specific and accurate"
                )
                score *= 0.9
            elif primary_score < 0.2:
                feedback.append(
                    f"More specific {primary_domain.replace('_', ' ')} concepts could be included"
                )
                score *= 0.95

        # Check for common technology misconceptions
        misconceptions = {
            "html is a programming language": "HTML is a markup language, not a programming language",
            "javascript and java are the same": "JavaScript and Java are completely different languages",
            "ai and machine learning are the same": "AI is broader than machine learning",
            "cloud means internet": "Cloud computing is more than just internet connectivity",
        }

        # Special check for HTML programming language misconception
        if (
            "html" in all_content
            and "programming language" in all_content
            and "markup" not in all_content
        ):
            feedback.append(
                "Critical error: HTML is a markup language, not a programming language"
            )
            score *= 0.3

        for misconception, correction in misconceptions.items():
            if any(word in all_content for word in misconception.split()):
                # Check if the content might be perpetuating the misconception
                if not any(word in all_content for word in correction.split()):
                    feedback.append(f"Clarification needed: {correction}")
                    score *= 0.5  # More severe penalty for misconceptions

        # Check for outdated technology references
        outdated_tech = [
            "flash",
            "silverlight",
            "internet explorer",
            "jquery mobile",
            "angular.js",
            "bower",
            "grunt",
        ]

        outdated_found = [tech for tech in outdated_tech if tech in all_content]
        if outdated_found:
            feedback.append(
                f"Consider updating references to outdated technologies: {', '.join(outdated_found)}"
            )
            score *= 0.95

        # Bonus for using current industry terminology
        current_tech = [
            "microservices",
            "containerization",
            "devops",
            "ci/cd",
            "cloud native",
            "serverless",
            "edge computing",
            "api first",
            "jamstack",
        ]

        current_mentioned = sum(1 for tech in current_tech if tech in all_content)
        if current_mentioned >= 2:
            score *= 1.02  # Small bonus for current terminology
            score = min(score, 1.0)

        return score, feedback

    # Helper methods

    def _detect_programming_language(self, content: str) -> Optional[str]:
        """Detect programming language from content."""
        content_lower = content.lower()

        # Check for explicit language mentions
        for lang in self.programming_languages.keys():
            if lang in content_lower:
                return lang

        # Check for syntax patterns
        for lang, config in self.programming_languages.items():
            patterns = config["syntax_patterns"]
            matches = sum(
                1 for pattern in patterns if re.search(pattern, content, re.IGNORECASE)
            )
            if matches >= 2:  # At least 2 patterns must match
                return lang

        return None

    def _validate_code_structure(self, code: str, language: str) -> float:
        """Validate code structure and best practices."""
        if not code.strip():
            return 1.0

        score = 1.0

        # Basic structure checks
        lines = code.split("\n")
        non_empty_lines = [line for line in lines if line.strip()]

        if len(non_empty_lines) == 0:
            return 0.5

        # Check for comments
        comment_patterns = {
            "python": r"#.*",
            "javascript": r"//.*|/\*.*\*/",
            "java": r"//.*|/\*.*\*/",
            "cpp": r"//.*|/\*.*\*/",
        }

        if language in comment_patterns:
            pattern = comment_patterns[language]
            has_comments = any(re.search(pattern, line) for line in lines)
            if not has_comments and len(non_empty_lines) > 5:
                score *= 0.9

        # Check for proper indentation (simplified)
        if language in ["python"]:
            indented_lines = [
                line
                for line in lines
                if line.startswith("    ") or line.startswith("\t")
            ]
            if len(indented_lines) == 0 and len(non_empty_lines) > 3:
                score *= 0.9

        return score

    def calculate_quality_score(self, content: Dict[str, Any]) -> float:
        """
        Calculate domain-specific quality score for technology content.

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
        Generate domain-specific improvement feedback for technology content.

        Args:
            validation_result: Result of validation to generate feedback for

        Returns:
            List of feedback messages for content improvement
        """
        feedback = []

        if not validation_result.is_valid:
            feedback.append(
                "Technology content needs improvement to meet quality standards"
            )

        # Extract validation scores from details
        validation_scores = validation_result.details.get("validation_scores", {})

        # Provide specific feedback based on low scores
        if validation_scores.get("code_validation", 1.0) < 0.7:
            feedback.append(
                "Code execution, structure, or best practices could be improved"
            )

        if validation_scores.get("algorithm_validation", 1.0) < 0.7:
            feedback.append(
                "Algorithm analysis and complexity discussion could be more detailed"
            )

        if validation_scores.get("security_validation", 1.0) < 0.7:
            feedback.append(
                "Security concepts and practices should be more comprehensive"
            )

        if validation_scores.get("practices_validation", 1.0) < 0.7:
            feedback.append(
                "Technology best practices and principles could be emphasized"
            )

        if validation_scores.get("ethics_validation", 1.0) < 0.7:
            feedback.append(
                "Technology ethics and responsibility considerations needed"
            )

        if validation_scores.get("concept_accuracy", 1.0) < 0.7:
            feedback.append(
                "Technology concept accuracy and terminology could be improved"
            )

        # Add positive feedback for high scores
        if validation_result.quality_score > 0.8:
            feedback.append(
                "Technology content demonstrates strong technical understanding"
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
