"""
Unit tests for the Technology domain validator.

This module tests all aspects of technology content validation including
code execution, algorithm analysis, security validation, and
technology best practices.
"""

# Standard Library
from unittest.mock import MagicMock, patch

# Third-Party Library
import pytest

# SynThesisAI Modules
from core.validation.config import ValidationConfig
from core.validation.domains.technology import (
    SandboxedCodeExecutor,
    TechnologyValidator,
)


class TestSandboxedCodeExecutor:
    """Test suite for SandboxedCodeExecutor class."""

    @pytest.fixture
    def executor(self):
        """Create a code executor instance for testing."""
        return SandboxedCodeExecutor(timeout_seconds=3, memory_limit_mb=64)

    def test_executor_initialization(self, executor):
        """Test code executor initialization."""
        assert executor.timeout_seconds == 3
        assert executor.memory_limit_mb == 64
        assert len(executor.language_configs) > 0
        assert "python" in executor.language_configs
        assert "javascript" in executor.language_configs

    def test_security_check_safe_code(self, executor):
        """Test security check with safe code."""
        safe_python_code = "print('Hello, World!')\nx = 5 + 3\nprint(x)"
        assert executor._security_check(safe_python_code, "python") is True

    def test_security_check_unsafe_code(self, executor):
        """Test security check with unsafe code."""
        unsafe_python_code = "import os\nos.system('rm -rf /')"
        assert executor._security_check(unsafe_python_code, "python") is False

    def test_execute_python_code_success(self, executor):
        """Test successful Python code execution."""
        code = "print('Hello, World!')"
        result = executor.execute_code(code, "python")

        assert result["success"] is True
        assert "Hello, World!" in result["output"]
        assert result["execution_time"] > 0

    def test_execute_python_code_with_error(self, executor):
        """Test Python code execution with error."""
        code = "print(undefined_variable)"
        result = executor.execute_code(code, "python")

        assert result["success"] is False
        assert "NameError" in result["error"] or "undefined_variable" in result["error"]

    def test_execute_unsupported_language(self, executor):
        """Test execution with unsupported language."""
        code = "some code"
        result = executor.execute_code(code, "unsupported_lang")

        assert result["success"] is False
        assert "Unsupported language" in result["error"]

    def test_execute_javascript_code(self, executor):
        """Test JavaScript code execution."""
        code = "console.log('Hello from Node.js');"
        result = executor.execute_code(code, "javascript")

        # Should either succeed or fail gracefully
        assert "success" in result
        assert "output" in result
        assert "error" in result


class TestTechnologyValidator:
    """Test suite for TechnologyValidator class."""

    @pytest.fixture
    def validator(self):
        """Create a technology validator instance for testing."""
        config = ValidationConfig(
            domain="technology", quality_thresholds={"technology_score": 0.7}
        )
        return TechnologyValidator("technology", config)

    @pytest.fixture
    def cs_validator(self):
        """Create a computer science validator instance."""
        config = ValidationConfig(
            domain="technology", quality_thresholds={"technology_score": 0.7}
        )
        return TechnologyValidator("computer_science", config)

    def test_validator_initialization(self, validator):
        """Test technology validator initialization."""
        assert validator.domain == "technology"
        assert validator.subdomain == "technology"
        assert validator.config is not None
        assert len(validator.programming_languages) > 0
        assert len(validator.security_concepts) > 0
        assert validator.code_executor is not None

    def test_invalid_subdomain_raises_error(self):
        """Test that invalid subdomain raises ValueError."""
        config = ValidationConfig(domain="technology")

        with pytest.raises(ValueError, match="Invalid technology subdomain"):
            TechnologyValidator("invalid_subdomain", config)

    def test_valid_subdomains(self):
        """Test that all valid technology subdomains work."""
        config = ValidationConfig(domain="technology")
        valid_subdomains = [
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
        ]

        for subdomain in valid_subdomains:
            validator = TechnologyValidator(subdomain, config)
            assert validator.subdomain == subdomain

    def test_validate_python_code_execution(self, validator):
        """Test validation of Python code execution."""
        content = {
            "problem": "Write a Python function to calculate factorial.",
            "answer": "The function uses recursion to calculate factorial.",
            "explanation": "Factorial of n is n * factorial(n-1), with base case factorial(0) = 1.",
            "code": "def factorial(n):\n    if n == 0:\n        return 1\n    return n * factorial(n-1)\n\nprint(factorial(5))",
        }

        result = validator.validate_content(content)

        assert result.domain == "technology"
        assert result.validation_details["subdomain"] == "technology"
        assert result.is_valid is True
        assert result.quality_score > 0.7
        assert "code_validation" in result.validation_details["validation_scores"]

    def test_validate_algorithm_complexity_analysis(self, validator):
        """Test validation of algorithm complexity analysis."""
        content = {
            "problem": "Analyze the time complexity of binary search.",
            "answer": "Binary search has O(log n) time complexity.",
            "explanation": "Binary search divides the search space in half at each step, leading to logarithmic time complexity. The algorithm is efficient for sorted arrays and has constant space complexity O(1).",
        }

        result = validator.validate_content(content)

        assert result.is_valid is True
        assert result.quality_score > 0.7
        assert "algorithm_validation" in result.validation_details["validation_scores"]

    def test_validate_security_concepts(self, validator):
        """Test validation of cybersecurity concepts."""
        content = {
            "problem": "Explain SQL injection attacks and prevention.",
            "answer": "SQL injection occurs when user input is not properly sanitized in database queries.",
            "explanation": "Attackers can inject malicious SQL code through input fields. Prevention includes using parameterized queries, input validation, and least privilege principles. Proper authentication and authorization are also essential.",
        }

        result = validator.validate_content(content)

        assert result.is_valid is True
        assert "security_validation" in result.validation_details["validation_scores"]
        assert (
            result.validation_details["validation_scores"]["security_validation"] > 0.7
        )

    def test_validate_software_engineering_best_practices(self, validator):
        """Test validation of software engineering best practices."""
        content = {
            "problem": "Discuss SOLID principles in software design.",
            "answer": "SOLID principles guide object-oriented design for maintainable code.",
            "explanation": "The SOLID principles include Single Responsibility, Open/Closed, Liskov Substitution, Interface Segregation, and Dependency Inversion. These principles promote code that is easy to maintain, extend, and test. Code review and version control are also essential practices.",
        }

        result = validator.validate_content(content)

        assert result.is_valid is True
        assert "practices_validation" in result.validation_details["validation_scores"]
        assert (
            result.validation_details["validation_scores"]["practices_validation"] > 0.6
        )

    def test_validate_technology_ethics(self, validator):
        """Test validation of technology ethics considerations."""
        content = {
            "problem": "What ethical considerations apply to AI systems?",
            "answer": "AI systems must consider bias, privacy, and accountability.",
            "explanation": "Algorithmic bias can lead to unfair discrimination. Privacy protection requires careful data handling and user consent. Transparency and explainability help ensure accountability. Social impact assessment is crucial for responsible AI development.",
        }

        result = validator.validate_content(content)

        assert result.is_valid is True
        assert "ethics_validation" in result.validation_details["validation_scores"]
        assert result.validation_details["validation_scores"]["ethics_validation"] > 0.7

    def test_validate_data_structures_content(self, validator):
        """Test validation of data structures content."""
        content = {
            "problem": "Compare arrays and linked lists.",
            "answer": "Arrays provide O(1) access but fixed size, while linked lists allow dynamic size but O(n) access.",
            "explanation": "Arrays store elements in contiguous memory with constant-time access by index. Linked lists use pointers to connect nodes, allowing efficient insertion and deletion but requiring traversal for access. The choice depends on the specific use case and operation requirements.",
        }

        result = validator.validate_content(content)

        assert result.is_valid is True
        assert result.quality_score > 0.7

    def test_validate_javascript_code_detection(self, validator):
        """Test detection of JavaScript code."""
        content = {
            "problem": "Write a JavaScript function to reverse a string.",
            "answer": "Use array methods to reverse the string.",
            "explanation": "The function splits the string into an array, reverses it, and joins back to a string.",
            "code": "function reverseString(str) {\n    return str.split('').reverse().join('');\n}\n\nconsole.log(reverseString('hello'));",
        }

        result = validator.validate_content(content)

        assert result.is_valid is True
        assert "code_validation" in result.validation_details["validation_scores"]

    def test_programming_language_detection(self, validator):
        """Test programming language detection."""
        # Test Python detection
        python_content = "def function_name(): import sys class MyClass:"
        assert validator._detect_programming_language(python_content) == "python"

        # Test JavaScript detection
        js_content = "function myFunc() { const x = 5; let y = 10; }"
        assert validator._detect_programming_language(js_content) == "javascript"

        # Test Java detection
        java_content = "public class MyClass { public static void main import java.util"
        assert validator._detect_programming_language(java_content) == "java"

        # Test no detection
        generic_content = "This is just regular text without code patterns"
        assert validator._detect_programming_language(generic_content) is None

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
            "problem": "Test technology problem about algorithms and security",
            "answer": "Algorithms have complexity. Security requires encryption.",
            "explanation": "This covers algorithm analysis and security concepts",
        }

        result = validator.validate_content(content)

        # Check that weights are applied correctly
        expected_weights = {
            "code_validation": 0.22,
            "algorithm_validation": 0.18,
            "security_validation": 0.18,
            "practices_validation": 0.22,
            "ethics_validation": 0.12,
            "concept_accuracy": 0.08,
        }

        assert result.validation_details["weights"] == expected_weights

    def test_quality_threshold_application(self, validator):
        """Test that quality threshold is applied correctly."""
        # Set a high threshold
        validator.config.quality_thresholds["technology_score"] = 0.95

        content = {
            "problem": "Simple technology question",
            "answer": "Simple answer",
            "explanation": "Basic explanation",
        }

        result = validator.validate_content(content)

        # Should fail with high threshold (or check that threshold is applied)
        assert result.validation_details["threshold"] == 0.95
        # The result may still pass if content is good enough, so just check threshold is applied

    def test_computer_science_subdomain(self, cs_validator):
        """Test computer science specific validation."""
        content = {
            "problem": "Explain the concept of recursion in computer science.",
            "answer": "Recursion is when a function calls itself with a base case.",
            "explanation": "Recursive functions solve problems by breaking them into smaller subproblems. A base case prevents infinite recursion. Examples include factorial calculation and tree traversal.",
        }

        result = cs_validator.validate_content(content)

        assert result.validation_details["subdomain"] == "computer_science"
        assert result.is_valid is True

    def test_validation_error_handling(self, validator):
        """Test error handling during validation."""
        # Test with invalid content type
        with patch.object(
            validator,
            "_validate_code_execution",
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

    def test_code_structure_validation(self, validator):
        """Test code structure validation."""
        # Test Python code with comments
        good_code = "# This function calculates factorial\ndef factorial(n):\n    if n == 0:\n        return 1\n    return n * factorial(n-1)"
        score = validator._validate_code_structure(good_code, "python")
        assert score > 0.8

        # Test code without comments
        basic_code = (
            "def factorial(n):\nif n == 0:\nreturn 1\nreturn n * factorial(n-1)"
        )
        score = validator._validate_code_structure(basic_code, "python")
        assert score < 1.0

        # Test empty code
        empty_code = ""
        score = validator._validate_code_structure(empty_code, "python")
        assert score == 1.0

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

    def test_system_design_validation(self, validator):
        """Test validation of system design concepts."""
        content = {
            "problem": "Design a scalable web application architecture.",
            "answer": "Use microservices with load balancing and caching for scalability.",
            "explanation": "Microservices architecture allows independent scaling of components. Load balancing distributes traffic across multiple servers. Caching improves performance by storing frequently accessed data. Database sharding can handle large datasets. Monitoring and logging are essential for system reliability.",
        }

        result = validator.validate_content(content)

        assert result.is_valid is True
        assert result.quality_score > 0.7
        assert "practices_validation" in result.validation_details["validation_scores"]

    def test_ai_ethics_validation(self, validator):
        """Test validation of AI ethics without explicit ethics discussion."""
        content = {
            "problem": "Implement a machine learning model for hiring decisions.",
            "answer": "Use supervised learning with historical hiring data.",
            "explanation": "Train a model on past hiring decisions to predict candidate success. Feature engineering is important for model performance.",
        }

        result = validator.validate_content(content)

        # Should detect missing ethics discussion for AI content
        ethics_score = result.validation_details["validation_scores"][
            "ethics_validation"
        ]
        assert ethics_score < 1.0
        feedback_text = " ".join(result.feedback).lower()
        assert "ethical" in feedback_text or "bias" in feedback_text

    def test_mock_code_executor_failure(self, validator):
        """Test handling of code executor failures."""
        # Mock the code executor to simulate failure
        with patch.object(validator.code_executor, "execute_code") as mock_execute:
            mock_execute.return_value = {
                "success": False,
                "error": "Execution timeout",
                "output": "",
                "execution_time": 5.0,
            }

            content = {
                "problem": "Write a Python function",
                "answer": "Function implementation",
                "explanation": "This function does something",
                "code": "def test(): pass",
            }

            result = validator.validate_content(content)

            # Should handle the failure gracefully
            assert "code_validation" in result.validation_details["validation_scores"]
            code_score = result.validation_details["validation_scores"][
                "code_validation"
            ]
            assert code_score < 1.0


class TestTechnologyValidatorIntegration:
    """Integration tests for technology validator."""

    def test_comprehensive_technology_problem(self):
        """Test validation of a comprehensive technology problem."""
        config = ValidationConfig(
            domain="technology", quality_thresholds={"technology_score": 0.6}
        )
        validator = TechnologyValidator("technology", config)

        content = {
            "problem": "Implement a secure web API with proper authentication, analyze its time complexity, and discuss ethical considerations for user data handling.",
            "answer": "Use JWT tokens for authentication, implement rate limiting, and ensure O(1) lookup for user sessions. Consider privacy and data protection regulations.",
            "explanation": "The API uses JSON Web Tokens for stateless authentication. Rate limiting prevents abuse with O(1) time complexity using hash tables. Input validation prevents SQL injection. Privacy by design principles ensure user data protection. GDPR compliance requires explicit consent and data minimization.",
            "code": "def authenticate_user(token):\n    # Validate JWT token\n    if validate_jwt(token):\n        return get_user_from_token(token)\n    return None",
        }

        result = validator.validate_content(content)

        assert result.is_valid is True
        assert result.quality_score > 0.6
        assert all(
            score > 0
            for score in result.validation_details["validation_scores"].values()
        )

    def test_technology_problem_with_all_components(self):
        """Test technology problem covering all validation aspects."""
        config = ValidationConfig(
            domain="technology", quality_thresholds={"technology_score": 0.5}
        )
        validator = TechnologyValidator("cybersecurity", config)

        content = {
            "problem": "Design a secure password hashing system with proper algorithm analysis, implementation best practices, and ethical considerations.",
            "answer": "Use bcrypt with salt for password hashing, implement proper error handling, and ensure user privacy protection.",
            "explanation": "Bcrypt provides adaptive hashing with configurable work factor, achieving O(2^n) time complexity for security. Salt prevents rainbow table attacks. Secure coding practices include input validation and error handling. Privacy considerations require secure storage and user consent for data processing. The system follows OWASP guidelines for authentication security.",
            "code": "import bcrypt\n\ndef hash_password(password):\n    # Generate salt and hash password\n    salt = bcrypt.gensalt()\n    return bcrypt.hashpw(password.encode('utf-8'), salt)",
        }

        result = validator.validate_content(content)

        assert result.is_valid is True
        assert result.validation_details["subdomain"] == "cybersecurity"

        # Check all validation components were evaluated
        validation_scores = result.validation_details["validation_scores"]
        assert "code_validation" in validation_scores
        assert "algorithm_validation" in validation_scores
        assert "security_validation" in validation_scores
        assert "practices_validation" in validation_scores
        assert "ethics_validation" in validation_scores
