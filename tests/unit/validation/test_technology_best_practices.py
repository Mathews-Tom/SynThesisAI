"""
Unit tests for technology best practices validation.

This module tests the enhanced technology validator's ability to validate
current industry standards, best practices, and technology concept accuracy.
"""

# Standard Library
from typing import Any, Dict

# Third-Party Library
import pytest

# SynThesisAI Modules
from core.validation.config import ValidationConfig
from core.validation.domains.technology import TechnologyValidator


class TestTechnologyBestPracticesValidation:
    """Test technology best practices validation functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = ValidationConfig(domain="technology")
        self.validator = TechnologyValidator(config=self.config)

    def test_validate_solid_principles(self):
        """Test validation of SOLID principles in software engineering content."""
        content = {
            "problem": "Explain the SOLID principles in software engineering",
            "answer": """The SOLID principles are five design principles:
            1. Single Responsibility Principle - A class should have only one reason to change
            2. Open/Closed Principle - Software entities should be open for extension, closed for modification
            3. Liskov Substitution Principle - Objects should be replaceable with instances of their subtypes
            4. Interface Segregation Principle - Many client-specific interfaces are better than one general-purpose interface
            5. Dependency Inversion Principle - Depend on abstractions, not concretions""",
            "explanation": "These principles help create maintainable, flexible, and robust software designs.",
        }

        result = self.validator.validate_content(content)

        assert result.is_valid
        assert result.quality_score > 0.8
        assert "practices_validation" in result.validation_details["validation_scores"]
        assert (
            result.validation_details["validation_scores"]["practices_validation"] > 0.8
        )

    def test_validate_coding_standards(self):
        """Test validation of coding standards and conventions."""
        content = {
            "problem": "What are important coding standards for Python development?",
            "answer": """Important Python coding standards include:
            - Use snake_case for variable and function names
            - Use PascalCase for class names
            - Follow PEP 8 for formatting and indentation
            - Write comprehensive docstrings for functions and classes
            - Keep line length under 88 characters
            - Use meaningful variable names""",
            "explanation": "Following coding standards improves code readability and maintainability.",
        }

        result = self.validator.validate_content(content)

        assert result.is_valid
        assert result.quality_score > 0.75
        assert "practices_validation" in result.validation_details["validation_scores"]

    def test_validate_testing_best_practices(self):
        """Test validation of testing methodologies and best practices."""
        content = {
            "problem": "Explain Test-Driven Development (TDD) best practices",
            "answer": """TDD follows the Red-Green-Refactor cycle:
            1. Write a failing test (Red)
            2. Write minimal code to make it pass (Green)
            3. Refactor the code while keeping tests passing
            
            Best practices include:
            - Write unit tests for individual components
            - Use mocking for external dependencies
            - Maintain high test coverage (>80%)
            - Use descriptive test names
            - Keep tests independent and isolated""",
            "explanation": "TDD improves code quality and reduces bugs through systematic testing.",
        }

        result = self.validator.validate_content(content)

        assert result.is_valid
        assert result.quality_score > 0.8
        assert "practices_validation" in result.validation_details["validation_scores"]

    def test_validate_system_design_principles(self):
        """Test validation of system design principles and patterns."""
        content = {
            "problem": "Design a scalable microservices architecture",
            "answer": """A scalable microservices architecture should include:
            - Service decomposition based on business domains
            - API Gateway for request routing and load balancing
            - Service discovery mechanism
            - Circuit breaker pattern for fault tolerance
            - Event-driven communication between services
            - Centralized logging and monitoring
            - Database per service pattern
            - Containerization with Docker and Kubernetes""",
            "explanation": "This architecture provides scalability, reliability, and maintainability.",
        }

        result = self.validator.validate_content(content)

        assert result.is_valid
        assert result.quality_score > 0.85
        assert "practices_validation" in result.validation_details["validation_scores"]
        assert (
            result.validation_details["validation_scores"]["practices_validation"] > 0.8
        )

    def test_validate_performance_optimization(self):
        """Test validation of performance optimization practices."""
        content = {
            "problem": "How to optimize application performance?",
            "answer": """Performance optimization strategies include:
            - Algorithmic optimization to reduce time complexity
            - Caching frequently accessed data
            - Database query optimization and indexing
            - Load balancing to distribute traffic
            - Code profiling to identify bottlenecks
            - Memory management and garbage collection tuning
            - CDN usage for static content delivery
            - Asynchronous processing for I/O operations""",
            "explanation": "Systematic performance optimization improves user experience and resource efficiency.",
        }

        result = self.validator.validate_content(content)

        assert result.is_valid
        assert result.quality_score > 0.8
        assert "practices_validation" in result.validation_details["validation_scores"]

    def test_validate_security_best_practices(self):
        """Test validation of security best practices."""
        content = {
            "problem": "What are essential security best practices for web applications?",
            "answer": """Essential security best practices include:
            - Input validation and sanitization
            - Output encoding to prevent XSS
            - Parameterized queries to prevent SQL injection
            - Authentication and authorization mechanisms
            - HTTPS encryption for data in transit
            - Secure session management
            - Regular security audits and penetration testing
            - Principle of least privilege
            - Security headers (CSP, HSTS, etc.)""",
            "explanation": "These practices protect against common vulnerabilities and attacks.",
        }

        result = self.validator.validate_content(content)

        assert result.is_valid
        assert result.quality_score > 0.85
        assert "practices_validation" in result.validation_details["validation_scores"]
        assert "security_validation" in result.validation_details["validation_scores"]

    def test_validate_industry_standard_tools(self):
        """Test validation of current industry standard tools and practices."""
        content = {
            "problem": "What tools are essential for modern software development?",
            "answer": """Essential modern development tools include:
            - Git for version control and collaboration
            - Docker for containerization and deployment
            - Kubernetes for container orchestration
            - Jenkins or GitHub Actions for CI/CD pipelines
            - Cloud platforms like AWS, Azure, or Google Cloud
            - Monitoring tools like Prometheus and Grafana
            - Agile methodologies and Scrum framework
            - REST APIs and GraphQL for service communication""",
            "explanation": "These tools enable efficient, scalable, and collaborative development.",
        }

        result = self.validator.validate_content(content)

        assert result.is_valid
        assert result.quality_score > 0.8
        assert "practices_validation" in result.validation_details["validation_scores"]

    def test_validate_concept_accuracy(self):
        """Test validation of technology concept accuracy."""
        content = {
            "problem": "Explain the difference between JavaScript and Java",
            "answer": """JavaScript and Java are completely different programming languages:
            
            JavaScript:
            - Interpreted scripting language
            - Primarily used for web development
            - Dynamic typing
            - Runs in browsers and Node.js
            - Prototype-based object-oriented
            
            Java:
            - Compiled to bytecode
            - Platform-independent (JVM)
            - Static typing
            - Used for enterprise applications
            - Class-based object-oriented""",
            "explanation": "Despite similar names, these languages have different purposes and characteristics.",
        }

        result = self.validator.validate_content(content)

        assert result.is_valid
        assert result.quality_score > 0.8
        assert "concept_accuracy" in result.validation_details["validation_scores"]
        assert result.validation_details["validation_scores"]["concept_accuracy"] > 0.9

    def test_detect_technology_misconceptions(self):
        """Test detection of common technology misconceptions."""
        content = {
            "problem": "Is HTML a programming language?",
            "answer": "Yes, HTML is a programming language used to create websites.",
            "explanation": "HTML allows you to program the structure of web pages.",
        }

        result = self.validator.validate_content(content)

        # Check that the misconception is detected in feedback
        assert "concept_accuracy" in result.validation_details["validation_scores"]
        assert result.validation_details["validation_scores"]["concept_accuracy"] < 0.5
        assert any(
            "HTML is a markup language" in feedback for feedback in result.feedback
        )

    def test_detect_outdated_technology_references(self):
        """Test detection of outdated technology references."""
        content = {
            "problem": "How to create interactive web content?",
            "answer": """You can create interactive web content using:
            - Adobe Flash for animations and games
            - jQuery Mobile for mobile web apps
            - Internet Explorer specific features
            - Silverlight for rich media applications""",
            "explanation": "These technologies provide rich interactive experiences.",
        }

        result = self.validator.validate_content(content)

        assert "concept_accuracy" in result.validation_details["validation_scores"]
        assert result.validation_details["validation_scores"]["concept_accuracy"] < 0.8
        assert any("outdated technologies" in feedback for feedback in result.feedback)

    def test_bonus_for_current_technology_terminology(self):
        """Test bonus scoring for using current technology terminology."""
        content = {
            "problem": "Describe modern application architecture",
            "answer": """Modern application architecture includes:
            - Microservices for service decomposition
            - Containerization with Docker
            - DevOps practices for continuous delivery
            - CI/CD pipelines for automated deployment
            - Cloud native design principles
            - Serverless functions for event-driven processing
            - Edge computing for reduced latency
            - API-first development approach""",
            "explanation": "This architecture leverages current best practices and technologies.",
        }

        result = self.validator.validate_content(content)

        assert result.is_valid
        assert result.quality_score > 0.85
        assert "concept_accuracy" in result.validation_details["validation_scores"]

    def test_comprehensive_best_practices_coverage(self):
        """Test comprehensive coverage of multiple best practice areas."""
        content = {
            "problem": "Design a complete software development process",
            "answer": """A complete software development process should include:
            
            Software Engineering:
            - SOLID principles for design
            - DRY (Don't Repeat Yourself) principle
            - Design patterns like MVC and Observer
            - Code review processes
            
            Testing:
            - Test-driven development (TDD)
            - Unit testing with pytest
            - Integration testing
            - Continuous testing in CI/CD
            
            System Design:
            - Microservices architecture
            - Load balancing and caching
            - Database sharding for scalability
            - Circuit breaker pattern for fault tolerance
            
            Performance:
            - Code profiling and optimization
            - Monitoring with metrics and logging
            - Performance testing and benchmarks
            
            Security:
            - Input validation and sanitization
            - Authentication and authorization
            - Static code analysis
            - Penetration testing""",
            "explanation": "This comprehensive approach ensures high-quality, maintainable software.",
        }

        result = self.validator.validate_content(content)

        assert result.is_valid
        assert result.quality_score > 0.9
        assert "practices_validation" in result.validation_details["validation_scores"]
        assert (
            result.validation_details["validation_scores"]["practices_validation"] > 0.9
        )

    def test_insufficient_best_practices_content(self):
        """Test handling of content with insufficient best practices coverage."""
        content = {
            "problem": "Write a simple function",
            "answer": "def add(a, b): return a + b",
            "explanation": "This function adds two numbers.",
        }

        result = self.validator.validate_content(content)

        # Should still be valid but with lower score due to minimal content
        assert result.quality_score < 0.95
        practices_score = result.validation_details["validation_scores"][
            "practices_validation"
        ]
        assert practices_score <= 1.0

    def test_validate_with_mixed_quality_practices(self):
        """Test validation with mixed quality best practices content."""
        content = {
            "problem": "Explain software testing",
            "answer": """Software testing involves:
            - Writing some tests
            - Running tests sometimes
            - Fixing bugs when found
            - Using testing tools""",
            "explanation": "Testing helps find problems in code.",
        }

        result = self.validator.validate_content(content)

        # Should have moderate score due to basic coverage
        assert 0.7 < result.quality_score < 0.9
        assert "practices_validation" in result.validation_details["validation_scores"]
        assert any("best practices" in feedback for feedback in result.feedback)


class TestTechnologyValidatorIntegration:
    """Test integration of all technology validation components."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = ValidationConfig(domain="technology")
        self.validator = TechnologyValidator(config=self.config)

    def test_comprehensive_technology_validation(self):
        """Test comprehensive validation across all technology areas."""
        content = {
            "problem": "Design and implement a secure, scalable web API",
            "answer": """To design a secure, scalable web API:

            Architecture:
            - Use microservices architecture with API Gateway
            - Implement RESTful design principles
            - Use JSON for data exchange
            - Apply circuit breaker pattern for fault tolerance

            Security:
            - Implement OAuth 2.0 for authentication
            - Use HTTPS for all communications
            - Apply input validation and sanitization
            - Implement rate limiting to prevent abuse

            Code Implementation (Python):
            ```python
            from flask import Flask, request, jsonify
            from flask_limiter import Limiter
            import jwt

            app = Flask(__name__)
            limiter = Limiter(app, key_func=lambda: request.remote_addr)

            @app.route('/api/users', methods=['GET'])
            @limiter.limit("100 per hour")
            def get_users():
                # Validate authentication token
                token = request.headers.get('Authorization')
                if not validate_token(token):
                    return jsonify({'error': 'Unauthorized'}), 401
                
                # Return user data
                return jsonify({'users': get_user_data()})
            ```

            Performance:
            - Use caching for frequently accessed data
            - Implement database connection pooling
            - Use asynchronous processing for heavy operations
            - Monitor API performance with metrics

            Testing:
            - Write unit tests for all endpoints
            - Implement integration tests
            - Use automated testing in CI/CD pipeline
            - Perform load testing for scalability

            Best Practices:
            - Follow SOLID principles in code design
            - Use version control with Git
            - Implement proper error handling
            - Document API with OpenAPI/Swagger""",
            "explanation": """This comprehensive approach ensures the API is secure, scalable, and maintainable.
            The implementation follows current industry standards and best practices.""",
            "code": """
from flask import Flask, request, jsonify
from flask_limiter import Limiter
import jwt

app = Flask(__name__)
limiter = Limiter(app, key_func=lambda: request.remote_addr)

@app.route('/api/users', methods=['GET'])
@limiter.limit("100 per hour")
def get_users():
    token = request.headers.get('Authorization')
    if not validate_token(token):
        return jsonify({'error': 'Unauthorized'}), 401
    
    return jsonify({'users': get_user_data()})
""",
        }

        result = self.validator.validate_content(content)

        assert result.is_valid
        assert result.quality_score > 0.8

        # Check all validation components
        validation_scores = result.validation_details["validation_scores"]
        assert "code_validation" in validation_scores
        assert "algorithm_validation" in validation_scores
        assert "security_validation" in validation_scores
        assert "practices_validation" in validation_scores
        assert "ethics_validation" in validation_scores
        assert "concept_accuracy" in validation_scores

        # Most scores should be reasonably high (code validation may be lower due to security checks)
        for score_type, score in validation_scores.items():
            if score_type == "code_validation":
                assert score > 0.5, f"{score_type} score too low: {score}"
            else:
                assert score > 0.7, f"{score_type} score too low: {score}"

    def test_technology_validation_feedback_generation(self):
        """Test comprehensive feedback generation for technology content."""
        content = {
            "problem": "Create a web application",
            "answer": "Use HTML and some JavaScript to make a website.",
            "explanation": "Web development is easy with these tools.",
        }

        result = self.validator.validate_content(content)

        # Check that feedback is provided for improvement
        assert len(result.feedback) > 0

        # Should provide specific feedback for improvement
        feedback_text = " ".join(result.feedback)
        # Check for any improvement-related keywords
        improvement_keywords = [
            "best practices",
            "security",
            "standards",
            "principles",
            "could",
            "should",
            "improve",
        ]
        assert any(
            keyword in feedback_text.lower() for keyword in improvement_keywords
        ), f"Expected improvement feedback, got: {feedback_text}"
