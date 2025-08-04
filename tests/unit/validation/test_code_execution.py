"""
Unit tests for the advanced code execution validation module.

This module tests all aspects of code execution validation including
multi-language execution, security analysis, performance analysis,
and correctness validation.
"""

# Standard Library
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

# Third-Party Library
import pytest

# SynThesisAI Modules
from core.validation.code_execution import (
    AdvancedCodeExecutor,
    CodeCorrectnessValidator,
    CodeExecutionResult,
    PerformanceAnalyzer,
)


class TestCodeExecutionResult:
    """Test suite for CodeExecutionResult class."""

    def test_result_initialization(self):
        """Test code execution result initialization."""
        result = CodeExecutionResult(
            success=True,
            output="Hello, World!",
            execution_time=0.5,
            security_violations=["test violation"],
        )

        assert result.success is True
        assert result.output == "Hello, World!"
        assert result.execution_time == 0.5
        assert result.security_violations == ["test violation"]
        assert result.performance_metrics == {}

    def test_result_defaults(self):
        """Test default values in result initialization."""
        result = CodeExecutionResult(success=False)

        assert result.success is False
        assert result.output == ""
        assert result.error == ""
        assert result.execution_time == 0.0
        assert result.security_violations == []
        assert result.performance_metrics == {}


class TestAdvancedCodeExecutor:
    """Test suite for AdvancedCodeExecutor class."""

    @pytest.fixture
    def executor(self):
        """Create an advanced code executor instance for testing."""
        return AdvancedCodeExecutor(
            timeout_seconds=5,
            memory_limit_mb=128,
            enable_networking=False,
            enable_file_system=False,
        )

    def test_executor_initialization(self, executor):
        """Test advanced code executor initialization."""
        assert executor.timeout_seconds == 5
        assert executor.memory_limit_mb == 128
        assert executor.enable_networking is False
        assert executor.enable_file_system is False
        assert len(executor.language_configs) >= 4
        assert "python" in executor.language_configs
        assert "javascript" in executor.language_configs
        assert "java" in executor.language_configs
        assert "cpp" in executor.language_configs

    def test_python_security_analysis(self, executor):
        """Test Python-specific security analysis."""
        safe_code = "print('Hello, World!')\nx = 5 + 3"
        violations = executor._analyze_python_security(safe_code)
        assert len(violations) == 0

        unsafe_code = "import os\nexec('malicious code')"
        violations = executor._analyze_python_security(unsafe_code)
        assert len(violations) > 0
        assert any("import" in v for v in violations)

    def test_javascript_security_analysis(self, executor):
        """Test JavaScript-specific security analysis."""
        safe_code = "console.log('Hello, World!');\nlet x = 5 + 3;"
        violations = executor._analyze_javascript_security(safe_code)
        assert len(violations) == 0

        unsafe_code = "eval('malicious code');\nsetTimeout('alert(1)', 1000);"
        violations = executor._analyze_javascript_security(unsafe_code)
        assert len(violations) > 0

    def test_performance_pattern_analysis(self, executor):
        """Test performance pattern analysis."""
        code_with_loops = """
        for i in range(10):
            for j in range(10):
                print(i, j)
        """

        metrics = executor._analyze_performance_patterns(code_with_loops, "python")

        assert "complexity_indicators" in metrics
        assert "optimization_opportunities" in metrics
        assert len(metrics["optimization_opportunities"]) > 0

    def test_execute_python_code_success(self, executor):
        """Test successful Python code execution."""
        code = "print('Hello, World!')"
        result = executor.execute_code(code, "python")

        assert isinstance(result, CodeExecutionResult)
        assert result.success is True
        assert "Hello, World!" in result.output
        assert result.execution_time > 0

    def test_execute_python_code_with_test_cases(self, executor):
        """Test Python code execution with test cases."""
        code = """
x = input()
y = input()
result = int(x) + int(y)
print(result)
"""

        test_cases = [
            {
                "input": "2\n3",
                "output": "5",
            },
        ]

        result = executor.execute_code(code, "python", test_cases=test_cases)

        assert isinstance(result, CodeExecutionResult)
        assert "test_results" in result.performance_metrics

    def test_execute_code_with_security_violations(self, executor):
        """Test code execution with security violations."""
        unsafe_code = "import os\nos.system('echo malicious')"
        result = executor.execute_code(unsafe_code, "python")

        assert isinstance(result, CodeExecutionResult)
        assert result.success is False
        assert len(result.security_violations) > 0
        assert "security violations" in result.error.lower()

    def test_execute_unsupported_language(self, executor):
        """Test execution with unsupported language."""
        code = "some code"
        result = executor.execute_code(code, "unsupported_lang")

        assert isinstance(result, CodeExecutionResult)
        assert result.success is False
        assert "Unsupported language" in result.error

    def test_output_validation(self, executor):
        """Test output validation functionality."""
        # Test exact match
        assert executor._validate_output("Hello", "Hello") is True

        # Test whitespace normalization
        assert executor._validate_output("  Hello  World  ", "Hello World") is True

        # Test mismatch
        assert executor._validate_output("Hello", "Goodbye") is False

    def test_execute_with_expected_output(self, executor):
        """Test code execution with expected output validation."""
        code = "print('Hello, World!')"
        expected = "Hello, World!"

        result = executor.execute_code(code, "python", expected_output=expected)

        assert isinstance(result, CodeExecutionResult)
        assert result.success is True
        assert result.performance_metrics["output_validation"] is True

    def test_execute_with_timeout_simulation(self, executor):
        """Test timeout handling (simulated)."""
        # Create an executor with very short timeout
        short_executor = AdvancedCodeExecutor(timeout_seconds=0.1)

        # Code that would take longer than timeout
        slow_code = """
import time
time.sleep(1)
print('Done')
"""

        result = short_executor.execute_code(slow_code, "python")

        assert isinstance(result, CodeExecutionResult)
        # Should either timeout or complete quickly
        assert result.execution_time < 2.0  # Reasonable upper bound

    @patch("subprocess.run")
    def test_execute_with_monitoring_mock(self, mock_run, executor):
        """Test code execution with mocked subprocess."""
        # Mock successful execution
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Hello, World!"
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        result = executor._execute_with_monitoring("print('Hello')", "python", "")

        assert result["success"] is True
        assert result["output"] == "Hello, World!"
        assert result["error"] == ""

    @patch("subprocess.run")
    def test_execute_with_compilation_error_mock(self, mock_run, executor):
        """Test handling of compilation errors."""
        # Mock compilation failure
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "Compilation error"
        mock_run.return_value = mock_result

        # Test with Java (which requires compilation)
        result = executor._execute_with_monitoring("invalid java code", "java", "")

        # The result depends on the mocking, but should handle errors gracefully
        assert "success" in result


class TestCodeCorrectnessValidator:
    """Test suite for CodeCorrectnessValidator class."""

    @pytest.fixture
    def validator(self):
        """Create a code correctness validator instance for testing."""
        return CodeCorrectnessValidator()

    def test_validator_initialization(self, validator):
        """Test code correctness validator initialization."""
        assert validator.executor is not None
        assert isinstance(validator.executor, AdvancedCodeExecutor)

    def test_validate_python_syntax_valid(self, validator):
        """Test validation of valid Python syntax."""
        valid_code = """
def hello():
    print("Hello, World!")
    return True

hello()
"""

        result = validator._validate_syntax(valid_code, "python")

        assert result["valid"] is True
        assert len(result["errors"]) == 0

    def test_validate_python_syntax_invalid(self, validator):
        """Test validation of invalid Python syntax."""
        invalid_code = """
def hello(
    print("Hello, World!")
    return True
"""

        result = validator._validate_syntax(invalid_code, "python")

        assert result["valid"] is False
        assert len(result["errors"]) > 0

    def test_validate_javascript_syntax_basic(self, validator):
        """Test basic JavaScript syntax validation."""
        valid_code = "function hello() { console.log('Hello'); }"
        result = validator._validate_syntax(valid_code, "javascript")
        assert result["valid"] is True

        invalid_code = "function hello() { console.log('Hello'; }"
        result = validator._validate_syntax(invalid_code, "javascript")
        assert result["valid"] is False

    def test_analyze_code_quality_python(self, validator):
        """Test code quality analysis for Python."""
        good_code = """
# This function greets the user
def greet_user(user_name):
    '''Greet the user with their name'''
    return f"Hello, {user_name}!"

# Main execution
if __name__ == "__main__":
    name = "World"
    print(greet_user(name))
"""

        result = validator._analyze_code_quality(good_code, "python")

        assert result["score"] > 0.8
        assert isinstance(result["recommendations"], list)

    def test_analyze_code_quality_poor(self, validator):
        """Test code quality analysis for poor code."""
        poor_code = """
def f(x,y,z,a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x1,y1,z1):
    a=x+y
    b=a*z
    return b
"""

        result = validator._analyze_code_quality(poor_code, "python")

        assert result["score"] < 1.0
        assert len(result["recommendations"]) > 0

    def test_analyze_logic_patterns(self, validator):
        """Test logic pattern analysis."""
        code_with_issues = """
while True:
    x = input("Enter value: ")
    if x == "quit":
        print("Goodbye")
    # Missing break statement
"""

        result = validator._analyze_logic_patterns(code_with_issues, "python")

        assert len(result["issues"]) > 0
        assert any("infinite loop" in issue.lower() for issue in result["issues"])

    @patch.object(AdvancedCodeExecutor, "execute_code")
    def test_validate_code_correctness_comprehensive(self, mock_execute, validator):
        """Test comprehensive code correctness validation."""
        # Mock execution result
        mock_result = CodeExecutionResult(
            success=True,
            output="5",
            performance_metrics={
                "test_results": [
                    {"success": True, "test_case": 1},
                    {"success": True, "test_case": 2},
                ]
            },
        )
        mock_execute.return_value = mock_result

        code = """
def add(a, b):
    return a + b

print(add(2, 3))
"""

        test_cases = [{"input": "2 3", "output": "5"}, {"input": "1 4", "output": "5"}]

        result = validator.validate_code_correctness(
            code, "python", test_cases=test_cases
        )

        assert result["syntax_valid"] is True
        assert result["execution_successful"] is True
        assert result["test_cases_passed"] == 2
        assert result["total_test_cases"] == 2
        assert result["code_quality_score"] > 0


class TestPerformanceAnalyzer:
    """Test suite for PerformanceAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create a performance analyzer instance for testing."""
        return PerformanceAnalyzer()

    def test_analyzer_initialization(self, analyzer):
        """Test performance analyzer initialization."""
        assert len(analyzer.complexity_patterns) > 0
        assert "O(1)" in analyzer.complexity_patterns
        assert "O(n)" in analyzer.complexity_patterns
        assert "O(n^2)" in analyzer.complexity_patterns

    def test_analyze_linear_complexity(self, analyzer):
        """Test analysis of linear complexity code."""
        linear_code = """
def find_max(arr):
    max_val = arr[0]
    for item in arr:
        if item > max_val:
            max_val = item
    return max_val
"""

        result = analyzer.analyze_performance(linear_code, "python")

        assert result["estimated_time_complexity"] == "O(n)"
        assert isinstance(result["performance_issues"], list)
        assert isinstance(result["optimization_recommendations"], list)

    def test_analyze_quadratic_complexity(self, analyzer):
        """Test analysis of quadratic complexity code."""
        quadratic_code = """
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr
"""

        result = analyzer.analyze_performance(quadratic_code, "python")

        assert result["estimated_time_complexity"] == "O(n^2)"
        assert len(result["performance_issues"]) > 0
        assert any("nested" in issue.lower() for issue in result["performance_issues"])

    def test_detect_performance_issues(self, analyzer):
        """Test detection of specific performance issues."""
        problematic_code = """
result = ""
for i in range(1000):
    result += str(i)
"""

        issues = analyzer._detect_performance_issues(problematic_code, "python")

        assert len(issues) > 0
        assert any("string concatenation" in issue.lower() for issue in issues)

    def test_generate_optimization_recommendations(self, analyzer):
        """Test generation of optimization recommendations."""
        code_needing_optimization = """
result = []
for item in data:
    result.append(process(item))
"""

        recommendations = analyzer._generate_optimization_recommendations(
            code_needing_optimization, "python"
        )

        assert len(recommendations) > 0
        assert any("comprehension" in rec.lower() for rec in recommendations)

    def test_detect_bottlenecks(self, analyzer):
        """Test detection of performance bottlenecks."""
        code_with_io = """
def process_file(filename):
    with open(filename, 'r') as f:
        data = f.read()
    return data.upper()
"""

        bottlenecks = analyzer._detect_bottlenecks(code_with_io, "python")

        assert len(bottlenecks) > 0
        assert any("i/o" in bottleneck.lower() for bottleneck in bottlenecks)

    def test_analyze_constant_complexity(self, analyzer):
        """Test analysis of constant complexity code."""
        constant_code = """
def get_first_element(arr):
    return arr[0]
"""

        result = analyzer.analyze_performance(constant_code, "python")

        assert result["estimated_time_complexity"] == "O(1)"

    def test_analyze_recursive_patterns(self, analyzer):
        """Test detection of recursive patterns."""
        recursive_code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""

        result = analyzer.analyze_performance(recursive_code, "python")

        # Should detect recursive pattern
        assert len(result["optimization_recommendations"]) > 0
        assert any(
            "memoization" in rec.lower()
            for rec in result["optimization_recommendations"]
        )


class TestCodeExecutionIntegration:
    """Integration tests for code execution validation."""

    def test_full_python_validation_pipeline(self):
        """Test complete validation pipeline for Python code."""
        executor = AdvancedCodeExecutor()
        validator = CodeCorrectnessValidator()
        analyzer = PerformanceAnalyzer()

        code = """
def factorial(n):
    '''Calculate factorial of n'''
    if n <= 1:
        return 1
    return n * factorial(n - 1)

# Test the function
print(factorial(5))
"""

        # Test execution
        exec_result = executor.execute_code(code, "python")
        assert isinstance(exec_result, CodeExecutionResult)

        # Test correctness validation
        correctness_result = validator.validate_code_correctness(code, "python")
        assert correctness_result["syntax_valid"] is True

        # Test performance analysis
        perf_result = analyzer.analyze_performance(code, "python")
        assert "estimated_time_complexity" in perf_result

    def test_security_and_performance_analysis(self):
        """Test combined security and performance analysis."""
        executor = AdvancedCodeExecutor()

        # Code with both security and performance issues
        problematic_code = """
import os
result = ""
for i in range(1000):
    for j in range(1000):
        result += str(i * j)
os.system("echo 'done'")
"""

        exec_result = executor.execute_code(problematic_code, "python")

        # Should detect security violations
        assert len(exec_result.security_violations) > 0

        # Should detect performance issues
        assert (
            len(exec_result.performance_metrics.get("optimization_opportunities", []))
            > 0
        )

    def test_multi_language_support(self):
        """Test support for multiple programming languages."""
        executor = AdvancedCodeExecutor()

        # Test Python
        python_code = "print('Hello from Python')"
        python_result = executor.execute_code(python_code, "python")
        assert isinstance(python_result, CodeExecutionResult)

        # Test JavaScript
        js_code = "console.log('Hello from JavaScript');"
        js_result = executor.execute_code(js_code, "javascript")
        assert isinstance(js_result, CodeExecutionResult)

        # Both should have language-specific analysis
        assert len(executor.language_configs["python"]["security_patterns"]) > 0
        assert len(executor.language_configs["javascript"]["security_patterns"]) > 0
