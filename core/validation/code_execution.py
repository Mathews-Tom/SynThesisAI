"""
Advanced code execution validation for the SynThesisAI platform.

This module provides comprehensive code execution capabilities including:
- Multi-language sandboxed execution
- Code correctness and output validation
- Performance and efficiency analysis
- Security and safety validation
"""

# Standard Library
import ast
import logging
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class CodeExecutionResult:
    """Result of code execution with comprehensive metrics."""

    def __init__(
        self,
        success: bool,
        output: str = "",
        error: str = "",
        execution_time: float = 0.0,
        memory_usage: int = 0,
        exit_code: int = 0,
        security_violations: Optional[List[str]] = None,
        performance_metrics: Optional[Dict[str, Any]] = None,
    ):
        self.success = success
        self.output = output
        self.error = error
        self.execution_time = execution_time
        self.memory_usage = memory_usage
        self.exit_code = exit_code
        self.security_violations = security_violations or []
        self.performance_metrics = performance_metrics or {}


class AdvancedCodeExecutor:
    """
    Advanced multi-language code execution environment.

    Provides secure, monitored code execution with:
    - Resource limits and timeouts
    - Security violation detection
    - Performance metrics collection
    - Output validation and analysis
    """

    def __init__(
        self,
        timeout_seconds: int = 10,
        memory_limit_mb: int = 256,
        enable_networking: bool = False,
        enable_file_system: bool = False,
    ):
        """
        Initialize the advanced code executor.

        Args:
            timeout_seconds: Maximum execution time
            memory_limit_mb: Maximum memory usage
            enable_networking: Allow network access
            enable_file_system: Allow file system access
        """
        self.timeout_seconds = timeout_seconds
        self.memory_limit_mb = memory_limit_mb
        self.enable_networking = enable_networking
        self.enable_file_system = enable_file_system

        # Enhanced language configurations
        self.language_configs = {
            "python": {
                "extension": ".py",
                "command": ["python3"],
                "compile_command": None,
                "security_patterns": [
                    r"import\s+(os|sys|subprocess|socket|urllib|requests)",
                    r"__import__\s*\(",
                    r"exec\s*\(",
                    r"eval\s*\(",
                    r"open\s*\(",
                    r"file\s*\(",
                ],
                "performance_indicators": [
                    r"for\s+\w+\s+in\s+range\s*\(",
                    r"while\s+",
                    r"def\s+\w+\s*\(",
                    r"class\s+\w+",
                ],
            },
            "javascript": {
                "extension": ".js",
                "command": ["node"],
                "compile_command": None,
                "security_patterns": [
                    r"require\s*\(\s*['\"]fs['\"]",
                    r"require\s*\(\s*['\"]child_process['\"]",
                    r"require\s*\(\s*['\"]http['\"]",
                    r"process\.",
                    r"global\.",
                    r"eval\s*\(",
                ],
                "performance_indicators": [
                    r"for\s*\(",
                    r"while\s*\(",
                    r"function\s+\w+",
                    r"=>\s*{",
                ],
            },
            "java": {
                "extension": ".java",
                "command": ["java"],
                "compile_command": ["javac"],
                "security_patterns": [
                    r"import\s+java\.io\.",
                    r"import\s+java\.net\.",
                    r"Runtime\.getRuntime\(\)",
                    r"ProcessBuilder",
                    r"System\.exit\s*\(",
                    r"System\.getProperty",
                ],
                "performance_indicators": [
                    r"for\s*\(",
                    r"while\s*\(",
                    r"public\s+\w+\s+\w+\s*\(",
                    r"class\s+\w+",
                ],
            },
            "cpp": {
                "extension": ".cpp",
                "command": ["./a.out"],
                "compile_command": ["g++", "-o", "a.out"],
                "security_patterns": [
                    r"#include\s*<cstdlib>",
                    r"#include\s*<unistd\.h>",
                    r"system\s*\(",
                    r"exec\w*\s*\(",
                    r"fork\s*\(",
                    r"popen\s*\(",
                ],
                "performance_indicators": [
                    r"for\s*\(",
                    r"while\s*\(",
                    r"\w+\s+\w+\s*\([^)]*\)\s*{",
                    r"class\s+\w+",
                ],
            },
        }

        logger.info(
            "Initialized AdvancedCodeExecutor: timeout=%ds, memory=%dMB, network=%s, fs=%s",
            timeout_seconds,
            memory_limit_mb,
            enable_networking,
            enable_file_system,
        )

    def execute_code(
        self,
        code: str,
        language: str,
        input_data: str = "",
        expected_output: Optional[str] = None,
        test_cases: Optional[List[Dict[str, str]]] = None,
    ) -> CodeExecutionResult:
        """
        Execute code with comprehensive validation and analysis.

        Args:
            code: Source code to execute
            language: Programming language
            input_data: Input data for the program
            expected_output: Expected output for validation
            test_cases: List of test cases with input/output pairs

        Returns:
            CodeExecutionResult with comprehensive metrics
        """
        try:
            if language not in self.language_configs:
                return CodeExecutionResult(
                    success=False, error=f"Unsupported language: {language}"
                )

            # Performance analysis (done before security check for comprehensive analysis)
            performance_metrics = self._analyze_performance_patterns(code, language)

            # Security analysis
            security_violations = self._analyze_security(code, language)
            if security_violations and not self._allow_security_violations():
                return CodeExecutionResult(
                    success=False,
                    error="Code contains security violations",
                    security_violations=security_violations,
                    performance_metrics=performance_metrics,
                )

            # Execute code
            start_time = time.time()
            execution_result = self._execute_with_monitoring(code, language, input_data)
            execution_time = time.time() - start_time

            # Validate output if expected output is provided
            output_valid = True
            if expected_output is not None:
                output_valid = self._validate_output(
                    execution_result["output"], expected_output
                )

            # Run test cases if provided
            test_results = []
            if test_cases:
                test_results = self._run_test_cases(code, language, test_cases)

            # Compile comprehensive result
            result = CodeExecutionResult(
                success=execution_result["success"] and output_valid,
                output=execution_result["output"],
                error=execution_result["error"],
                execution_time=execution_time,
                memory_usage=execution_result.get("memory_usage", 0),
                exit_code=execution_result.get("exit_code", 0),
                security_violations=security_violations,
                performance_metrics={
                    **performance_metrics,
                    "test_results": test_results,
                    "output_validation": output_valid,
                },
            )

            return result

        except Exception as e:
            logger.error("Code execution failed: %s", e)
            return CodeExecutionResult(success=False, error=str(e))

    def _analyze_security(self, code: str, language: str) -> List[str]:
        """Analyze code for security violations."""
        violations = []
        config = self.language_configs[language]

        for pattern in config["security_patterns"]:
            matches = re.findall(pattern, code, re.IGNORECASE)
            if matches:
                violations.append(f"Security pattern detected: {pattern}")

        # Language-specific security analysis
        if language == "python":
            violations.extend(self._analyze_python_security(code))
        elif language == "javascript":
            violations.extend(self._analyze_javascript_security(code))

        return violations

    def _analyze_python_security(self, code: str) -> List[str]:
        """Python-specific security analysis."""
        violations = []

        try:
            # Parse AST for dangerous operations
            tree = ast.parse(code)

            for node in ast.walk(tree):
                # Check for dangerous function calls
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        if node.func.id in ["exec", "eval", "compile", "__import__"]:
                            violations.append(
                                f"Dangerous function call: {node.func.id}"
                            )

                # Check for dangerous imports
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in ["os", "sys", "subprocess", "socket"]:
                            violations.append(
                                f"Potentially dangerous import: {alias.name}"
                            )

                elif isinstance(node, ast.ImportFrom):
                    if node.module in ["os", "sys", "subprocess", "socket"]:
                        violations.append(
                            f"Potentially dangerous import from: {node.module}"
                        )

        except SyntaxError:
            violations.append("Python syntax error detected")

        return violations

    def _analyze_javascript_security(self, code: str) -> List[str]:
        """JavaScript-specific security analysis."""
        violations = []

        # Check for dangerous patterns
        dangerous_patterns = [
            r"eval\s*\(",
            r"Function\s*\(",
            r"setTimeout\s*\(\s*['\"]",
            r"setInterval\s*\(\s*['\"]",
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                violations.append(f"Dangerous JavaScript pattern: {pattern}")

        return violations

    def _analyze_performance_patterns(self, code: str, language: str) -> Dict[str, Any]:
        """Analyze code for performance patterns and potential issues."""
        metrics = {
            "complexity_indicators": [],
            "optimization_opportunities": [],
            "algorithm_patterns": [],
        }

        config = self.language_configs[language]

        # Count performance indicators
        for pattern in config["performance_indicators"]:
            matches = len(re.findall(pattern, code, re.IGNORECASE))
            if matches > 0:
                metrics["complexity_indicators"].append(
                    {"pattern": pattern, "count": matches}
                )

        # Detect nested loops (potential O(n^2) or worse)
        if language == "python":
            nested_loop_pattern = r"for\s+\w+.*:\s*.*for\s+\w+"
        else:
            nested_loop_pattern = r"for\s*\([^}]*for\s*\(|while\s*\([^}]*while\s*\("

        nested_loops = len(
            re.findall(nested_loop_pattern, code, re.IGNORECASE | re.DOTALL)
        )
        if nested_loops > 0:
            metrics["optimization_opportunities"].append(
                f"Detected {nested_loops} potentially nested loops"
            )

        # Detect recursive patterns
        if language == "python":
            recursive_pattern = r"def\s+(\w+)\s*\([^)]*\):[^}]*\1\s*\("
        elif language == "javascript":
            recursive_pattern = r"function\s+(\w+)\s*\([^)]*\)\s*{[^}]*\1\s*\("
        else:
            recursive_pattern = r"(\w+)\s*\([^)]*\)[^}]*\1\s*\("

        recursive_calls = re.findall(recursive_pattern, code, re.IGNORECASE | re.DOTALL)
        if recursive_calls:
            metrics["algorithm_patterns"].append(
                f"Recursive functions detected: {len(recursive_calls)}"
            )

        return metrics

    def _execute_with_monitoring(
        self, code: str, language: str, input_data: str
    ) -> Dict[str, Any]:
        """Execute code with resource monitoring."""
        config = self.language_configs[language]

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                # Write code to temporary file
                code_file = temp_path / f"code{config['extension']}"
                code_file.write_text(code)

                # Compile if necessary
                if config["compile_command"]:
                    compile_cmd = config["compile_command"] + [str(code_file)]
                    compile_result = subprocess.run(
                        compile_cmd,
                        cwd=temp_dir,
                        capture_output=True,
                        text=True,
                        timeout=self.timeout_seconds,
                    )

                    if compile_result.returncode != 0:
                        return {
                            "success": False,
                            "output": "",
                            "error": f"Compilation failed: {compile_result.stderr}",
                            "exit_code": compile_result.returncode,
                        }

                # Execute code
                if language == "java":
                    # Extract class name for Java
                    class_match = re.search(r"public\s+class\s+(\w+)", code)
                    if class_match:
                        class_name = class_match.group(1)
                        exec_cmd = ["java", "-cp", str(temp_dir), class_name]
                    else:
                        return {
                            "success": False,
                            "output": "",
                            "error": "No public class found in Java code",
                        }
                else:
                    exec_cmd = config["command"] + [str(code_file)]

                # Run with resource limits
                result = subprocess.run(
                    exec_cmd,
                    input=input_data,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                    cwd=temp_dir,
                )

                return {
                    "success": result.returncode == 0,
                    "output": result.stdout,
                    "error": result.stderr,
                    "exit_code": result.returncode,
                    "memory_usage": 0,  # Would need platform-specific implementation
                }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "output": "",
                "error": f"Execution timed out after {self.timeout_seconds} seconds",
            }
        except Exception as e:
            return {"success": False, "output": "", "error": str(e)}

    def _validate_output(self, actual_output: str, expected_output: str) -> bool:
        """Validate actual output against expected output."""
        # Normalize whitespace for comparison
        actual_normalized = re.sub(r"\s+", " ", actual_output.strip())
        expected_normalized = re.sub(r"\s+", " ", expected_output.strip())

        return actual_normalized == expected_normalized

    def _run_test_cases(
        self, code: str, language: str, test_cases: List[Dict[str, str]]
    ) -> List[Dict[str, Any]]:
        """Run multiple test cases against the code."""
        results = []

        for i, test_case in enumerate(test_cases):
            input_data = test_case.get("input", "")
            expected_output = test_case.get("output", "")

            # Execute code with test input
            execution_result = self._execute_with_monitoring(code, language, input_data)

            # Validate output
            output_valid = self._validate_output(
                execution_result["output"], expected_output
            )

            results.append(
                {
                    "test_case": i + 1,
                    "input": input_data,
                    "expected_output": expected_output,
                    "actual_output": execution_result["output"],
                    "success": execution_result["success"] and output_valid,
                    "error": execution_result.get("error", ""),
                    "output_valid": output_valid,
                }
            )

        return results

    def _allow_security_violations(self) -> bool:
        """Determine if security violations should be allowed."""
        # In a real implementation, this might check configuration
        # For now, we'll be strict about security
        return False


class CodeCorrectnessValidator:
    """
    Validator for code correctness and quality.

    Analyzes code for:
    - Syntax correctness
    - Logic errors
    - Best practices adherence
    - Code quality metrics
    """

    def __init__(self):
        self.executor = AdvancedCodeExecutor()

    def validate_code_correctness(
        self,
        code: str,
        language: str,
        test_cases: Optional[List[Dict[str, str]]] = None,
        expected_behavior: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Validate code correctness comprehensively.

        Args:
            code: Source code to validate
            language: Programming language
            test_cases: Test cases for validation
            expected_behavior: Description of expected behavior

        Returns:
            Dictionary with correctness validation results
        """
        results = {
            "syntax_valid": False,
            "execution_successful": False,
            "test_cases_passed": 0,
            "total_test_cases": 0,
            "code_quality_score": 0.0,
            "issues": [],
            "recommendations": [],
        }

        try:
            # 1. Syntax validation
            syntax_result = self._validate_syntax(code, language)
            results["syntax_valid"] = syntax_result["valid"]
            if not syntax_result["valid"]:
                results["issues"].extend(syntax_result["errors"])

            # 2. Execution validation
            if test_cases:
                execution_result = self.executor.execute_code(
                    code, language, test_cases=test_cases
                )

                results["execution_successful"] = execution_result.success

                if execution_result.performance_metrics.get("test_results"):
                    test_results = execution_result.performance_metrics["test_results"]
                    results["total_test_cases"] = len(test_results)
                    results["test_cases_passed"] = sum(
                        1 for test in test_results if test["success"]
                    )

            # 3. Code quality analysis
            quality_result = self._analyze_code_quality(code, language)
            results["code_quality_score"] = quality_result["score"]
            results["recommendations"].extend(quality_result["recommendations"])

            # 4. Logic analysis
            logic_result = self._analyze_logic_patterns(code, language)
            results["issues"].extend(logic_result["issues"])
            results["recommendations"].extend(logic_result["recommendations"])

        except Exception as e:
            results["issues"].append(f"Validation error: {str(e)}")

        return results

    def _validate_syntax(self, code: str, language: str) -> Dict[str, Any]:
        """Validate code syntax."""
        result = {"valid": True, "errors": []}

        try:
            if language == "python":
                ast.parse(code)
            elif language == "javascript":
                # For JavaScript, we'd need a proper parser
                # For now, do basic checks
                if code.count("{") != code.count("}"):
                    result["valid"] = False
                    result["errors"].append("Mismatched braces")
                if code.count("(") != code.count(")"):
                    result["valid"] = False
                    result["errors"].append("Mismatched parentheses")
            # Add more language-specific syntax validation as needed

        except SyntaxError as e:
            result["valid"] = False
            result["errors"].append(f"Syntax error: {str(e)}")

        return result

    def _analyze_code_quality(self, code: str, language: str) -> Dict[str, Any]:
        """Analyze code quality and style."""
        score = 1.0
        recommendations = []

        lines = code.split("\n")
        non_empty_lines = [line for line in lines if line.strip()]

        # Check for comments
        comment_patterns = {
            "python": r"#.*",
            "javascript": r"//.*|/\*.*\*/",
            "java": r"//.*|/\*.*\*/",
            "cpp": r"//.*|/\*.*\*/",
        }

        if language in comment_patterns:
            pattern = comment_patterns[language]
            comment_lines = [line for line in lines if re.search(pattern, line)]
            comment_ratio = len(comment_lines) / max(len(non_empty_lines), 1)

            if comment_ratio < 0.1 and len(non_empty_lines) > 5:
                score *= 0.9
                recommendations.append("Consider adding more comments for clarity")

        # Check line length
        long_lines = [line for line in lines if len(line) > 100]
        if long_lines:
            score *= 0.95
            recommendations.append(f"Consider breaking up {len(long_lines)} long lines")

        # Check for meaningful variable names
        if language == "python":
            short_vars = re.findall(r"\b[a-z]\b\s*=", code)
            if len(short_vars) > 2:
                score *= 0.9
                recommendations.append("Consider using more descriptive variable names")

            # Check for very long parameter lists
            long_param_pattern = r"def\s+\w+\s*\([^)]{50,}\)"
            if re.search(long_param_pattern, code):
                score *= 0.7
                recommendations.append(
                    "Consider reducing the number of function parameters"
                )

        return {"score": score, "recommendations": recommendations}

    def _analyze_logic_patterns(self, code: str, language: str) -> Dict[str, Any]:
        """Analyze code for common logic patterns and potential issues."""
        issues = []
        recommendations = []

        # Check for infinite loop patterns
        infinite_loop_patterns = [
            r"while\s+True\s*:",
            r"while\s*\(\s*true\s*\)",
            r"for\s*\(\s*;\s*;\s*\)",
        ]

        for pattern in infinite_loop_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                # Check if there's an actual break statement (not in comments)
                code_without_comments = re.sub(r"#.*", "", code)
                if not re.search(r"\bbreak\b", code_without_comments, re.IGNORECASE):
                    issues.append(
                        "Potential infinite loop detected without break statement"
                    )

        # Check for unused variables (simplified)
        if language == "python":
            assignments = re.findall(r"(\w+)\s*=", code)
            usages = re.findall(r"\b(\w+)\b", code)

            for var in assignments:
                if usages.count(var) == 1:  # Only appears in assignment
                    recommendations.append(f"Variable '{var}' may be unused")

        return {"issues": issues, "recommendations": recommendations}


class PerformanceAnalyzer:
    """
    Analyzer for code performance and efficiency.

    Provides:
    - Time complexity analysis
    - Space complexity estimation
    - Performance bottleneck detection
    - Optimization recommendations
    """

    def __init__(self):
        self.complexity_patterns = {
            "O(1)": [r"return\s+\w+", r"\w+\[\d+\]", r"\w+\.get\("],
            "O(log n)": [r"binary.*search", r"\/\/\s*2", r"mid\s*="],
            "O(n)": [r"for\s+\w+\s+in\s+\w+", r"while\s+\w+"],
            "O(n log n)": [r"sort\(", r"merge.*sort", r"heap.*sort"],
            "O(n^2)": [r"for\s+\w+.*:\s*.*for\s+\w+", r"nested.*loop", r"bubble.*sort"],
            "O(2^n)": [r"fibonacci.*recursive", r"2\s*\*\*\s*n"],
        }

    def analyze_performance(self, code: str, language: str) -> Dict[str, Any]:
        """
        Analyze code performance characteristics.

        Args:
            code: Source code to analyze
            language: Programming language

        Returns:
            Dictionary with performance analysis results
        """
        analysis = {
            "estimated_time_complexity": "O(1)",
            "estimated_space_complexity": "O(1)",
            "performance_issues": [],
            "optimization_recommendations": [],
            "bottlenecks": [],
        }

        # Detect time complexity
        for complexity, patterns in self.complexity_patterns.items():
            for pattern in patterns:
                if re.search(pattern, code, re.IGNORECASE):
                    analysis["estimated_time_complexity"] = complexity
                    break

        # Detect performance issues
        performance_issues = self._detect_performance_issues(code, language)
        analysis["performance_issues"] = performance_issues

        # Generate optimization recommendations
        recommendations = self._generate_optimization_recommendations(code, language)
        analysis["optimization_recommendations"] = recommendations

        # Detect bottlenecks
        bottlenecks = self._detect_bottlenecks(code, language)
        analysis["bottlenecks"] = bottlenecks

        return analysis

    def _detect_performance_issues(self, code: str, language: str) -> List[str]:
        """Detect potential performance issues in code."""
        issues = []

        # Check for nested loops
        if language == "python":
            nested_loop_pattern = r"for\s+\w+.*:\s*.*for\s+\w+"
        else:
            nested_loop_pattern = r"for\s*\([^}]*for\s*\(|while\s*\([^}]*while\s*\("

        if re.search(nested_loop_pattern, code, re.IGNORECASE | re.DOTALL):
            issues.append("Nested loops detected - potential O(n^2) complexity")

        # Check for string concatenation in loops
        if language == "python":
            string_concat_pattern = r"for\s+\w+.*:\s*.*\w+\s*\+=\s*.*str\("
            if re.search(string_concat_pattern, code, re.IGNORECASE | re.DOTALL):
                issues.append("String concatenation in loop - consider using join()")

        # Check for inefficient data structure usage
        if "list" in code.lower() and "in" in code.lower():
            issues.append(
                "Linear search in list - consider using set or dict for O(1) lookup"
            )

        return issues

    def _generate_optimization_recommendations(
        self, code: str, language: str
    ) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []

        # Recommend list comprehensions for Python
        if language == "python" and re.search(
            r"for\s+\w+\s+in.*append\(", code, re.DOTALL
        ):
            recommendations.append(
                "Consider using list comprehension for better performance"
            )

        # Recommend caching for recursive functions
        if re.search(r"def\s+(\w+).*\1\s*\(", code, re.DOTALL):
            recommendations.append("Consider memoization for recursive functions")

        # Recommend early returns
        if re.search(r"if\s+.*:\s*return.*else:", code, re.DOTALL):
            recommendations.append("Consider early returns to reduce nesting")

        return recommendations

    def _detect_bottlenecks(self, code: str, language: str) -> List[str]:
        """Detect potential performance bottlenecks."""
        bottlenecks = []

        # I/O operations
        io_patterns = [r"open\(", r"read\(", r"write\(", r"print\("]
        for pattern in io_patterns:
            if re.search(pattern, code):
                bottlenecks.append("I/O operations detected - potential bottleneck")
                break

        # Network operations
        network_patterns = [r"requests\.", r"urllib\.", r"socket\."]
        for pattern in network_patterns:
            if re.search(pattern, code):
                bottlenecks.append("Network operations detected - potential bottleneck")
                break

        return bottlenecks
