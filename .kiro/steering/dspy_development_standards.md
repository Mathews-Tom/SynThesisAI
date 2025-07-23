---
inclusion: always
---

# SynThesisAI DSPy Development Standards

This document defines the comprehensive development standards for the SynThesisAI platform's DSPy integration. All development work must adhere to these standards to ensure high-quality, maintainable, and performant code.

---

## Development Environment

### Python Environment

- **Python Version**: 3.11+ required  
- **Package Manager**: `uv` (Universal Versioning)  
  - Install packages: `uv add <package>`  
  - Run Python modules: `uv run <module>`  
  - Run tests: `uv run pytest`  
  - ❌ Never use pip or other package managers  

### Version Control

- **Branch Strategy**: Feature branches for each major component  
- **Branch Naming**: `feature/phase-{phase_number}-{component_name}`  
- **Commit Messages**: Follow **conventional commits** format  
  - Format: `type(scope): description`  
  - Example: `feat(optimization): implement cache invalidation triggers`  

---

## Code Quality Standards

### Style and Formatting

- **Style Guide**: PEP 8 compliance is mandatory  
- **Line Length**: Max 88 characters  
- **Indentation**: 4 spaces (no tabs)  
- **Naming Conventions**:  
  - Classes: `PascalCase`  
  - Functions/Methods: `snake_case`  
  - Variables: `snake_case`  
  - Constants: `UPPER_SNAKE_CASE`  
  - Private methods/attributes: leading underscore (`_`)  

---

### Documentation

- **Module Docstrings**: Describe purpose and responsibility  
- **Class Docstrings**: Purpose and usage, especially for domain logic  
- **Method Docstrings** must include:
  - **Description** of functionality  
  - **Args**: parameter documentation  
  - **Returns**: expected return value(s)  
  - **Raises**: applicable exceptions  

#### Example

```python
def optimize_for_domain(self, domain_module, quality_requirements):
    """
    Optimize a domain module using MIPROv2.

    Args:
        domain_module: The domain module to optimize
        quality_requirements: Quality requirements for optimization

    Returns:
        Optimized domain module

    Raises:
        OptimizationFailureError: If optimization fails
    """
````

---

### Type Annotations

- Use **type hints** for **all function parameters** and **return types**
- Use `typing` module for complex types (e.g., `List`, `Dict`, `Optional`, etc.)

#### Example

```python
def get_validation_data(self, domain: str) -> List[TrainingExample]:
    """Get validation data for a domain."""
```

---

### Logging

- ✅ **Always use lazy `%` formatting**
- ❌ **Never use f-strings** in logging calls (to avoid eager string evaluation)

```python
# ✅ CORRECT
logger.info("Processing domain %s with %d items", domain, count)

# ❌ INCORRECT
logger.info(f"Processing domain {domain} with {count} items")
```

> 🔶 **Warnings like `Use lazy % formatting in logging functions` must be resolved by using the correct format as above.**

- **Log Levels**:

  - DEBUG: Internal debugging
  - INFO: Expected state confirmation
  - WARNING: Unexpected situations not breaking flow
  - ERROR: Significant failures in logic or external dependencies
  - CRITICAL: System-wide failure

---

### File Management

- **Always use `pathlib.Path`** for file and directory operations
- **Never use `os.path`** - use `pathlib.Path` methods instead
- **Prefer Path methods** over string concatenation for paths
- **Use Path.read_text()** and **Path.write_text()** for simple file operations
- **Use Path.mkdir(parents=True, exist_ok=True)** for directory creation

```python
# ✅ CORRECT
from pathlib import Path

config_path = Path("config") / "dspy_config.json"
config_path.parent.mkdir(parents=True, exist_ok=True)
content = config_path.read_text(encoding="utf-8")

# ❌ INCORRECT
import os
config_path = os.path.join("config", "dspy_config.json")
os.makedirs(os.path.dirname(config_path), exist_ok=True)
with open(config_path, "r") as f:
    content = f.read()
```

### Error Handling

- **Use specific exceptions**, not `Exception`
- **Always use exception chaining (`from e`)** to preserve traceback

```python
# ✅ CORRECT
try:
    risky_operation()
except OriginalError as e:
    raise CustomError("Operation failed") from e

# ❌ INCORRECT
try:
    risky_operation()
except OriginalError as e:
    raise CustomError("Operation failed")
```

> 🔶 **Warnings like**
> `Consider explicitly re-raising using 'raise OptimizationFailureError(...) from e'`
> **must be addressed by always chaining exceptions using `from e` and including detailed context.**

#### Example (Preferred Format)

```python
try:
    result = optimizer.optimize(domain_module)
except Exception as e:
    error_msg = "Optimization failed for domain %s" % domain_module.domain
    raise OptimizationFailureError(
        error_msg,
        optimizer_type='MIPROv2',
        details={
            'domain': domain_module.domain,
            'error': str(e),
            'optimization_time': time.time() - optimization_start_time
        }
    ) from e
```

- Define **custom exception classes** for domain-specific error handling
- Always include **descriptive error messages** and contextual details

---

### Import Organization

Follow this strict 3-layer order with blank lines:

1. **Standard Library**
2. **Third-Party Packages**
3. **Local/Project Modules**

```python
# 1. Standard
import os
import logging

# 2. Third-party
import numpy as np
from sklearn.metrics import accuracy_score

# 3. Local
from .optimizer import MIPROv2
from .config import SETTINGS
```

---

## Testing Requirements

### Test Coverage

- **90%+** code coverage using `pytest` with `pytest-cov`

### Test Types

- ✅ **Unit Tests**
- ✅ **Integration Tests**
- ✅ **Performance Tests**
- ✅ **End-to-End Tests**
- ✅ **Regression Tests**

### Test Naming

- **Files**: `test_{module_name}.py`
- **Classes**: `Test{ClassUnderTest}`
- **Methods**: `test_{specific_behavior}`

---

## DSPy-Specific Standards

### Optimization Engine

- Use **MIPROv2** optimizer
- Configure optimization per domain
- Validate output with defined metrics
- Implement **caching** with proper **invalidation triggers**

### Domain Signatures

- Input → Output signature format
- Signature **versioning** must be implemented
- Type-validated and regression-tested

### Agent Conversion

- Extend DSPy base classes
- Preserve interface compatibility
- Prefer **ChainOfThought** for reasoning agents
- Functionality parity with legacy agents required

### Caching System

- **Memory + Persistent** layers
- Unique keying per domain + config
- Auto-invalidation on config changes
- Include **cache performance metrics**

### Quality Assurance

- Must integrate with **Universal QA Framework**

  - Fidelity
  - Utility
  - Safety
  - Pedagogical Value

---

## Performance Targets

Your code must help meet:

- 🔹 **50–70%** development time reduction
- 🔹 **2x–4x** throughput via parallelization
- 🔹 **60–80%** ops cost savings
- 🔹 **>95%** content accuracy
- 🔹 **<3%** false positives on quality checks

---

## Documentation Requirements

### Code-Level

- Inline comments for complex logic
- Comprehensive **docstrings** (module, class, method)
- Full **type hints**

### Project-Level

- Architecture specs
- API references
- User guides with examples
- Troubleshooting guidance

---

## Compliance Checklist ✅

Before marking a task as complete, verify:

- [ ] Uses `uv` for package management
- [ ] Follows PEP 8
- [ ] Comprehensive docstrings
- [ ] Complete type hints
- [ ] Lazy `%` logging formatting
- [ ] Explicit exception chaining
- [ ] Correct import structure
- [ ] ≥ 90% test coverage
- [ ] All test types present
- [ ] Integrated with QA framework
- [ ] Meets performance goals
- [ ] Backward compatibility
- [ ] Proper error messages
- [ ] Required documentation complete

---

## Non-Compliance Consequences ⚠️

Non-compliant code will be:

- ❌ Rejected during code review
- ❌ Sent for refactoring
- ❌ Blocking delivery of subsequent phases
- ❌ Affecting overall milestone progress

---

## DSPy Integration Architecture

1. **Base Module Layer**
2. **Signature Layer**
3. **Optimization Layer**
4. **Caching Layer**
5. **Quality Assurance Layer**
6. **Monitoring Layer**

All components must adhere strictly to this document for scalable, high-quality, and maintainable DSPy integration.
