# DSPy Integration for SynThesisAI

This module provides DSPy (Declarative Self-improving Python) integration for the SynThesisAI platform, enabling automated prompt optimization and self-improving content generation pipelines.

## Overview

DSPy integration replaces manual prompt engineering with automated optimization using MIPROv2 optimizer and ChainOfThought modules, reducing development time by 50-70% while maintaining backward compatibility with existing systems.

## Key Components

- **STREAMContentGenerator**: Base DSPy module for content generation across STREAM domains
- **DSPyOptimizationEngine**: Optimization engine using MIPROv2 for automated prompt improvement
- **OptimizationCache**: Caching system for optimized DSPy modules
- **SignatureManager**: Management of domain-specific DSPy signatures
- **DSPyConfig**: Configuration management for DSPy integration

## Usage

### Basic Usage

```python
from core.dspy import STREAMContentGenerator

# Create a domain-specific content generator
math_generator = STREAMContentGenerator("mathematics")

# Generate content
result = math_generator(
    mathematical_concept="quadratic_equations",
    difficulty_level="high_school",
    learning_objectives=["solve quadratic equations", "understand discriminant"]
)

# Access generated content
problem = result.problem_statement
solution = result.solution
reasoning = result.reasoning_trace
```

### Optimization

```python
from core.dspy import DSPyOptimizationEngine

# Create optimization engine
optimizer = DSPyOptimizationEngine()

# Optimize a domain module
optimized_module = optimizer.optimize_for_domain(
    math_generator,
    quality_requirements={"min_accuracy": 0.9}
)

# Use optimized module
result = optimized_module(
    mathematical_concept="calculus_integration",
    difficulty_level="undergraduate",
    learning_objectives=["compute definite integrals"]
)
```

### Custom Signatures

```python
from core.dspy.signatures import create_custom_signature, SignatureManager

# Create custom signature
signature = create_custom_signature(
    inputs=["concept", "difficulty", "objectives"],
    outputs=["problem", "solution", "explanation"]
)

# Register with signature manager
manager = SignatureManager()
manager.register_custom_signature(
    domain="custom_domain",
    signature_type="generation",
    signature=signature,
    version="1.0.0"
)
```

## Dependencies

- DSPy: `dspy-ai>=2.5.0`
- Optuna: `optuna>=3.6.1`

## Installation

Install dependencies using `uv`:

```bash
uv add "dspy-ai>=2.5.0" "optuna>=3.6.1"
```