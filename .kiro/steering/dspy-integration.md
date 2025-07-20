---
inclusion: fileMatch
fileMatchPattern: '**/dspy_*.py'
---

# DSPy Integration Guidelines

## DSPy Overview

DSPy is a framework for programming foundation models using techniques like Chain-of-Thought prompting, few-shot learning, and self-improvement through feedback. It enables declarative programming of LLMs, allowing the system to automatically optimize prompts and improve performance over time.

## Key DSPy Components for SynThesisAI

### ChainOfThought Module

Use ChainOfThought for structured reasoning in content generation:

```python
class STREAMContentGenerator(dspy.Module):
    def __init__(self, domain):
        self.domain = domain
        self.generate = dspy.ChainOfThought(
            "domain, topic, difficulty_level, learning_objectives -> "
            "content, solution, reasoning_trace, pedagogical_hints"
        )
        self.refine = dspy.ChainOfThought(
            "content, feedback, quality_metrics -> "
            "refined_content, improvements, confidence_score"
        )
```

### MIPROv2 Optimizer

Implement MIPROv2 for automated prompt optimization:

```python
class DSPyOptimizationEngine:
    def __init__(self):
        self.optimizers = {
            'mipro_v2': MIPROv2Optimizer(),
            'bootstrap': BootstrapFewShotOptimizer()
        }
        self.cache = OptimizationCache()
    
    def optimize_for_domain(self, domain_module, quality_requirements):
        # Auto-optimize prompts using MIPROv2
        optimizer = self.optimizers['mipro_v2']
        optimized_module = optimizer.compile(
            student=domain_module,
            trainset=self.get_training_data(domain_module),
            valset=self.get_validation_data(domain_module),
            optuna_trials_num=100
        )
        
        return optimized_module
```

### Signature Optimization

Create domain-specific signatures for each STREAM field:

```python
# Science domain signature
science_signature = dspy.Signature(
    inputs=["scientific_concept", "difficulty_level", "learning_objectives"],
    outputs=["problem_statement", "solution", "experimental_design", "reasoning_trace"]
)

# Mathematics domain signature
math_signature = dspy.Signature(
    inputs=["mathematical_concept", "difficulty_level", "learning_objectives"],
    outputs=["problem_statement", "solution", "proof", "reasoning_trace"]
)
```

## DSPy Integration Best Practices

1. **Modular Design**: Create separate DSPy modules for different aspects of content generation
2. **Caching Optimization Results**: Store and reuse optimization results to avoid redundant computations
3. **Feedback Integration**: Use validation results to improve DSPy modules over time
4. **Signature Design**: Create clear input/output signatures for each module
5. **Training Data Management**: Maintain high-quality training and validation datasets for each domain

## DSPy Resources

- DSPy GitHub Repository: https://github.com/stanfordnlp/dspy
- DSPy Documentation: https://dspy-docs.vercel.app/
- MIPROv2 Paper: https://arxiv.org/abs/2306.03781
