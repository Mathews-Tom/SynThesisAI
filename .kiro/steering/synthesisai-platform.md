---
inclusion: always
---

# SynThesisAI Platform Development Guidelines

## Overview

SynThesisAI is a comprehensive, self-optimizing, multi-domain AI platform that leverages DSPy's declarative programming paradigm and agentic reinforcement learning to generate high-quality educational content across Science, Technology, Reading, Engineering, Arts, and Mathematics (STREAM) domains.

## Key Architecture Components

- **Domain Classification Layer**: Routes requests to appropriate STREAM domain modules
- **DSPy Optimization Layer**: Automates prompt engineering using MIPROv2 optimizer
- **Multi-Agent RL Coordination**: Coordinates Generator, Validator, and Curriculum agents
- **Quality Assurance Framework**: Validates content across multiple dimensions
- **Reasoning Trace Generation**: Provides educational transparency and pedagogical insights

## Performance Targets

- **50-70% reduction** in development time through automated prompt optimization
- **200-400% increase** in throughput through parallel processing
- **60-80% reduction** in operational costs through intelligent resource management
- **>95% accuracy** in generated content validation
- **<3% false positive rate** in quality assessment

## Development Standards

### Code Structure

- Use Python 3.9+ with type hints
- Follow PEP 8 style guidelines
- Implement comprehensive docstrings for all classes and methods
- Use dataclasses for data models
- Implement proper exception handling and logging

### Testing Requirements

- Write unit tests for all components
- Implement integration tests for end-to-end workflows
- Create performance benchmarks to validate improvement claims
- Test all error handling and recovery mechanisms

### DSPy Integration

- Use DSPy's ChainOfThought for structured reasoning
- Implement MIPROv2 optimizer for automated prompt engineering
- Create domain-specific signatures for each STREAM field
- Cache optimization results to avoid redundant computations

### Multi-Agent RL Implementation

- Use reinforcement learning for agent policy optimization
- Implement proper reward functions based on quality metrics
- Create effective coordination mechanisms to avoid conflicts
- Design consensus protocols for multi-agent decision making

## Resources

- DSPy Documentation: https://dspy-docs.vercel.app/
- Reinforcement Learning from Human Feedback: https://arxiv.org/abs/2009.01325
- Multi-Agent Coordination: https://arxiv.org/abs/2103.01955
