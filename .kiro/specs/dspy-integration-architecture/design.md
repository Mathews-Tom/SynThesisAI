# DSPy Integration Architecture - Design Document

## Overview

This design document outlines the integration of DSPy (Declarative Self-improving Python) framework into the SynThesisAI platform. The integration will replace manual prompt engineering with automated optimization using MIPROv2 optimizer and ChainOfThought modules, enabling self-improving content generation pipelines that reduce development time by 50-70% while maintaining backward compatibility with existing systems.

## Architecture

### High-Level DSPy Integration Architecture

The DSPy integration follows a layered approach that wraps existing agent functionality while providing new optimization capabilities:

1. **DSPy Module Layer**: Converts existing agents to DSPy modules with ChainOfThought reasoning
2. **Optimization Engine Layer**: Implements MIPROv2 optimizer for automated prompt improvement
3. **Signature Management Layer**: Manages domain-specific input/output signatures
4. **Caching Layer**: Stores and retrieves optimized prompts and modules
5. **Integration Layer**: Provides seamless integration with existing architecture

### DSPy Module Architecture

```python
# Base DSPy Module for STREAM Content Generation
class STREAMContentGenerator(dspy.Module):
    def __init__(self, domain: str, signature: str):
        super().__init__()
        self.domain = domain
        self.generate = dspy.ChainOfThought(signature)
        self.refine = dspy.ChainOfThought(
            "content, feedback, quality_metrics -> refined_content, improvements, confidence_score"
        )
        
    def forward(self, **inputs):
        # Generate initial content using optimized prompts
        draft_content = self.generate(**inputs)
        
        # Apply domain-specific refinements if needed
        if self.needs_refinement(draft_content):
            refined_content = self.refine(
                content=draft_content,
                feedback=self.get_domain_feedback(draft_content),
                quality_metrics=self.calculate_quality_metrics(draft_content)
            )
            return refined_content
        
        return draft_content
```

### Domain-Specific Signatures

Each STREAM domain will have specialized signatures:

```python
# Mathematics Domain Signature
MATH_SIGNATURE = "mathematical_concept, difficulty_level, learning_objectives -> problem_statement, solution, proof, reasoning_trace, pedagogical_hints"

# Science Domain Signature  
SCIENCE_SIGNATURE = "scientific_concept, difficulty_level, learning_objectives -> problem_statement, solution, experimental_design, evidence_evaluation, reasoning_trace"

# Technology Domain Signature
TECH_SIGNATURE = "technical_concept, difficulty_level, learning_objectives -> problem_statement, solution, algorithm_explanation, system_design, reasoning_trace"

# Reading Domain Signature
READING_SIGNATURE = "literary_concept, difficulty_level, learning_objectives -> comprehension_question, analysis_prompt, critical_thinking_exercise, reasoning_trace"

# Engineering Domain Signature
ENGINEERING_SIGNATURE = "engineering_concept, difficulty_level, learning_objectives -> design_challenge, optimization_problem, constraint_analysis, reasoning_trace"

# Arts Domain Signature
ARTS_SIGNATURE = "artistic_concept, difficulty_level, learning_objectives -> creative_prompt, aesthetic_analysis, cultural_context, reasoning_trace"
```

## Components and Interfaces

### DSPy Optimization Engine

```python
class DSPyOptimizationEngine:
    def __init__(self):
        self.optimizers = {
            'mipro_v2': MIPROv2Optimizer(),
            'bootstrap': BootstrapFewShotOptimizer(),
            'copro': COPROOptimizer()
        }
        self.cache = OptimizationCache()
        self.training_data_manager = TrainingDataManager()
        
    def optimize_for_domain(self, domain_module: STREAMContentGenerator, 
                           quality_requirements: Dict[str, Any]) -> STREAMContentGenerator:
        """Optimize a domain module using MIPROv2"""
        cache_key = self.generate_cache_key(domain_module, quality_requirements)
        
        # Check cache first
        if cached_result := self.cache.get(cache_key):
            return cached_result
        
        # Get training and validation data
        trainset = self.training_data_manager.get_training_data(domain_module.domain)
        valset = self.training_data_manager.get_validation_data(domain_module.domain)
        
        # Optimize using MIPROv2
        optimizer = self.optimizers['mipro_v2']
        optimized_module = optimizer.compile(
            student=domain_module,
            trainset=trainset,
            valset=valset,
            optuna_trials_num=100,
            max_bootstrapped_demos=4,
            max_labeled_demos=16
        )
        
        # Cache the result
        self.cache.store(cache_key, optimized_module)
        
        return optimized_module
```

### Agent Conversion Architecture

```python
class DSPyEngineerAgent(Agent):
    """DSPy-powered Engineer Agent"""
    
    def __init__(self):
        super().__init__("engineer", "engineer_model")
        self.dspy_module = None
        self.optimization_engine = DSPyOptimizationEngine()
        
    def initialize_dspy_module(self, domain: str):
        """Initialize DSPy module for specific domain"""
        signature = self.get_domain_signature(domain)
        self.dspy_module = STREAMContentGenerator(domain, signature)
        
        # Optimize the module
        self.dspy_module = self.optimization_engine.optimize_for_domain(
            self.dspy_module, 
            self.get_quality_requirements()
        )
    
    def generate(self, subject: str, topic: str, **kwargs) -> Dict[str, Any]:
        """Generate content using DSPy module"""
        if not self.dspy_module or self.dspy_module.domain != subject.lower():
            self.initialize_dspy_module(subject.lower())
        
        # Use DSPy module for generation
        result = self.dspy_module(
            domain=subject,
            topic=topic,
            difficulty_level=kwargs.get('difficulty_level'),
            learning_objectives=kwargs.get('learning_objectives', [])
        )
        
        # Convert DSPy result to expected format
        return self.convert_dspy_result(result, subject, topic, **kwargs)
```

### Caching Architecture

```python
class OptimizationCache:
    def __init__(self, cache_dir: str = ".cache/dspy"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.memory_cache = {}
        self.cache_ttl = 3600 * 24  # 24 hours
        
    def generate_cache_key(self, domain_module: STREAMContentGenerator, 
                          quality_requirements: Dict[str, Any]) -> str:
        """Generate unique cache key for optimization"""
        key_components = [
            domain_module.domain,
            domain_module.generate.signature,
            json.dumps(quality_requirements, sort_keys=True),
            self.get_dspy_version()
        ]
        return hashlib.md5("|".join(key_components).encode()).hexdigest()
    
    def store(self, cache_key: str, optimized_module: STREAMContentGenerator):
        """Store optimized module in cache"""
        cache_path = self.cache_dir / f"{cache_key}.pkl"
        
        # Store in memory cache
        self.memory_cache[cache_key] = {
            'module': optimized_module,
            'timestamp': time.time()
        }
        
        # Store in persistent cache
        with open(cache_path, 'wb') as f:
            pickle.dump({
                'module': optimized_module,
                'timestamp': time.time(),
                'metadata': {
                    'domain': optimized_module.domain,
                    'signature': optimized_module.generate.signature
                }
            }, f)
    
    def get(self, cache_key: str) -> Optional[STREAMContentGenerator]:
        """Retrieve optimized module from cache"""
        # Check memory cache first
        if cache_key in self.memory_cache:
            cached_item = self.memory_cache[cache_key]
            if time.time() - cached_item['timestamp'] < self.cache_ttl:
                return cached_item['module']
        
        # Check persistent cache
        cache_path = self.cache_dir / f"{cache_key}.pkl"
        if cache_path.exists():
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
                if time.time() - cached_data['timestamp'] < self.cache_ttl:
                    # Update memory cache
                    self.memory_cache[cache_key] = {
                        'module': cached_data['module'],
                        'timestamp': cached_data['timestamp']
                    }
                    return cached_data['module']
        
        return None
```

## Data Models

### DSPy Module Configuration

```python
@dataclass
class DSPyModuleConfig:
    domain: str
    signature: str
    optimization_params: Dict[str, Any]
    quality_requirements: Dict[str, Any]
    training_data_path: str
    validation_data_path: str
```

### Optimization Result

```python
@dataclass
class OptimizationResult:
    optimized_module: STREAMContentGenerator
    optimization_metrics: Dict[str, float]
    training_time: float
    validation_score: float
    cache_key: str
    timestamp: datetime
```

### Training Data Structure

```python
@dataclass
class TrainingExample:
    inputs: Dict[str, Any]
    expected_outputs: Dict[str, Any]
    quality_score: float
    domain: str
    metadata: Dict[str, Any]
```

## Integration Strategy

### Backward Compatibility

The DSPy integration maintains backward compatibility through adapter patterns:

```python
class DSPyAgentAdapter:
    """Adapter to maintain backward compatibility"""
    
    def __init__(self, legacy_agent: Agent, dspy_agent: DSPyEngineerAgent):
        self.legacy_agent = legacy_agent
        self.dspy_agent = dspy_agent
        self.use_dspy = True
        
    def generate(self, *args, **kwargs):
        """Route to DSPy or legacy implementation"""
        try:
            if self.use_dspy:
                return self.dspy_agent.generate(*args, **kwargs)
        except Exception as e:
            logger.warning(f"DSPy generation failed, falling back to legacy: {e}")
            self.use_dspy = False
            
        return self.legacy_agent.generate(*args, **kwargs)
```

### Gradual Migration Strategy

1. **Phase 1**: Implement DSPy modules alongside existing agents
2. **Phase 2**: Add optimization engine and caching
3. **Phase 3**: Enable DSPy by default with legacy fallback
4. **Phase 4**: Remove legacy implementations after validation

## Error Handling

### DSPy-Specific Error Handling

```python
class DSPyIntegrationError(Exception):
    """Base exception for DSPy integration issues"""
    pass

class OptimizationFailureError(DSPyIntegrationError):
    """Raised when DSPy optimization fails"""
    pass

class SignatureValidationError(DSPyIntegrationError):
    """Raised when DSPy signature validation fails"""
    pass

class CacheCorruptionError(DSPyIntegrationError):
    """Raised when optimization cache is corrupted"""
    pass
```

### Fallback Mechanisms

1. **Optimization Failure**: Fall back to non-optimized DSPy modules
2. **DSPy Module Failure**: Fall back to legacy agent implementations
3. **Cache Corruption**: Regenerate optimizations from scratch
4. **Signature Errors**: Use default signatures with logging

## Testing Strategy

### DSPy Integration Testing

1. **Unit Tests**: Test individual DSPy modules and optimization components
2. **Integration Tests**: Test DSPy integration with existing architecture
3. **Performance Tests**: Validate 50-70% development time reduction claims
4. **Optimization Tests**: Verify MIPROv2 optimizer effectiveness
5. **Caching Tests**: Validate cache performance and correctness
6. **Fallback Tests**: Ensure graceful degradation when DSPy fails

### Test Data Management

```python
class DSPyTestDataManager:
    def __init__(self):
        self.test_datasets = {}
        
    def create_test_dataset(self, domain: str, size: int = 100):
        """Create test dataset for DSPy optimization"""
        return [
            TrainingExample(
                inputs=self.generate_test_inputs(domain),
                expected_outputs=self.generate_expected_outputs(domain),
                quality_score=random.uniform(0.7, 1.0),
                domain=domain,
                metadata={}
            )
            for _ in range(size)
        ]
```

This design provides a comprehensive foundation for integrating DSPy into the SynThesisAI platform while maintaining system stability and enabling significant performance improvements through automated prompt optimization.
