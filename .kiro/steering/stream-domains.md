---
inclusion: fileMatch
fileMatchPattern: '**/domain_*.py'
---

# STREAM Domain Implementation Guidelines

## Domain-Specific Requirements

### Science Domain

- Implement physics, chemistry, and biology validators
- Create simulation-based validation for physical phenomena
- Ensure chemical equation balancing and verification
- Validate biological system models and processes

### Technology Domain

- Support programming challenges with automated code execution
- Implement system design problem generation and validation
- Create digital literacy assessments with objective criteria
- Validate technology concepts against current standards

### Reading Domain

- Generate comprehension questions with answer validation
- Create literary analysis prompts with rubric-based assessment
- Develop critical thinking exercises with reasoning validation
- Ensure cultural and linguistic sensitivity in content

### Engineering Domain

- Create design challenges with constraint satisfaction verification
- Implement optimization problems with solution validation
- Generate real-world engineering scenarios with practical assessment
- Validate engineering solutions against safety and efficiency standards

### Arts Domain

- Develop creative prompts with subjective assessment criteria
- Create aesthetic analysis questions with rubric-based validation
- Generate interdisciplinary connections with cultural context
- Ensure cultural sensitivity and diversity in content

### Mathematics Domain

- Extend existing math validation with advanced CAS integration
- Support multi-step problem solving with step validation
- Implement visual mathematics problems with diagram generation
- Create proof-based problems with logical verification

## Domain Validation Implementation

```python
class DomainValidator:
    def __init__(self, domain_config):
        self.domain = domain_config["domain"]
        self.validation_rules = domain_config["validation_rules"]
        self.quality_thresholds = domain_config["quality_thresholds"]
    
    def validate_content(self, content):
        # Apply domain-specific validation rules
        validation_results = {}
        for rule_name, rule_func in self.validation_rules.items():
            validation_results[rule_name] = rule_func(content)
        
        # Calculate overall quality score
        quality_score = self.calculate_quality_score(validation_results)
        
        return ValidationResult(
            is_valid=quality_score >= self.quality_thresholds["minimum_score"],
            quality_score=quality_score,
            validation_details=validation_results
        )
```

## Domain-Specific Reasoning Traces

Each domain should implement appropriate reasoning trace generation:

- **Mathematics**: Logical proof steps, algebraic manipulations
- **Science**: Hypothesis formation, experimental design, evidence evaluation
- **Technology**: Algorithm explanation, system design rationale
- **Engineering**: Design constraints, optimization criteria
- **Arts**: Creative process explanation, aesthetic analysis
- **Reading**: Comprehension strategies, literary analysis
