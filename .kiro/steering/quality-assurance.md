---
inclusion: fileMatch
fileMatchPattern: '**/quality_*.py'
---

# Quality Assurance Framework Guidelines

## Quality Assurance Architecture

SynThesisAI implements a comprehensive quality assurance framework with four primary dimensions:

1. **Fidelity**: Accuracy and correctness of content
2. **Utility**: Educational value and usefulness
3. **Safety**: Ethical guidelines and safety standards
4. **Pedagogical Value**: Learning objective alignment and educational effectiveness

## Universal Quality Assurance Implementation

```python
class UniversalQualityAssurance:
    def __init__(self):
        self.fidelity_checker = FidelityAssessmentModule()
        self.utility_evaluator = UtilityEvaluationModule()
        self.safety_validator = SafetyValidationModule()
        self.pedagogical_scorer = PedagogicalScoringModule()
    
    async def comprehensive_validation(self, content, domain, target_audience):
        # Run all validation checks in parallel
        fidelity_task = asyncio.create_task(
            self.fidelity_checker.assess(content, domain)
        )
        utility_task = asyncio.create_task(
            self.utility_evaluator.evaluate(content, target_audience)
        )
        safety_task = asyncio.create_task(
            self.safety_validator.check(content)
        )
        pedagogical_task = asyncio.create_task(
            self.pedagogical_scorer.score(content)
        )
        
        # Gather results
        results = {
            'fidelity': await fidelity_task,
            'utility': await utility_task,
            'safety': await safety_task,
            'pedagogical': await pedagogical_task
        }
        
        # Aggregate quality score
        quality_score = self.aggregate_quality_score(results)
        approved = self.is_approved(quality_score, results)
        
        return QualityResult(
            approved=approved,
            metrics=quality_score,
            validation_details=results,
            feedback=self.generate_feedback(results) if not approved else None
        )
    
    def aggregate_quality_score(self, results):
        # Calculate weighted average of all quality dimensions
        weights = {
            'fidelity': 0.4,
            'utility': 0.3,
            'safety': 0.2,
            'pedagogical': 0.1
        }
        
        overall_score = sum(
            results[dimension]['score'] * weights[dimension]
            for dimension in results
        )
        
        return {
            'fidelity_score': results['fidelity']['score'],
            'utility_score': results['utility']['score'],
            'safety_score': results['safety']['score'],
            'pedagogical_score': results['pedagogical']['score'],
            'overall_score': overall_score
        }
    
    def is_approved(self, quality_score, results):
        # Check if content meets all quality thresholds
        if quality_score['overall_score'] < 0.7:
            return False
        
        if results['safety']['score'] < 0.9:  # Safety is critical
            return False
        
        if results['fidelity']['score'] < 0.8:  # Accuracy is important
            return False
        
        return True
    
    def generate_feedback(self, results):
        # Create detailed feedback for improvement
        feedback = []
        
        if results['fidelity']['score'] < 0.8:
            feedback.append({
                'dimension': 'fidelity',
                'issue': 'Content accuracy needs improvement',
                'suggestions': results['fidelity']['improvement_suggestions']
            })
        
        if results['utility']['score'] < 0.7:
            feedback.append({
                'dimension': 'utility',
                'issue': 'Educational value needs enhancement',
                'suggestions': results['utility']['improvement_suggestions']
            })
        
        if results['safety']['score'] < 0.9:
            feedback.append({
                'dimension': 'safety',
                'issue': 'Content has safety or ethical concerns',
                'suggestions': results['safety']['improvement_suggestions']
            })
        
        if results['pedagogical']['score'] < 0.7:
            feedback.append({
                'dimension': 'pedagogical',
                'issue': 'Learning objective alignment needs improvement',
                'suggestions': results['pedagogical']['improvement_suggestions']
            })
        
        return feedback
```

## Quality Dimension Modules

### Fidelity Assessment

```python
class FidelityAssessmentModule:
    def __init__(self):
        self.domain_validators = self._initialize_validators()
    
    def _initialize_validators(self):
        return {
            'mathematics': MathematicsValidator(),
            'science': ScienceValidator(),
            'technology': TechnologyValidator(),
            'reading': ReadingValidator(),
            'engineering': EngineeringValidator(),
            'arts': ArtsValidator()
        }
    
    async def assess(self, content, domain):
        # Get domain-specific validator
        validator = self.domain_validators.get(domain)
        if not validator:
            raise ValueError(f"No validator available for domain: {domain}")
        
        # Perform domain-specific validation
        validation_result = await validator.validate(content)
        
        # Calculate fidelity score
        fidelity_score = self.calculate_fidelity_score(validation_result)
        
        return {
            'score': fidelity_score,
            'validation_details': validation_result,
            'improvement_suggestions': self.generate_improvement_suggestions(validation_result)
        }
```

### Safety Validation

```python
class SafetyValidationModule:
    def __init__(self):
        self.safety_checks = [
            EthicalGuidelinesCheck(),
            BiasDetectionCheck(),
            SensitiveContentCheck(),
            AgeAppropriatenessCheck()
        ]
    
    async def check(self, content):
        # Run all safety checks
        check_results = []
        for check in self.safety_checks:
            result = await check.run(content)
            check_results.append(result)
        
        # Calculate safety score
        safety_score = self.calculate_safety_score(check_results)
        
        return {
            'score': safety_score,
            'check_results': check_results,
            'improvement_suggestions': self.generate_improvement_suggestions(check_results)
        }
```

## Quality Metrics

### Content Accuracy Metrics

- **Mathematical Correctness**: Validation through Computer Algebra Systems
- **Scientific Accuracy**: Verification against established scientific principles
- **Technological Correctness**: Validation of code, algorithms, and technical concepts
- **Factual Accuracy**: Verification of facts and information

### Educational Value Metrics

- **Learning Objective Alignment**: Alignment with specified learning goals
- **Cognitive Level Appropriateness**: Match to target audience cognitive abilities
- **Engagement Potential**: Assessment of interest and motivation factors
- **Knowledge Transfer Effectiveness**: Evaluation of explanation clarity and effectiveness

### Safety and Ethics Metrics

- **Bias Detection**: Identification of cultural, gender, or other biases
- **Age Appropriateness**: Assessment of content suitability for target age group
- **Ethical Alignment**: Compliance with ethical guidelines and standards
- **Cultural Sensitivity**: Evaluation of cultural appropriateness and inclusivity

## Quality Assurance Best Practices

1. **Multi-Dimensional Assessment**: Evaluate content across all quality dimensions
2. **Domain-Specific Validation**: Apply appropriate validation rules for each STREAM domain
3. **Automated and Human Validation**: Combine automated checks with human review
4. **Continuous Improvement**: Use validation results to improve generation over time
5. **Transparent Reporting**: Provide detailed quality metrics and validation results
6. **Threshold-Based Approval**: Establish clear quality thresholds for content approval
