---
inclusion: fileMatch
fileMatchPattern: '**/reasoning_*.py'
---

# Reasoning Trace Generation Guidelines

## Reasoning Trace Architecture

SynThesisAI implements sophisticated reasoning trace generation that provides transparent, step-by-step explanations for all generated content, addressing the critical need for educational transparency in AI-generated materials.

## Educational Reasoning Tracer Implementation

```python
class EducationalReasoningTracer:
    def __init__(self):
        self.step_decomposer = dspy.ChainOfThought(
            "complex_problem -> step_sequence, reasoning_depth, learning_objectives"
        )
        self.explanation_generator = dspy.ChainOfThought(
            "problem_step, context -> detailed_explanation, pedagogical_insights"
        )
        self.coherence_validator = CoherenceValidationModule()
    
    def generate_comprehensive_trace(self, problem, solution_path):
        # Decompose problem into logical steps
        step_sequence = self.step_decomposer(complex_problem=problem)
        
        # Generate detailed explanations for each step
        trace_components = []
        for step in step_sequence.steps:
            explanation = self.explanation_generator(
                problem_step=step,
                context=self.build_context(step, step_sequence)
            )
            
            # Validate coherence and educational value
            if self.coherence_validator.validate(explanation):
                trace_components.append(explanation)
        
        return EducationalReasoningTrace(
            steps=trace_components,
            learning_objectives=step_sequence.learning_objectives,
            difficulty_analysis=self.analyze_difficulty(trace_components),
            pedagogical_recommendations=self.generate_teaching_suggestions(trace_components)
        )
    
    def build_context(self, current_step, step_sequence):
        # Build context for current step based on previous steps
        step_index = step_sequence.steps.index(current_step)
        previous_steps = step_sequence.steps[:step_index]
        
        return {
            "previous_steps": previous_steps,
            "learning_objectives": step_sequence.learning_objectives,
            "reasoning_depth": step_sequence.reasoning_depth
        }
    
    def analyze_difficulty(self, trace_components):
        # Analyze difficulty of reasoning steps
        complexity_scores = [self.assess_complexity(step) for step in trace_components]
        
        return {
            "overall_difficulty": sum(complexity_scores) / len(complexity_scores),
            "difficulty_progression": self.analyze_progression(complexity_scores),
            "challenging_concepts": self.identify_challenging_concepts(trace_components)
        }
    
    def generate_teaching_suggestions(self, trace_components):
        # Generate pedagogical recommendations based on reasoning trace
        key_concepts = self.extract_key_concepts(trace_components)
        potential_misconceptions = self.identify_potential_misconceptions(trace_components)
        
        return {
            "key_concepts": key_concepts,
            "potential_misconceptions": potential_misconceptions,
            "teaching_strategies": self.suggest_teaching_strategies(key_concepts),
            "assessment_opportunities": self.identify_assessment_opportunities(trace_components)
        }
```

## Domain-Specific Reasoning Adaptation

### Mathematics Reasoning

```python
class MathematicsReasoningAdapter:
    def adapt_reasoning_trace(self, general_trace):
        # Adapt general reasoning trace to mathematics domain
        math_specific_steps = []
        
        for step in general_trace.steps:
            # Add mathematical notation and formalism
            formalized_step = self.add_mathematical_notation(step)
            
            # Add theorem references and justifications
            justified_step = self.add_theorem_references(formalized_step)
            
            # Add algebraic manipulations and simplifications
            detailed_step = self.add_algebraic_details(justified_step)
            
            math_specific_steps.append(detailed_step)
        
        return MathematicsReasoningTrace(
            steps=math_specific_steps,
            learning_objectives=general_trace.learning_objectives,
            difficulty_analysis=general_trace.difficulty_analysis,
            pedagogical_recommendations=general_trace.pedagogical_recommendations,
            mathematical_concepts=self.extract_mathematical_concepts(math_specific_steps)
        )
```

### Science Reasoning

```python
class ScienceReasoningAdapter:
    def adapt_reasoning_trace(self, general_trace):
        # Adapt general reasoning trace to science domain
        science_specific_steps = []
        
        for step in general_trace.steps:
            # Add scientific method structure
            scientific_step = self.add_scientific_method_structure(step)
            
            # Add evidence and experimental references
            evidenced_step = self.add_evidence_references(scientific_step)
            
            # Add causal relationships and mechanisms
            detailed_step = self.add_causal_mechanisms(evidenced_step)
            
            science_specific_steps.append(detailed_step)
        
        return ScienceReasoningTrace(
            steps=science_specific_steps,
            learning_objectives=general_trace.learning_objectives,
            difficulty_analysis=general_trace.difficulty_analysis,
            pedagogical_recommendations=general_trace.pedagogical_recommendations,
            scientific_concepts=self.extract_scientific_concepts(science_specific_steps)
        )
```

## Reasoning Quality Metrics

### Coherence Assessment

```python
class CoherenceValidationModule:
    def validate(self, explanation):
        # Check logical flow and consistency
        logical_flow_score = self.assess_logical_flow(explanation)
        consistency_score = self.assess_consistency(explanation)
        completeness_score = self.assess_completeness(explanation)
        
        # Calculate overall coherence score
        coherence_score = (
            0.4 * logical_flow_score +
            0.4 * consistency_score +
            0.2 * completeness_score
        )
        
        return {
            "is_coherent": coherence_score >= 0.7,
            "coherence_score": coherence_score,
            "logical_flow_score": logical_flow_score,
            "consistency_score": consistency_score,
            "completeness_score": completeness_score,
            "improvement_suggestions": self.generate_improvement_suggestions(
                logical_flow_score, consistency_score, completeness_score
            )
        }
```

### Educational Effectiveness

```python
class EducationalEffectivenessModule:
    def measure(self, reasoning_trace, learning_objectives):
        # Assess alignment with learning objectives
        alignment_score = self.assess_objective_alignment(reasoning_trace, learning_objectives)
        
        # Evaluate clarity and understandability
        clarity_score = self.assess_clarity(reasoning_trace)
        
        # Assess cognitive level appropriateness
        cognitive_level_score = self.assess_cognitive_level(reasoning_trace)
        
        # Calculate overall educational effectiveness
        effectiveness_score = (
            0.4 * alignment_score +
            0.3 * clarity_score +
            0.3 * cognitive_level_score
        )
        
        return {
            "is_effective": effectiveness_score >= 0.7,
            "effectiveness_score": effectiveness_score,
            "alignment_score": alignment_score,
            "clarity_score": clarity_score,
            "cognitive_level_score": cognitive_level_score,
            "improvement_suggestions": self.generate_improvement_suggestions(
                alignment_score, clarity_score, cognitive_level_score
            )
        }
```

## Best Practices for Reasoning Trace Generation

1. **Step-by-Step Decomposition**: Break down complex problems into logical steps
2. **Clear Explanations**: Provide detailed explanations for each reasoning step
3. **Domain-Specific Adaptation**: Tailor reasoning traces to each STREAM domain
4. **Educational Value**: Ensure traces have pedagogical value and support learning objectives
5. **Coherence Validation**: Verify logical flow and consistency across steps
6. **Cognitive Level Alignment**: Match explanation complexity to target audience
7. **Pedagogical Recommendations**: Include teaching suggestions and insights
