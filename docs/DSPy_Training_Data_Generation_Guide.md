# DSPy Training Data Generation Guide

## Overview

This guide provides comprehensive solutions for generating training data for DSPy optimization when you don't have existing datasets. The SynThesisAI system can bootstrap its own training data from successful problem generations, enabling DSPy optimization without requiring external data sources.

## 🎯 Solutions for DSPy Training Data

### The Challenge

DSPy optimization requires training examples to improve prompt engineering and content generation. However, many systems don't have pre-existing training datasets. This guide shows you how to create high-quality training data from your existing problem generation pipeline.

### The Solution

Your SynThesisAI system is already generating exactly the kind of high-quality examples that DSPy needs! Every successful problem generation can become a training example.

## 🚀 Immediate Solutions (Quick Start)

### 1. Bootstrap from Your Current System

Your system is already generating high-quality problems. You can collect these as training data:

```bash
# Run your current system to collect training data
uv run python core/cli/interface.py \
  --config config/settings.yaml \
  --batch-id training_bootstrap \
  --num-problems 50 \
  --engineer-provider gemini \
  --engineer-model gemini-2.5-pro \
  --checker-provider openai \
  --checker-model o3-mini \
  --target-provider openai \
  --target-model o1
```

**Key Insight**: Problems where the target model fails are perfect training examples because they represent challenging, high-quality problems.

### 2. Use Existing High-Quality Examples

From any successful run, you already have training examples! Look for:

- Problems where target model failed (indicates good difficulty)
- High checker scores (indicates quality)
- Diverse topics and difficulty levels

### 3. Synthetic Data Generation

Generate training data using different strategies:

- **Vary difficulty levels**: easy, medium, hard
- **Cover different topics**: algebra, calculus, geometry, statistics
- **Use different seed prompts**: optimization, word problems, proofs
- **Vary problem types**: computational, conceptual, applied

## 🛠️ Implementation Approaches

### Option A: Quick Manual Collection (Immediate - 30 minutes)

Create a simple script to collect training data from your existing results:

```python
import json
from pathlib import Path
from datetime import datetime

def collect_training_data_from_results(results_dir: str):
    """Collect training examples from existing results."""
    
    training_examples = []
    results_path = Path(results_dir)
    
    # Read valid_prompts.json from your runs
    valid_prompts_file = results_path / "valid_prompts.json"
    if valid_prompts_file.exists():
        with open(valid_prompts_file) as f:
            valid_prompts = json.load(f)
            
        for prompt in valid_prompts:
            training_example = {
                "timestamp": datetime.now().isoformat(),
                "inputs": {
                    "subject": prompt.get("subject"),
                    "topic": prompt.get("topic"),
                    "difficulty_level": prompt.get("difficulty_level", "medium"),
                    "seed_prompt": prompt.get("seed_prompt", "")
                },
                "outputs": {
                    "problem": prompt.get("problem"),
                    "answer": prompt.get("answer"),
                    "hints": prompt.get("hints", {}),
                    "solution": prompt.get("solution", "")
                },
                "quality_metrics": {
                    "quality_score": 0.9,  # These are already validated
                    "checker_score": prompt.get("checker_score", 0.8),
                    "target_model_failed": True  # Assume true for valid prompts
                },
                "domain": "mathematics"
            }
            training_examples.append(training_example)
    
    return training_examples

# Usage example
examples = collect_training_data_from_results("./results/dspy_test")
print(f"Collected {len(examples)} training examples")

# Save to training data file
training_file = Path(".cache/training_data/bootstrap_examples.json")
training_file.parent.mkdir(parents=True, exist_ok=True)

with open(training_file, "w") as f:
    json.dump(examples, f, indent=2)
```

### Option B: Automated Collection System (Recommended - 1-2 hours)

Add automatic training data collection to your pipeline:

```python
def collect_training_example(generation_result):
    """Collect successful generations as training data."""
    
    # Only collect high-quality examples where target model failed
    if (generation_result.get("target_model_failed", False) and 
        generation_result.get("quality_score", 0) > 0.7):
        
        training_example = {
            "timestamp": datetime.now().isoformat(),
            "inputs": {
                "subject": generation_result["subject"],
                "topic": generation_result["topic"],
                "difficulty_level": generation_result.get("difficulty_level", "medium"),
                "seed_prompt": generation_result.get("seed_prompt", "")
            },
            "outputs": {
                "problem": generation_result["problem"],
                "answer": generation_result["answer"],
                "hints": generation_result["hints"],
                "solution": generation_result.get("solution", "")
            },
            "quality_metrics": {
                "quality_score": generation_result["quality_score"],
                "checker_score": generation_result.get("checker_score", 0.0),
                "target_model_failed": generation_result["target_model_failed"]
            },
            "domain": classify_domain(generation_result["subject"])
        }
        
        # Append to training data file (JSONL format for streaming)
        training_file = Path(".cache/training_data/examples.jsonl")
        training_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(training_file, "a") as f:
            f.write(json.dumps(training_example) + "\n")
        
        return training_example
    
    return None

def classify_domain(subject: str) -> str:
    """Classify subject into STREAM domain."""
    math_subjects = ["algebra", "calculus", "geometry", "statistics", "number theory", 
                     "combinatorics", "probability", "linear algebra", "differential equations"]
    
    if any(math_sub in subject.lower() for math_sub in math_subjects):
        return "mathematics"
    
    return "mathematics"  # Default for now
```

### Option C: Bootstrap Dataset Generation (Comprehensive - 2-4 hours)

Generate a comprehensive training dataset by running multiple targeted batches:

```bash
#!/bin/bash
# bootstrap_training_data.sh

echo "🚀 Bootstrapping DSPy training data..."

# Define topics and difficulty levels
topics=("algebra" "calculus" "geometry" "statistics" "number_theory" "combinatorics")
difficulties=("easy" "medium" "hard")

# Generate training data for each topic and difficulty
for topic in "${topics[@]}"; do
    for difficulty in "${difficulties[@]}"; do
        echo "📚 Generating ${difficulty} ${topic} problems..."
        
        uv run python core/cli/interface.py \
            --config config/settings.yaml \
            --batch-id "bootstrap_${topic}_${difficulty}" \
            --num-problems 10 \
            --engineer-provider gemini \
            --engineer-model gemini-2.5-pro \
            --checker-provider openai \
            --checker-model o3-mini \
            --target-provider openai \
            --target-model o1
            
        sleep 5  # Brief pause between batches
    done
done

echo "✅ Bootstrap training data generation complete!"
echo "📊 Check results in ./results/ directories"
```

### Option D: External Data Import (Advanced)

Import training data from external sources:

```python
def import_external_training_data(source_file: str, format_type: str = "json"):
    """Import training data from external sources."""
    
    if format_type == "json":
        with open(source_file) as f:
            external_data = json.load(f)
    elif format_type == "csv":
        import pandas as pd
        df = pd.read_csv(source_file)
        external_data = df.to_dict('records')
    else:
        raise ValueError(f"Unsupported format: {format_type}")
    
    training_examples = []
    
    for item in external_data:
        # Convert external format to internal format
        training_example = {
            "timestamp": datetime.now().isoformat(),
            "inputs": {
                "subject": item.get("subject", "mathematics"),
                "topic": item.get("topic", "general"),
                "difficulty_level": item.get("difficulty", "medium"),
                "seed_prompt": item.get("context", "")
            },
            "outputs": {
                "problem": item["problem"],
                "answer": item["answer"],
                "hints": item.get("hints", {}),
                "solution": item.get("solution", "")
            },
            "quality_metrics": {
                "quality_score": item.get("quality_score", 0.8),
                "checker_score": 0.8,  # Default for imported data
                "target_model_failed": True  # Assume good quality
            },
            "domain": "mathematics",
            "source": "external_import"
        }
        
        training_examples.append(training_example)
    
    return training_examples

# Example usage
# external_examples = import_external_training_data("math_problems.json")
```

## 📊 Data Requirements and Guidelines

### Minimum Data Requirements

Based on DSPy best practices:

- **Minimum**: 10-20 examples per domain
- **Recommended**: 50-100 examples per domain  
- **Optimal**: 200+ examples per domain

### Quality Guidelines

**High-Quality Training Examples Should Have:**

1. **Clear Problem Statements**: Well-defined, unambiguous problems
2. **Correct Answers**: Verified solutions with proper mathematical notation
3. **Helpful Hints**: Progressive hints that guide toward the solution
4. **Appropriate Difficulty**: Challenging enough that target models sometimes fail
5. **Educational Value**: Problems that teach important concepts

**Quality Metrics to Track:**

- **Quality Score**: Overall assessment (aim for >0.7)
- **Checker Score**: Validation accuracy (aim for >0.8)
- **Target Model Failure Rate**: Indicates good difficulty (aim for 20-40%)
- **Topic Diversity**: Coverage across different mathematical areas
- **Difficulty Distribution**: Balance of easy, medium, and hard problems

### Data Diversity Requirements

**Topic Coverage:**

- Algebra (linear equations, polynomials, systems)
- Calculus (derivatives, integrals, limits)
- Geometry (proofs, coordinate geometry, trigonometry)
- Statistics (probability, distributions, hypothesis testing)
- Number Theory (primes, modular arithmetic, divisibility)
- Combinatorics (counting, permutations, combinations)

**Difficulty Levels:**

- **Easy**: Basic concept application (20-30% of dataset)
- **Medium**: Multi-step problems requiring reasoning (40-50% of dataset)
- **Hard**: Complex problems requiring advanced techniques (20-30% of dataset)

**Problem Types:**

- Computational problems (direct calculation)
- Conceptual problems (understanding and explanation)
- Proof-based problems (logical reasoning)
- Applied problems (real-world applications)

## 🔧 DSPy Integration Steps

### Step 1: Collect Training Data (30 minutes - 2 hours)

Choose one of the approaches above and collect 20-50 training examples.

### Step 2: Convert to DSPy Format (15 minutes)

```python
def convert_to_dspy_format(training_examples):
    """Convert internal training examples to DSPy format."""
    
    dspy_examples = []
    
    for example in training_examples:
        # Create DSPy example (simplified - actual implementation would use DSPy classes)
        dspy_example = {
            'subject': example['inputs']['subject'],
            'topic': example['inputs']['topic'],
            'difficulty_level': example['inputs']['difficulty_level'],
            'seed_prompt': example['inputs']['seed_prompt'],
            'problem': example['outputs']['problem'],
            'answer': example['outputs']['answer'],
            'hints': example['outputs']['hints'],
            'solution': example['outputs']['solution']
        }
        
        dspy_examples.append(dspy_example)
    
    return dspy_examples

# Usage
training_examples = collect_training_data_from_results("./results/bootstrap_algebra")
dspy_examples = convert_to_dspy_format(training_examples)

# Save DSPy training data
with open(".cache/dspy/training_data.json", "w") as f:
    json.dump(dspy_examples, f, indent=2)
```

### Step 3: Update DSPy Configuration (5 minutes)

Update your DSPy configuration to use the training data:

```json
{
  "enabled": true,
  "cache_dir": ".cache/dspy",
  "training_data": {
    "min_examples": 10,
    "max_examples": 100,
    "validation_split": 0.2,
    "data_path": ".cache/dspy/training_data.json"
  },
  "optimization": {
    "mipro_v2": {
      "optuna_trials_num": 20,
      "max_bootstrapped_demos": 2,
      "max_labeled_demos": 8,
      "init_temperature": 1.0
    }
  }
}
```

### Step 4: Enable DSPy (2 minutes)

```yaml
# config/settings.yaml
dspy_enabled: true
dspy_config_path: "config/dspy_config.json"
dspy_fallback_on_error: true
```

### Step 5: Test DSPy Optimization (5-10 minutes)

```bash
# Test with DSPy enabled
uv run python core/cli/interface.py \
  --config config/settings.yaml \
  --batch-id dspy_test \
  --num-problems 5 \
  --engineer-provider gemini \
  --engineer-model gemini-2.5-pro \
  --checker-provider openai \
  --checker-model o3-mini \
  --target-provider openai \
  --target-model o1
```

## 🎯 Practical Implementation Timeline

### Phase 1: Quick Start (1 hour)

1. **Collect existing examples** (30 min): Use Option A to gather current results
2. **Convert to DSPy format** (15 min): Transform examples for DSPy use
3. **Enable DSPy** (15 min): Update configuration and test

### Phase 2: Automated Collection (2-3 hours)

1. **Implement collection hooks** (1 hour): Add automatic collection to pipeline
2. **Generate bootstrap dataset** (1-2 hours): Run targeted generation batches
3. **Validate and optimize** (30 min): Check quality and tune parameters

### Phase 3: Advanced Features (4-8 hours)

1. **Build comprehensive system** (4-6 hours): Implement full training data management
2. **Add monitoring and analytics** (1-2 hours): Track performance and quality
3. **Create management interfaces** (1-2 hours): Build tools for data curation

## 💡 Pro Tips and Best Practices

### Quality Over Quantity

- **20 high-quality examples** are better than 100 poor ones
- Focus on problems that challenge the target model
- Ensure mathematical correctness and clear explanations

### Diversity Matters

- Cover different mathematical topics and concepts
- Include various difficulty levels and problem types
- Use different prompting strategies and contexts

### Incremental Approach

- Start with a small, high-quality dataset
- Gradually expand as you identify gaps
- Monitor DSPy performance improvements over time

### Monitor and Iterate

- Track which examples contribute most to DSPy performance
- Remove or improve low-value training examples
- Continuously add new examples from successful generations

### Automation is Key

- Set up automatic collection from your pipeline
- Use scripts to generate diverse training datasets
- Implement quality checks and filtering

## 🔍 Troubleshooting Common Issues

### Issue: Not Enough Training Data

**Solution**: Run bootstrap generation script with more topics and difficulties

### Issue: Poor DSPy Performance

**Solutions**:

- Check training data quality scores
- Ensure diversity across topics and difficulties
- Remove low-quality examples
- Add more challenging problems

### Issue: DSPy Optimization Fails

**Solutions**:

- Verify DSPy example format is correct
- Check that training data meets minimum requirements
- Use fallback to legacy agents while building more data

### Issue: Training Data Collection Not Working

**Solutions**:

- Verify collection hooks are properly integrated
- Check file permissions and storage paths
- Monitor logs for collection errors

## 📚 Additional Resources

### Related Documentation

- [DSPy Documentation](https://dspy-docs.vercel.app/)
- [SynThesisAI User Workflow Guide](docs/USER_WORKFLOW_GUIDE.md)
- [DSPy Integration Architecture Spec](.kiro/specs/dspy-integration-architecture/)

### Training Data Generation Spec

For a comprehensive implementation plan, see:

- [Requirements](.kiro/specs/dspy-training-data-generation/requirements.md)
- [Design](.kiro/specs/dspy-training-data-generation/design.md)
- [Implementation Tasks](.kiro/specs/dspy-training-data-generation/tasks.md)

### Example Datasets

Look for inspiration in:

- Mathematical competition problems (AMC, AIME, USAMO)
- Educational problem databases
- Open-source mathematics repositories
- Academic course materials (with proper licensing)

## 🎉 Conclusion

You don't need external training data to enable DSPy optimization! Your SynThesisAI system is already generating exactly the kind of high-quality examples that DSPy needs. By implementing automatic collection and bootstrap generation, you can quickly build a comprehensive training dataset that enables powerful DSPy optimization.

**Key Takeaway**: Every successful problem generation is a potential training example. Start collecting today, and you'll have DSPy optimization running within hours!

---

**Next Steps**: Choose an implementation approach above and start building your training dataset. Begin with the Quick Start approach for immediate results, then expand to automated collection for long-term success.
