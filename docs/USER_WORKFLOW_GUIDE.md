# SynThesisAI User Workflow Guide

This guide provides comprehensive instructions for running the SynThesisAI system, including DSPy optimization and end-to-end testing.

## 🚀 Quick Start Commands

### 1. Interactive Mode (Recommended for First-Time Users)

```bash
uv run python core/cli/run_interactive.py
```

This will guide you through the configuration process step by step.

### 2. Command Line Interface with Default Configuration

```bash
uv run python core/cli/interface.py --config config/settings.yaml
```

### 3. Command Line with Custom Parameters

```bash
uv run python core/cli/interface.py \
  --config config/settings.yaml \
  --batch-id my_custom_batch \
  --num-problems 10 \
  --engineer-provider gemini \
  --engineer-model gemini-2.5-pro \
  --checker-provider openai \
  --checker-model o3-mini \
  --target-provider openai \
  --target-model o1
```

### 4. Web API Service

```bash
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## 🧠 DSPy Optimization

### Enable DSPy Optimization

DSPy optimization is controlled by the `config/dspy_config.json` file. Ensure `"enabled": true`.

### Run with Automatic DSPy Optimization

```bash
# DSPy optimization happens automatically when enabled
uv run python core/cli/interface.py --config config/settings.yaml
```

### Manual DSPy Optimization Workflow

```python
from core.dspy.optimization_workflows import get_workflow_manager

# Start the workflow manager
workflow_manager = get_workflow_manager()
workflow_manager.start()

# Optimize a single domain
job_id = workflow_manager.optimize_domain(
    domain="mathematics",
    quality_requirements={
        "min_accuracy": 0.9,
        "min_coherence": 0.8,
        "min_relevance": 0.9
    }
)

# Check optimization status
status = workflow_manager.get_status(job_id)
print(f"Optimization status: {status}")

# Generate optimization report
report = workflow_manager.generate_report(time_range_hours=24)
print(f"Optimization report: {report}")
```

### Batch DSPy Optimization

```python
# Optimize multiple domains at once
batch_id = workflow_manager.optimize_domains_batch(
    domains=["mathematics", "science", "technology"],
    quality_requirements={
        "min_accuracy": 0.85,
        "min_coherence": 0.8,
        "min_relevance": 0.85
    }
)

# Monitor batch progress
batch_status = workflow_manager.get_status(batch_id)
```

## 🧪 End-to-End Testing

### Run Complete User Workflow Tests

#### Basic Workflow Test

```bash
uv run python scripts/run_user_workflow_test.py --test-type basic
```

#### DSPy Optimization Workflow Test

```bash
uv run python scripts/run_user_workflow_test.py --test-type dspy
```

#### Validation Integration Test

```bash
uv run python scripts/run_user_workflow_test.py --test-type validation
```

#### Performance Monitoring Test

```bash
uv run python scripts/run_user_workflow_test.py --test-type performance
```

#### Error Handling Test

```bash
uv run python scripts/run_user_workflow_test.py --test-type error
```

#### Full System Integration Test

```bash
uv run python scripts/run_user_workflow_test.py --test-type full --verbose
```

### Run All End-to-End Tests

```bash
uv run pytest tests/end_to_end/test_synthesisai_user_workflow.py -v
```

## 📊 System Monitoring and Analysis

### Check System Health

```python
from core.dspy.optimization_workflows import get_workflow_manager

workflow_manager = get_workflow_manager()
workflow_manager.start()

# Get queue status
queue_status = workflow_manager.scheduler.get_queue_status()
print(f"Queue status: {queue_status}")

# Generate performance report
report = workflow_manager.generate_report(time_range_hours=24)
print(f"Performance report: {report}")
```

### Validation System Integration

```python
from core.validation import UniversalValidator

# Initialize validator
validator = UniversalValidator()

# Validate content
content = {
    'problem': 'What is 2 + 2?',
    'answer': '4',
    'explanation': 'Adding 2 and 2 gives 4'
}

result = await validator.validate_content(content, domain='mathematics')
print(f"Validation result: {result}")
```

## ⚙️ Configuration

### Main Configuration (`config/settings.yaml`)

```yaml
num_problems: 10
max_workers: 10
taxonomy: "taxonomy/enhanced_math_taxonomy.json"
output_dir: "./results"
default_batch_id: "batch_01"

engineer_model:
  provider: "gemini"
  model_name: "gemini-2.5-pro"

checker_model:
  provider: "openai"
  model_name: "o3-mini"

target_model:
  provider: "openai"
  model_name: "o1"
```

### DSPy Configuration (`config/dspy_config.json`)

```json
{
  "enabled": true,
  "cache_dir": ".cache/dspy",
  "optimization": {
    "mipro_v2": {
      "optuna_trials_num": 100,
      "max_bootstrapped_demos": 4,
      "max_labeled_demos": 16,
      "num_candidate_programs": 16,
      "init_temperature": 1.4
    }
  },
  "quality_requirements": {
    "min_accuracy": 0.8,
    "min_coherence": 0.7,
    "min_relevance": 0.8
  }
}
```

### MARL Configuration (`config/marl_config.yaml`)

Multi-Agent Reinforcement Learning settings for advanced coordination.

## 🔧 Advanced Usage

### Custom Taxonomy

```bash
uv run python core/cli/interface.py \
  --config config/settings.yaml \
  --taxonomy-path custom_taxonomy.json
```

### Performance Optimization

```bash
# Disable caching for fresh results
uv run python core/cli/interface.py \
  --config config/settings.yaml \
  --disable-cache

# Use legacy processing
uv run python core/cli/interface.py \
  --config config/settings.yaml \
  --legacy-processing

# Disable pre-filtering
uv run python core/cli/interface.py \
  --config config/settings.yaml \
  --disable-prefiltering
```

### Benchmark Mode

```bash
# Use benchmark seed data
uv run python core/cli/run_interactive.py
# Then select "y" for benchmark seed and specify benchmark name (e.g., "AIME")
```

## 📈 Output and Results

### Generated Files

- `valid_prompts.json` - Successfully generated problems
- `rejected_prompts.json` - Problems that didn't meet quality standards
- `generation_summary.json` - Summary statistics and metadata
- `cost_tracking.json` - API usage and cost information

### Validation Reports

- `validation_report.json` - STREAM domain validation results
- `quality_metrics.json` - Quality assessment details

### Performance Reports

- `performance_report.json` - System performance metrics
- `optimization_history.json` - DSPy optimization results

## 🐛 Troubleshooting

### Common Issues

1. **DSPy Import Error**

   ```bash
   pip install dspy-ai
   ```

2. **Configuration File Not Found**

   ```bash
   # Ensure config file exists
   ls -la config/settings.yaml
   ```

3. **Taxonomy Loading Error**

   ```bash
   # Validate JSON syntax
   python -m json.tool taxonomy/enhanced_math_taxonomy.json
   ```

4. **Memory Issues**

   ```bash
   # Reduce number of problems or workers
   uv run python core/cli/interface.py --num-problems 5 --max-workers 2
   ```

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
uv run python core/cli/interface.py --config config/settings.yaml
```

## 📚 Additional Resources

- [DSPy Documentation](https://dspy-docs.vercel.app/)
- [MARL Architecture Guide](docs/marl_architecture_guide.md)
- [Phase 1 Completion Report](docs/phase1_completion_report.md)
- [Phase 2 Completion Report](docs/phase2_completion_report.md)

## 🤝 Contributing

To contribute to the SynThesisAI system:

1. Run the full test suite:

   ```bash
   uv run python scripts/run_user_workflow_test.py --test-type full
   ```

2. Ensure all validation tests pass:

   ```bash
   uv run pytest tests/unit_tests/ tests/integration_tests/ -v
   ```

3. Test DSPy optimization:

   ```bash
   uv run python scripts/run_user_workflow_test.py --test-type dspy
   ```

## 📞 Support

For issues or questions:

1. Check the troubleshooting section above
2. Run the error handling test: `uv run python scripts/run_user_workflow_test.py --test-type error`
3. Review the comprehensive logs in the output directory
