# MARL Configuration Guide

## Overview

This guide provides comprehensive instructions for configuring the Multi-Agent Reinforcement Learning (MARL) coordination system. It covers all configuration parameters, deployment scenarios, and optimization strategies.

## Configuration Structure

The MARL system uses a hierarchical configuration structure with the following main sections:

```yaml
# marl_config.yaml
version: "1.0.0"
name: "production_marl_config"
description: "Production MARL configuration"

agents:
  generator:
    # Generator agent configuration
  validator:
    # Validator agent configuration
  curriculum:
    # Curriculum agent configuration

coordination:
  # Coordination mechanism configuration

learning:
  # Shared learning configuration

system:
  # System-level configuration
```

## Agent Configuration

### Base Agent Parameters

All agents share common base parameters:

```yaml
agents:
  generator:
    agent_id: "generator"
    agent_type: "generator"
    
    # Neural Network Configuration
    network:
      architecture: "feedforward"
      hidden_layers: [256, 128, 64]
      activation_function: "relu"
      dropout_rate: 0.1
      batch_normalization: true
      weight_initialization: "xavier_uniform"
    
    # Optimization Configuration
    optimization:
      optimizer_type: "adam"
      learning_rate: 0.001
      learning_rate_decay: 0.95
      learning_rate_schedule: "exponential"
      weight_decay: 0.0001
      gradient_clipping: 1.0
      beta1: 0.9
      beta2: 0.999
      epsilon: 1e-8
    
    # Exploration Configuration
    exploration:
      strategy: "epsilon_greedy"
      initial_epsilon: 1.0
      final_epsilon: 0.01
      epsilon_decay: 0.995
      epsilon_decay_steps: 10000
      temperature: 1.0
    
    # Replay Buffer Configuration
    replay_buffer:
      capacity: 100000
      batch_size: 32
      min_size_to_sample: 1000
      prioritized_replay: false
      alpha: 0.6
      beta: 0.4
      beta_increment: 0.001
    
    # RL Parameters
    gamma: 0.99
    tau: 0.005
    update_frequency: 4
    target_update_frequency: 1000
    double_dqn: true
    dueling_dqn: true
    
    # Agent-specific Parameters
    state_dim: 128
    action_dim: 8
    reward_scaling: 1.0
```

### Generator Agent Configuration

Specialized parameters for the Generator agent:

```yaml
agents:
  generator:
    # ... base parameters ...
    
    # Generation-specific parameters
    generation_strategies:
      - "step_by_step"
      - "concept_based"
      - "creative"
      - "structured"
      - "interactive"
      - "visual"
      - "collaborative"
      - "adaptive"
    
    # Reward function weights
    reward_weights:
      quality: 0.5
      novelty: 0.3
      efficiency: 0.2
    
    # Quality thresholds
    quality_threshold: 0.8
    novelty_threshold: 0.6
    
    # Performance bonuses
    coordination_bonus: 0.1
    validation_penalty: 0.2
```

### Validator Agent Configuration

Specialized parameters for the Validator agent:

```yaml
agents:
  validator:
    # ... base parameters ...
    
    # Validation-specific parameters
    validation_strategies:
      - "strict_validation"
      - "balanced_validation"
      - "lenient_validation"
      - "adaptive_threshold"
      - "domain_specific"
      - "pedagogical_focus"
      - "efficiency_optimized"
      - "comprehensive_review"
    
    # Reward function weights
    reward_weights:
      accuracy: 0.7
      efficiency: 0.3
      feedback_quality: 0.2
    
    # Validation thresholds
    accuracy_threshold: 0.85
    confidence_threshold: 0.7
    
    # Penalty weights
    false_positive_penalty: 0.1
    false_negative_penalty: 0.15
    coordination_bonus: 0.1
```

### Curriculum Agent Configuration

Specialized parameters for the Curriculum agent:

```yaml
agents:
  curriculum:
    # ... base parameters ...
    
    # Curriculum-specific parameters
    curriculum_strategies:
      - "linear_progression"
      - "spiral_curriculum"
      - "mastery_based"
      - "adaptive_pacing"
      - "competency_based"
      - "project_based"
      - "inquiry_driven"
      - "differentiated"
    
    # Reward function weights
    reward_weights:
      pedagogical_coherence: 0.4
      learning_progression: 0.4
      objective_alignment: 0.2
    
    # Curriculum thresholds
    coherence_threshold: 0.7
    progression_threshold: 0.7
    
    # Performance bonuses
    integration_bonus: 0.15
    alignment_bonus: 0.1
```

## Coordination Configuration

### Consensus Mechanism

```yaml
coordination:
  consensus:
    strategy: "weighted_average"  # "majority_vote", "expert_priority", "adaptive"
    voting_threshold: 0.5
    confidence_threshold: 0.7
    timeout_seconds: 30.0
    max_iterations: 10
    convergence_threshold: 0.01
    
    # Strategy-specific parameters
    expert_weights:
      generator: 1.0
      validator: 1.2
      curriculum: 1.1
    
    adaptive_learning_rate: 0.1
```

### Communication Protocol

```yaml
coordination:
  communication:
    message_queue_size: 1000
    message_timeout: 10.0
    max_retries: 3
    retry_delay: 1.0
    compression_enabled: true
    encryption_enabled: false
    heartbeat_interval: 5.0
```

### Coordination Parameters

```yaml
coordination:
  coordination_timeout: 60.0
  max_concurrent_coordinations: 10
  conflict_resolution_strategy: "priority_based"
  deadlock_detection_enabled: true
  deadlock_timeout: 120.0
```

## Learning Configuration

### Shared Learning

```yaml
learning:
  shared_learning:
    enabled: true
    experience_sharing_rate: 0.1
    experience_buffer_size: 50000
    sharing_strategy: "high_reward"  # "all", "novel", "adaptive"
    novelty_threshold: 0.8
    reward_threshold: 0.7
    
    # Continuous learning
    continuous_learning_enabled: true
    learning_update_interval: 10.0
    performance_window_size: 100
    adaptation_threshold: 0.05
```

### Training Parameters

```yaml
learning:
  training_enabled: true
  evaluation_interval: 1000
  save_interval: 10000
  max_episodes: 100000
  max_steps_per_episode: 1000
  
  # Performance monitoring
  performance_tracking_enabled: true
  metrics_collection_interval: 1.0
  trend_analysis_enabled: true
```

## System Configuration

### Hardware and Resources

```yaml
system:
  device: "auto"  # "cpu", "cuda", "auto"
  num_workers: 4
  seed: 42
  deterministic: false
  
  # Resource management
  memory_limit_gb: 8.0
  gpu_memory_fraction: 0.8
```

### Logging and Monitoring

```yaml
system:
  log_level: "INFO"
  log_to_file: true
  log_file_path: "logs/marl.log"
  metrics_enabled: true
  
  # Performance monitoring
  monitoring_enabled: true
  metrics_collection_interval: 100
  performance_report_interval: 1000
```

## Deployment Scenarios

### Development Configuration

For development and testing:

```yaml
# development_config.yaml
version: "1.0.0"
name: "development_config"

agents:
  generator:
    network:
      hidden_layers: [128, 64]
    optimization:
      learning_rate: 0.01
    exploration:
      initial_epsilon: 0.5
      epsilon_decay: 0.99
    replay_buffer:
      capacity: 10000
      batch_size: 16

system:
  log_level: "DEBUG"
  num_workers: 2
  memory_limit_gb: 4.0
```

### Production Configuration

For production deployment:

```yaml
# production_config.yaml
version: "1.0.0"
name: "production_config"

agents:
  generator:
    network:
      hidden_layers: [512, 256, 128]
    optimization:
      learning_rate: 0.0001
    exploration:
      initial_epsilon: 0.1
      epsilon_decay: 0.9995
    replay_buffer:
      capacity: 1000000
      batch_size: 64
      prioritized_replay: true

coordination:
  consensus:
    timeout_seconds: 60.0
    max_iterations: 20
  
system:
  log_level: "INFO"
  num_workers: 8
  memory_limit_gb: 16.0
  gpu_memory_fraction: 0.9
```

### High-Performance Configuration

For maximum performance:

```yaml
# high_performance_config.yaml
version: "1.0.0"
name: "high_performance_config"

agents:
  generator:
    network:
      hidden_layers: [1024, 512, 256, 128]
      batch_normalization: true
    optimization:
      optimizer_type: "adamw"
      learning_rate: 0.0005
      gradient_clipping: 0.5
    replay_buffer:
      capacity: 2000000
      batch_size: 128
      prioritized_replay: true

learning:
  shared_learning:
    experience_buffer_size: 100000
    continuous_learning_enabled: true
    learning_update_interval: 5.0

system:
  num_workers: 16
  memory_limit_gb: 32.0
  deterministic: true
```

## Configuration Validation

### Automatic Validation

The system automatically validates configurations:

```python
from core.marl.config.config_schema import MARLConfig

# Load and validate configuration
config = MARLConfig.from_dict(config_data)
errors = config.validate()

if errors:
    print("Configuration errors:")
    for error in errors:
        print(f"  - {error}")
else:
    print("Configuration is valid")
```

### Common Validation Errors

1. **Invalid Learning Rate**: Must be between 0 and 1
2. **Invalid Gamma**: Must be between 0 and 1
3. **Invalid Buffer Size**: Must be positive
4. **Invalid Batch Size**: Must be positive and <= buffer size
5. **Invalid Timeout**: Must be positive
6. **Invalid Threshold**: Must be between 0 and 1

## Performance Tuning

### Learning Rate Optimization

```yaml
# Adaptive learning rate schedule
optimization:
  learning_rate: 0.001
  learning_rate_decay: 0.95
  learning_rate_schedule: "exponential"  # "linear", "cosine", "step"
```

### Exploration Strategy Tuning

```yaml
# Balanced exploration-exploitation
exploration:
  strategy: "epsilon_greedy"
  initial_epsilon: 1.0
  final_epsilon: 0.01
  epsilon_decay: 0.995
  epsilon_decay_steps: 10000
```

### Memory Optimization

```yaml
# Optimized replay buffer
replay_buffer:
  capacity: 100000
  batch_size: 32
  prioritized_replay: true
  alpha: 0.6  # Prioritization strength
  beta: 0.4   # Importance sampling
```

### Coordination Optimization

```yaml
# Efficient coordination
coordination:
  consensus:
    strategy: "adaptive"
    timeout_seconds: 30.0
    convergence_threshold: 0.01
  
  communication:
    compression_enabled: true
    message_queue_size: 1000
```

## Environment-Specific Configurations

### Mathematics Domain

```yaml
# math_domain_config.yaml
agents:
  generator:
    state_dim: 64
    action_dim: 8
    reward_weights:
      quality: 0.6
      novelty: 0.2
      efficiency: 0.2
  
  validator:
    accuracy_threshold: 0.9
    false_negative_penalty: 0.2
  
  curriculum:
    coherence_threshold: 0.8
    progression_threshold: 0.8
```

### Science Domain

```yaml
# science_domain_config.yaml
agents:
  generator:
    state_dim: 96
    action_dim: 8
    reward_weights:
      quality: 0.5
      novelty: 0.4
      efficiency: 0.1
  
  validator:
    accuracy_threshold: 0.85
    domain_specific_validation: true
  
  curriculum:
    curriculum_strategies:
      - "inquiry_driven"
      - "project_based"
      - "spiral_curriculum"
```

## Monitoring and Debugging

### Performance Metrics Configuration

```yaml
system:
  metrics_enabled: true
  metrics_collection_interval: 1.0
  
  # Specific metrics to track
  tracked_metrics:
    - "coordination_success_rate"
    - "agent_performance"
    - "learning_progress"
    - "resource_utilization"
    - "error_rates"
```

### Debug Configuration

```yaml
# debug_config.yaml
system:
  log_level: "DEBUG"
  debug_mode: true
  
  # Debug-specific settings
  debug_settings:
    log_agent_actions: true
    log_coordination_details: true
    log_learning_updates: true
    save_debug_data: true
    debug_data_path: "debug_data/"
```

## Configuration Management

### Version Control

```yaml
# Always include version information
version: "1.0.0"
name: "config_name"
description: "Configuration description"
created_at: "2024-01-01T00:00:00Z"
```

### Configuration Templates

The system provides pre-built templates:

- `default_config.yaml`: Balanced configuration for general use
- `development_config.yaml`: Optimized for development and testing
- `production_config.yaml`: Production-ready configuration
- `high_performance_config.yaml`: Maximum performance configuration
- `research_config.yaml`: Configuration for research and experimentation

### Dynamic Configuration Updates

```python
# Update configuration at runtime
from core.marl.config.config_manager import MARLConfigManager

config_manager = MARLConfigManager()

# Update learning rate
config_manager.update_agent_config(
    "generator", 
    {"optimization.learning_rate": 0.0005}
)

# Update coordination timeout
config_manager.update_coordination_config({
    "consensus.timeout_seconds": 45.0
})
```

## Best Practices

### Configuration Organization

1. **Use Descriptive Names**: Choose clear, descriptive configuration names
2. **Version Control**: Always version your configurations
3. **Environment Separation**: Use separate configurations for different environments
4. **Documentation**: Document custom configuration parameters
5. **Validation**: Always validate configurations before deployment

### Performance Optimization

1. **Start Simple**: Begin with default configurations and optimize incrementally
2. **Monitor Metrics**: Use performance monitoring to guide optimization
3. **A/B Testing**: Test configuration changes systematically
4. **Resource Monitoring**: Monitor resource usage and adjust accordingly
5. **Regular Review**: Regularly review and update configurations

### Security Considerations

1. **Sensitive Data**: Never include sensitive data in configuration files
2. **Access Control**: Implement proper access control for configuration files
3. **Encryption**: Use encryption for sensitive configuration parameters
4. **Audit Trail**: Maintain audit trails for configuration changes
5. **Backup**: Regularly backup configuration files

## Troubleshooting

### Common Configuration Issues

#### Poor Performance
- Check learning rates (too high/low)
- Verify exploration parameters
- Review network architecture
- Monitor resource utilization

#### Coordination Failures
- Increase consensus timeout
- Adjust voting thresholds
- Check communication settings
- Review conflict resolution strategy

#### Memory Issues
- Reduce replay buffer size
- Decrease batch size
- Limit concurrent coordinations
- Enable memory optimization

#### Learning Instability
- Reduce learning rate
- Increase target update frequency
- Enable gradient clipping
- Use more stable exploration strategy

This configuration guide provides comprehensive coverage of all MARL system parameters and optimization strategies. For specific deployment scenarios, refer to the provided templates and examples.