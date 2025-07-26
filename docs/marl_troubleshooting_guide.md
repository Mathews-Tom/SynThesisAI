# MARL Troubleshooting Guide

## Overview

This guide provides comprehensive troubleshooting information for the Multi-Agent Reinforcement Learning (MARL) coordination system. It covers common issues, diagnostic procedures, and resolution strategies.

## Quick Diagnostic Checklist

When experiencing issues with the MARL system, follow this quick diagnostic checklist:

1. **System Status**: Check overall system health
2. **Agent Status**: Verify all agents are running and responsive
3. **Coordination Status**: Check coordination mechanism operation
4. **Resource Usage**: Monitor CPU, memory, and GPU utilization
5. **Error Logs**: Review recent error logs and patterns
6. **Configuration**: Validate configuration parameters
7. **Network Connectivity**: Verify inter-agent communication
8. **Performance Metrics**: Check key performance indicators

## Common Issues and Solutions

### 1. Low Coordination Success Rate

**Symptoms:**
- Coordination success rate below 85%
- Frequent coordination timeouts
- High conflict resolution failures

**Diagnostic Steps:**
```bash
# Check coordination metrics
uv run python -c "
from core.marl.monitoring.performance_monitor import MARLPerformanceMonitor
monitor = MARLPerformanceMonitor()
print(f'Success rate: {monitor.get_coordination_success_rate():.2%}')
print(f'System summary: {monitor.get_system_performance_summary()}')
"

# Check agent health
uv run python -c "
from core.marl.fault_tolerance.agent_monitor import AgentMonitor
monitor = AgentMonitor()
print(monitor.get_system_health_summary())
"
```

**Common Causes and Solutions:**

#### Misaligned Reward Functions
```yaml
# Fix: Align reward functions across agents
agents:
  generator:
    reward_weights:
      quality: 0.5
      novelty: 0.3
      efficiency: 0.2
      coordination_bonus: 0.1  # Add coordination incentive
  
  validator:
    reward_weights:
      accuracy: 0.7
      efficiency: 0.3
      coordination_bonus: 0.1  # Add coordination incentive
```

#### Inappropriate Consensus Strategy
```yaml
# Fix: Use adaptive consensus for complex scenarios
coordination:
  consensus:
    strategy: "adaptive"  # Instead of "majority_vote"
    timeout_seconds: 60.0  # Increase timeout
    convergence_threshold: 0.05  # Relax convergence
```

#### Communication Bottlenecks
```yaml
# Fix: Optimize communication settings
coordination:
  communication:
    message_queue_size: 2000  # Increase queue size
    compression_enabled: true  # Enable compression
    max_retries: 5  # Increase retry attempts
```

### 2. Poor Learning Performance

**Symptoms:**
- Slow learning convergence
- Unstable policy updates
- Poor knowledge sharing effectiveness

**Diagnostic Steps:**
```bash
# Check learning metrics
uv run python -c "
from core.marl.learning.continuous_learning import ContinuousLearningManager
manager = ContinuousLearningManager()
print(manager.get_learning_summary())
"

# Check shared experience effectiveness
uv run python -c "
from core.marl.learning.shared_experience import SharedExperienceManager
manager = SharedExperienceManager()
print(manager.get_statistics())
"
```

**Common Causes and Solutions:**

#### Inappropriate Learning Rate
```yaml
# Fix: Adjust learning rate based on performance
agents:
  generator:
    optimization:
      learning_rate: 0.0001  # Reduce if unstable
      learning_rate_decay: 0.99  # Add decay
      learning_rate_schedule: "exponential"
```

#### Poor Experience Sharing
```yaml
# Fix: Optimize experience sharing parameters
learning:
  shared_learning:
    experience_sharing_rate: 0.2  # Increase sharing
    sharing_strategy: "adaptive"  # Use adaptive strategy
    novelty_threshold: 0.6  # Lower threshold
    reward_threshold: 0.5  # Lower threshold
```

#### Inadequate Exploration
```yaml
# Fix: Improve exploration strategy
agents:
  generator:
    exploration:
      strategy: "epsilon_greedy"
      initial_epsilon: 0.8  # Increase initial exploration
      epsilon_decay: 0.999  # Slower decay
      epsilon_decay_steps: 50000  # More steps
```

### 3. System Instability

**Symptoms:**
- Frequent system crashes
- Memory overflow errors
- Agent disconnections
- Deadlock situations

**Diagnostic Steps:**
```bash
# Check system health
uv run python -c "
from core.marl.fault_tolerance.fault_tolerance_manager import FaultToleranceManager
manager = FaultToleranceManager()
print(manager.get_system_health_summary())
"

# Check for deadlocks
uv run python -c "
from core.marl.fault_tolerance.deadlock_detector import DeadlockDetector
detector = DeadlockDetector()
print(detector.get_deadlock_statistics())
"
```

**Common Causes and Solutions:**

#### Memory Overflow
```yaml
# Fix: Optimize memory usage
agents:
  generator:
    replay_buffer:
      capacity: 50000  # Reduce buffer size
      batch_size: 16   # Reduce batch size

system:
  memory_limit_gb: 8.0  # Set memory limit
  num_workers: 2        # Reduce workers
```

#### Agent Failures
```yaml
# Fix: Improve fault tolerance
system:
  fault_tolerance:
    agent_health_check_interval: 5.0
    max_failure_count: 3
    recovery_timeout: 30.0
    auto_restart_enabled: true
```

#### Deadlock Detection
```yaml
# Fix: Enable deadlock prevention
coordination:
  deadlock_detection_enabled: true
  deadlock_timeout: 60.0
  max_concurrent_coordinations: 5  # Limit concurrency
```

### 4. Performance Degradation

**Symptoms:**
- Increasing response times
- Decreasing throughput
- High resource utilization
- Slow coordination resolution

**Diagnostic Steps:**
```bash
# Monitor performance trends
uv run python -c "
from core.marl.monitoring.performance_analyzer import PerformanceAnalyzer
analyzer = PerformanceAnalyzer()
report = analyzer.generate_comprehensive_report()
print(f'Overall score: {report.overall_score}')
print(f'Bottlenecks: {report.bottlenecks}')
"

# Check resource utilization
uv run python -c "
from core.marl.monitoring.system_monitor import SystemMonitor
monitor = SystemMonitor()
print(monitor.get_resource_usage())
"
```

**Common Causes and Solutions:**

#### Resource Bottlenecks
```yaml
# Fix: Optimize resource allocation
system:
  num_workers: 8  # Increase workers
  gpu_memory_fraction: 0.9  # Use more GPU memory
  
agents:
  generator:
    network:
      hidden_layers: [256, 128]  # Reduce network size
    replay_buffer:
      batch_size: 64  # Increase batch size for efficiency
```

#### Inefficient Coordination
```yaml
# Fix: Streamline coordination process
coordination:
  consensus:
    strategy: "weighted_average"  # Faster than adaptive
    timeout_seconds: 30.0  # Reduce timeout
    max_iterations: 5  # Limit iterations
```

#### Suboptimal Configuration
```yaml
# Fix: Use performance-optimized settings
learning:
  shared_learning:
    continuous_learning_enabled: false  # Disable if not needed
    learning_update_interval: 20.0  # Reduce frequency
```

## Error Analysis

### Error Classification

The MARL system classifies errors into several categories:

#### Agent Errors
- **Learning Divergence**: Agent policy becomes unstable
- **Action Selection Failure**: Agent cannot select valid actions
- **Network Update Failure**: Neural network update fails
- **Memory Overflow**: Agent memory usage exceeds limits

#### Coordination Errors
- **Consensus Failure**: Agents cannot reach consensus
- **Communication Timeout**: Inter-agent communication fails
- **Conflict Resolution Failure**: Cannot resolve agent conflicts
- **Deadlock Detection**: Circular waiting detected

#### System Errors
- **Resource Exhaustion**: System runs out of resources
- **Configuration Error**: Invalid configuration parameters
- **Network Connectivity**: Communication infrastructure fails
- **Hardware Failure**: Underlying hardware issues

### Error Pattern Analysis

```python
# Analyze error patterns
from core.marl.error_handling.error_analyzer import ErrorAnalyzer

analyzer = ErrorAnalyzer()
patterns = analyzer.analyze_error_patterns(time_window=3600)  # Last hour

print("Error patterns:")
for pattern in patterns:
    print(f"  {pattern.error_type}: {pattern.frequency} occurrences")
    print(f"    Trend: {pattern.trend}")
    print(f"    Recommendations: {pattern.recommendations}")
```

## Diagnostic Tools

### Performance Dashboard

Access the real-time performance dashboard:

```bash
# Start performance monitoring dashboard
uv run python -m core.marl.monitoring.dashboard --port 8080
```

Navigate to `http://localhost:8080` to view:
- Real-time coordination metrics
- Agent performance graphs
- Resource utilization charts
- Error rate trends

### Log Analysis

```bash
# Analyze recent logs for patterns
uv run python -c "
from core.marl.monitoring.log_analyzer import LogAnalyzer
analyzer = LogAnalyzer()
analysis = analyzer.analyze_recent_logs(hours=1)
print(f'Error rate: {analysis.error_rate:.2%}')
print(f'Top errors: {analysis.top_errors}')
print(f'Performance trends: {analysis.performance_trends}')
"
```

### Health Check

```bash
# Comprehensive system health check
uv run python -c "
from core.marl.monitoring.health_checker import HealthChecker
checker = HealthChecker()
health = checker.comprehensive_health_check()
print(f'Overall health: {health.overall_status}')
print(f'Issues found: {len(health.issues)}')
for issue in health.issues:
    print(f'  - {issue.severity}: {issue.description}')
"
```

## Recovery Procedures

### Agent Recovery

```python
# Restart failed agent
from core.marl.fault_tolerance.agent_monitor import AgentMonitor

monitor = AgentMonitor()
failed_agents = monitor.get_failed_agents()

for agent_id in failed_agents:
    print(f"Restarting agent: {agent_id}")
    success = monitor.restart_agent(agent_id)
    print(f"Restart {'successful' if success else 'failed'}")
```

### Coordination Recovery

```python
# Reset coordination mechanism
from core.marl.coordination.coordination_policy import CoordinationPolicy

policy = CoordinationPolicy()
policy.reset_coordination_state()
print("Coordination state reset")
```

### System Recovery

```python
# Full system recovery
from core.marl.fault_tolerance.fault_tolerance_manager import FaultToleranceManager

manager = FaultToleranceManager()
recovery_result = manager.full_system_recovery()
print(f"Recovery status: {recovery_result.status}")
print(f"Recovery time: {recovery_result.recovery_time:.2f}s")
```

## Performance Optimization

### Profiling

```bash
# Profile system performance
uv run python -m cProfile -o marl_profile.prof -c "
from core.marl.coordination.marl_coordinator import MultiAgentRLCoordinator
coordinator = MultiAgentRLCoordinator()
# Run coordination tasks
"

# Analyze profile
uv run python -c "
import pstats
stats = pstats.Stats('marl_profile.prof')
stats.sort_stats('cumulative').print_stats(20)
"
```

### Memory Profiling

```bash
# Profile memory usage
uv run python -m memory_profiler -c "
from core.marl.agents.generator_agent import GeneratorRLAgent
# Profile agent memory usage
"
```

### Optimization Recommendations

Based on profiling results, common optimizations include:

1. **Reduce Network Complexity**: Smaller neural networks for faster inference
2. **Optimize Batch Sizes**: Balance memory usage and training efficiency
3. **Implement Caching**: Cache frequently computed values
4. **Parallel Processing**: Utilize multiple cores for coordination
5. **Memory Management**: Implement efficient memory cleanup

## Monitoring and Alerting

### Setting Up Alerts

```yaml
# monitoring_config.yaml
alerts:
  coordination_success_rate:
    threshold: 0.85
    severity: "critical"
    action: "restart_coordination"
  
  memory_usage:
    threshold: 0.9
    severity: "warning"
    action: "cleanup_memory"
  
  error_rate:
    threshold: 0.05
    severity: "warning"
    action: "analyze_errors"
```

### Custom Monitoring

```python
# Custom monitoring script
from core.marl.monitoring.custom_monitor import CustomMonitor

monitor = CustomMonitor()

# Add custom metrics
monitor.add_metric("custom_coordination_metric", lambda: calculate_custom_metric())
monitor.add_alert("custom_alert", threshold=0.8, action="custom_action")

# Start monitoring
monitor.start_monitoring(interval=60)  # Check every minute
```

## Best Practices for Troubleshooting

### Systematic Approach

1. **Reproduce the Issue**: Ensure the issue is reproducible
2. **Gather Information**: Collect logs, metrics, and system state
3. **Isolate the Problem**: Narrow down the root cause
4. **Test Solutions**: Implement and test potential fixes
5. **Monitor Results**: Verify the fix resolves the issue
6. **Document Findings**: Record the issue and solution

### Preventive Measures

1. **Regular Health Checks**: Implement automated health monitoring
2. **Performance Baselines**: Establish performance baselines
3. **Configuration Validation**: Always validate configurations
4. **Gradual Rollouts**: Deploy changes incrementally
5. **Backup Strategies**: Maintain system backups and rollback plans

### Emergency Procedures

#### System Shutdown
```bash
# Emergency system shutdown
uv run python -c "
from core.marl.system.emergency_shutdown import EmergencyShutdown
shutdown = EmergencyShutdown()
shutdown.graceful_shutdown(timeout=30)
"
```

#### Rollback Configuration
```bash
# Rollback to previous configuration
uv run python -c "
from core.marl.config.config_manager import MARLConfigManager
manager = MARLConfigManager()
manager.rollback_to_previous_config()
"
```

#### Data Recovery
```bash
# Recover from backup
uv run python -c "
from core.marl.system.backup_manager import BackupManager
backup = BackupManager()
backup.restore_from_backup('latest')
"
```

## Support and Resources

### Log Locations

- **System Logs**: `logs/marl.log`
- **Agent Logs**: `logs/agents/`
- **Coordination Logs**: `logs/coordination/`
- **Performance Logs**: `logs/performance/`
- **Error Logs**: `logs/errors/`

### Configuration Files

- **Main Config**: `config/marl_config.yaml`
- **Agent Configs**: `config/agents/`
- **System Config**: `config/system/`
- **Monitoring Config**: `config/monitoring/`

### Useful Commands

```bash
# Check system status
uv run python -m core.marl.system.status

# Validate configuration
uv run python -m core.marl.config.validator config/marl_config.yaml

# Run diagnostics
uv run python -m core.marl.diagnostics.full_diagnostic

# Generate performance report
uv run python -m core.marl.monitoring.report_generator

# Export system metrics
uv run python -m core.marl.monitoring.metrics_exporter --format json
```

### Getting Help

1. **Check Documentation**: Review architecture and configuration guides
2. **Search Logs**: Look for similar error patterns in logs
3. **Community Forums**: Check community discussions and solutions
4. **Issue Tracking**: Report bugs and feature requests
5. **Professional Support**: Contact support team for critical issues

This troubleshooting guide provides comprehensive coverage of common MARL system issues and their solutions. For specific problems not covered here, refer to the detailed logs and diagnostic tools provided by the system.