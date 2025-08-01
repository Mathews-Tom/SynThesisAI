"""
MARL Configuration Templates.

This module provides pre-defined configuration templates for different
deployment scenarios and use cases.
"""

# Standard Library
from __future__ import annotations
import logging
from typing import Callable, Dict, List, Optional

# Third-Party Library

# SynThesisAI Modules
from core.marl.config.config_schema import (
    AgentConfig,
    ConsensusConfig,
    ConsensusStrategy,
    CoordinationConfig,
    ExplorationConfig,
    ExplorationStrategy,
    LearningConfig,
    MARLConfig,
    NetworkConfig,
    OptimizationConfig,
    OptimizationType,
    ReplayBufferConfig,
    SharedLearningConfig,
    SystemConfig,
)


class ConfigTemplateManager:
    """
    Manager for MARL configuration templates.

    Provides pre-defined templates for common deployment scenarios
    and the ability to create custom templates.
    """

    def __init__(self):
        """Initialize the template manager."""
        self.logger = logging.getLogger(__name__)

        # Register built-in templates
        self._templates: Dict[str, Callable] = {
            "development": self._create_development_template,
            "production": self._create_production_template,
            "research": self._create_research_template,
            "high_performance": self._create_high_performance_template,
            "low_resource": self._create_low_resource_template,
            "distributed": self._create_distributed_template,
            "single_agent": self._create_single_agent_template,
            "multi_domain": self._create_multi_domain_template,
        }

        self.logger.info(
            "Configuration template manager initialized with %d templates",
            len(self._templates),
        )

    def get_available_templates(self) -> List[str]:
        """Get list of available template names."""
        return list(self._templates.keys())

    def create_from_template(
        self, template_name: str, name: Optional[str] = None, **kwargs
    ) -> MARLConfig:
        """
        Create configuration from template.

        Args:
            template_name: Name of the template to use
            name: Custom name for the configuration
            **kwargs: Additional parameters for template customization

        Returns:
            MARL configuration based on template

        Raises:
            ValueError: If template not found
        """
        if template_name not in self._templates:
            available = ", ".join(self._templates.keys())
            raise ValueError(f"Template '{template_name}' not found. Available: {available}")

        template_func = self._templates[template_name]
        config = template_func(**kwargs)

        if name:
            config.name = name

        self.logger.info("Created configuration from template: %s", template_name)
        return config

    def _create_development_template(self, **kwargs) -> MARLConfig:
        """Create development template with fast training and debugging features."""
        # Development-optimized agents
        agents = {}

        # Generator agent - optimized for quick iteration
        agents["generator"] = AgentConfig(
            agent_id="generator",
            agent_type="generator",
            network=NetworkConfig(
                hidden_layers=[128, 64],  # Smaller network
                dropout_rate=0.1,
            ),
            optimization=OptimizationConfig(
                learning_rate=0.003,  # Higher learning rate
                learning_rate_decay=0.99,
            ),
            exploration=ExplorationConfig(
                initial_epsilon=0.8,  # Lower initial exploration
                epsilon_decay_steps=5000,  # Faster decay
            ),
            replay_buffer=ReplayBufferConfig(
                capacity=10000,  # Smaller buffer
                batch_size=32,
            ),
            state_dim=128,
            action_dim=10,
            update_frequency=2,  # More frequent updates
        )

        # Validator agent
        agents["validator"] = AgentConfig(
            agent_id="validator",
            agent_type="validator",
            network=NetworkConfig(hidden_layers=[128, 64], dropout_rate=0.1),
            optimization=OptimizationConfig(learning_rate=0.003, learning_rate_decay=0.99),
            exploration=ExplorationConfig(initial_epsilon=0.8, epsilon_decay_steps=5000),
            replay_buffer=ReplayBufferConfig(capacity=10000, batch_size=32),
            state_dim=256,
            action_dim=5,
            update_frequency=2,
        )

        # Curriculum agent
        agents["curriculum"] = AgentConfig(
            agent_id="curriculum",
            agent_type="curriculum",
            network=NetworkConfig(
                hidden_layers=[64, 32],  # Even smaller for curriculum
                dropout_rate=0.1,
            ),
            optimization=OptimizationConfig(learning_rate=0.003, learning_rate_decay=0.99),
            exploration=ExplorationConfig(initial_epsilon=0.8, epsilon_decay_steps=5000),
            replay_buffer=ReplayBufferConfig(capacity=5000, batch_size=16),
            state_dim=64,
            action_dim=8,
            update_frequency=2,
        )

        return MARLConfig(
            name="development_config",
            description="Development configuration with fast training and debugging features",
            version="1.0.0",
            agents=agents,
            coordination=CoordinationConfig(
                coordination_timeout=30.0,  # Shorter timeouts
                max_concurrent_coordinations=5,
            ),
            learning=LearningConfig(
                max_episodes=10000,  # Fewer episodes
                evaluation_interval=500,  # More frequent evaluation
                save_interval=2000,
                shared_learning=SharedLearningConfig(
                    experience_buffer_size=10000  # Smaller shared buffer
                ),
            ),
            system=SystemConfig(
                device="auto",
                num_workers=2,  # Fewer workers
                log_level="DEBUG",  # More verbose logging
            ),
        )

    def _create_production_template(self, **kwargs) -> MARLConfig:
        """Create production template with stability and performance focus."""
        # Production-optimized agents
        agents = {}

        # Generator agent - stable and robust
        agents["generator"] = AgentConfig(
            agent_id="generator",
            agent_type="generator",
            network=NetworkConfig(
                hidden_layers=[512, 256, 128],  # Larger network
                dropout_rate=0.2,
                batch_normalization=True,
            ),
            optimization=OptimizationConfig(
                learning_rate=0.0005,  # Conservative learning rate
                learning_rate_decay=0.995,
                gradient_clipping=0.5,  # Stricter clipping
            ),
            exploration=ExplorationConfig(
                initial_epsilon=1.0,
                final_epsilon=0.01,
                epsilon_decay=0.9995,  # Slower decay
                epsilon_decay_steps=50000,
            ),
            replay_buffer=ReplayBufferConfig(
                capacity=100000,  # Large buffer
                batch_size=64,
                prioritized_replay=True,  # Use prioritized replay
            ),
            state_dim=256,
            action_dim=20,
            gamma=0.99,
            tau=0.001,  # Slower target updates
            update_frequency=4,
        )

        # Validator agent
        agents["validator"] = AgentConfig(
            agent_id="validator",
            agent_type="validator",
            network=NetworkConfig(
                hidden_layers=[512, 256, 128],
                dropout_rate=0.2,
                batch_normalization=True,
            ),
            optimization=OptimizationConfig(
                learning_rate=0.0005, learning_rate_decay=0.995, gradient_clipping=0.5
            ),
            exploration=ExplorationConfig(
                initial_epsilon=1.0,
                final_epsilon=0.01,
                epsilon_decay=0.9995,
                epsilon_decay_steps=50000,
            ),
            replay_buffer=ReplayBufferConfig(
                capacity=100000, batch_size=64, prioritized_replay=True
            ),
            state_dim=512,
            action_dim=10,
            gamma=0.99,
            tau=0.001,
            update_frequency=4,
        )

        # Curriculum agent
        agents["curriculum"] = AgentConfig(
            agent_id="curriculum",
            agent_type="curriculum",
            network=NetworkConfig(
                hidden_layers=[256, 128, 64], dropout_rate=0.2, batch_normalization=True
            ),
            optimization=OptimizationConfig(
                learning_rate=0.0005, learning_rate_decay=0.995, gradient_clipping=0.5
            ),
            exploration=ExplorationConfig(
                initial_epsilon=1.0,
                final_epsilon=0.01,
                epsilon_decay=0.9995,
                epsilon_decay_steps=50000,
            ),
            replay_buffer=ReplayBufferConfig(
                capacity=50000, batch_size=32, prioritized_replay=True
            ),
            state_dim=128,
            action_dim=15,
            gamma=0.99,
            tau=0.001,
            update_frequency=4,
        )

        return MARLConfig(
            name="production_config",
            description="Production configuration optimized for stability and performance",
            version="1.0.0",
            agents=agents,
            coordination=CoordinationConfig(
                coordination_timeout=120.0,  # Longer timeouts
                max_concurrent_coordinations=20,
                consensus=ConsensusConfig(
                    strategy=ConsensusStrategy.WEIGHTED_AVERAGE,
                    timeout_seconds=60.0,
                    max_iterations=20,
                ),
            ),
            learning=LearningConfig(
                max_episodes=1000000,
                evaluation_interval=5000,
                save_interval=25000,
                shared_learning=SharedLearningConfig(
                    enabled=True,
                    experience_buffer_size=200000,
                    sharing_strategy="adaptive",
                ),
            ),
            system=SystemConfig(
                device="auto",
                num_workers=8,
                log_level="INFO",
                memory_limit_gb=32.0,
                gpu_memory_fraction=0.8,
            ),
        )

    def _create_research_template(self, **kwargs) -> MARLConfig:
        """Create research template with extensive logging and experimentation features."""
        # Research-focused agents with extensive monitoring
        agents = {}

        # Generator agent - research configuration
        agents["generator"] = AgentConfig(
            agent_id="generator",
            agent_type="generator",
            network=NetworkConfig(
                hidden_layers=[256, 128, 64],
                dropout_rate=0.15,
                activation_function="relu",
            ),
            optimization=OptimizationConfig(
                optimizer_type=OptimizationType.ADAM,
                learning_rate=0.001,
                learning_rate_decay=0.99,
                learning_rate_schedule="exponential",
            ),
            exploration=ExplorationConfig(
                strategy=ExplorationStrategy.EPSILON_GREEDY,
                initial_epsilon=1.0,
                final_epsilon=0.05,
                epsilon_decay=0.995,
                epsilon_decay_steps=20000,
            ),
            replay_buffer=ReplayBufferConfig(
                capacity=50000,
                batch_size=32,
                prioritized_replay=True,
                alpha=0.6,
                beta=0.4,
            ),
            state_dim=256,
            action_dim=20,
            double_dqn=True,
            dueling_dqn=True,
        )

        # Validator agent
        agents["validator"] = AgentConfig(
            agent_id="validator",
            agent_type="validator",
            network=NetworkConfig(
                hidden_layers=[512, 256, 128],
                dropout_rate=0.15,
                activation_function="relu",
            ),
            optimization=OptimizationConfig(
                optimizer_type=OptimizationType.ADAM,
                learning_rate=0.001,
                learning_rate_decay=0.99,
            ),
            exploration=ExplorationConfig(
                strategy=ExplorationStrategy.EPSILON_GREEDY,
                initial_epsilon=1.0,
                final_epsilon=0.05,
                epsilon_decay=0.995,
                epsilon_decay_steps=20000,
            ),
            replay_buffer=ReplayBufferConfig(
                capacity=75000, batch_size=64, prioritized_replay=True
            ),
            state_dim=512,
            action_dim=10,
            double_dqn=True,
            dueling_dqn=True,
        )

        # Curriculum agent
        agents["curriculum"] = AgentConfig(
            agent_id="curriculum",
            agent_type="curriculum",
            network=NetworkConfig(
                hidden_layers=[128, 64, 32],
                dropout_rate=0.15,
                activation_function="relu",
            ),
            optimization=OptimizationConfig(
                optimizer_type=OptimizationType.ADAM,
                learning_rate=0.001,
                learning_rate_decay=0.99,
            ),
            exploration=ExplorationConfig(
                strategy=ExplorationStrategy.EPSILON_GREEDY,
                initial_epsilon=1.0,
                final_epsilon=0.05,
                epsilon_decay=0.995,
                epsilon_decay_steps=20000,
            ),
            replay_buffer=ReplayBufferConfig(
                capacity=25000, batch_size=32, prioritized_replay=True
            ),
            state_dim=128,
            action_dim=15,
            double_dqn=True,
            dueling_dqn=True,
        )

        return MARLConfig(
            name="research_config",
            description="Research configuration with extensive logging and experimentation features",
            version="1.0.0",
            agents=agents,
            coordination=CoordinationConfig(
                coordination_timeout=90.0,
                max_concurrent_coordinations=10,
                consensus=ConsensusConfig(
                    strategy=ConsensusStrategy.ADAPTIVE,
                    timeout_seconds=45.0,
                    max_iterations=15,
                    adaptive_learning_rate=0.1,
                ),
            ),
            learning=LearningConfig(
                max_episodes=500000,
                evaluation_interval=2000,
                save_interval=10000,
                performance_tracking_enabled=True,
                metrics_collection_interval=0.5,  # More frequent metrics
                trend_analysis_enabled=True,
                shared_learning=SharedLearningConfig(
                    enabled=True,
                    experience_buffer_size=100000,
                    sharing_strategy="novel",
                    novelty_threshold=0.8,
                    continuous_learning_enabled=True,
                    learning_update_interval=5.0,
                ),
            ),
            system=SystemConfig(
                device="auto",
                num_workers=4,
                log_level="DEBUG",
                log_to_file=True,
                metrics_enabled=True,
                deterministic=False,  # Allow randomness for research
            ),
        )

    def _create_high_performance_template(self, **kwargs) -> MARLConfig:
        """Create high-performance template optimized for speed and throughput."""
        # High-performance optimized agents
        agents = {}

        # Streamlined agents for maximum performance
        for agent_type, dims in [
            ("generator", (256, 20)),
            ("validator", (512, 10)),
            ("curriculum", (128, 15)),
        ]:
            agents[agent_type] = AgentConfig(
                agent_id=agent_type,
                agent_type=agent_type,
                network=NetworkConfig(
                    hidden_layers=[256, 128],  # Simpler networks
                    dropout_rate=0.1,
                    batch_normalization=False,  # Disable for speed
                ),
                optimization=OptimizationConfig(
                    optimizer_type=OptimizationType.ADAM,
                    learning_rate=0.002,  # Higher learning rate
                    gradient_clipping=1.0,
                ),
                exploration=ExplorationConfig(
                    initial_epsilon=0.5,  # Start with less exploration
                    final_epsilon=0.01,
                    epsilon_decay=0.99,
                    epsilon_decay_steps=10000,  # Faster decay
                ),
                replay_buffer=ReplayBufferConfig(
                    capacity=50000,  # Moderate buffer size
                    batch_size=128,  # Larger batches
                    prioritized_replay=False,  # Disable for speed
                ),
                state_dim=dims[0],
                action_dim=dims[1],
                update_frequency=1,  # Update every step
                target_update_frequency=500,  # Frequent target updates
            )

        return MARLConfig(
            name="high_performance_config",
            description="High-performance configuration optimized for speed and throughput",
            version="1.0.0",
            agents=agents,
            coordination=CoordinationConfig(
                coordination_timeout=30.0,  # Short timeouts
                max_concurrent_coordinations=50,  # High concurrency
                consensus=ConsensusConfig(
                    strategy=ConsensusStrategy.MAJORITY_VOTE,  # Fastest consensus
                    timeout_seconds=15.0,
                    max_iterations=5,  # Quick decisions
                ),
            ),
            learning=LearningConfig(
                max_episodes=200000,
                evaluation_interval=10000,  # Less frequent evaluation
                save_interval=50000,
                performance_tracking_enabled=False,  # Disable for speed
                shared_learning=SharedLearningConfig(
                    enabled=True,
                    experience_buffer_size=50000,
                    sharing_strategy="high_reward",  # Simple strategy
                    continuous_learning_enabled=False,  # Disable for speed
                ),
            ),
            system=SystemConfig(
                device="cuda",  # Force GPU usage
                num_workers=16,  # Maximum parallelism
                log_level="WARNING",  # Minimal logging
                log_to_file=False,
                metrics_enabled=False,  # Disable metrics collection
                gpu_memory_fraction=0.95,  # Use most GPU memory
            ),
        )

    def _create_low_resource_template(self, **kwargs) -> MARLConfig:
        """Create low-resource template for constrained environments."""
        # Resource-constrained agents
        agents = {}

        # Minimal agents for low-resource environments
        for agent_type, dims in [("generator", (128, 10)), ("validator", (256, 5))]:
            agents[agent_type] = AgentConfig(
                agent_id=agent_type,
                agent_type=agent_type,
                network=NetworkConfig(
                    hidden_layers=[64, 32],  # Very small networks
                    dropout_rate=0.1,
                    batch_normalization=False,
                ),
                optimization=OptimizationConfig(learning_rate=0.001, gradient_clipping=0.5),
                exploration=ExplorationConfig(
                    initial_epsilon=0.8,
                    final_epsilon=0.05,
                    epsilon_decay=0.995,
                    epsilon_decay_steps=5000,
                ),
                replay_buffer=ReplayBufferConfig(
                    capacity=5000,  # Very small buffer
                    batch_size=16,  # Small batches
                    prioritized_replay=False,
                ),
                state_dim=dims[0],
                action_dim=dims[1],
                update_frequency=8,  # Less frequent updates
            )

        return MARLConfig(
            name="low_resource_config",
            description="Low-resource configuration for constrained environments",
            version="1.0.0",
            agents=agents,
            coordination=CoordinationConfig(
                coordination_timeout=60.0,
                max_concurrent_coordinations=2,  # Very limited concurrency
                consensus=ConsensusConfig(
                    strategy=ConsensusStrategy.MAJORITY_VOTE,
                    timeout_seconds=30.0,
                    max_iterations=3,
                ),
            ),
            learning=LearningConfig(
                max_episodes=50000,
                evaluation_interval=5000,
                save_interval=20000,
                shared_learning=SharedLearningConfig(
                    enabled=False  # Disable shared learning to save memory
                ),
            ),
            system=SystemConfig(
                device="cpu",  # Force CPU usage
                num_workers=1,  # Single worker
                log_level="ERROR",  # Minimal logging
                memory_limit_gb=2.0,  # Strict memory limit
                metrics_enabled=False,
            ),
        )

    def _create_distributed_template(self, **kwargs) -> MARLConfig:
        """Create distributed template for multi-node deployment."""
        # Distributed-optimized agents
        agents = {}

        # Agents optimized for distributed training
        for agent_type, dims in [
            ("generator", (256, 20)),
            ("validator", (512, 10)),
            ("curriculum", (128, 15)),
        ]:
            agents[agent_type] = AgentConfig(
                agent_id=agent_type,
                agent_type=agent_type,
                network=NetworkConfig(
                    hidden_layers=[512, 256, 128],  # Larger networks for distributed
                    dropout_rate=0.2,
                    batch_normalization=True,
                ),
                optimization=OptimizationConfig(
                    learning_rate=0.0005,  # Conservative for distributed
                    gradient_clipping=0.5,
                ),
                exploration=ExplorationConfig(
                    initial_epsilon=1.0,
                    final_epsilon=0.01,
                    epsilon_decay=0.9995,
                    epsilon_decay_steps=100000,  # Longer decay for distributed
                ),
                replay_buffer=ReplayBufferConfig(
                    capacity=200000,  # Large buffers for distributed
                    batch_size=256,  # Large batches
                    prioritized_replay=True,
                ),
                state_dim=dims[0],
                action_dim=dims[1],
                update_frequency=4,
            )

        return MARLConfig(
            name="distributed_config",
            description="Distributed configuration for multi-node deployment",
            version="1.0.0",
            agents=agents,
            coordination=CoordinationConfig(
                coordination_timeout=300.0,  # Long timeouts for network latency
                max_concurrent_coordinations=100,
                consensus=ConsensusConfig(
                    strategy=ConsensusStrategy.WEIGHTED_AVERAGE,
                    timeout_seconds=120.0,
                    max_iterations=50,
                ),
            ),
            learning=LearningConfig(
                max_episodes=2000000,  # More episodes for distributed
                evaluation_interval=10000,
                save_interval=50000,
                shared_learning=SharedLearningConfig(
                    enabled=True,
                    experience_buffer_size=500000,  # Very large shared buffer
                    sharing_strategy="adaptive",
                    continuous_learning_enabled=True,
                    learning_update_interval=30.0,  # Less frequent updates
                ),
            ),
            system=SystemConfig(
                device="auto",
                num_workers=32,  # Many workers for distributed
                log_level="INFO",
                memory_limit_gb=64.0,  # High memory limit
                gpu_memory_fraction=0.9,
            ),
        )

    def _create_single_agent_template(self, **kwargs) -> MARLConfig:
        """Create single-agent template for testing and development."""
        agent_type = kwargs.get("agent_type", "generator")

        agents = {
            agent_type: AgentConfig(
                agent_id=agent_type,
                agent_type=agent_type,
                network=NetworkConfig(hidden_layers=[256, 128, 64], dropout_rate=0.1),
                optimization=OptimizationConfig(learning_rate=0.001),
                exploration=ExplorationConfig(
                    initial_epsilon=1.0, final_epsilon=0.01, epsilon_decay=0.995
                ),
                replay_buffer=ReplayBufferConfig(capacity=50000, batch_size=32),
                state_dim=256,
                action_dim=10,
            )
        }

        return MARLConfig(
            name="single_agent_config",
            description=f"Single-agent configuration for {agent_type} testing",
            version="1.0.0",
            agents=agents,
            coordination=CoordinationConfig(max_concurrent_coordinations=1),  # Single agent
            learning=LearningConfig(
                shared_learning=SharedLearningConfig(enabled=False)  # No sharing with single agent
            ),
            system=SystemConfig(num_workers=1),
        )

    def _create_multi_domain_template(self, **kwargs) -> MARLConfig:
        """Create multi-domain template with specialized agents for different domains."""
        domains = kwargs.get("domains", ["science", "technology", "mathematics"])

        agents = {}

        # Create specialized agents for each domain
        for i, domain in enumerate(domains):
            for agent_type in ["generator", "validator"]:
                agent_id = f"{domain}_{agent_type}"
                agents[agent_id] = AgentConfig(
                    agent_id=agent_id,
                    agent_type=agent_type,
                    network=NetworkConfig(
                        hidden_layers=[
                            512,
                            256,
                            128,
                        ],  # Larger for domain specialization
                        dropout_rate=0.15,
                    ),
                    optimization=OptimizationConfig(
                        learning_rate=0.0008  # Slightly different rates per domain
                    ),
                    exploration=ExplorationConfig(
                        initial_epsilon=0.9 + i * 0.05,  # Varied exploration
                        final_epsilon=0.01,
                        epsilon_decay=0.995,
                    ),
                    replay_buffer=ReplayBufferConfig(capacity=75000, batch_size=64),
                    state_dim=512,  # Larger state space for domain complexity
                    action_dim=15 + i * 5,  # Varied action spaces
                )

        # Add a curriculum agent for cross-domain coordination
        agents["curriculum"] = AgentConfig(
            agent_id="curriculum",
            agent_type="curriculum",
            network=NetworkConfig(hidden_layers=[256, 128, 64], dropout_rate=0.2),
            state_dim=256,
            action_dim=len(domains) * 5,  # Actions scale with domains
        )

        return MARLConfig(
            name="multi_domain_config",
            description=f"Multi-domain configuration for {', '.join(domains)}",
            version="1.0.0",
            agents=agents,
            coordination=CoordinationConfig(
                coordination_timeout=180.0,  # Longer for complex coordination
                max_concurrent_coordinations=len(domains) * 2,
                consensus=ConsensusConfig(
                    strategy=ConsensusStrategy.EXPERT_PRIORITY,
                    expert_weights={f"{domain}_validator": 1.5 for domain in domains},
                ),
            ),
            learning=LearningConfig(
                shared_learning=SharedLearningConfig(
                    enabled=True,
                    experience_buffer_size=150000,
                    sharing_strategy="adaptive",
                )
            ),
            system=SystemConfig(num_workers=len(domains) * 2),  # Scale workers with domains
        )

    def register_custom_template(self, name: str, template_func: Callable):
        """
        Register a custom template function.

        Args:
            name: Template name
            template_func: Function that returns MARLConfig
        """
        self._templates[name] = template_func
        self.logger.info("Registered custom template: %s", name)

    def get_template_description(self, template_name: str) -> Optional[str]:
        """
        Get description of a template.

        Args:
            template_name: Name of the template

        Returns:
            Template description or None if not found
        """
        if template_name not in self._templates:
            return None

        # Create a sample config to get description
        try:
            config = self.create_from_template(template_name)
            return config.description
        except Exception:
            return "Template description unavailable"


class ConfigTemplateManagerFactory:
    """Factory for creating configuration template managers."""

    @staticmethod
    def create() -> ConfigTemplateManager:
        """Create a configuration template manager."""
        return ConfigTemplateManager()

    @staticmethod
    def create_with_custom_templates(
        custom_templates: Dict[str, Callable],
    ) -> ConfigTemplateManager:
        """
        Create a template manager with custom templates.

        Args:
            custom_templates: Dictionary of custom template functions

        Returns:
            Template manager with custom templates registered
        """
        manager = ConfigTemplateManager()

        for name, template_func in custom_templates.items():
            manager.register_custom_template(name, template_func)

        return manager
