"""Mock MARL Environment for Testing.

This module provides mock environments for isolated testing of MARL agents
and coordination mechanisms without external dependencies.
"""

# Standard Library
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

# Third-Party Library
import numpy as np

# SynThesisAI Modules
from utils.logging_config import get_logger


class EnvironmentType(Enum):
    """Environment type enumeration."""

    CONTENT_GENERATION = "content_generation"
    VALIDATION = "validation"
    CURRICULUM = "curriculum"
    COORDINATION = "coordination"
    MIXED = "mixed"


class DifficultyLevel(Enum):
    """Difficulty level enumeration."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    ADAPTIVE = "adaptive"


@dataclass
class MockEnvironmentConfig:
    """Configuration for mock MARL environment."""

    # Environment settings
    environment_type: EnvironmentType = EnvironmentType.MIXED
    difficulty_level: DifficultyLevel = DifficultyLevel.MEDIUM
    max_episodes: int = 100
    max_steps_per_episode: int = 50

    # Agent settings
    num_agents: int = 3
    agent_types: List[str] = field(default_factory=lambda: ["generator", "validator", "curriculum"])

    # Reward settings
    base_reward: float = 1.0
    penalty_factor: float = -0.5
    coordination_bonus: float = 0.2
    quality_threshold: float = 0.7

    # Environment dynamics
    state_space_size: int = 100
    action_space_size: int = 10
    observation_noise: float = 0.1
    reward_noise: float = 0.05

    # Coordination settings
    coordination_required: bool = True
    coordination_timeout: float = 10.0
    consensus_threshold: float = 0.6

    # Performance settings
    enable_performance_tracking: bool = True
    track_coordination_success: bool = True
    track_learning_progress: bool = True

    def __post_init__(self):
        """Validate configuration."""
        if self.num_agents != len(self.agent_types):
            raise ValueError("Number of agents must match agent types length")

        if self.max_episodes <= 0 or self.max_steps_per_episode <= 0:
            raise ValueError("Episode and step limits must be positive")


@dataclass
class EnvironmentState:
    """Environment state representation."""

    episode: int
    step: int
    agent_states: Dict[str, np.ndarray]
    global_state: np.ndarray
    last_actions: Dict[str, int]
    coordination_status: str
    performance_metrics: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "episode": self.episode,
            "step": self.step,
            "agent_states": {k: v.tolist() for k, v in self.agent_states.items()},
            "global_state": self.global_state.tolist(),
            "last_actions": self.last_actions,
            "coordination_status": self.coordination_status,
            "performance_metrics": self.performance_metrics,
        }


@dataclass
class ActionResult:
    """Result of an action in the environment."""

    agent_id: str
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    info: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agent_id": self.agent_id,
            "action": self.action,
            "reward": self.reward,
            "next_state": self.next_state.tolist(),
            "done": self.done,
            "info": self.info,
        }


class MockMARLEnvironment:
    """Mock MARL environment for testing.

    Provides a controlled environment for testing MARL agents and coordination
    mechanisms with configurable scenarios and predictable behaviors.
    """

    def __init__(self, config: MockEnvironmentConfig) -> None:
        """
        Initialize mock MARL environment.

        Args:
            config: Environment configuration
        """
        self.config = config
        self.logger = get_logger(__name__)

        # Environment state
        self.current_episode = 0
        self.current_step = 0
        self.is_active = False
        self.episode_done = False

        # Agent management
        self.agents: Dict[str, Dict[str, Any]] = {}
        self.agent_states: Dict[str, np.ndarray] = {}
        self.agent_rewards: Dict[str, List[float]] = {}

        # Environment dynamics
        self.global_state = np.zeros(self.config.state_space_size)
        self.state_history: List[EnvironmentState] = []
        self.action_history: List[Dict[str, int]] = []

        # Coordination tracking
        self.coordination_requests: List[Dict[str, Any]] = []
        self.coordination_results: List[Dict[str, Any]] = []
        self.coordination_success_rate = 0.0

        # Performance tracking
        self.performance_metrics = {
            "total_episodes": 0,
            "total_steps": 0,
            "average_reward": 0.0,
            "coordination_success_rate": 0.0,
            "learning_progress": 0.0,
            "quality_score": 0.0,
        }

        # Initialize environment
        self._initialize_environment()

        self.logger.info("Mock MARL environment initialized")

    def _initialize_environment(self) -> None:
        """Initialize environment components."""
        # Initialize agents
        for i, agent_type in enumerate(self.config.agent_types):
            agent_id = f"{agent_type}_{i}"
            self.agents[agent_id] = {
                "type": agent_type,
                "index": i,
                "active": True,
                "performance": 0.0,
            }

            # Initialize agent state
            self.agent_states[agent_id] = self._generate_initial_state()
            self.agent_rewards[agent_id] = []

        # Initialize global state
        self.global_state = self._generate_global_state()

        self.logger.debug("Environment initialized with %d agents", len(self.agents))

    def _generate_initial_state(self) -> np.ndarray:
        """Generate initial state for an agent."""
        state = np.random.normal(0, 1, self.config.state_space_size)

        # Add some structure based on difficulty
        if self.config.difficulty_level == DifficultyLevel.EASY:
            state = state * 0.5  # Reduce variance for easier learning
        elif self.config.difficulty_level == DifficultyLevel.HARD:
            state = state * 2.0  # Increase variance for harder learning

        return state

    def _generate_global_state(self) -> np.ndarray:
        """Generate global environment state."""
        # Combine agent states with some global dynamics
        combined_state = np.zeros(self.config.state_space_size)

        for agent_state in self.agent_states.values():
            combined_state += agent_state * 0.1

        # Add environment-specific dynamics
        if self.config.environment_type == EnvironmentType.CONTENT_GENERATION:
            combined_state += np.random.normal(0, 0.2, self.config.state_space_size)
        elif self.config.environment_type == EnvironmentType.VALIDATION:
            combined_state += np.random.uniform(-0.1, 0.1, self.config.state_space_size)
        elif self.config.environment_type == EnvironmentType.CURRICULUM:
            # Add progressive difficulty
            difficulty_factor = self.current_episode / self.config.max_episodes
            combined_state += np.random.normal(
                0, 0.1 + difficulty_factor * 0.3, self.config.state_space_size
            )

        return combined_state

    async def reset(self) -> Dict[str, np.ndarray]:
        """Reset environment for new episode.

        Returns:
            Initial observations for all agents
        """
        self.current_step = 0
        self.episode_done = False

        # Reset agent states
        for agent_id in self.agents:
            self.agent_states[agent_id] = self._generate_initial_state()
            self.agent_rewards[agent_id] = []

        # Reset global state
        self.global_state = self._generate_global_state()

        # Clear episode history
        self.state_history.clear()
        self.action_history.clear()
        self.coordination_requests.clear()
        self.coordination_results.clear()

        self.is_active = True

        self.logger.debug("Environment reset for episode %d", self.current_episode)

        return self.agent_states.copy()

    async def step(
        self, actions: Dict[str, int]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float], bool, Dict[str, Any]]:
        """Execute one step in the environment.

        Args:
            actions: Actions for each agent

        Returns:
            Tuple of (observations, rewards, done, info)
        """
        if not self.is_active:
            raise RuntimeError("Environment not active. Call reset() first.")

        self.current_step += 1

        # Record actions
        self.action_history.append(actions.copy())

        # Process actions and generate results
        action_results = {}
        rewards = {}

        for agent_id, action in actions.items():
            if agent_id not in self.agents:
                continue

            # Process individual agent action
            result = await self._process_agent_action(agent_id, action)
            action_results[agent_id] = result
            rewards[agent_id] = result.reward

            # Update agent state
            self.agent_states[agent_id] = result.next_state
            self.agent_rewards[agent_id].append(result.reward)

        # Process coordination if required
        if self.config.coordination_required:
            coordination_bonus = await self._process_coordination(actions)
            for agent_id in rewards:
                rewards[agent_id] += coordination_bonus

        # Update global state
        self.global_state = self._generate_global_state()

        # Check if episode is done
        self.episode_done = (
            self.current_step >= self.config.max_steps_per_episode
            or self._check_termination_conditions()
        )

        # Record state
        current_state = EnvironmentState(
            episode=self.current_episode,
            step=self.current_step,
            agent_states=self.agent_states.copy(),
            global_state=self.global_state.copy(),
            last_actions=actions.copy(),
            coordination_status=("active" if self.config.coordination_required else "disabled"),
            performance_metrics=self._calculate_step_metrics(),
        )
        self.state_history.append(current_state)

        # Update performance metrics
        self._update_performance_metrics(rewards)

        # Prepare info
        info = {
            "episode": self.current_episode,
            "step": self.current_step,
            "coordination_success": len(self.coordination_results) > 0,
            "performance_metrics": self.performance_metrics.copy(),
            "action_results": {k: v.to_dict() for k, v in action_results.items()},
        }

        if self.episode_done:
            self.current_episode += 1
            self.is_active = False
            info["episode_summary"] = self._generate_episode_summary()

        return self.agent_states.copy(), rewards, self.episode_done, info

    async def _process_agent_action(self, agent_id: str, action: int) -> ActionResult:
        """Process action for a specific agent."""
        agent = self.agents[agent_id]
        current_state = self.agent_states[agent_id]

        # Validate action
        if action < 0 or action >= self.config.action_space_size:
            action = 0  # Default action for invalid input

        # Calculate base reward based on agent type and action
        base_reward = self._calculate_base_reward(agent, action, current_state)

        # Add noise
        reward = base_reward + np.random.normal(0, self.config.reward_noise)

        # Generate next state
        next_state = self._generate_next_state(current_state, action)

        # Check if agent is done
        done = self._check_agent_done(agent_id, action, next_state)

        # Additional info
        info = {
            "agent_type": agent["type"],
            "action_valid": True,
            "state_quality": self._evaluate_state_quality(next_state),
            "learning_signal": self._generate_learning_signal(agent_id, action, reward),
        }

        return ActionResult(
            agent_id=agent_id,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            info=info,
        )

    def _calculate_base_reward(
        self, agent: Dict[str, Any], action: int, state: np.ndarray
    ) -> float:
        """Calculate base reward for an agent action."""
        agent_type = agent["type"]

        # Type-specific reward calculation
        if agent_type == "generator":
            # Reward for generating quality content
            quality_score = self._evaluate_generation_quality(action, state)
            return self.config.base_reward * quality_score

        elif agent_type == "validator":
            # Reward for accurate validation
            validation_accuracy = self._evaluate_validation_accuracy(action, state)
            return self.config.base_reward * validation_accuracy

        elif agent_type == "curriculum":
            # Reward for pedagogical coherence
            coherence_score = self._evaluate_curriculum_coherence(action, state)
            return self.config.base_reward * coherence_score

        else:
            # Default reward
            return self.config.base_reward * random.uniform(0.5, 1.0)

    def _evaluate_generation_quality(self, action: int, state: np.ndarray) -> float:
        """Evaluate generation quality score."""
        # Simulate quality evaluation based on action and state
        state_quality = np.mean(np.abs(state))
        action_quality = 1.0 - (action / self.config.action_space_size)

        quality = (state_quality + action_quality) / 2.0

        # Add difficulty-based adjustment
        if self.config.difficulty_level == DifficultyLevel.HARD:
            quality *= 0.8  # Harder to achieve high quality
        elif self.config.difficulty_level == DifficultyLevel.EASY:
            quality = min(1.0, quality * 1.2)  # Easier to achieve high quality

        return np.clip(quality, 0.0, 1.0)

    def _evaluate_validation_accuracy(self, action: int, state: np.ndarray) -> float:
        """Evaluate validation accuracy score."""
        # Simulate validation accuracy based on action and state
        state_consistency = 1.0 - np.std(state)
        action_appropriateness = 1.0 - abs(action - self.config.action_space_size // 2) / (
            self.config.action_space_size // 2
        )

        accuracy = (state_consistency + action_appropriateness) / 2.0
        return np.clip(accuracy, 0.0, 1.0)

    def _evaluate_curriculum_coherence(self, action: int, state: np.ndarray) -> float:
        """Evaluate curriculum coherence score."""
        # Simulate coherence evaluation based on progression
        progression_factor = self.current_step / self.config.max_steps_per_episode
        state_progression = np.mean(state) * progression_factor
        action_progression = (action / self.config.action_space_size) * progression_factor

        coherence = (state_progression + action_progression) / 2.0
        return np.clip(coherence, 0.0, 1.0)

    def _generate_next_state(self, current_state: np.ndarray, action: int) -> np.ndarray:
        """Generate next state based on current state and action."""
        # Apply action effect
        action_effect = np.zeros_like(current_state)
        action_effect[action % len(action_effect)] = 1.0

        # State transition dynamics
        next_state = current_state * 0.9 + action_effect * 0.1

        # Add observation noise
        noise = np.random.normal(0, self.config.observation_noise, current_state.shape)
        next_state += noise

        # Apply environment-specific dynamics
        if self.config.environment_type == EnvironmentType.CONTENT_GENERATION:
            # Add creativity dynamics
            creativity_boost = np.random.uniform(-0.1, 0.2, current_state.shape)
            next_state += creativity_boost

        return next_state

    def _check_agent_done(self, agent_id: str, action: int, state: np.ndarray) -> bool:
        """Check if agent is done."""
        # Agent is done if state quality is very poor
        state_quality = self._evaluate_state_quality(state)
        return state_quality < 0.1

    def _evaluate_state_quality(self, state: np.ndarray) -> float:
        """Evaluate quality of a state."""
        # Simple quality metric based on state statistics
        mean_abs = np.mean(np.abs(state))
        std_dev = np.std(state)

        quality = 1.0 / (1.0 + mean_abs + std_dev)
        return np.clip(quality, 0.0, 1.0)

    def _generate_learning_signal(
        self, agent_id: str, action: int, reward: float
    ) -> Dict[str, float]:
        """Generate learning signal for agent."""
        return {
            "reward_signal": reward,
            "exploration_bonus": 0.1 if random.random() < 0.2 else 0.0,
            "coordination_signal": 0.05 if self.config.coordination_required else 0.0,
            "progress_signal": self.current_step / self.config.max_steps_per_episode,
        }

    async def _process_coordination(self, actions: Dict[str, int]) -> float:
        """Process coordination between agents."""
        if not self.config.coordination_required:
            return 0.0

        # Create coordination request
        coordination_request = {
            "timestamp": time.time(),
            "actions": actions.copy(),
            "step": self.current_step,
            "episode": self.current_episode,
        }
        self.coordination_requests.append(coordination_request)

        # Simulate coordination process
        coordination_success = await self._simulate_coordination(actions)

        # Record coordination result
        coordination_result = {
            "timestamp": time.time(),
            "success": coordination_success,
            "actions": actions.copy(),
            "consensus_reached": coordination_success,
            "coordination_time": 0.1,  # Simulated time
        }
        self.coordination_results.append(coordination_result)

        # Calculate coordination bonus
        if coordination_success:
            return self.config.coordination_bonus
        else:
            return self.config.penalty_factor * 0.1

    async def _simulate_coordination(self, actions: Dict[str, int]) -> bool:
        """Simulate coordination process."""
        # Simple coordination simulation
        action_values = list(actions.values())

        if not action_values:
            return False

        # Check if actions are coordinated (similar values)
        action_std = np.std(action_values)
        coordination_threshold = self.config.action_space_size * 0.3

        # Success if actions are coordinated enough
        coordination_success = action_std < coordination_threshold

        # Add some randomness based on difficulty
        if self.config.difficulty_level == DifficultyLevel.HARD:
            coordination_success = coordination_success and random.random() > 0.3
        elif self.config.difficulty_level == DifficultyLevel.EASY:
            coordination_success = coordination_success or random.random() > 0.7

        return coordination_success

    def _check_termination_conditions(self) -> bool:
        """Check if episode should terminate early."""
        # Terminate if all agents perform poorly
        if len(self.agent_rewards) == 0:
            return False

        recent_rewards = []
        for agent_rewards in self.agent_rewards.values():
            if len(agent_rewards) >= 5:  # Check last 5 rewards
                recent_rewards.extend(agent_rewards[-5:])

        if recent_rewards:
            avg_recent_reward = np.mean(recent_rewards)
            return avg_recent_reward < self.config.penalty_factor

        return False

    def _calculate_step_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics for current step."""
        metrics = {}

        # Calculate average reward
        if self.agent_rewards:
            all_rewards = []
            for rewards in self.agent_rewards.values():
                all_rewards.extend(rewards)
            metrics["step_average_reward"] = np.mean(all_rewards) if all_rewards else 0.0

        # Calculate coordination success rate
        if self.coordination_results:
            successful_coordinations = sum(1 for r in self.coordination_results if r["success"])
            metrics["coordination_success_rate"] = successful_coordinations / len(
                self.coordination_results
            )
        else:
            metrics["coordination_success_rate"] = 0.0

        # Calculate state quality
        state_qualities = [
            self._evaluate_state_quality(state) for state in self.agent_states.values()
        ]
        metrics["average_state_quality"] = np.mean(state_qualities) if state_qualities else 0.0

        return metrics

    def _update_performance_metrics(self, rewards: Dict[str, float]) -> None:
        """Update overall performance metrics."""
        self.performance_metrics["total_steps"] += 1

        # Update average reward
        if rewards:
            step_avg_reward = np.mean(list(rewards.values()))
            total_steps = self.performance_metrics["total_steps"]
            current_avg = self.performance_metrics["average_reward"]

            self.performance_metrics["average_reward"] = (
                current_avg * (total_steps - 1) + step_avg_reward
            ) / total_steps

        # Update coordination success rate
        if self.coordination_results:
            successful = sum(1 for r in self.coordination_results if r["success"])
            self.performance_metrics["coordination_success_rate"] = successful / len(
                self.coordination_results
            )

        # Update learning progress (based on reward trend)
        if len(self.state_history) > 10:
            recent_rewards = []
            for state in self.state_history[-10:]:
                recent_rewards.append(state.performance_metrics.get("step_average_reward", 0.0))

            if len(recent_rewards) >= 2:
                # Simple trend calculation
                trend = np.polyfit(range(len(recent_rewards)), recent_rewards, 1)[0]
                self.performance_metrics["learning_progress"] = max(0.0, trend)

        # Update quality score
        if self.state_history:
            recent_quality = self.state_history[-1].performance_metrics.get(
                "average_state_quality", 0.0
            )
            self.performance_metrics["quality_score"] = recent_quality

    def _generate_episode_summary(self) -> Dict[str, Any]:
        """Generate summary for completed episode."""
        summary = {
            "episode": self.current_episode - 1,
            "total_steps": self.current_step,
            "total_reward": sum(sum(rewards) for rewards in self.agent_rewards.values()),
            "average_reward": self.performance_metrics["average_reward"],
            "coordination_attempts": len(self.coordination_requests),
            "coordination_successes": len([r for r in self.coordination_results if r["success"]]),
            "coordination_success_rate": self.performance_metrics["coordination_success_rate"],
            "final_state_quality": self.performance_metrics["quality_score"],
            "learning_progress": self.performance_metrics["learning_progress"],
        }

        # Agent-specific summaries
        summary["agent_summaries"] = {}
        for agent_id, rewards in self.agent_rewards.items():
            if rewards:
                summary["agent_summaries"][agent_id] = {
                    "total_reward": sum(rewards),
                    "average_reward": np.mean(rewards),
                    "best_reward": max(rewards),
                    "worst_reward": min(rewards),
                    "reward_std": np.std(rewards),
                }

        return summary

    def get_environment_info(self) -> Dict[str, Any]:
        """Get environment information."""
        return {
            "config": {
                "environment_type": self.config.environment_type.value,
                "difficulty_level": self.config.difficulty_level.value,
                "num_agents": self.config.num_agents,
                "agent_types": self.config.agent_types,
                "max_episodes": self.config.max_episodes,
                "max_steps_per_episode": self.config.max_steps_per_episode,
            },
            "current_state": {
                "episode": self.current_episode,
                "step": self.current_step,
                "is_active": self.is_active,
                "episode_done": self.episode_done,
            },
            "agents": {
                agent_id: {
                    "type": agent["type"],
                    "active": agent["active"],
                    "current_reward": sum(self.agent_rewards.get(agent_id, [])),
                    "state_quality": self._evaluate_state_quality(
                        self.agent_states.get(agent_id, np.zeros(1))
                    ),
                }
                for agent_id, agent in self.agents.items()
            },
            "performance_metrics": self.performance_metrics.copy(),
        }

    def get_state_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get state history.

        Args:
            limit: Maximum number of states to return

        Returns:
            List of state dictionaries
        """
        history = self.state_history.copy()

        if limit:
            history = history[-limit:]

        return [state.to_dict() for state in history]

    def get_coordination_history(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get coordination history."""
        return {
            "requests": self.coordination_requests.copy(),
            "results": self.coordination_results.copy(),
        }

    def reset_performance_metrics(self) -> None:
        """Reset performance metrics."""
        self.performance_metrics = {
            "total_episodes": 0,
            "total_steps": 0,
            "average_reward": 0.0,
            "coordination_success_rate": 0.0,
            "learning_progress": 0.0,
            "quality_score": 0.0,
        }

        # Clear histories
        self.state_history.clear()
        self.action_history.clear()
        self.coordination_requests.clear()
        self.coordination_results.clear()

        # Reset agent rewards
        for agent_id in self.agent_rewards:
            self.agent_rewards[agent_id].clear()

    async def run_episode(
        self, agent_policies: Dict[str, Callable[[np.ndarray], int]]
    ) -> Dict[str, Any]:
        """Run a complete episode with given agent policies.

        Args:
            agent_policies: Dictionary mapping agent_id to policy function

        Returns:
            Episode summary
        """
        observations = await self.reset()
        episode_data = {
            "observations": [observations],
            "actions": [],
            "rewards": [],
            "infos": [],
        }

        while not self.episode_done:
            # Get actions from policies
            actions = {}
            for agent_id, observation in observations.items():
                if agent_id in agent_policies:
                    policy = agent_policies[agent_id]
                    actions[agent_id] = policy(observation)
                else:
                    # Random action if no policy provided
                    actions[agent_id] = random.randint(0, self.config.action_space_size - 1)

            # Execute step
            observations, rewards, done, info = await self.step(actions)

            # Record data
            episode_data["observations"].append(observations)
            episode_data["actions"].append(actions)
            episode_data["rewards"].append(rewards)
            episode_data["infos"].append(info)

        # Return episode summary
        return info.get("episode_summary", {})
