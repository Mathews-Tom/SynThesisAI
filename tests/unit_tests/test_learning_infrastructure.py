"""
Unit tests for MARL learning infrastructure.

This module tests the advanced learning infrastructure components including
optimization, experience sharing, and distributed training capabilities.
"""

# Standard Library
import time
from unittest.mock import patch

# Third-Party Library
import numpy as np
import pytest
import torch
import torch.nn as nn

# SynThesisAI Modules
from core.marl.agents.experience import Experience
from core.marl.config import AgentConfig, ExperienceConfig, MARLConfig
from core.marl.exceptions import LearningDivergenceError
from core.marl.learning.distributed_training import (
    DistributedCoordinationManager,
    DistributedTrainingManager,
    TrainingNode,
    TrainingNodeType,
)
from core.marl.learning.experience_sharing import (
    ActionFrequencyTracker,
    CoordinationSuccessTracker,
    ExperienceValue,
    SharedExperienceManager,
    StateNoveltyTracker,
)
from core.marl.learning.optimization import (
    AdaptiveLearningRateScheduler,
    LearningStabilizer,
    MARLOptimizer,
    OptimizationMetrics,
)


# Pytest fixtures for common objects
@pytest.fixture
def mock_network():
    return MockNetwork()


@pytest.fixture
def agent_config():
    return AgentConfig()


@pytest.fixture
def default_optimizer(mock_network, agent_config):
    return MARLOptimizer(mock_network, agent_config, "test_agent")


# Simple test network for optimization tests
class MockNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, x):
        return self.linear(x)


class TestMARLOptimizer:
    """Test MARL optimizer functionality."""

    def test_optimizer_initialization(self, mock_network, agent_config, default_optimizer):
        """Test optimizer initialization."""
        optimizer = default_optimizer
        network = mock_network
        config = agent_config

        assert optimizer.agent_id == "test_agent"
        assert optimizer.config == config
        assert optimizer.network == network
        assert optimizer.step_count == 0
        assert isinstance(optimizer.metrics, OptimizationMetrics)

    def test_optimization_step(self, default_optimizer):
        """Test optimization step execution."""
        optimizer = default_optimizer
        network = optimizer.network

        # Create dummy input and target
        x = torch.randn(4, 4)
        target = torch.randn(4, 2)

        # Forward pass and loss calculation
        output = network(x)
        loss = nn.MSELoss()(output, target)

        # Optimization step
        metrics = optimizer.step(loss)

        assert "loss" in metrics
        assert "gradient_norm" in metrics
        assert "learning_rate" in metrics
        assert optimizer.step_count == 1
        assert optimizer.metrics.total_updates == 1

    def test_learning_rate_adjustment(self):
        """Test learning rate adjustment."""
        network = MockNetwork()
        config = AgentConfig()
        config.learning_rate = 0.001
        optimizer = MARLOptimizer(network, config, "test_agent")

        initial_lr = optimizer.optimizer.param_groups[0]["lr"]
        assert initial_lr == 0.001

        # Adjust learning rate
        optimizer.adjust_learning_rate(0.5)

        new_lr = optimizer.optimizer.param_groups[0]["lr"]
        assert new_lr == 0.0005

    def test_divergence_detection(self, default_optimizer):
        """Test learning divergence detection."""
        optimizer = default_optimizer

        # Create loss that exceeds divergence threshold
        divergent_loss = torch.tensor(20.0)  # Above default threshold of 10.0

        with pytest.raises(LearningDivergenceError):
            optimizer.step(divergent_loss)

    def test_optimization_state_save_load(self, default_optimizer):
        """Test optimization state save and load."""
        optimizer = default_optimizer

        # Perform some optimization steps
        for _ in range(5):
            x = torch.randn(2, 4)
            target = torch.randn(2, 2)
            output = optimizer.network(x)
            loss = nn.MSELoss()(output, target)
            optimizer.step(loss)

        # Save state
        state = optimizer.get_optimization_state()

        # Create new optimizer and load state
        new_network = MockNetwork()
        new_optimizer = MARLOptimizer(new_network, optimizer.config, optimizer.agent_id)
        new_optimizer.load_optimization_state(state)

        assert new_optimizer.step_count == optimizer.step_count


class TestAdaptiveLearningRateScheduler:
    """Test adaptive learning rate scheduler."""

    def test_scheduler_initialization(self):
        """Test scheduler initialization."""
        network = MockNetwork()
        config = AgentConfig()
        optimizer = MARLOptimizer(network, config, "test_agent")
        scheduler = AdaptiveLearningRateScheduler(optimizer, config)

        assert scheduler.optimizer == optimizer
        assert scheduler.config == config
        assert scheduler.agent_id == "test_agent"
        assert scheduler.best_loss == float("inf")
        assert scheduler.patience_counter == 0

    def test_learning_rate_reduction(self):
        """Test learning rate reduction on plateau."""
        network = MockNetwork()
        config = AgentConfig()
        config.lr_patience = 3  # Reduce patience for testing
        optimizer = MARLOptimizer(network, config, "test_agent")
        scheduler = AdaptiveLearningRateScheduler(optimizer, config)

        initial_lr = optimizer.optimizer.param_groups[0]["lr"]

        # Simulate plateau (no improvement)
        for _ in range(5):
            scheduler.step(1.0)  # Constant loss

        final_lr = optimizer.optimizer.param_groups[0]["lr"]
        assert final_lr < initial_lr  # Learning rate should be reduced

    def test_scheduler_state_save_load(self):
        """Test scheduler state save and load."""
        network = MockNetwork()
        config = AgentConfig()
        optimizer = MARLOptimizer(network, config, "test_agent")
        scheduler = AdaptiveLearningRateScheduler(optimizer, config)

        # Update scheduler state
        scheduler.step(0.5)
        scheduler.step(0.4)
        scheduler.step(0.6)

        # Save state
        state = scheduler.get_scheduler_state()

        # Create new scheduler and load state
        new_optimizer = MARLOptimizer(MockNetwork(), config, "test_agent")
        new_scheduler = AdaptiveLearningRateScheduler(new_optimizer, config)
        new_scheduler.load_scheduler_state(state)

        assert new_scheduler.best_loss == scheduler.best_loss
        assert new_scheduler.patience_counter == scheduler.patience_counter


class TestLearningStabilizer:
    """Test learning stabilizer functionality."""

    def test_stabilizer_initialization(self):
        """Test stabilizer initialization."""
        stabilizer = LearningStabilizer("test_agent")

        assert stabilizer.agent_id == "test_agent"
        assert len(stabilizer.loss_history) == 0
        assert len(stabilizer.gradient_history) == 0

    def test_stability_check_normal(self):
        """Test stability check with normal training."""
        stabilizer = LearningStabilizer("test_agent")

        # Simulate normal training
        for i in range(20):
            loss = 1.0 - i * 0.01  # Decreasing loss
            gradient_norm = 0.5 + np.random.normal(0, 0.1)

            report = stabilizer.check_stability(loss, gradient_norm)

        assert report["stable"] is True
        assert len(report["issues"]) == 0

    def test_stability_check_loss_spike(self):
        """Test stability check with loss spike."""
        stabilizer = LearningStabilizer("test_agent")

        # Add normal losses first
        for i in range(15):
            stabilizer.check_stability(0.5, 0.5)

        # Add loss spike
        report = stabilizer.check_stability(10.0, 0.5)  # Large loss spike

        assert report["stable"] is False
        assert "loss_spike" in report["issues"]
        assert "reduce_learning_rate" in report["recommendations"]

    def test_stability_check_gradient_explosion(self):
        """Test stability check with gradient explosion."""
        stabilizer = LearningStabilizer("test_agent")

        # Add some normal history first (need at least 10 for stability checks)
        for i in range(10):
            stabilizer.check_stability(0.5, 0.5)

        # Add gradient explosion
        report = stabilizer.check_stability(0.5, 15.0)  # Large gradient norm

        assert report["stable"] is False
        assert "gradient_explosion" in report["issues"]
        assert "clip_gradients" in report["recommendations"]


class TestSharedExperienceManager:
    """Test shared experience manager functionality."""

    def test_manager_initialization(self):
        """Test experience manager initialization."""
        config = ExperienceConfig()
        manager = SharedExperienceManager(config)

        assert manager.config == config
        assert len(manager.shared_experiences) == 0
        assert len(manager.experience_metadata) == 0
        assert manager.sharing_stats["total_shared"] == 0

    def test_experience_value_evaluation(self):
        """Test experience value evaluation."""
        config = ExperienceConfig()
        config.high_reward_threshold = 0.8
        config.novelty_threshold = 0.7
        manager = SharedExperienceManager(config)

        # High reward experience
        high_reward_exp = Experience(
            state=np.array([1.0, 2.0, 3.0]),
            action=1,
            reward=1.0,  # Above threshold
            next_state=np.array([1.1, 2.1, 3.1]),
            done=False,
        )

        should_share, value_type, score = manager.evaluate_experience_value(
            high_reward_exp, "agent1", {}
        )

        assert value_type == ExperienceValue.HIGH_REWARD
        assert score > 0.0

    def test_experience_sharing(self):
        """Test experience sharing mechanism."""
        config = ExperienceConfig()
        manager = SharedExperienceManager(config)

        experience = Experience(
            state=np.array([1.0, 2.0]),
            action=0,
            reward=0.9,
            next_state=np.array([1.1, 2.1]),
            done=False,
        )

        # Share experience
        exp_id = manager.share_experience(experience, "agent1", ExperienceValue.HIGH_REWARD, 0.9)

        assert exp_id in manager.shared_experiences
        assert exp_id in manager.experience_metadata
        assert manager.sharing_stats["total_shared"] == 1
        assert manager.agent_contributions["agent1"] == 1

    def test_experience_retrieval(self):
        """Test experience retrieval for agents."""
        config = ExperienceConfig()
        manager = SharedExperienceManager(config)

        # Share some experiences
        for i in range(3):
            exp = Experience(
                state=np.array([float(i)]),
                action=i,
                reward=0.8 + i * 0.1,
                next_state=np.array([float(i + 1)]),
                done=False,
            )
            manager.share_experience(exp, f"agent{i}", ExperienceValue.HIGH_REWARD, 0.8 + i * 0.1)

        # Retrieve experiences for different agent
        experiences = manager.get_experiences_for_agent("agent_new", max_experiences=2)

        assert len(experiences) == 2
        assert all(isinstance(exp, Experience) for _, exp in experiences)
        assert manager.sharing_stats["total_consumed"] == 2

    def test_experience_feedback(self):
        """Test experience feedback mechanism."""
        config = ExperienceConfig()
        manager = SharedExperienceManager(config)

        # Share experience
        exp = Experience(
            state=np.array([1.0]),
            action=0,
            reward=0.9,
            next_state=np.array([1.1]),
            done=False,
        )
        exp_id = manager.share_experience(exp, "agent1", ExperienceValue.HIGH_REWARD, 0.9)

        # Provide feedback
        manager.provide_feedback(exp_id, True, "agent2")

        metadata = manager.experience_metadata[exp_id]
        assert len(metadata.success_feedback) == 1
        assert metadata.success_feedback[0] is True
        assert manager.sharing_stats["successful_shares"] == 1

    def test_experience_cleanup(self):
        """Test old experience cleanup."""
        config = ExperienceConfig()
        manager = SharedExperienceManager(config)

        # Share experience
        exp = Experience(
            state=np.array([1.0]),
            action=0,
            reward=0.9,
            next_state=np.array([1.1]),
            done=False,
        )
        exp_id = manager.share_experience(exp, "agent1", ExperienceValue.HIGH_REWARD, 0.9)

        # Manually set old timestamp
        manager.experience_metadata[exp_id].creation_time = time.time() - 25 * 3600  # 25 hours ago

        # Cleanup old experiences
        removed_count = manager.cleanup_old_experiences(max_age_hours=24.0)

        assert removed_count == 1
        assert exp_id not in manager.shared_experiences
        assert exp_id not in manager.experience_metadata

    def test_sharing_statistics(self):
        """Test sharing statistics generation."""
        config = ExperienceConfig()
        manager = SharedExperienceManager(config)

        # Share some experiences and provide feedback
        for i in range(3):
            exp = Experience(
                state=np.array([float(i)]),
                action=i,
                reward=0.8,
                next_state=np.array([float(i + 1)]),
                done=False,
            )
            exp_id = manager.share_experience(exp, f"agent{i}", ExperienceValue.HIGH_REWARD, 0.8)
            manager.provide_feedback(exp_id, i % 2 == 0, "feedback_agent")

        stats = manager.get_sharing_statistics()

        assert stats["total_shared"] == 3
        assert stats["successful_shares"] == 2
        assert stats["failed_shares"] == 1
        assert stats["sharing_effectiveness"] == 2 / 3
        assert "agent_contributions" in stats
        assert "value_type_distribution" in stats


class TestStateNoveltyTracker:
    """Test state novelty tracking."""

    def test_novelty_tracker_initialization(self):
        """Test novelty tracker initialization."""
        tracker = StateNoveltyTracker()

        assert tracker.novelty_threshold == 0.1
        assert len(tracker.agent_state_histories) == 0

    def test_first_state_novelty(self):
        """Test that first state is always novel."""
        tracker = StateNoveltyTracker()

        state = np.array([1.0, 2.0, 3.0])
        novelty = tracker.assess_novelty(state, "agent1")

        assert novelty == 1.0
        assert len(tracker.agent_state_histories["agent1"]) == 1

    def test_similar_state_novelty(self):
        """Test novelty assessment for similar states."""
        tracker = StateNoveltyTracker()

        # Add first state
        state1 = np.array([1.0, 2.0, 3.0])
        novelty1 = tracker.assess_novelty(state1, "agent1")

        # Add very similar state
        state2 = np.array([1.01, 2.01, 3.01])
        novelty2 = tracker.assess_novelty(state2, "agent1")

        assert novelty1 == 1.0
        assert novelty2 < novelty1  # Should be less novel

    def test_different_state_novelty(self):
        """Test novelty assessment for different states."""
        tracker = StateNoveltyTracker()

        # Add first state
        state1 = np.array([1.0, 2.0, 3.0])
        tracker.assess_novelty(state1, "agent1")

        # Add very different state
        state2 = np.array([10.0, 20.0, 30.0])
        novelty2 = tracker.assess_novelty(state2, "agent1")

        assert novelty2 > 0.5  # Should be quite novel


class TestActionFrequencyTracker:
    """Test action frequency tracking."""

    def test_frequency_tracker_initialization(self):
        """Test frequency tracker initialization."""
        tracker = ActionFrequencyTracker()

        assert len(tracker.agent_action_counts) == 0
        assert len(tracker.agent_total_actions) == 0

    def test_first_action_rarity(self):
        """Test that first action has high rarity."""
        tracker = ActionFrequencyTracker()

        rarity = tracker.assess_rarity(0, "agent1")

        assert rarity == 0.0  # First action has frequency 1.0, so rarity is 0.0
        assert tracker.agent_action_counts["agent1"][0] == 1
        assert tracker.agent_total_actions["agent1"] == 1

    def test_repeated_action_rarity(self):
        """Test rarity assessment for repeated actions."""
        tracker = ActionFrequencyTracker()

        # Perform same action multiple times
        for _ in range(10):
            rarity = tracker.assess_rarity(0, "agent1")

        # Action 0 should have low rarity (high frequency)
        assert rarity < 0.1

        # Try different action
        rarity_new = tracker.assess_rarity(1, "agent1")

        # New action should have higher rarity
        assert rarity_new > rarity


class TestCoordinationSuccessTracker:
    """Test coordination success tracking."""

    def test_success_tracker_initialization(self):
        """Test success tracker initialization."""
        tracker = CoordinationSuccessTracker()

        assert len(tracker.success_history) == 0
        assert len(tracker.success_patterns) == 0

    def test_coordination_recording(self):
        """Test coordination recording."""
        tracker = CoordinationSuccessTracker()

        # Record successful coordination
        context = {
            "participating_agents": ["agent1", "agent2"],
            "coordination_strategy": "consensus",
        }
        tracker.record_coordination(True, context)

        assert len(tracker.success_history) == 1
        assert tracker.success_history[0] is True
        assert len(tracker.success_patterns) == 1

    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        tracker = CoordinationSuccessTracker()

        # Record mixed results
        contexts = [{"participating_agents": ["agent1"], "coordination_strategy": "simple"}]

        for i in range(10):
            tracker.record_coordination(i % 2 == 0, contexts[0])  # 50% success rate

        success_rate = tracker.get_success_rate()
        assert abs(success_rate - 0.5) < 0.1  # Should be around 50%


class TestDistributedTraining:
    """Test distributed training components."""

    def test_training_node_creation(self):
        """Test training node creation."""
        node = TrainingNode(
            node_id="node1",
            node_type=TrainingNodeType.WORKER,
            rank=1,
            world_size=4,
            gpu_id=0,
        )

        assert node.node_id == "node1"
        assert node.node_type == TrainingNodeType.WORKER
        assert node.rank == 1
        assert node.world_size == 4
        assert node.status == "initialized"

    def test_node_heartbeat(self):
        """Test node heartbeat functionality."""
        node = TrainingNode(
            node_id="node1", node_type=TrainingNodeType.WORKER, rank=0, world_size=1
        )

        # Node should be alive initially
        assert node.is_alive()

        # Update heartbeat
        node.update_heartbeat()
        assert node.is_alive()

        # Simulate old heartbeat
        node.last_heartbeat = time.time() - 60  # 60 seconds ago
        assert not node.is_alive(timeout_seconds=30)

    @patch("torch.distributed.is_available")
    def test_distributed_training_manager_single_node(self, mock_dist_available):
        """Test distributed training manager in single node mode."""
        mock_dist_available.return_value = False

        config = MARLConfig()
        config.distributed_training = False

        node = TrainingNode(
            node_id="node1", node_type=TrainingNodeType.MASTER, rank=0, world_size=1
        )

        manager = DistributedTrainingManager(config, node)

        assert manager.config == config
        assert manager.node == node
        assert not manager.is_initialized

        # Initialize
        manager.initialize_distributed_training()

        assert manager.is_initialized
        assert manager.node.status == "single_node"

    def test_training_status(self):
        """Test training status reporting."""
        config = MARLConfig()
        node = TrainingNode(
            node_id="test_node", node_type=TrainingNodeType.WORKER, rank=0, world_size=1
        )

        manager = DistributedTrainingManager(config, node)
        status = manager.get_training_status()

        assert "node_info" in status
        assert "training_state" in status
        assert "metrics" in status
        assert "node_health" in status

        assert status["node_info"]["node_id"] == "test_node"
        assert status["training_state"]["is_initialized"] is False


class TestDistributedCoordination:
    """Test distributed coordination functionality."""

    def test_coordination_manager_initialization(self):
        """Test coordination manager initialization."""
        config = MARLConfig()
        node = TrainingNode(
            node_id="coord_node",
            node_type=TrainingNodeType.MASTER,
            rank=0,
            world_size=1,
        )

        training_manager = DistributedTrainingManager(config, node)
        coord_manager = DistributedCoordinationManager(training_manager)

        assert coord_manager.training_manager == training_manager
        assert coord_manager.node == node
        assert len(coord_manager.consensus_proposals) == 0

    def test_coordination_proposal(self):
        """Test coordination proposal mechanism."""
        config = MARLConfig()
        node = TrainingNode(
            node_id="coord_node",
            node_type=TrainingNodeType.MASTER,
            rank=0,
            world_size=1,
        )

        training_manager = DistributedTrainingManager(config, node)
        coord_manager = DistributedCoordinationManager(training_manager)

        # Propose coordination action
        action_proposal = {"action": "generate", "strategy": "creative"}
        proposal_id = coord_manager.propose_coordination_action("agent1", action_proposal)

        assert proposal_id in coord_manager.consensus_proposals
        proposal = coord_manager.consensus_proposals[proposal_id]
        assert proposal["agent_id"] == "agent1"
        assert proposal["proposal"] == action_proposal
        assert proposal["status"] == "proposed"

    def test_coordination_voting(self):
        """Test coordination voting mechanism."""
        config = MARLConfig()
        node = TrainingNode(
            node_id="coord_node",
            node_type=TrainingNodeType.MASTER,
            rank=0,
            world_size=2,  # Two nodes for voting
        )

        training_manager = DistributedTrainingManager(config, node)
        coord_manager = DistributedCoordinationManager(training_manager)

        # Propose and vote
        proposal_id = coord_manager.propose_coordination_action("agent1", {"action": "validate"})

        # Vote from both nodes
        coord_manager.vote_on_proposal(proposal_id, "agent1", True)
        coord_manager.vote_on_proposal(proposal_id, "agent2", True)

        proposal = coord_manager.consensus_proposals[proposal_id]
        assert proposal["status"] == "approved"

    def test_coordination_status(self):
        """Test coordination status reporting."""
        config = MARLConfig()
        node = TrainingNode(
            node_id="coord_node",
            node_type=TrainingNodeType.MASTER,
            rank=0,
            world_size=1,
        )

        training_manager = DistributedTrainingManager(config, node)
        coord_manager = DistributedCoordinationManager(training_manager)

        status = coord_manager.get_coordination_status()

        assert "node_id" in status
        assert "global_state_size" in status
        assert "active_proposals" in status
        assert "total_proposals" in status

        assert status["node_id"] == "coord_node"
        assert status["active_proposals"] == 0
        assert status["total_proposals"] == 0
