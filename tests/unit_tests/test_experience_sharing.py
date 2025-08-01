# Standard Library
import time

# Third-Party Library
import numpy as np
import pytest

# SynThesisAI Modules
from core.marl.agents.experience import Experience
from core.marl.config_legacy import ExperienceConfig
from core.marl.learning.experience_sharing import (
    ActionFrequencyTracker,
    CoordinationSuccessTracker,
    ExperienceValue,
    SharedExperienceManager,
    StateNoveltyTracker,
)


class TestExperienceConfig:
    def test_defaults(self):
        config = ExperienceConfig()
        assert config.high_reward_threshold == 0.8
        assert config.novelty_threshold == 0.7
        assert config.sharing_probability == 0.3
        assert config.agent_buffer_size > 0
        assert config.shared_buffer_size > 0


class TestStateNoveltyTracker:
    def test_novelty_scores(self):
        tracker = StateNoveltyTracker(novelty_threshold=0.1)
        state = np.array([1.0, 2.0, 3.0])
        score1 = tracker.assess_novelty(state, "agent1")
        assert score1 == pytest.approx(1.0)
        score2 = tracker.assess_novelty(state, "agent1")
        assert 0.0 <= score2 <= 1.0


class TestActionFrequencyTracker:
    def test_rarity(self):
        tracker = ActionFrequencyTracker()
        for _ in range(3):
            tracker.assess_rarity(1, "agent1")
        rarity = tracker.assess_rarity(1, "agent1")
        assert 0.0 <= rarity <= 1.0


class TestCoordinationSuccessTracker:
    def test_success_rate(self):
        tracker = CoordinationSuccessTracker()
        assert tracker.get_success_rate() == 0.0
        tracker.record_coordination(
            True, {"participating_agents": ["a", "b"], "coordination_strategy": "test"}
        )
        tracker.record_coordination(False, {})
        assert tracker.get_success_rate() == pytest.approx(0.5)


class TestSharedExperienceManager:
    def test_initialization(self):
        config = ExperienceConfig()
        manager = SharedExperienceManager(config)
        assert hasattr(manager, "shared_experiences")
        assert hasattr(manager, "experience_metadata")
        assert isinstance(manager.shared_experiences, dict)

    def test_register_and_share(self):
        config = ExperienceConfig()
        config.sharing_probability = 1.0
        manager = SharedExperienceManager(config)
        # Registration no longer required
        exp = Experience(
            state=np.array([0]),
            action=0,
            reward=0.9,
            next_state=np.array([1]),
            done=False,
        )
        exp_id = manager.share_experience(
            exp, "agent1", ExperienceValue.HIGH_REWARD, 1.0
        )
        assert isinstance(exp_id, str)
        assert exp_id in manager.shared_experiences

    def test_retrieve_for_agent(self):
        config = ExperienceConfig()
        config.sharing_probability = 1.0
        manager = SharedExperienceManager(config)
        # Registration no longer required
        exp = Experience(
            state=np.array([0]),
            action=0,
            reward=0.9,
            next_state=np.array([1]),
            done=False,
        )
        exp_id = manager.share_experience(
            exp, "agent1", ExperienceValue.HIGH_REWARD, 1.0
        )
        samples = manager.get_experiences_for_agent("agent2", max_experiences=1)
        assert len(samples) == 1
        assert samples[0][0] == exp_id

    def test_cleanup_old_experiences(self):
        config = ExperienceConfig()
        manager = SharedExperienceManager(config)
        manager.experience_metadata.setdefault(
            "e1", type("Meta", (), {"creation_time": time.time() - 3600 * 25})
        )
        manager.shared_experiences["e1"] = "dummy"
        removed = manager.cleanup_old_experiences(max_age_hours=24)
        assert removed == 1
