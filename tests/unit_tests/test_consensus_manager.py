"""Unit tests for Consensus mechanisms."""

import asyncio
import time
from unittest.mock import Mock, patch

import pytest

from core.marl.config import CoordinationConfig
from core.marl.config.config_schema import ConsensusConfig
from core.marl.coordination.consensus_mechanism import ConsensusMechanism


class TestConsensusMechanism:
    """Test ConsensusMechanism functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.coordination_config = CoordinationConfig()
        self.mechanism = ConsensusMechanism(self.coordination_config)

    def test_mechanism_initialization(self):
        """Test mechanism initialization."""
        assert self.mechanism.config == self.coordination_config
        assert hasattr(self.mechanism, "consensus_strategies")
        assert len(self.mechanism.consensus_strategies) > 0

    def test_consensus_strategies_available(self):
        """Test that consensus strategies are available."""
        expected_strategies = [
            "weighted_average",
            "majority_vote",
            "expert_priority",
            "adaptive_consensus",
        ]

        for strategy in expected_strategies:
            assert strategy in self.mechanism.consensus_strategies

    def test_consensus_history_tracking(self):
        """Test consensus history tracking."""
        assert hasattr(self.mechanism, "consensus_history")
        assert isinstance(self.mechanism.consensus_history, list)
        assert len(self.mechanism.consensus_history) == 0  # Initially empty

    def test_mechanism_has_logger(self):
        """Test that mechanism has proper logging."""
        assert hasattr(self.mechanism, "logger")
        assert self.mechanism.logger is not None


class TestConsensusConfig:
    """Test ConsensusConfig functionality."""

    def test_config_creation(self):
        """Test config creation with default values."""
        config = ConsensusConfig()

        assert config.voting_threshold == 0.5  # default
        assert config.timeout_seconds == 30.0  # default
        assert config.confidence_threshold == 0.7  # default

    def test_config_validation(self):
        """Test config validation."""
        # Valid config
        config = ConsensusConfig(voting_threshold=0.7, timeout_seconds=5.0)
        assert 0 < config.voting_threshold <= 1.0
        assert config.timeout_seconds > 0

        # Test that validation works in __post_init__
        try:
            config.__post_init__()
        except ValueError:
            pytest.fail("Valid config should not raise ValueError")

    def test_config_serialization(self):
        """Test config serialization."""
        config = ConsensusConfig(voting_threshold=0.7, timeout_seconds=5.0)

        # Test that config can be converted to dict
        config_dict = config.__dict__
        assert isinstance(config_dict, dict)
        assert config_dict["voting_threshold"] == 0.7
        assert config_dict["timeout_seconds"] == 5.0


@pytest.mark.integration
class TestConsensusIntegration:
    """Integration tests for consensus mechanisms."""

    def test_consensus_mechanism_creation(self):
        """Test creating consensus mechanism with coordination config."""
        coordination_config = CoordinationConfig()
        mechanism = ConsensusMechanism(coordination_config)

        assert mechanism is not None
        assert mechanism.config == coordination_config

    def test_consensus_strategies_callable(self):
        """Test that consensus strategies are callable."""
        coordination_config = CoordinationConfig()
        mechanism = ConsensusMechanism(coordination_config)

        # Test that strategies are callable functions
        for strategy_name, strategy_func in mechanism.consensus_strategies.items():
            assert callable(strategy_func), (
                f"Strategy {strategy_name} should be callable"
            )

    def test_mechanism_with_different_configs(self):
        """Test mechanism with different coordination configurations."""
        # Test with default config
        default_config = CoordinationConfig()
        mechanism1 = ConsensusMechanism(default_config)
        assert mechanism1 is not None

        # Test with custom config
        custom_config = CoordinationConfig()
        custom_config.consensus_strategy = "majority_vote"
        mechanism2 = ConsensusMechanism(custom_config)
        assert mechanism2 is not None
        assert mechanism2.config.consensus_strategy == "majority_vote"


if __name__ == "__main__":
    pytest.main([__file__])
