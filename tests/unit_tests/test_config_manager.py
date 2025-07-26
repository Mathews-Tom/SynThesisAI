"""
Unit tests for MARL Configuration Manager.

Tests the comprehensive configuration management capabilities including
loading, saving, validation, and configuration updates.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import yaml

from core.marl.config.config_manager import (
    ConfigurationError,
    MARLConfigManager,
    MARLConfigManagerFactory,
)
from core.marl.config.config_schema import AgentConfig, MARLConfig


class TestMARLConfigManager:
    """Test MARL configuration manager."""

    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary configuration directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def config_manager(self, temp_config_dir):
        """Create configuration manager for testing."""
        return MARLConfigManager(temp_config_dir)

    @pytest.fixture
    def sample_config(self):
        """Create sample configuration for testing."""
        return {
            "name": "test_config",
            "version": "1.0.0",
            "description": "Test configuration",
            "agents": {
                "generator": {
                    "agent_id": "generator",
                    "agent_type": "generator",
                    "state_dim": 128,
                    "action_dim": 10,
                    "network": {
                        "hidden_layers": [64, 32],
                        "activation_function": "relu",
                        "dropout_rate": 0.1,
                        "batch_normalization": True,
                        "weight_initialization": "xavier_uniform",
                        "architecture": "feedforward",
                    },
                    "optimization": {
                        "optimizer_type": "adam",
                        "learning_rate": 0.001,
                        "learning_rate_decay": 0.95,
                        "learning_rate_schedule": "exponential",
                        "weight_decay": 0.0001,
                        "gradient_clipping": 1.0,
                        "momentum": 0.9,
                        "beta1": 0.9,
                        "beta2": 0.999,
                        "epsilon": 1e-8,
                    },
                    "exploration": {
                        "strategy": "epsilon_greedy",
                        "initial_epsilon": 1.0,
                        "final_epsilon": 0.01,
                        "epsilon_decay": 0.995,
                        "epsilon_decay_steps": 10000,
                        "temperature": 1.0,
                        "ucb_c": 2.0,
                    },
                    "replay_buffer": {
                        "capacity": 50000,
                        "batch_size": 32,
                        "min_size_to_sample": 1000,
                        "prioritized_replay": False,
                        "alpha": 0.6,
                        "beta": 0.4,
                        "beta_increment": 0.001,
                    },
                    "gamma": 0.99,
                    "tau": 0.005,
                    "update_frequency": 4,
                    "target_update_frequency": 1000,
                    "double_dqn": True,
                    "dueling_dqn": True,
                    "reward_scaling": 1.0,
                }
            },
            "coordination": {
                "consensus": {
                    "strategy": "weighted_average",
                    "voting_threshold": 0.5,
                    "confidence_threshold": 0.7,
                    "timeout_seconds": 30.0,
                    "max_iterations": 10,
                    "convergence_threshold": 0.01,
                    "expert_weights": {},
                    "adaptive_learning_rate": 0.1,
                },
                "communication": {
                    "message_queue_size": 1000,
                    "message_timeout": 10.0,
                    "max_retries": 3,
                    "retry_delay": 1.0,
                    "compression_enabled": True,
                    "encryption_enabled": False,
                    "heartbeat_interval": 5.0,
                },
                "coordination_timeout": 60.0,
                "max_concurrent_coordinations": 10,
                "conflict_resolution_strategy": "priority_based",
                "deadlock_detection_enabled": True,
                "deadlock_timeout": 120.0,
            },
            "learning": {
                "shared_learning": {
                    "enabled": True,
                    "experience_sharing_rate": 0.1,
                    "experience_buffer_size": 50000,
                    "sharing_strategy": "high_reward",
                    "novelty_threshold": 0.8,
                    "reward_threshold": 0.7,
                    "continuous_learning_enabled": True,
                    "learning_update_interval": 10.0,
                    "performance_window_size": 100,
                    "adaptation_threshold": 0.05,
                },
                "training_enabled": True,
                "evaluation_interval": 1000,
                "save_interval": 10000,
                "max_episodes": 100000,
                "max_steps_per_episode": 1000,
                "performance_tracking_enabled": True,
                "metrics_collection_interval": 1.0,
                "trend_analysis_enabled": True,
            },
            "system": {
                "device": "auto",
                "num_workers": 4,
                "seed": None,
                "deterministic": False,
                "memory_limit_gb": None,
                "gpu_memory_fraction": 0.8,
                "log_level": "INFO",
                "log_to_file": True,
                "log_file_path": "logs/marl.log",
                "metrics_enabled": True,
            },
            "environment": {},
        }

    def test_initialization(self, config_manager, temp_config_dir):
        """Test configuration manager initialization."""
        assert config_manager.config_dir == temp_config_dir
        assert config_manager.config_dir.exists()
        assert isinstance(config_manager.validator, object)
        assert len(config_manager._config_cache) == 0
        assert config_manager._supported_formats == {".json", ".yaml", ".yml"}

    def test_create_default_config(self, config_manager):
        """Test creating default configuration."""
        config = config_manager.create_default_config()

        assert isinstance(config, MARLConfig)
        assert config.name == "default_marl_config"
        assert len(config.agents) == 3  # generator, validator, curriculum
        assert "generator" in config.agents
        assert "validator" in config.agents
        assert "curriculum" in config.agents

    def test_save_config_json(self, config_manager, sample_config, temp_config_dir):
        """Test saving configuration as JSON."""
        config = MARLConfig.from_dict(sample_config)
        config_path = temp_config_dir / "test_config.json"

        success = config_manager.save_config(config, config_path, format="json")

        assert success is True
        assert config_path.exists()

        # Verify content
        with config_path.open("r") as f:
            saved_data = json.load(f)

        assert saved_data["name"] == "test_config"
        assert saved_data["version"] == "1.0.0"

    def test_save_config_yaml(self, config_manager, sample_config, temp_config_dir):
        """Test saving configuration as YAML."""
        config = MARLConfig.from_dict(sample_config)
        config_path = temp_config_dir / "test_config.yaml"

        success = config_manager.save_config(config, config_path, format="yaml")

        assert success is True
        assert config_path.exists()

        # Verify content
        with config_path.open("r") as f:
            saved_data = yaml.safe_load(f)

        assert saved_data["name"] == "test_config"
        assert saved_data["version"] == "1.0.0"

    def test_save_config_with_backup(
        self, config_manager, sample_config, temp_config_dir
    ):
        """Test saving configuration with backup."""
        config = MARLConfig.from_dict(sample_config)
        config_path = temp_config_dir / "test_config.yaml"

        # Create initial file
        config_path.write_text("initial content")

        success = config_manager.save_config(config, config_path, backup=True)

        assert success is True
        assert config_path.exists()

        # Check backup was created
        backup_path = config_path.with_suffix(".yaml.backup")
        assert backup_path.exists()
        assert backup_path.read_text() == "initial content"

    def test_load_config_json(self, config_manager, sample_config, temp_config_dir):
        """Test loading configuration from JSON."""
        config_path = temp_config_dir / "test_config.json"

        # Save sample config
        with config_path.open("w") as f:
            json.dump(sample_config, f)

        loaded_config = config_manager.load_config(config_path)

        assert isinstance(loaded_config, MARLConfig)
        assert loaded_config.name == "test_config"
        assert loaded_config.version == "1.0.0"
        assert len(loaded_config.agents) == 1

    def test_load_config_yaml(self, config_manager, sample_config, temp_config_dir):
        """Test loading configuration from YAML."""
        config_path = temp_config_dir / "test_config.yaml"

        # Save sample config
        with config_path.open("w") as f:
            yaml.dump(sample_config, f)

        loaded_config = config_manager.load_config(config_path)

        assert isinstance(loaded_config, MARLConfig)
        assert loaded_config.name == "test_config"
        assert loaded_config.version == "1.0.0"

    def test_load_config_with_cache(
        self, config_manager, sample_config, temp_config_dir
    ):
        """Test loading configuration with caching."""
        config_path = temp_config_dir / "test_config.json"

        # Save sample config
        with config_path.open("w") as f:
            json.dump(sample_config, f)

        # Load first time
        config1 = config_manager.load_config(config_path, use_cache=True)

        # Load second time (should use cache)
        config2 = config_manager.load_config(config_path, use_cache=True)

        assert config1 is config2  # Same object from cache
        assert len(config_manager._config_cache) == 1

    def test_load_config_file_not_found(self, config_manager, temp_config_dir):
        """Test loading non-existent configuration file."""
        config_path = temp_config_dir / "nonexistent.json"

        with pytest.raises(ConfigurationError, match="Configuration file not found"):
            config_manager.load_config(config_path)

    def test_load_config_unsupported_format(self, config_manager, temp_config_dir):
        """Test loading configuration with unsupported format."""
        config_path = temp_config_dir / "test_config.txt"
        config_path.write_text("some content")

        with pytest.raises(
            ConfigurationError, match="Unsupported configuration format"
        ):
            config_manager.load_config(config_path)

    def test_load_config_invalid_json(self, config_manager, temp_config_dir):
        """Test loading invalid JSON configuration."""
        config_path = temp_config_dir / "invalid.json"
        config_path.write_text("{ invalid json }")

        with pytest.raises(ConfigurationError):
            config_manager.load_config(config_path)

    def test_save_config_unsupported_format(self, config_manager, sample_config):
        """Test saving configuration with unsupported format."""
        config = MARLConfig.from_dict(sample_config)

        with pytest.raises(ConfigurationError, match="Unsupported format"):
            config_manager.save_config(config, "test.txt", format="txt")

    def test_list_configs(self, config_manager, temp_config_dir):
        """Test listing configuration files."""
        # Create test files
        (temp_config_dir / "config1.json").write_text("{}")
        (temp_config_dir / "config2.yaml").write_text("test: value")
        (temp_config_dir / "config3.yml").write_text("test: value")
        (temp_config_dir / "not_config.txt").write_text("text")

        config_files = config_manager.list_configs()

        assert len(config_files) == 3
        assert any(f.name == "config1.json" for f in config_files)
        assert any(f.name == "config2.yaml" for f in config_files)
        assert any(f.name == "config3.yml" for f in config_files)

    def test_validate_config_file(self, config_manager, sample_config, temp_config_dir):
        """Test validating configuration file."""
        config_path = temp_config_dir / "test_config.json"

        # Save valid config
        with config_path.open("w") as f:
            json.dump(sample_config, f)

        result = config_manager.validate_config_file(config_path)

        assert result["valid"] is True
        assert result["config_name"] == "test_config"
        assert result["config_version"] == "1.0.0"
        assert len(result["errors"]) == 0

    def test_validate_config_file_invalid(self, config_manager, temp_config_dir):
        """Test validating invalid configuration file."""
        config_path = temp_config_dir / "invalid_config.json"

        # Save invalid config (missing required fields)
        invalid_config = {"name": "test"}
        with config_path.open("w") as f:
            json.dump(invalid_config, f)

        result = config_manager.validate_config_file(config_path)

        assert result["valid"] is False
        assert len(result["errors"]) > 0

    def test_compare_configs(self, config_manager, sample_config, temp_config_dir):
        """Test comparing two configurations."""
        config1_path = temp_config_dir / "config1.json"
        config2_path = temp_config_dir / "config2.json"

        # Create two similar configs
        config1 = sample_config.copy()
        config2 = sample_config.copy()
        config2["version"] = "1.1.0"  # Different version

        with config1_path.open("w") as f:
            json.dump(config1, f)
        with config2_path.open("w") as f:
            json.dump(config2, f)

        result = config_manager.compare_configs(config1_path, config2_path)

        assert "Version: 1.0.0 vs 1.1.0" in result["differences"]
        assert result["config1_name"] == "test_config"
        assert result["config2_name"] == "test_config"

    def test_update_config(self, config_manager, sample_config):
        """Test updating configuration."""
        config = MARLConfig.from_dict(sample_config)

        updates = {"name": "updated_config", "system": {"num_workers": 8}}

        updated_config = config_manager.update_config(config, updates)

        assert updated_config.name == "updated_config"
        assert updated_config.system.num_workers == 8
        # Original should be unchanged
        assert config.name == "test_config"
        assert config.system.num_workers == 4

    def test_get_config_summary(self, config_manager, sample_config):
        """Test getting configuration summary."""
        config = MARLConfig.from_dict(sample_config)

        summary = config_manager.get_config_summary(config)

        assert summary["name"] == "test_config"
        assert summary["version"] == "1.0.0"
        assert summary["num_agents"] == 1
        assert "generator" in summary["agent_types"]
        assert summary["consensus_strategy"] == "weighted_average"
        assert summary["shared_learning_enabled"] is True
        assert summary["device"] == "auto"
        assert summary["num_workers"] == 4

    def test_export_config_template(
        self, config_manager, sample_config, temp_config_dir
    ):
        """Test exporting configuration as template."""
        config = MARLConfig.from_dict(sample_config)
        template_path = temp_config_dir / "template.yaml"

        success = config_manager.export_config_template(config, template_path)

        assert success is True
        assert template_path.exists()

        # Check template content
        content = template_path.read_text()
        assert "MARL Configuration Template" in content
        assert "Generated from: test_config" in content

    def test_clear_cache(self, config_manager, sample_config, temp_config_dir):
        """Test clearing configuration cache."""
        config_path = temp_config_dir / "test_config.json"

        # Save and load config to populate cache
        with config_path.open("w") as f:
            json.dump(sample_config, f)

        config_manager.load_config(config_path, use_cache=True)
        assert len(config_manager._config_cache) == 1

        # Clear cache
        config_manager.clear_cache()
        assert len(config_manager._config_cache) == 0

    def test_get_cache_info(self, config_manager, sample_config, temp_config_dir):
        """Test getting cache information."""
        config_path = temp_config_dir / "test_config.json"

        # Save and load config to populate cache
        with config_path.open("w") as f:
            json.dump(sample_config, f)

        config_manager.load_config(config_path, use_cache=True)

        cache_info = config_manager.get_cache_info()

        assert cache_info["cached_configs"] == 1
        assert len(cache_info["cache_keys"]) == 1


class TestMARLConfigManagerFactory:
    """Test MARL configuration manager factory."""

    def test_create_default(self):
        """Test creating manager with default settings."""
        manager = MARLConfigManagerFactory.create()

        assert isinstance(manager, MARLConfigManager)
        assert manager.config_dir == Path("configs")

    def test_create_with_config_dir(self):
        """Test creating manager with custom config directory."""
        custom_dir = Path("custom_configs")
        manager = MARLConfigManagerFactory.create(custom_dir)

        assert isinstance(manager, MARLConfigManager)
        assert manager.config_dir == custom_dir

    def test_create_with_validation(self):
        """Test creating manager with custom validation settings."""
        manager = MARLConfigManagerFactory.create_with_validation(
            strict_validation=True
        )

        assert isinstance(manager, MARLConfigManager)
        # Check that strict validation settings were applied
        assert manager.validator._warning_thresholds["high_learning_rate"] == 0.005
