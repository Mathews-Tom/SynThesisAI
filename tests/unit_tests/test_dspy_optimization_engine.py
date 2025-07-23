"""
Unit tests for DSPy Optimization Engine.

These tests verify the functionality of the DSPyOptimizationEngine and
TrainingDataManager classes.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.dspy.config import TrainingExample
from core.dspy.exceptions import OptimizationFailureError, TrainingDataError
from core.dspy.optimization_engine import DSPyOptimizationEngine, TrainingDataManager


class TestTrainingDataManager:
    """Test TrainingDataManager functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.manager = TrainingDataManager()
        # Override data directories for testing
        self.manager.training_data_dir = Path(self.temp_dir.name) / "training"
        self.manager.validation_data_dir = Path(self.temp_dir.name) / "validation"
        self.manager.training_data_dir.mkdir(parents=True, exist_ok=True)
        self.manager.validation_data_dir.mkdir(parents=True, exist_ok=True)

    def teardown_method(self):
        """Clean up test environment."""
        self.temp_dir.cleanup()

    def test_initialization(self):
        """Test TrainingDataManager initialization."""
        manager = TrainingDataManager()
        assert manager.config is not None
        assert manager.data_cache == {}
        assert manager.training_data_dir.exists()
        assert manager.validation_data_dir.exists()

    def test_generate_synthetic_training_data(self):
        """Test generating synthetic training data."""
        # Test mathematics domain
        math_data = self.manager._generate_synthetic_training_data("mathematics")
        assert len(math_data) >= 2
        assert all(isinstance(example, TrainingExample) for example in math_data)
        assert all(example.domain == "mathematics" for example in math_data)
        assert all(example.quality_score > 0 for example in math_data)

        # Test generic domain
        generic_data = self.manager._generate_synthetic_training_data("physics")
        assert len(generic_data) >= 1
        assert all(isinstance(example, TrainingExample) for example in generic_data)
        assert all(example.domain == "physics" for example in generic_data)

    def test_generate_synthetic_validation_data(self):
        """Test generating synthetic validation data."""
        # Test mathematics domain
        math_data = self.manager._generate_synthetic_validation_data("mathematics")
        assert len(math_data) >= 1
        assert all(isinstance(example, TrainingExample) for example in math_data)
        assert all(example.domain == "mathematics" for example in math_data)
        assert all(example.metadata.get("validation") is True for example in math_data)

    def test_save_and_load_training_data(self):
        """Test saving and loading training data."""
        # Create test data
        test_data = [
            TrainingExample(
                inputs={"subject": "Mathematics", "topic": "Algebra"},
                expected_outputs={
                    "problem": "Test problem",
                    "solution": "Test solution",
                },
                quality_score=0.9,
                domain="mathematics",
                metadata={"test": True},
            )
        ]

        # Save data
        self.manager.save_training_data("mathematics", test_data, "training")

        # Verify file was created
        training_file = self.manager.training_data_dir / "mathematics_training.json"
        assert training_file.exists()

        # Load data
        loaded_data = self.manager._load_training_data_from_file(training_file)
        assert len(loaded_data) == 1
        assert loaded_data[0].inputs == test_data[0].inputs
        assert loaded_data[0].expected_outputs == test_data[0].expected_outputs
        assert loaded_data[0].quality_score == test_data[0].quality_score

    @patch("core.dspy.optimization_engine.get_dspy_config")
    def test_get_training_data_with_file(self, mock_get_config):
        """Test getting training data when file exists."""
        # Setup mock config
        mock_config = MagicMock()
        mock_config.get_training_config.return_value = {
            "min_examples": 1,
            "min_quality_score": 0.5,
        }
        mock_get_config.return_value = mock_config

        # Create test data file
        test_data = [
            {
                "inputs": {"subject": "Mathematics"},
                "expected_outputs": {"problem": "Test problem"},
                "quality_score": 0.8,
                "domain": "mathematics",
                "metadata": {},
            }
        ]

        training_file = self.manager.training_data_dir / "mathematics_training.json"
        with open(training_file, "w") as f:
            json.dump(test_data, f)

        # Get training data
        training_data = self.manager.get_training_data("mathematics")
        assert len(training_data) == 1
        assert training_data[0].quality_score == 0.8

    @patch("core.dspy.optimization_engine.get_dspy_config")
    def test_get_training_data_synthetic(self, mock_get_config):
        """Test getting training data when no file exists (synthetic)."""
        # Setup mock config
        mock_config = MagicMock()
        mock_config.get_training_config.return_value = {
            "min_examples": 1,
            "min_quality_score": 0.5,
        }
        mock_get_config.return_value = mock_config

        # Get training data (should generate synthetic)
        training_data = self.manager.get_training_data("mathematics")
        assert (
            len(training_data) >= 2
        )  # Mathematics should have at least 2 synthetic examples
        assert all(
            example.metadata.get("synthetic") is True for example in training_data
        )

    def test_validate_training_data(self):
        """Test training data validation."""
        # Create test data with mixed quality
        test_data = [
            TrainingExample(
                inputs={"subject": "Mathematics"},
                expected_outputs={"problem": "Good problem"},
                quality_score=0.9,
                domain="mathematics",
                metadata={},
            ),
            TrainingExample(
                inputs={},  # Invalid - empty inputs
                expected_outputs={"problem": "Bad problem"},
                quality_score=0.8,
                domain="mathematics",
                metadata={},
            ),
            TrainingExample(
                inputs={"subject": "Mathematics"},
                expected_outputs={"problem": "Low quality problem"},
                quality_score=0.3,  # Low quality score
                domain="mathematics",
                metadata={},
            ),
        ]

        # Mock config
        self.manager.config = MagicMock()
        self.manager.config.get_training_config.return_value = {
            "min_quality_score": 0.5
        }

        # Validate data
        validated_data = self.manager._validate_training_data(test_data, "mathematics")

        # Should only have 1 valid example
        assert len(validated_data) == 1
        assert validated_data[0].quality_score == 0.9


class TestDSPyOptimizationEngine:
    """Test DSPyOptimizationEngine functionality."""

    @patch("core.dspy.optimization_engine.get_optimization_cache")
    @patch("core.dspy.optimization_engine.get_dspy_config")
    def test_initialization(self, mock_get_config, mock_get_cache):
        """Test DSPyOptimizationEngine initialization."""
        # Setup mocks
        mock_config = MagicMock()
        mock_cache = MagicMock()
        mock_get_config.return_value = mock_config
        mock_get_cache.return_value = mock_cache

        # Create engine
        engine = DSPyOptimizationEngine()

        assert engine.config == mock_config
        assert engine.cache == mock_cache
        assert isinstance(engine.training_manager, TrainingDataManager)
        assert engine.optimization_history == []

    @patch("core.dspy.optimization_engine.get_optimization_cache")
    @patch("core.dspy.optimization_engine.get_dspy_config")
    def test_optimize_for_domain_disabled(self, mock_get_config, mock_get_cache):
        """Test optimization when disabled."""
        # Setup mocks
        mock_config = MagicMock()
        mock_config.is_enabled.return_value = False
        mock_get_config.return_value = mock_config
        mock_get_cache.return_value = MagicMock()

        # Create engine and mock module
        engine = DSPyOptimizationEngine()
        mock_module = MagicMock()
        mock_module.domain = "mathematics"

        # Test optimization (should skip)
        result = engine.optimize_for_domain(mock_module, {})
        assert result == mock_module  # Should return original module

    @patch("core.dspy.optimization_engine.get_optimization_cache")
    @patch("core.dspy.optimization_engine.get_dspy_config")
    def test_optimize_for_domain_no_dspy(self, mock_get_config, mock_get_cache):
        """Test optimization when DSPy is not available."""
        # Setup mocks
        mock_config = MagicMock()
        mock_config.is_enabled.return_value = True
        mock_get_config.return_value = mock_config
        mock_get_cache.return_value = MagicMock()

        # Create engine and mock module
        engine = DSPyOptimizationEngine()
        mock_module = MagicMock()
        mock_module.domain = "mathematics"

        # Mock DSPy import failure
        with patch("builtins.__import__", side_effect=ImportError("DSPy not found")):
            result = engine.optimize_for_domain(mock_module, {})
            assert result == mock_module  # Should return original module

    @patch("core.dspy.optimization_engine.get_optimization_cache")
    @patch("core.dspy.optimization_engine.get_dspy_config")
    def test_optimize_for_domain_cached(self, mock_get_config, mock_get_cache):
        """Test optimization with cached result."""
        # Setup mocks
        mock_config = MagicMock()
        mock_config.is_enabled.return_value = True
        mock_config.get_dspy_version.return_value = "1.0.0"
        mock_config.get_optimization_config.return_value = {"param": "value"}
        mock_cache = MagicMock()
        mock_cached_module = MagicMock()
        mock_cache.get.return_value = mock_cached_module
        mock_get_config.return_value = mock_config
        mock_get_cache.return_value = mock_cache

        # Create engine and mock module
        engine = DSPyOptimizationEngine()
        mock_module = MagicMock()
        mock_module.domain = "mathematics"
        mock_module.get_optimization_data.return_value = {
            "domain": "mathematics",
            "signature": "test_signature",
        }

        # Test optimization (should use cache)
        result = engine.optimize_for_domain(mock_module, {})
        assert result == mock_cached_module
        mock_cache.get.assert_called_once()

    def test_generate_cache_key(self):
        """Test cache key generation."""
        # Setup mocks
        engine = DSPyOptimizationEngine()
        engine.config = MagicMock()
        engine.config.get_dspy_version.return_value = "1.0.0"
        engine.config.get_optimization_config.return_value = {"param": "value"}

        mock_module = MagicMock()
        mock_module.get_optimization_data.return_value = {
            "domain": "mathematics",
            "signature": "test_signature",
        }

        quality_requirements = {"min_quality": 0.8}

        # Generate cache key
        cache_key = engine._generate_cache_key(mock_module, quality_requirements)

        assert isinstance(cache_key, str)
        assert len(cache_key) == 32  # MD5 hash length

        # Same inputs should generate same key
        cache_key2 = engine._generate_cache_key(mock_module, quality_requirements)
        assert cache_key == cache_key2

    def test_convert_to_dspy_format(self):
        """Test converting training examples to DSPy format."""
        engine = DSPyOptimizationEngine()

        # Create test training examples
        training_examples = [
            TrainingExample(
                inputs={"subject": "Mathematics"},
                expected_outputs={"problem": "Test problem"},
                quality_score=0.9,
                domain="mathematics",
                metadata={},
            )
        ]

        # Convert to DSPy format
        dspy_examples = engine._convert_to_dspy_format(training_examples)

        assert len(dspy_examples) == 1
        assert hasattr(dspy_examples[0], "subject")
        assert hasattr(dspy_examples[0], "problem")
        assert hasattr(dspy_examples[0], "quality_score")
        assert dspy_examples[0].subject == "Mathematics"
        assert dspy_examples[0].problem == "Test problem"
        assert dspy_examples[0].quality_score == 0.9

    def test_evaluate_optimization(self):
        """Test optimization evaluation."""
        engine = DSPyOptimizationEngine()

        # Create mock optimized module
        mock_module = MagicMock()
        mock_module.return_value = {"result": "test"}

        # Create mock validation set
        mock_example = MagicMock()
        mock_example.__dict__ = {"input1": "value1", "quality_score": 0.8}
        valset = [mock_example]

        # Evaluate optimization
        score = engine._evaluate_optimization(mock_module, valset)

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_optimization_history(self):
        """Test optimization history management."""
        engine = DSPyOptimizationEngine()

        # Initially empty
        assert engine.get_optimization_history() == []

        # Add some history (simulate)
        from datetime import datetime

        from core.dspy.config import OptimizationResult

        result = OptimizationResult(
            optimized_module=MagicMock(),
            optimization_metrics={"score": 0.8},
            training_time=10.0,
            validation_score=0.8,
            cache_key="test_key",
            timestamp=datetime.now(),
        )

        engine.optimization_history.append(result)

        # Check history
        history = engine.get_optimization_history()
        assert len(history) == 1
        assert history[0] == result

        # Clear history
        engine.clear_optimization_history()
        assert engine.get_optimization_history() == []


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
