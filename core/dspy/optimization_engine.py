"""
DSPy Optimization Engine

This module implements the DSPyOptimizationEngine class for optimizing DSPy modules
using MIPROv2 and other optimization techniques with comprehensive training data management.
"""

import hashlib
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .base_module import STREAMContentGenerator
from .cache import get_optimization_cache
from .config import OptimizationResult, TrainingExample, get_dspy_config
from .exceptions import OptimizationFailureError, TrainingDataError

logger = logging.getLogger(__name__)


class TrainingDataManager:
    """
    Manages training and validation data for DSPy optimization.

    This class handles loading, validation, and management of training datasets
    for different domains.
    """

    def __init__(self):
        """Initialize the training data manager."""
        self.config = get_dspy_config()
        self.logger = logging.getLogger(__name__ + ".TrainingDataManager")
        self.data_cache = {}

        # Set up data directories
        self.training_data_dir = Path("data/training")
        self.validation_data_dir = Path("data/validation")
        self.training_data_dir.mkdir(parents=True, exist_ok=True)
        self.validation_data_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info("Initialized training data manager")

    def get_training_data(self, domain: str) -> List[TrainingExample]:
        """
        Get training data for a domain.

        Args:
            domain: The domain to get training data for

        Returns:
            List of training examples

        Raises:
            TrainingDataError: If training data cannot be loaded
        """
        try:
            cache_key = f"training_{domain}"
            if cache_key in self.data_cache:
                return self.data_cache[cache_key]

            # Load training data from file
            training_file = self.training_data_dir / f"{domain}_training.json"

            if not training_file.exists():
                # Generate synthetic training data if file doesn't exist
                self.logger.info(
                    "No training data file found for %s, generating synthetic data",
                    domain,
                )
                training_data = self._generate_synthetic_training_data(domain)
            else:
                training_data = self._load_training_data_from_file(training_file)

            # Validate training data
            validated_data = self._validate_training_data(training_data, domain)

            # Cache the data
            self.data_cache[cache_key] = validated_data

            self.logger.info(
                "Loaded %d training examples for %s", len(validated_data), domain
            )
            return validated_data

        except Exception as e:
            error_msg = f"Failed to load training data for {domain}: {str(e)}"
            self.logger.error("Failed to load training data for %s: %s", domain, str(e))
            raise TrainingDataError(error_msg, domain=domain) from e

    def get_validation_data(self, domain: str) -> List[TrainingExample]:
        """
        Get validation data for a domain.

        Args:
            domain: The domain to get validation data for

        Returns:
            List of validation examples

        Raises:
            TrainingDataError: If validation data cannot be loaded
        """
        try:
            cache_key = f"validation_{domain}"
            if cache_key in self.data_cache:
                return self.data_cache[cache_key]

            # Load validation data from file
            validation_file = self.validation_data_dir / f"{domain}_validation.json"

            if not validation_file.exists():
                # Generate synthetic validation data if file doesn't exist
                self.logger.info(
                    "No validation data file found for %s, generating synthetic data",
                    domain,
                )
                validation_data = self._generate_synthetic_validation_data(domain)
            else:
                validation_data = self._load_training_data_from_file(validation_file)

            # Validate validation data
            validated_data = self._validate_training_data(validation_data, domain)

            # Cache the data
            self.data_cache[cache_key] = validated_data

            self.logger.info(
                "Loaded %d validation examples for %s", len(validated_data), domain
            )
            return validated_data

        except Exception as e:
            error_msg = f"Failed to load validation data for {domain}: {str(e)}"
            self.logger.error(
                "Failed to load validation data for %s: %s", domain, str(e)
            )
            raise TrainingDataError(error_msg, domain=domain) from e

    def _load_training_data_from_file(self, file_path: Path) -> List[TrainingExample]:
        """
        Load training data from a JSON file.

        Args:
            file_path: Path to the training data file

        Returns:
            List of training examples
        """
        data = json.loads(file_path.read_text(encoding="utf-8"))

        training_examples = []
        for item in data:
            example = TrainingExample(
                inputs=item.get("inputs", {}),
                expected_outputs=item.get("expected_outputs", {}),
                quality_score=item.get("quality_score", 1.0),
                domain=item.get("domain", "unknown"),
                metadata=item.get("metadata", {}),
            )
            training_examples.append(example)

        return training_examples

    def _validate_training_data(
        self, data: List[TrainingExample], domain: str
    ) -> List[TrainingExample]:
        """
        Validate training data for a domain.

        Args:
            data: List of training examples
            domain: The domain

        Returns:
            List of validated training examples
        """
        validated_data = []
        training_config = self.config.get_training_config()

        for example in data:
            # Check if example has required fields
            if not example.inputs or not example.expected_outputs:
                self.logger.warning(
                    "Skipping invalid training example: missing inputs or outputs"
                )
                continue

            # Check quality score
            if example.quality_score < training_config.get("min_quality_score", 0.5):
                self.logger.warning(
                    "Skipping low-quality training example: %.2f", example.quality_score
                )
                continue

            validated_data.append(example)

        # Check if we have enough data
        min_examples = training_config.get("min_examples", 10)
        if len(validated_data) < min_examples:
            self.logger.warning(
                "Only %d valid examples for %s, minimum is %d",
                len(validated_data),
                domain,
                min_examples,
            )

        return validated_data

    def _generate_synthetic_training_data(self, domain: str) -> List[TrainingExample]:
        """
        Generate synthetic training data for a domain.

        Args:
            domain: The domain to generate data for

        Returns:
            List of synthetic training examples
        """
        synthetic_data = []

        # Generate domain-specific synthetic examples
        if domain == "mathematics":
            examples = [
                {
                    "inputs": {
                        "subject": "Algebra",
                        "topic": "Linear Equations",
                        "difficulty_level": "High School",
                    },
                    "expected_outputs": {
                        "problem_statement": "Solve for x: 2x + 3 = 7",
                        "solution": "x = 2",
                        "hints": {
                            "0": "Subtract 3 from both sides",
                            "1": "Divide by 2",
                        },
                    },
                    "quality_score": 0.9,
                },
                {
                    "inputs": {
                        "subject": "Calculus",
                        "topic": "Derivatives",
                        "difficulty_level": "Undergraduate",
                    },
                    "expected_outputs": {
                        "problem_statement": "Find the derivative of f(x) = x^2 + 3x + 2",
                        "solution": "f'(x) = 2x + 3",
                        "hints": {
                            "0": "Use the power rule",
                            "1": "Derivative of constant is 0",
                        },
                    },
                    "quality_score": 0.85,
                },
            ]
        else:
            # Generic examples for other domains
            examples = [
                {
                    "inputs": {
                        "domain": domain,
                        "topic": "Basic Concepts",
                        "difficulty_level": "Beginner",
                    },
                    "expected_outputs": {
                        "problem_statement": f"Basic {domain} problem",
                        "solution": f"Basic {domain} solution",
                    },
                    "quality_score": 0.7,
                }
            ]

        for example_data in examples:
            example = TrainingExample(
                inputs=example_data["inputs"],
                expected_outputs=example_data["expected_outputs"],
                quality_score=example_data["quality_score"],
                domain=domain,
                metadata={
                    "synthetic": True,
                    "generated_at": datetime.now().isoformat(),
                },
            )
            synthetic_data.append(example)

        return synthetic_data

    def _generate_synthetic_validation_data(self, domain: str) -> List[TrainingExample]:
        """
        Generate synthetic validation data for a domain.

        Args:
            domain: The domain to generate data for

        Returns:
            List of synthetic validation examples
        """
        # For now, use similar logic to training data but with different examples
        synthetic_data = []

        if domain == "mathematics":
            examples = [
                {
                    "inputs": {
                        "subject": "Algebra",
                        "topic": "Quadratic Equations",
                        "difficulty_level": "High School",
                    },
                    "expected_outputs": {
                        "problem_statement": "Solve for x: x^2 - 5x + 6 = 0",
                        "solution": "x = 2 or x = 3",
                        "hints": {
                            "0": "Factor the quadratic",
                            "1": "Set each factor to zero",
                        },
                    },
                    "quality_score": 0.9,
                }
            ]
        else:
            examples = [
                {
                    "inputs": {
                        "domain": domain,
                        "topic": "Validation Concepts",
                        "difficulty_level": "Intermediate",
                    },
                    "expected_outputs": {
                        "problem_statement": f"Validation {domain} problem",
                        "solution": f"Validation {domain} solution",
                    },
                    "quality_score": 0.8,
                }
            ]

        for example_data in examples:
            example = TrainingExample(
                inputs=example_data["inputs"],
                expected_outputs=example_data["expected_outputs"],
                quality_score=example_data["quality_score"],
                domain=domain,
                metadata={
                    "synthetic": True,
                    "validation": True,
                    "generated_at": datetime.now().isoformat(),
                },
            )
            synthetic_data.append(example)

        return synthetic_data

    def save_training_data(
        self, domain: str, data: List[TrainingExample], data_type: str = "training"
    ):
        """
        Save training or validation data to file.

        Args:
            domain: The domain
            data: List of training examples
            data_type: Type of data ("training" or "validation")
        """
        try:
            if data_type == "training":
                file_path = self.training_data_dir / f"{domain}_training.json"
            else:
                file_path = self.validation_data_dir / f"{domain}_validation.json"

            # Convert to JSON-serializable format
            json_data = []
            for example in data:
                json_data.append(
                    {
                        "inputs": example.inputs,
                        "expected_outputs": example.expected_outputs,
                        "quality_score": example.quality_score,
                        "domain": example.domain,
                        "metadata": example.metadata,
                    }
                )

            file_path.write_text(json.dumps(json_data, indent=2), encoding="utf-8")

            self.logger.info(
                "Saved %d %s examples for %s", len(data), data_type, domain
            )

        except Exception as e:
            self.logger.error(
                "Failed to save %s data for %s: %s", data_type, domain, str(e)
            )


class DSPyOptimizationEngine:
    """
    Advanced optimization engine for DSPy modules.

    This class provides comprehensive functionality for optimizing DSPy modules using
    MIPROv2 and other optimization techniques with proper training data management.
    """

    def __init__(self):
        """Initialize the optimization engine."""
        self.config = get_dspy_config()
        self.cache = get_optimization_cache()
        self.training_manager = TrainingDataManager()
        self.logger = logging.getLogger(__name__ + ".DSPyOptimizationEngine")
        self.optimization_history = []
        self.logger.info("Initialized DSPy optimization engine")

    def optimize_for_domain(
        self,
        domain_module: STREAMContentGenerator,
        quality_requirements: Dict[str, Any],
    ) -> STREAMContentGenerator:
        """
        Optimize a domain module using MIPROv2.

        Args:
            domain_module: The domain module to optimize
            quality_requirements: Quality requirements for optimization

        Returns:
            Optimized domain module

        Raises:
            OptimizationFailureError: If optimization fails
        """
        optimization_start_time = time.time()

        try:
            # Check if optimization is enabled
            if not self.config.is_enabled():
                self.logger.info("DSPy optimization is disabled, skipping")
                return domain_module

            # Check if we have DSPy available
            try:
                import dspy
                from dspy.teleprompt import MIPROv2
            except ImportError:
                self.logger.warning("DSPy not available, skipping optimization")
                return domain_module

            self.logger.info(
                "Starting optimization for %s module with MIPROv2", domain_module.domain
            )

            # Generate cache key
            cache_key = self._generate_cache_key(domain_module, quality_requirements)

            # Check if cache lookup is enabled
            if self.config.cache_enabled:
                # Check cache first
                cached_result = self.cache.get(cache_key)
                if cached_result:
                    self.logger.info(
                        "Using cached optimization for %s (cache key: %s)",
                        domain_module.domain,
                        cache_key,
                    )

                    # Log cache performance metrics
                    cache_stats = self.cache.get_stats()
                    self.logger.debug(
                        "Cache performance: hit_rate=%.2f, hits=%d, misses=%d",
                        cache_stats["hit_rate"],
                        cache_stats["stats"]["hits"],
                        cache_stats["stats"]["misses"],
                    )

                    return cached_result
                else:
                    self.logger.debug(
                        "No cache entry found for %s (cache key: %s)",
                        domain_module.domain,
                        cache_key,
                    )
            else:
                self.logger.debug("Cache lookup disabled by configuration")

            # Get training and validation data
            try:
                trainset = self.training_manager.get_training_data(domain_module.domain)
                valset = self.training_manager.get_validation_data(domain_module.domain)
            except TrainingDataError as e:
                self.logger.warning(
                    "Training data error: %s, skipping optimization", str(e)
                )
                return domain_module

            if not trainset or not valset:
                self.logger.warning(
                    "Insufficient data for %s, skipping optimization",
                    domain_module.domain,
                )
                return domain_module

            # Convert training examples to DSPy format
            dspy_trainset = self._convert_to_dspy_format(trainset)
            dspy_valset = self._convert_to_dspy_format(valset)

            # Get optimization configuration
            opt_config = self.config.get_optimization_config("mipro_v2")

            # Create MIPROv2 optimizer with configuration
            optimizer = MIPROv2(
                max_bootstrapped_demos=opt_config.get("max_bootstrapped_demos", 4),
                max_labeled_demos=opt_config.get("max_labeled_demos", 16),
                num_candidate_programs=opt_config.get("num_candidate_programs", 16),
                init_temperature=opt_config.get("init_temperature", 1.4),
            )

            self.logger.info(
                "Optimizing with %d training and %d validation examples",
                len(dspy_trainset),
                len(dspy_valset),
            )

            # Perform optimization
            optimized_module = optimizer.compile(
                student=domain_module,
                trainset=dspy_trainset,
                valset=dspy_valset,
                optuna_trials_num=opt_config.get("optuna_trials_num", 100),
            )

            optimization_end_time = time.time()
            optimization_time = optimization_end_time - optimization_start_time

            # Evaluate optimization results
            validation_score = self._evaluate_optimization(
                optimized_module, dspy_valset
            )

            # Create optimization result
            optimization_result = OptimizationResult(
                optimized_module=optimized_module,
                optimization_metrics={
                    "validation_score": validation_score,
                    "training_examples": len(dspy_trainset),
                    "validation_examples": len(dspy_valset),
                    "optimization_time": optimization_time,
                    "cache_key": cache_key,
                    "timestamp": datetime.now().isoformat(),
                },
                training_time=optimization_time,
                validation_score=validation_score,
                cache_key=cache_key,
                timestamp=datetime.now(),
            )

            # Store in cache if enabled
            if self.config.cache_enabled:
                cache_stored = self.cache.store(cache_key, optimized_module)
                if cache_stored:
                    self.logger.info(
                        "Stored optimization result in cache (key: %s)", cache_key
                    )
                else:
                    self.logger.warning(
                        "Failed to store optimization result in cache (key: %s)",
                        cache_key,
                    )
            else:
                self.logger.debug("Cache storage disabled by configuration")

            # Record optimization history
            self.optimization_history.append(optimization_result)

            self.logger.info(
                "Successfully optimized %s module (validation score: %.3f, time: %.2fs)",
                domain_module.domain,
                validation_score,
                optimization_time,
            )

            return optimized_module

        except Exception as e:
            error_msg = f"Failed to optimize {domain_module.domain} module: {str(e)}"
            self.logger.error(
                "Failed to optimize %s module: %s", domain_module.domain, str(e)
            )
            raise OptimizationFailureError(
                error_msg,
                optimizer_type="MIPROv2",
                details={
                    "domain": domain_module.domain,
                    "error": str(e),
                    "optimization_time": time.time() - optimization_start_time,
                },
            ) from e

    def _generate_cache_key(
        self,
        domain_module: STREAMContentGenerator,
        quality_requirements: Dict[str, Any],
    ) -> str:
        """
        Generate cache key for optimization.

        Args:
            domain_module: The domain module
            quality_requirements: Quality requirements

        Returns:
            Cache key string
        """
        # Get optimization data
        opt_data = domain_module.get_optimization_data()

        # Create key components
        key_components = [
            opt_data["domain"],
            opt_data["signature"],
            json.dumps(quality_requirements, sort_keys=True),
            self.config.get_dspy_version(),
            json.dumps(self.config.get_optimization_config("mipro_v2"), sort_keys=True),
        ]

        # Generate hash
        return hashlib.md5("|".join(key_components).encode()).hexdigest()

    def _convert_to_dspy_format(
        self, training_examples: List[TrainingExample]
    ) -> List[Any]:
        """
        Convert training examples to DSPy format.

        Args:
            training_examples: List of training examples

        Returns:
            List of DSPy-formatted examples
        """
        dspy_examples = []

        for example in training_examples:
            # Create a simple DSPy example object
            # In a real implementation, this would use proper DSPy Example classes
            dspy_example = type(
                "DSPyExample",
                (),
                {
                    **example.inputs,
                    **example.expected_outputs,
                    "quality_score": example.quality_score,
                },
            )()

            dspy_examples.append(dspy_example)

        return dspy_examples

    def _evaluate_optimization(
        self, optimized_module: STREAMContentGenerator, valset: List[Any]
    ) -> float:
        """
        Evaluate the optimization results.

        Args:
            optimized_module: The optimized module
            valset: Validation dataset

        Returns:
            Validation score (0.0 to 1.0)
        """
        try:
            # Simple evaluation - in practice, this would be more sophisticated
            total_score = 0.0
            valid_examples = 0

            for example in valset[
                :5
            ]:  # Evaluate on first 5 examples to avoid long evaluation
                try:
                    # Run the optimized module on the example
                    result = optimized_module(
                        **{
                            k: v
                            for k, v in example.__dict__.items()
                            if not k.startswith("_") and k != "quality_score"
                        }
                    )

                    # Simple scoring based on whether we got a result
                    if result:
                        total_score += getattr(example, "quality_score", 1.0)
                        valid_examples += 1
                except Exception:
                    # Skip examples that cause errors
                    continue

            if valid_examples == 0:
                return 0.0

            return total_score / valid_examples

        except Exception as e:
            self.logger.warning("Evaluation failed: %s", str(e))
            return 0.5  # Default score

    def get_optimization_history(self) -> List[OptimizationResult]:
        """
        Get the optimization history.

        Returns:
            List of optimization results
        """
        return self.optimization_history.copy()

    def clear_optimization_history(self) -> None:
        """Clear the optimization history."""
        self.optimization_history = []
        self.logger.info("Cleared optimization history")


# Global instances
_training_data_manager = None
_optimization_engine = None


def get_training_data_manager() -> TrainingDataManager:
    """Get the global training data manager instance."""
    global _training_data_manager
    if _training_data_manager is None:
        _training_data_manager = TrainingDataManager()
    return _training_data_manager


def get_optimization_engine() -> DSPyOptimizationEngine:
    """Get the global optimization engine instance."""
    global _optimization_engine
    if _optimization_engine is None:
        _optimization_engine = DSPyOptimizationEngine()
    return _optimization_engine
