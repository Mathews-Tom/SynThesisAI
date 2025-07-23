"""
Unit tests for DSPy quality assessment.

Tests the quality assessment functionality for DSPy optimization results.
"""

import unittest
from datetime import datetime
from unittest import mock

import pytest

from core.dspy.base_module import STREAMContentGenerator
from core.dspy.config import OptimizationResult
from core.dspy.quality_assessment import (
    QualityAssessor,
    QualityMetric,
    TrainingTimeMetric,
    ValidationAccuracyMetric,
    get_quality_assessor,
)


class TestQualityMetrics(unittest.TestCase):
    """Test quality metrics."""

    def setUp(self):
        """Set up test fixtures."""
        self.module = mock.MagicMock(spec=STREAMContentGenerator)
        self.module.domain = "mathematics"

        self.result = OptimizationResult(
            optimized_module=self.module,
            optimization_metrics={"accuracy": 0.95},
            training_time=10.5,
            validation_score=0.92,
            cache_key="test_cache_key",
            timestamp=datetime.now(),
        )

    def test_validation_accuracy_metric(self):
        """Test validation accuracy metric."""
        metric = ValidationAccuracyMetric(weight=1.0)

        # Test calculation
        value = metric.calculate(self.module, self.result)
        self.assertEqual(value, 0.92)

        # Test with different validation score
        result_with_lower_score = OptimizationResult(
            optimized_module=self.module,
            optimization_metrics={"accuracy": 0.8},
            training_time=10.5,
            validation_score=0.75,
            cache_key="test_cache_key",
            timestamp=datetime.now(),
        )
        value = metric.calculate(self.module, result_with_lower_score)
        self.assertEqual(value, 0.75)

    def test_training_time_metric(self):
        """Test training time metric."""
        # Target time of 60 seconds
        metric = TrainingTimeMetric(weight=0.5, target_time=60.0)

        # Test calculation - should be 1.0 if training time is less than target
        result_fast = OptimizationResult(
            optimized_module=self.module,
            optimization_metrics={"accuracy": 0.95},
            training_time=30.0,  # Faster than target
            validation_score=0.92,
            cache_key="test_cache_key",
            timestamp=datetime.now(),
        )
        value = metric.calculate(self.module, result_fast)
        self.assertEqual(value, 1.0)

        # Test with longer training time
        result_slow = OptimizationResult(
            optimized_module=self.module,
            optimization_metrics={"accuracy": 0.95},
            training_time=120.0,  # Slower than target
            validation_score=0.92,
            cache_key="test_cache_key",
            timestamp=datetime.now(),
        )
        value = metric.calculate(self.module, result_slow)
        self.assertEqual(value, 0.5)  # 60/120 = 0.5

        # Test with zero training time (edge case)
        result_zero = OptimizationResult(
            optimized_module=self.module,
            optimization_metrics={"accuracy": 0.95},
            training_time=0.0,
            validation_score=0.92,
            cache_key="test_cache_key",
            timestamp=datetime.now(),
        )
        value = metric.calculate(self.module, result_zero)
        self.assertEqual(value, 1.0)  # Should handle zero gracefully


class TestQualityAssessor(unittest.TestCase):
    """Test quality assessor."""

    def setUp(self):
        """Set up test fixtures."""
        self.module = mock.MagicMock(spec=STREAMContentGenerator)
        self.module.domain = "mathematics"

        self.result = OptimizationResult(
            optimized_module=self.module,
            optimization_metrics={"accuracy": 0.95},
            training_time=10.5,
            validation_score=0.92,
            cache_key="test_cache_key",
            timestamp=datetime.now(),
        )

        self.quality_requirements = {
            "min_accuracy": 0.9,
            "max_training_time": 60.0,
        }

        self.assessor = QualityAssessor()

    def test_assess_quality(self):
        """Test quality assessment."""
        assessment = self.assessor.assess_quality(
            self.module, self.result, self.quality_requirements
        )

        # Check assessment structure
        self.assertIn("domain", assessment)
        self.assertIn("overall_score", assessment)
        self.assertIn("metrics", assessment)
        self.assertIn("requirements_met", assessment)
        self.assertIn("timestamp", assessment)

        # Check domain
        self.assertEqual(assessment["domain"], "mathematics")

        # Check metrics
        self.assertIn("validation_accuracy", assessment["metrics"])
        self.assertIn("training_time", assessment["metrics"])

        # Check requirements met
        self.assertTrue(assessment["requirements_met"])

    def test_validate_result_success(self):
        """Test validation with successful result."""
        is_valid, assessment = self.assessor.validate_result(
            self.module, self.result, self.quality_requirements
        )

        self.assertTrue(is_valid)
        self.assertTrue(assessment["requirements_met"])

    def test_validate_result_failure(self):
        """Test validation with failing result."""
        # Create result that doesn't meet requirements
        failing_result = OptimizationResult(
            optimized_module=self.module,
            optimization_metrics={"accuracy": 0.85},
            training_time=10.5,
            validation_score=0.85,  # Below min_accuracy of 0.9
            cache_key="test_cache_key",
            timestamp=datetime.now(),
        )

        is_valid, assessment = self.assessor.validate_result(
            self.module, failing_result, self.quality_requirements
        )

        self.assertFalse(is_valid)
        self.assertFalse(assessment["requirements_met"])

    def test_get_quality_assessor(self):
        """Test global quality assessor singleton."""
        assessor1 = get_quality_assessor()
        assessor2 = get_quality_assessor()

        # Should be the same instance
        self.assertIs(assessor1, assessor2)

        # Should be a QualityAssessor
        self.assertIsInstance(assessor1, QualityAssessor)
