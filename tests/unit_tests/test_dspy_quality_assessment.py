"""
Unit tests for DSPy quality assessment.

Tests the quality assessment functionality for DSPy optimization results.
"""

# Standard Library
from datetime import datetime
from unittest import mock

# Third-Party Library
import pytest

# SynThesisAI Modules
from core.dspy.base_module import STREAMContentGenerator
from core.dspy.config import OptimizationResult
from core.dspy.quality_assessment import (
    QualityAssessor,
    TrainingTimeMetric,
    ValidationAccuracyMetric,
    get_quality_assessor,
)


@pytest.fixture
def module():
    module = mock.MagicMock(spec=STREAMContentGenerator)
    module.domain = "mathematics"
    return module


@pytest.fixture
def result(module):
    return OptimizationResult(
        optimized_module=module,
        optimization_metrics={"accuracy": 0.95},
        training_time=10.5,
        validation_score=0.92,
        cache_key="test_cache_key",
        timestamp=datetime.now(),
    )


@pytest.fixture
def quality_requirements():
    return {"min_accuracy": 0.9, "max_training_time": 60.0}


def test_validation_accuracy_metric(module, result):
    metric = ValidationAccuracyMetric(weight=1.0)
    assert metric.calculate(module, result) == 0.92

    lower_result = OptimizationResult(
        optimized_module=module,
        optimization_metrics={"accuracy": 0.8},
        training_time=10.5,
        validation_score=0.75,
        cache_key="test_cache_key",
        timestamp=datetime.now(),
    )
    assert metric.calculate(module, lower_result) == 0.75


def test_training_time_metric(module, result):
    metric = TrainingTimeMetric(weight=0.5, target_time=60.0)

    fast_result = OptimizationResult(
        optimized_module=module,
        optimization_metrics={"accuracy": 0.95},
        training_time=30.0,
        validation_score=0.92,
        cache_key="test_cache_key",
        timestamp=datetime.now(),
    )
    assert metric.calculate(module, fast_result) == 1.0

    slow_result = OptimizationResult(
        optimized_module=module,
        optimization_metrics={"accuracy": 0.95},
        training_time=120.0,
        validation_score=0.92,
        cache_key="test_cache_key",
        timestamp=datetime.now(),
    )
    assert metric.calculate(module, slow_result) == 0.5

    zero_result = OptimizationResult(
        optimized_module=module,
        optimization_metrics={"accuracy": 0.95},
        training_time=0.0,
        validation_score=0.92,
        cache_key="test_cache_key",
        timestamp=datetime.now(),
    )
    assert metric.calculate(module, zero_result) == 1.0


def test_assess_quality(module, result, quality_requirements):
    assessor = QualityAssessor()
    assessment = assessor.assess_quality(module, result, quality_requirements)

    assert assessment["domain"] == "mathematics"
    assert "overall_score" in assessment
    assert "metrics" in assessment
    assert "requirements_met" in assessment
    assert assessment["requirements_met"] is True


def test_validate_result_success(module, result, quality_requirements):
    assessor = QualityAssessor()
    is_valid, assessment = assessor.validate_result(module, result, quality_requirements)
    assert is_valid is True
    assert assessment["requirements_met"] is True


def test_validate_result_failure(module, quality_requirements):
    assessor = QualityAssessor()
    failing_result = OptimizationResult(
        optimized_module=module,
        optimization_metrics={"accuracy": 0.85},
        training_time=10.5,
        validation_score=0.85,
        cache_key="test_cache_key",
        timestamp=datetime.now(),
    )
    is_valid, assessment = assessor.validate_result(module, failing_result, quality_requirements)
    assert is_valid is False
    assert assessment["requirements_met"] is False


def test_get_quality_assessor_singleton():
    assessor1 = get_quality_assessor()
    assessor2 = get_quality_assessor()
    assert assessor1 is assessor2
    assert isinstance(assessor1, QualityAssessor)
