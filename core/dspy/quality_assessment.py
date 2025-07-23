"""
Quality assessment for DSPy optimization results.

This module provides functionality for validating optimization results
against quality requirements and generating quality metrics.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

from .base_module import STREAMContentGenerator
from .config import OptimizationResult
from .exceptions import QualityAssessmentError

logger = logging.getLogger(__name__)


class QualityMetric:
    """Base class for quality metrics."""

    def __init__(self, name: str, weight: float = 1.0):
        """
        Initialize quality metric.

        Args:
            name: Name of the metric
            weight: Weight of the metric in overall quality score
        """
        self.name = name
        self.weight = weight

    def calculate(
        self, module: STREAMContentGenerator, result: OptimizationResult
    ) -> float:
        """
        Calculate metric value.

        Args:
            module: Optimized module
            result: Optimization result

        Returns:
            Metric value between 0.0 and 1.0
        """
        raise NotImplementedError("Subclasses must implement calculate()")


class ValidationAccuracyMetric(QualityMetric):
    """Metric based on validation accuracy."""

    def __init__(self, weight: float = 1.0):
        """Initialize validation accuracy metric."""
        super().__init__("validation_accuracy", weight)

    def calculate(
        self, module: STREAMContentGenerator, result: OptimizationResult
    ) -> float:
        """Calculate validation accuracy metric."""
        # In a real implementation, this would use the validation score
        # For now, we'll use the validation_score from the result
        return result.validation_score


class TrainingTimeMetric(QualityMetric):
    """Metric based on training time efficiency."""

    def __init__(self, weight: float = 0.5, target_time: float = 60.0):
        """
        Initialize training time metric.

        Args:
            weight: Weight of the metric
            target_time: Target training time in seconds
        """
        super().__init__("training_time", weight)
        self.target_time = target_time

    def calculate(
        self, module: STREAMContentGenerator, result: OptimizationResult
    ) -> float:
        """Calculate training time metric."""
        # Lower is better, with diminishing returns
        if result.training_time <= 0:
            return 1.0  # Avoid division by zero

        # Normalize: 1.0 at target_time, approaching 0 as time increases
        return min(1.0, self.target_time / result.training_time)


class QualityAssessor:
    """
    Assesses quality of optimization results.

    Validates optimization results against quality requirements and
    generates quality metrics.
    """

    def __init__(self):
        """Initialize quality assessor."""
        self.metrics: Dict[str, QualityMetric] = {
            "validation_accuracy": ValidationAccuracyMetric(),
            "training_time": TrainingTimeMetric(),
        }
        self.logger = logging.getLogger(__name__ + ".QualityAssessor")

    def assess_quality(
        self,
        module: STREAMContentGenerator,
        result: OptimizationResult,
        quality_requirements: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Assess quality of optimization result.

        Args:
            module: Optimized module
            result: Optimization result
            quality_requirements: Quality requirements

        Returns:
            Quality assessment results
        """
        self.logger.info("Assessing quality for domain %s", module.domain)

        # Calculate individual metrics
        metric_results = {}
        for name, metric in self.metrics.items():
            try:
                value = metric.calculate(module, result)
                metric_results[name] = value
            except Exception as e:
                self.logger.error("Error calculating metric %s: %s", name, str(e))
                metric_results[name] = 0.0

        # Calculate overall quality score (weighted average)
        total_weight = sum(metric.weight for metric in self.metrics.values())
        if total_weight > 0:
            overall_score = (
                sum(
                    metric_results[name] * metric.weight
                    for name, metric in self.metrics.items()
                )
                / total_weight
            )
        else:
            overall_score = 0.0

        # Check if quality requirements are met
        requirements_met = self._check_requirements(
            metric_results, quality_requirements
        )

        assessment = {
            "domain": module.domain,
            "overall_score": overall_score,
            "metrics": metric_results,
            "requirements_met": requirements_met,
            "timestamp": result.timestamp.isoformat(),
        }

        self.logger.info(
            "Quality assessment for domain %s: overall_score=%.2f, requirements_met=%s",
            module.domain,
            overall_score,
            requirements_met,
        )

        return assessment

    def _check_requirements(
        self, metrics: Dict[str, float], requirements: Dict[str, Any]
    ) -> bool:
        """
        Check if metrics meet requirements.

        Args:
            metrics: Calculated metrics
            requirements: Quality requirements

        Returns:
            True if all requirements are met
        """
        # Check minimum accuracy requirement
        if (
            "min_accuracy" in requirements
            and metrics.get("validation_accuracy", 0) < requirements["min_accuracy"]
        ):
            return False

        # Check maximum training time requirement
        if (
            "max_training_time" in requirements
            and metrics.get("training_time", 0) < 0.5
        ):  # Normalized value < 0.5 means training took too long
            return False

        # All requirements met
        return True

    def validate_result(
        self,
        module: STREAMContentGenerator,
        result: OptimizationResult,
        quality_requirements: Dict[str, Any],
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate optimization result against quality requirements.

        Args:
            module: Optimized module
            result: Optimization result
            quality_requirements: Quality requirements

        Returns:
            Tuple of (is_valid, assessment)
        """
        assessment = self.assess_quality(module, result, quality_requirements)
        is_valid = assessment["requirements_met"]

        if not is_valid:
            self.logger.warning(
                "Optimization result for domain %s did not meet quality requirements",
                module.domain,
            )

        return is_valid, assessment


# Global quality assessor instance
_quality_assessor = None


def get_quality_assessor() -> QualityAssessor:
    """Get the global quality assessor instance."""
    global _quality_assessor
    if _quality_assessor is None:
        _quality_assessor = QualityAssessor()
    return _quality_assessor
