"""
MARL A/B Testing Framework.

This module provides specialized A/B testing capabilities for MARL systems,
including statistical analysis, significance testing, and result interpretation.
"""

# Standard Library
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

# Third-Party Library
import numpy as np
from scipy import stats

# SynThesisAI Modules
from utils.logging_config import get_logger
from .experiment_manager import Experiment, ExperimentType


class ABTestResult(Enum):
    """A/B test result interpretation."""

    TREATMENT_WINS = "treatment_wins"
    CONTROL_WINS = "control_wins"
    NO_SIGNIFICANT_DIFFERENCE = "no_significant_difference"
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass
class StatisticalTest:
    """Results of a statistical test."""

    test_name: str
    statistic: float
    p_value: float
    is_significant: bool
    confidence_interval: Optional[Tuple[float, float]] = None
    effect_size: Optional[float] = None
    power: Optional[float] = None

    def __post_init__(self):
        """Validate statistical test results."""
        if not (0.0 <= self.p_value <= 1.0):
            raise ValueError("P-value must be between 0 and 1")


@dataclass
class ABTestAnalysis:
    """Comprehensive A/B test analysis results."""

    experiment_id: str
    metric_name: str
    control_mean: float
    treatment_mean: float
    control_std: float
    treatment_std: float
    control_n: int
    treatment_n: int

    # Statistical tests
    t_test: StatisticalTest
    mann_whitney_test: StatisticalTest

    # Effect size measures
    cohens_d: float
    hedges_g: float

    # Practical significance
    relative_improvement: float
    absolute_improvement: float

    # Overall result
    result: ABTestResult
    recommendation: str

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of A/B test analysis."""
        return {
            "metric": self.metric_name,
            "control": {
                "mean": self.control_mean,
                "std": self.control_std,
                "n": self.control_n,
            },
            "treatment": {
                "mean": self.treatment_mean,
                "std": self.treatment_std,
                "n": self.treatment_n,
            },
            "statistical_significance": {
                "t_test_p_value": self.t_test.p_value,
                "t_test_significant": self.t_test.is_significant,
                "mann_whitney_p_value": self.mann_whitney_test.p_value,
                "mann_whitney_significant": self.mann_whitney_test.is_significant,
            },
            "effect_size": {
                "cohens_d": self.cohens_d,
                "hedges_g": self.hedges_g,
                "relative_improvement": self.relative_improvement,
                "absolute_improvement": self.absolute_improvement,
            },
            "result": self.result.value,
            "recommendation": self.recommendation,
        }


class ABTestManager:
    """
    A/B testing manager for MARL systems.

    Provides specialized functionality for designing, running, and analyzing
    A/B tests for MARL coordination strategies and parameters.
    """

    def __init__(
        self,
        significance_level: float = 0.05,
        minimum_effect_size: float = 0.1,
        power_threshold: float = 0.8,
    ):
        """
        Initialize the A/B test manager.

        Args:
            significance_level: Statistical significance threshold
            minimum_effect_size: Minimum practical effect size
            power_threshold: Minimum statistical power threshold
        """
        self.logger = get_logger(__name__)
        self.significance_level = significance_level
        self.minimum_effect_size = minimum_effect_size
        self.power_threshold = power_threshold

        self.logger.info(
            "A/B test manager initialized with α=%.3f, min_effect=%.3f, power=%.3f",
            significance_level,
            minimum_effect_size,
            power_threshold,
        )

    def calculate_sample_size(
        self,
        expected_effect_size: float,
        baseline_std: float,
        power: float = 0.8,
        two_sided: bool = True,
    ) -> int:
        """
        Calculate required sample size for A/B test.

        Args:
            expected_effect_size: Expected effect size (Cohen's d)
            baseline_std: Standard deviation of baseline metric
            power: Desired statistical power
            two_sided: Whether to use two-sided test

        Returns:
            Required sample size per group
        """
        try:
            # Calculate critical values
            alpha = self.significance_level
            if two_sided:
                z_alpha = stats.norm.ppf(1 - alpha / 2)
            else:
                z_alpha = stats.norm.ppf(1 - alpha)

            z_beta = stats.norm.ppf(power)

            # Calculate sample size using standard formula
            n = 2 * ((z_alpha + z_beta) ** 2) / (expected_effect_size**2)

            # Round up to nearest integer
            sample_size = math.ceil(n)

            self.logger.info(
                "Calculated sample size: %d per group (effect_size=%.3f, power=%.3f)",
                sample_size,
                expected_effect_size,
                power,
            )

            return sample_size

        except Exception as e:
            self.logger.error("Sample size calculation failed: %s", str(e))
            return 100  # Default fallback

    def analyze_ab_test(self, experiment: Experiment, metric_name: str) -> Optional[ABTestAnalysis]:
        """
        Analyze A/B test results for a specific metric.

        Args:
            experiment: A/B test experiment
            metric_name: Name of metric to analyze

        Returns:
            A/B test analysis results or None if insufficient data
        """
        if experiment.experiment_type != ExperimentType.AB_TEST:
            self.logger.error("Experiment is not an A/B test: %s", experiment.experiment_id)
            return None

        # Get control and treatment results
        control_result = experiment.get_result("control")
        treatment_result = experiment.get_result("treatment")

        if not control_result or not treatment_result:
            self.logger.error("Missing control or treatment results")
            return None

        # Extract metric data
        control_data = self._extract_metric_data(control_result, metric_name)
        treatment_data = self._extract_metric_data(treatment_result, metric_name)

        if not control_data or not treatment_data:
            self.logger.error("Insufficient data for metric: %s", metric_name)
            return None

        try:
            return self._perform_statistical_analysis(
                experiment.experiment_id, metric_name, control_data, treatment_data
            )

        except Exception as e:
            self.logger.error("A/B test analysis failed: %s", str(e))
            return None

    def _extract_metric_data(self, result, metric_name: str) -> Optional[List[float]]:
        """Extract metric data from experiment result."""
        try:
            # Try performance data first (time series)
            if metric_name in result.performance_data:
                data = result.performance_data[metric_name]
                return [float(x) for x in data if isinstance(x, (int, float))]

            # Try metrics (individual measurements)
            if metric_name in result.metrics:
                metric_entries = result.metrics[metric_name]
                if isinstance(metric_entries, list):
                    values = []
                    for entry in metric_entries:
                        if isinstance(entry, dict) and "value" in entry:
                            value = entry["value"]
                            if isinstance(value, (int, float)):
                                values.append(float(value))
                        elif isinstance(entry, (int, float)):
                            values.append(float(entry))
                    return values if values else None
                elif isinstance(metric_entries, (int, float)):
                    return [float(metric_entries)]

            return None

        except Exception as e:
            self.logger.warning("Failed to extract metric data for %s: %s", metric_name, str(e))
            return None

    def _perform_statistical_analysis(
        self,
        experiment_id: str,
        metric_name: str,
        control_data: List[float],
        treatment_data: List[float],
    ) -> ABTestAnalysis:
        """Perform comprehensive statistical analysis."""

        # Calculate descriptive statistics
        control_mean = np.mean(control_data)
        control_std = np.std(control_data, ddof=1)
        control_n = len(control_data)

        treatment_mean = np.mean(treatment_data)
        treatment_std = np.std(treatment_data, ddof=1)
        treatment_n = len(treatment_data)

        # Perform t-test
        t_stat, t_p_value = stats.ttest_ind(treatment_data, control_data, equal_var=False)
        t_test = StatisticalTest(
            test_name="Welch's t-test",
            statistic=t_stat,
            p_value=t_p_value,
            is_significant=t_p_value < self.significance_level,
        )

        # Calculate confidence interval for difference in means
        pooled_se = math.sqrt((control_std**2 / control_n) + (treatment_std**2 / treatment_n))
        df = ((control_std**2 / control_n) + (treatment_std**2 / treatment_n)) ** 2 / (
            (control_std**2 / control_n) ** 2 / (control_n - 1)
            + (treatment_std**2 / treatment_n) ** 2 / (treatment_n - 1)
        )
        t_critical = stats.t.ppf(1 - self.significance_level / 2, df)
        margin_error = t_critical * pooled_se

        mean_diff = treatment_mean - control_mean
        t_test.confidence_interval = (
            mean_diff - margin_error,
            mean_diff + margin_error,
        )

        # Perform Mann-Whitney U test (non-parametric)
        u_stat, u_p_value = stats.mannwhitneyu(
            treatment_data, control_data, alternative="two-sided"
        )
        mann_whitney_test = StatisticalTest(
            test_name="Mann-Whitney U test",
            statistic=u_stat,
            p_value=u_p_value,
            is_significant=u_p_value < self.significance_level,
        )

        # Calculate effect sizes
        cohens_d = self._calculate_cohens_d(
            control_data,
            treatment_data,
            control_mean,
            treatment_mean,
            control_std,
            treatment_std,
            control_n,
            treatment_n,
        )

        hedges_g = self._calculate_hedges_g(cohens_d, control_n, treatment_n)

        # Calculate practical significance
        relative_improvement = (
            (treatment_mean - control_mean) / control_mean * 100 if control_mean != 0 else 0.0
        )
        absolute_improvement = treatment_mean - control_mean

        # Determine overall result
        result, recommendation = self._interpret_results(
            t_test, mann_whitney_test, cohens_d, relative_improvement
        )

        return ABTestAnalysis(
            experiment_id=experiment_id,
            metric_name=metric_name,
            control_mean=control_mean,
            treatment_mean=treatment_mean,
            control_std=control_std,
            treatment_std=treatment_std,
            control_n=control_n,
            treatment_n=treatment_n,
            t_test=t_test,
            mann_whitney_test=mann_whitney_test,
            cohens_d=cohens_d,
            hedges_g=hedges_g,
            relative_improvement=relative_improvement,
            absolute_improvement=absolute_improvement,
            result=result,
            recommendation=recommendation,
        )

    def _calculate_cohens_d(
        self,
        control_data: List[float],
        treatment_data: List[float],
        control_mean: float,
        treatment_mean: float,
        control_std: float,
        treatment_std: float,
        control_n: int,
        treatment_n: int,
    ) -> float:
        """Calculate Cohen's d effect size."""
        try:
            # Pooled standard deviation
            pooled_std = math.sqrt(
                ((control_n - 1) * control_std**2 + (treatment_n - 1) * treatment_std**2)
                / (control_n + treatment_n - 2)
            )

            if pooled_std == 0:
                return 0.0

            cohens_d = (treatment_mean - control_mean) / pooled_std
            return cohens_d

        except Exception as e:
            self.logger.warning("Failed to calculate Cohen's d: %s", str(e))
            return 0.0

    def _calculate_hedges_g(self, cohens_d: float, n1: int, n2: int) -> float:
        """Calculate Hedges' g (bias-corrected effect size)."""
        try:
            # Correction factor for small samples
            df = n1 + n2 - 2
            correction = 1 - (3 / (4 * df - 1))
            hedges_g = cohens_d * correction
            return hedges_g

        except Exception as e:
            self.logger.warning("Failed to calculate Hedges' g: %s", str(e))
            return cohens_d

    def _interpret_results(
        self,
        t_test: StatisticalTest,
        mann_whitney_test: StatisticalTest,
        cohens_d: float,
        relative_improvement: float,
    ) -> Tuple[ABTestResult, str]:
        """Interpret A/B test results and provide recommendation."""

        # Check for statistical significance
        is_statistically_significant = t_test.is_significant or mann_whitney_test.is_significant

        # Check for practical significance
        is_practically_significant = abs(cohens_d) >= self.minimum_effect_size

        # Determine result
        if not is_statistically_significant:
            if abs(relative_improvement) < 1.0:  # Less than 1% change
                result = ABTestResult.NO_SIGNIFICANT_DIFFERENCE
                recommendation = (
                    "No significant difference detected. Consider running the test longer "
                    "or increasing sample size if a smaller effect is important."
                )
            else:
                result = ABTestResult.INSUFFICIENT_DATA
                recommendation = (
                    "Trend observed but not statistically significant. "
                    "Increase sample size to achieve statistical power."
                )

        elif is_statistically_significant and is_practically_significant:
            if relative_improvement > 0:
                result = ABTestResult.TREATMENT_WINS
                recommendation = (
                    f"Treatment shows significant improvement ({relative_improvement:.1f}% increase). "
                    f"Effect size is {self._interpret_effect_size(abs(cohens_d))}. "
                    "Recommend implementing the treatment."
                )
            else:
                result = ABTestResult.CONTROL_WINS
                recommendation = (
                    f"Control performs significantly better ({abs(relative_improvement):.1f}% decrease in treatment). "
                    f"Effect size is {self._interpret_effect_size(abs(cohens_d))}. "
                    "Recommend keeping the control configuration."
                )

        elif is_statistically_significant and not is_practically_significant:
            result = ABTestResult.NO_SIGNIFICANT_DIFFERENCE
            recommendation = (
                "Statistically significant but practically insignificant difference. "
                f"Effect size ({abs(cohens_d):.3f}) is below minimum threshold ({self.minimum_effect_size}). "
                "Consider the cost-benefit of implementation."
            )

        else:
            result = ABTestResult.NO_SIGNIFICANT_DIFFERENCE
            recommendation = "No significant difference detected."

        return result, recommendation

    def _interpret_effect_size(self, effect_size: float) -> str:
        """Interpret effect size magnitude."""
        if effect_size < 0.2:
            return "negligible"
        elif effect_size < 0.5:
            return "small"
        elif effect_size < 0.8:
            return "medium"
        else:
            return "large"

    def analyze_multiple_metrics(
        self, experiment: Experiment, metric_names: List[str]
    ) -> Dict[str, ABTestAnalysis]:
        """
        Analyze A/B test for multiple metrics.

        Args:
            experiment: A/B test experiment
            metric_names: List of metric names to analyze

        Returns:
            Dictionary mapping metric names to analysis results
        """
        results = {}

        for metric_name in metric_names:
            analysis = self.analyze_ab_test(experiment, metric_name)
            if analysis:
                results[metric_name] = analysis
            else:
                self.logger.warning("Failed to analyze metric: %s", metric_name)

        # Apply multiple comparison correction if needed
        if len(results) > 1:
            self._apply_multiple_comparison_correction(results)

        return results

    def _apply_multiple_comparison_correction(self, results: Dict[str, ABTestAnalysis]):
        """Apply Bonferroni correction for multiple comparisons."""
        try:
            num_tests = len(results)
            corrected_alpha = self.significance_level / num_tests

            for analysis in results.values():
                # Update significance based on corrected alpha
                analysis.t_test.is_significant = analysis.t_test.p_value < corrected_alpha
                analysis.mann_whitney_test.is_significant = (
                    analysis.mann_whitney_test.p_value < corrected_alpha
                )

                # Re-interpret results with corrected significance
                result, recommendation = self._interpret_results(
                    analysis.t_test,
                    analysis.mann_whitney_test,
                    analysis.cohens_d,
                    analysis.relative_improvement,
                )
                analysis.result = result
                analysis.recommendation = f"[Bonferroni corrected] {recommendation}"

            self.logger.info(
                "Applied Bonferroni correction for %d tests (α=%.4f)",
                num_tests,
                corrected_alpha,
            )

        except Exception as e:
            self.logger.warning("Failed to apply multiple comparison correction: %s", str(e))

    def generate_ab_test_report(
        self, experiment: Experiment, metric_names: List[str]
    ) -> Dict[str, Any]:
        """
        Generate comprehensive A/B test report.

        Args:
            experiment: A/B test experiment
            metric_names: List of metrics to include in report

        Returns:
            Comprehensive A/B test report
        """
        # Analyze all metrics
        analyses = self.analyze_multiple_metrics(experiment, metric_names)

        # Generate summary
        summary = {
            "experiment_id": experiment.experiment_id,
            "experiment_name": experiment.name,
            "status": experiment.status.value,
            "created_at": experiment.created_at.isoformat(),
            "completed_at": experiment.completed_at.isoformat()
            if experiment.completed_at
            else None,
            "significance_level": self.significance_level,
            "minimum_effect_size": self.minimum_effect_size,
            "metrics_analyzed": len(analyses),
            "overall_recommendation": self._generate_overall_recommendation(analyses),
        }

        # Add detailed analyses
        detailed_results = {}
        for metric_name, analysis in analyses.items():
            detailed_results[metric_name] = analysis.get_summary()

        return {
            "summary": summary,
            "detailed_results": detailed_results,
            "methodology": {
                "statistical_tests": ["Welch's t-test", "Mann-Whitney U test"],
                "effect_size_measures": ["Cohen's d", "Hedges' g"],
                "multiple_comparison_correction": "Bonferroni" if len(analyses) > 1 else None,
            },
        }

    def _generate_overall_recommendation(self, analyses: Dict[str, ABTestAnalysis]) -> str:
        """Generate overall recommendation based on multiple metric analyses."""
        if not analyses:
            return "No valid analyses available."

        # Count results by type
        result_counts = {}
        for analysis in analyses.values():
            result_type = analysis.result
            result_counts[result_type] = result_counts.get(result_type, 0) + 1

        total_metrics = len(analyses)

        # Generate recommendation based on majority
        if result_counts.get(ABTestResult.TREATMENT_WINS, 0) > total_metrics / 2:
            return "Overall recommendation: Implement treatment. Majority of metrics show significant improvement."

        elif result_counts.get(ABTestResult.CONTROL_WINS, 0) > total_metrics / 2:
            return "Overall recommendation: Keep control. Majority of metrics favor the control configuration."

        elif result_counts.get(ABTestResult.NO_SIGNIFICANT_DIFFERENCE, 0) > total_metrics / 2:
            return "Overall recommendation: No clear winner. Consider business factors and implementation costs."

        else:
            return "Overall recommendation: Mixed results. Analyze individual metrics and consider trade-offs."


class ABTestManagerFactory:
    """Factory for creating A/B test managers."""

    @staticmethod
    def create(significance_level: float = 0.05) -> ABTestManager:
        """
        Create an A/B test manager.

        Args:
            significance_level: Statistical significance threshold

        Returns:
            A/B test manager
        """
        return ABTestManager(significance_level)

    @staticmethod
    def create_strict(
        significance_level: float = 0.01, minimum_effect_size: float = 0.2
    ) -> ABTestManager:
        """
        Create a strict A/B test manager with higher thresholds.

        Args:
            significance_level: Statistical significance threshold
            minimum_effect_size: Minimum practical effect size

        Returns:
            Strict A/B test manager
        """
        return ABTestManager(significance_level, minimum_effect_size)

    @staticmethod
    def create_permissive(
        significance_level: float = 0.1, minimum_effect_size: float = 0.05
    ) -> ABTestManager:
        """
        Create a permissive A/B test manager with lower thresholds.

        Args:
            significance_level: Statistical significance threshold
            minimum_effect_size: Minimum practical effect size

        Returns:
            Permissive A/B test manager
        """
        return ABTestManager(significance_level, minimum_effect_size)
