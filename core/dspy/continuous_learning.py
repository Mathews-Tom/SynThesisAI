"""
Continuous learning system for DSPy modules.

This module implements automated reoptimization, performance tracking,
and adaptive parameter tuning for continuous improvement of DSPy modules.
"""

import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .base_module import STREAMContentGenerator
from .config import get_dspy_config
from .exceptions import DSPyIntegrationError
from .feedback import get_feedback_integrator
from .optimization_engine import get_optimization_engine
from .quality_assessment import get_quality_assessor

logger = logging.getLogger(__name__)


class LearningMetrics:
    """Metrics for tracking learning progress."""

    def __init__(self):
        """Initialize learning metrics."""
        self.optimization_count = 0
        self.performance_history: List[Dict[str, Any]] = []
        self.improvement_rate = 0.0
        self.last_optimization = None
        self.best_performance = 0.0
        self.current_performance = 0.0

    def add_performance_record(self, performance_data: Dict[str, Any]) -> None:
        """
        Add a performance record.

        Args:
            performance_data: Performance metrics data
        """
        record = {
            "timestamp": datetime.now().isoformat(),
            "optimization_count": self.optimization_count,
            **performance_data,
        }
        self.performance_history.append(record)

        # Update current performance
        if "overall_score" in performance_data:
            self.current_performance = performance_data["overall_score"]
            if self.current_performance > self.best_performance:
                self.best_performance = self.current_performance

        # Calculate improvement rate
        self._calculate_improvement_rate()

    def _calculate_improvement_rate(self) -> None:
        """Calculate the improvement rate over recent optimizations."""
        if len(self.performance_history) < 2:
            self.improvement_rate = 0.0
            return

        # Compare last 5 records vs previous 5 records
        recent_records = self.performance_history[-5:]
        previous_records = (
            self.performance_history[-10:-5]
            if len(self.performance_history) >= 10
            else []
        )

        if not previous_records:
            self.improvement_rate = 0.0
            return

        recent_avg = sum(r.get("overall_score", 0) for r in recent_records) / len(
            recent_records
        )
        previous_avg = sum(r.get("overall_score", 0) for r in previous_records) / len(
            previous_records
        )

        if previous_avg > 0:
            self.improvement_rate = (recent_avg - previous_avg) / previous_avg
        else:
            self.improvement_rate = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "optimization_count": self.optimization_count,
            "performance_history": self.performance_history,
            "improvement_rate": self.improvement_rate,
            "last_optimization": self.last_optimization.isoformat()
            if self.last_optimization
            else None,
            "best_performance": self.best_performance,
            "current_performance": self.current_performance,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LearningMetrics":
        """Create metrics from dictionary."""
        metrics = cls()
        metrics.optimization_count = data.get("optimization_count", 0)
        metrics.performance_history = data.get("performance_history", [])
        metrics.improvement_rate = data.get("improvement_rate", 0.0)
        metrics.best_performance = data.get("best_performance", 0.0)
        metrics.current_performance = data.get("current_performance", 0.0)

        if data.get("last_optimization"):
            metrics.last_optimization = datetime.fromisoformat(
                data["last_optimization"]
            )

        return metrics


class ContinuousLearningManager:
    """Manager for continuous learning of DSPy modules."""

    def __init__(self, learning_dir: str = ".learning"):
        """
        Initialize continuous learning manager.

        Args:
            learning_dir: Directory for storing learning data
        """
        self.learning_dir = Path(learning_dir)
        self.learning_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__ + ".ContinuousLearningManager")
        self.config = get_dspy_config()

        # Initialize components
        self.optimization_engine = get_optimization_engine()
        self.feedback_integrator = get_feedback_integrator()
        self.quality_assessor = get_quality_assessor()

        # Learning metrics for each domain
        self.domain_metrics: Dict[str, LearningMetrics] = {}

        # Load existing metrics
        self._load_learning_metrics()

    def should_reoptimize(self, domain: str) -> Tuple[bool, str]:
        """
        Determine if a domain module should be reoptimized.

        Args:
            domain: Domain to check

        Returns:
            Tuple of (should_reoptimize, reason)
        """
        try:
            metrics = self.domain_metrics.get(domain, LearningMetrics())

            # Check if never optimized (but allow other checks to override for testing)
            never_optimized = metrics.optimization_count == 0

            # Check time-based reoptimization
            if metrics.last_optimization:
                time_since_last = datetime.now() - metrics.last_optimization
                max_age = timedelta(days=7)  # Default 7 days
                if time_since_last > max_age:
                    return (
                        True,
                        f"Last optimization was {time_since_last.days} days ago",
                    )

            # Check performance degradation
            if (
                metrics.current_performance < metrics.best_performance * 0.9
            ):  # 10% degradation threshold
                return (
                    True,
                    f"Performance degraded from {metrics.best_performance:.3f} to {metrics.current_performance:.3f}",
                )

            # Check improvement stagnation
            if (
                len(metrics.performance_history) >= 10
                and metrics.improvement_rate < 0.01
            ):  # Less than 1% improvement
                return (
                    True,
                    f"Improvement rate stagnated at {metrics.improvement_rate:.3f}",
                )

            # Check feedback-based triggers
            feedback_summary = (
                self.feedback_integrator.feedback_manager.get_feedback_summary(domain)
            )
            high_priority_feedback = feedback_summary.get("by_severity", {}).get(
                "high", 0
            ) + feedback_summary.get("by_severity", {}).get("critical", 0)

            if high_priority_feedback >= 5:  # Threshold for high-priority feedback
                return (
                    True,
                    f"Accumulated {high_priority_feedback} high-priority feedback items",
                )

            # Finally check if never optimized
            if never_optimized:
                return True, "Initial optimization required"

            return False, "No reoptimization needed"

        except Exception as e:
            self.logger.error(
                "Error checking reoptimization need for domain %s: %s", domain, str(e)
            )
            return False, f"Error: {str(e)}"

    def perform_continuous_learning(
        self, domain_module: STREAMContentGenerator
    ) -> Dict[str, Any]:
        """
        Perform continuous learning for a domain module.

        Args:
            domain_module: The domain module to optimize

        Returns:
            Learning results and metrics
        """
        domain = domain_module.domain
        learning_start_time = time.time()

        try:
            self.logger.info("Starting continuous learning for domain: %s", domain)

            # Check if reoptimization is needed
            should_reopt, reason = self.should_reoptimize(domain)
            if not should_reopt:
                self.logger.info("Skipping reoptimization for %s: %s", domain, reason)
                return {
                    "domain": domain,
                    "reoptimized": False,
                    "reason": reason,
                    "learning_time": time.time() - learning_start_time,
                }

            self.logger.info("Reoptimizing %s: %s", domain, reason)

            # Get current performance baseline
            baseline_performance = self._evaluate_current_performance(domain_module)

            # Integrate feedback into optimization parameters
            feedback_adjustments = (
                self.feedback_integrator.integrate_feedback_for_optimization(domain)
            )

            # Apply adaptive parameter tuning
            adaptive_params = self._apply_adaptive_tuning(domain)

            # Merge optimization parameters
            optimization_params = {
                **self.config.get_optimization_config("mipro_v2"),
                **feedback_adjustments.get("optimization_params", {}),
                **adaptive_params,
            }

            quality_requirements = {
                **self.config.default_quality_requirements,
                **feedback_adjustments.get("quality_requirements", {}),
            }

            # Perform optimization
            optimized_module = self.optimization_engine.optimize_for_domain(
                domain_module, quality_requirements
            )

            # Evaluate post-optimization performance
            post_optimization_performance = self._evaluate_current_performance(
                optimized_module
            )

            # Update learning metrics
            metrics = self.domain_metrics.get(domain, LearningMetrics())
            metrics.optimization_count += 1
            metrics.last_optimization = datetime.now()
            metrics.add_performance_record(
                {
                    "overall_score": post_optimization_performance["overall_score"],
                    "accuracy": post_optimization_performance.get("accuracy", 0),
                    "relevance": post_optimization_performance.get("relevance", 0),
                    "coherence": post_optimization_performance.get("coherence", 0),
                    "pedagogical_value": post_optimization_performance.get(
                        "pedagogical_value", 0
                    ),
                    "optimization_reason": reason,
                    "baseline_score": baseline_performance["overall_score"],
                    "improvement": post_optimization_performance["overall_score"]
                    - baseline_performance["overall_score"],
                }
            )

            self.domain_metrics[domain] = metrics

            # Save updated metrics
            self._save_learning_metrics()

            learning_time = time.time() - learning_start_time

            learning_results = {
                "domain": domain,
                "reoptimized": True,
                "reason": reason,
                "baseline_performance": baseline_performance,
                "post_optimization_performance": post_optimization_performance,
                "improvement": post_optimization_performance["overall_score"]
                - baseline_performance["overall_score"],
                "optimization_count": metrics.optimization_count,
                "learning_time": learning_time,
                "feedback_adjustments": feedback_adjustments,
                "adaptive_params": adaptive_params,
            }

            self.logger.info(
                "Completed continuous learning for %s: improvement=%.3f, time=%.2fs",
                domain,
                learning_results["improvement"],
                learning_time,
            )

            return learning_results

        except Exception as e:
            self.logger.error(
                "Continuous learning failed for domain %s: %s", domain, str(e)
            )
            return {
                "domain": domain,
                "reoptimized": False,
                "error": str(e),
                "learning_time": time.time() - learning_start_time,
            }

    def _evaluate_current_performance(
        self, domain_module: STREAMContentGenerator
    ) -> Dict[str, Any]:
        """
        Evaluate current performance of a domain module.

        Args:
            domain_module: Module to evaluate

        Returns:
            Performance metrics
        """
        try:
            # Generate sample content for evaluation
            sample_inputs = self._get_sample_inputs(domain_module.domain)

            performance_scores = []
            for inputs in sample_inputs:
                try:
                    # Generate content
                    result = domain_module.generate(**inputs)

                    # Assess quality
                    quality_assessment = self.quality_assessor.assess_quality(
                        result, domain_module.domain, inputs
                    )

                    performance_scores.append(quality_assessment)

                except Exception as e:
                    self.logger.warning("Failed to evaluate sample input: %s", str(e))
                    continue

            if not performance_scores:
                return {"overall_score": 0.0, "error": "No valid evaluations"}

            # Calculate average performance
            avg_performance = {
                "overall_score": sum(
                    s.get("overall_score", 0) for s in performance_scores
                )
                / len(performance_scores),
                "accuracy": sum(s.get("accuracy", 0) for s in performance_scores)
                / len(performance_scores),
                "relevance": sum(s.get("relevance", 0) for s in performance_scores)
                / len(performance_scores),
                "coherence": sum(s.get("coherence", 0) for s in performance_scores)
                / len(performance_scores),
                "pedagogical_value": sum(
                    s.get("pedagogical_value", 0) for s in performance_scores
                )
                / len(performance_scores),
                "sample_count": len(performance_scores),
            }

            return avg_performance

        except Exception as e:
            self.logger.error("Performance evaluation failed: %s", str(e))
            return {"overall_score": 0.0, "error": str(e)}

    def _get_sample_inputs(self, domain: str) -> List[Dict[str, Any]]:
        """
        Get sample inputs for performance evaluation.

        Args:
            domain: Domain to get samples for

        Returns:
            List of sample input dictionaries
        """
        # Domain-specific sample inputs for evaluation
        sample_inputs = {
            "mathematics": [
                {
                    "topic": "algebra",
                    "difficulty_level": "high_school",
                    "learning_objectives": ["solve linear equations"],
                },
                {
                    "topic": "calculus",
                    "difficulty_level": "undergraduate",
                    "learning_objectives": ["find derivatives"],
                },
                {
                    "topic": "geometry",
                    "difficulty_level": "middle_school",
                    "learning_objectives": ["calculate area"],
                },
            ],
            "science": [
                {
                    "topic": "physics",
                    "difficulty_level": "high_school",
                    "learning_objectives": ["understand motion"],
                },
                {
                    "topic": "chemistry",
                    "difficulty_level": "undergraduate",
                    "learning_objectives": ["chemical reactions"],
                },
                {
                    "topic": "biology",
                    "difficulty_level": "middle_school",
                    "learning_objectives": ["cell structure"],
                },
            ],
            "technology": [
                {
                    "topic": "programming",
                    "difficulty_level": "beginner",
                    "learning_objectives": ["basic algorithms"],
                },
                {
                    "topic": "data_structures",
                    "difficulty_level": "intermediate",
                    "learning_objectives": ["arrays and lists"],
                },
                {
                    "topic": "databases",
                    "difficulty_level": "advanced",
                    "learning_objectives": ["query optimization"],
                },
            ],
            "reading": [
                {
                    "topic": "literature",
                    "difficulty_level": "high_school",
                    "learning_objectives": ["analyze themes"],
                },
                {
                    "topic": "poetry",
                    "difficulty_level": "middle_school",
                    "learning_objectives": ["identify metaphors"],
                },
                {
                    "topic": "essays",
                    "difficulty_level": "undergraduate",
                    "learning_objectives": ["critical analysis"],
                },
            ],
            "engineering": [
                {
                    "topic": "mechanical",
                    "difficulty_level": "undergraduate",
                    "learning_objectives": ["design principles"],
                },
                {
                    "topic": "electrical",
                    "difficulty_level": "advanced",
                    "learning_objectives": ["circuit analysis"],
                },
                {
                    "topic": "software",
                    "difficulty_level": "intermediate",
                    "learning_objectives": ["system design"],
                },
            ],
            "arts": [
                {
                    "topic": "visual_arts",
                    "difficulty_level": "beginner",
                    "learning_objectives": ["color theory"],
                },
                {
                    "topic": "music",
                    "difficulty_level": "intermediate",
                    "learning_objectives": ["composition basics"],
                },
                {
                    "topic": "theater",
                    "difficulty_level": "advanced",
                    "learning_objectives": ["character development"],
                },
            ],
        }

        return sample_inputs.get(
            domain,
            [
                {
                    "topic": "general",
                    "difficulty_level": "intermediate",
                    "learning_objectives": ["basic concepts"],
                }
            ],
        )

    def _apply_adaptive_tuning(self, domain: str) -> Dict[str, Any]:
        """
        Apply adaptive parameter tuning based on learning history.

        Args:
            domain: Domain to tune parameters for

        Returns:
            Adaptive parameter adjustments
        """
        metrics = self.domain_metrics.get(domain, LearningMetrics())
        adaptive_params = {}

        try:
            # Adjust based on optimization count
            if metrics.optimization_count > 5:
                # Increase exploration for experienced modules
                adaptive_params["init_temperature"] = 1.6
                adaptive_params["num_candidate_programs"] = 20
            elif metrics.optimization_count > 2:
                # Moderate exploration
                adaptive_params["init_temperature"] = 1.5
                adaptive_params["num_candidate_programs"] = 18
            else:
                # Conservative for new modules
                adaptive_params["init_temperature"] = 1.3
                adaptive_params["num_candidate_programs"] = 16

            # Adjust based on improvement rate
            if metrics.improvement_rate < 0.01:  # Slow improvement
                adaptive_params["max_labeled_demos"] = 20  # More training data
                adaptive_params["optuna_trials_num"] = 150  # More optimization trials
            elif metrics.improvement_rate > 0.1:  # Fast improvement
                adaptive_params["max_labeled_demos"] = 12  # Less training data
                adaptive_params["optuna_trials_num"] = 80  # Fewer optimization trials

            # Adjust based on current performance
            if metrics.current_performance < 0.7:  # Low performance
                adaptive_params["max_bootstrapped_demos"] = 6  # More bootstrapping
                adaptive_params["max_labeled_demos"] = 24  # More labeled examples

            self.logger.debug(
                "Applied adaptive tuning for %s: %s", domain, adaptive_params
            )

        except Exception as e:
            self.logger.error(
                "Adaptive tuning failed for domain %s: %s", domain, str(e)
            )

        return adaptive_params

    def _load_learning_metrics(self) -> None:
        """Load learning metrics from storage."""
        try:
            metrics_file = self.learning_dir / "learning_metrics.json"
            if metrics_file.exists():
                data = json.loads(metrics_file.read_text(encoding="utf-8"))

                for domain, metrics_data in data.items():
                    self.domain_metrics[domain] = LearningMetrics.from_dict(
                        metrics_data
                    )

                self.logger.info(
                    "Loaded learning metrics for %d domains", len(self.domain_metrics)
                )

        except Exception as e:
            self.logger.error("Failed to load learning metrics: %s", str(e))

    def _save_learning_metrics(self) -> None:
        """Save learning metrics to storage."""
        try:
            metrics_data = {}
            for domain, metrics in self.domain_metrics.items():
                metrics_data[domain] = metrics.to_dict()

            metrics_file = self.learning_dir / "learning_metrics.json"
            metrics_file.write_text(
                json.dumps(metrics_data, indent=2), encoding="utf-8"
            )

            self.logger.debug(
                "Saved learning metrics for %d domains", len(self.domain_metrics)
            )

        except Exception as e:
            self.logger.error("Failed to save learning metrics: %s", str(e))

    def get_learning_summary(self, domain: Optional[str] = None) -> Dict[str, Any]:
        """
        Get learning summary for domain(s).

        Args:
            domain: Specific domain or None for all domains

        Returns:
            Learning summary
        """
        if domain:
            metrics = self.domain_metrics.get(domain, LearningMetrics())
            return {
                "domain": domain,
                "metrics": metrics.to_dict(),
                "feedback_summary": self.feedback_integrator.feedback_manager.get_feedback_summary(
                    domain
                ),
            }
        else:
            summary = {"total_domains": len(self.domain_metrics), "domains": {}}

            for domain_name, metrics in self.domain_metrics.items():
                summary["domains"][domain_name] = {
                    "optimization_count": metrics.optimization_count,
                    "current_performance": metrics.current_performance,
                    "best_performance": metrics.best_performance,
                    "improvement_rate": metrics.improvement_rate,
                    "last_optimization": metrics.last_optimization.isoformat()
                    if metrics.last_optimization
                    else None,
                }

            return summary

    def run_continuous_learning_cycle(
        self, domains: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Run a continuous learning cycle for specified domains.

        Args:
            domains: List of domains to process, or None for all domains

        Returns:
            Cycle results
        """
        cycle_start_time = time.time()
        results = {
            "cycle_start": datetime.now().isoformat(),
            "domains_processed": [],
            "domains_optimized": [],
            "total_improvements": 0.0,
            "cycle_time": 0.0,
        }

        try:
            # Get domains to process
            if domains is None:
                # Get all available domains from signature registry
                from .signature_registry import get_signature_registry

                registry = get_signature_registry()
                domains = registry.list_domains()

            self.logger.info(
                "Starting continuous learning cycle for %d domains", len(domains)
            )

            for domain in domains:
                try:
                    # Create a mock domain module for testing
                    # In practice, this would get the actual domain module
                    from .base_module import STREAMContentGenerator

                    domain_module = STREAMContentGenerator(domain)

                    # Perform continuous learning
                    learning_result = self.perform_continuous_learning(domain_module)

                    results["domains_processed"].append(domain)

                    if learning_result.get("reoptimized", False):
                        results["domains_optimized"].append(domain)
                        improvement = learning_result.get("improvement", 0.0)
                        results["total_improvements"] += improvement

                except Exception as e:
                    self.logger.error(
                        "Failed to process domain %s in learning cycle: %s",
                        domain,
                        str(e),
                    )
                    continue

            results["cycle_time"] = time.time() - cycle_start_time

            self.logger.info(
                "Completed continuous learning cycle: processed=%d, optimized=%d, total_improvement=%.3f, time=%.2fs",
                len(results["domains_processed"]),
                len(results["domains_optimized"]),
                results["total_improvements"],
                results["cycle_time"],
            )

        except Exception as e:
            self.logger.error("Continuous learning cycle failed: %s", str(e))
            results["error"] = str(e)

        return results


# Global continuous learning manager instance
_continuous_learning_manager = None


def get_continuous_learning_manager() -> ContinuousLearningManager:
    """Get the global continuous learning manager instance."""
    global _continuous_learning_manager
    if _continuous_learning_manager is None:
        _continuous_learning_manager = ContinuousLearningManager()
    return _continuous_learning_manager
