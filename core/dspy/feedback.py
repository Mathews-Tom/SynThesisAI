"""
DSPy feedback integration for continuous learning.

This module provides functionality for collecting and integrating feedback
into the DSPy optimization process for continuous improvement.
"""

import json
import logging
import time
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .base_module import STREAMContentGenerator
from .config import get_dspy_config
from .exceptions import DSPyIntegrationError

logger = logging.getLogger(__name__)


class FeedbackSource(Enum):
    """Sources of feedback for DSPy modules."""

    USER = "user"  # Direct user feedback
    SYSTEM = "system"  # Automated system feedback
    EXPERT = "expert"  # Subject matter expert feedback
    PEER = "peer"  # Peer review feedback
    SELF = "self"  # Self-evaluation feedback


class FeedbackType(Enum):
    """Types of feedback for DSPy modules."""

    ACCURACY = "accuracy"  # Factual correctness
    RELEVANCE = "relevance"  # Relevance to the query/topic
    COHERENCE = "coherence"  # Logical flow and structure
    COMPLETENESS = "completeness"  # Coverage of necessary information
    PEDAGOGICAL = "pedagogical"  # Educational value
    SAFETY = "safety"  # Ethical and safety considerations
    GENERAL = "general"  # General feedback


class FeedbackSeverity(Enum):
    """Severity levels for feedback."""

    CRITICAL = "critical"  # Must be addressed immediately
    HIGH = "high"  # Should be addressed soon
    MEDIUM = "medium"  # Should be addressed when possible
    LOW = "low"  # Minor issue
    SUGGESTION = "suggestion"  # Optional improvement


class Feedback:
    """Feedback for DSPy modules."""

    def __init__(
        self,
        content: str,
        feedback_type: Union[FeedbackType, str],
        source: Union[FeedbackSource, str],
        domain: str,
        severity: Union[FeedbackSeverity, str] = FeedbackSeverity.MEDIUM,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
    ):
        """
        Initialize feedback.

        Args:
            content: Feedback content
            feedback_type: Type of feedback
            source: Source of feedback
            domain: Domain the feedback applies to
            severity: Severity level of the feedback
            metadata: Additional metadata
            timestamp: When the feedback was created
        """
        self.content = content
        self.feedback_type = (
            feedback_type
            if isinstance(feedback_type, FeedbackType)
            else FeedbackType(feedback_type)
        )
        self.source = (
            source if isinstance(source, FeedbackSource) else FeedbackSource(source)
        )
        self.domain = domain
        self.severity = (
            severity
            if isinstance(severity, FeedbackSeverity)
            else FeedbackSeverity(severity)
        )
        self.metadata = metadata or {}
        self.timestamp = timestamp or datetime.now()
        self.feedback_id = (
            f"{self.domain}_{int(time.time())}_{hash(self.content) % 10000}"
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert feedback to dictionary.

        Returns:
            Dictionary representation of feedback
        """
        return {
            "feedback_id": self.feedback_id,
            "content": self.content,
            "feedback_type": self.feedback_type.value,
            "source": self.source.value,
            "domain": self.domain,
            "severity": self.severity.value,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Feedback":
        """
        Create feedback from dictionary.

        Args:
            data: Dictionary representation of feedback

        Returns:
            Feedback instance
        """
        feedback = cls(
            content=data["content"],
            feedback_type=data["feedback_type"],
            source=data["source"],
            domain=data["domain"],
            severity=data.get("severity", FeedbackSeverity.MEDIUM.value),
            metadata=data.get("metadata", {}),
            timestamp=(
                datetime.fromisoformat(data["timestamp"])
                if "timestamp" in data
                else None
            ),
        )
        feedback.feedback_id = data.get("feedback_id", feedback.feedback_id)
        return feedback


class FeedbackManager:
    """Manager for DSPy feedback."""

    def __init__(self, feedback_dir: str = ".feedback"):
        """
        Initialize feedback manager.

        Args:
            feedback_dir: Directory for storing feedback
        """
        self.feedback_dir = Path(feedback_dir)
        self.feedback_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__ + ".FeedbackManager")
        self.config = get_dspy_config()
        self.feedback_cache: Dict[str, List[Feedback]] = {}

    def add_feedback(
        self,
        content: str,
        feedback_type: Union[FeedbackType, str],
        source: Union[FeedbackSource, str],
        domain: str,
        severity: Union[FeedbackSeverity, str] = FeedbackSeverity.MEDIUM,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Feedback:
        """
        Add feedback for a domain.

        Args:
            content: Feedback content
            feedback_type: Type of feedback
            source: Source of feedback
            domain: Domain the feedback applies to
            severity: Severity level of the feedback
            metadata: Additional metadata

        Returns:
            Created feedback
        """
        feedback = Feedback(
            content=content,
            feedback_type=feedback_type,
            source=source,
            domain=domain,
            severity=severity,
            metadata=metadata,
        )

        # Add to cache
        if domain not in self.feedback_cache:
            self.feedback_cache[domain] = []
        self.feedback_cache[domain].append(feedback)

        # Save to file
        self._save_feedback(feedback)

        self.logger.info(
            "Added %s feedback for domain %s: %s",
            feedback.source.value,
            domain,
            feedback.feedback_id,
        )

        return feedback

    def get_feedback(
        self,
        domain: str,
        feedback_type: Optional[Union[FeedbackType, str]] = None,
        source: Optional[Union[FeedbackSource, str]] = None,
        min_severity: Optional[Union[FeedbackSeverity, str]] = None,
        limit: int = 100,
    ) -> List[Feedback]:
        """
        Get feedback for a domain.

        Args:
            domain: Domain to get feedback for
            feedback_type: Filter by feedback type
            source: Filter by feedback source
            min_severity: Filter by minimum severity
            limit: Maximum number of feedback items to return

        Returns:
            List of feedback items
        """
        # Convert enum string values to enum objects
        if isinstance(feedback_type, str):
            feedback_type = FeedbackType(feedback_type)
        if isinstance(source, str):
            source = FeedbackSource(source)
        if isinstance(min_severity, str):
            min_severity = FeedbackSeverity(min_severity)

        # Get severity rank for comparison
        severity_ranks = {
            FeedbackSeverity.CRITICAL: 4,
            FeedbackSeverity.HIGH: 3,
            FeedbackSeverity.MEDIUM: 2,
            FeedbackSeverity.LOW: 1,
            FeedbackSeverity.SUGGESTION: 0,
        }
        min_severity_rank = (
            severity_ranks[min_severity] if min_severity is not None else 0
        )

        # Load feedback from cache or file
        all_feedback = self._load_feedback(domain)

        # Apply filters
        filtered_feedback = []
        for feedback in all_feedback:
            # Filter by type
            if feedback_type is not None and feedback.feedback_type != feedback_type:
                continue

            # Filter by source
            if source is not None and feedback.source != source:
                continue

            # Filter by severity
            if min_severity is not None:
                feedback_severity_rank = severity_ranks[feedback.severity]
                if feedback_severity_rank < min_severity_rank:
                    continue

            filtered_feedback.append(feedback)

        # Sort by timestamp (newest first) and limit
        filtered_feedback.sort(key=lambda x: x.timestamp, reverse=True)
        return filtered_feedback[:limit]

    def _save_feedback(self, feedback: Feedback) -> bool:
        """
        Save feedback to file.

        Args:
            feedback: Feedback to save

        Returns:
            True if saved successfully
        """
        try:
            feedback_file = self.feedback_dir / f"{feedback.domain}_feedback.json"

            # Load existing feedback
            existing_feedback = []
            if feedback_file.exists():
                existing_data = json.loads(feedback_file.read_text(encoding="utf-8"))
                existing_feedback = existing_data.get("feedback", [])

            # Add new feedback
            existing_feedback.append(feedback.to_dict())

            # Save updated feedback
            feedback_data = {
                "domain": feedback.domain,
                "feedback": existing_feedback,
                "last_updated": datetime.now().isoformat(),
            }

            feedback_file.write_text(
                json.dumps(feedback_data, indent=2), encoding="utf-8"
            )

            return True
        except Exception as e:
            self.logger.error("Failed to save feedback: %s", str(e))
            return False

    def _load_feedback(self, domain: str) -> List[Feedback]:
        """
        Load feedback from file.

        Args:
            domain: Domain to load feedback for

        Returns:
            List of feedback items
        """
        # Check cache first
        if domain in self.feedback_cache:
            return self.feedback_cache[domain]

        try:
            feedback_file = self.feedback_dir / f"{domain}_feedback.json"
            if not feedback_file.exists():
                return []

            feedback_data = json.loads(feedback_file.read_text(encoding="utf-8"))
            feedback_list = []

            for item in feedback_data.get("feedback", []):
                feedback_list.append(Feedback.from_dict(item))

            # Cache the loaded feedback
            self.feedback_cache[domain] = feedback_list
            return feedback_list

        except Exception as e:
            self.logger.error(
                "Failed to load feedback for domain %s: %s", domain, str(e)
            )
            return []

    def get_feedback_summary(self, domain: str) -> Dict[str, Any]:
        """
        Get feedback summary for a domain.

        Args:
            domain: Domain to get summary for

        Returns:
            Feedback summary statistics
        """
        feedback_list = self._load_feedback(domain)

        if not feedback_list:
            return {
                "domain": domain,
                "total_feedback": 0,
                "by_type": {},
                "by_source": {},
                "by_severity": {},
                "recent_feedback": [],
            }

        # Count by type
        by_type = {}
        for feedback in feedback_list:
            type_name = feedback.feedback_type.value
            by_type[type_name] = by_type.get(type_name, 0) + 1

        # Count by source
        by_source = {}
        for feedback in feedback_list:
            source_name = feedback.source.value
            by_source[source_name] = by_source.get(source_name, 0) + 1

        # Count by severity
        by_severity = {}
        for feedback in feedback_list:
            severity_name = feedback.severity.value
            by_severity[severity_name] = by_severity.get(severity_name, 0) + 1

        # Get recent feedback (last 5)
        recent_feedback = sorted(
            feedback_list, key=lambda x: x.timestamp, reverse=True
        )[:5]
        recent_feedback_data = [f.to_dict() for f in recent_feedback]

        return {
            "domain": domain,
            "total_feedback": len(feedback_list),
            "by_type": by_type,
            "by_source": by_source,
            "by_severity": by_severity,
            "recent_feedback": recent_feedback_data,
        }


class DSPyFeedbackIntegrator:
    """Integrates feedback with DSPy optimization process."""

    def __init__(self):
        """Initialize feedback integrator."""
        self.feedback_manager = FeedbackManager()
        self.logger = logging.getLogger(__name__ + ".DSPyFeedbackIntegrator")
        self.config = get_dspy_config()

    def collect_validation_feedback(
        self, domain_module: STREAMContentGenerator, validation_results: Dict[str, Any]
    ) -> List[Feedback]:
        """
        Collect feedback from validation results.

        Args:
            domain_module: The domain module being validated
            validation_results: Results from validation process

        Returns:
            List of generated feedback items
        """
        feedback_items = []

        try:
            domain = domain_module.domain

            # Generate accuracy feedback
            if "accuracy_score" in validation_results:
                accuracy = validation_results["accuracy_score"]
                if accuracy < 0.7:
                    feedback = self.feedback_manager.add_feedback(
                        content=f"Low accuracy score: {accuracy:.2f}. Consider improving factual correctness.",
                        feedback_type=FeedbackType.ACCURACY,
                        source=FeedbackSource.SYSTEM,
                        domain=domain,
                        severity=(
                            FeedbackSeverity.HIGH
                            if accuracy < 0.5
                            else FeedbackSeverity.MEDIUM
                        ),
                        metadata={
                            "accuracy_score": accuracy,
                            "validation_results": validation_results,
                        },
                    )
                    feedback_items.append(feedback)

            # Generate relevance feedback
            if "relevance_score" in validation_results:
                relevance = validation_results["relevance_score"]
                if relevance < 0.8:
                    feedback = self.feedback_manager.add_feedback(
                        content=f"Low relevance score: {relevance:.2f}. Content may not be sufficiently relevant to the topic.",
                        feedback_type=FeedbackType.RELEVANCE,
                        source=FeedbackSource.SYSTEM,
                        domain=domain,
                        severity=FeedbackSeverity.MEDIUM,
                        metadata={
                            "relevance_score": relevance,
                            "validation_results": validation_results,
                        },
                    )
                    feedback_items.append(feedback)

            # Generate coherence feedback
            if "coherence_score" in validation_results:
                coherence = validation_results["coherence_score"]
                if coherence < 0.7:
                    feedback = self.feedback_manager.add_feedback(
                        content=f"Low coherence score: {coherence:.2f}. Improve logical flow and structure.",
                        feedback_type=FeedbackType.COHERENCE,
                        source=FeedbackSource.SYSTEM,
                        domain=domain,
                        severity=FeedbackSeverity.MEDIUM,
                        metadata={
                            "coherence_score": coherence,
                            "validation_results": validation_results,
                        },
                    )
                    feedback_items.append(feedback)

            # Generate pedagogical feedback
            if "pedagogical_score" in validation_results:
                pedagogical = validation_results["pedagogical_score"]
                if pedagogical < 0.8:
                    feedback = self.feedback_manager.add_feedback(
                        content=f"Low pedagogical score: {pedagogical:.2f}. Enhance educational value and clarity.",
                        feedback_type=FeedbackType.PEDAGOGICAL,
                        source=FeedbackSource.SYSTEM,
                        domain=domain,
                        severity=FeedbackSeverity.MEDIUM,
                        metadata={
                            "pedagogical_score": pedagogical,
                            "validation_results": validation_results,
                        },
                    )
                    feedback_items.append(feedback)

            self.logger.info(
                "Collected %d feedback items from validation for domain %s",
                len(feedback_items),
                domain,
            )

        except Exception as e:
            self.logger.error("Failed to collect validation feedback: %s", str(e))

        return feedback_items

    def integrate_feedback_for_optimization(self, domain: str) -> Dict[str, Any]:
        """
        Integrate feedback into optimization parameters.

        Args:
            domain: Domain to integrate feedback for

        Returns:
            Updated optimization parameters based on feedback
        """
        try:
            # Get recent high-priority feedback
            feedback_items = self.feedback_manager.get_feedback(
                domain=domain, min_severity=FeedbackSeverity.MEDIUM, limit=50
            )

            if not feedback_items:
                self.logger.info("No feedback available for domain %s", domain)
                return {}

            # Analyze feedback patterns
            feedback_analysis = self._analyze_feedback_patterns(feedback_items)

            # Generate optimization adjustments
            optimization_adjustments = self._generate_optimization_adjustments(
                feedback_analysis
            )

            self.logger.info(
                "Generated optimization adjustments for domain %s based on %d feedback items",
                domain,
                len(feedback_items),
            )

            return optimization_adjustments

        except Exception as e:
            self.logger.error(
                "Failed to integrate feedback for optimization: %s", str(e)
            )
            return {}

    def _analyze_feedback_patterns(
        self, feedback_items: List[Feedback]
    ) -> Dict[str, Any]:
        """
        Analyze patterns in feedback.

        Args:
            feedback_items: List of feedback items to analyze

        Returns:
            Analysis results
        """
        analysis = {
            "total_feedback": len(feedback_items),
            "type_distribution": {},
            "severity_distribution": {},
            "common_issues": [],
            "improvement_areas": [],
        }

        # Analyze feedback types
        for feedback in feedback_items:
            type_name = feedback.feedback_type.value
            analysis["type_distribution"][type_name] = (
                analysis["type_distribution"].get(type_name, 0) + 1
            )

            severity_name = feedback.severity.value
            analysis["severity_distribution"][severity_name] = (
                analysis["severity_distribution"].get(severity_name, 0) + 1
            )

        # Identify common issues (types with high frequency)
        for feedback_type, count in analysis["type_distribution"].items():
            if count >= len(feedback_items) * 0.3:  # 30% threshold
                analysis["common_issues"].append(
                    {
                        "type": feedback_type,
                        "frequency": count,
                        "percentage": (count / len(feedback_items)) * 100,
                    }
                )

        # Identify improvement areas based on severity
        high_severity_count = analysis["severity_distribution"].get(
            "high", 0
        ) + analysis["severity_distribution"].get("critical", 0)
        if high_severity_count > 0:
            analysis["improvement_areas"].append("high_priority_issues")

        return analysis

    def _generate_optimization_adjustments(
        self, feedback_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate optimization parameter adjustments based on feedback analysis.

        Args:
            feedback_analysis: Results from feedback analysis

        Returns:
            Optimization parameter adjustments
        """
        adjustments = {
            "quality_requirements": {},
            "optimization_params": {},
            "training_focus": [],
        }

        # Adjust quality requirements based on common issues
        for issue in feedback_analysis.get("common_issues", []):
            issue_type = issue["type"]

            if issue_type == "accuracy":
                adjustments["quality_requirements"]["min_accuracy"] = 0.85
                adjustments["training_focus"].append("factual_correctness")
            elif issue_type == "relevance":
                adjustments["quality_requirements"]["min_relevance"] = 0.85
                adjustments["training_focus"].append("topic_relevance")
            elif issue_type == "coherence":
                adjustments["quality_requirements"]["min_coherence"] = 0.8
                adjustments["training_focus"].append("logical_structure")
            elif issue_type == "pedagogical":
                adjustments["quality_requirements"]["min_pedagogical_value"] = 0.85
                adjustments["training_focus"].append("educational_clarity")

        # Adjust optimization parameters for high-priority issues
        if "high_priority_issues" in feedback_analysis.get("improvement_areas", []):
            adjustments["optimization_params"][
                "max_labeled_demos"
            ] = 20  # Increase training examples
            # Removed num_candidate_programs as it's not supported by MIPROv2

        return adjustments


# Global feedback integrator instance
_feedback_integrator = None


def get_feedback_integrator() -> DSPyFeedbackIntegrator:
    """Get the global feedback integrator instance."""
    global _feedback_integrator
    if _feedback_integrator is None:
        _feedback_integrator = DSPyFeedbackIntegrator()
    return _feedback_integrator


def get_feedback_manager() -> FeedbackManager:
    """Get the global feedback manager instance."""
    integrator = get_feedback_integrator()
    return integrator.feedback_manager
