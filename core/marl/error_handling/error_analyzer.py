"""
MARL Error Analyzer.

This module provides error pattern analysis and recognition capabilities
for the multi-agent reinforcement learning coordination system.
"""

import asyncio
import json
from collections import defaultdict, deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from utils.logging_config import get_logger

from .error_types import ErrorPattern, ErrorStatistics, MARLError


class ErrorAnalyzer:
    """
    Analyzes error patterns and provides insights for system improvement.

    Tracks error occurrences, identifies patterns, and provides
    recommendations for preventing recurring errors.
    """

    def __init__(
        self,
        pattern_window_size: int = 100,
        pattern_threshold: int = 3,
        analysis_interval: float = 300.0,  # 5 minutes
        enable_persistence: bool = True,
        persistence_path: Optional[Path] = None,
    ):
        """
        Initialize error analyzer.

        Args:
            pattern_window_size: Number of recent errors to analyze for patterns
            pattern_threshold: Minimum occurrences to consider a pattern
            analysis_interval: Interval between pattern analysis runs (seconds)
            enable_persistence: Enable pattern persistence to disk
            persistence_path: Path for pattern persistence
        """
        self.logger = get_logger(__name__)

        # Configuration
        self.pattern_window_size = pattern_window_size
        self.pattern_threshold = pattern_threshold
        self.analysis_interval = analysis_interval
        self.enable_persistence = enable_persistence
        self.persistence_path = persistence_path or Path(
            "data/marl/error_patterns.json"
        )

        # Error tracking
        self.recent_errors: deque = deque(maxlen=pattern_window_size)
        self.error_patterns: Dict[str, ErrorPattern] = {}
        self.error_sequences: List[List[str]] = []

        # Analysis state
        self.last_analysis_time = datetime.now()
        self.analysis_task: Optional[asyncio.Task] = None
        self.is_running = False

        # Pattern statistics
        self.pattern_statistics = {
            "total_patterns_identified": 0,
            "active_patterns": 0,
            "pattern_accuracy": 0.0,
            "last_analysis": None,
        }

        # Load existing patterns if persistence enabled
        if self.enable_persistence:
            self._load_patterns()

        self.logger.info("Error analyzer initialized")

    async def start_analysis(self) -> None:
        """Start continuous error pattern analysis."""
        if self.is_running:
            self.logger.warning("Error analysis already running")
            return

        self.is_running = True
        self.analysis_task = asyncio.create_task(self._analysis_loop())
        self.logger.info("Error pattern analysis started")

    async def stop_analysis(self) -> None:
        """Stop continuous error pattern analysis."""
        if not self.is_running:
            return

        self.is_running = False

        if self.analysis_task:
            self.analysis_task.cancel()
            try:
                await self.analysis_task
            except asyncio.CancelledError:
                pass

        # Save patterns if persistence enabled
        if self.enable_persistence:
            self._save_patterns()

        self.logger.info("Error pattern analysis stopped")

    async def analyze_error(self, error: MARLError) -> Dict[str, Any]:
        """
        Analyze a single error and update patterns.

        Args:
            error: The error to analyze

        Returns:
            Analysis results including pattern matches and recommendations
        """
        # Add error to recent errors
        error_data = {
            "error_id": error.error_id,
            "error_code": error.error_code,
            "error_type": error.__class__.__name__,
            "timestamp": error.timestamp,
            "context": error.context,
            "severity": error.severity,
        }

        self.recent_errors.append(error_data)

        # Check for immediate pattern matches
        matching_patterns = self._find_matching_patterns(error)

        # Update pattern frequencies
        for pattern in matching_patterns:
            pattern.update_frequency()

        # Analyze error context for new patterns
        potential_patterns = await self._identify_potential_patterns(error)

        # Generate recommendations
        recommendations = self._generate_error_recommendations(error, matching_patterns)

        analysis_result = {
            "error_id": error.error_id,
            "matching_patterns": [p.pattern_id for p in matching_patterns],
            "potential_new_patterns": len(potential_patterns),
            "recommendations": recommendations,
            "severity_assessment": self._assess_error_severity(error),
            "prediction_confidence": self._calculate_prediction_confidence(error),
        }

        self.logger.debug(
            "Error analysis complete for %s: %d matching patterns, %d recommendations",
            error.error_id,
            len(matching_patterns),
            len(recommendations),
        )

        return analysis_result

    async def _analysis_loop(self) -> None:
        """Main analysis loop for pattern identification."""
        while self.is_running:
            try:
                await asyncio.sleep(self.analysis_interval)

                if not self.is_running:
                    break

                await self._perform_pattern_analysis()

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Error in analysis loop: %s", str(e), exc_info=True)
                await asyncio.sleep(60)  # Wait before retrying

    async def _perform_pattern_analysis(self) -> None:
        """Perform comprehensive pattern analysis on recent errors."""
        if len(self.recent_errors) < self.pattern_threshold:
            return

        self.logger.debug(
            "Performing pattern analysis on %d recent errors", len(self.recent_errors)
        )

        # Analyze error sequences
        await self._analyze_error_sequences()

        # Analyze error contexts
        await self._analyze_error_contexts()

        # Analyze temporal patterns
        await self._analyze_temporal_patterns()

        # Update pattern statistics
        self._update_pattern_statistics()

        # Save patterns if persistence enabled
        if self.enable_persistence:
            self._save_patterns()

        self.last_analysis_time = datetime.now()
        self.pattern_statistics["last_analysis"] = self.last_analysis_time.isoformat()

        self.logger.info(
            "Pattern analysis complete: %d active patterns identified",
            len(
                [
                    p
                    for p in self.error_patterns.values()
                    if p.frequency >= self.pattern_threshold
                ]
            ),
        )

    async def _analyze_error_sequences(self) -> None:
        """Analyze sequences of errors to identify patterns."""
        # Group errors by time windows
        time_windows = self._group_errors_by_time_window(minutes=10)

        for window_errors in time_windows:
            if len(window_errors) >= self.pattern_threshold:
                # Create sequence pattern
                error_codes = [e["error_code"] for e in window_errors]
                sequence_id = f"sequence_{hash(tuple(error_codes))}"

                if sequence_id not in self.error_patterns:
                    self.error_patterns[sequence_id] = ErrorPattern(
                        pattern_id=sequence_id,
                        error_codes=error_codes,
                        context_patterns={"type": "sequence", "window_minutes": 10},
                    )

                self.error_patterns[sequence_id].update_frequency()

    async def _analyze_error_contexts(self) -> None:
        """Analyze error contexts to identify common patterns."""
        context_groups = defaultdict(list)

        # Group errors by context similarity
        for error_data in self.recent_errors:
            context = error_data.get("context", {})

            # Group by agent_id if present
            if "agent_id" in context:
                key = f"agent_{context['agent_id']}"
                context_groups[key].append(error_data)

            # Group by coordination_id if present
            if "coordination_id" in context:
                key = f"coordination_{context['coordination_id']}"
                context_groups[key].append(error_data)

            # Group by source_component if present
            if "source_component" in context:
                key = f"component_{context['source_component']}"
                context_groups[key].append(error_data)

        # Create patterns for significant context groups
        for group_key, group_errors in context_groups.items():
            if len(group_errors) >= self.pattern_threshold:
                pattern_id = f"context_{group_key}"
                error_codes = list(set(e["error_code"] for e in group_errors))

                if pattern_id not in self.error_patterns:
                    self.error_patterns[pattern_id] = ErrorPattern(
                        pattern_id=pattern_id,
                        error_codes=error_codes,
                        context_patterns={"type": "context", "group": group_key},
                    )

                self.error_patterns[pattern_id].frequency = len(group_errors)

    async def _analyze_temporal_patterns(self) -> None:
        """Analyze temporal patterns in error occurrences."""
        # Group errors by hour of day
        hourly_groups = defaultdict(list)

        for error_data in self.recent_errors:
            timestamp = error_data["timestamp"]
            hour = timestamp.hour
            hourly_groups[hour].append(error_data)

        # Identify peak error hours
        peak_hours = []
        avg_errors_per_hour = len(self.recent_errors) / 24

        for hour, errors in hourly_groups.items():
            if len(errors) > avg_errors_per_hour * 1.5:  # 50% above average
                peak_hours.append(hour)

        if peak_hours:
            pattern_id = f"temporal_peak_hours_{hash(tuple(peak_hours))}"

            if pattern_id not in self.error_patterns:
                all_error_codes = []
                for hour in peak_hours:
                    all_error_codes.extend(
                        [e["error_code"] for e in hourly_groups[hour]]
                    )

                self.error_patterns[pattern_id] = ErrorPattern(
                    pattern_id=pattern_id,
                    error_codes=list(set(all_error_codes)),
                    context_patterns={"type": "temporal", "peak_hours": peak_hours},
                )

            self.error_patterns[pattern_id].frequency = sum(
                len(hourly_groups[hour]) for hour in peak_hours
            )

    def _group_errors_by_time_window(self, minutes: int) -> List[List[Dict[str, Any]]]:
        """Group errors by time windows."""
        if not self.recent_errors:
            return []

        # Sort errors by timestamp
        sorted_errors = sorted(self.recent_errors, key=lambda x: x["timestamp"])

        windows = []
        current_window = []
        window_start = None

        for error_data in sorted_errors:
            timestamp = error_data["timestamp"]

            if window_start is None:
                window_start = timestamp
                current_window = [error_data]
            elif (timestamp - window_start).total_seconds() <= minutes * 60:
                current_window.append(error_data)
            else:
                if len(current_window) >= self.pattern_threshold:
                    windows.append(current_window)

                window_start = timestamp
                current_window = [error_data]

        # Add final window if significant
        if len(current_window) >= self.pattern_threshold:
            windows.append(current_window)

        return windows

    def _find_matching_patterns(self, error: MARLError) -> List[ErrorPattern]:
        """Find existing patterns that match the given error."""
        matching_patterns = []

        for pattern in self.error_patterns.values():
            if pattern.matches(error):
                matching_patterns.append(pattern)

        return matching_patterns

    async def _identify_potential_patterns(
        self, error: MARLError
    ) -> List[Dict[str, Any]]:
        """Identify potential new patterns based on the error."""
        potential_patterns = []

        # Look for similar recent errors
        similar_errors = []
        for recent_error in self.recent_errors:
            if (
                recent_error["error_code"] == error.error_code
                and recent_error["error_type"] == error.__class__.__name__
            ):
                similar_errors.append(recent_error)

        if len(similar_errors) >= self.pattern_threshold:
            potential_patterns.append(
                {
                    "type": "recurring_error",
                    "error_code": error.error_code,
                    "frequency": len(similar_errors),
                }
            )

        # Look for context-based patterns
        if error.context:
            for key, value in error.context.items():
                context_matches = []
                for recent_error in self.recent_errors:
                    if recent_error.get("context", {}).get(key) == value:
                        context_matches.append(recent_error)

                if len(context_matches) >= self.pattern_threshold:
                    potential_patterns.append(
                        {
                            "type": "context_pattern",
                            "context_key": key,
                            "context_value": value,
                            "frequency": len(context_matches),
                        }
                    )

        return potential_patterns

    def _generate_error_recommendations(
        self, error: MARLError, matching_patterns: List[ErrorPattern]
    ) -> List[str]:
        """Generate recommendations based on error and patterns."""
        recommendations = []

        # Add error-specific recommendations
        if error.recovery_hint:
            recommendations.append(error.recovery_hint)

        # Add pattern-based recommendations
        for pattern in matching_patterns:
            if pattern.recovery_success_rate < 0.5:  # Low success rate
                recommendations.append(
                    f"Pattern {pattern.pattern_id} has low recovery success rate. "
                    "Consider reviewing recovery strategies."
                )

            if pattern.frequency > 10:  # High frequency
                recommendations.append(
                    f"Pattern {pattern.pattern_id} occurs frequently. "
                    "Consider addressing root cause."
                )

        # Add general recommendations based on error type
        error_type = error.__class__.__name__

        if error_type == "AgentError":
            recommendations.append("Monitor agent health and consider load balancing")
        elif error_type == "CoordinationError":
            recommendations.append(
                "Review coordination timeouts and consensus thresholds"
            )
        elif error_type == "LearningError":
            recommendations.append(
                "Check learning parameters and experience buffer health"
            )
        elif error_type == "CommunicationError":
            recommendations.append(
                "Verify network connectivity and message queue status"
            )
        elif error_type == "PerformanceError":
            recommendations.append("Monitor system resources and optimize performance")

        return list(set(recommendations))  # Remove duplicates

    def _assess_error_severity(self, error: MARLError) -> Dict[str, Any]:
        """Assess the severity and impact of an error."""
        severity_score = 0
        factors = []

        # Base severity from error
        severity_map = {"DEBUG": 1, "INFO": 2, "WARNING": 3, "ERROR": 4, "CRITICAL": 5}
        severity_score += severity_map.get(error.severity, 3)
        factors.append(f"base_severity_{error.severity}")

        # Check if error affects multiple agents
        if "participating_agents" in error.context:
            agent_count = len(error.context["participating_agents"])
            if agent_count > 1:
                severity_score += min(agent_count, 3)
                factors.append(f"affects_{agent_count}_agents")

        # Check if error is part of a pattern
        matching_patterns = self._find_matching_patterns(error)
        if matching_patterns:
            pattern_severity = sum(p.frequency for p in matching_patterns) / len(
                matching_patterns
            )
            severity_score += min(pattern_severity / 5, 2)
            factors.append(f"pattern_frequency_{pattern_severity:.1f}")

        # Normalize severity score
        severity_score = min(severity_score, 10)

        return {
            "severity_score": severity_score,
            "severity_level": self._get_severity_level(severity_score),
            "contributing_factors": factors,
        }

    def _get_severity_level(self, score: float) -> str:
        """Convert severity score to level."""
        if score <= 2:
            return "LOW"
        elif score <= 4:
            return "MEDIUM"
        elif score <= 6:
            return "HIGH"
        else:
            return "CRITICAL"

    def _calculate_prediction_confidence(self, error: MARLError) -> float:
        """Calculate confidence in error prediction/classification."""
        confidence = 0.5  # Base confidence

        # Increase confidence if error matches known patterns
        matching_patterns = self._find_matching_patterns(error)
        if matching_patterns:
            avg_success_rate = sum(
                p.recovery_success_rate for p in matching_patterns
            ) / len(matching_patterns)
            confidence += 0.3 * avg_success_rate

        # Increase confidence based on context completeness
        context_completeness = (
            len(error.context) / 10
        )  # Assume 10 is max useful context items
        confidence += 0.2 * min(context_completeness, 1.0)

        return min(confidence, 1.0)

    def _update_pattern_statistics(self) -> None:
        """Update pattern analysis statistics."""
        active_patterns = [
            p
            for p in self.error_patterns.values()
            if p.frequency >= self.pattern_threshold
        ]

        self.pattern_statistics.update(
            {
                "total_patterns_identified": len(self.error_patterns),
                "active_patterns": len(active_patterns),
                "pattern_accuracy": self._calculate_pattern_accuracy(),
            }
        )

    def _calculate_pattern_accuracy(self) -> float:
        """Calculate overall pattern prediction accuracy."""
        if not self.error_patterns:
            return 0.0

        total_predictions = sum(p.frequency for p in self.error_patterns.values())
        if total_predictions == 0:
            return 0.0

        successful_predictions = sum(
            p.frequency * p.recovery_success_rate for p in self.error_patterns.values()
        )

        return successful_predictions / total_predictions

    def _save_patterns(self) -> None:
        """Save error patterns to disk."""
        try:
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)

            patterns_data = {
                "patterns": {
                    pattern_id: {
                        "pattern_id": pattern.pattern_id,
                        "error_codes": pattern.error_codes,
                        "frequency": pattern.frequency,
                        "last_occurrence": pattern.last_occurrence.isoformat()
                        if pattern.last_occurrence
                        else None,
                        "context_patterns": pattern.context_patterns,
                        "recovery_success_rate": pattern.recovery_success_rate,
                    }
                    for pattern_id, pattern in self.error_patterns.items()
                },
                "statistics": self.pattern_statistics,
                "saved_at": datetime.now().isoformat(),
            }

            with open(self.persistence_path, "w") as f:
                json.dump(patterns_data, f, indent=2)

            self.logger.debug("Error patterns saved to %s", self.persistence_path)

        except Exception as e:
            self.logger.error("Failed to save error patterns: %s", str(e))

    def _load_patterns(self) -> None:
        """Load error patterns from disk."""
        try:
            if not self.persistence_path.exists():
                self.logger.debug(
                    "No existing pattern file found at %s", self.persistence_path
                )
                return

            with open(self.persistence_path, "r") as f:
                patterns_data = json.load(f)

            # Load patterns
            for pattern_id, pattern_data in patterns_data.get("patterns", {}).items():
                pattern = ErrorPattern(
                    pattern_id=pattern_data["pattern_id"],
                    error_codes=pattern_data["error_codes"],
                    frequency=pattern_data["frequency"],
                    context_patterns=pattern_data["context_patterns"],
                    recovery_success_rate=pattern_data["recovery_success_rate"],
                )

                if pattern_data["last_occurrence"]:
                    pattern.last_occurrence = datetime.fromisoformat(
                        pattern_data["last_occurrence"]
                    )

                self.error_patterns[pattern_id] = pattern

            # Load statistics
            if "statistics" in patterns_data:
                self.pattern_statistics.update(patterns_data["statistics"])

            self.logger.info(
                "Loaded %d error patterns from %s",
                len(self.error_patterns),
                self.persistence_path,
            )

        except Exception as e:
            self.logger.error("Failed to load error patterns: %s", str(e))

    def get_pattern_summary(self) -> Dict[str, Any]:
        """Get summary of identified patterns."""
        active_patterns = [
            p
            for p in self.error_patterns.values()
            if p.frequency >= self.pattern_threshold
        ]

        return {
            "total_patterns": len(self.error_patterns),
            "active_patterns": len(active_patterns),
            "most_frequent_patterns": sorted(
                [(p.pattern_id, p.frequency) for p in active_patterns],
                key=lambda x: x[1],
                reverse=True,
            )[:10],
            "pattern_statistics": self.pattern_statistics,
            "recent_errors_analyzed": len(self.recent_errors),
        }

    def get_pattern_details(self, pattern_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific pattern."""
        if pattern_id not in self.error_patterns:
            return None

        pattern = self.error_patterns[pattern_id]

        return {
            "pattern_id": pattern.pattern_id,
            "error_codes": pattern.error_codes,
            "frequency": pattern.frequency,
            "last_occurrence": pattern.last_occurrence.isoformat()
            if pattern.last_occurrence
            else None,
            "context_patterns": pattern.context_patterns,
            "recovery_success_rate": pattern.recovery_success_rate,
            "recommendations": self._get_pattern_recommendations(pattern),
        }

    def _get_pattern_recommendations(self, pattern: ErrorPattern) -> List[str]:
        """Get recommendations for a specific pattern."""
        recommendations = []

        if pattern.recovery_success_rate < 0.3:
            recommendations.append(
                "Low recovery success rate - review recovery strategies"
            )

        if pattern.frequency > 20:
            recommendations.append("High frequency pattern - investigate root cause")

        if pattern.context_patterns.get("type") == "temporal":
            recommendations.append(
                "Temporal pattern detected - consider scheduled maintenance"
            )

        if pattern.context_patterns.get("type") == "sequence":
            recommendations.append("Error sequence pattern - review system workflows")

        return recommendations

    async def shutdown(self) -> None:
        """Shutdown error analyzer."""
        await self.stop_analysis()
        self.logger.info("Error analyzer shutdown complete")
