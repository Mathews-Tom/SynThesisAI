"""
MARL Experiment Manager.

This module provides comprehensive experiment management for MARL systems,
including experiment design, execution, and result analysis.
"""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from utils.logging_config import get_logger

from ..config.config_schema import MARLConfig


class ExperimentStatus(Enum):
    """Experiment execution status."""

    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ExperimentType(Enum):
    """Types of experiments."""

    AB_TEST = "ab_test"
    PARAMETER_SWEEP = "parameter_sweep"
    ALGORITHM_COMPARISON = "algorithm_comparison"
    PERFORMANCE_BENCHMARK = "performance_benchmark"
    ABLATION_STUDY = "ablation_study"
    SCALABILITY_TEST = "scalability_test"


@dataclass
class ExperimentCondition:
    """
    Represents a single experimental condition.

    An experimental condition defines a specific configuration
    and parameters to be tested.
    """

    condition_id: str
    name: str
    description: str
    config: MARLConfig
    parameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate experiment condition."""
        if not self.condition_id:
            raise ValueError("Condition ID cannot be empty")
        if not self.name:
            raise ValueError("Condition name cannot be empty")


@dataclass
class ExperimentResult:
    """
    Represents the results of an experimental condition.

    Contains all metrics, performance data, and analysis
    results from running an experimental condition.
    """

    condition_id: str
    status: ExperimentStatus
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    performance_data: Dict[str, List[float]] = field(default_factory=dict)
    error_message: Optional[str] = None
    logs: List[str] = field(default_factory=list)
    artifacts: Dict[str, str] = field(
        default_factory=dict
    )  # artifact_name -> file_path

    def add_metric(self, name: str, value: Any, timestamp: Optional[datetime] = None):
        """Add a metric value."""
        if timestamp is None:
            timestamp = datetime.now()

        if name not in self.metrics:
            self.metrics[name] = []

        self.metrics[name].append({"value": value, "timestamp": timestamp.isoformat()})

    def add_performance_data(self, metric_name: str, values: List[float]):
        """Add performance data series."""
        if metric_name not in self.performance_data:
            self.performance_data[metric_name] = []

        self.performance_data[metric_name].extend(values)

    def get_final_metric(self, name: str) -> Optional[Any]:
        """Get the final (most recent) value of a metric."""
        if name not in self.metrics or not self.metrics[name]:
            return None

        return self.metrics[name][-1]["value"]

    def get_average_metric(self, name: str) -> Optional[float]:
        """Get the average value of a numeric metric."""
        if name not in self.metrics or not self.metrics[name]:
            return None

        try:
            values = [entry["value"] for entry in self.metrics[name]]
            numeric_values = [v for v in values if isinstance(v, (int, float))]

            if not numeric_values:
                return None

            return sum(numeric_values) / len(numeric_values)
        except (TypeError, ValueError):
            return None


@dataclass
class Experiment:
    """
    Represents a complete experiment with multiple conditions.

    An experiment contains multiple conditions to be tested,
    configuration for execution, and results analysis.
    """

    experiment_id: str
    name: str
    description: str
    experiment_type: ExperimentType
    conditions: List[ExperimentCondition]
    status: ExperimentStatus = ExperimentStatus.CREATED
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    results: Dict[str, ExperimentResult] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Execution configuration
    max_episodes_per_condition: int = 1000
    max_duration_per_condition: Optional[float] = None  # seconds
    parallel_execution: bool = False
    random_seed: Optional[int] = None

    # Analysis configuration
    significance_level: float = 0.05
    minimum_effect_size: float = 0.1

    def __post_init__(self):
        """Validate experiment."""
        if not self.experiment_id:
            raise ValueError("Experiment ID cannot be empty")
        if not self.name:
            raise ValueError("Experiment name cannot be empty")
        if not self.conditions:
            raise ValueError("Experiment must have at least one condition")

        # Initialize results for each condition
        for condition in self.conditions:
            if condition.condition_id not in self.results:
                self.results[condition.condition_id] = ExperimentResult(
                    condition_id=condition.condition_id, status=ExperimentStatus.CREATED
                )

    def get_condition(self, condition_id: str) -> Optional[ExperimentCondition]:
        """Get condition by ID."""
        for condition in self.conditions:
            if condition.condition_id == condition_id:
                return condition
        return None

    def get_result(self, condition_id: str) -> Optional[ExperimentResult]:
        """Get result by condition ID."""
        return self.results.get(condition_id)

    def is_completed(self) -> bool:
        """Check if experiment is completed."""
        return self.status == ExperimentStatus.COMPLETED

    def get_completion_percentage(self) -> float:
        """Get experiment completion percentage."""
        if not self.results:
            return 0.0

        completed_count = sum(
            1
            for result in self.results.values()
            if result.status in [ExperimentStatus.COMPLETED, ExperimentStatus.FAILED]
        )

        return (completed_count / len(self.results)) * 100.0


class ExperimentManager:
    """
    Comprehensive experiment manager for MARL systems.

    Manages experiment lifecycle, execution coordination,
    and result analysis for MARL research and optimization.
    """

    def __init__(
        self, experiments_dir: Optional[Union[str, Path]] = None, auto_save: bool = True
    ):
        """
        Initialize the experiment manager.

        Args:
            experiments_dir: Directory for storing experiments
            auto_save: Whether to automatically save experiments
        """
        self.logger = get_logger(__name__)
        self.experiments_dir = (
            Path(experiments_dir) if experiments_dir else Path("experiments")
        )
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self.auto_save = auto_save

        # Active experiments
        self._experiments: Dict[str, Experiment] = {}

        # Load existing experiments
        self._load_experiments()

        self.logger.info(
            "Experiment manager initialized with %d experiments", len(self._experiments)
        )

    def create_experiment(
        self,
        name: str,
        description: str,
        experiment_type: ExperimentType,
        conditions: List[ExperimentCondition],
        **kwargs,
    ) -> Experiment:
        """
        Create a new experiment.

        Args:
            name: Experiment name
            description: Experiment description
            experiment_type: Type of experiment
            conditions: List of experimental conditions
            **kwargs: Additional experiment parameters

        Returns:
            Created experiment
        """
        experiment_id = str(uuid.uuid4())

        experiment = Experiment(
            experiment_id=experiment_id,
            name=name,
            description=description,
            experiment_type=experiment_type,
            conditions=conditions,
            **kwargs,
        )

        self._experiments[experiment_id] = experiment

        if self.auto_save:
            self._save_experiment(experiment)

        self.logger.info("Created experiment: %s (%s)", name, experiment_id)
        return experiment

    def create_ab_test(
        self,
        name: str,
        description: str,
        control_config: MARLConfig,
        treatment_config: MARLConfig,
        **kwargs,
    ) -> Experiment:
        """
        Create an A/B test experiment.

        Args:
            name: Test name
            description: Test description
            control_config: Control group configuration
            treatment_config: Treatment group configuration
            **kwargs: Additional experiment parameters

        Returns:
            Created A/B test experiment
        """
        conditions = [
            ExperimentCondition(
                condition_id="control",
                name="Control",
                description="Control group configuration",
                config=control_config,
            ),
            ExperimentCondition(
                condition_id="treatment",
                name="Treatment",
                description="Treatment group configuration",
                config=treatment_config,
            ),
        ]

        return self.create_experiment(
            name=name,
            description=description,
            experiment_type=ExperimentType.AB_TEST,
            conditions=conditions,
            **kwargs,
        )

    def create_parameter_sweep(
        self,
        name: str,
        description: str,
        base_config: MARLConfig,
        parameter_ranges: Dict[str, List[Any]],
        **kwargs,
    ) -> Experiment:
        """
        Create a parameter sweep experiment.

        Args:
            name: Experiment name
            description: Experiment description
            base_config: Base configuration
            parameter_ranges: Dictionary of parameter names to value lists
            **kwargs: Additional experiment parameters

        Returns:
            Created parameter sweep experiment
        """
        conditions = []

        # Generate all parameter combinations
        import itertools

        param_names = list(parameter_ranges.keys())
        param_values = list(parameter_ranges.values())

        for i, combination in enumerate(itertools.product(*param_values)):
            condition_id = f"sweep_{i}"
            param_dict = dict(zip(param_names, combination))

            # Create condition name from parameters
            param_str = "_".join(f"{k}={v}" for k, v in param_dict.items())
            condition_name = f"Sweep {i}: {param_str}"

            conditions.append(
                ExperimentCondition(
                    condition_id=condition_id,
                    name=condition_name,
                    description=f"Parameter combination: {param_dict}",
                    config=base_config,  # Will be modified with parameters
                    parameters=param_dict,
                )
            )

        return self.create_experiment(
            name=name,
            description=description,
            experiment_type=ExperimentType.PARAMETER_SWEEP,
            conditions=conditions,
            **kwargs,
        )

    def create_algorithm_comparison(
        self,
        name: str,
        description: str,
        algorithm_configs: Dict[str, MARLConfig],
        **kwargs,
    ) -> Experiment:
        """
        Create an algorithm comparison experiment.

        Args:
            name: Experiment name
            description: Experiment description
            algorithm_configs: Dictionary of algorithm names to configurations
            **kwargs: Additional experiment parameters

        Returns:
            Created algorithm comparison experiment
        """
        conditions = []

        for algo_name, config in algorithm_configs.items():
            conditions.append(
                ExperimentCondition(
                    condition_id=algo_name.lower().replace(" ", "_"),
                    name=algo_name,
                    description=f"Algorithm: {algo_name}",
                    config=config,
                )
            )

        return self.create_experiment(
            name=name,
            description=description,
            experiment_type=ExperimentType.ALGORITHM_COMPARISON,
            conditions=conditions,
            **kwargs,
        )

    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """
        Get experiment by ID.

        Args:
            experiment_id: Experiment ID

        Returns:
            Experiment or None if not found
        """
        return self._experiments.get(experiment_id)

    def list_experiments(
        self,
        experiment_type: Optional[ExperimentType] = None,
        status: Optional[ExperimentStatus] = None,
    ) -> List[Experiment]:
        """
        List experiments with optional filtering.

        Args:
            experiment_type: Filter by experiment type
            status: Filter by status

        Returns:
            List of matching experiments
        """
        experiments = list(self._experiments.values())

        if experiment_type:
            experiments = [
                e for e in experiments if e.experiment_type == experiment_type
            ]

        if status:
            experiments = [e for e in experiments if e.status == status]

        return experiments

    def delete_experiment(self, experiment_id: str) -> bool:
        """
        Delete an experiment.

        Args:
            experiment_id: Experiment ID

        Returns:
            True if deleted successfully
        """
        if experiment_id not in self._experiments:
            return False

        experiment = self._experiments[experiment_id]

        # Don't delete running experiments
        if experiment.status == ExperimentStatus.RUNNING:
            self.logger.warning("Cannot delete running experiment: %s", experiment_id)
            return False

        # Remove from memory
        del self._experiments[experiment_id]

        # Remove from disk
        experiment_file = self.experiments_dir / f"{experiment_id}.json"
        if experiment_file.exists():
            experiment_file.unlink()

        self.logger.info("Deleted experiment: %s", experiment_id)
        return True

    def update_experiment_result(
        self, experiment_id: str, condition_id: str, result_update: Dict[str, Any]
    ) -> bool:
        """
        Update experiment result.

        Args:
            experiment_id: Experiment ID
            condition_id: Condition ID
            result_update: Result update data

        Returns:
            True if updated successfully
        """
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            return False

        result = experiment.get_result(condition_id)
        if not result:
            return False

        # Update result fields
        for key, value in result_update.items():
            if hasattr(result, key):
                setattr(result, key, value)

        # Auto-save if enabled
        if self.auto_save:
            self._save_experiment(experiment)

        return True

    def get_experiment_summary(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """
        Get experiment summary.

        Args:
            experiment_id: Experiment ID

        Returns:
            Experiment summary dictionary
        """
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            return None

        # Calculate summary statistics
        completed_conditions = [
            r
            for r in experiment.results.values()
            if r.status == ExperimentStatus.COMPLETED
        ]

        failed_conditions = [
            r
            for r in experiment.results.values()
            if r.status == ExperimentStatus.FAILED
        ]

        # Get common metrics
        common_metrics = set()
        for result in completed_conditions:
            common_metrics.update(result.metrics.keys())

        metric_summaries = {}
        for metric in common_metrics:
            values = []
            for result in completed_conditions:
                final_value = result.get_final_metric(metric)
                if final_value is not None and isinstance(final_value, (int, float)):
                    values.append(final_value)

            if values:
                metric_summaries[metric] = {
                    "count": len(values),
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "std": (
                        sum((x - sum(values) / len(values)) ** 2 for x in values)
                        / len(values)
                    )
                    ** 0.5,
                }

        return {
            "experiment_id": experiment.experiment_id,
            "name": experiment.name,
            "type": experiment.experiment_type.value,
            "status": experiment.status.value,
            "created_at": experiment.created_at.isoformat(),
            "started_at": experiment.started_at.isoformat()
            if experiment.started_at
            else None,
            "completed_at": experiment.completed_at.isoformat()
            if experiment.completed_at
            else None,
            "total_conditions": len(experiment.conditions),
            "completed_conditions": len(completed_conditions),
            "failed_conditions": len(failed_conditions),
            "completion_percentage": experiment.get_completion_percentage(),
            "metric_summaries": metric_summaries,
        }

    def export_experiment_results(
        self, experiment_id: str, output_path: Union[str, Path], format: str = "json"
    ) -> bool:
        """
        Export experiment results.

        Args:
            experiment_id: Experiment ID
            output_path: Output file path
            format: Export format ('json' or 'csv')

        Returns:
            True if exported successfully
        """
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            return False

        output_path = Path(output_path)

        try:
            if format.lower() == "json":
                export_data = {
                    "experiment": {
                        "id": experiment.experiment_id,
                        "name": experiment.name,
                        "description": experiment.description,
                        "type": experiment.experiment_type.value,
                        "status": experiment.status.value,
                        "created_at": experiment.created_at.isoformat(),
                        "metadata": experiment.metadata,
                    },
                    "conditions": [
                        {
                            "id": condition.condition_id,
                            "name": condition.name,
                            "description": condition.description,
                            "parameters": condition.parameters,
                            "metadata": condition.metadata,
                        }
                        for condition in experiment.conditions
                    ],
                    "results": {
                        condition_id: {
                            "status": result.status.value,
                            "start_time": result.start_time.isoformat()
                            if result.start_time
                            else None,
                            "end_time": result.end_time.isoformat()
                            if result.end_time
                            else None,
                            "duration_seconds": result.duration_seconds,
                            "metrics": result.metrics,
                            "performance_data": result.performance_data,
                            "error_message": result.error_message,
                        }
                        for condition_id, result in experiment.results.items()
                    },
                }

                with output_path.open("w", encoding="utf-8") as f:
                    json.dump(export_data, f, indent=2, default=str)

            elif format.lower() == "csv":
                import csv

                # Flatten results for CSV export
                rows = []
                for condition in experiment.conditions:
                    result = experiment.results.get(condition.condition_id)
                    if not result:
                        continue

                    base_row = {
                        "experiment_id": experiment.experiment_id,
                        "experiment_name": experiment.name,
                        "condition_id": condition.condition_id,
                        "condition_name": condition.name,
                        "status": result.status.value,
                        "duration_seconds": result.duration_seconds,
                        "error_message": result.error_message,
                    }

                    # Add parameters
                    for param_name, param_value in condition.parameters.items():
                        base_row[f"param_{param_name}"] = param_value

                    # Add final metric values
                    for metric_name in result.metrics:
                        final_value = result.get_final_metric(metric_name)
                        base_row[f"metric_{metric_name}"] = final_value

                    rows.append(base_row)

                if rows:
                    with output_path.open("w", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                        writer.writeheader()
                        writer.writerows(rows)

            else:
                self.logger.error("Unsupported export format: %s", format)
                return False

            self.logger.info("Exported experiment results to: %s", output_path)
            return True

        except Exception as e:
            self.logger.error("Failed to export experiment results: %s", str(e))
            return False

    def _save_experiment(self, experiment: Experiment):
        """Save experiment to disk."""
        try:
            experiment_file = self.experiments_dir / f"{experiment.experiment_id}.json"

            # Convert experiment to serializable format
            experiment_data = {
                "experiment_id": experiment.experiment_id,
                "name": experiment.name,
                "description": experiment.description,
                "experiment_type": experiment.experiment_type.value,
                "status": experiment.status.value,
                "created_at": experiment.created_at.isoformat(),
                "started_at": experiment.started_at.isoformat()
                if experiment.started_at
                else None,
                "completed_at": experiment.completed_at.isoformat()
                if experiment.completed_at
                else None,
                "metadata": experiment.metadata,
                "max_episodes_per_condition": experiment.max_episodes_per_condition,
                "max_duration_per_condition": experiment.max_duration_per_condition,
                "parallel_execution": experiment.parallel_execution,
                "random_seed": experiment.random_seed,
                "significance_level": experiment.significance_level,
                "minimum_effect_size": experiment.minimum_effect_size,
                "conditions": [
                    {
                        "condition_id": condition.condition_id,
                        "name": condition.name,
                        "description": condition.description,
                        "config": condition.config.to_dict(),
                        "parameters": condition.parameters,
                        "metadata": condition.metadata,
                    }
                    for condition in experiment.conditions
                ],
                "results": {
                    condition_id: {
                        "condition_id": result.condition_id,
                        "status": result.status.value,
                        "start_time": result.start_time.isoformat()
                        if result.start_time
                        else None,
                        "end_time": result.end_time.isoformat()
                        if result.end_time
                        else None,
                        "duration_seconds": result.duration_seconds,
                        "metrics": result.metrics,
                        "performance_data": result.performance_data,
                        "error_message": result.error_message,
                        "logs": result.logs,
                        "artifacts": result.artifacts,
                    }
                    for condition_id, result in experiment.results.items()
                },
            }

            with experiment_file.open("w", encoding="utf-8") as f:
                json.dump(experiment_data, f, indent=2, default=str)

        except Exception as e:
            self.logger.error(
                "Failed to save experiment %s: %s", experiment.experiment_id, str(e)
            )

    def _load_experiments(self):
        """Load experiments from disk."""
        try:
            for experiment_file in self.experiments_dir.glob("*.json"):
                try:
                    with experiment_file.open("r", encoding="utf-8") as f:
                        experiment_data = json.load(f)

                    # Reconstruct experiment object
                    experiment = self._reconstruct_experiment(experiment_data)
                    self._experiments[experiment.experiment_id] = experiment

                except Exception as e:
                    self.logger.warning(
                        "Failed to load experiment from %s: %s", experiment_file, str(e)
                    )

        except Exception as e:
            self.logger.error("Failed to load experiments: %s", str(e))

    def _reconstruct_experiment(self, data: Dict[str, Any]) -> Experiment:
        """Reconstruct experiment object from saved data."""
        from ..config.config_schema import MARLConfig

        # Reconstruct conditions
        conditions = []
        for condition_data in data["conditions"]:
            config = MARLConfig.from_dict(condition_data["config"])
            condition = ExperimentCondition(
                condition_id=condition_data["condition_id"],
                name=condition_data["name"],
                description=condition_data["description"],
                config=config,
                parameters=condition_data.get("parameters", {}),
                metadata=condition_data.get("metadata", {}),
            )
            conditions.append(condition)

        # Reconstruct results
        results = {}
        for condition_id, result_data in data["results"].items():
            result = ExperimentResult(
                condition_id=result_data["condition_id"],
                status=ExperimentStatus(result_data["status"]),
                start_time=datetime.fromisoformat(result_data["start_time"])
                if result_data["start_time"]
                else None,
                end_time=datetime.fromisoformat(result_data["end_time"])
                if result_data["end_time"]
                else None,
                duration_seconds=result_data.get("duration_seconds"),
                metrics=result_data.get("metrics", {}),
                performance_data=result_data.get("performance_data", {}),
                error_message=result_data.get("error_message"),
                logs=result_data.get("logs", []),
                artifacts=result_data.get("artifacts", {}),
            )
            results[condition_id] = result

        # Reconstruct experiment
        experiment = Experiment(
            experiment_id=data["experiment_id"],
            name=data["name"],
            description=data["description"],
            experiment_type=ExperimentType(data["experiment_type"]),
            conditions=conditions,
            status=ExperimentStatus(data["status"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            started_at=datetime.fromisoformat(data["started_at"])
            if data.get("started_at")
            else None,
            completed_at=datetime.fromisoformat(data["completed_at"])
            if data.get("completed_at")
            else None,
            results=results,
            metadata=data.get("metadata", {}),
            max_episodes_per_condition=data.get("max_episodes_per_condition", 1000),
            max_duration_per_condition=data.get("max_duration_per_condition"),
            parallel_execution=data.get("parallel_execution", False),
            random_seed=data.get("random_seed"),
            significance_level=data.get("significance_level", 0.05),
            minimum_effect_size=data.get("minimum_effect_size", 0.1),
        )

        return experiment


class ExperimentManagerFactory:
    """Factory for creating experiment managers."""

    @staticmethod
    def create(experiments_dir: Optional[Union[str, Path]] = None) -> ExperimentManager:
        """
        Create an experiment manager.

        Args:
            experiments_dir: Directory for storing experiments

        Returns:
            Experiment manager
        """
        return ExperimentManager(experiments_dir)

    @staticmethod
    def create_with_auto_save(
        experiments_dir: Optional[Union[str, Path]] = None, auto_save: bool = True
    ) -> ExperimentManager:
        """
        Create an experiment manager with auto-save configuration.

        Args:
            experiments_dir: Directory for storing experiments
            auto_save: Whether to automatically save experiments

        Returns:
            Experiment manager
        """
        return ExperimentManager(experiments_dir, auto_save)
