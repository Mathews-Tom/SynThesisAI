"""
MARL Research Logger.

This module provides comprehensive research data logging capabilities
for MARL experiments, including structured data collection and export.
"""

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from utils.logging_config import get_logger


class ResearchDataPoint:
    """
    Represents a single research data point.

    Contains timestamped data with metadata for research analysis.
    """

    def __init__(
        self,
        timestamp: datetime,
        experiment_id: str,
        condition_id: str,
        data_type: str,
        data: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize research data point.

        Args:
            timestamp: When the data was collected
            experiment_id: Associated experiment ID
            condition_id: Associated condition ID
            data_type: Type of data (e.g., 'metric', 'state', 'action')
            data: The actual data
            metadata: Additional metadata
        """
        self.timestamp = timestamp
        self.experiment_id = experiment_id
        self.condition_id = condition_id
        self.data_type = data_type
        self.data = data
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "experiment_id": self.experiment_id,
            "condition_id": self.condition_id,
            "data_type": self.data_type,
            "data": self._serialize_data(self.data),
            "metadata": self.metadata,
        }

    def _serialize_data(self, data: Any) -> Any:
        """Serialize data for JSON storage."""
        if isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, (np.integer, np.floating)):
            return data.item()
        elif hasattr(data, "to_dict"):
            return data.to_dict()
        else:
            return data


class ResearchLogger:
    """
    Comprehensive research logger for MARL experiments.

    Provides structured logging of experimental data, metrics,
    and system states for research analysis and reproducibility.
    """

    def __init__(
        self,
        output_dir: Union[str, Path],
        experiment_id: str,
        auto_flush: bool = True,
        buffer_size: int = 1000,
    ):
        """
        Initialize the research logger.

        Args:
            output_dir: Directory for output files
            experiment_id: Associated experiment ID
            auto_flush: Whether to automatically flush data to disk
            buffer_size: Number of data points to buffer before flushing
        """
        self.logger = get_logger(__name__)
        self.output_dir = Path(output_dir)
        self.experiment_id = experiment_id
        self.auto_flush = auto_flush
        self.buffer_size = buffer_size

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Data storage
        self._data_buffer: List[ResearchDataPoint] = []
        self._data_counts: Dict[str, int] = {}

        # File handles
        self._json_file = None
        self._csv_file = None

        # Initialize output files
        self._initialize_output_files()

        self.logger.info(
            "Research logger initialized for experiment: %s", experiment_id
        )

    def _initialize_output_files(self):
        """Initialize output files."""
        try:
            # JSON file for structured data
            json_path = self.output_dir / f"{self.experiment_id}_research_data.json"
            self._json_file = json_path.open("w", encoding="utf-8")
            self._json_file.write("[\n")  # Start JSON array

            self.logger.debug("Initialized JSON output: %s", json_path)

        except Exception as e:
            self.logger.error("Failed to initialize output files: %s", str(e))

    def log_metric(
        self,
        condition_id: str,
        metric_name: str,
        value: Union[float, int],
        episode: Optional[int] = None,
        step: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Log a metric value.

        Args:
            condition_id: Condition ID
            metric_name: Name of the metric
            value: Metric value
            episode: Episode number (optional)
            step: Step number (optional)
            metadata: Additional metadata
        """
        data = {
            "metric_name": metric_name,
            "value": value,
            "episode": episode,
            "step": step,
        }

        self._log_data_point(
            condition_id=condition_id, data_type="metric", data=data, metadata=metadata
        )

    def log_agent_state(
        self,
        condition_id: str,
        agent_id: str,
        state: Any,
        episode: int,
        step: int,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Log agent state information.

        Args:
            condition_id: Condition ID
            agent_id: Agent ID
            state: Agent state
            episode: Episode number
            step: Step number
            metadata: Additional metadata
        """
        data = {"agent_id": agent_id, "state": state, "episode": episode, "step": step}

        self._log_data_point(
            condition_id=condition_id,
            data_type="agent_state",
            data=data,
            metadata=metadata,
        )

    def log_agent_action(
        self,
        condition_id: str,
        agent_id: str,
        action: Any,
        episode: int,
        step: int,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Log agent action information.

        Args:
            condition_id: Condition ID
            agent_id: Agent ID
            action: Agent action
            episode: Episode number
            step: Step number
            metadata: Additional metadata
        """
        data = {
            "agent_id": agent_id,
            "action": action,
            "episode": episode,
            "step": step,
        }

        self._log_data_point(
            condition_id=condition_id,
            data_type="agent_action",
            data=data,
            metadata=metadata,
        )

    def log_coordination_event(
        self,
        condition_id: str,
        coordination_id: str,
        event_type: str,
        participants: List[str],
        result: Optional[Any] = None,
        episode: Optional[int] = None,
        step: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Log coordination event.

        Args:
            condition_id: Condition ID
            coordination_id: Coordination ID
            event_type: Type of event ('start', 'end', 'timeout', etc.)
            participants: List of participating agent IDs
            result: Coordination result (optional)
            episode: Episode number (optional)
            step: Step number (optional)
            metadata: Additional metadata
        """
        data = {
            "coordination_id": coordination_id,
            "event_type": event_type,
            "participants": participants,
            "result": result,
            "episode": episode,
            "step": step,
        }

        self._log_data_point(
            condition_id=condition_id,
            data_type="coordination_event",
            data=data,
            metadata=metadata,
        )

    def log_learning_update(
        self,
        condition_id: str,
        agent_id: str,
        update_type: str,
        loss: Optional[float] = None,
        gradient_norm: Optional[float] = None,
        episode: Optional[int] = None,
        step: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Log learning update information.

        Args:
            condition_id: Condition ID
            agent_id: Agent ID
            update_type: Type of update ('policy', 'value', 'target', etc.)
            loss: Training loss (optional)
            gradient_norm: Gradient norm (optional)
            episode: Episode number (optional)
            step: Step number (optional)
            metadata: Additional metadata
        """
        data = {
            "agent_id": agent_id,
            "update_type": update_type,
            "loss": loss,
            "gradient_norm": gradient_norm,
            "episode": episode,
            "step": step,
        }

        self._log_data_point(
            condition_id=condition_id,
            data_type="learning_update",
            data=data,
            metadata=metadata,
        )

    def log_system_event(
        self,
        condition_id: str,
        event_type: str,
        event_data: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Log system-level event.

        Args:
            condition_id: Condition ID
            event_type: Type of event
            event_data: Event data
            metadata: Additional metadata
        """
        data = {"event_type": event_type, "event_data": event_data}

        self._log_data_point(
            condition_id=condition_id,
            data_type="system_event",
            data=data,
            metadata=metadata,
        )

    def log_custom_data(
        self,
        condition_id: str,
        data_type: str,
        data: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Log custom research data.

        Args:
            condition_id: Condition ID
            data_type: Custom data type
            data: Data to log
            metadata: Additional metadata
        """
        self._log_data_point(
            condition_id=condition_id, data_type=data_type, data=data, metadata=metadata
        )

    def _log_data_point(
        self,
        condition_id: str,
        data_type: str,
        data: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Log a data point to the buffer."""
        data_point = ResearchDataPoint(
            timestamp=datetime.now(),
            experiment_id=self.experiment_id,
            condition_id=condition_id,
            data_type=data_type,
            data=data,
            metadata=metadata,
        )

        self._data_buffer.append(data_point)

        # Update counts
        self._data_counts[data_type] = self._data_counts.get(data_type, 0) + 1

        # Auto-flush if buffer is full
        if self.auto_flush and len(self._data_buffer) >= self.buffer_size:
            self.flush()

    def flush(self):
        """Flush buffered data to disk."""
        if not self._data_buffer:
            return

        try:
            # Write to JSON file
            if self._json_file:
                for i, data_point in enumerate(self._data_buffer):
                    if i > 0 or self._json_file.tell() > 2:  # Not first entry
                        self._json_file.write(",\n")

                    json.dump(
                        data_point.to_dict(), self._json_file, indent=2, default=str
                    )

                self._json_file.flush()

            self.logger.debug("Flushed %d data points to disk", len(self._data_buffer))

            # Clear buffer
            self._data_buffer.clear()

        except Exception as e:
            self.logger.error("Failed to flush data: %s", str(e))

    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary of logged data."""
        return {
            "experiment_id": self.experiment_id,
            "total_data_points": sum(self._data_counts.values()),
            "data_types": dict(self._data_counts),
            "buffer_size": len(self._data_buffer),
            "output_directory": str(self.output_dir),
        }

    def export_to_csv(
        self,
        data_types: Optional[List[str]] = None,
        output_file: Optional[Union[str, Path]] = None,
    ) -> bool:
        """
        Export logged data to CSV format.

        Args:
            data_types: Specific data types to export (None for all)
            output_file: Output CSV file path

        Returns:
            True if export successful
        """
        try:
            # Flush any remaining data
            self.flush()

            # Read JSON data
            json_path = self.output_dir / f"{self.experiment_id}_research_data.json"
            if not json_path.exists():
                self.logger.warning("No data file found for export")
                return False

            # Load data
            with json_path.open("r", encoding="utf-8") as f:
                content = f.read().strip()
                if content.endswith(","):
                    content = content[:-1]  # Remove trailing comma
                content += "\n]"  # Close JSON array

                # Parse JSON
                data_points = json.loads(content)

            # Filter by data types if specified
            if data_types:
                data_points = [
                    dp for dp in data_points if dp["data_type"] in data_types
                ]

            if not data_points:
                self.logger.warning("No data points to export")
                return False

            # Convert to DataFrame
            rows = []
            for dp in data_points:
                row = {
                    "timestamp": dp["timestamp"],
                    "experiment_id": dp["experiment_id"],
                    "condition_id": dp["condition_id"],
                    "data_type": dp["data_type"],
                }

                # Flatten data and metadata
                if isinstance(dp["data"], dict):
                    for key, value in dp["data"].items():
                        row[f"data_{key}"] = value
                else:
                    row["data_value"] = dp["data"]

                if dp["metadata"]:
                    for key, value in dp["metadata"].items():
                        row[f"metadata_{key}"] = value

                rows.append(row)

            df = pd.DataFrame(rows)

            # Determine output file
            if output_file is None:
                output_file = (
                    self.output_dir / f"{self.experiment_id}_research_data.csv"
                )
            else:
                output_file = Path(output_file)

            # Export to CSV
            df.to_csv(output_file, index=False)

            self.logger.info(
                "Exported %d data points to CSV: %s", len(rows), output_file
            )
            return True

        except Exception as e:
            self.logger.error("Failed to export to CSV: %s", str(e))
            return False

    def export_to_pickle(self, output_file: Optional[Union[str, Path]] = None) -> bool:
        """
        Export logged data to pickle format for Python analysis.

        Args:
            output_file: Output pickle file path

        Returns:
            True if export successful
        """
        try:
            # Flush any remaining data
            self.flush()

            # Read JSON data
            json_path = self.output_dir / f"{self.experiment_id}_research_data.json"
            if not json_path.exists():
                self.logger.warning("No data file found for export")
                return False

            # Load data
            with json_path.open("r", encoding="utf-8") as f:
                content = f.read().strip()
                if content.endswith(","):
                    content = content[:-1]  # Remove trailing comma
                content += "\n]"  # Close JSON array

                # Parse JSON
                data_points = json.loads(content)

            # Determine output file
            if output_file is None:
                output_file = (
                    self.output_dir / f"{self.experiment_id}_research_data.pkl"
                )
            else:
                output_file = Path(output_file)

            # Export to pickle
            with output_file.open("wb") as f:
                pickle.dump(data_points, f)

            self.logger.info(
                "Exported %d data points to pickle: %s", len(data_points), output_file
            )
            return True

        except Exception as e:
            self.logger.error("Failed to export to pickle: %s", str(e))
            return False

    def create_analysis_notebook(
        self, notebook_path: Optional[Union[str, Path]] = None
    ) -> bool:
        """
        Create a Jupyter notebook template for data analysis.

        Args:
            notebook_path: Path for the notebook file

        Returns:
            True if notebook created successfully
        """
        try:
            if notebook_path is None:
                notebook_path = self.output_dir / f"{self.experiment_id}_analysis.ipynb"
            else:
                notebook_path = Path(notebook_path)

            # Create notebook template
            notebook_content = {
                "cells": [
                    {
                        "cell_type": "markdown",
                        "metadata": {},
                        "source": [
                            f"# MARL Experiment Analysis: {self.experiment_id}\n\n",
                            "This notebook provides analysis templates for the research data.\n",
                        ],
                    },
                    {
                        "cell_type": "code",
                        "execution_count": None,
                        "metadata": {},
                        "source": [
                            "import json\n",
                            "import pandas as pd\n",
                            "import numpy as np\n",
                            "import matplotlib.pyplot as plt\n",
                            "import seaborn as sns\n",
                            "from pathlib import Path\n\n",
                            "# Set up plotting\n",
                            "plt.style.use('seaborn-v0_8')\n",
                            "sns.set_palette('husl')\n",
                        ],
                    },
                    {
                        "cell_type": "code",
                        "execution_count": None,
                        "metadata": {},
                        "source": [
                            "# Load research data\n",
                            f"data_file = Path('{self.output_dir}') / '{self.experiment_id}_research_data.json'\n",
                            "with data_file.open('r') as f:\n",
                            "    content = f.read().strip()\n",
                            "    if content.endswith(','):\n",
                            "        content = content[:-1]\n",
                            "    content += '\\n]'\n",
                            "    data_points = json.loads(content)\n\n",
                            "print(f'Loaded {len(data_points)} data points')\n",
                        ],
                    },
                    {
                        "cell_type": "code",
                        "execution_count": None,
                        "metadata": {},
                        "source": [
                            "# Data summary\n",
                            "data_types = {}\n",
                            "for dp in data_points:\n",
                            "    dt = dp['data_type']\n",
                            "    data_types[dt] = data_types.get(dt, 0) + 1\n\n",
                            "print('Data types:')\n",
                            "for dt, count in data_types.items():\n",
                            "    print(f'  {dt}: {count}')\n",
                        ],
                    },
                    {
                        "cell_type": "markdown",
                        "metadata": {},
                        "source": ["## Metric Analysis\n"],
                    },
                    {
                        "cell_type": "code",
                        "execution_count": None,
                        "metadata": {},
                        "source": [
                            "# Extract metrics\n",
                            "metrics_data = [dp for dp in data_points if dp['data_type'] == 'metric']\n",
                            "metrics_df = pd.DataFrame([\n",
                            "    {\n",
                            "        'timestamp': dp['timestamp'],\n",
                            "        'condition_id': dp['condition_id'],\n",
                            "        'metric_name': dp['data']['metric_name'],\n",
                            "        'value': dp['data']['value'],\n",
                            "        'episode': dp['data'].get('episode'),\n",
                            "        'step': dp['data'].get('step')\n",
                            "    }\n",
                            "    for dp in metrics_data\n",
                            "])\n\n",
                            "print(f'Metrics data shape: {metrics_df.shape}')\n",
                            "print('Available metrics:', metrics_df['metric_name'].unique())\n",
                        ],
                    },
                    {
                        "cell_type": "code",
                        "execution_count": None,
                        "metadata": {},
                        "source": [
                            "# Plot metrics over time\n",
                            "fig, axes = plt.subplots(2, 2, figsize=(15, 10))\n",
                            "axes = axes.flatten()\n\n",
                            "unique_metrics = metrics_df['metric_name'].unique()[:4]  # First 4 metrics\n",
                            "for i, metric in enumerate(unique_metrics):\n",
                            "    metric_data = metrics_df[metrics_df['metric_name'] == metric]\n",
                            "    \n",
                            "    for condition in metric_data['condition_id'].unique():\n",
                            "        condition_data = metric_data[metric_data['condition_id'] == condition]\n",
                            "        axes[i].plot(condition_data['episode'], condition_data['value'], \n",
                            "                    label=condition, alpha=0.7)\n",
                            "    \n",
                            "    axes[i].set_title(f'{metric}')\n",
                            "    axes[i].set_xlabel('Episode')\n",
                            "    axes[i].set_ylabel('Value')\n",
                            "    axes[i].legend()\n",
                            "    axes[i].grid(True, alpha=0.3)\n\n",
                            "plt.tight_layout()\n",
                            "plt.show()\n",
                        ],
                    },
                ],
                "metadata": {
                    "kernelspec": {
                        "display_name": "Python 3",
                        "language": "python",
                        "name": "python3",
                    },
                    "language_info": {"name": "python", "version": "3.8.0"},
                },
                "nbformat": 4,
                "nbformat_minor": 4,
            }

            # Write notebook
            with notebook_path.open("w", encoding="utf-8") as f:
                json.dump(notebook_content, f, indent=2)

            self.logger.info("Created analysis notebook: %s", notebook_path)
            return True

        except Exception as e:
            self.logger.error("Failed to create analysis notebook: %s", str(e))
            return False

    def close(self):
        """Close the research logger and finalize files."""
        try:
            # Flush any remaining data
            self.flush()

            # Close JSON file
            if self._json_file:
                self._json_file.write("\n]")  # Close JSON array
                self._json_file.close()
                self._json_file = None

            self.logger.info(
                "Research logger closed for experiment: %s", self.experiment_id
            )

        except Exception as e:
            self.logger.error("Error closing research logger: %s", str(e))

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


class ResearchLoggerFactory:
    """Factory for creating research loggers."""

    @staticmethod
    def create(output_dir: Union[str, Path], experiment_id: str) -> ResearchLogger:
        """
        Create a research logger.

        Args:
            output_dir: Output directory
            experiment_id: Experiment ID

        Returns:
            Research logger
        """
        return ResearchLogger(output_dir, experiment_id)

    @staticmethod
    def create_with_config(
        output_dir: Union[str, Path],
        experiment_id: str,
        auto_flush: bool = True,
        buffer_size: int = 1000,
    ) -> ResearchLogger:
        """
        Create a research logger with custom configuration.

        Args:
            output_dir: Output directory
            experiment_id: Experiment ID
            auto_flush: Whether to auto-flush data
            buffer_size: Buffer size before flushing

        Returns:
            Configured research logger
        """
        return ResearchLogger(output_dir, experiment_id, auto_flush, buffer_size)
