from pathlib import Path
import json
from typing import Any, Dict, List

# Map benchmark names to local file paths
BENCHMARK_PATHS: Dict[str, str] = {
    "AIME": "taxonomy/benchmarks/AIME.json",
    "HMMT": "taxonomy/benchmarks/HMMT.json",
    "GPQA_DIAMOND": "taxonomy/benchmarks/GPQA_DIAMOND.json",
}


def load_benchmark(name: str) -> List[Dict[str, Any]]:
    """
    Load a benchmark dataset by name.

    Args:
        name: Name of the benchmark.

    Returns:
        List of dicts, each with 'problem' and 'answer' keys.

    Raises:
        ValueError: If the benchmark name is unknown or data is invalid.
        FileNotFoundError: If the benchmark file does not exist.
    """
    if name not in BENCHMARK_PATHS:
        raise ValueError(f"Unknown benchmark: {name}")

    benchmark_file = Path(__file__).parent.parent / BENCHMARK_PATHS[name]

    if not benchmark_file.exists():
        raise FileNotFoundError(f"Benchmark file not found: {benchmark_file}")

    with benchmark_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"{name} benchmark must be a list of problems")

    if not all(isinstance(item, dict) and "problem" in item and "answer" in item for item in data):
        raise ValueError(f"{name} benchmark data is not normalized")

    return data
