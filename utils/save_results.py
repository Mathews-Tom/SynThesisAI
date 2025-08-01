import json
from pathlib import Path
from typing import Any, List, Optional

from utils.costs import CostTracker
from utils.logging_config import get_logger

logger = get_logger(__name__)


def save_prompts(
    valid_list: List[Any],
    discarded_list: List[Any],
    save_path: Path | str,
    cost_tracker: Optional[CostTracker] = None,
) -> None:
    """
    Save valid and discarded prompts to JSON files, along with costs if provided.

    Args:
        valid_list: List of valid prompts.
        discarded_list: List of discarded prompts.
        save_path: Directory path to save results.
        cost_tracker: Optional CostTracker instance to save cost data.
    """
    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    valid_file = save_dir / "valid.json"
    with open(valid_file, "w", encoding="utf-8") as f:
        json.dump(valid_list, f, indent=2, ensure_ascii=False)

    discarded_file = save_dir / "discarded.json"
    with open(discarded_file, "w", encoding="utf-8") as f:
        json.dump(discarded_list, f, indent=2, ensure_ascii=False)

    if cost_tracker:
        run_id = save_dir.name
        costs_file = save_dir / "costs.json"
        with open(costs_file, "w", encoding="utf-8") as f:
            json.dump(cost_tracker.as_dict(run_id=run_id), f, indent=2, ensure_ascii=False)

    logger.info("Results saved to: %s", save_dir)
    logger.info("valid.json (%d entries)", len(valid_list))
    logger.info("discarded.json (%d entries)", len(discarded_list))
    if cost_tracker:
        total_cost = cost_tracker.get_total_cost()
        logger.info("costs.json saved (Total: $%.6f)", total_cost)
