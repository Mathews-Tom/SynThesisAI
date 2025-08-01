"""
Service layer for handling the prompt generation pipeline.

This module provides functions for running the generation pipeline both
synchronously and as a background task, and for saving the results to the
database.
"""

# Standard Library
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict

# Third-Party Library
from fastapi import BackgroundTasks
from sqlalchemy.orm import Session

# SynThesisAI Modules
from app.models.database import SessionLocal
from app.models.schemas import BatchCreate, GenerationRequest, ProblemCreate
from app.services.batch_service import create_batch, update_batch_cost
from app.services.problem_service import create_problem
from core.runner import run_pipeline_from_config
from utils.similarity_utils import fetch_embedding

# Set up logging
logger = logging.getLogger(__name__)


def run_pipeline(request: GenerationRequest) -> Dict[str, Any]:
    """
    Run the prompt generation pipeline synchronously.

    Args:
        request: The request containing the generation parameters.

    Returns:
        A dictionary containing the results of the pipeline run.
    """
    config: Dict[str, Any] = {
        "num_problems": request.num_problems,
        "engineer_model": request.engineer_model.dict(),
        "checker_model": request.checker_model.dict(),
        "target_model": request.target_model.dict(),
        "use_search": request.use_search,
    }

    if request.use_seed_data:
        config["use_seed_data"] = True
        if request.benchmark_name:
            config["benchmark_name"] = request.benchmark_name
        if request.seed_data:
            config["seed_data"] = request.seed_data
    else:
        config["taxonomy"] = request.taxonomy

    return run_pipeline_from_config(config)


async def run_pipeline_background(batch_id: int, config: Dict[str, Any]) -> None:
    """
    Run the generation pipeline as a background task and save results.

    This function runs the blocking pipeline in a separate thread to avoid
    blocking the main application's event loop.

    Args:
        batch_id: The ID of the batch to associate the results with.
        config: The configuration dictionary for the pipeline.
    """
    db: Session = SessionLocal()
    try:
        # Run the blocking pipeline in a separate thread to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            result = await loop.run_in_executor(
                executor, run_pipeline_from_config, config
            )

        valid_prompts = result.get("valid_prompts", [])
        discarded_prompts = result.get("discarded_prompts", [])
        total_cost = result.get("total_cost", 0.0)

        # Update batch cost with the total cost from the pipeline
        update_batch_cost(db, batch_id, total_cost)

        # Save valid prompts to database
        for prompt in valid_prompts:
            _save_prompt_to_db(db, prompt, batch_id, "valid")

        for prompt in discarded_prompts:
            _save_prompt_to_db(db, prompt, batch_id, "discarded")

        logger.info(
            "Background pipeline completed for batch %d. Total cost: $%.6f",
            batch_id,
            total_cost,
        )

    except Exception as e:
        logger.error(
            "Error in background pipeline for batch %d: %s", batch_id, e, exc_info=True
        )
    finally:
        db.close()


def _save_prompt_to_db(
    db: Session, prompt: Dict[str, Any], batch_id: int, status: str
) -> None:
    """
    Helper function to save a single prompt to the database.

    Args:
        db: The database session.
        prompt: The prompt data dictionary.
        batch_id: The ID of the batch.
        status: The status of the prompt ('valid' or 'discarded').
    """
    # Fetch embedding for the problem text (even for discarded problems)
    problem_text = prompt.get("problem", "")
    problem_embedding = None
    if problem_text:
        try:
            embedding_list = fetch_embedding(problem_text)
            # Convert list to dict for database storage
            problem_embedding = {"embedding": embedding_list}
        except Exception as e:
            logger.warning("Failed to fetch embedding for problem: %s", e)

    problem_data = ProblemCreate(
        batch_id=batch_id,
        subject=prompt.get("subject", ""),
        topic=prompt.get("topic", ""),
        question=prompt.get("problem", ""),
        answer=prompt.get("answer", ""),
        hints=prompt.get("hints", {}),
        status=status,
        rejection_reason=prompt.get("rejection_reason"),
        target_model_answer=prompt.get("target_model_answer"),
        hints_were_corrected=prompt.get("hints_were_corrected", False),
        cost=Decimal("0.00"),  # Cost is handled at the batch level
        problem_embedding=problem_embedding,
        similar_problems=prompt.get("similar_problems", {}),
        reference=prompt.get("reference"),
    )
    create_problem(db, problem_data)


def start_generation_with_database(
    request: GenerationRequest, background_tasks: BackgroundTasks, db: Session
) -> Dict[str, Any]:
    """
    Create a batch record and start the generation pipeline in the background.

    Args:
        request: The request containing the generation parameters.
        background_tasks: FastAPI background tasks manager.
        db: The database session.

    Returns:
        A dictionary confirming that the generation has started.

    Raises:
        RuntimeError: If starting the generation fails.
    """
    try:
        # Create batch record with initial cost of 0.00
        batch_data = BatchCreate(
            name=(
                f"Batch_{request.target_model.model_name}_"
                f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            ),
            taxonomy_json=request.taxonomy or {},
            pipeline={
                "engineer_model": request.engineer_model.dict(),
                "checker_model": request.checker_model.dict(),
                "target_model": request.target_model.dict(),
                "use_seed_data": request.use_seed_data,
                "benchmark_name": request.benchmark_name,
            },
            num_problems=request.num_problems,
            batch_cost=Decimal("0.00"),
        )
        batch = create_batch(db, batch_data)

        # Prepare config for background task
        config: Dict[str, Any] = {
            "num_problems": request.num_problems,
            "engineer_model": request.engineer_model.dict(),
            "checker_model": request.checker_model.dict(),
            "target_model": request.target_model.dict(),
            "use_search": request.use_search,
        }

        # Add generation mode config
        if request.use_seed_data:
            config["use_seed_data"] = True
            config["benchmark_name"] = request.benchmark_name
            if request.seed_data:
                config["seed_data"] = request.seed_data
        else:
            config["taxonomy"] = request.taxonomy

        # Add generation mode config
        background_tasks.add_task(run_pipeline_background, batch.id, config)

        return {
            "status": "started",
            "batch_id": batch.id,
            "message": (
                f"Generation started for batch {batch.id}. "
                f"Generating {request.num_problems} valid problems."
            ),
            "total_cost": 0.00,
        }
    except Exception as e:
        logger.error("Failed to start generation: %s", e, exc_info=True)
        raise RuntimeError(f"Failed to start generation: {e}") from e
