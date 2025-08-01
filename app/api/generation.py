# -*- coding: utf-8 -*-
"""
This module defines the API endpoints for managing and monitoring the generation of math problems.

It provides endpoints to initiate the generation process for a new batch of problems and
to check the status of an ongoing generation task. The API handles asynchronous processing
of generation tasks using background workers and provides real-time progress updates.
"""

# Standard Library
import logging
from typing import Any, Dict

# Third-Party Library
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.orm import Session

# SynThesisAI Modules
from app.models.database import get_db
from app.models.schemas import GenerationRequest, GenerationStatus
from app.services.batch_service import get_batch
from app.services.pipeline_service import start_generation_with_database
from app.services.problem_service import get_problem_stats

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/", status_code=202)
async def start_generation(
    request: GenerationRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """
    Starts the asynchronous generation of a new batch of problems.

    This endpoint initiates the problem generation pipeline for the specified batch
    configuration. The actual generation is performed in the background to avoid
    blocking the API response.

    Args:
        request: The request payload containing the batch generation parameters.
        background_tasks: FastAPI's background task runner.
        db: The database session dependency.

    Returns:
        A dictionary containing the initial status and details of the generation task.

    Raises:
        HTTPException: If there is an error initiating the generation process.
    """
    try:
        logger.info(
            "Received request to start generation for batch: %s", request.batch_name
        )
        result = start_generation_with_database(request, background_tasks, db)
        return result
    except Exception as e:
        logger.error(
            "Failed to start generation for batch '%s': %s",
            request.batch_name,
            e,
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail=f"Failed to start generation process: {e}"
        ) from e


@router.get("/status/{batch_id}", response_model=GenerationStatus)
async def get_generation_status(
    batch_id: int, db: Session = Depends(get_db)
) -> GenerationStatus:
    """
    Retrieves the current status of a problem generation batch.

    This endpoint provides detailed statistics about the generation progress for a given
    batch ID, including the number of problems needed, generated, and validated.

    Args:
        batch_id: The unique identifier for the batch.
        db: The database session dependency.

    Returns:
        A GenerationStatus object with detailed progress information.

    Raises:
        HTTPException: If the batch with the specified ID is not found.
    """
    logger.info("Fetching generation status for batch_id: %d", batch_id)
    batch = get_batch(db, batch_id)
    if not batch:
        logger.warning("Batch with id %d not found.", batch_id)
        raise HTTPException(
            status_code=404, detail=f"Batch with id {batch_id} not found."
        )

    stats = get_problem_stats(db, batch_id)
    total_generated = stats["discarded"] + stats["solved"] + stats["valid"]
    progress = (
        (stats["valid"] / batch.num_problems * 100) if batch.num_problems > 0 else 0.0
    )

    status = "completed" if stats["valid"] >= batch.num_problems else "in_progress"

    return GenerationStatus(
        batch_id=batch_id,
        total_needed=batch.num_problems,
        valid_generated=stats["valid"],
        total_generated=total_generated,
        progress_percentage=round(progress, 2),
        stats=stats,
        batch_cost=float(batch.batch_cost),
        status=status,
    )
