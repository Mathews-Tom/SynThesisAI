# app/api/batches.py
"""API endpoints for managing batches."""

# Standard Library
import logging
from typing import Any, Dict, List, Optional

# Third-Party Library
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

# SynThesisAI Modules
from app.models.database import get_db
from app.models.models import Batch as BatchModel
from app.models.schemas import Batch, BatchWithStats, TargetModelUpdate
from app.services.batch_service import (
    delete_batch,
    get_batch,
    get_batches,
    get_problems_count,
    update_batch_target_model,
)
from app.services.problem_service import get_problem_stats

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/", response_model=List[BatchWithStats])
def get_all_batches(db: Session = Depends(get_db)) -> List[BatchWithStats]:
    """Retrieve all batches with their problem statistics.

    Args:
        db: The database session dependency.

    Returns:
        A list of batches, each including its statistics.

    Raises:
        HTTPException: If an internal server error occurs.
    """
    logger.info("Fetching all batches.")
    try:
        batches: List[BatchModel] = get_batches(db)
        result: List[BatchWithStats] = []
        for batch in batches:
            stats = get_problem_stats(db, batch.id)
            batch_dict = Batch.from_orm(batch).dict()
            batch_dict["stats"] = stats
            result.append(BatchWithStats(**batch_dict))
        logger.info("Successfully fetched %d batches.", len(result))
        return result
    except Exception as e:
        logger.error("Failed to fetch batches: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500, detail="Internal server error while fetching batches."
        ) from e


@router.get("/{batch_id}", response_model=BatchWithStats)
def get_batch_by_id(batch_id: int, db: Session = Depends(get_db)) -> BatchWithStats:
    """Retrieve a single batch by its ID, including problem statistics.

    Args:
        batch_id: The ID of the batch to retrieve.
        db: The database session dependency.

    Returns:
        The batch with its statistics.

    Raises:
        HTTPException: If the batch is not found or an internal error occurs.
    """
    logger.info("Fetching batch with ID: %d.", batch_id)
    batch: Optional[BatchModel] = get_batch(db, batch_id)
    if not batch:
        logger.warning("Batch with ID %d not found.", batch_id)
        raise HTTPException(
            status_code=404, detail=f"Batch with ID {batch_id} not found."
        )

    try:
        stats = get_problem_stats(db, batch_id)
        batch_dict = Batch.from_orm(batch).dict()
        batch_dict["stats"] = stats
        logger.info("Successfully fetched batch with ID: %d.", batch_id)
        return BatchWithStats(**batch_dict)
    except Exception as e:
        logger.error(
            "Failed to fetch stats for batch %d: %s", batch_id, e, exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error while fetching stats for batch {batch_id}.",
        ) from e


@router.delete("/{batch_id}", status_code=200)
def delete_batch_by_id(batch_id: int, db: Session = Depends(get_db)) -> Dict[str, str]:
    """Delete a batch by its ID.

    Args:
        batch_id: The ID of the batch to delete.
        db: The database session dependency.

    Returns:
        A confirmation message.

    Raises:
        HTTPException: If the batch with the specified ID is not found.
    """
    logger.info("Attempting to delete batch with ID: %d.", batch_id)
    success = delete_batch(db, batch_id)
    if not success:
        logger.warning(
            "Failed to delete batch with ID %d because it was not found.", batch_id
        )
        raise HTTPException(
            status_code=404, detail=f"Batch with ID {batch_id} not found."
        )

    logger.info("Successfully deleted batch with ID: %d.", batch_id)
    return {"message": f"Batch with ID {batch_id} deleted successfully."}


@router.patch("/{batch_id}/target-model", response_model=Dict[str, Any])
def update_target_model(
    batch_id: int, target_update: TargetModelUpdate, db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Update the target model for a specific batch.

    Args:
        batch_id: The ID of the batch to update.
        target_update: The new target model information.
        db: The database session dependency.

    Returns:
        A confirmation message along with the updated details.

    Raises:
        HTTPException: If the batch is not found or the update fails.
    """
    logger.info("Updating target model for batch ID: %d.", batch_id)
    batch: Optional[BatchModel] = get_batch(db, batch_id)
    if not batch:
        logger.warning(
            "Cannot update target model. Batch with ID %d not found.", batch_id
        )
        raise HTTPException(
            status_code=404, detail=f"Batch with ID {batch_id} not found."
        )

    try:
        target_model_data = target_update.target_model.dict()
        updated_batch = update_batch_target_model(db, batch_id, target_model_data)
        if not updated_batch:
            logger.error("Failed to update target model for batch ID: %d.", batch_id)
            raise HTTPException(
                status_code=500, detail="Failed to update target model."
            )

        logger.info("Successfully updated target model for batch ID: %d.", batch_id)
        return {
            "message": "Target model updated successfully",
            "batch_id": batch_id,
            "new_target_model": target_model_data,
        }
    except Exception as e:
        logger.error(
            "An unexpected error occurred while updating target model for batch %d: %s",
            batch_id,
            e,
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail="Internal server error during target model update."
        ) from e


@router.get("/problems/count", response_model=Dict[str, Any])
def get_problems_count_endpoint(
    batch_id: Optional[int] = Query(
        None, description="Optional batch ID to get count for."
    ),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Get the number of problems, either for a specific batch or total across all batches.

    Args:
        batch_id: Optional ID of the batch to get the problem count for.
        db: The database session dependency.

    Returns:
        A dictionary containing the problem count.

    Raises:
        HTTPException: If a specific batch_id is provided but the batch is not found.
    """
    if batch_id is not None:
        logger.info("Getting problem count for batch ID: %d.", batch_id)
        batch: Optional[BatchModel] = get_batch(db, batch_id)
        if not batch:
            logger.warning(
                "Cannot get problem count. Batch with ID %d not found.", batch_id
            )
            raise HTTPException(
                status_code=404, detail=f"Batch with ID {batch_id} not found."
            )

        result = get_problems_count(db, batch_id)
        result["batch_name"] = batch.name
        logger.info(
            "Problem count for batch %d is %d.", batch_id, result.get("count", 0)
        )
        return result
    else:
        logger.info("Getting total problem count across all batches.")
        total_count = get_problems_count(db)
        logger.info("Total problem count is %d.", total_count.get("total_count", 0))
        return total_count
