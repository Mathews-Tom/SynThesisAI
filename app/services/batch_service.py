"""
Service layer for handling batch-related operations.

This module provides functions for creating, retrieving, updating, and deleting
batches in the database.
"""

# Standard Library
import logging
from typing import Dict, List, Optional

# Third-Party Library
from sqlalchemy.orm import Session, exc

# SynThesisAI Modules
from app.models.models import Batch, Problem
from app.models.schemas import BatchCreate

# Initialize logger
logger = logging.getLogger(__name__)


class BatchServiceError(Exception):
    """Base exception class for batch service errors."""


class BatchNotFoundError(BatchServiceError):
    """Exception raised when a batch is not found."""


def create_batch(db: Session, batch: BatchCreate) -> Batch:
    """
    Creates a new batch in the database.

    Args:
        db: The database session.
        batch: The batch data to create.

    Returns:
        The newly created batch.
    """
    db_batch = Batch(**batch.dict())
    db.add(db_batch)
    db.commit()
    db.refresh(db_batch)
    logger.info("Successfully created batch with ID: %d", db_batch.id)
    return db_batch


def get_batch(db: Session, batch_id: int) -> Optional[Batch]:
    """
    Retrieves a single batch by its ID.

    Args:
        db: The database session.
        batch_id: The ID of the batch to retrieve.

    Returns:
        The batch object if found, otherwise None.
    """
    return db.query(Batch).filter(Batch.id == batch_id).first()


def get_batches(db: Session, skip: int = 0, limit: int = 100) -> List[Batch]:
    """
    Retrieves a list of batches with pagination.

    Args:
        db: The database session.
        skip: The number of batches to skip.
        limit: The maximum number of batches to return.

    Returns:
        A list of batch objects.
    """
    return db.query(Batch).offset(skip).limit(limit).all()


def delete_batch(db: Session, batch_id: int) -> bool:
    """
    Deletes a batch from the database.

    Args:
        db: The database session.
        batch_id: The ID of the batch to delete.

    Returns:
        True if the batch was deleted, False otherwise.

    Raises:
        BatchNotFoundError: If the batch with the specified ID is not found.
    """
    try:
        batch = db.query(Batch).filter(Batch.id == batch_id).one()
        db.delete(batch)
        db.commit()
        logger.info("Successfully deleted batch with ID: %d", batch_id)
        return True
    except exc.NoResultFound:
        logger.warning("Attempted to delete non-existent batch with ID: %d", batch_id)
        raise BatchNotFoundError(f"Batch with ID {batch_id} not found.")


def update_batch_cost(db: Session, batch_id: int, cost: float) -> Optional[Batch]:
    """
    Updates the cost of a batch.

    Args:
        db: The database session.
        batch_id: The ID of the batch to update.
        cost: The new cost of the batch.

    Returns:
        The updated batch object if found, otherwise None.

    Raises:
        BatchNotFoundError: If the batch with the specified ID is not found.
    """
    batch = get_batch(db, batch_id)
    if not batch:
        raise BatchNotFoundError(f"Batch with ID {batch_id} not found.")

    batch.batch_cost = cost
    db.commit()
    db.refresh(batch)
    logger.info("Successfully updated cost for batch with ID: %d", batch_id)
    return batch


def update_batch_target_model(
    db: Session, batch_id: int, target_model: Dict
) -> Optional[Batch]:
    """
    Updates the target model configuration for a batch.

    Args:
        db: The database session.
        batch_id: The ID of the batch to update.
        target_model: The new target model configuration.

    Returns:
        The updated batch object if found, otherwise None.

    Raises:
        BatchNotFoundError: If the batch with the specified ID is not found.
    """
    batch = get_batch(db, batch_id)
    if not batch:
        raise BatchNotFoundError(f"Batch with ID {batch_id} not found.")

    pipeline = batch.pipeline.copy()
    pipeline["target_model"] = target_model
    batch.pipeline = pipeline
    db.commit()
    db.refresh(batch)
    logger.info("Successfully updated target model for batch ID: %d", batch_id)
    return batch


def get_problems_count(db: Session, batch_id: Optional[int] = None) -> Dict[str, int]:
    """
    Gets the number of problems for a specific batch or all batches.

    Args:
        db: The database session.
        batch_id: The ID of the batch to count problems for. If None,
                    counts problems across all batches.

    Returns:
        A dictionary containing the batch ID and problem count, or the
        total problem count.
    """
    if batch_id is not None:
        count = db.query(Problem).filter(Problem.batch_id == batch_id).count()
        return {"batch_id": batch_id, "problems_count": count}

    total_count = db.query(Problem).count()
    return {"total_problems_count": total_count}
