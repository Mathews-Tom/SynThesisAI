# -*- coding: utf-8 -*-
"""
This module defines the API endpoints for managing problems.

It provides routes for retrieving individual problems, lists of problems with pagination,
and all problems associated with a specific batch.
"""

# Standard Library
import logging
from typing import List, Optional

# Third-Party Library
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

# SynThesisAI Modules
from app.models.database import get_db
from app.models.schemas import Problem as ProblemResponse
from app.services.problem_service import (
    get_problem,
    get_problems,
    get_problems_by_batch,
)

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/", response_model=List[ProblemResponse])
def read_problems(
    batch_id: Optional[int] = Query(None, description="Filter problems by batch ID."),
    status: Optional[str] = Query(None, description="Filter problems by status."),
    skip: int = Query(0, ge=0, description="Number of problems to skip for pagination."),
    limit: int = Query(1000, ge=1, le=10000, description="Maximum number of problems to return."),
    db: Session = Depends(get_db),
) -> List[ProblemResponse]:
    """
    Retrieve a list of problems with optional filtering and pagination.

    Args:
        batch_id: An optional batch ID to filter the problems.
        status: An optional status to filter the problems.
        skip: The number of records to skip.
        limit: The maximum number of records to return.
        db: The database session dependency.

    Returns:
        A list of problem objects matching the criteria.
    """
    try:
        problems = get_problems(db, skip=skip, limit=limit, batch_id=batch_id, status=status)
        return problems
    except Exception as e:
        logger.error("Failed to retrieve problems: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while fetching problems.",
        ) from e


@router.get("/problem/{problem_id}", response_model=ProblemResponse)
def read_problem_by_id(problem_id: int, db: Session = Depends(get_db)) -> ProblemResponse:
    """
    Retrieve a single problem by its unique ID.

    Args:
        problem_id: The unique identifier of the problem.
        db: The database session dependency.

    Returns:
        The problem object if found.

    Raises:
        HTTPException: If the problem with the specified ID is not found.
    """
    problem = get_problem(db, problem_id)
    if not problem:
        raise HTTPException(status_code=404, detail=f"Problem with ID {problem_id} not found.")
    return problem


@router.get("/batch/{batch_id}/problems", response_model=List[ProblemResponse])
def read_problems_from_batch(batch_id: int, db: Session = Depends(get_db)) -> List[ProblemResponse]:
    """
    Retrieve all problems associated with a specific batch ID.

    Args:
        batch_id: The unique identifier of the batch.
        db: The database session dependency.

    Returns:
        A list of problem objects belonging to the specified batch.
    """
    try:
        problems = get_problems_by_batch(db, batch_id)
        return problems
    except Exception as e:
        logger.error(
            "Failed to retrieve problems for batch ID %d: %s",
            batch_id,
            e,
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while fetching problems for batch {batch_id}.",
        ) from e


@router.get("/all", response_model=List[ProblemResponse])
def read_all_problems(
    batch_id: Optional[int] = Query(None, description="Filter problems by batch ID."),
    status: Optional[str] = Query(None, description="Filter problems by status."),
    db: Session = Depends(get_db),
) -> List[ProblemResponse]:
    """
    Retrieve all problems without pagination.

    Warning:
        This endpoint can be resource-intensive for large datasets and should be used
        with caution.

    Args:
        batch_id: An optional batch ID to filter the problems.
        status: An optional status to filter the problems.
        db: The database session dependency.

    Returns:
        A list of all problem objects matching the criteria.
    """
    try:
        # A high limit is used to effectively retrieve all problems.
        problems = get_problems(db, skip=0, limit=100000, batch_id=batch_id, status=status)
        return problems
    except Exception as e:
        logger.error("Failed to retrieve all problems: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while fetching all problems.",
        ) from e
