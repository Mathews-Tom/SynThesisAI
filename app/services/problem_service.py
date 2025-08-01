"""
Service layer for handling problem-related operations.

This module provides functions for creating and retrieving problems from the
database.
"""

# Standard Library
from typing import Dict, List, Optional

# Third-Party Library
from sqlalchemy.orm import Session

# SynThesisAI Modules
from app.models.models import Problem
from app.models.schemas import ProblemCreate


def create_problem(db: Session, problem: ProblemCreate) -> Problem:
    """
    Create a new problem in the database.

    Args:
        db: The database session.
        problem: The problem data to create.

    Returns:
        The newly created problem.
    """
    db_problem = Problem(**problem.dict())
    db.add(db_problem)
    db.commit()
    db.refresh(db_problem)
    return db_problem


def get_problem(db: Session, problem_id: int) -> Optional[Problem]:
    """
    Retrieve a single problem by its ID.

    Args:
        db: The database session.
        problem_id: The ID of the problem to retrieve.

    Returns:
        The problem object if found, otherwise None.
    """
    return db.query(Problem).filter(Problem.id == problem_id).first()


def get_problems(
    db: Session,
    skip: int = 0,
    limit: int = 1000,
    batch_id: Optional[int] = None,
    status: Optional[str] = None,
) -> List[Problem]:
    """
    Retrieve a list of problems with optional filtering and pagination.

    Args:
        db: The database session.
        skip: The number of problems to skip.
        limit: The maximum number of problems to return.
        batch_id: The ID of the batch to filter by.
        status: The status to filter by ('valid' or 'discarded').

    Returns:
        A list of problem objects.
    """
    query = db.query(Problem)

    if batch_id is not None:
        query = query.filter(Problem.batch_id == batch_id)
    if status is not None:
        query = query.filter(Problem.status == status)

    return query.offset(skip).limit(limit).all()


def get_problems_by_batch(db: Session, batch_id: int) -> List[Problem]:
    """
    Retrieve all problems associated with a specific batch.

    Args:
        db: The database session.
        batch_id: The ID of the batch.

    Returns:
        A list of problem objects.
    """
    return db.query(Problem).filter(Problem.batch_id == batch_id).all()


def get_problem_stats(db: Session, batch_id: int) -> Dict[str, int]:
    """
    Get statistics on the number of valid and discarded problems for a batch.

    Args:
        db: The database session.
        batch_id: The ID of the batch to get stats for.

    Returns:
        A dictionary with the counts of 'valid' and 'discarded' problems.
    """
    discarded_count = (
        db.query(Problem)
        .filter(Problem.batch_id == batch_id, Problem.status == "discarded")
        .count()
    )
    valid_count = (
        db.query(Problem)
        .filter(Problem.batch_id == batch_id, Problem.status == "valid")
        .count()
    )
    return {"discarded": discarded_count, "valid": valid_count}
