"""
API router for the SynThesisAI application.

This module defines the main API router and includes sub-routers
for different API functionalities.
"""

# Third-Party Library
from fastapi import APIRouter, Depends

# SynThesisAI Modules
from app.api import batches, generation, problems
from app.models.schemas import GenerationRequest, GenerationResponse
from app.services.pipeline_service import run_pipeline
from utils.google_auth import verify_company_email

router: APIRouter = APIRouter()


@router.post("/generate", response_model=GenerationResponse)
def generate_prompts(request: GenerationRequest) -> GenerationResponse:
    """
    Generate synthetic math prompts based on the provided request.

    Args:
        request: The request containing the generation parameters.

    Returns:
        The response containing the generated prompts.
    """
    return run_pipeline(request)


# Include sub-routers for different API functionalities
router.include_router(
    batches.router,
    prefix="/batches",
    tags=["batches"],
    dependencies=[Depends(verify_company_email)],
)
router.include_router(
    problems.router,
    prefix="/problems",
    tags=["problems"],
    dependencies=[Depends(verify_company_email)],
)
router.include_router(
    generation.router,
    prefix="/generation",
    tags=["generation"],
    dependencies=[Depends(verify_company_email)],
)
