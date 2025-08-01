"""
Main application file for the Synthetic Math Prompts API.

This module initializes the FastAPI application, sets up middleware,
and includes the main API routes.
"""

# Standard Library
from typing import Dict

# Third-Party Library
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# SynThesisAI Modules
from app.api.routes import router
from app.config import settings
from app.models.database import Base, engine

# Create database tables
# This ensures that all tables are created based on the models defined.
Base.metadata.create_all(bind=engine)

app: FastAPI = FastAPI(title=settings.APP_NAME, version=settings.APP_VERSION)

# Add CORS middleware to allow cross-origin requests.
# This is configured to be open for development purposes.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the main API router
# All routes defined in the router will be prefixed with /api.
app.include_router(router, prefix="/api")


@app.get("/")
async def root() -> Dict[str, str]:
    """
    Root endpoint for the API.

    Returns:
        A dictionary with a welcome message.
    """
    return {"message": "Synthetic Math Prompts API"}


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """
    Health check endpoint.

    Provides a simple health status check for monitoring purposes.

    Returns:
        A dictionary indicating the service status.
    """
    return {"status": "healthy"}
