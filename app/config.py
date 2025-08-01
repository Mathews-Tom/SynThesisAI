"""Module for loading and validating application configuration from environment variables."""

# Standard Library
import logging
import os

# Third-Party Library
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class Settings:
    """Application settings loaded from environment variables."""

    # API Keys
    OPENAI_KEY: str = os.getenv("OPENAI_KEY", "")
    GEMINI_KEY: str = os.getenv("GEMINI_KEY", "")
    DEEPSEEK_KEY: str = os.getenv("DEEPSEEK_KEY", "")

    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./database/math_agent.db")

    # App settings
    APP_NAME: str = "Synthetic Math Prompts API"
    APP_VERSION: str = "1.0.0"

    # Similarity settings
    SIMILARITY_THRESHOLD: float = 0.82
    EMBEDDING_MODEL: str = "text-embedding-3-small"

    # Validation
    def validate(self) -> None:
        """
        Validate that required settings are present.

        Raises:
            ValueError: If any required key is missing.
        """
        if not self.OPENAI_KEY:
            raise ValueError("OPENAI_KEY is required")
        if not self.GEMINI_KEY:
            raise ValueError("GEMINI_KEY is required")
        if not self.DEEPSEEK_KEY:
            raise ValueError("DEEPSEEK_KEY is required")


# Create settings instance
settings = Settings()

# Validate settings on import
try:
    settings.validate()
except ValueError as e:
    logger.critical("Configuration error: %s", e)
    raise SystemExit("Configuration error: %s" % e) from e
