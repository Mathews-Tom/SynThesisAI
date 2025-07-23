#!/usr/bin/env python3
"""
Setup script for SynThesisAI - The Next-Gen Platform for Generative Intelligence Across STREAM
"""

import os
from pathlib import Path

from setuptools import find_packages, setup

# Read the long description from README.md
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


# Create necessary directories
def create_directories():
    """Create necessary directories for the project"""
    directories = ["database", ".cache/dspy", "data/training", "data/validation"]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")


# Create .env file if it doesn't exist
def create_env_file():
    """Create .env file if it doesn't exist"""
    env_path = Path(".env")
    if not env_path.exists():
        env_content = """# API Keys
OPENAI_KEY=your_openai_api_key_here
GEMINI_KEY=your_gemini_api_key_here
DEEPSEEK_KEY=your_deepseek_api_key_here

# Database
DATABASE_URL=sqlite:///./database/math_agent.db

# Optional Settings
SIMILARITY_THRESHOLD=0.82
EMBEDDING_MODEL=text-embedding-3-small
"""
        with open(env_path, "w") as f:
            f.write(env_content)
        print("✅ Created .env file - please update with your API keys")
    else:
        print("✅ .env file already exists")


# Create directories and .env file
print("🚀 Setting up SynThesisAI...")
create_directories()
create_env_file()

# Define package requirements
requirements = [
    # LLM Clients
    "openai>=1.0.0",
    "google-generativeai>=0.4.1",
    "requests>=2.31.0",
    # Web Framework
    "fastapi>=0.110.0",
    "uvicorn[standard]>=0.29.0",
    # Utility & Config
    "python-dotenv>=1.0.0",
    "PyYAML>=6.0.1",
    "pandas>=2.2.1",
    "tqdm>=4.66.2",
    # Pydantic (for schemas)
    "pydantic>=2.7.1",
    "json-repair>=0.47.4",
    "pytest>=8.4.1",
    "sympy>=1.14.0",
    # DSPy Framework
    "dspy-ai>=2.5.0",
    "optuna>=3.6.1",
]

# Setup configuration
setup(
    name="synthesisai",
    version="0.1.0",
    author="SynThesisAI Team",
    author_email="info@synthesisai.example.com",
    description="The Next-Gen Platform for Generative Intelligence Across STREAM",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/synthesisai",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "synthesisai=app.cli:main",
        ],
    },
)

print("\n🎉 Setup complete!")
print("\nNext steps:")
print("1. Update .env file with your API keys")
print("2. Run: uv run uvicorn app.main:app --reload")
print("3. Access API at: http://localhost:8000")
print("4. View docs at: http://localhost:8000/docs")
