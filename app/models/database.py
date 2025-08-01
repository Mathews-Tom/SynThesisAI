"""Database module for setting up SQLAlchemy engine, sessionmaker, and base.

This module loads environment variables, configures the database connection, creates an SQLAlchemy 
engine, sessionmaker, and declarative base, and provides the `get_db` generator for obtaining 
database sessions.
"""

# Standard Library
import os

from dotenv import load_dotenv

# Third-Party Library
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Load environment variables
load_dotenv()

# Database URL - default to SQLite
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./database/math_agent.db")

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
