# CRITICAL: Set Windows event loop policy FIRST, before any other imports
# This must be the very first thing that happens to fix psycopg compatibility
import os
import sys

if sys.platform == "win32":
    import asyncio

    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Load environment variables early
from dotenv import load_dotenv

load_dotenv()

# Constants
try:
    from pathlib import Path

    BASE_DIR = Path(__file__).resolve().parents[2]
except NameError:
    BASE_DIR = Path(os.getcwd()).parents[0]

# Standard imports
import uuid
from typing import Optional

from pydantic import BaseModel, Field, field_validator

# ============================================================
# REQUEST MODELS
# ============================================================


class AnalyzeRequest(BaseModel):
    prompt: str = Field(
        ..., min_length=1, max_length=10000, description="The prompt to analyze"
    )
    thread_id: str = Field(
        ..., min_length=1, max_length=100, description="The thread ID"
    )

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, v):
        if not v or not v.strip():
            raise ValueError("Prompt cannot be empty or only whitespace")
        return v.strip()

    @field_validator("thread_id")
    @classmethod
    def validate_thread_id(cls, v):
        if not v or not v.strip():
            raise ValueError("Thread ID cannot be empty or only whitespace")
        return v.strip()


class FeedbackRequest(BaseModel):
    run_id: str = Field(..., min_length=1, description="The run ID (UUID format)")
    feedback: Optional[int] = Field(
        None,
        ge=0,
        le=1,
        description="Feedback score: 1 for thumbs up, 0 for thumbs down",
    )
    comment: Optional[str] = Field(
        None, max_length=1000, description="Optional comment"
    )

    @field_validator("run_id")
    @classmethod
    def validate_run_id(cls, v):
        if not v or not v.strip():
            raise ValueError("Run ID cannot be empty")
        # Basic UUID format validation
        import uuid

        try:
            uuid.UUID(v.strip())
        except ValueError:
            raise ValueError("Run ID must be a valid UUID format")
        return v.strip()

    @field_validator("comment")
    @classmethod
    def validate_comment(cls, v):
        if v is not None and len(v.strip()) == 0:
            return None  # Convert empty string to None
        return v


class SentimentRequest(BaseModel):
    run_id: str = Field(..., min_length=1, description="The run ID (UUID format)")
    sentiment: Optional[bool] = Field(
        None,
        description="Sentiment: true for positive, false for negative, null to clear",
    )

    @field_validator("run_id")
    @classmethod
    def validate_run_id(cls, v):
        if not v or not v.strip():
            raise ValueError("Run ID cannot be empty")
        # Basic UUID format validation
        import uuid

        try:
            uuid.UUID(v.strip())
        except ValueError:
            raise ValueError("Run ID must be a valid UUID format")
        return v.strip()
