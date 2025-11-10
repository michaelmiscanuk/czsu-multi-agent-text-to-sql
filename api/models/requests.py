"""Request models for the CZSU multi-agent text-to-SQL API."""

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
    """Request model for analyzing natural language queries.

    This model represents a user's natural language question that will be
    converted to SQL and executed against the CZSU database.
    """

    prompt: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Natural language query to analyze and convert to SQL",
        examples=["What was the population of Prague in 2020?"],
    )
    thread_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Unique identifier for the conversation thread",
        examples=["thread_abc123"],
    )
    run_id: Optional[str] = Field(
        None,
        min_length=1,
        description="Optional run ID in UUID format. Auto-generated if not provided.",
        examples=["550e8400-e29b-41d4-a716-446655440000"],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "prompt": "Show me unemployment rates for 2023",
                    "thread_id": "thread_12345",
                    "run_id": "550e8400-e29b-41d4-a716-446655440000",
                }
            ]
        }
    }

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

    @field_validator("run_id")
    @classmethod
    def validate_run_id(cls, v):
        """Validate that run_id is a valid UUID string if provided."""
        if v is not None:
            if not v or not v.strip():
                raise ValueError("Run ID cannot be empty")
            # Basic UUID format validation
            try:
                uuid.UUID(v.strip())
            except ValueError as exc:
                raise ValueError("Run ID must be a valid UUID format") from exc
            return v.strip()
        return v


class FeedbackRequest(BaseModel):
    """Request model for submitting user feedback on query results.

    Allows users to rate AI responses and optionally provide text comments.
    Feedback is tracked in LangSmith for quality monitoring.
    """

    run_id: str = Field(
        ...,
        min_length=1,
        description="UUID of the analysis run to provide feedback for",
        examples=["550e8400-e29b-41d4-a716-446655440000"],
    )
    feedback: Optional[int] = Field(
        None,
        ge=0,
        le=1,
        description="Binary feedback score: 1 = positive (👍), 0 = negative (👎)",
        examples=[1],
    )
    comment: Optional[str] = Field(
        None,
        max_length=1000,
        description="Optional text comment explaining the feedback",
        examples=["The query results were accurate and helpful."],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "run_id": "550e8400-e29b-41d4-a716-446655440000",
                    "feedback": 1,
                    "comment": "Great results!",
                }
            ]
        }
    }

    @field_validator("run_id")
    @classmethod
    def validate_run_id(cls, v):
        """Validate that run_id is a non-empty valid UUID string."""
        if not v or not v.strip():
            raise ValueError("Run ID cannot be empty")

        try:
            uuid.UUID(v.strip())
        except ValueError as exc:
            raise ValueError("Run ID must be a valid UUID format") from exc
        return v.strip()

    @field_validator("comment")
    @classmethod
    def validate_comment(cls, v):
        """Convert empty comment strings to None."""
        if v is not None and len(v.strip()) == 0:
            return None  # Convert empty string to None
        return v


class SentimentRequest(BaseModel):
    """Request model for tracking user sentiment on query responses.

    Simpler alternative to FeedbackRequest for quick positive/negative tracking.
    """

    run_id: str = Field(
        ...,
        min_length=1,
        description="UUID of the analysis run to track sentiment for",
        examples=["550e8400-e29b-41d4-a716-446655440000"],
    )
    sentiment: Optional[bool] = Field(
        None,
        description=(
            "Sentiment value: true = positive, false = negative, "
            "null = clear/remove sentiment"
        ),
        examples=[True],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"run_id": "550e8400-e29b-41d4-a716-446655440000", "sentiment": True}
            ]
        }
    }

    @field_validator("run_id")
    @classmethod
    def validate_run_id(cls, v):
        """Validate that run_id is a non-empty valid UUID string."""
        if not v or not v.strip():
            raise ValueError("Run ID cannot be empty")

        try:
            uuid.UUID(v.strip())
        except ValueError as exc:
            raise ValueError("Run ID must be a valid UUID format") from exc
        return v.strip()
