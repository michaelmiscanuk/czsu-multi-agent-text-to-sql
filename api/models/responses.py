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
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel

# ============================================================
# RESPONSE MODELS
# ============================================================


class ChatThreadResponse(BaseModel):
    """Response model for a single chat thread summary.

    Represents metadata about a conversation thread including the latest activity
    and a preview of the conversation topic.
    """

    thread_id: str = Field(
        description="Unique identifier for the conversation thread",
        examples=["thread_abc123"],
    )
    latest_timestamp: datetime = Field(
        description="Timestamp of the most recent message in ISO 8601 format"
    )
    run_count: int = Field(
        description="Total number of query runs in this thread", examples=[5]
    )
    title: str = Field(
        description="Thread title derived from the first user prompt",
        examples=["Population Statistics Query"],
    )
    full_prompt: str = Field(
        description="Complete text of the first prompt for tooltip display",
        examples=["What was the total population of Prague in 2020?"],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "thread_id": "thread_abc123",
                    "latest_timestamp": "2024-01-15T10:30:00Z",
                    "run_count": 3,
                    "title": "Population Statistics",
                    "full_prompt": "What was the population of Prague in 2020?",
                }
            ]
        }
    }


class PaginatedChatThreadsResponse(BaseModel):
    """Paginated response containing a list of chat threads.

    Includes pagination metadata for implementing infinite scroll or page navigation.
    """

    threads: List[ChatThreadResponse] = Field(
        description="List of chat thread summaries for the current page"
    )
    total_count: int = Field(
        description="Total number of threads across all pages", examples=[42]
    )
    page: int = Field(description="Current page number (1-indexed)", examples=[1])
    limit: int = Field(description="Maximum number of threads per page", examples=[10])
    has_more: bool = Field(
        description="Whether more pages of results are available", examples=[True]
    )


class ChatMessage(BaseModel):
    """Complete message object containing user query and AI response with metadata.

    Includes the natural language query, generated SQL, execution results,
    and supporting information like datasets used and PDF references.
    """

    id: str = Field(description="Unique message identifier", examples=["msg_123"])
    threadId: str = Field(
        description="Thread ID this message belongs to", examples=["thread_abc123"]
    )
    user: str = Field(
        description="Email address of the user who created the message",
        examples=["user@example.com"],
    )
    createdAt: int = Field(
        description="Unix timestamp (milliseconds) when message was created",
        examples=[1705320600000],
    )
    prompt: Optional[str] = Field(
        None,
        description="User's natural language query",
        examples=["What was the population in 2020?"],
    )
    final_answer: Optional[str] = Field(
        None, description="AI-generated answer to the user's query"
    )
    queries_and_results: Optional[List[List[str]]] = Field(
        None,
        description="List of [query_text, result_text] pairs from database execution",
    )
    datasets_used: Optional[List[str]] = Field(
        None,
        description="Names of CZSU datasets referenced in the query",
        examples=[["population_data", "census_2020"]],
    )
    top_chunks: Optional[List[dict]] = Field(
        None,
        description="Relevant PDF documentation chunks retrieved via vector search",
    )
    sql_query: Optional[str] = Field(
        None,
        description="Generated SQL query executed against the database",
        examples=["SELECT COUNT(*) FROM population WHERE year = 2020"],
    )
    error: Optional[str] = Field(
        None, description="Error message if query execution failed"
    )
    isLoading: Optional[bool] = Field(
        None, description="Whether the message is currently being processed"
    )
    startedAt: Optional[int] = Field(
        None, description="Unix timestamp (milliseconds) when processing started"
    )
    isError: Optional[bool] = Field(
        None, description="Whether the message resulted in an error"
    )
    run_id: Optional[str] = Field(
        None,
        description="UUID of the analysis run for tracking and feedback",
        examples=["550e8400-e29b-41d4-a716-446655440000"],
    )
    followup_prompts: Optional[List[str]] = Field(
        None,
        description="AI-suggested follow-up questions",
        examples=[["What about 2021?", "Show me by region"]],
    )
