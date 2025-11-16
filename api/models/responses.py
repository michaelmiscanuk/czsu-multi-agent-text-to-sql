"""
MODULE_DESCRIPTION: Response Models - Pydantic Schemas for API Response Serialization

===================================================================================
PURPOSE AND OVERVIEW
===================================================================================

This module defines Pydantic BaseModel classes that serve as response schemas for
the CZSU Multi-Agent Text-to-SQL API. These models ensure consistent response
formats, automatic serialization, type safety, and OpenAPI documentation generation
for all API responses.

Response Models:
    1. ChatThreadResponse: Summary of a single conversation thread
    2. PaginatedChatThreadsResponse: Paginated list of conversation threads
    3. ChatMessage: Complete message with query, results, and metadata

Each model provides:
    - Type-safe response structure
    - Automatic JSON serialization
    - OpenAPI schema generation
    - Field validation and documentation
    - Example responses for API docs

===================================================================================
KEY FEATURES
===================================================================================

1. Type Safety
   - Ensures response fields have correct types
   - Prevents runtime type errors
   - IDE auto-completion support
   - Better code maintainability

2. Automatic Serialization
   - Converts Python objects to JSON
   - Handles datetime formatting
   - Nested model serialization
   - List/dict serialization

3. OpenAPI Documentation
   - Automatic schema generation for /docs
   - Field descriptions in Swagger UI
   - Example responses
   - Response type documentation

4. Consistent Structure
   - Standardized response formats
   - Predictable field names
   - Consistent data types
   - Clear naming conventions

5. Optional Fields
   - Flexible response structure
   - Supports partial data
   - Null safety
   - Error state handling

===================================================================================
RESPONSE MODELS
===================================================================================

Model 1: ChatThreadResponse
    Purpose: Summary of a single conversation thread

    Fields:
        thread_id (str):
            - Unique identifier for the conversation thread
            - Matches AnalyzeRequest.thread_id
            - Example: "thread_abc123"

        latest_timestamp (datetime):
            - Timestamp of most recent message in thread
            - ISO 8601 format in JSON
            - Used for sorting threads by recency
            - Example: "2024-01-15T10:30:00Z"

        run_count (int):
            - Total number of query runs in this thread
            - Indicates conversation length
            - Example: 5

        title (str):
            - Thread title derived from first user prompt
            - Truncated for display (typically ~50 chars)
            - Example: "Population Statistics Query"

        full_prompt (str):
            - Complete text of the first prompt
            - Used for tooltip/preview
            - Example: "What was the total population of Prague in 2020?"

    Usage:
        Returned by GET /chat/threads
        Part of PaginatedChatThreadsResponse

Model 2: PaginatedChatThreadsResponse
    Purpose: Paginated list of conversation threads

    Fields:
        threads (List[ChatThreadResponse]):
            - Array of thread summaries for current page
            - Ordered by latest_timestamp descending
            - Empty array if no threads

        total_count (int):
            - Total number of threads across all pages
            - Used for pagination UI
            - Example: 42

        page (int):
            - Current page number (1-indexed)
            - Example: 1

        limit (int):
            - Maximum number of threads per page
            - Typically 10 or 20
            - Example: 10

        has_more (bool):
            - Whether more pages are available
            - true if page < ceil(total_count / limit)
            - Example: true

    Usage:
        GET /chat/threads?page=1&limit=10
        Supports infinite scroll or page navigation

Model 3: ChatMessage
    Purpose: Complete message with query, results, and metadata

    Core Fields:
        id (str):
            - Unique message identifier
            - Example: "msg_123"

        threadId (str):
            - Thread ID this message belongs to
            - Example: "thread_abc123"

        user (str):
            - Email address of the user
            - From JWT authentication
            - Example: "user@example.com"

        createdAt (int):
            - Unix timestamp in milliseconds
            - When message was created
            - Example: 1705320600000

    Query Fields:
        prompt (Optional[str]):
            - User's natural language query
            - None for system messages
            - Example: "What was the population in 2020?"

        sql_query (Optional[str]):
            - Generated SQL query
            - None if query generation failed
            - Example: "SELECT COUNT(*) FROM population WHERE year = 2020"

    Result Fields:
        final_answer (Optional[str]):
            - AI-generated answer to user's query
            - Markdown formatted
            - None if analysis incomplete/failed

        queries_and_results (Optional[List[List[str]]]):
            - List of [query_text, result_text] pairs
            - Multiple queries if analysis required iteration
            - None if no queries executed

        datasets_used (Optional[List[str]]):
            - Names of CZSU datasets referenced
            - Example: ["population_data", "census_2020"]

        top_chunks (Optional[List[dict]]):
            - Relevant PDF documentation chunks
            - Retrieved via vector search
            - Contains metadata and content

    State Fields:
        isLoading (Optional[bool]):
            - Whether message is currently being processed
            - true during analysis
            - false when complete

        startedAt (Optional[int]):
            - Unix timestamp (ms) when processing started
            - Used to calculate processing duration

        isError (Optional[bool]):
            - Whether message resulted in an error
            - true if analysis failed
            - false if successful

        error (Optional[str]):
            - Error message if query execution failed
            - None if successful

    Tracking Fields:
        run_id (Optional[str]):
            - UUID of the analysis run
            - Used for feedback submission
            - Example: "550e8400-e29b-41d4-a716-446655440000"

        followup_prompts (Optional[List[str]]):
            - AI-suggested follow-up questions
            - Example: ["What about 2021?", "Show me by region"]

    Usage:
        GET /chat/messages/{thread_id}
        GET /chat/all-messages-for-all-threads
        Streaming updates during analysis

===================================================================================
FIELD NAMING CONVENTIONS
===================================================================================

Python vs JSON:
    Python (snake_case):
        - latest_timestamp
        - run_count
        - full_prompt

    JSON (camelCase):
        - threadId
        - isLoading
        - startedAt

    Reason:
        - Python models use snake_case
        - JSON API uses camelCase
        - Pydantic handles conversion with aliases

Pydantic Aliases:
    class ChatMessage(BaseModel):
        thread_id: str = Field(..., alias="threadId")

    Python code: message.thread_id
    JSON output: {"threadId": "..."}

===================================================================================
DATETIME HANDLING
===================================================================================

Datetime Fields:
    latest_timestamp: datetime

    Python:
        datetime.datetime(2024, 1, 15, 10, 30, 0)

    JSON:
        "2024-01-15T10:30:00Z" (ISO 8601 format)

Pydantic Serialization:
    Automatically converts datetime to ISO 8601
    Timezone-aware recommended
    UTC preferred for consistency

Unix Timestamps:
    createdAt: int (milliseconds)
    startedAt: int (milliseconds)

    Python:
        int(time.time() * 1000)

    JSON:
        1705320600000

Why Both Formats:
    - datetime: Human-readable, sortable
    - Unix ms: Frontend compatibility, calculations

===================================================================================
OPTIONAL FIELDS
===================================================================================

Purpose:
    - Support partial data (loading states)
    - Handle errors gracefully
    - Flexible response structure
    - Backward compatibility

Loading State Example:
    {
        "id": "msg_123",
        "threadId": "thread_abc",
        "user": "user@example.com",
        "createdAt": 1705320600000,
        "prompt": "What was the population?",
        "isLoading": true,
        "startedAt": 1705320601000,
        "final_answer": null,
        "sql_query": null
    }

Complete State Example:
    {
        "id": "msg_123",
        "threadId": "thread_abc",
        "user": "user@example.com",
        "createdAt": 1705320600000,
        "prompt": "What was the population?",
        "isLoading": false,
        "final_answer": "The population was 1.3 million.",
        "sql_query": "SELECT * FROM population",
        "run_id": "550e8400-e29b-41d4-a716-446655440000"
    }

Error State Example:
    {
        "id": "msg_123",
        "threadId": "thread_abc",
        "user": "user@example.com",
        "createdAt": 1705320600000,
        "prompt": "What was the population?",
        "isError": true,
        "error": "Database connection failed",
        "final_answer": null
    }

===================================================================================
PAGINATION PATTERN
===================================================================================

Request:
    GET /chat/threads?page=2&limit=10

Response:
    {
        "threads": [...],  // 10 thread summaries
        "total_count": 42,
        "page": 2,
        "limit": 10,
        "has_more": true  // page 2 of 5
    }

Client Logic:
    if (response.has_more) {
        // Load next page
        fetchThreads(page + 1)
    }

Infinite Scroll:
    - Load page 1 initially
    - Append page 2 when scrolling
    - Continue until has_more = false

Page Navigation:
    total_pages = ceil(total_count / limit)
    can_go_next = page < total_pages
    can_go_prev = page > 1

===================================================================================
OPENAPI INTEGRATION
===================================================================================

Automatic Schema Generation:
    Response models → OpenAPI schemas
    Visible in /docs (Swagger UI)
    Shows example responses

Field Documentation:
    description parameter → Field description
    examples parameter → Example values

Response Example in Docs:
    {
        "thread_id": "thread_abc123",
        "latest_timestamp": "2024-01-15T10:30:00Z",
        "run_count": 3,
        "title": "Population Statistics",
        "full_prompt": "What was the population of Prague in 2020?"
    }

===================================================================================
SERIALIZATION
===================================================================================

Python to JSON:
    message = ChatMessage(
        id="msg_123",
        threadId="thread_abc",
        user="user@example.com",
        createdAt=1705320600000,
        prompt="Test"
    )

    json_str = message.model_dump_json()
    # {"id":"msg_123","threadId":"thread_abc",...}

Nested Models:
    threads_response = PaginatedChatThreadsResponse(
        threads=[thread1, thread2],
        total_count=2,
        page=1,
        limit=10,
        has_more=False
    )

    # Automatically serializes nested ChatThreadResponse objects

List Fields:
    datasets_used: Optional[List[str]]
    # ["dataset1", "dataset2"]

    top_chunks: Optional[List[dict]]
    # [{"content": "...", "metadata": {...}}, ...]

===================================================================================
TESTING CONSIDERATIONS
===================================================================================

Unit Tests:
    - Test model creation
    - Test serialization
    - Test optional fields
    - Test field types

Test Examples:
    def test_chat_thread_response():
        thread = ChatThreadResponse(
            thread_id="test_thread",
            latest_timestamp=datetime.now(),
            run_count=5,
            title="Test Thread",
            full_prompt="Test prompt"
        )
        assert thread.thread_id == "test_thread"
        assert thread.run_count == 5

    def test_chat_message_optional_fields():
        message = ChatMessage(
            id="msg_1",
            threadId="thread_1",
            user="test@example.com",
            createdAt=1705320600000
        )
        assert message.prompt is None
        assert message.final_answer is None

Integration Tests:
    - Test endpoint responses
    - Verify JSON structure
    - Test pagination
    - Test error responses

===================================================================================
DEPENDENCIES
===================================================================================

Standard Library:
    - datetime: Timestamp handling
    - typing: Type hints (List, Optional)

Third-Party:
    - pydantic: BaseModel, Field

===================================================================================
FUTURE ENHANCEMENTS
===================================================================================

1. Response Versioning
   - API version in response
   - Backward compatibility
   - Gradual migration

2. Enhanced Metadata
   - Performance metrics
   - Cache hit/miss
   - Processing time

3. Richer Error Information
   - Error codes
   - Retry suggestions
   - Help links

4. Additional Fields
   - User preferences
   - Personalization data
   - A/B test variants

===================================================================================
"""

# Response models for the CZSU multi-agent text-to-SQL API
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

from pydantic import BaseModel, Field

# ==============================================================================
# RESPONSE MODELS - PYDANTIC SCHEMAS FOR API SERIALIZATION
# ==============================================================================


class ChatThreadResponse(BaseModel):
    """Response model for a single chat thread summary.

    Represents metadata about a conversation thread including the latest activity
    and a preview of the conversation topic. Used in paginated thread listings
    to give users an overview of their conversations without loading full history.

    Contains:
        - Thread identification
        - Latest activity timestamp
        - Run count (number of messages)
        - Thread title (user-provided or auto-generated)
        - Full prompt text for tooltips

    Example Response:
        {
            "thread_id": "thread_12345",
            "latest_timestamp": "2024-01-15T10:30:00Z",
            "run_count": 5,
            "title": "Population Analysis",
            "full_prompt": "Show unemployment rates for 2023"
        }
    """

    # =======================================================================
    # FIELD DEFINITIONS
    # =======================================================================

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

    # =======================================================================
    # OPENAPI CONFIGURATION
    # =======================================================================

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
    Wraps thread summaries with total count and pagination state to support efficient
    loading of large thread lists.

    Pagination Pattern:
        - Request: GET /threads?page=1&limit=20
        - Response includes: threads array, total count, page info, has_more flag
        - Client can request next page if has_more is true

    Example Response:
        {
            "threads": [...],
            "total_count": 47,
            "page": 1,
            "limit": 20,
            "has_more": true
        }
    """

    # =======================================================================
    # FIELD DEFINITIONS
    # =======================================================================

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
    and supporting information like datasets used and PDF references. Used in both
    real-time streaming responses and historical message retrieval.

    Field Groups:
        - Core Fields: id, threadId, user, createdAt, prompt
        - Query Fields: queries_and_results, datasets_used, top_chunks, sql_query
        - Result Fields: final_answer
        - State Fields: isLoading, isError, error
        - Tracking Fields: startedAt, run_id, followup_prompts

    Loading States:
        - Loading: isLoading=true, final_answer=null
        - Complete: isLoading=false, final_answer="...", error=null
        - Error: isLoading=false, isError=true, error="..."

    Example Response:
        {
            "id": "msg_123",
            "threadId": "thread_12345",
            "user": "user@example.com",
            "createdAt": 1705315800000,
            "prompt": "Show unemployment rates",
            "final_answer": "Here are the results...",
            "queries_and_results": [...],
            "datasets_used": ["dataset_1"],
            "top_chunks": [...],
            "sql_query": "SELECT ...",
            "isLoading": false,
            "isError": false,
            "error": null,
            "startedAt": 1705315800000,
            "run_id": "550e8400-e29b-41d4-a716-446655440000",
            "followup_prompts": ["Related query?"]
        }
    """

    # =======================================================================
    # CORE IDENTIFICATION FIELDS
    # =======================================================================

    id: str = Field(description="Unique message identifier", examples=["msg_123"])
    threadId: str = Field(
        description="Thread ID this message belongs to", examples=["thread_abc123"]
    )
    user: str = Field(
        description="Email address of the user who created the message",
        examples=["user@example.com"],
    )

    # =======================================================================
    # TIMESTAMP FIELDS (Unix milliseconds for frontend compatibility)
    # =======================================================================

    createdAt: int = Field(
        description="Unix timestamp (milliseconds) when message was created",
        examples=[1705320600000],
    )
    startedAt: Optional[int] = Field(
        None, description="Unix timestamp (milliseconds) when processing started"
    )

    # =======================================================================
    # USER INPUT AND AI RESPONSE
    # =======================================================================

    prompt: Optional[str] = Field(
        None,
        description="User's natural language query",
        examples=["What was the population in 2020?"],
    )
    final_answer: Optional[str] = Field(
        None, description="AI-generated answer to the user's query"
    )

    # =======================================================================
    # SQL QUERY ANALYSIS DETAILS
    # =======================================================================

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

    # =======================================================================
    # UI STATE FLAGS
    # =======================================================================

    error: Optional[str] = Field(
        None, description="Error message if query execution failed"
    )
    isLoading: Optional[bool] = Field(
        None, description="Whether the message is currently being processed"
    )
    isError: Optional[bool] = Field(
        None, description="Whether the message resulted in an error"
    )

    # =======================================================================
    # TRACKING AND FOLLOW-UP
    # =======================================================================

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
