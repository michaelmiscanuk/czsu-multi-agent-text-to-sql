"""Message API Routes for Multi-Agent Text-to-SQL System

This module provides FastAPI REST endpoints for retrieving conversation messages and run metadata
from the LangGraph-based multi-agent text-to-SQL system. It handles checkpoint-based message
extraction, run ID tracking for sentiment correlation, and complete metadata retrieval for
chat conversations.
"""

MODULE_DESCRIPTION = r"""Message API Routes for Multi-Agent Text-to-SQL System

This module serves as a specialized REST API interface focused on message retrieval and run tracking
for the multi-agent text-to-SQL conversation system. It provides endpoints for accessing conversation
history stored in LangGraph checkpoints and correlating messages with run IDs for sentiment tracking.

Key Features:
-------------
1. Checkpoint-Based Message Retrieval:
   - Complete conversation history extraction from PostgreSQL checkpoints
   - Chronological message ordering by checkpoint step numbers
   - Preservation of original user prompts (not sanitized or modified)
   - AI response extraction with complete metadata
   - Efficient single-pass checkpoint processing
   - Security-verified access to thread data

2. Message Metadata Extraction:
   - SQL queries and execution results per interaction
   - Dataset selection codes (CZSU data sources) used
   - PDF document chunks retrieved for context
   - Follow-up prompt suggestions for continued conversation
   - Complete checkpoint metadata preservation
   - Per-interaction metadata isolation (not accumulated)

3. Run ID Management:
   - Run ID retrieval for sentiment tracking correlation
   - Chronological run ordering by timestamp
   - UUID validation and formatting
   - Prompt text association with run IDs
   - Timestamp preservation in ISO format
   - Invalid UUID filtering and error handling

4. Security and Authentication:
   - JWT-based user authentication via dependency injection
   - Thread ownership validation before data access
   - Email-based user identification and authorization
   - Database-level security checks for thread access
   - Protected endpoints requiring valid authentication tokens

5. Error Handling and Resilience:
   - Comprehensive exception catching and logging
   - Graceful degradation on database connection failures
   - Custom traceback JSON responses for client debugging
   - Connection error detection and appropriate responses
   - Empty result handling for unavailable resources
   - Detailed error logging for troubleshooting

6. Checkpoint Integration:
   - LangGraph AsyncPostgresSaver checkpoint access
   - Fallback mechanisms for checkpoint retrieval (alist → aget_tuple)
   - Checkpoint tuple processing for message reconstruction
   - Metadata extraction from writes.__start__ and submit_final_answer
   - State snapshot handling for complete conversation context
   - Step-based chronological ordering

7. Database Operations:
   - PostgreSQL connection pooling for efficiency
   - Parameterized queries for SQL injection prevention
   - Transactional consistency for data retrieval
   - Connection health checking and retry logic
   - Optimized queries with proper indexing
   - Cursor-based result iteration for memory efficiency

API Endpoints:
-------------
1. GET /chat/{thread_id}/messages
   - Retrieve complete conversation messages from checkpoints
   - Path parameter: thread_id (conversation identifier)
   - Returns: Array of ChatMessage objects with full metadata
   - Authentication: Required (JWT token with ownership verification)
   - Source: PostgreSQL checkpoints table (LangGraph state)

2. GET /chat/{thread_id}/run-ids
   - Get run_ids for messages to enable sentiment submission
   - Path parameter: thread_id (conversation identifier)
   - Returns: Array of objects with run_id, prompt, timestamp
   - Authentication: Required (JWT token)
   - Source: users_threads_runs table

Processing Flow:
--------------
1. Message Retrieval Workflow:
   - User requests messages for specific thread_id
   - System validates authentication token and extracts user email
   - Thread ownership verified via database lookup
   - LangGraph checkpointer accessed to retrieve all checkpoints
   - Checkpoints sorted chronologically by step number
   - Message extraction from checkpoint metadata:
     * User prompts from writes.__start__.prompt
     * AI answers from writes.submit_final_answer.final_answer
     * Queries/results from writes.submit_final_answer.queries_and_results
     * Datasets from writes.submit_final_answer.top_selection_codes
     * PDF chunks from writes.submit_final_answer.top_chunks
     * Follow-up prompts from writes.submit_final_answer.followup_prompts
   - ChatMessage objects created with complete metadata
   - Response returned as array of messages

2. Run ID Retrieval Workflow:
   - User requests run_ids for specific thread_id
   - System validates authentication token and extracts user email
   - Database query retrieves all runs for thread from users_threads_runs
   - Results ordered chronologically by timestamp
   - UUID validation performed on each run_id
   - Invalid UUIDs filtered out with logging
   - Valid run_ids packaged with prompt text and timestamp
   - Response returned as dictionary with run_ids array

3. Security Verification Process:
   - Extract user email from JWT token (via get_current_user dependency)
   - Query users_threads_runs table for matching email + thread_id
   - Count matching entries (should be > 0 for owned threads)
   - Deny access if no matching entries found
   - Proceed with data access only after successful verification

4. Checkpoint Processing Details:
   - Primary method: checkpointer.alist(config, limit=200)
     * Retrieves all checkpoints for complete conversation
     * Provides full checkpoint tuples with metadata
   - Fallback method: checkpointer.aget_tuple(config)
     * Used if alist() fails due to compatibility issues
     * Retrieves only latest checkpoint (limited context)
   - Checkpoint tuple structure:
     * metadata.step: Chronological step number
     * metadata.writes: Dictionary of agent outputs
     * metadata.writes.__start__: User input data
     * metadata.writes.submit_final_answer: AI response data

5. Error Handling Strategy:
   - Connection errors: Return empty results with logging
   - Authentication errors: Raise HTTPException 401
   - Database errors: Log and return empty or raise 500
   - UUID validation errors: Filter invalid, continue processing
   - Checkpoint retrieval errors: Try fallback, then return empty
   - Generic exceptions: Custom traceback response or HTTPException 500

Data Models:
-----------
1. ChatMessage (Response Model):
   - id: Message identifier (generated)
   - threadId: Parent thread identifier
   - user: User email address
   - createdAt: Message creation timestamp (milliseconds)
   - prompt: User's natural language query (optional)
   - final_answer: AI's generated response (optional)
   - queries_and_results: SQL queries and execution results (optional)
   - datasets_used: CZSU selection codes used (optional)
   - top_chunks: PDF document chunks for context (optional)
   - sql_query: Primary SQL query executed (optional)
   - followup_prompts: Suggested follow-up questions (optional)
   - run_id: Unique run identifier for sentiment tracking (optional)
   - error: Error message if execution failed (optional)
   - isLoading: Loading state indicator (boolean)
   - isError: Error state indicator (boolean)

2. Run ID Response Format:
   - run_ids: Array of run metadata objects
     * run_id: UUID string for the run
     * prompt: User's query text for this run
     * timestamp: ISO formatted timestamp of run creation

Configuration:
-------------
- Windows compatibility: WindowsSelectorEventLoopPolicy for psycopg
- Debug toggles: print__* functions for detailed logging
- Connection pooling: Managed by checkpointer factory
- Timeout handling: Database connection error detection
- Retry logic: Graceful degradation on transient failures

Database Schema Dependencies:
----------------------------
1. checkpoints table (LangGraph):
   - Stores conversation state snapshots
   - Contains metadata with agent outputs
   - Used for message reconstruction
   - Indexed by thread_id and checkpoint_id

2. users_threads_runs table:
   - Maps users to threads and runs
   - Stores run_id, prompts, timestamps, sentiments
   - Used for authentication and tracking
   - Indexed by email and thread_id

Required Environment:
-------------------
- Python 3.8+ with async/await support
- PostgreSQL database with LangGraph schema
- FastAPI web framework
- JWT authentication configured
- LangGraph AsyncPostgresSaver initialized
- Environment variables loaded via python-dotenv

Usage Examples:
--------------
# Get conversation messages from checkpoints
GET /chat/thread_abc123/messages
Authorization: Bearer <jwt_token>
Response: [
    {
        "id": "msg_1",
        "threadId": "thread_abc123",
        "prompt": "Show population data",
        "final_answer": "Here is the data...",
        "sql_query": "SELECT * FROM...",
        "datasets_used": ["OBY01PDT01"]
    }
]

# Get run IDs for sentiment tracking
GET /chat/thread_abc123/run-ids
Authorization: Bearer <jwt_token>
Response: {
    "run_ids": [
        {
            "run_id": "550e8400-e29b-41d4-a716-446655440000",
            "prompt": "Show population data",
            "timestamp": "2024-01-15T10:30:00"
        }
    ]
}

Error Handling:
-------------
- HTTPException 401: Missing or invalid authentication token
- HTTPException 500: Database connection failures, checkpoint errors
- Empty results: Returned on connection issues without raising error
- Detailed logging: Debug output for troubleshooting (development mode)
- Custom traceback responses: JSON-formatted error details for debugging
- Connection retry handling: Automatic reconnection for transient failures

Performance Considerations:
--------------------------
- Single-pass checkpoint processing reduces query overhead
- Connection pooling improves database efficiency
- Checkpoint limit (200) prevents memory issues
- UUID validation filters prevent invalid data propagation
- Cursor-based iteration for memory-efficient result processing
- Parameterized queries prevent SQL injection and enable query caching

Integration Points:
------------------
- LangGraph multi-agent system for conversation state
- PostgreSQL for persistent storage
- JWT authentication service for security
- Sentiment tracking service for user feedback
- chat.py module for shared message extraction logic
- checkpointer module for database connections

Key Differences from chat.py:
-----------------------------
- Focused on message/run retrieval vs. complete thread management
- Preserves original user prompts from checkpoints (not sanitized)
- Provides run_ids endpoint for sentiment correlation
- Simpler response models (arrays vs. comprehensive dictionaries)
- No thread creation, deletion, or listing functionality
- No sentiment retrieval (handled in chat.py)
- Shares get_thread_messages_with_metadata utility with chat.py

Debugging and Monitoring:
------------------------
- print__api_postgresql: PostgreSQL operation diagnostics
- print__chat_messages_debug: Message processing diagnostics
- print__feedback_flow: Feedback/run-id flow tracking
- Traceback preservation: Full error stack traces for debugging
- Connection error detection: Specific keyword matching for DB issues
- UUID validation logging: Invalid UUID reporting"""

# ==============================================================================
# CRITICAL: WINDOWS EVENT LOOP POLICY CONFIGURATION
# ==============================================================================

# CRITICAL: Set Windows event loop policy FIRST, before any other imports
# This must be the very first thing that happens to fix psycopg compatibility
# with Windows systems. The WindowsSelectorEventLoopPolicy resolves issues
# with the default ProactorEventLoop that can cause database connection problems.
# This is especially important for async database operations with psycopg3.
import os
import sys

if sys.platform == "win32":
    import asyncio

    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# ==============================================================================
# ENVIRONMENT CONFIGURATION
# ==============================================================================

# Load environment variables early to ensure configuration availability
# This must happen before any modules that depend on environment variables
from dotenv import load_dotenv

load_dotenv()

# ==============================================================================
# PATH CONFIGURATION
# ==============================================================================

# Determine base directory for the project, handling both regular execution
# and special cases where __file__ might not be defined (e.g., REPL)
try:
    from pathlib import Path

    BASE_DIR = Path(__file__).resolve().parents[2]  # Navigate up to project root
except NameError:
    # Fallback for environments where __file__ is not available
    BASE_DIR = Path(os.getcwd()).parents[0]

import traceback

# ==============================================================================
# STANDARD LIBRARY IMPORTS
# ==============================================================================

# Standard library imports for data structures and type hints
import uuid  # For UUID validation of run_ids
from typing import Dict, List  # Type hints for improved code clarity

# ==============================================================================
# THIRD-PARTY IMPORTS - WEB FRAMEWORK
# ==============================================================================

# FastAPI components for building REST API endpoints
from fastapi import APIRouter, Depends, HTTPException

# ==============================================================================
# API DEPENDENCIES - AUTHENTICATION
# ==============================================================================

# JWT-based authentication dependency for protecting endpoints
# This ensures only authenticated users can access their conversation data
from api.dependencies.auth import get_current_user

# ==============================================================================
# API MODELS - RESPONSE SCHEMAS
# ==============================================================================

# Pydantic model for message response serialization
# ChatMessage includes prompt, response, metadata, and tracking information
from api.models.responses import ChatMessage

# ==============================================================================
# UTILITY FUNCTIONS - DEBUGGING
# ==============================================================================

# Conditional debug output functions for development and troubleshooting
# These can be toggled via configuration to control verbosity
from api.utils.debug import (
    print__api_postgresql,  # PostgreSQL operation diagnostics
    print__chat_messages_debug,  # Message processing diagnostics
    print__feedback_flow,  # Feedback/sentiment flow tracking
)

# ==============================================================================
# PROJECT PATH SETUP
# ==============================================================================

# Add base directory to Python path for module resolution
# This allows imports from the project root directory
sys.path.insert(0, str(BASE_DIR))

# ==============================================================================
# API HELPERS AND SHARED UTILITIES
# ==============================================================================

# Custom error response formatting for client debugging
from api.helpers import traceback_json_response

# Shared message extraction logic from chat.py module
# This function consolidates checkpoint processing and metadata extraction
from api.routes.chat import get_thread_messages_with_metadata

# ==============================================================================
# DATABASE OPERATIONS - CHECKPOINTER MODULE
# ==============================================================================

# Global checkpointer factory for LangGraph checkpoint access
# Provides connection pooling and health monitoring
from checkpointer.checkpointer.factory import get_global_checkpointer

# ==============================================================================
# FASTAPI ROUTER INITIALIZATION
# ==============================================================================

# Create router instance for message-related endpoints
# This router will be included in the main FastAPI application
router = APIRouter()


# ==============================================================================
# API ENDPOINT: GET CHAT MESSAGES FROM CHECKPOINTS
# ==============================================================================


@router.get("/chat/{thread_id}/messages")
async def get_chat_messages(
    thread_id: str, user=Depends(get_current_user)
) -> List[ChatMessage]:
    """Load conversation messages from PostgreSQL checkpoint history that preserves original user messages.

    This endpoint retrieves the complete conversation history for a specific thread by
    extracting data from LangGraph checkpoints stored in PostgreSQL. It preserves the
    original, unmodified user prompts and includes all metadata such as SQL queries,
    datasets used, and PDF chunks.

    Key Features:
        - Extracts messages from checkpoint writes metadata
        - Preserves original user prompts (not sanitized)
        - Includes complete AI response metadata
        - Verifies thread ownership before access
        - Handles checkpoint retrieval failures gracefully

    Security:
        - Requires JWT authentication token
        - Verifies thread ownership via database lookup
        - Returns empty list if user doesn't own thread

    Args:
        thread_id: Unique identifier for the conversation thread
        user: Authenticated user object (injected by dependency)

    Returns:
        List[ChatMessage]: Chronologically ordered messages with full metadata.
                          Empty list if thread doesn't exist, user lacks access,
                          or checkpoint retrieval fails.

    Raises:
        HTTPException 401: User email not found in authentication token
        HTTPException 500: Database errors (non-connection related)

    Note:
        - Connection errors return empty list (graceful degradation)
        - Uses shared get_thread_messages_with_metadata from chat.py
        - Source: PostgreSQL checkpoints table (LangGraph state)
    """

    # =======================================================================
    # AUTHENTICATION AND VALIDATION
    # =======================================================================

    # Extract user email from JWT token for ownership verification
    user_email = user.get("email")
    if not user_email:
        raise HTTPException(status_code=401, detail="User email not found in token")

    print__api_postgresql(
        f"📥 Loading checkpoint messages for thread {thread_id}, user: {user_email}"
    )

    try:
        # =======================================================================
        # CHECKPOINTER ACCESS
        # =======================================================================

        # Get global checkpointer instance for conversation state access
        # This provides connection pooling and health monitoring
        checkpointer = await get_global_checkpointer()

        # =======================================================================
        # CHECKPOINTER VALIDATION
        # =======================================================================

        # Verify that checkpointer has active database connection
        # If no connection available, return empty list (graceful degradation)
        if not hasattr(checkpointer, "conn"):
            print__api_postgresql(
                f"⚠️ No PostgreSQL checkpointer available - returning empty messages"
            )
            return []

        # =======================================================================
        # MESSAGE EXTRACTION FROM CHECKPOINTS
        # =======================================================================

        # Use the consolidated function that handles all checkpoint processing and metadata extraction
        # This function:
        # - Verifies thread ownership
        # - Retrieves all checkpoints for the thread
        # - Extracts user prompts and AI responses
        # - Includes metadata (SQL queries, datasets, PDF chunks)
        # - Returns chronologically ordered ChatMessage objects
        print__api_postgresql(
            f"🔍 Using consolidated get_thread_messages_with_metadata function"
        )
        chat_messages = await get_thread_messages_with_metadata(
            checkpointer, thread_id, user_email, "checkpoint_history"
        )

        # =======================================================================
        # MESSAGE VALIDATION
        # =======================================================================

        # Check if any messages were extracted from checkpoints
        # Return empty list if no conversation data found
        if not chat_messages:
            print__api_postgresql(f"⚠ No messages found for thread {thread_id}")
            return []

        print__api_postgresql(
            f"✅ Converted {len(chat_messages)} messages to frontend format"
        )

        # =======================================================================
        # DEBUG LOGGING
        # =======================================================================

        # Log detailed message information for debugging and monitoring
        # Shows message type, content preview, and metadata presence
        for i, msg in enumerate(chat_messages):
            # Determine message type for logging
            user_type = "👤 User" if msg.isUser else "🤖 AI"

            # Create content preview (truncated for readability)
            content_preview = (
                msg.content[:50] + "..." if len(msg.content) > 50 else msg.content
            )

            # Extract datasets metadata if present
            datasets_info = (
                f" (datasets: {msg.meta.get('datasetsUsed', [])})"
                if msg.meta and msg.meta.get("datasetsUsed")
                else ""
            )

            # Extract SQL query metadata if present
            sql_info = (
                f" (SQL: {msg.meta.get('sqlQuery', 'None')[:30]}...)"
                if msg.meta and msg.meta.get("sqlQuery")
                else ""
            )

            # Log complete message summary
            print__api_postgresql(
                f"{i+1}. {user_type}: {content_preview}{datasets_info}{sql_info}"
            )

        # =======================================================================
        # RESPONSE RETURN
        # =======================================================================

        return chat_messages

    except Exception as e:
        # =======================================================================
        # ERROR HANDLING
        # =======================================================================

        error_msg = str(e)
        print__api_postgresql(
            f"❌ Failed to load checkpoint messages for thread {thread_id}: {e}"
        )

        # =======================================================================
        # CONNECTION ERROR DETECTION
        # =======================================================================

        # Handle specific database connection errors gracefully
        # Return empty list instead of raising error for better UX
        # This allows the frontend to function even when database is unavailable
        if any(
            keyword in error_msg.lower()
            for keyword in [
                "ssl error",  # SSL/TLS connection issues
                "connection",  # General connection failures
                "timeout",  # Connection or query timeouts
                "operational error",  # Database operational issues
                "server closed",  # Server closed connection
                "bad connection",  # Invalid or corrupted connection
                "consuming input failed",  # Network I/O failures
            ]
        ):
            print__api_postgresql(
                f"⚠ Database connection error - returning empty messages"
            )
            return []  # Graceful degradation
        else:
            # =======================================================================
            # GENERIC ERROR HANDLING
            # =======================================================================

            # Try custom traceback response for debugging
            resp = traceback_json_response(e)
            if resp:
                return resp

            # Fallback to standard HTTP exception
            raise HTTPException(
                status_code=500, detail=f"Failed to load checkpoint messages: {e}"
            )


# ==============================================================================
# API ENDPOINT: GET MESSAGE RUN IDS FOR SENTIMENT TRACKING
# ==============================================================================


@router.get("/chat/{thread_id}/run-ids")
async def get_message_run_ids(thread_id: str, user=Depends(get_current_user)):
    """Get run_ids for messages in a thread to enable feedback submission.

    This endpoint retrieves run IDs associated with messages in a conversation thread.
    Run IDs are used to correlate user feedback/sentiment with specific AI responses,
    enabling sentiment tracking and quality monitoring.

    Each run represents one complete execution of the multi-agent system for a user query.
    The run_id is stored in the users_threads_runs table along with the prompt text
    and timestamp, providing a link between the message and its sentiment rating.

    Key Features:
        - Retrieves all run IDs for a thread in chronological order
        - Validates UUIDs and filters out invalid entries
        - Includes prompt text and timestamp for each run
        - Enables sentiment submission by providing run identifiers

    Security:
        - Requires JWT authentication token
        - Only returns runs for the authenticated user
        - Email-based filtering ensures data isolation

    Args:
        thread_id: Unique identifier for the conversation thread
        user: Authenticated user object (injected by dependency)

    Returns:
        Dict containing:
            - run_ids: Array of run metadata objects
                * run_id: UUID string for the run
                * prompt: User's query text for this run
                * timestamp: ISO formatted timestamp of run creation

    Raises:
        HTTPException 401: User email not found in authentication token

    Note:
        - Invalid UUIDs are logged and filtered from results
        - Returns empty array if no runs found or database unavailable
        - Source: users_threads_runs table

    Example Response:
        {
            "run_ids": [
                {
                    "run_id": "550e8400-e29b-41d4-a716-446655440000",
                    "prompt": "What was the population in 2020?",
                    "timestamp": "2024-01-15T10:30:00"
                }
            ]
        }
    """

    # =======================================================================
    # AUTHENTICATION AND VALIDATION
    # =======================================================================

    # Extract user email from JWT token for data filtering
    user_email = user.get("email")
    if not user_email:
        raise HTTPException(status_code=401, detail="User email not found in token")

    print__feedback_flow(f"🔍 Fetching run_ids for thread {thread_id}")

    try:
        # =======================================================================
        # DATABASE CONNECTION
        # =======================================================================

        # Get database connection pool from global checkpointer
        # This reuses the same connection pool as message retrieval
        pool = await get_global_checkpointer()
        pool = pool.conn if hasattr(pool, "conn") else None

        # =======================================================================
        # CONNECTION VALIDATION
        # =======================================================================

        # Check if connection pool is available
        # Return empty result if no database connection (graceful degradation)
        if not pool:
            print__feedback_flow("⚠ No pool available for run_id lookup")
            return {"run_ids": []}

        # =======================================================================
        # DATABASE QUERY EXECUTION
        # =======================================================================

        # Execute query to retrieve run metadata for the thread
        # Uses connection pool context manager for automatic connection cleanup
        async with pool.connection() as conn:
            print__feedback_flow(f"📊 Executing SQL query for thread {thread_id}")
            async with conn.cursor() as cur:
                # Parameterized query to prevent SQL injection
                # Filters by email and thread_id for security and data isolation
                # Orders by timestamp to maintain chronological order
                await cur.execute(
                    """
                    SELECT run_id, prompt, timestamp
                    FROM users_threads_runs 
                    WHERE email = %s AND thread_id = %s
                    ORDER BY timestamp ASC
                """,
                    (user_email, thread_id),
                )

                # =======================================================================
                # RESULT PROCESSING
                # =======================================================================

                # Initialize list for validated run data
                run_id_data = []

                # Fetch all results from query
                rows = await cur.fetchall()
                # =======================================================================
                # UUID VALIDATION AND DATA PACKAGING
                # =======================================================================

                # Process each database row, validating UUIDs and packaging data
                for row in rows:
                    print__feedback_flow(
                        f"📝 Processing database row - run_id: {row[0]}, prompt: {row[1][:50]}..."
                    )
                    try:
                        # Validate and format run_id as UUID
                        # uuid.UUID() constructor validates UUID format and raises ValueError if invalid
                        run_uuid = str(uuid.UUID(row[0])) if row[0] else None

                        if run_uuid:
                            # Valid UUID found - add to result list
                            run_id_data.append(
                                {
                                    "run_id": run_uuid,  # Validated UUID string
                                    "prompt": row[1],  # User's query text
                                    "timestamp": row[
                                        2
                                    ].isoformat(),  # ISO formatted timestamp
                                }
                            )
                            print__feedback_flow(f"✅ Valid UUID found: {run_uuid}")
                        else:
                            # Null run_id in database - log warning and skip
                            print__feedback_flow(
                                f"⚠ Null run_id found for prompt: {row[1][:50]}..."
                            )
                    except ValueError:
                        # Invalid UUID format - log error and skip this entry
                        print__feedback_flow(f"❌ Invalid UUID in database: {row[0]}")
                        continue

                # =======================================================================
                # RESPONSE RETURN
                # =======================================================================

                print__feedback_flow(
                    f"📊 Total valid run_ids found: {len(run_id_data)}"
                )
                return {"run_ids": run_id_data}

    except Exception as e:
        # =======================================================================
        # ERROR HANDLING
        # =======================================================================

        # Log error details for debugging
        print__feedback_flow(f"🚨 Error fetching run_ids: {str(e)}")

        # Try custom traceback response for detailed error information
        resp = traceback_json_response(e)
        if resp:
            return resp

        # Fallback: return empty result (graceful degradation)
        # This allows the frontend to function even if run_id retrieval fails
        return {"run_ids": []}
