"""FastAPI Analysis Endpoint - Natural Language to SQL Query Processing

This module provides the main analysis endpoint for converting natural language queries
into SQL queries and executing them against the CZSU (Czech Statistical Office) database
using a multi-agent system architecture with LangGraph.
"""

MODULE_DESCRIPTION = r"""FastAPI Analysis Endpoint - Natural Language to SQL Query Processing

This module implements the core analysis functionality for the CZSU multi-agent text-to-SQL
system. It provides REST API endpoints for processing natural language queries through a
sophisticated multi-agent workflow, managing execution lifecycle, and returning structured
results with comprehensive metadata.

Key Features:
-------------
1. Natural Language Query Processing:
   - Accepts user queries in natural language
   - Orchestrates multi-agent workflow for query understanding
   - Identifies relevant CZSU datasets automatically
   - Generates and executes SQL queries
   - Returns structured results with natural language answers

2. Multi-Agent Workflow Management:
   - Integration with LangGraph-based agent system
   - Thread-based conversation management
   - Checkpointing for state persistence
   - Run tracking with unique run_id generation
   - Execution cancellation support
   - Background task monitoring

3. Database Checkpointer Management:
   - PostgreSQL-based checkpoint persistence (primary)
   - Automatic health monitoring and recovery
   - InMemorySaver fallback for database failures
   - Prepared statement error handling with retry logic
   - Connection pool management
   - Graceful degradation strategies

4. Authentication and Authorization:
   - JWT-based user authentication
   - Email extraction from tokens
   - Thread-level access control
   - User-specific execution tracking
   - Session management

5. Concurrency and Resource Management:
   - Semaphore-based concurrency limiting
   - Maximum concurrent analysis prevention
   - Memory usage monitoring and logging
   - Garbage collection optimization
   - Resource cleanup after execution
   - Timeout protection (4-minute limit)

6. Error Handling and Recovery:
   - Comprehensive exception handling
   - Database connection fallback mechanisms
   - Prepared statement error recovery
   - Network timeout handling
   - User cancellation support
   - Detailed error logging and tracing

7. Metadata Extraction and Response:
   - Thread message retrieval via single-thread endpoint
   - Dataset and selection code extraction
   - SQL query capture
   - Top chunks (RAG context) tracking
   - Follow-up prompt generation
   - Query results compilation

8. Progress Tracking and Debugging:
   - Detailed debug logging throughout execution
   - Analysis tracing with numbered checkpoints
   - Feedback flow monitoring
   - Memory usage logging
   - Performance metrics collection
   - Error diagnostics with traceback capture

Processing Flow:
--------------
1. Request Authentication:
   - Validates JWT token from request headers
   - Extracts user email for authorization
   - Verifies user permissions

2. Request Validation:
   - Validates AnalyzeRequest structure
   - Checks thread_id and prompt
   - Handles optional run_id parameter

3. Resource Acquisition:
   - Acquires analysis semaphore for concurrency control
   - Logs memory usage baseline
   - Prepares execution environment

4. Checkpointer Initialization:
   - Retrieves healthy global checkpointer
   - Falls back to InMemorySaver on database failures
   - Handles prepared statement errors with retry

5. Thread Run Creation:
   - Creates or uses provided run_id
   - Registers execution in database
   - Enables cancellation tracking
   - Links run to user and thread

6. Analysis Execution:
   - Invokes main analysis workflow (analysis_main)
   - Monitors for user cancellation requests
   - Polls cancellation status every 0.5 seconds
   - Enforces 4-minute timeout limit
   - Handles async cancellation gracefully

7. Metadata Retrieval:
   - Calls single-thread endpoint for thread messages
   - Extracts latest AI message with metadata
   - Retrieves datasets_used, SQL queries, top_chunks
   - Compiles follow-up prompts

8. Response Preparation:
   - Constructs structured response JSON
   - Includes query results and metadata
   - Adds run_id, thread_id, iteration counts
   - Formats for frontend consumption

9. Cleanup and Return:
   - Unregisters execution from cancellation tracking
   - Forces garbage collection for memory cleanup
   - Logs final memory usage
   - Releases analysis semaphore
   - Returns JSON response to client

10. Error Recovery:
    - Handles database connection failures with fallback
    - Manages prepared statement errors with retry
    - Processes cancellation and timeout exceptions
    - Returns appropriate HTTP status codes
    - Logs detailed error traces

Endpoint Details:
---------------
POST /analyze
- Summary: Analyze natural language query
- Description: Converts natural language to SQL, executes against CZSU database
- Request Body: AnalyzeRequest (prompt, thread_id, optional run_id)
- Response: JSON with results, metadata, run_id, datasets_used, SQL, etc.
- Status Codes:
  * 200 - Successful analysis with results
  * 401 - Authentication failed (invalid/missing token)
  * 408 - Request timeout (exceeded 4-minute limit)
  * 429 - Rate limit exceeded
  * 499 - Client cancelled request
  * 500 - Internal server error

Configuration Parameters:
-----------------------
- MAX_CONCURRENT_ANALYSES: Maximum simultaneous analysis requests (default: from settings)
- INMEMORY_FALLBACK_ENABLED: Enable InMemorySaver fallback (default: from settings)
- analysis_semaphore: AsyncIO semaphore for concurrency control
- Analysis timeout: 240 seconds (4 minutes)
- Cancellation poll interval: 0.5 seconds

Checkpointer Fallback Strategy:
------------------------------
1. Primary: PostgreSQL-based persistent checkpointer
   - Handles prepared statement errors with automatic retry
   - Connection pool management
   - Health monitoring

2. Fallback: InMemorySaver (if enabled)
   - Triggered by database connection failures
   - Temporary in-memory state storage
   - No persistence across restarts
   - Generates fallback run_id if needed

3. Error Conditions for Fallback:
   - Pool connection errors
   - SSL/TLS errors
   - Connection timeout
   - Database closed errors
   - Prepared statement errors (after retry exhaustion)

Metadata Extraction Details:
---------------------------
The endpoint retrieves comprehensive metadata by calling the single-thread
endpoint to get all messages for the current thread. It extracts:

- datasets_used: List of CZSU selection codes used in analysis
- top_selection_codes: Primary datasets identified (same as datasets_used)
- queries_and_results: List of SQL queries executed with results
- sql_query: Final SQL query generated (if any)
- top_chunks: RAG context chunks used for answer generation
- followup_prompts: Suggested follow-up questions for user

This metadata is extracted from the latest AI message that contains
final_answer and datasets_used fields.

Cancellation Support:
-------------------
The endpoint supports user-initiated cancellation through:
- Execution registration with thread_id and run_id
- Periodic cancellation checks during analysis
- Graceful task cancellation via asyncio.CancelledError
- Automatic cleanup and unregistration
- 499 status code for cancelled requests

Memory Management:
----------------
- Logs memory usage at analysis start
- Monitors memory before/after garbage collection
- Forces garbage collection after analysis completion
- Uses log_memory_usage utility for tracking
- Helps prevent memory leaks in long-running processes

Debug and Tracing:
----------------
Three debug logging levels:
1. print__analyze_debug: Detailed execution flow logging
2. print__analysis_tracing_debug: Numbered checkpoint tracing
3. print__feedback_flow: User-facing progress updates

Usage Example:
-------------
# From frontend or API client
POST /analyze
{
  "prompt": "How many people live in Prague?",
  "thread_id": "user123_conv456",
  "run_id": null  // Optional, will be generated if not provided
}

# Response
{
  "prompt": "How many people live in Prague?",
  "result": "According to the latest data, Prague has approximately 1.3 million inhabitants.",
  "queries_and_results": [...],
  "thread_id": "user123_conv456",
  "datasets_used": ["OBY01PDT01"],
  "sql": "SELECT ...",
  "run_id": "abc123-...",
  "top_chunks": [...],
  "followup_prompts": [...]
}

Required Environment:
-------------------
- Python 3.10+
- FastAPI framework
- LangGraph for multi-agent workflows
- PostgreSQL database for checkpointing
- JWT authentication configured
- Environment variables for database connection
- Required packages: fastapi, asyncio, httpx, psycopg, langgraph

Error Handling Strategy:
-----------------------
1. Database Errors:
   - Detect prepared statement errors
   - Retry with exponential backoff
   - Fall back to InMemorySaver if enabled
   - Return 500 with user-friendly message

2. Timeout Errors:
   - 4-minute limit enforced
   - Return 408 status code
   - Clean up resources
   - Unregister execution

3. Cancellation:
   - Detect user cancellation
   - Cancel async task gracefully
   - Return 499 status code
   - Clean up execution state

4. Authentication Errors:
   - Validate JWT token
   - Check email extraction
   - Return 401 for auth failures

5. General Errors:
   - Catch all unexpected exceptions
   - Log detailed tracebacks
   - Return 500 with generic message
   - Ensure resource cleanup

Integration Points:
-----------------
- api/config/settings.py: Configuration parameters
- api/dependencies/auth.py: Authentication functions
- api/models/requests.py: Request model definitions
- api/utils/debug.py: Debug logging utilities
- api/utils/memory.py: Memory monitoring utilities
- api/utils/cancellation.py: Cancellation tracking utilities
- api/helpers.py: Error response helpers
- main.py: Core analysis_main function
- checkpointer/: Database checkpointer management
- api/routes/chat.py: Single-thread endpoint for metadata

Windows Compatibility:
--------------------
- Sets WindowsSelectorEventLoopPolicy for asyncio
- Must be done before other imports
- Fixes psycopg compatibility issues on Windows
- Essential for database connection stability"""

# CRITICAL: Set Windows event loop policy FIRST, before any other imports
# This must be the very first thing that happens to fix psycopg compatibility
# with asyncio on Windows platforms. This prevents "Event loop is closed" errors
# and ensures proper async database operations.
import asyncio
import gc
import os
import sys
import traceback
import uuid

# Load environment variables early (before other module imports)
# This ensures configuration is available for all subsequent imports
from dotenv import load_dotenv

if sys.platform == "win32":
    import asyncio

    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

load_dotenv()

# ==============================================================================
# PATH AND DIRECTORY CONSTANTS
# ==============================================================================

# Determine base directory for the project
# Handles both normal execution and special environments (e.g., REPL, Jupyter)
try:
    from pathlib import Path

    # Navigate up two directories from this file to reach project root
    # Structure: project_root/api/routes/analysis.py -> go up 2 levels
    BASE_DIR = Path(__file__).resolve().parents[2]
except NameError:
    # Fallback for environments where __file__ is not defined
    BASE_DIR = Path(os.getcwd()).parents[0]

# ==============================================================================
# FASTAPI AND HTTP IMPORTS
# ==============================================================================
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

# ==============================================================================
# CONFIGURATION AND SETTINGS
# ==============================================================================

# Import configuration parameters for concurrency control and fallback behavior
from api.config.settings import (
    INMEMORY_FALLBACK_ENABLED,  # Flag to enable/disable InMemorySaver fallback
    MAX_CONCURRENT_ANALYSES,  # Maximum simultaneous analysis requests allowed
    analysis_semaphore,  # AsyncIO semaphore for concurrency control
)

# ==============================================================================
# AUTHENTICATION AND AUTHORIZATION
# ==============================================================================

# Import JWT-based authentication dependency
from api.dependencies.auth import get_current_user

# ==============================================================================
# REQUEST/RESPONSE MODELS
# ==============================================================================

# Import Pydantic models for request validation
from api.models.requests import AnalyzeRequest

# ==============================================================================
# DEBUG AND LOGGING UTILITIES
# ==============================================================================

# Import specialized debug logging functions for different verbosity levels
from api.utils.debug import (
    print__analysis_tracing_debug,  # Numbered checkpoint tracing
    print__analyze_debug,  # Detailed execution flow logging
    print__feedback_flow,  # User-facing progress updates
)

# ==============================================================================
# MEMORY MONITORING UTILITIES
# ==============================================================================

# Import memory usage tracking for resource management
from api.utils.memory import log_memory_usage

# ==============================================================================
# EXECUTION CANCELLATION UTILITIES
# ==============================================================================

# Import cancellation tracking for user-initiated request cancellation
from api.utils.cancellation import (
    register_execution,  # Register execution for cancellation tracking
    unregister_execution,  # Remove execution from tracking after completion
    is_cancelled,  # Check if execution has been cancelled by user
)

# ==============================================================================
# PROJECT-SPECIFIC IMPORTS
# ==============================================================================

# Add project root to Python path for direct imports
sys.path.insert(0, str(BASE_DIR))

# HTTP client for internal API calls (currently unused but available)
import httpx

# Import error response formatting helper
from api.helpers import traceback_json_response

# Import core analysis workflow orchestrator
from main import main as analysis_main

# Import database operations for thread and run management
from checkpointer.user_management.thread_operations import create_thread_run_entry
from checkpointer.checkpointer.factory import get_global_checkpointer

# ==============================================================================
# ENVIRONMENT CONFIGURATION
# ==============================================================================

# Load environment variables from .env file (second load to ensure all imports have access)
load_dotenv()

# ==============================================================================
# FASTAPI ROUTER INITIALIZATION
# ==============================================================================

# Create router instance for analysis endpoints
# This router will be included in the main FastAPI application
router = APIRouter()


async def get_thread_metadata_from_single_thread_endpoint(
    thread_id: str, user_email: str
) -> dict:
    """Retrieve comprehensive metadata for a thread by calling the single-thread endpoint.

    This function extracts metadata from the latest AI message in a thread, including
    datasets used, SQL queries, RAG chunks, and follow-up prompts. It calls the
    get_all_chat_messages_for_one_thread function directly instead of making an HTTP
    request for better performance.

    Args:
        thread_id (str): The unique identifier for the conversation thread
        user_email (str): The email address of the authenticated user

    Returns:
        dict: Dictionary containing extracted metadata with keys:
            - top_selection_codes: List of CZSU selection codes (datasets used)
            - datasets_used: Same as top_selection_codes
            - queries_and_results: List of SQL queries executed with results
            - sql: Final SQL query string (or None)
            - dataset_url: URL to dataset (deprecated, always None)
            - top_chunks: RAG context chunks used in answer generation
            - followup_prompts: Suggested follow-up questions

    Note:
        - Returns empty metadata structure on any error
        - Searches messages in reverse order to find latest completed AI message
        - Requires message to have both final_answer and datasets_used fields
    """
    try:
        print__analyze_debug(
            f"🔍 Calling single-thread endpoint for thread: {thread_id}"
        )
        print__analysis_tracing_debug(
            f"METADATA EXTRACTION: Calling /chat/all-messages-for-one-thread/{thread_id}"
        )

        # Import the single-thread function directly to avoid HTTP overhead
        # This is more efficient than making an HTTP request to our own endpoint
        from unittest.mock import Mock

        from fastapi import Request

        from api.routes.chat import get_all_chat_messages_for_one_thread

        # Create a mock user object with email for authentication
        # This simulates the authenticated user context without HTTP overhead
        mock_user = {"email": user_email}

        # Call the function directly instead of making HTTP request for better performance
        print__analyze_debug(f"🔍 Calling single-thread function directly")
        result = await get_all_chat_messages_for_one_thread(thread_id, mock_user)

        # Handle different response types from the single-thread endpoint
        if isinstance(result, dict):
            # Direct dict response (from cache hit or direct return path)
            response_data = result
        else:
            # JSONResponse object - need to extract the actual data
            if hasattr(result, "body"):
                import json

                # Decode the response body from bytes and parse JSON
                response_data = json.loads(result.body.decode())
            else:
                # Unexpected response type - log warning and return empty metadata
                print__analyze_debug(
                    f"⚠️ Unexpected response type from single-thread endpoint: {type(result)}"
                )
                return {}

        print__analyze_debug(
            f"🔍 Single-thread endpoint returned {len(response_data.get('messages', []))} messages"
        )

        # Extract data from the single-thread endpoint response
        messages = response_data.get("messages", [])
        run_ids = response_data.get(
            "runIds", []
        )  # Currently not used in metadata extraction
        sentiments = response_data.get(
            "sentiments", {}
        )  # Currently not used in metadata extraction

        # Find the latest completed AI message that has both final_answer and metadata
        # We search in reverse order to get the most recent completed analysis
        latest_ai_message = None
        for message in reversed(messages):
            # With unified message structure, look for messages that have final_answer
            # and datasets_used (indicating a completed analysis)
            if message.get("final_answer") and message.get("datasets_used"):
                latest_ai_message = message
                break

        # Build metadata dictionary from the extracted AI message
        metadata = {}
        if latest_ai_message:
            # Extract all relevant metadata fields from the unified message structure
            # Each field is extracted with a default fallback to prevent KeyError
            metadata.update(
                {
                    "top_selection_codes": latest_ai_message.get(
                        "datasets_used", []
                    ),  # CZSU selection codes
                    "datasets_used": latest_ai_message.get(
                        "datasets_used", []
                    ),  # Duplicate for backward compatibility
                    "queries_and_results": latest_ai_message.get(
                        "queries_and_results", []
                    ),  # SQL queries with results
                    "sql": latest_ai_message.get(
                        "sql_query"
                    ),  # Final SQL query (may be None)
                    "dataset_url": None,  # Deprecated field, no longer used in current structure
                    "top_chunks": latest_ai_message.get(
                        "top_chunks", []
                    ),  # RAG context chunks
                    "followup_prompts": latest_ai_message.get(
                        "followup_prompts", []
                    ),  # Suggested follow-ups
                }
            )
            print__analyze_debug(
                f"🔍 Extracted metadata: top_selection_codes={len(metadata.get('top_selection_codes', []))}"
            )
        else:
            # No completed AI message found - return empty metadata structure
            print__analyze_debug(
                f"⚠️ No AI message with metadata found in single-thread response"
            )
            # Return empty metadata with all expected keys to prevent frontend errors
            metadata = {
                "top_selection_codes": [],
                "datasets_used": [],
                "queries_and_results": [],
                "sql": None,
                "dataset_url": None,
                "top_chunks": [],
                "followup_prompts": [],
            }

        return metadata

    except Exception as e:
        # Handle any errors during metadata extraction gracefully
        print__analyze_debug(f"🚨 Error calling single-thread endpoint: {e}")
        print__analysis_tracing_debug(f"METADATA EXTRACTION ERROR: {e}")

        # Try to generate error response for logging
        resp = traceback_json_response(
            e, run_id=None
        )  # run_id not yet generated at this point
        if resp:
            return resp

        # Return empty metadata structure on error to prevent cascading failures
        # This allows the analysis to complete even if metadata extraction fails
        return {
            "top_selection_codes": [],
            "datasets_used": [],
            "queries_and_results": [],
            "sql": None,
            "dataset_url": None,
            "top_chunks": [],
            "followup_prompts": [],
        }


@router.post(
    "/analyze",
    summary="Analyze natural language query",
    description="""
    **Convert a natural language query to SQL and execute it against the CZSU database.**
    
    This endpoint uses a multi-agent system to:
    1. Parse the user's natural language query
    2. Identify relevant CZSU datasets
    3. Generate appropriate SQL queries
    4. Execute queries and return results
    5. Format a natural language answer
    
    **Rate Limiting:** Subject to per-IP rate limits. See 429 responses for retry information.
    
    **Concurrency:** Limited to {MAX_CONCURRENT_ANALYSES} concurrent requests to prevent resource exhaustion.
    """,
    response_description="Streaming response with query results and metadata",
    responses={
        200: {
            "description": "Successful analysis with streaming response",
            "content": {
                "text/event-stream": {
                    "example": "data: {'status': 'processing', 'message': 'Analyzing query...'}\n\n"
                }
            },
        },
        401: {"description": "Authentication failed - Invalid or missing token"},
        429: {"description": "Rate limit exceeded - Too many requests"},
        500: {"description": "Internal error during query processing"},
    },
)
async def analyze(request: AnalyzeRequest, user=Depends(get_current_user)):
    """Main analysis endpoint - converts natural language to SQL and executes against CZSU database.

    This endpoint orchestrates the complete multi-agent workflow for processing user queries:
    1. Authenticates the user via JWT token
    2. Acquires resources (semaphore, checkpointer)
    3. Creates or retrieves run_id for tracking
    4. Executes the analysis workflow with cancellation support
    5. Retrieves metadata from completed thread
    6. Returns structured response with results and metadata

    Args:
        request (AnalyzeRequest): Request object containing:
            - prompt: Natural language query from user
            - thread_id: Unique conversation thread identifier
            - run_id: Optional run identifier (generated if not provided)
        user (dict): Authenticated user object from JWT token dependency

    Returns:
        dict: JSON response containing:
            - prompt: Original user query
            - result: Natural language answer
            - queries_and_results: List of SQL queries and their results
            - thread_id: Conversation thread identifier
            - datasets_used: List of CZSU selection codes used
            - sql: Final SQL query (if any)
            - run_id: Unique execution identifier
            - top_chunks: RAG context chunks used
            - followup_prompts: Suggested follow-up questions

    Raises:
        HTTPException:
            - 401: Authentication failed (no email in token)
            - 408: Request timeout (exceeded 4-minute limit)
            - 499: Client cancelled request
            - 500: Internal server error

    Note:
        - Limited to MAX_CONCURRENT_ANALYSES simultaneous requests
        - Uses PostgreSQL checkpointer with InMemorySaver fallback
        - Supports user-initiated cancellation
        - Includes comprehensive error handling and logging
    """

    # ==========================================================================
    # ENTRY POINT LOGGING AND REQUEST VALIDATION
    # ==========================================================================
    print__analysis_tracing_debug("01 - ANALYZE ENDPOINT ENTRY: Request received")
    print__analyze_debug("🔍 ANALYZE ENDPOINT - ENTRY POINT")
    print__analyze_debug(f"🔍 Request received: thread_id={request.thread_id}")
    print__analyze_debug(f"🔍 Request prompt length: {len(request.prompt)}")

    try:
        # ======================================================================
        # USER AUTHENTICATION AND EMAIL EXTRACTION
        # ======================================================================
        print__analysis_tracing_debug(
            "02 - USER EXTRACTION: Getting user email from token"
        )
        # Extract user email from the JWT token payload
        # The user object is injected by the get_current_user dependency
        user_email = user.get("email")
        print__analyze_debug(f"🔍 User extraction: {user_email}")

        # Validate that user email exists in token payload
        if not user_email:
            print__analysis_tracing_debug("03 - ERROR: No user email found in token")
            print__analyze_debug("🚨 No user email found in token")
            raise HTTPException(status_code=401, detail="User email not found in token")

        # ======================================================================
        # AUTHENTICATION SUCCESS - LOG USER INFO
        # ======================================================================
        print__analysis_tracing_debug(
            f"04 - USER VALIDATION SUCCESS: User {user_email} validated"
        )
        print__feedback_flow(
            f"📝 New analysis request - Thread: {request.thread_id}, User: {user_email}"
        )
        print__analyze_debug(
            f"🔍 ANALYZE REQUEST RECEIVED: thread_id={request.thread_id}, user={user_email}"
        )

        # ======================================================================
        # MEMORY BASELINE MONITORING
        # ======================================================================
        print__analysis_tracing_debug("05 - MEMORY MONITORING: Starting memory logging")
        # Establish baseline memory usage for comparison after analysis
        print__analyze_debug("🔍 Starting memory logging")
        log_memory_usage("analysis_start")
        run_id = None  # Initialize run_id for error handling scope

        # ======================================================================
        # CONCURRENCY CONTROL - SEMAPHORE ACQUISITION
        # ======================================================================
        print__analysis_tracing_debug(
            "06 - SEMAPHORE ACQUISITION: Attempting to acquire analysis semaphore"
        )
        print__analyze_debug("🔍 About to acquire analysis semaphore")
        # Limit concurrent analyses to prevent resource exhaustion and ensure platform stability
        # The semaphore blocks if MAX_CONCURRENT_ANALYSES is already running
        async with analysis_semaphore:
            print__analysis_tracing_debug(
                "07 - SEMAPHORE ACQUIRED: Analysis semaphore acquired"
            )
            print__feedback_flow("🔒 Acquired analysis semaphore")
            print__analyze_debug("🔍 Semaphore acquired successfully")

            try:
                # ==============================================================
                # CHECKPOINTER INITIALIZATION
                # ==============================================================
                print__analysis_tracing_debug(
                    "08 - CHECKPOINTER INITIALIZATION: Getting healthy checkpointer"
                )
                print__analyze_debug("🔍 About to get healthy checkpointer")
                print__feedback_flow("🔄 Getting healthy checkpointer")
                # Retrieve the global checkpointer instance with health monitoring
                # This handles automatic recovery and fallback if needed
                checkpointer = await get_global_checkpointer()
                print__analysis_tracing_debug(
                    f"09 - CHECKPOINTER SUCCESS: Checkpointer obtained ({type(checkpointer).__name__})"
                )
                print__analyze_debug(
                    f"🔍 Checkpointer obtained: {type(checkpointer).__name__}"
                )

                # ==============================================================
                # THREAD RUN ENTRY CREATION
                # ==============================================================
                print__analysis_tracing_debug(
                    "10 - THREAD RUN ENTRY: Creating thread run entry in database"
                )
                print__analyze_debug("🔍 About to create thread run entry")
                print__feedback_flow("🔄 Creating thread run entry")

                # Use run_id from request if provided, otherwise generate new UUID
                run_id = request.run_id if request.run_id else None
                if run_id:
                    print__analyze_debug(f"🔍 Using run_id from request: {run_id}")
                else:
                    print__analyze_debug("🔍 No run_id in request, will be generated")

                # Create database entry for this analysis run, linking it to user and thread
                run_id = await create_thread_run_entry(
                    user_email, request.thread_id, request.prompt, run_id=run_id
                )
                print__analysis_tracing_debug(
                    f"11 - THREAD RUN SUCCESS: Thread run entry created with run_id {run_id}"
                )
                print__feedback_flow(f"✅ Generated new run_id: {run_id}")
                print__analyze_debug(
                    f"🔍 Thread run entry created successfully: {run_id}"
                )

                print__analysis_tracing_debug(
                    f"11.5 - REGISTER CANCELLATION: Registering execution for cancellation tracking"
                )
                # Register this execution for potential user-initiated cancellation
                # This enables the frontend to cancel long-running analyses
                register_execution(request.thread_id, run_id)

                print__analysis_tracing_debug(
                    "12 - ANALYSIS MAIN START: Starting analysis_main function"
                )
                print__analyze_debug("🔍 About to start analysis_main")
                print__feedback_flow("🚀 Starting analysis")

                # ==============================================================
                # CANCELLABLE ANALYSIS WRAPPER
                # ==============================================================

                # Create a wrapper that allows user-initiated cancellation
                async def cancellable_analysis():
                    """Wrapper that checks for cancellation periodically.

                    This function wraps the main analysis workflow and polls
                    for cancellation requests every 0.5 seconds. If cancellation
                    is detected, it gracefully cancels the analysis task.
                    """
                    # Start the analysis as an async task so we can monitor it
                    task = asyncio.create_task(
                        analysis_main(
                            request.prompt,
                            thread_id=request.thread_id,
                            checkpointer=checkpointer,
                            run_id=run_id,
                        )
                    )

                    # Poll for cancellation every 5 seconds while task runs
                    while not task.done():
                        # Check if user requested cancellation via API
                        if is_cancelled(request.thread_id, run_id):
                            print__analyze_debug(
                                f"🛑 Cancellation detected for run_id: {run_id}"
                            )
                            # Cancel the async task gracefully
                            task.cancel()
                            try:
                                await task
                            except asyncio.CancelledError:
                                pass
                            raise asyncio.CancelledError("Execution cancelled by user")

                        # Wait 5 seconds before next cancellation check
                        # Use shield to prevent timeout from cancelling the main task
                        try:
                            await asyncio.wait_for(asyncio.shield(task), timeout=5)
                        except asyncio.TimeoutError:
                            # Timeout just means we should check cancellation again
                            continue
                        except asyncio.CancelledError:
                            # Task was cancelled - re-raise
                            raise

                    # Task completed normally without cancellation
                    return await task

                # ==============================================================
                # ANALYSIS EXECUTION WITH TIMEOUT
                # ==============================================================

                # Execute the cancellable analysis with a 4-minute timeout
                # This prevents indefinite execution and ensures platform stability
                result = await asyncio.wait_for(
                    cancellable_analysis(),
                    timeout=240,  # 4 minutes timeout for platform stability
                )

                # Analysis completed successfully without timeout or cancellation
                print__analysis_tracing_debug(
                    "13 - ANALYSIS MAIN SUCCESS: Analysis completed successfully"
                )
                print__analyze_debug("🔍 Analysis completed successfully")
                print__feedback_flow("✅ Analysis completed successfully")

                # Unregister execution from cancellation tracking after successful completion
                unregister_execution(request.thread_id, run_id)

            except Exception as analysis_error:
                # ==============================================================
                # ANALYSIS EXCEPTION HANDLER
                # ==============================================================
                print__analysis_tracing_debug(
                    f"14 - ANALYSIS ERROR: Exception in analysis block - {type(analysis_error).__name__}"
                )
                print__analyze_debug(
                    f"🚨 Exception in analysis block: {type(analysis_error).__name__}: {str(analysis_error)}"
                )

                # Convert error message to lowercase for case-insensitive matching
                error_msg = str(analysis_error).lower()
                print__analyze_debug(f"🔍 Error message (lowercase): {error_msg}")

                # ==============================================================
                # PREPARED STATEMENT ERROR DETECTION
                # ==============================================================

                # PostgreSQL prepared statement errors occur when statement names are reused
                # These are typically transient and should be retried by the calling code
                is_prepared_stmt_error = any(
                    indicator in error_msg
                    for indicator in [
                        "prepared statement",  # Direct prepared statement error
                        "does not exist",  # Statement doesn't exist error
                        "_pg3_",  # psycopg3 prepared statement prefix
                        "_pg_",  # Generic PostgreSQL prepared statement prefix
                        "invalidsqlstatementname",  # PostgreSQL error code
                    ]
                )

                if is_prepared_stmt_error:
                    # Prepared statement error detected - should be handled by retry decorator
                    print__analysis_tracing_debug(
                        "15 - PREPARED STATEMENT ERROR: Prepared statement error detected"
                    )
                    print__analyze_debug(
                        f"🔧 PREPARED STATEMENT ERROR DETECTED: {analysis_error}"
                    )
                    print__feedback_flow(
                        f"🔧 Prepared statement error detected - this should be handled by retry logic: {analysis_error}"
                    )

                    # Generate error response for logging
                    resp = traceback_json_response(analysis_error, run_id=run_id)
                    if resp:
                        return resp

                    # Re-raise prepared statement errors - they should be handled by the retry decorator
                    # The calling code should have retry logic to handle these transient errors
                    raise HTTPException(
                        status_code=500,
                        detail="Database prepared statement error. Please try again. {analysis_error}",
                    ) from analysis_error
                elif any(
                    keyword in error_msg
                    for keyword in [
                        "pool",  # Connection pool errors
                        "connection",  # Generic connection errors
                        "closed",  # Connection closed errors
                        "timeout",  # Connection timeout errors
                        "ssl",  # SSL/TLS errors
                        "postgres",  # General PostgreSQL errors
                    ]
                ):
                    # ==============================================================
                    # DATABASE CONNECTION ERROR - ATTEMPT FALLBACK
                    # ==============================================================
                    print__analysis_tracing_debug(
                        "16 - DATABASE FALLBACK: Database issue detected, attempting fallback"
                    )
                    print__analyze_debug(
                        "🔍 Database issue detected, attempting fallback"
                    )
                    print__feedback_flow(
                        f"⚠️ Database issue detected, trying with InMemorySaver fallback: {analysis_error}"
                    )

                    # Check if InMemorySaver fallback is enabled in configuration
                    if not INMEMORY_FALLBACK_ENABLED:
                        print__analysis_tracing_debug(
                            "17 - FALLBACK DISABLED: InMemorySaver fallback is disabled by configuration"
                        )
                        print__analyze_debug(
                            f"🚫 InMemorySaver fallback is DISABLED by configuration - re-raising database error"
                        )
                        print__feedback_flow(
                            f"🚫 InMemorySaver fallback disabled - propagating database error: {analysis_error}"
                        )
                        resp = traceback_json_response(analysis_error, run_id=run_id)
                        if resp:
                            return resp

                        raise HTTPException(
                            status_code=500,
                            detail="Database connection error. Please try again. {analysis_error}",
                        ) from analysis_error

                    try:
                        # ==========================================================
                        # INMEMORY FALLBACK INITIALIZATION
                        # ==========================================================
                        print__analysis_tracing_debug(
                            "17 - FALLBACK INITIALIZATION: Importing InMemorySaver"
                        )
                        print__analyze_debug(f"🔍 Importing InMemorySaver")
                        from langgraph.checkpoint.memory import InMemorySaver

                        # Create in-memory checkpointer as fallback
                        # Note: This will not persist state across server restarts
                        fallback_checkpointer = InMemorySaver()
                        print__analysis_tracing_debug(
                            "18 - FALLBACK CHECKPOINTER: InMemorySaver created"
                        )
                        print__analyze_debug(f"🔍 InMemorySaver created")

                        # Generate a fallback run_id since database creation might have failed
                        if run_id is None:
                            run_id = str(uuid.uuid4())
                            print__analysis_tracing_debug(
                                f"19 - FALLBACK RUN ID: Generated fallback run_id {run_id}"
                            )
                            print__feedback_flow(
                                f"✅ Generated fallback run_id: {run_id}"
                            )
                            print__analyze_debug(
                                f"🔍 Generated fallback run_id: {run_id}"
                            )

                        # ==========================================================
                        # RETRY ANALYSIS WITH FALLBACK CHECKPOINTER
                        # ==========================================================
                        print__analysis_tracing_debug(
                            "20 - FALLBACK ANALYSIS: Starting fallback analysis"
                        )
                        print__analyze_debug(f"🔍 Starting fallback analysis")
                        print__feedback_flow(
                            f"🚀 Starting analysis with InMemorySaver fallback"
                        )
                        # Retry the analysis with in-memory checkpointer
                        result = await asyncio.wait_for(
                            analysis_main(
                                request.prompt,
                                thread_id=request.thread_id,
                                checkpointer=fallback_checkpointer,
                                run_id=run_id,
                            ),
                            timeout=240,  # 4 minutes timeout
                        )

                        print__analysis_tracing_debug(
                            "21 - FALLBACK SUCCESS: Fallback analysis completed successfully"
                        )
                        print__analyze_debug(
                            f"🔍 Fallback analysis completed successfully"
                        )
                        print__feedback_flow(
                            f"✅ Analysis completed successfully with fallback"
                        )

                    except Exception as fallback_error:
                        # ==========================================================
                        # FALLBACK ALSO FAILED
                        # ==========================================================
                        print__analysis_tracing_debug(
                            f"22 - FALLBACK FAILED: Fallback also failed - {type(fallback_error).__name__}"
                        )
                        print__analyze_debug(
                            f"🚨 Fallback also failed: {type(fallback_error).__name__}: {str(fallback_error)}"
                        )
                        print__feedback_flow(
                            f"🚨 Fallback analysis also failed: {fallback_error}"
                        )
                        resp = traceback_json_response(fallback_error)
                        if resp:
                            return resp

                        raise HTTPException(
                            status_code=500,
                            detail="Sorry, there was an error processing your request. Please try again.",
                        )
                else:
                    # ==============================================================
                    # NON-DATABASE ERROR - RE-RAISE
                    # ==============================================================

                    # Re-raise errors that are not database-related
                    print__analysis_tracing_debug(
                        f"23 - NON-DATABASE ERROR: Non-database error - {type(analysis_error).__name__}"
                    )
                    print__analyze_debug(
                        f"🚨 Non-database error, re-raising: {type(analysis_error).__name__}: {str(analysis_error)}"
                    )
                    print__feedback_flow(f"🚨 Non-database error: {analysis_error}")

                    resp = traceback_json_response(analysis_error, run_id=run_id)
                    if resp:
                        return resp

                    raise HTTPException(
                        status_code=500,
                        detail="Sorry, there was an error processing your request. Please try again.",
                    )

            # ==================================================================
            # RESPONSE PREPARATION
            # ==================================================================
            print__analysis_tracing_debug(
                "24 - RESPONSE PREPARATION: Preparing response data"
            )
            print__analyze_debug(f"🔍 About to prepare response data")

            # ==================================================================
            # METADATA EXTRACTION FROM SINGLE-THREAD ENDPOINT
            # ==================================================================

            # Instead of using metadata from the analysis result, we call the
            # single-thread endpoint to get the most up-to-date thread metadata
            print__analysis_tracing_debug(
                "24a - METADATA EXTRACTION: Getting metadata from single-thread endpoint"
            )
            print__analyze_debug(
                f"🔍 Getting metadata from single-thread endpoint for thread: {request.thread_id}"
            )
            thread_metadata = await get_thread_metadata_from_single_thread_endpoint(
                request.thread_id, user_email
            )
            print__analyze_debug(
                f"🔍 Retrieved metadata from single-thread endpoint: {list(thread_metadata.keys())}"
            )

            # ==================================================================
            # RESPONSE DATA CONSTRUCTION
            # ==================================================================

            # Build the response dictionary with all required fields
            # Metadata comes from single-thread endpoint, result from analysis_main
            response_data = {
                "prompt": request.prompt,  # Original user query
                "result": (  # Natural language answer
                    result["result"]
                    if isinstance(result, dict) and "result" in result
                    else str(result)
                ),
                "queries_and_results": thread_metadata.get(
                    "queries_and_results", []
                ),  # SQL queries executed
                "thread_id": request.thread_id,  # Conversation thread ID
                "top_selection_codes": thread_metadata.get(
                    "top_selection_codes", []
                ),  # Dataset codes used
                "datasets_used": thread_metadata.get(
                    "datasets_used", []
                ),  # Same as top_selection_codes
                "iteration": (  # Current iteration count (for multi-iteration workflows)
                    result.get("iteration", 0) if isinstance(result, dict) else 0
                ),
                "max_iterations": (  # Maximum iterations allowed
                    result.get("max_iterations", 2) if isinstance(result, dict) else 2
                ),
                "sql": thread_metadata.get("sql", None),  # Final SQL query generated
                "datasetUrl": thread_metadata.get(
                    "dataset_url", None
                ),  # Dataset URL (deprecated)
                "run_id": run_id,  # Unique execution identifier
                "top_chunks": thread_metadata.get(
                    "top_chunks", []
                ),  # RAG context chunks
                "followup_prompts": thread_metadata.get(
                    "followup_prompts", []
                ),  # Suggested follow-up questions
            }

            # ==================================================================
            # DEBUG LOGGING FOR RESPONSE VALIDATION
            # ==================================================================

            # Log extracted metadata for debugging and validation
            print__analyze_debug(
                f"🔍 DEBUG RESPONSE: datasets_used extracted from single-thread: {response_data['datasets_used']}"
            )
            print__analyze_debug(
                f"🔍 DEBUG RESPONSE: top_selection_codes extracted from single-thread: {response_data['top_selection_codes']}"
            )
            print__analyze_debug(
                f"🔍 DEBUG RESPONSE: queries_and_results count: {len(response_data['queries_and_results'])}"
            )
            print__analyze_debug(
                f"🔍 DEBUG RESPONSE: sql query available: {'Yes' if response_data['sql'] else 'No'}"
            )
            print__analyze_debug(
                f"🔍 DEBUG RESPONSE: top_chunks count: {len(response_data['top_chunks'])}"
            )
            print__analyze_debug(
                f"🔍 DEBUG RESPONSE: datasetUrl: {response_data['datasetUrl']}"
            )
            # Critical validation: Ensure run_id is present and valid
            print__analyze_debug(
                f"🔍 CRITICAL - RESPONSE run_id: {response_data.get('run_id', 'MISSING')}"
            )
            print__analyze_debug(
                f"🔍 CRITICAL - run_id type: {type(response_data.get('run_id')).__name__}"
            )
            print__analyze_debug(
                f"🔍 CRITICAL - run_id length: {len(response_data.get('run_id', '')) if response_data.get('run_id') else 0}"
            )

            # ==================================================================
            # RESPONSE PREPARATION SUCCESS
            # ==================================================================
            print__analysis_tracing_debug(
                f"25 - RESPONSE SUCCESS: Response data prepared with {len(response_data.keys())} keys"
            )
            print__analyze_debug(f"🔍 Response data prepared successfully")
            print__analyze_debug(f"🔍 Response data keys: {list(response_data.keys())}")

            # ==================================================================
            # MEMORY CLEANUP AND GARBAGE COLLECTION
            # ==================================================================

            # Force garbage collection to free memory after analysis completion
            # This is important for long-running processes to prevent memory leaks
            print__analyze_debug("🧹 Running garbage collection to free memory")
            log_memory_usage("before_gc")  # Log memory usage before GC
            gc.collect()  # Force garbage collection
            log_memory_usage("after_gc")  # Log memory usage after GC to measure impact
            print__analyze_debug("🧹 Garbage collection completed")

            # ==================================================================
            # SUCCESSFUL EXIT
            # ==================================================================
            print__analyze_debug(f"🔍 ANALYZE ENDPOINT - SUCCESSFUL EXIT")
            return response_data

    # ==========================================================================
    # EXCEPTION HANDLERS FOR DIFFERENT ERROR TYPES
    # ==========================================================================
    except asyncio.CancelledError:
        # ======================================================================
        # USER-INITIATED CANCELLATION HANDLER
        # ======================================================================
        error_msg = "Analysis was cancelled by user"
        print__analysis_tracing_debug(
            "26 - CANCELLED ERROR: Analysis was cancelled by user"
        )
        print__analyze_debug(f"🛑 CANCELLED: {error_msg}")
        print__feedback_flow(f"🛑 {error_msg}")

        # Unregister execution from cancellation tracking on cancellation
        if run_id:
            unregister_execution(request.thread_id, run_id)

        # Return 499 status code (Client Closed Request) for cancelled requests
        raise HTTPException(
            status_code=499, detail=error_msg
        )  # 499 Client Closed Request

    except asyncio.TimeoutError:
        # ======================================================================
        # TIMEOUT ERROR HANDLER (4-MINUTE LIMIT EXCEEDED)
        # ======================================================================
        error_msg = "Analysis timed out after 8 minutes"
        print__analysis_tracing_debug(
            "27 - TIMEOUT ERROR: Analysis timed out after 8 minutes"
        )
        print__analyze_debug(f"🚨 TIMEOUT ERROR: {error_msg}")
        print__feedback_flow(f"🚨 {error_msg}")

        # Unregister execution from cancellation tracking on timeout
        if run_id:
            unregister_execution(request.thread_id, run_id)

        # Generate error response with traceback for logging
        resp = traceback_json_response(asyncio.TimeoutError(), run_id=run_id)
        if resp:
            return resp

        # Return 408 status code (Request Timeout)
        raise HTTPException(status_code=408, detail=error_msg)

    except HTTPException as http_exc:
        # ======================================================================
        # HTTP EXCEPTION HANDLER (RE-RAISE WITH CLEANUP)
        # ======================================================================
        print__analysis_tracing_debug(
            f"28 - HTTP EXCEPTION: HTTP exception {http_exc.status_code}"
        )
        print__analyze_debug(
            f"🚨 HTTP EXCEPTION: {http_exc.status_code} - {http_exc.detail}"
        )

        # Unregister execution from cancellation tracking on HTTP exception
        if run_id:
            unregister_execution(request.thread_id, run_id)

        # Generate error response with traceback for logging
        resp = traceback_json_response(http_exc, run_id=run_id)
        if resp:
            return resp

        # Re-raise the HTTP exception to be handled by FastAPI
        raise http_exc

    except Exception as e:
        # ======================================================================
        # UNEXPECTED EXCEPTION HANDLER (CATCH-ALL)
        # ======================================================================
        error_msg = f"Analysis failed: {str(e)}"
        print__analysis_tracing_debug(
            f"29 - UNEXPECTED EXCEPTION: Unexpected exception - {type(e).__name__}"
        )
        print__analyze_debug(f"🚨 UNEXPECTED EXCEPTION: {type(e).__name__}: {str(e)}")
        print__analyze_debug(f"🚨 Exception traceback: {traceback.format_exc()}")
        print__feedback_flow(f"🚨 {error_msg}")

        # Unregister execution from cancellation tracking on unexpected exception
        if run_id:
            unregister_execution(request.thread_id, run_id)

        # Generate error response with full traceback for debugging
        resp = traceback_json_response(e, run_id=run_id)
        if resp:
            return resp

        # Return generic error message to user while logging detailed error
        raise HTTPException(
            status_code=500,
            detail="Sorry, there was an error processing your request. Please try again.",
        )
