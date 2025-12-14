"""Chat API Routes for Multi-Agent Text-to-SQL System

This module provides FastAPI REST endpoints for managing chat threads, messages, and user interactions
with the LangGraph-based multi-agent text-to-SQL system that queries Czech Statistical Office (CZSU) data.
"""

MODULE_DESCRIPTION = r"""Chat API Routes for Multi-Agent Text-to-SQL System

This module serves as the primary REST API interface for the chat functionality of a sophisticated
multi-agent system that converts natural language queries into SQL, executes them against CZSU
statistical databases, and returns conversational responses with rich metadata.

Key Features:
-------------
1. Chat Thread Management:
   - Paginated retrieval of user chat threads with metadata
   - Thread-level information including run counts and timestamps
   - Title generation from first user prompts
   - Thread ownership verification and security controls
   - Thread deletion with cascading checkpoint cleanup

2. Message Retrieval and Processing:
   - Complete conversation history extraction from LangGraph checkpoints
   - Chronological message ordering by checkpoint step numbers
   - Dual extraction of user prompts and AI responses
   - Per-interaction metadata isolation (queries, datasets, chunks)
   - Efficient bulk processing with caching mechanisms
   - Security-verified access to thread data

3. Metadata Extraction:
   - SQL queries and execution results per interaction
   - Dataset selection codes (CZSU data sources) used
   - PDF document chunks retrieved for context
   - Follow-up prompt suggestions for continued conversation
   - Source file references and page numbers from documents
   - Complete checkpoint metadata preservation

4. Sentiment Tracking:
   - User satisfaction ratings per interaction (run_id based)
   - Sentiment retrieval for individual threads
   - Integration with PostgreSQL sentiment storage
   - Run ID matching for sentiment-to-message correlation

5. Checkpoint Integration:
   - LangGraph AsyncPostgresSaver checkpoint access
   - Fallback mechanisms for checkpoint retrieval (alist → aget_tuple)
   - Checkpoint tuple processing for message reconstruction
   - Metadata extraction from writes.__start__ and submit_final_answer
   - State snapshot handling for complete conversation context

6. Security and Authentication:
   - JWT-based user authentication via dependency injection
   - Thread ownership validation before data access
   - Email-based user identification and authorization
   - Database-level security checks for thread access
   - Protected endpoints requiring valid authentication tokens

7. Error Handling and Diagnostics:
   - Comprehensive exception catching and logging
   - Detailed debug output for troubleshooting (toggle-controlled)
   - Graceful degradation on database connection failures
   - Custom traceback JSON responses for client debugging
   - Memory usage logging and performance monitoring

8. Performance Optimization:
   - Bulk loading caches for multi-thread retrieval
   - Concurrent analysis limiting to prevent overload
   - Connection pooling for database efficiency
   - Single-pass checkpoint processing to minimize queries
   - Pagination support to reduce payload sizes

API Endpoints:
-------------
1. GET /chat-threads
   - Retrieve paginated list of user's chat threads
   - Query parameters: page (1-indexed), limit (1-50)
   - Returns: Thread IDs, timestamps, run counts, titles, prompts
   - Authentication: Required (JWT token)

2. GET /chat/all-messages-for-one-thread/{thread_id}
   - Get complete conversation history for single thread
   - Path parameter: thread_id (thread identifier)
   - Returns: Messages array, run IDs, sentiments dictionary
   - Authentication: Required with ownership verification

3. GET /chat/{thread_id}/sentiments
   - Retrieve sentiment values for all runs in a thread
   - Path parameter: thread_id (thread identifier)
   - Returns: Dictionary mapping run_id to sentiment value
   - Authentication: Required with ownership verification

4. DELETE /chat/{thread_id}
   - Delete all checkpoints and records for a thread
   - Path parameter: thread_id (thread to delete)
   - Returns: Deletion confirmation with record counts
   - Authentication: Required with ownership verification

Processing Flow:
--------------
1. Thread Listing Workflow:
   - User requests chat threads with pagination parameters
   - System validates authentication token and extracts user email
   - Database query retrieves thread metadata with LIMIT/OFFSET
   - Total count calculated for pagination controls
   - Response includes threads array and pagination metadata

2. Message Retrieval Workflow:
   - User requests messages for specific thread_id
   - System verifies thread ownership via database lookup
   - LangGraph checkpointer accessed to retrieve all checkpoints
   - Checkpoints sorted chronologically by step number
   - Message extraction from checkpoint metadata in single pass:
     * User prompts from writes.__start__.prompt
     * AI answers from writes.submit_final_answer.final_answer
     * Queries/results from writes.submit_final_answer.queries_and_results
     * Datasets from writes.submit_final_answer.top_selection_codes
     * PDF chunks from writes.submit_final_answer.top_chunks
     * Follow-up prompts from writes.submit_final_answer.followup_prompts
   - Run IDs matched to AI messages by chronological index
   - ChatMessage objects created with complete metadata
   - Response packaged with messages, run IDs, and sentiments

3. Sentiment Retrieval Workflow:
   - User requests sentiments for specific thread_id
   - System validates authentication and thread ownership
   - Database query retrieves all sentiment values for thread
   - Sentiments returned as dictionary: {run_id: sentiment_value}

4. Thread Deletion Workflow:
   - User requests deletion of specific thread_id
   - System verifies thread ownership before deletion
   - Database connection established for transactional safety
   - Deletion operations performed:
     * LangGraph checkpoints removed from checkpoints table
     * Thread entries deleted from users_threads_runs table
     * Cascade deletions handled automatically
   - Confirmation returned with deletion statistics
   - Memory cleanup performed after deletion

5. Checkpoint Processing Details:
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

6. Security Verification Process:
   - Extract user email from JWT token (via get_current_user dependency)
   - Query users_threads_runs table for matching email + thread_id
   - Count matching entries (should be > 0 for owned threads)
   - Deny access if no matching entries found
   - Proceed with data access only after successful verification

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

2. ChatThreadResponse (Response Model):
   - thread_id: Unique thread identifier
   - latest_timestamp: Most recent activity timestamp
   - run_count: Number of query runs in thread
   - title: Thread title (from first prompt)
   - full_prompt: Complete first user prompt

3. PaginatedChatThreadsResponse (Response Model):
   - threads: Array of ChatThreadResponse objects
   - total_count: Total threads for user across all pages
   - page: Current page number (1-indexed)
   - limit: Items per page
   - has_more: Boolean indicating additional pages exist

Configuration:
-------------
- BULK_CACHE_TIMEOUT: Cache expiration time for bulk operations (seconds)
- MAX_CONCURRENT_ANALYSES: Maximum parallel analysis operations
- Debug toggles: print__chat_*_debug functions for detailed logging
- Windows compatibility: WindowsSelectorEventLoopPolicy for psycopg

Database Schema Dependencies:
----------------------------
1. checkpoints table (LangGraph):
   - Stores conversation state snapshots
   - Contains metadata with agent outputs
   - Used for message reconstruction

2. users_threads_runs table:
   - Maps users to threads and runs
   - Stores prompts, timestamps, sentiments
   - Used for authentication and tracking

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
# Get user's chat threads (first page, 10 items)
GET /chat-threads?page=1&limit=10
Authorization: Bearer <jwt_token>

# Get complete conversation for thread
GET /chat/all-messages-for-one-thread/thread_abc123
Authorization: Bearer <jwt_token>

# Get sentiment ratings for thread
GET /chat/thread_abc123/sentiments
Authorization: Bearer <jwt_token>

# Delete a chat thread
DELETE /chat/thread_abc123
Authorization: Bearer <jwt_token>

Error Handling:
-------------
- HTTPException 401: Missing or invalid authentication token
- HTTPException 500: Database connection failures, checkpoint errors
- Graceful degradation: Returns empty results on non-critical failures
- Detailed logging: Debug output for troubleshooting (development mode)
- Custom traceback responses: JSON-formatted error details for debugging
- Connection retry handling: Automatic reconnection for transient failures

Performance Considerations:
--------------------------
- Pagination reduces payload sizes and database load
- Bulk caching minimizes redundant checkpoint retrievals
- Single-pass checkpoint processing reduces query overhead
- Connection pooling improves database efficiency
- Concurrent operation limiting prevents resource exhaustion
- Memory cleanup after deletions prevents memory leaks

Integration Points:
------------------
- LangGraph multi-agent system for conversation state
- PostgreSQL for persistent storage
- JWT authentication service for security
- Sentiment tracking service for user feedback
- PDF document retrieval for context provision
- CZSU database for statistical data queries

Debugging and Monitoring:
------------------------
- print__chat_threads_debug: Thread listing diagnostics
- print__chat_all_messages_one_thread_debug: Single thread diagnostics
- print__chat_all_messages_debug: General message processing diagnostics
- print__chat_sentiments_debug: Sentiment retrieval diagnostics
- print__delete_chat_debug: Deletion operation diagnostics
- print__sentiment_flow: Sentiment processing flow tracking
- log_memory_usage: Memory consumption monitoring
- Traceback preservation: Full error stack traces for debugging"""

# ==============================================================================
# ENVIRONMENT AND IMPORT INITIALIZATION
# ==============================================================================

# Load environment variables early to ensure configuration availability
import os

# CRITICAL: Set Windows event loop policy FIRST, before any other imports
# This must be the very first thing that happens to fix psycopg compatibility
# with Windows systems. The WindowsSelectorEventLoopPolicy resolves issues
# with the default ProactorEventLoop that can cause database connection problems.
import sys
import time

# Standard library imports
import traceback
from datetime import datetime
from typing import Dict, List

# Third-party imports - environment and web framework
from dotenv import load_dotenv
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse

# Platform-specific configuration for Windows compatibility
if sys.platform == "win32":
    import asyncio

    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

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

# ==============================================================================
# API DEPENDENCIES - AUTHENTICATION
# ==============================================================================

# JWT-based authentication dependency for protecting endpoints
from api.dependencies.auth import get_current_user

# ==============================================================================
# API MODELS - REQUEST AND RESPONSE SCHEMAS
# ==============================================================================

# Pydantic models for request validation and response serialization
from api.models.requests import FeedbackRequest, SentimentRequest
from api.models.responses import (
    ChatMessage,  # Individual message with metadata
    ChatThreadResponse,  # Thread summary information
    PaginatedChatThreadsResponse,  # Paginated thread list
)

# ==============================================================================
# UTILITY FUNCTIONS - DEBUGGING
# ==============================================================================

# Conditional debug output functions for development and troubleshooting
from api.utils.debug import (
    print__chat_all_messages_one_thread_debug,  # Single thread message diagnostics
    print__chat_sentiments_debug,  # Sentiment retrieval diagnostics
    print__chat_threads_debug,  # Thread listing diagnostics
    print__delete_chat_debug,  # Thread deletion diagnostics
    print__sentiment_flow,  # Sentiment flow tracking
)

# ==============================================================================
# UTILITY FUNCTIONS - MEMORY AND OPERATIONS
# ==============================================================================

# Memory monitoring and database operation utilities
from api.utils.memory import log_memory_usage, perform_deletion_operations

# ==============================================================================
# PROJECT PATH SETUP
# ==============================================================================

# Add base directory to Python path for module resolution
sys.path.insert(0, str(BASE_DIR))

# ==============================================================================
# CONFIGURATION SETTINGS
# ==============================================================================

# Global configuration for caching and concurrency control
from api.config.settings import (
    BULK_CACHE_TIMEOUT,  # Cache expiration time for bulk operations
    MAX_CONCURRENT_ANALYSES,  # Limit for parallel analysis tasks
    _bulk_loading_cache,  # Cache storage for bulk thread data
    _bulk_loading_locks,  # Locks for thread-safe cache access
)

# ==============================================================================
# API HELPERS
# ==============================================================================

# Custom error response formatting for client debugging
from api.helpers import traceback_json_response

# Additional debug functions
from api.utils.debug import print__chat_all_messages_debug

# ==============================================================================
# DATABASE OPERATIONS - CHECKPOINTER MODULE
# ==============================================================================

# Thread and sentiment management functions
from checkpointer.user_management.thread_operations import (
    get_user_chat_threads,  # Retrieve paginated thread list
    get_user_chat_threads_count,  # Get total thread count
)
from checkpointer.user_management.sentiment_tracking import (
    get_thread_run_sentiments,  # Retrieve sentiment values
)

# Database connection and LangGraph checkpointer
from checkpointer.database.connection import get_direct_connection
from checkpointer.checkpointer.factory import get_global_checkpointer

# ==============================================================================
# ENVIRONMENT CONFIGURATION
# ==============================================================================

# Load environment variables from .env file
load_dotenv()

# ==============================================================================
# FASTAPI ROUTER INITIALIZATION
# ==============================================================================

# Create router instance for chat-related endpoints
router = APIRouter()


async def get_thread_messages_with_metadata(
    checkpointer, thread_id: str, user_email: str, source_context: str = "general"
) -> List[ChatMessage]:
    """
    Extract and process all messages for a single thread with complete metadata.

    This function performs a single-pass extraction of conversation data from LangGraph
    checkpoints, efficiently consolidating all metadata for each interaction. It handles
    security verification, checkpoint retrieval, message extraction, and metadata association
    in one cohesive workflow.

    The function extracts:
    - User prompts from checkpoint writes.__start__.prompt
    - AI responses from checkpoint writes.submit_final_answer.final_answer
    - SQL queries and results from queries_and_results
    - CZSU dataset codes from top_selection_codes
    - PDF document chunks with source references from top_chunks
    - Follow-up question suggestions from followup_prompts

    Security:
        - Verifies thread ownership before accessing checkpoint data
        - Returns empty list if user doesn't own the thread
        - Database-level validation of email + thread_id combination

    Checkpoint Retrieval Strategy:
        1. Primary: checkpointer.alist(config, limit=200) for complete history
        2. Fallback: checkpointer.aget_tuple(config) for latest checkpoint only
        3. Empty list returned if both methods fail

    Args:
        checkpointer: LangGraph AsyncPostgresSaver instance for checkpoint access
        thread_id: Unique identifier for the conversation thread
        user_email: User's email address for ownership verification
        source_context: Processing context for debugging ("single_thread", "bulk_processing")

    Returns:
        List[ChatMessage]: Chronologically ordered messages with full metadata.
                          Empty list if thread doesn't exist, user lacks access,
                          or checkpoint retrieval fails.

    Raises:
        Exception: Caught and logged internally, returns empty list on errors

    Note:
        - Checkpoints are sorted by step number for chronological ordering
        - Each interaction produces one ChatMessage with both prompt and response
        - Metadata is isolated per interaction (not accumulated across thread)
        - PDF chunks include page_content, source_file, and page_number
    """

    print__chat_all_messages_debug(
        f"🔄 Processing thread {thread_id} for user {user_email}"
    )

    def get_channel_values(checkpoint_tuple):
        """Best-effort extraction of channel_values from LangGraph checkpoint tuples."""

        checkpoint_data = None
        if isinstance(checkpoint_tuple, dict):
            checkpoint_data = checkpoint_tuple.get("checkpoint")
        else:
            checkpoint_data = getattr(checkpoint_tuple, "checkpoint", None)

        if checkpoint_data is None:
            return {}

        # Handle dict-based checkpoints as well as objects with attributes
        if isinstance(checkpoint_data, dict):
            channel_dict = checkpoint_data.get("channel_values") or {}
        else:
            channel_dict = (
                getattr(checkpoint_data, "channel_values", None)
                or getattr(checkpoint_data, "values", None)
                or {}
            )

        if channel_dict is None:
            return {}

        if isinstance(channel_dict, dict):
            return channel_dict

        # Fallback: try to coerce other mapping types to dict
        try:
            return dict(channel_dict)
        except Exception:
            return {}

    def normalize_top_chunks(raw_chunks):
        """Convert LangGraph chunk objects into serializable dictionaries."""

        if not raw_chunks:
            return []

        chunks_processed = []
        for chunk in raw_chunks:
            chunk_data = {}

            if hasattr(chunk, "page_content"):
                chunk_data["page_content"] = chunk.page_content
            elif isinstance(chunk, dict):
                chunk_data["page_content"] = chunk.get("page_content", "")
            else:
                chunk_data["page_content"] = str(chunk)

            metadata = None
            if hasattr(chunk, "metadata"):
                metadata = chunk.metadata
            elif isinstance(chunk, dict):
                metadata = chunk.get("metadata")

            if metadata:
                chunk_data["metadata"] = metadata
                if isinstance(metadata, dict):
                    if "source_file" in metadata:
                        chunk_data["source_file"] = metadata["source_file"]
                    if "page_number" in metadata:
                        chunk_data["page_number"] = metadata["page_number"]

            chunks_processed.append(chunk_data)

        return chunks_processed

    try:
        # =======================================================================
        # SECURITY VERIFICATION
        # =======================================================================

        # Verify thread ownership before accessing any checkpoint data
        # This prevents unauthorized access to other users' conversations
        if user_email:
            print__chat_all_messages_debug(
                f"🔍 SECURITY CHECK: Verifying thread ownership for user: {user_email}"
            )

            try:
                async with get_direct_connection() as conn:
                    async with conn.cursor() as cur:
                        await cur.execute(
                            """
                            SELECT COUNT(*) FROM users_threads_runs 
                            WHERE email = %s AND thread_id = %s
                        """,
                            (user_email, thread_id),
                        )
                        result = await cur.fetchone()
                        thread_entries_count = result[0] if result else 0

                    if thread_entries_count == 0:
                        print__chat_all_messages_debug(
                            f"🔍 SECURITY DENIED: User {user_email} does not own thread {thread_id} - access denied"
                        )
                        return []

                    print__chat_all_messages_debug(
                        f"🔍 SECURITY GRANTED: User {user_email} owns thread {thread_id} ({thread_entries_count} entries) - access granted"
                    )
            except Exception as e:
                print__chat_all_messages_debug(
                    f"🔍 SECURITY ERROR: Could not verify thread ownership: {e}"
                )
                return []

        # =======================================================================
        # CHECKPOINT CONFIGURATION
        # =======================================================================

        # Configure checkpointer with thread_id for data retrieval
        config = {"configurable": {"thread_id": thread_id}}

        # =======================================================================
        # CHECKPOINT RETRIEVAL WITH FALLBACK STRATEGY
        # =======================================================================

        # Retrieve checkpoint tuples using primary method with fallback
        # Primary: alist() retrieves complete conversation history
        # Fallback: aget_tuple() retrieves only latest checkpoint if alist() fails
        checkpoint_tuples = []
        try:
            # PRIMARY METHOD: Use official alist() for complete checkpoint history
            print__chat_all_messages_debug(
                "🔍 ALIST METHOD: Using official AsyncPostgresSaver.alist() method"
            )

            # Retrieve all checkpoints (up to 200) to capture complete conversation
            # Each checkpoint represents one step in the agent execution
            async for checkpoint_tuple in checkpointer.alist(config, limit=200):
                checkpoint_tuples.append(checkpoint_tuple)

        except Exception as alist_error:
            # FALLBACK METHOD: Try aget_tuple() if alist() fails
            print__chat_all_messages_debug(
                f"🔍 ALIST ERROR: Error using alist(): {alist_error}"
            )

            # Attempt fallback only if primary method produced no results
            if not checkpoint_tuples:
                print__chat_all_messages_debug(
                    "🔍 FALLBACK METHOD: Trying fallback method using aget_tuple()"
                )
                try:
                    # Get only the latest checkpoint (limited conversation context)
                    state_snapshot = await checkpointer.aget_tuple(config)
                    if state_snapshot:
                        checkpoint_tuples = [state_snapshot]
                        print__chat_all_messages_debug(
                            "🔍 FALLBACK SUCCESS: Using fallback method - got latest checkpoint only"
                        )
                except Exception as fallback_error:
                    # Both methods failed - return empty list
                    print__chat_all_messages_debug(
                        f"🔍 FALLBACK ERROR: Fallback method also failed: {fallback_error}"
                    )
                    return []  # No checkpoint data available

        # =======================================================================
        # CHECKPOINT VALIDATION
        # =======================================================================

        # Validate that checkpoints were successfully retrieved
        if not checkpoint_tuples:
            print__chat_all_messages_debug(
                f"🔍 NO CHECKPOINTS: No checkpoints found for thread: {thread_id}"
            )
            return []  # No conversation data to process

        print__chat_all_messages_debug(
            f"🔍 CHECKPOINTS FOUND: Found {len(checkpoint_tuples)} checkpoints for verified thread"
        )

        # =======================================================================
        # CHECKPOINT SORTING
        # =======================================================================

        # Sort checkpoints chronologically by step number
        # Each step represents one agent execution in the conversation
        checkpoint_tuples.sort(
            key=lambda x: x.metadata.get("step", 0) if x.metadata else 0
        )

        # =======================================================================
        # INTERACTION EXTRACTION - SINGLE PASS PROCESSING
        # =======================================================================

        # Extract all user-AI interactions in one pass through checkpoints
        # Each interaction contains: prompt, response, queries, datasets, chunks
        interactions = []
        active_interaction: Dict[str, object] | None = None

        def start_new_interaction(initial_step: int) -> Dict[str, object]:
            interaction = {"step": initial_step}
            interactions.append(interaction)
            return interaction

        print__chat_all_messages_debug(
            f"🔍 INTERACTION EXTRACTION: Extracting complete interactions from {len(checkpoint_tuples)} checkpoints"
        )

        # Process each checkpoint to extract interaction data
        for checkpoint_index, checkpoint_tuple in enumerate(checkpoint_tuples):
            # Extract metadata from checkpoint tuple
            metadata = checkpoint_tuple.metadata or {}
            channel_values = get_channel_values(checkpoint_tuple)
            step = metadata.get("step", 0)  # Chronological step number
            writes = metadata.get("writes", {})  # Agent outputs

            data_updates: Dict[str, object] = {}
            prompt_from_writes = False

            # ===================================================================
            # EXTRACT USER PROMPT
            # ===================================================================

            # User prompts are stored in writes.__start__.prompt
            # This is the user's natural language query
            if isinstance(writes, dict) and "__start__" in writes:
                start_data = writes["__start__"]
                if isinstance(start_data, dict) and "prompt" in start_data:
                    prompt = start_data["prompt"]
                    if prompt and prompt.strip():
                        data_updates["prompt"] = prompt.strip()
                        prompt_from_writes = True
                        print__chat_all_messages_debug(
                            f"🔍 USER PROMPT FOUND: Step {step}: {prompt[:50]}..."
                        )

            # Fallback to channel_values when writes lacks the prompt
            if "prompt" not in data_updates:
                channel_prompt = channel_values.get("prompt")
                if isinstance(channel_prompt, str) and channel_prompt.strip():
                    data_updates["prompt"] = channel_prompt.strip()
                    print__chat_all_messages_debug(
                        f"🔍 USER PROMPT FOUND (channel_values): Step {step}: {channel_prompt[:50]}..."
                    )

            # ===================================================================
            # EXTRACT AI RESPONSE AND ALL METADATA
            # ===================================================================

            # AI responses and metadata are in writes.submit_final_answer
            # This contains the complete agent output for this interaction
            if isinstance(writes, dict) and "submit_final_answer" in writes:
                submit_data = writes["submit_final_answer"]
                if isinstance(submit_data, dict):
                    # ---------------------------------------------------------------
                    # Extract final answer (AI's natural language response)
                    # ---------------------------------------------------------------
                    if "final_answer" in submit_data:
                        final_answer = submit_data["final_answer"]
                        if final_answer and final_answer.strip():
                            data_updates["final_answer"] = final_answer.strip()
                            print__chat_all_messages_debug(
                                f"🔍 AI ANSWER FOUND: Step {step}: {final_answer[:50]}..."
                            )

                    # ---------------------------------------------------------------
                    # Extract follow-up prompts (suggested next questions)
                    # ---------------------------------------------------------------
                    # These are AI-generated suggestions for continuing the conversation
                    if "followup_prompts" in submit_data:
                        followup_prompts = submit_data["followup_prompts"]
                        if followup_prompts:
                            data_updates["followup_prompts"] = followup_prompts
                            print__chat_all_messages_debug(
                                f"🔍 FOLLOWUP PROMPTS FOUND: Step {step}: Found {len(followup_prompts)} follow-up prompts"
                            )

                    # ---------------------------------------------------------------
                    # Extract queries and results (SQL execution data)
                    # ---------------------------------------------------------------
                    # Contains SQL queries executed and their result sets
                    if "queries_and_results" in submit_data:
                        queries_and_results = submit_data["queries_and_results"]
                        if queries_and_results:
                            data_updates["queries_and_results"] = queries_and_results
                            print__chat_all_messages_debug(
                                f"🔍 QUERIES FOUND: Step {step}: Found queries and results for this interaction"
                            )

                    # ---------------------------------------------------------------
                    # Extract datasets used (CZSU selection codes)
                    # ---------------------------------------------------------------
                    # Contains codes for CZSU statistical datasets accessed
                    if "top_selection_codes" in submit_data:
                        top_selection_codes = submit_data["top_selection_codes"]
                        if top_selection_codes:
                            data_updates["datasets_used"] = top_selection_codes
                            print__chat_all_messages_debug(
                                f"🔍 DATASETS FOUND: Step {step}: Found {len(top_selection_codes)} datasets for this interaction"
                            )

                    # ---------------------------------------------------------------
                    # Extract PDF document chunks (context sources)
                    # ---------------------------------------------------------------
                    # Contains chunks from PDF documents used for answering the query
                    # Each chunk includes: page_content, source_file, page_number
                    if "top_chunks" in submit_data:
                        processed_chunks = normalize_top_chunks(
                            submit_data.get("top_chunks")
                        )
                        if processed_chunks:
                            data_updates["top_chunks"] = processed_chunks
                            print__chat_all_messages_debug(
                                f"🔍 CHUNKS FOUND: Step {step}: Found {len(processed_chunks)} chunks for this interaction"
                            )

                    # ---------------------------------------------------------------
                    # Extension point for additional metadata
                    # ---------------------------------------------------------------

                    # Future metadata fields can be extracted here
                    # (e.g., timing data, confidence scores, etc.)

            # ===================================================================
            # FALLBACKS USING CHANNEL VALUES
            # ===================================================================

            if channel_values:
                if "final_answer" not in data_updates:
                    channel_final_answer = channel_values.get("final_answer")
                    if (
                        isinstance(channel_final_answer, str)
                        and channel_final_answer.strip()
                    ):
                        data_updates["final_answer"] = channel_final_answer.strip()
                        print__chat_all_messages_debug(
                            f"🔍 AI ANSWER FOUND (channel_values): Step {step}: {channel_final_answer[:50]}..."
                        )

                if "followup_prompts" not in data_updates:
                    followups = channel_values.get("followup_prompts")
                    if followups:
                        data_updates["followup_prompts"] = followups
                        print__chat_all_messages_debug(
                            f"🔍 FOLLOWUP PROMPTS FOUND (channel_values): Step {step}: Found {len(followups)} follow-up prompts"
                        )

                if "queries_and_results" not in data_updates:
                    queries = channel_values.get("queries_and_results")
                    if queries:
                        data_updates["queries_and_results"] = queries
                        print__chat_all_messages_debug(
                            f"🔍 QUERIES FOUND (channel_values): Step {step}: Found queries and results for this interaction"
                        )

                if "datasets_used" not in data_updates:
                    datasets = channel_values.get("top_selection_codes")
                    if datasets:
                        data_updates["datasets_used"] = datasets
                        print__chat_all_messages_debug(
                            f"🔍 DATASETS FOUND (channel_values): Step {step}: Found {len(datasets)} datasets for this interaction"
                        )

                if "top_chunks" not in data_updates:
                    processed_chunks = normalize_top_chunks(
                        channel_values.get("top_chunks")
                    )
                    if processed_chunks:
                        data_updates["top_chunks"] = processed_chunks
                        print__chat_all_messages_debug(
                            f"🔍 CHUNKS FOUND (channel_values): Step {step}: Found {len(processed_chunks)} chunks for this interaction"
                        )

            if not data_updates:
                continue

            prompt_in_update = "prompt" in data_updates
            target_interaction = active_interaction

            if prompt_in_update:
                stripped_prompt = data_updates["prompt"].strip()
                data_updates["prompt"] = stripped_prompt

                is_same_prompt = (
                    target_interaction is not None
                    and target_interaction.get("prompt") == stripped_prompt
                )

                should_start_new = False
                if target_interaction is None:
                    should_start_new = True
                elif prompt_from_writes:
                    should_start_new = True
                elif not is_same_prompt:
                    should_start_new = True

                if should_start_new:
                    target_interaction = start_new_interaction(step)
                    active_interaction = target_interaction
                    print__chat_all_messages_debug(
                        f"🔄 STARTED NEW INTERACTION: Step {step}"
                    )
                else:
                    active_interaction = target_interaction

                if target_interaction and not target_interaction.get("prompt"):
                    target_interaction["prompt"] = stripped_prompt
            else:
                if target_interaction is None:
                    target_interaction = start_new_interaction(step)
                active_interaction = target_interaction

            target_interaction["step"] = min(target_interaction.get("step", step), step)

            for key, value in data_updates.items():
                if key == "prompt":
                    continue
                target_interaction[key] = value

        # =======================================================================
        # INTERACTION SORTING
        # =======================================================================

        # Sort interactions chronologically by step number
        # Ensures messages appear in conversation order
        interactions.sort(key=lambda x: x["step"])

        print__chat_all_messages_debug(
            f"🔍 INTERACTION SUCCESS: Created {len(interactions)} complete interactions"
        )

        # Validate that we extracted at least one interaction
        if not interactions:
            print__chat_all_messages_debug(
                f"⚠ No interactions found for thread {thread_id}"
            )
            return []  # No conversation content to return

        # =======================================================================
        # CHATMESSAGE OBJECT CREATION
        # =======================================================================

        # Convert raw interactions to structured ChatMessage objects
        chat_messages = []
        message_counter = 0  # Counter for generating unique message IDs

        print__chat_all_messages_debug(
            f"🔍 Converting {len(interactions)} interactions to ChatMessage objects"
        )

        # Process each interaction into a ChatMessage
        for i, interaction in enumerate(interactions):
            print__chat_all_messages_debug(
                f"🔍 Processing interaction {i+1}/{len(interactions)}: Step {interaction['step']}"
            )

            # Generate unique message ID and timestamp
            message_counter += 1

            # Create ChatMessage with all extracted data
            # Each message contains both user prompt and AI response for the interaction
            chat_message = ChatMessage(
                id=f"msg_{message_counter}",  # Unique message identifier
                threadId=thread_id,  # Parent thread reference
                user=user_email,  # User who owns this message
                createdAt=int(  # Timestamp in milliseconds
                    datetime.fromtimestamp(
                        1700000000 + message_counter * 1000  # Synthetic timestamp
                    ).timestamp()
                    * 1000
                ),
                prompt=interaction.get("prompt"),  # User's query
                final_answer=interaction.get("final_answer"),  # AI's response
                queries_and_results=interaction.get("queries_and_results"),  # SQL data
                datasets_used=interaction.get("datasets_used"),  # CZSU datasets
                top_chunks=interaction.get("top_chunks"),  # PDF context
                sql_query=None,  # Populated below from queries_and_results
                error=None,  # Error message if execution failed
                isLoading=False,  # Loading state (always False for historical data)
                startedAt=None,  # Execution start time
                isError=False,  # Error state indicator
                followup_prompts=interaction.get(
                    "followup_prompts"
                ),  # Suggested next questions
            )

            # ===================================================================
            # EXTRACT PRIMARY SQL QUERY
            # ===================================================================

            # Extract the main SQL query from queries_and_results for easy access
            # Format: queries_and_results = [[sql_query, result_set], ...]
            if (
                chat_message.queries_and_results
                and len(chat_message.queries_and_results) > 0
            ):
                try:
                    # Get first query (primary SQL execution)
                    chat_message.sql_query = (
                        chat_message.queries_and_results[0][0]  # First query string
                        if chat_message.queries_and_results[0]
                        else None
                    )
                except (IndexError, TypeError):
                    # Handle malformed queries_and_results structure
                    chat_message.sql_query = None

            # Add message to output list
            chat_messages.append(chat_message)
            print__chat_all_messages_debug(
                f"🔍 ADDED MESSAGE: Step {interaction['step']}: prompt={interaction.get('prompt', '')[:50]} final_answer={interaction.get('final_answer', '')[:50]}..."
            )

        # =======================================================================
        # COMPLETION AND RETURN
        # =======================================================================
        print__chat_all_messages_debug(
            f"✅ Processed {len(chat_messages)} messages for thread {thread_id}"
        )
        return chat_messages  # Return chronologically ordered messages

    except Exception as e:
        # =======================================================================
        # ERROR HANDLING
        # =======================================================================

        # Catch all exceptions and return empty list for graceful degradation
        print__chat_all_messages_debug(f"❌ Error processing thread {thread_id}: {e}")
        print__chat_all_messages_debug(
            f"🔍 Thread processing error type: {type(e).__name__}"
        )
        print__chat_all_messages_debug(
            f"🔍 Thread processing error traceback: {traceback.format_exc()}"
        )
        return []  # Return empty list on any error


# ==============================================================================
# API ENDPOINT: GET THREAD SENTIMENTS
# ==============================================================================


@router.get("/chat/{thread_id}/sentiments")
async def get_thread_sentiments(thread_id: str, user=Depends(get_current_user)):
    """
    Retrieve sentiment values for all messages in a specific chat thread.

    This endpoint returns user sentiment ratings (satisfaction scores) for each
    interaction in the conversation. Sentiments are stored per run_id and range
    from positive (satisfied) to negative (dissatisfied).

    Authentication:
        Requires valid JWT token via get_current_user dependency

    Args:
        thread_id: Unique identifier for the chat thread
        user: Authenticated user object (injected by dependency)

    Returns:
        Dict[str, float]: Dictionary mapping run_id to sentiment value

    Raises:
        HTTPException 401: User email not found in authentication token
        HTTPException 500: Database error or sentiment retrieval failure

    Example Response:
        {
            "run_abc123": 0.8,
            "run_def456": -0.2,
            "run_ghi789": 0.5
        }
    """

    print__chat_sentiments_debug("🔍 CHAT_SENTIMENTS ENDPOINT - ENTRY POINT")
    print__chat_sentiments_debug(f"🔍 Request received: thread_id={thread_id}")

    # Extract user email from authentication token
    user_email = user.get("email")
    print__chat_sentiments_debug(f"🔍 User email extracted: {user_email}")

    # Validate user email presence
    if not user_email:
        print__chat_sentiments_debug("🚨 No user email found in token")
        raise HTTPException(status_code=401, detail="User email not found in token")

    try:
        print__chat_sentiments_debug(
            f"🔍 Getting sentiments for thread {thread_id}, user: {user_email}"
        )
        print__sentiment_flow(
            f"📥 Getting sentiments for thread {thread_id}, user: {user_email}"
        )

        # Retrieve sentiments from database
        sentiments = await get_thread_run_sentiments(user_email, thread_id)
        print__chat_sentiments_debug(f"🔍 Retrieved {len(sentiments)} sentiment values")

        print__sentiment_flow(f"✅ Retrieved {len(sentiments)} sentiment values")
        print__chat_sentiments_debug("🔍 CHAT_SENTIMENTS ENDPOINT - SUCCESSFUL EXIT")
        return sentiments

    except Exception as e:
        # Log detailed error information for debugging
        print__chat_sentiments_debug(
            f"🚨 Exception in chat sentiments processing: {type(e).__name__}: {str(e)}"
        )
        print__chat_sentiments_debug(
            f"🚨 Chat sentiments processing traceback: {traceback.format_exc()}"
        )
        print__sentiment_flow(
            f"❌ Failed to get sentiments for thread {thread_id}: {e}"
        )

        # Try custom error response first
        resp = traceback_json_response(e)
        if resp:
            return resp

        # Fallback to standard HTTP exception
        raise HTTPException(
            status_code=500, detail=f"Failed to get sentiments: {e}"
        ) from e


# ==============================================================================
# API ENDPOINT: GET CHAT THREADS
# ==============================================================================


@router.get(
    "/chat-threads",
    summary="Get user's chat threads",
    description="""
    **Retrieve a paginated list of chat threads for the authenticated user.**
    
    Threads are ordered by most recent activity first. Each thread includes:
    - Thread ID for referencing in other endpoints
    - Latest timestamp of activity
    - Number of query runs in the thread
    - Title derived from the first query
    - Full first prompt for context
    
    **Pagination:** Use `page` and `limit` parameters to control results.
    """,
    response_model=PaginatedChatThreadsResponse,
    response_description="Paginated list of chat threads with metadata",
    responses={200: {"description": "Successfully retrieved chat threads"}},
)
async def get_chat_threads(
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    limit: int = Query(10, ge=1, le=50, description="Number of threads per page"),
    user=Depends(get_current_user),
) -> PaginatedChatThreadsResponse:
    """Get paginated chat threads for the authenticated user."""

    print__chat_threads_debug("🔍 CHAT_THREADS ENDPOINT - ENTRY POINT")
    print__chat_threads_debug(f"🔍 Request parameters: page={page}, limit={limit}")

    try:
        user_email = user["email"]
        print__chat_threads_debug(f"🔍 User email extracted: {user_email}")
        print__chat_threads_debug(
            f"Loading chat threads for user: {user_email} (page: {page}, limit: {limit})"
        )

        print__chat_threads_debug("🔍 Starting simplified approach")
        print__chat_threads_debug("Getting chat threads with simplified approach")

        # Get total count first
        print__chat_threads_debug("🔍 Getting total threads count")
        print__chat_threads_debug(f"Getting chat threads count for user: {user_email}")
        total_count = await get_user_chat_threads_count(user_email)
        print__chat_threads_debug(f"🔍 Total count retrieved: {total_count}")
        print__chat_threads_debug(
            f"Total threads count for user {user_email}: {total_count}"
        )

        # Calculate offset for pagination
        offset = (page - 1) * limit
        print__chat_threads_debug(f"🔍 Calculated offset: {offset}")

        # Get threads for this page
        print__chat_threads_debug(
            f"🔍 Getting chat threads for user: {user_email} (limit: {limit}, offset: {offset})"
        )
        print__chat_threads_debug(
            f"Getting chat threads for user: {user_email} (limit: {limit}, offset: {offset})"
        )
        threads = await get_user_chat_threads(user_email, limit=limit, offset=offset)
        print__chat_threads_debug(f"🔍 Retrieved threads: {threads}")
        if threads is None:
            print__chat_threads_debug(
                "get_user_chat_threads returned None! Setting to empty list."
            )
            threads = []
        print__chat_threads_debug(f"🔍 Retrieved {len(threads)} threads from database")
        print__chat_threads_debug(
            f"Retrieved {len(threads)} threads for user {user_email}"
        )

        # Try/except around the for-loop to catch and print any errors
        try:
            chat_thread_responses = []
            for thread in threads:
                print__chat_threads_debug(f"Processing thread dict: {thread}")
                chat_thread_response = ChatThreadResponse(
                    thread_id=thread["thread_id"],
                    latest_timestamp=thread["latest_timestamp"],
                    run_count=thread["run_count"],
                    title=thread["title"],
                    full_prompt=thread["full_prompt"],
                )
                chat_thread_responses.append(chat_thread_response)
        except Exception as e:
            print__chat_threads_debug(f"Exception in /chat-threads for-loop: {e}")
            print__chat_threads_debug(f"Traceback: {traceback.format_exc()}")
            # Return empty result on error
            return PaginatedChatThreadsResponse(
                threads=[], total_count=0, page=page, limit=limit, has_more=False
            )

        # Convert to response format
        print__chat_threads_debug("🔍 Converting threads to response format")
        chat_thread_responses = []
        for thread in threads:
            chat_thread_response = ChatThreadResponse(
                thread_id=thread["thread_id"],
                latest_timestamp=thread["latest_timestamp"],
                run_count=thread["run_count"],
                title=thread["title"],
                full_prompt=thread["full_prompt"],
            )
            chat_thread_responses.append(chat_thread_response)

        # Calculate pagination info
        has_more = (offset + len(chat_thread_responses)) < total_count
        print__chat_threads_debug(f"🔍 Pagination calculated: has_more={has_more}")

        print__chat_threads_debug(
            f"Retrieved {len(threads)} threads for user {user_email} (total: {total_count})"
        )
        print__chat_threads_debug(
            f"Returning {len(chat_thread_responses)} threads to frontend (page {page}/{(total_count + limit - 1) // limit})"
        )

        result = PaginatedChatThreadsResponse(
            threads=chat_thread_responses,
            total_count=total_count,
            page=page,
            limit=limit,
            has_more=has_more,
        )
        print__chat_threads_debug("🔍 CHAT_THREADS ENDPOINT - SUCCESSFUL EXIT")
        return result

    except Exception as e:
        print__chat_threads_debug(
            f"🚨 Exception in chat threads processing: {type(e).__name__}: {str(e)}"
        )
        print__chat_threads_debug(
            f"🚨 Chat threads processing traceback: {traceback.format_exc()}"
        )
        print__chat_threads_debug(f"❌ Error getting chat threads: {e}")
        print__chat_threads_debug(f"Full traceback: {traceback.format_exc()}")
        resp = traceback_json_response(e)
        if resp:
            return resp
        # Return error response
        result = PaginatedChatThreadsResponse(
            threads=[], total_count=0, page=page, limit=limit, has_more=False
        )
        print__chat_threads_debug("🔍 CHAT_THREADS ENDPOINT - ERROR EXIT")
        return result


# ==============================================================================
# API ENDPOINT: DELETE CHAT THREAD
# ==============================================================================


@router.delete("/chat/{thread_id}")
async def delete_chat_checkpoints(thread_id: str, user=Depends(get_current_user)):
    """
    Delete all PostgreSQL checkpoint records and thread entries for a specific thread.

    This endpoint permanently removes all data associated with a chat thread, including:
    - LangGraph checkpoints (conversation state snapshots)
    - Thread entries in users_threads_runs table
    - Associated sentiment data (via cascade delete)

    The operation is transactional - either all deletions succeed or none do.

    Security:
        - Requires valid JWT token via get_current_user dependency
        - Only the thread owner can delete their threads
        - Ownership verified before deletion

    Args:
        thread_id: Unique identifier for the thread to delete
        user: Authenticated user object (injected by dependency)

    Returns:
        Dict: Deletion confirmation with statistics
            - message: Success message
            - thread_id: Deleted thread identifier
            - user_email: User who performed deletion
            - deleted_counts: Number of records deleted per table

    Raises:
        HTTPException 401: User email not found in authentication token
        HTTPException 500: Database error during deletion

    Note:
        - Operation is irreversible - deleted data cannot be recovered
        - Performs memory cleanup after deletion
        - Handles database connection failures gracefully

    Example Response:
        {
            \"message\": \"Successfully deleted thread\",
            \"thread_id\": \"thread_abc123\",
            \"user_email\": \"user@example.com\",
            \"deleted_counts\": {
                \"checkpoints\": 15,
                \"thread_entries\": 5
            }
        }
    """

    print__delete_chat_debug("🔍 DELETE_CHAT ENDPOINT - ENTRY POINT")
    print__delete_chat_debug(f"🔍 Request received: thread_id={thread_id}")

    # Extract user email from authentication token
    user_email = user.get("email")
    print__delete_chat_debug(f"🔍 User email extracted: {user_email}")

    # Validate user email presence
    if not user_email:
        print__delete_chat_debug("🚨 No user email found in token")
        raise HTTPException(status_code=401, detail="User email not found in token")

    print__delete_chat_debug(
        f"🗑️ Deleting chat thread {thread_id} for user {user_email}"
    )

    try:
        # Perform deletion operations with transactional database connection
        # Uses same connection method as get_user_chat_threads for consistency
        async with get_direct_connection() as conn:
            # Delegate to perform_deletion_operations for actual deletion logic
            # This function handles ownership verification and cascading deletes
            result_data = await perform_deletion_operations(conn, user_email, thread_id)
            return result_data

    except Exception as e:
        error_msg = str(e)
        print__delete_chat_debug(
            f"❌ Failed to delete checkpoint records for thread {thread_id}: {e}"
        )
        print__delete_chat_debug(f"🔧 DEBUG: Main exception type: {type(e).__name__}")
        print__delete_chat_debug(
            f"🔧 DEBUG: Main exception traceback: {traceback.format_exc()}"
        )

        # Check if error is database connection-related
        # Gracefully handle connection issues without raising 500 error
        if any(
            keyword in error_msg.lower()
            for keyword in [
                "ssl error",
                "connection",
                "timeout",
                "operational error",
                "server closed",
                "bad connection",
                "consuming input failed",
            ]
        ):
            # Database unavailable - return warning instead of error
            print__delete_chat_debug(
                "⚠️ PostgreSQL connection unavailable - no records to delete"
            )
            return {
                "message": "PostgreSQL connection unavailable - no records to delete",
                "thread_id": thread_id,
                "user_email": user_email,
                "warning": "Database connection issues",
            }
        else:
            # Non-connection error - try custom error response
            resp = traceback_json_response(e)
            if resp:
                return resp

            # Fallback to standard HTTP exception
            raise HTTPException(
                status_code=500, detail=f"Failed to delete checkpoint records: {e}"
            ) from e


# ==============================================================================
# API ENDPOINT: GET ALL MESSAGES FOR SINGLE THREAD
# ==============================================================================


@router.get("/chat/all-messages-for-one-thread/{thread_id}")
async def get_all_chat_messages_for_one_thread(
    thread_id: str, user=Depends(get_current_user)
) -> Dict:
    """
    Retrieve complete conversation history for a specific chat thread with all metadata.

    This endpoint returns the complete conversation for a single thread, including:
    - All user prompts and AI responses in chronological order
    - SQL queries executed and their results
    - CZSU datasets referenced
    - PDF document chunks used for context
    - Follow-up prompt suggestions
    - Run IDs for sentiment tracking
    - Sentiment values for each interaction

    The response is packaged as a comprehensive dictionary containing messages,
    run IDs, and sentiments, suitable for rendering a complete chat interface.

    Authentication:
        - Requires valid JWT token via get_current_user dependency
        - Thread ownership verified via get_thread_messages_with_metadata

    Args:
        thread_id: Unique identifier for the chat thread
        user: Authenticated user object (injected by dependency)

    Returns:
        Dict containing:
            - messages: List[Dict] - Serialized ChatMessage objects with all metadata
            - runIds: List[Dict] - Run information (run_id, prompt, timestamp)
            - sentiments: Dict[str, float] - Mapping of run_id to sentiment value

    Raises:
        HTTPException 401: User email not found in authentication token
        HTTPException 500: Checkpoint retrieval or database error

    Note:
        - Messages include both user prompts and AI responses
        - Run IDs matched to messages by chronological index
        - Only AI messages (with final_answer) receive run_ids
        - Sentiment data optional - may be empty if no ratings provided

    Example Response:
        {
            \"messages\": [
                {
                    \"id\": \"msg_1\",
                    \"threadId\": \"thread_abc123\",
                    \"prompt\": \"Show me population data\",
                    \"final_answer\": \"Here is the population data...\",
                    \"sql_query\": \"SELECT * FROM population WHERE...\",
                    \"datasets_used\": [\"OBY01PDT01\"],
                    \"run_id\": \"run_xyz789\"
                }
            ],
            \"runIds\": [
                {
                    \"run_id\": \"run_xyz789\",
                    \"prompt\": \"Show me population data\",
                    \"timestamp\": \"2024-01-15T10:30:00\"
                }
            ],
            \"sentiments\": {
                \"run_xyz789\": 0.8
            }
        }
    """

    print__chat_all_messages_one_thread_debug(
        "🔍 CHAT_SINGLE_THREAD ENDPOINT - ENTRY POINT"
    )

    # Extract user email from authentication token
    user_email = user["email"]
    print__chat_all_messages_one_thread_debug(
        f"📥 SINGLE THREAD REQUEST: Loading chat messages for thread: {thread_id}, user: {user_email}"
    )

    try:
        # =======================================================================
        # CHECKPOINT ACCESS
        # =======================================================================

        # Get global checkpointer instance for conversation state access
        checkpointer = await get_global_checkpointer()

        # =======================================================================
        # MESSAGE EXTRACTION
        # =======================================================================

        # Extract all messages and per-interaction metadata from checkpoints
        # This performs security verification and returns chronologically ordered messages
        chat_messages = await get_thread_messages_with_metadata(
            checkpointer, thread_id, user_email, "single_thread_processing"
        )

        # =======================================================================
        # RUN ID AND SENTIMENT RETRIEVAL
        # =======================================================================

        # Query database for run metadata and sentiment values
        # Run IDs link messages to sentiment ratings and execution tracking
        thread_run_ids = []
        thread_sentiments = {}

        async with get_direct_connection() as conn:
            async with conn.cursor() as cur:
                # Retrieve all runs for this thread in chronological order
                await cur.execute(
                    """
                    SELECT run_id, prompt, timestamp, sentiment
                    FROM users_threads_runs 
                    WHERE email = %s AND thread_id = %s
                    ORDER BY timestamp ASC
                """,
                    (user_email, thread_id),
                )
                rows = await cur.fetchall()

            # Process query results into structured data
            for row in rows:
                run_id, prompt, timestamp, sentiment = row

                # Build run ID metadata list
                thread_run_ids.append(
                    {
                        "run_id": run_id,  # Unique run identifier
                        "prompt": prompt,  # User's query for this run
                        "timestamp": timestamp.isoformat(),  # ISO-formatted timestamp
                    }
                )

                # Build sentiment mapping (only if sentiment was provided)
                if sentiment is not None:
                    thread_sentiments[run_id] = sentiment

        # =======================================================================
        # RUN ID MATCHING TO MESSAGES
        # =======================================================================

        # Match run_ids to AI messages by chronological index
        # Only messages with final_answer (AI responses) receive run_ids
        print__chat_all_messages_one_thread_debug(
            f"🔍 MATCHING RUN_IDS: Found {len(thread_run_ids)} run_ids and {len(chat_messages)} messages"
        )

        ai_message_index = 0  # Track index within AI messages only

        for idx, msg in enumerate(chat_messages):
            print__chat_all_messages_one_thread_debug(
                f"🔍 Message {idx}: has_prompt={bool(msg.prompt)}, has_final_answer={bool(msg.final_answer)}"
            )

            # Only AI messages (with final_answer) get run_ids
            if msg.final_answer and ai_message_index < len(thread_run_ids):
                # Match this AI message to the corresponding run_id
                msg.run_id = thread_run_ids[ai_message_index]["run_id"]
                print__chat_all_messages_one_thread_debug(
                    f"🔍 MATCHED: Message {idx} -> run_id {msg.run_id}"
                )
                ai_message_index += 1
            else:
                # Warn if AI message has no corresponding run_id
                if msg.final_answer:
                    print__chat_all_messages_one_thread_debug(
                        f"⚠️ WARNING: Message {idx} has final_answer but no run_id available (ai_message_index={ai_message_index}, run_ids_count={len(thread_run_ids)})"
                    )

        print__chat_all_messages_one_thread_debug(
            f"🔍 MATCHING COMPLETE: Matched {ai_message_index} messages to run_ids"
        )

        # =======================================================================
        # RESPONSE SERIALIZATION
        # =======================================================================

        # Convert ChatMessage objects to dictionaries for JSON response
        # exclude_none=True removes fields with None values for cleaner JSON
        chat_messages_serialized = [
            msg.model_dump(exclude_none=True) for msg in chat_messages
        ]

        # =======================================================================
        # RESPONSE PACKAGING
        # =======================================================================

        # Package all data into the structure expected by the frontend
        result = {
            "messages": chat_messages_serialized,  # Complete message history
            "runIds": thread_run_ids,  # Run metadata for tracking
            "sentiments": thread_sentiments,  # User satisfaction ratings
        }

        print__chat_all_messages_one_thread_debug(
            f"✅ SINGLE THREAD PROCESSING COMPLETE: Returning {len(chat_messages)} messages and metadata."
        )

        return result

    except Exception as e:
        # =======================================================================
        # ERROR HANDLING
        # =======================================================================

        # Log detailed error information for debugging
        print__chat_all_messages_one_thread_debug(
            f"❌ SINGLE THREAD ERROR: Failed to process request for thread {thread_id}: {e}"
        )
        print__chat_all_messages_one_thread_debug(
            f"Full error traceback: {traceback.format_exc()}"
        )

        # Try custom error response first
        resp = traceback_json_response(e)
        if resp:
            return resp

        # Fallback to standard HTTP exception
        raise HTTPException(
            status_code=500, detail="Failed to process chat thread"
        ) from e
