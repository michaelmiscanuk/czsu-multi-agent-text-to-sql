"""User Thread Management for PostgreSQL Checkpointer System

This module provides comprehensive functionality for managing conversation threads
in a multi-agent text-to-SQL system using PostgreSQL as the backend storage. It handles
the complete lifecycle of user conversation threads including creation, retrieval,
pagination, counting, and deletion operations.
"""

from __future__ import annotations

MODULE_DESCRIPTION = r"""User Thread Management for PostgreSQL Checkpointer System

This module provides comprehensive functionality for managing conversation threads
in a multi-agent text-to-SQL system using PostgreSQL as the backend storage. It handles
the complete lifecycle of user conversation threads including creation, retrieval,
pagination, counting, and deletion operations.

Key Features:
-------------
1. Thread Run Entry Management:
   - Creation of new thread run entries with unique run IDs
   - The run_id is passed to LangGraph and used by LangSmith for tracing
   - Automatic generation of UUID-based identifiers
   - Upsert operations to handle duplicate IDs gracefully
   - Timestamp tracking for each thread run entry
   - Association of prompts with thread runs
   - Resilient fallback to return run IDs even on database failures

2. Thread Retrieval and Listing:
   - Retrieval of all conversation threads for a specific user
   - Grouping of thread runs by thread ID
   - Automatic sorting by latest activity (most recent first)
   - Metadata extraction including run count and timestamps
   - Thread title generation from first prompt with smart truncation
   - Support for pagination with configurable limit and offset
   - Graceful error handling with empty list returns on failures

3. Thread Statistics:
   - Counting of total threads per user
   - Efficient distinct thread ID counting
   - Support for pagination metadata
   - Error-resilient counting with zero fallback

4. Thread Deletion:
   - Complete deletion of all entries for a specific thread
   - Pre-deletion counting for verification
   - Detailed deletion reporting with affected row counts
   - User and thread ID validation
   - Comprehensive deletion status responses
   - Transaction-safe deletion operations

5. Database Operations:
   - Direct PostgreSQL connection management via connection pool
   - Asynchronous database operations using psycopg (async)
   - Prepared statement error handling with automatic retries
   - Cursor-based transaction safety
   - Efficient batch operations for improved performance
   - Connection lifecycle management with context managers

6. Error Handling and Resilience:
   - Retry mechanism for prepared statement errors
   - Configurable maximum retry attempts via DEFAULT_MAX_RETRIES
   - Detailed debug logging for all operations
   - Graceful degradation to prevent API crashes
   - Exception tracking with full traceback logging
   - Non-raising error handling for read operations

7. Title Generation:
   - Intelligent thread title creation from first prompt
   - Configurable maximum title length (THREAD_TITLE_MAX_LENGTH)
   - Smart truncation with ellipsis for long prompts
   - Fallback to "Untitled Conversation" for empty prompts
   - Preservation of full prompt text in separate field

Database Schema:
---------------
The module interacts with the 'users_threads_runs' table with the following structure:

Table: users_threads_runs
  Columns:
    - email (VARCHAR): User email address (foreign key)
    - thread_id (VARCHAR): Conversation thread identifier
    - run_id (VARCHAR): Unique run identifier (primary key)
    - prompt (TEXT): User prompt/query text
    - timestamp (TIMESTAMP): Entry creation timestamp (auto-updated)
  
  Constraints:
    - PRIMARY KEY: run_id
    - UNIQUE: run_id
    - ON CONFLICT: Upsert behavior updates email, thread_id, prompt, and timestamp

Processing Flow:
--------------
1. Thread Run Creation:
   - Validate input parameters (email, thread_id)
   - Generate new run_id if not provided
   - Attempt database insertion with upsert logic
   - Log operation with detailed debug messages
   - Return run_id on success or generate fallback on failure
   - Handle prepared statement errors with retry mechanism

2. Thread Retrieval:
   - Accept pagination parameters (limit, offset)
   - Query database for user's threads
   - Group runs by thread_id
   - Extract first prompt for title generation
   - Calculate run count and latest timestamp
   - Format thread data with metadata
   - Return structured list of thread objects
   - Handle errors gracefully with empty list return

3. Thread Counting:
   - Query for distinct thread IDs per user
   - Return total count for pagination metadata
   - Fallback to zero on database errors
   - Prevent API crashes with error-resilient design

4. Thread Deletion:
   - Validate user email and thread_id
   - Count entries to be deleted for verification
   - Execute deletion query with user and thread filters
   - Track affected row count
   - Return detailed deletion status report
   - Log full traceback on errors
   - Raise exceptions for deletion failures

Debug Logging:
-------------
The module uses a specialized debug logging system via print__checkpointers_debug()
with numbered debug points for operation tracking:

- 286: Thread entry creation start
- 287: Run ID generation
- 288: Database insertion operation
- 289: Successful thread entry creation
- 290: Thread entry creation error
- 291: Fallback run_id return on error
- Unnamed: Thread retrieval operations
- Unnamed: Thread counting operations
- Unnamed: Thread deletion operations

Configuration Dependencies:
-------------------------
The module relies on configuration constants from checkpointer.config:

- DEFAULT_MAX_RETRIES: Maximum retry attempts for database operations
- THREAD_TITLE_MAX_LENGTH: Maximum characters in thread title
- THREAD_TITLE_SUFFIX_LENGTH: Buffer for ellipsis truncation logic

Usage Examples:
--------------
# Create a new thread run entry
run_id = await create_thread_run_entry(
    email="user@example.com",
    thread_id="thread_abc123",
    prompt="What is the population of Prague?"
)

# Retrieve user's threads with pagination
threads = await get_user_chat_threads(
    email="user@example.com",
    limit=10,
    offset=0
)

# Get total thread count for user
total = await get_user_chat_threads_count(
    email="user@example.com"
)

# Delete a specific thread
result = await delete_user_thread_entries(
    email="user@example.com",
    thread_id="thread_abc123"
)

Return Data Structures:
---------------------
Thread List Item:
{
    "thread_id": str,           # Unique thread identifier
    "latest_timestamp": datetime, # Most recent activity timestamp
    "run_count": int,           # Number of runs in this thread
    "title": str,               # Truncated first prompt or "Untitled Conversation"
    "full_prompt": str          # Complete first prompt text
}

Deletion Result:
{
    "deleted_count": int,       # Number of entries deleted
    "message": str,             # Descriptive status message
    "thread_id": str,           # Thread identifier
    "user_email": str           # User email address
}

Error Handling Strategy:
----------------------
Read Operations (get_user_chat_threads, get_user_chat_threads_count):
- Return empty/zero values on errors to prevent API crashes
- Log detailed error information for debugging
- Enable graceful degradation of user-facing features

Write Operations (create_thread_run_entry):
- Return generated run_id even on database failures
- Enable system to continue operating despite storage issues
- Log errors for monitoring and troubleshooting

Delete Operations (delete_user_thread_entries):
- Raise exceptions to signal critical failures
- Provide detailed deletion status for verification
- Include full traceback for debugging

Required Environment:
-------------------
- Python 3.9+ with async/await support
- PostgreSQL database with users_threads_runs table
- psycopg async driver (psycopg[binary,pool])
- Configured database connection pool
- Appropriate database permissions for CRUD operations

Dependencies:
------------
- api.utils.debug: Debug logging functionality
- checkpointer.database.connection: Database connection management
- checkpointer.error_handling.retry_decorators: Retry logic for database errors
- checkpointer.config: Configuration constants

Performance Considerations:
-------------------------
- Uses database indexes on email and thread_id for fast queries
- Implements pagination to limit result set sizes
- Efficient GROUP BY queries for thread aggregation
- Connection pooling for optimal resource utilization
- Asynchronous operations for non-blocking I/O
- Cursor context managers for automatic cleanup

Security Considerations:
----------------------
- Parameterized queries prevent SQL injection attacks
- User email validation through authentication layer
- Thread ownership verification before deletion
- No raw SQL string interpolation
- Secure connection handling via connection pool"""

import traceback
import uuid
from typing import List, Dict, Any

# Debug logging utilities for checkpointer operations
from api.utils.debug import print__checkpointers_debug

# Database connection management with pooling support
from checkpointer.database.connection import get_direct_connection

# Retry decorator for handling prepared statement errors
from checkpointer.error_handling.retry_decorators import (
    retry_on_prepared_statement_error,
)

# Configuration constants for retry logic and title formatting
from checkpointer.config import (
    DEFAULT_MAX_RETRIES,  # Maximum retry attempts for database operations
    THREAD_TITLE_MAX_LENGTH,  # Maximum length for thread titles before truncation
    THREAD_TITLE_SUFFIX_LENGTH,  # Buffer length for ellipsis in truncated titles
)


# ==============================================================================
# MODULE FUNCTIONS
# ==============================================================================
# This module provides four main functions for thread management:
#
# 1. create_thread_run_entry()    - Creates new thread run entries with upsert logic
# 2. get_user_chat_threads()      - Retrieves paginated list of user's threads
# 3. get_user_chat_threads_count() - Counts total threads for pagination metadata
# 4. delete_user_thread_entries()  - Deletes all entries for a specific thread
# ==============================================================================


# ==============================================================================
# THREAD RUN ENTRY CREATION
# ==============================================================================


@retry_on_prepared_statement_error(max_retries=DEFAULT_MAX_RETRIES)
async def create_thread_run_entry(
    email: str, thread_id: str, prompt: str = None, run_id: str = None
) -> str:
    """Create a new thread run entry in the database with retry logic for prepared statement errors.

    This function creates a new entry in the users_threads_runs table, associating a user's
    conversation thread with a unique run identifier. It implements upsert logic to handle
    duplicate run IDs and provides resilient operation by returning a run_id even if
    database storage fails.

    Note:
        The run_id is passed to LangGraph and used by LangSmith for tracing.
        As the root run ID, it identifies the complete execution in LangSmith.

    Args:
        email (str): User's email address for thread ownership
        thread_id (str): Unique identifier for the conversation thread
        prompt (str, optional): User's query or prompt text. Defaults to None.
        run_id (str, optional): Unique run identifier (UUID). Auto-generated if not provided.

    Returns:
        str: The run_id for this thread entry (either provided or generated)

    Note:
        - Implements ON CONFLICT DO UPDATE for upsert behavior
        - Automatically generates UUID-based run_id if not provided
        - Returns run_id even on database failures for system resilience
        - Updates timestamp on conflict to maintain accurate activity tracking
    """
    # Log the start of thread entry creation operation
    print__checkpointers_debug(
        f"286 - CREATE THREAD ENTRY START: Creating thread run entry for user={email}, thread={thread_id}"
    )
    try:
        # Generate a new UUID-based run_id if not provided by caller
        if not run_id:
            run_id = str(uuid.uuid4())
            print__checkpointers_debug(
                f"287 - GENERATE RUN ID: Generated new run_id: {run_id}"
            )

        # Log the database insertion operation for debugging
        print__checkpointers_debug(
            f"288 - DATABASE INSERT: Inserting thread run entry with run_id={run_id}"
        )

        # Acquire database connection from pool using context manager
        async with get_direct_connection() as conn:
            # Create cursor for executing SQL commands
            async with conn.cursor() as cur:
                # Execute INSERT with ON CONFLICT DO UPDATE for upsert behavior
                # This ensures idempotency - same run_id will update existing record
                await cur.execute(
                    """
                    INSERT INTO users_threads_runs (email, thread_id, run_id, prompt)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (run_id) DO UPDATE SET
                        email = EXCLUDED.email,
                        thread_id = EXCLUDED.thread_id,
                        prompt = EXCLUDED.prompt,
                        timestamp = CURRENT_TIMESTAMP
                """,
                    (email, thread_id, run_id, prompt),
                )

        # Log successful creation and return the run_id
        print__checkpointers_debug(
            f"289 - CREATE THREAD ENTRY SUCCESS: Thread run entry created successfully: {run_id}"
        )
        return run_id
    except Exception as exc:
        # Log error but don't raise - implement graceful degradation
        print__checkpointers_debug(
            f"290 - CREATE THREAD ENTRY ERROR: Failed to create thread run entry: {exc}"
        )
        # Ensure we have a run_id to return even if database operation failed
        # This allows the system to continue operating despite storage failures
        if not run_id:
            run_id = str(uuid.uuid4())
        print__checkpointers_debug(
            f"291 - CREATE THREAD ENTRY FALLBACK: Returning run_id despite database error: {run_id}"
        )
        return run_id


# ==============================================================================
# THREAD RETRIEVAL AND LISTING
# ==============================================================================


@retry_on_prepared_statement_error(max_retries=DEFAULT_MAX_RETRIES)
async def get_user_chat_threads(
    email: str, limit: int = None, offset: int = 0
) -> List[Dict[str, Any]]:
    """Get chat threads for a user with optional pagination and retry logic for prepared statement errors.

    This function retrieves all conversation threads belonging to a specific user, with support
    for pagination. It aggregates thread run data, calculates statistics, generates user-friendly
    titles from the first prompt, and returns a structured list of thread metadata.

    Args:
        email (str): User's email address to filter threads
        limit (int, optional): Maximum number of threads to return. None for all threads.
        offset (int, optional): Number of threads to skip for pagination. Defaults to 0.

    Returns:
        List[Dict[str, Any]]: List of thread objects, each containing:
            - thread_id: Unique thread identifier
            - latest_timestamp: Timestamp of most recent activity
            - run_count: Number of runs in this thread
            - title: Truncated first prompt or "Untitled Conversation"
            - full_prompt: Complete text of the first prompt
        Returns empty list on error to prevent API crashes.

    Note:
        - Threads are sorted by latest_timestamp in descending order (most recent first)
        - Titles are intelligently truncated with ellipsis for long prompts
        - Uses GROUP BY for efficient aggregation of thread statistics
        - Implements graceful error handling with empty list return
    """
    try:
        # Log thread retrieval operation with pagination parameters
        print__checkpointers_debug(
            f"Getting chat threads for user: {email} (limit: {limit}, offset: {offset})"
        )

        # Acquire database connection from pool
        async with get_direct_connection() as conn:
            async with conn.cursor() as cur:
                # Build SQL query with aggregation and subquery for first prompt
                # The query:
                # 1. Groups runs by thread_id for aggregation
                # 2. Gets latest timestamp per thread for sorting
                # 3. Counts total runs per thread for statistics
                # 4. Retrieves first prompt via correlated subquery for title generation
                base_query = """
                    SELECT 
                        thread_id,
                        MAX(timestamp) as latest_timestamp,
                        COUNT(*) as run_count,
                        (SELECT prompt FROM users_threads_runs utr2 
                         WHERE utr2.email = %s AND utr2.thread_id = utr.thread_id 
                         ORDER BY timestamp ASC LIMIT 1) as first_prompt
                    FROM users_threads_runs utr
                    WHERE email = %s
                    GROUP BY thread_id
                    ORDER BY latest_timestamp DESC
                """

                # Build parameter list for parameterized query
                # Email appears twice: once for subquery, once for main WHERE clause
                params = [email, email]

                # Add pagination parameters if limit is specified
                if limit is not None:
                    base_query += " LIMIT %s OFFSET %s"
                    params.extend([limit, offset])

                # Execute query with all parameters
                await cur.execute(base_query, params)

                # Fetch all matching rows
                rows = await cur.fetchall()

                # Initialize list to collect thread objects
                threads = []

                # Process each row and build thread metadata objects
                for row in rows:
                    # Extract column values from result row
                    thread_id = row[0]
                    latest_timestamp = row[1]
                    run_count = row[2]
                    first_prompt = row[3]

                    # Generate user-friendly title from first prompt
                    # Logic:
                    # - If prompt exists and is longer than max length + suffix buffer:
                    #   Truncate to max length and add ellipsis
                    # - If prompt exists but is short enough: Use as-is
                    # - If no prompt: Use "Untitled Conversation" as fallback
                    title = (
                        (first_prompt[:THREAD_TITLE_MAX_LENGTH] + "...")
                        if first_prompt
                        and len(first_prompt)
                        > THREAD_TITLE_MAX_LENGTH + THREAD_TITLE_SUFFIX_LENGTH
                        else (first_prompt or "Untitled Conversation")
                    )

                    # Build thread metadata object and add to results list
                    threads.append(
                        {
                            "thread_id": thread_id,
                            "latest_timestamp": latest_timestamp,
                            "run_count": run_count,
                            "title": title,
                            "full_prompt": first_prompt or "",
                        }
                    )

                # Log successful retrieval with count
                print__checkpointers_debug(
                    f"Retrieved {len(threads)} threads for user {email}"
                )
                return threads

    except Exception as exc:
        # Log error details for debugging
        print__checkpointers_debug(
            f"Failed to get chat threads for user {email}: {exc}"
        )
        # Return empty list instead of raising exception to prevent API crashes
        # This implements graceful degradation - UI can show "no threads" instead of error
        print__checkpointers_debug("Returning empty threads list due to error")
        return []


# ==============================================================================
# THREAD COUNTING
# ==============================================================================


@retry_on_prepared_statement_error(max_retries=DEFAULT_MAX_RETRIES)
async def get_user_chat_threads_count(email: str) -> int:
    """Get total count of chat threads for a user with retry logic for prepared statement errors.

    This function counts the total number of distinct conversation threads for a user,
    which is useful for pagination metadata and displaying total thread counts in the UI.

    Args:
        email (str): User's email address to filter threads

    Returns:
        int: Total number of distinct threads for the user
        Returns 0 on error to prevent API crashes.

    Note:
        - Uses COUNT(DISTINCT thread_id) for accurate counting
        - Implements graceful error handling with zero fallback
        - Supports pagination by providing total count metadata
    """
    try:
        # Acquire database connection from pool
        async with get_direct_connection() as conn:
            async with conn.cursor() as cur:
                # Execute COUNT query with DISTINCT to count unique threads
                # This efficiently counts without loading all thread data
                await cur.execute(
                    """
                    SELECT COUNT(DISTINCT thread_id) as total_threads
                    FROM users_threads_runs
                    WHERE email = %s
                """,
                    (email,),
                )

                # Fetch the count result
                result = await cur.fetchone()
                total_count = result[0] if result else 0

            # Return count with fallback to 0 for null results
            return total_count or 0

    except Exception as exc:
        # Log error details for debugging
        print__checkpointers_debug(
            f"Failed to get chat threads count for user {email}: {exc}"
        )
        # Return 0 instead of raising exception to prevent API crashes
        # This allows pagination to work with empty/error state
        print__checkpointers_debug("Returning 0 thread count due to error")
        return 0


# ==============================================================================
# THREAD DELETION
# ==============================================================================


@retry_on_prepared_statement_error(max_retries=DEFAULT_MAX_RETRIES)
async def delete_user_thread_entries(email: str, thread_id: str) -> Dict[str, Any]:
    """Delete all entries for a user's thread from users_threads_runs table with retry logic for prepared statement errors.

    This function removes all run entries associated with a specific thread for a given user.
    It performs pre-deletion counting for verification and returns detailed deletion status.

    Args:
        email (str): User's email address for ownership verification
        thread_id (str): Unique identifier of the thread to delete

    Returns:
        Dict[str, Any]: Deletion status report containing:
            - deleted_count: Number of entries actually deleted
            - message: Descriptive status message
            - thread_id: The thread identifier that was deleted
            - user_email: The user email for verification

    Raises:
        Exception: Re-raises any database errors after logging for proper error handling

    Note:
        - Counts entries before deletion for verification
        - Returns early with zero count if no entries exist
        - Logs detailed error information including full traceback
        - Raises exceptions (unlike read operations) to signal critical failures
    """
    try:
        # Log deletion operation start with user and thread identifiers
        print__checkpointers_debug(
            f"Deleting thread entries for user: {email}, thread: {thread_id}"
        )

        # Acquire database connection from pool
        async with get_direct_connection() as conn:
            # First, count the entries to be deleted for verification and reporting
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT COUNT(*) FROM users_threads_runs 
                    WHERE email = %s AND thread_id = %s
                """,
                    (email, thread_id),
                )
                result = await cur.fetchone()
                entries_to_delete = result[0] if result else 0

            # Log the number of entries found for deletion
            print__checkpointers_debug(f"Found {entries_to_delete} entries to delete")

            # Early return if no entries exist - nothing to delete
            if entries_to_delete == 0:
                return {
                    "deleted_count": 0,
                    "message": "No entries found to delete",
                    "thread_id": thread_id,
                    "user_email": email,
                }

            # Proceed with deletion if entries exist
            async with conn.cursor() as cur:
                # Execute DELETE query with email and thread_id filters
                await cur.execute(
                    """
                    DELETE FROM users_threads_runs 
                    WHERE email = %s AND thread_id = %s
                """,
                    (email, thread_id),
                )
                # Capture the number of rows actually deleted
                deleted_count = cur.rowcount

            # Log successful deletion with count
            print__checkpointers_debug(
                f"Deleted {deleted_count} entries for user {email}, thread {thread_id}"
            )

            # Return detailed deletion status report
            return {
                "deleted_count": deleted_count,
                "message": f"Successfully deleted {deleted_count} entries",
                "thread_id": thread_id,
                "user_email": email,
            }

    except Exception as exc:
        # Log error details with exception message
        print__checkpointers_debug(
            f"Failed to delete thread entries for user {email}, thread {thread_id}: {exc}"
        )
        # Log full traceback for detailed debugging
        print__checkpointers_debug(f"Full traceback: {traceback.format_exc()}")
        # Re-raise exception for proper error handling at higher levels
        # Unlike read operations, deletion failures should be propagated
        raise
