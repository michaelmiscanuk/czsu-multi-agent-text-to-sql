"""User Feedback Sentiment Tracking for PostgreSQL Checkpointer System

This module provides robust sentiment tracking and retrieval functionality for user feedback
in the context of multi-agent conversation threads stored in a PostgreSQL database.
"""

from __future__ import annotations

MODULE_DESCRIPTION = r"""User Feedback Sentiment Tracking for PostgreSQL Checkpointer System

This module provides robust sentiment tracking and retrieval functionality for user feedback
in the context of multi-agent conversation threads stored in a PostgreSQL database.

Key Features:
-------------
1. Sentiment Update Operations:
   - Direct sentiment updates for individual thread runs
   - Boolean sentiment values (positive/negative feedback)
   - Run-level granularity for precise feedback tracking
   - Atomic database operations for data consistency
   - Automatic retry logic for prepared statement conflicts

2. Sentiment Retrieval Operations:
   - Bulk sentiment retrieval for entire threads
   - User-specific and thread-specific filtering
   - Dictionary-based mapping of run IDs to sentiments
   - Null-safe queries (excludes runs without sentiment)
   - Efficient batch retrieval for analytics

3. Error Resilience:
   - Automatic retry on prepared statement errors
   - Configurable retry limits from central configuration
   - Graceful degradation on persistent failures
   - Comprehensive error logging for diagnostics
   - Transaction safety with connection pooling

4. Database Integration:
   - PostgreSQL-backed persistent storage
   - Connection pool management via direct connection
   - Parameterized queries for SQL injection protection
   - Async/await pattern for non-blocking operations
   - Row count validation for update confirmation

5. Debug Support:
   - Detailed debug logging for all operations
   - Operation status tracking and reporting
   - Performance monitoring capabilities
   - Failure diagnostics and error tracing
   - Integration with system-wide debug framework

Processing Flow:
--------------
Sentiment Update Flow:
1. Receive run_id and sentiment value (bool)
2. Acquire database connection from pool
3. Execute UPDATE query with retry protection
4. Validate row count to confirm update
5. Log operation status and return result
6. Auto-retry on prepared statement conflicts

Sentiment Retrieval Flow:
1. Receive email and thread_id for filtering
2. Acquire database connection from pool
3. Execute SELECT query for all sentiments
4. Filter NULL sentiments automatically
5. Build dictionary mapping run_id → sentiment
6. Log retrieval count and return results

Database Schema:
--------------
Table: users_threads_runs
Columns used by this module:
- run_id (str): Unique identifier for conversation run [Primary filter]
- email (str): User email for ownership filtering [Secondary filter]
- thread_id (str): Thread identifier for grouping [Tertiary filter]
- sentiment (bool | NULL): User feedback sentiment value [Target field]
  * True: Positive feedback
  * False: Negative feedback
  * NULL: No feedback provided

Usage Examples:
-------------
# Update sentiment for a specific run
success = await update_thread_run_sentiment(
    run_id="run_abc123",
    sentiment=True  # Positive feedback
)

# Retrieve all sentiments for a thread
sentiments = await get_thread_run_sentiments(
    email="user@example.com",
    thread_id="thread_xyz789"
)
# Returns: {"run_abc123": True, "run_def456": False, ...}

Required Environment:
-------------------
- Python 3.11+ (for async/await support)
- PostgreSQL database with users_threads_runs table
- Database connection pool configured
- Required packages: asyncpg or psycopg (async)
- Retry decorator dependencies configured

API Functions:
-------------
update_thread_run_sentiment(run_id, sentiment) -> bool:
    Updates sentiment for a specific run with retry protection.
    Returns True if update successful, False on failure.

get_thread_run_sentiments(email, thread_id) -> Dict[str, bool]:
    Retrieves all sentiments for a thread filtered by user email.
    Returns dictionary mapping run_id to sentiment boolean values.

Error Handling:
-------------
- Prepared statement conflicts: Auto-retry with exponential backoff
- Connection failures: Handled by connection pool manager
- Transaction errors: Logged and returned as operation failure
- Query execution errors: Caught and logged, returns safe defaults
- Network timeouts: Managed by underlying connection layer

Configuration:
-------------
Retry behavior controlled by:
- DEFAULT_MAX_RETRIES: Maximum retry attempts (from config module)
- Exponential backoff: Automatic via retry decorator
- Connection timeout: Configured at connection pool level
- Statement timeout: Database server configuration

Integration Points:
-----------------
- checkpointer.database.connection: Database connection management
- checkpointer.error_handling.retry_decorators: Retry logic
- checkpointer.config: Configuration parameters
- api.utils.debug: Debug logging framework

Performance Considerations:
-------------------------
- Connection pooling: Reduces connection overhead
- Parameterized queries: Prevents SQL compilation overhead
- Batch retrieval: Single query for all thread sentiments
- Async operations: Non-blocking for concurrent requests
- Index requirements: run_id, (email, thread_id) composite
- Null filtering: Database-level for efficiency

Security:
--------
- SQL injection protection via parameterized queries
- Email-based access control for sentiment retrieval
- No direct SQL string interpolation
- Connection credentials managed externally
- Audit trail via debug logging"""

from typing import Dict

# Debug utilities for operation logging and diagnostics
from api.utils.debug import print__checkpointers_debug

# Database connection management with pooling support
from checkpointer.database.connection import get_direct_connection

# Retry decorator for handling transient database errors
from checkpointer.error_handling.retry_decorators import (
    retry_on_prepared_statement_error,
)

# Configuration constants for retry behavior
from checkpointer.config import DEFAULT_MAX_RETRIES


# ==============================================================================
# MODULE FUNCTIONS
# ==============================================================================
# This module provides two core functions for sentiment tracking:
# 1. update_thread_run_sentiment() - Updates sentiment for a specific run
# 2. get_thread_run_sentiments() - Retrieves all sentiments for a thread
#
# Both functions use async database operations with automatic retry logic
# to handle prepared statement conflicts and ensure data consistency.


# ==============================================================================
# SENTIMENT UPDATE FUNCTION
# ==============================================================================
@retry_on_prepared_statement_error(max_retries=DEFAULT_MAX_RETRIES)
async def update_thread_run_sentiment(run_id: str, sentiment: bool) -> bool:
    """Update sentiment for a thread run by run_id with retry logic for prepared statement errors.

    This function updates the sentiment field in the users_threads_runs table for a specific
    run identifier. The operation is protected by automatic retry logic to handle transient
    database errors such as prepared statement conflicts.

    Args:
        run_id (str): Unique identifier for the conversation run to update
        sentiment (bool): Boolean sentiment value (True=positive, False=negative)

    Returns:
        bool: True if at least one row was updated successfully, False on failure
              or if no matching run_id was found

    Note:
        - Uses parameterized query to prevent SQL injection
        - Automatically retries on prepared statement errors up to DEFAULT_MAX_RETRIES
        - Returns False for any exception to prevent caller crashes
        - Logs all operations for debugging and audit purposes
    """
    try:
        # Log the update operation for debugging and audit trail
        print__checkpointers_debug(f"Updating sentiment for run {run_id}: {sentiment}")

        # Acquire database connection from pool and ensure automatic cleanup
        async with get_direct_connection() as conn:
            # Create cursor for executing the SQL statement
            async with conn.cursor() as cur:
                # Execute UPDATE query with parameterized values for security
                # Sets sentiment field for the matching run_id
                await cur.execute(
                    """
                    UPDATE users_threads_runs 
                    SET sentiment = %s 
                    WHERE run_id = %s
                """,
                    (sentiment, run_id),
                )
                # Capture number of rows affected by the update operation
                updated = cur.rowcount

        # Log the result of the update operation
        print__checkpointers_debug(f"Updated sentiment for {updated} entries")

        # Return True if at least one row was updated, False otherwise
        return int(updated) > 0

    except Exception as exc:
        # Catch all exceptions to prevent function from crashing caller
        # Log the failure for diagnostics and debugging
        print__checkpointers_debug(f"Failed to update sentiment: {exc}")
        # Return False to indicate operation failure
        return False


# ==============================================================================
# SENTIMENT RETRIEVAL FUNCTION
# ==============================================================================
@retry_on_prepared_statement_error(max_retries=DEFAULT_MAX_RETRIES)
async def get_thread_run_sentiments(email: str, thread_id: str) -> Dict[str, bool]:
    """Get all sentiments for a thread with retry logic for prepared statement errors.

    This function retrieves all sentiment values for a specific conversation thread,
    filtered by user email and thread identifier. Only runs with non-null sentiment
    values are included in the results.

    Args:
        email (str): User email for ownership filtering and access control
        thread_id (str): Thread identifier to retrieve sentiments for

    Returns:
        Dict[str, bool]: Dictionary mapping run_id (str) to sentiment (bool)
                        Empty dictionary on failure or if no sentiments found
                        Example: {"run_abc123": True, "run_def456": False}

    Note:
        - Automatically filters out NULL sentiment values at database level
        - Uses composite filtering (email AND thread_id) for security
        - Automatically retries on prepared statement errors up to DEFAULT_MAX_RETRIES
        - Returns empty dict for any exception to prevent caller crashes
        - Logs all operations for debugging and audit purposes
    """
    try:
        # Log the retrieval operation for debugging and audit trail
        print__checkpointers_debug(f"Getting sentiments for thread {thread_id}")

        # Acquire database connection from pool and ensure automatic cleanup
        async with get_direct_connection() as conn:
            # Create cursor for executing the SQL statement
            async with conn.cursor() as cur:
                # Execute SELECT query with parameterized values for security
                # Filters by email (user ownership) and thread_id (conversation)
                # Excludes NULL sentiments to return only actual feedback
                await cur.execute(
                    """
                    SELECT run_id, sentiment 
                    FROM users_threads_runs 
                    WHERE email = %s AND thread_id = %s AND sentiment IS NOT NULL
                """,
                    (email, thread_id),
                )
                # Fetch all matching rows from the query result
                rows = await cur.fetchall()

        # Build dictionary mapping run_id to sentiment value
        # row[0] is run_id (string), row[1] is sentiment (boolean)
        sentiments = {row[0]: row[1] for row in rows}

        # Log the number of sentiments retrieved for diagnostics
        print__checkpointers_debug(f"Retrieved {len(sentiments)} sentiments")

        # Return the dictionary of run_id to sentiment mappings
        return sentiments

    except Exception as exc:
        # Catch all exceptions to prevent function from crashing caller
        # Log the failure for diagnostics and debugging
        print__checkpointers_debug(f"Failed to get sentiments: {exc}")
        # Return empty dictionary to indicate operation failure or no results
        return {}
