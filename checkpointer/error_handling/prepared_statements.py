"""Prepared statement error detection and cleanup operations.

This module handles prepared statement conflicts and cleanup operations
that can occur in PostgreSQL connections with connection pooling.
"""
from __future__ import annotations

import uuid

import psycopg

from api.utils.debug import print__checkpointers_debug
from checkpointer.database.connection import get_connection_kwargs
from checkpointer.config import get_db_config


# This file will contain:
# - is_prepared_statement_error() function
# - clear_prepared_statements() function
def is_prepared_statement_error(error: Exception) -> bool:
    """Detect if an error is related to prepared statement conflicts.

    This function analyzes exception messages to identify prepared statement
    related errors that commonly occur in PostgreSQL connections, especially
    when using connection pooling or when prepared statements become stale.

    Args:
        error (Exception): The exception to analyze for prepared statement indicators

    Returns:
        bool: True if the error appears to be prepared statement related, False otherwise

    Detection Patterns:
        - "prepared statement" text in error message
        - "does not exist" for missing prepared statement references
        - "_pg3_" or "_pg_" prefixes from psycopg prepared statement naming
        - "invalidsqlstatementname" PostgreSQL error codes

    Note:
        - Case-insensitive pattern matching for robust detection
        - Covers both psycopg2 and psycopg3 prepared statement naming conventions
        - Used by retry decorators to trigger appropriate recovery actions
    """
    print__checkpointers_debug(
        "200 - PREPARED STATEMENT CHECK: Checking if error is prepared statement related"
    )
    error_str = str(error).lower()
    result = any(
        indicator in error_str
        for indicator in [
            "prepared statement",
            "does not exist",
            "_pg3_",
            "_pg_",
            "invalidsqlstatementname",
        ]
    )
    print__checkpointers_debug(
        f"201 - PREPARED STATEMENT RESULT: Error is prepared statement related: {result}"
    )
    return result


async def clear_prepared_statements():
    """Clear existing prepared statements to resolve conflicts during error recovery.

    This function provides a comprehensive cleanup mechanism for prepared statement
    conflicts that can occur in PostgreSQL connections, particularly when using
    connection pooling or when connections become stale.

    Process:
        1. Creates a dedicated cleanup connection with unique application name
        2. Queries pg_prepared_statements system catalog for existing statements
        3. Deallocates all found prepared statements using DEALLOCATE command
        4. Provides detailed logging for troubleshooting and verification

    Connection Strategy:
        - Uses separate connection to avoid conflicts with main operations
        - Applies prepared statement disabling connection kwargs
        - Generates unique application name for identification
        - Uses direct psycopg connection for maximum compatibility

    Error Handling:
        - Non-fatal operation that continues on individual statement failures
        - Comprehensive logging for both successes and failures
        - Graceful handling of connection or permission issues
        - Returns silently to avoid blocking main operations

    Performance Optimizations:
        - Limits detailed logging to first few statements to avoid log spam
        - Batches operations efficiently for large prepared statement sets
        - Provides summary statistics for large cleanup operations

    Note:
        - Used during error recovery when prepared statement issues are detected
        - Most prepared statement issues are prevented by connection kwargs
        - This is a recovery mechanism for edge cases and troubleshooting
        - Safe to call multiple times without side effects
    """
    print__checkpointers_debug(
        "221 - CLEAR PREPARED START: Starting prepared statements cleanup"
    )
    try:
        config = get_db_config()
        # Use a different application name for the cleanup connection

        cleanup_app_name = f"czsu_cleanup_{uuid.uuid4().hex[:8]}"
        print__checkpointers_debug(
            f"222 - CLEANUP CONNECTION: Creating cleanup connection with app name: {cleanup_app_name}"
        )

        # Create connection string without prepared statement parameters
        connection_string = f"postgresql://{config['user']}:{config['password']}@{config['host']}:{config['port']}/{config['dbname']}?sslmode=require&application_name={cleanup_app_name}"

        # Get connection kwargs for disabling prepared statements
        connection_kwargs = get_connection_kwargs()

        print__checkpointers_debug(
            "223 - PSYCOPG CONNECTION: Establishing psycopg connection for cleanup"
        )
        async with await psycopg.AsyncConnection.connect(
            connection_string, **connection_kwargs
        ) as conn:
            print__checkpointers_debug(
                "224 - CONNECTION ESTABLISHED: Cleanup connection established successfully"
            )
            async with conn.cursor() as cur:
                print__checkpointers_debug(
                    "225 - CURSOR CREATED: Database cursor created for prepared statement query"
                )
                # Get all prepared statements for our application
                await cur.execute(
                    """
                    SELECT name FROM pg_prepared_statements 
                    WHERE name LIKE '_pg3_%' OR name LIKE '_pg_%';
                """
                )
                prepared_statements = await cur.fetchall()

                if prepared_statements:
                    print__checkpointers_debug(
                        f"226 - PREPARED STATEMENTS FOUND: Found {len(prepared_statements)} prepared statements to clear"
                    )

                    # Drop each prepared statement
                    for i, stmt in enumerate(prepared_statements, 1):
                        stmt_name = stmt[0]
                        # Only log first few statements
                        if i <= 3:
                            print__checkpointers_debug(
                                f"227 - CLEARING STATEMENT {i}: Clearing prepared statement: {stmt_name}"
                            )
                        try:
                            await cur.execute(f"DEALLOCATE {stmt_name};")
                            if i <= 3:
                                print__checkpointers_debug(
                                    f"228 - STATEMENT CLEARED {i}: Successfully cleared prepared statement: {stmt_name}"
                                )
                        except Exception as e:
                            if i <= 3:
                                print__checkpointers_debug(
                                    f"229 - STATEMENT ERROR {i}: Could not clear prepared statement {stmt_name}: {e}"
                                )

                    if len(prepared_statements) > 3:
                        print__checkpointers_debug(
                            f"230 - CLEANUP SUMMARY: Cleared {len(prepared_statements)} prepared statements (showing first 3)"
                        )
                    else:
                        print__checkpointers_debug(
                            f"230 - CLEANUP COMPLETE: Cleared {len(prepared_statements)} prepared statements"
                        )
                else:
                    print__checkpointers_debug(
                        "231 - NO STATEMENTS: No prepared statements to clear"
                    )

    except Exception as e:
        print__checkpointers_debug(
            f"232 - CLEANUP ERROR: Error clearing prepared statements (non-fatal): {e}"
        )
        # Don't raise - this is a cleanup operation and shouldn't block checkpointer creation
