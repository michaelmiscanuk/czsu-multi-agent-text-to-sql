"""PostgreSQL Prepared Statement Error Detection and Cleanup

This module provides specialized functionality for detecting and resolving prepared
statement conflicts that can occur in PostgreSQL database connections, particularly
when using connection pooling or when prepared statements become stale.
"""

from __future__ import annotations

MODULE_DESCRIPTION = r"""PostgreSQL Prepared Statement Error Detection and Cleanup

This module provides specialized functionality for detecting and resolving prepared
statement conflicts that can occur in PostgreSQL database connections. It is a critical
component of the checkpointer error handling system, ensuring robust operation when
using psycopg connection pools with prepared statements.

Key Features:
-------------
1. Prepared Statement Error Detection:
   - Pattern-based error message analysis
   - Support for both psycopg2 and psycopg3 naming conventions
   - Case-insensitive error matching for robustness
   - PostgreSQL error code recognition
   - Multiple error indicator support

2. Comprehensive Statement Cleanup:
   - System catalog query for existing prepared statements
   - Automatic DEALLOCATE command execution
   - Batch processing for multiple statements
   - Dedicated cleanup connection with unique app name
   - Non-blocking operation that continues on failures

3. Connection Management:
   - Separate cleanup connection to avoid conflicts
   - Prepared statement prevention through connection kwargs
   - Unique application naming for traceability
   - SSL/TLS connection requirement enforcement
   - Graceful connection handling and cleanup

4. Error Recovery Support:
   - Integration with retry decorator system
   - Triggers checkpointer recreation on cleanup
   - Detailed logging for troubleshooting
   - Performance optimizations for large cleanups
   - Summary statistics for cleanup operations

5. Debug and Monitoring:
   - Comprehensive debug logging throughout operations
   - Statement-level logging for first few operations
   - Summary logging for large cleanup sets
   - Error tracking for failed DEALLOCATE operations
   - Non-fatal error handling with detailed reporting

Prepared Statement Conflicts:
---------------------------
Prepared statements in PostgreSQL are pre-parsed SQL queries that can be executed
multiple times with different parameters. While they improve performance, they can
cause conflicts when:

1. Connection Pooling:
   - Multiple connections share prepared statements
   - Statement names collide between pool connections
   - Stale statements persist after connection reuse

2. Connection State:
   - Prepared statements survive connection errors
   - Statement cache becomes out of sync
   - Connection pool returns connection with stale state

3. Naming Conflicts:
   - psycopg auto-generated names (e.g., "_pg3_0", "_pg_1")
   - Multiple psycopg versions using different naming schemes
   - Statement name reuse across different queries

Error Detection Mechanism:
------------------------
The is_prepared_statement_error() function analyzes exception objects to identify
prepared statement related errors through multiple indicators:

1. Error Message Patterns:
   - "prepared statement" - Direct reference to prepared statements
   - "does not exist" - Missing statement references
   - "invalidsqlstatementname" - PostgreSQL error code

2. Statement Name Patterns:
   - "_pg3_" - psycopg3 auto-generated statement names
   - "_pg_" - psycopg2 auto-generated statement names

3. Detection Process:
   - Convert error to lowercase string for matching
   - Check all indicator patterns using any() function
   - Return True if any pattern matches
   - Provide debug logging for troubleshooting

Cleanup Process:
---------------
The clear_prepared_statements() function performs comprehensive cleanup:

1. Connection Setup:
   - Generate unique application name with UUID
   - Create PostgreSQL connection string with SSL
   - Apply prepared statement prevention kwargs
   - Establish dedicated psycopg async connection

2. Statement Discovery:
   - Query pg_prepared_statements system catalog
   - Filter for psycopg naming patterns (_pg3_%, _pg_%)
   - Retrieve all matching statement names
   - Count statements for logging

3. Statement Removal:
   - Iterate through all found prepared statements
   - Execute DEALLOCATE command for each statement
   - Log first few operations in detail
   - Continue on individual failures
   - Provide summary for large cleanup sets

4. Resource Cleanup:
   - Automatic cursor and connection cleanup via context managers
   - Non-fatal error handling throughout
   - Detailed error logging for troubleshooting
   - Graceful return to avoid blocking operations

Usage in Error Recovery:
-----------------------
This module integrates with the retry decorator system:

```python
@retry_on_prepared_statement_error(max_retries=3)
async def database_operation():
    # If prepared statement error occurs:
    # 1. is_prepared_statement_error() detects the error
    # 2. clear_prepared_statements() cleans up conflicts
    # 3. Global checkpointer recreated
    # 4. Operation retried with fresh state
    pass
```

Performance Optimizations:
------------------------
1. Logging Limits:
   - Detailed logging for first 3 statements only
   - Summary logging for operations beyond first 3
   - Prevents log spam on large cleanup operations

2. Batch Operations:
   - Single query to discover all statements
   - Efficient iteration through results
   - Minimal database round trips

3. Non-blocking Design:
   - Individual statement failures don't stop cleanup
   - Continues processing remaining statements
   - Returns gracefully to caller

Connection String Format:
-----------------------
The cleanup connection uses a comprehensive PostgreSQL connection string:

```
postgresql://{user}:{password}@{host}:{port}/{dbname}?sslmode=require&application_name={cleanup_app_name}
```

Components:
- Protocol: postgresql://
- Authentication: user and password from config
- Server: host and port from config
- Database: dbname from config
- SSL: sslmode=require for security
- Identification: Unique application_name for traceability

Prevention Strategy:
------------------
While this module handles cleanup, prevention is preferred:

1. Connection Configuration:
   - Use get_connection_kwargs() to disable prepared statements
   - Apply prepare_threshold=None to prevent auto-preparation
   - Configure connection pool with appropriate settings

2. Connection Pool Management:
   - Regular health checking of pooled connections
   - Proper pool size configuration
   - Connection validation before use

3. Application Design:
   - Minimize long-lived connections
   - Proper connection cleanup in error handlers
   - Avoid manual prepared statement creation

Debug Logging:
-------------
The module provides comprehensive debug output with numbered markers:

- 200: Prepared statement error check initiation
- 201: Error check result
- 221: Cleanup operation start
- 222: Cleanup connection creation
- 223: psycopg connection establishment
- 224: Connection success confirmation
- 225: Database cursor creation
- 226: Prepared statements discovery
- 227-229: Individual statement cleanup (first 3)
- 230: Cleanup summary/completion
- 231: No statements found
- 232: Cleanup error (non-fatal)

Error Handling Philosophy:
------------------------
This module follows a non-fatal error approach:

1. Detection Errors:
   - Always return boolean, never raise
   - Safe to call in exception handlers
   - Minimal overhead for non-error cases

2. Cleanup Errors:
   - Individual statement failures don't stop cleanup
   - Overall cleanup failure doesn't block checkpointer
   - Detailed logging for troubleshooting
   - Graceful degradation under all conditions

3. Recovery Integration:
   - Cleanup triggers checkpointer recreation
   - Fresh connections on next access
   - Automatic retry of failed operations

Dependencies:
------------
- psycopg: Async PostgreSQL driver (psycopg3)
- uuid: Unique identifier generation for app names
- api.utils.debug: Debug logging utilities (print__checkpointers_debug)
- checkpointer.database.connection: Connection kwargs provider
- checkpointer.config: Database configuration

PostgreSQL System Catalog:
-------------------------
The module queries pg_prepared_statements system view:

- name: Prepared statement identifier
- statement: Original SQL text
- prepare_time: When statement was prepared
- parameter_types: Parameter type array
- from_sql: Whether created by PREPARE SQL command

Query used:
```sql
SELECT name FROM pg_prepared_statements 
WHERE name LIKE '_pg3_%' OR name LIKE '_pg_%';
```

Security Considerations:
----------------------
1. Connection Security:
   - SSL/TLS required for all connections
   - Credentials from secure config module
   - Unique application names for audit trails

2. Query Safety:
   - Pattern-based filtering prevents broad deletion
   - Limited to psycopg naming conventions
   - No user-supplied input in queries

3. Permission Requirements:
   - Requires SELECT on pg_prepared_statements
   - Requires DEALLOCATE privilege
   - Standard user permissions sufficient

Known Limitations:
----------------
1. Scope:
   - Only cleans psycopg-named statements
   - Custom prepared statements not affected
   - Per-connection cleanup only

2. Concurrency:
   - Race conditions possible with concurrent operations
   - Cleanup may miss statements created during operation
   - Best effort approach, not atomic

3. Performance:
   - System catalog queries have overhead
   - Large statement sets may take time
   - Should not be called in hot paths

Troubleshooting:
---------------
1. Detection Issues:
   - Check error message format
   - Verify error type inheritance
   - Review debug log output (200-201 markers)

2. Cleanup Failures:
   - Verify database permissions
   - Check connection string format
   - Review debug logs (221-232 markers)
   - Ensure pg_prepared_statements accessible

3. Recovery Problems:
   - Verify checkpointer recreation logic
   - Check retry decorator application
   - Review global state management

Best Practices:
--------------
1. Always use with retry decorators
2. Monitor cleanup frequency for systemic issues
3. Keep connection pool size appropriate
4. Review debug logs for patterns
5. Prefer prevention over cleanup
6. Don't call cleanup in hot paths
7. Trust non-fatal error handling

Future Enhancements:
------------------
- Prepared statement age-based cleanup
- Connection-specific statement tracking
- Cleanup metrics and monitoring
- Automatic cleanup scheduling
- Enhanced pattern detection
- Statement usage statistics"""

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
    # Log the start of error detection analysis
    print__checkpointers_debug(
        "200 - PREPARED STATEMENT CHECK: Checking if error is prepared statement related"
    )

    # Convert error to lowercase string for case-insensitive matching
    error_str = str(error).lower()

    # Check if any of the prepared statement error indicators are present in error message
    # This covers multiple error patterns from both psycopg2 and psycopg3
    result = any(
        indicator in error_str
        for indicator in [
            "prepared statement",  # Direct reference to prepared statements
            "does not exist",  # Missing statement reference errors
            "_pg3_",  # psycopg3 auto-generated statement naming pattern
            "_pg_",  # psycopg2 auto-generated statement naming pattern
            "invalidsqlstatementname",  # PostgreSQL error code for invalid statement names
        ]
    )

    # Log the detection result for debugging and monitoring
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
    # Log the start of the cleanup operation for monitoring
    print__checkpointers_debug(
        "221 - CLEAR PREPARED START: Starting prepared statements cleanup"
    )
    try:
        # Retrieve database configuration (host, port, credentials, database name)
        config = get_db_config()

        # Generate unique application name for this cleanup connection
        # This helps identify cleanup operations in PostgreSQL logs and pg_stat_activity
        # Format: czsu_cleanup_<8-char-hex> (e.g., czsu_cleanup_a1b2c3d4)
        cleanup_app_name = f"czsu_cleanup_{uuid.uuid4().hex[:8]}"
        print__checkpointers_debug(
            f"222 - CLEANUP CONNECTION: Creating cleanup connection with app name: {cleanup_app_name}"
        )

        # Build PostgreSQL connection string with all required parameters:
        # - postgresql:// protocol prefix
        # - user:password for authentication
        # - host:port for server location
        # - /dbname for database selection
        # - sslmode=require for enforced SSL/TLS encryption
        # - application_name for connection identification in logs
        connection_string = f"postgresql://{config['user']}:{config['password']}@{config['host']}:{config['port']}/{config['dbname']}?sslmode=require&application_name={cleanup_app_name}"

        # Get connection kwargs that disable prepared statement auto-generation
        # This prevents the cleanup connection itself from creating new prepared statements
        connection_kwargs = get_connection_kwargs()

        # Log before establishing connection for troubleshooting connection issues
        print__checkpointers_debug(
            "223 - PSYCOPG CONNECTION: Establishing psycopg connection for cleanup"
        )

        # Establish async PostgreSQL connection using psycopg3
        # Context manager ensures automatic connection cleanup even if errors occur
        async with await psycopg.AsyncConnection.connect(
            connection_string, **connection_kwargs
        ) as conn:
            print__checkpointers_debug(
                "224 - CONNECTION ESTABLISHED: Cleanup connection established successfully"
            )

            # Create database cursor for executing SQL commands
            # Context manager ensures cursor cleanup after use
            async with conn.cursor() as cur:
                print__checkpointers_debug(
                    "225 - CURSOR CREATED: Database cursor created for prepared statement query"
                )

                # Query PostgreSQL system catalog for all prepared statements
                # matching psycopg naming patterns (_pg3_* for psycopg3, _pg_* for psycopg2)
                # This finds auto-generated statements that may be causing conflicts
                await cur.execute(
                    """
                    SELECT name FROM pg_prepared_statements 
                    WHERE name LIKE '_pg3_%' OR name LIKE '_pg_%';
                """
                )

                # Fetch all matching prepared statement names from query results
                prepared_statements = await cur.fetchall()

                # Process found prepared statements if any exist
                if prepared_statements:
                    print__checkpointers_debug(
                        f"226 - PREPARED STATEMENTS FOUND: Found {len(prepared_statements)} prepared statements to clear"
                    )

                    # Iterate through all prepared statements and deallocate each one
                    # Enumerate starting at 1 for human-readable logging
                    for i, stmt in enumerate(prepared_statements, 1):
                        # Extract statement name from result tuple (first column)
                        stmt_name = stmt[0]

                        # Limit detailed logging to first 3 statements to prevent log spam
                        # For large cleanup operations, we just show summary at the end
                        if i <= 3:
                            print__checkpointers_debug(
                                f"227 - CLEARING STATEMENT {i}: Clearing prepared statement: {stmt_name}"
                            )

                        try:
                            # Execute DEALLOCATE command to remove the prepared statement
                            # This frees up the statement name and associated resources
                            await cur.execute(f"DEALLOCATE {stmt_name};")

                            # Log successful deallocate for first 3 statements only
                            if i <= 3:
                                print__checkpointers_debug(
                                    f"228 - STATEMENT CLEARED {i}: Successfully cleared prepared statement: {stmt_name}"
                                )
                        except Exception as exc:
                            # Log individual statement failures but continue processing others
                            # Non-fatal: one failed DEALLOCATE shouldn't stop entire cleanup
                            if i <= 3:
                                print__checkpointers_debug(
                                    f"229 - STATEMENT ERROR {i}: Could not clear prepared statement {stmt_name}: {exc}"
                                )

                    # Provide summary logging based on total number of statements processed
                    if len(prepared_statements) > 3:
                        # For large cleanup operations, note that we only showed first 3 in detail
                        print__checkpointers_debug(
                            f"230 - CLEANUP SUMMARY: Cleared {len(prepared_statements)} prepared statements (showing first 3)"
                        )
                    else:
                        # For small cleanup operations, all statements were logged individually
                        print__checkpointers_debug(
                            f"230 - CLEANUP COMPLETE: Cleared {len(prepared_statements)} prepared statements"
                        )
                else:
                    # No prepared statements found matching psycopg patterns
                    # This is normal if prepared statements are properly disabled
                    print__checkpointers_debug(
                        "231 - NO STATEMENTS: No prepared statements to clear"
                    )

    except Exception as exc:
        # Catch and log any errors during the entire cleanup process
        # This is marked as non-fatal because cleanup failures shouldn't block
        # checkpointer operations - they can continue with recreated connections
        print__checkpointers_debug(
            f"232 - CLEANUP ERROR: Error clearing prepared statements (non-fatal): {exc}"
        )
        # Explicitly don't re-raise the exception
        # This is a cleanup operation and shouldn't block checkpointer creation or retries
