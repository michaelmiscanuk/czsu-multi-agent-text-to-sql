"""PostgreSQL Checkpointer Error Handling and Recovery System

This package provides comprehensive error detection, cleanup, and recovery mechanisms
for the PostgreSQL-based checkpointer system used in the CZSU multi-agent text-to-SQL
application. It handles common database connection issues, prepared statement conflicts,
and SSL connection errors that can occur during long-running agent workflows.
"""

MODULE_DESCRIPTION = r"""PostgreSQL Checkpointer Error Handling and Recovery System

This package provides comprehensive error detection, cleanup, and recovery mechanisms
for the PostgreSQL-based checkpointer system used in the CZSU multi-agent text-to-SQL
application. The error handling system ensures robust operation during agent execution
by automatically detecting and recovering from common database errors.

Key Features:
-------------
1. Prepared Statement Error Management:
   - Detection of prepared statement conflicts in PostgreSQL connections
   - Automatic cleanup of stale prepared statements
   - Connection pool recreation for recovery
   - Pattern-based error identification (psycopg2 and psycopg3)
   - System catalog queries for comprehensive cleanup

2. SSL Connection Error Recovery:
   - Detection of SSL connection failures and socket errors
   - Exponential backoff retry strategy
   - Connection pool health checking and recreation
   - Graceful degradation under network issues
   - Automatic state cleanup and recovery

3. Retry Logic and Decorators:
   - Configurable retry attempts with exponential backoff
   - Automatic error detection and recovery triggering
   - Function decoration for transparent retry behavior
   - Detailed debug logging for troubleshooting
   - Non-blocking cleanup operations

4. Connection Pool Management:
   - Automatic pool closure on persistent errors
   - Fresh pool creation after cleanup
   - Global checkpointer state management
   - Connection health verification
   - Resource cleanup and leak prevention

5. Debug and Monitoring:
   - Comprehensive debug logging throughout error handling
   - Error pattern tracking and analysis
   - Recovery attempt monitoring
   - Performance metrics for retry operations
   - Traceback capture for root cause analysis

Package Structure:
-----------------
- prepared_statements.py: Prepared statement error detection and cleanup
- retry_decorators.py: Retry logic decorators for error recovery
- __init__.py: Package initialization and documentation (this file)

Common Error Scenarios:
---------------------
1. Prepared Statement Conflicts:
   - Occur when prepared statements become stale or duplicated
   - Common in connection pooling with multiple concurrent operations
   - Detected by error message patterns ("prepared statement", "does not exist")
   - Resolved by clearing prepared statements and recreating connections

2. SSL Connection Failures:
   - "SSL connection has been closed unexpectedly" errors
   - "server closed the connection unexpectedly" messages
   - Connection reset or broken pipe errors
   - Resolved by recreating connection pool with backoff delay

3. Connection Pool Exhaustion:
   - Pool size limits reached during high concurrency
   - Stale connections blocking new checkpointer operations
   - Resolved by pool recreation and health checking

Error Recovery Flow:
------------------
1. Error Detection:
   - Function execution encounters database error
   - Error type and message analyzed by detection functions
   - Specific error category identified (prepared statement, SSL, etc.)

2. Cleanup Operations:
   - Prepared statements cleared from PostgreSQL catalog
   - Connection pool closed and resources released
   - Global checkpointer state reset to None
   - Short delay for cleanup completion

3. Retry Attempt:
   - Exponential backoff delay applied
   - New connection pool created on next checkpointer access
   - Function re-executed with fresh connection state
   - Success logged or next retry attempted

4. Final Resolution:
   - Either function succeeds after recovery
   - Or maximum retries exceeded and error propagated
   - Detailed logs available for troubleshooting

Usage in Checkpointer System:
---------------------------
The error handling decorators are applied to critical checkpointer operations:

```python
from checkpointer.error_handling.retry_decorators import (
    retry_on_prepared_statement_error,
    retry_on_ssl_connection_error
)

@retry_on_prepared_statement_error(max_retries=3)
@retry_on_ssl_connection_error(max_retries=3)
async def get_checkpointer(user_id: str):
    # Function automatically retries on prepared statement or SSL errors
    # Cleanup operations performed between retry attempts
    # Fresh connections created as needed
    pass
```

Configuration:
-------------
Retry behavior configured through:
- DEFAULT_MAX_RETRIES: Maximum retry attempts (from checkpointer.config)
- Exponential backoff: min(2^attempt, 30) seconds delay
- Debug logging: Controlled by print__checkpointers_debug utility

Dependencies:
------------
- psycopg: Async PostgreSQL adapter (psycopg3)
- api.utils.debug: Debug logging utilities
- checkpointer.database.connection: Connection management
- checkpointer.config: Configuration parameters
- checkpointer.globals: Global state management

Error Detection Patterns:
-----------------------
Prepared Statement Errors:
- "prepared statement" in error message
- "does not exist" for missing statement references
- "_pg3_" or "_pg_" statement name prefixes
- "invalidsqlstatementname" PostgreSQL error codes

SSL Connection Errors:
- "ssl connection has been closed unexpectedly"
- "consuming input failed"
- "server closed the connection unexpectedly"
- "connection closed", "ssl syscall error"
- "operationalerror", "connection reset", "broken pipe"

Best Practices:
--------------
1. Apply retry decorators to all critical checkpointer operations
2. Use appropriate max_retries based on operation criticality
3. Monitor debug logs for recurring error patterns
4. Keep prepared statement settings disabled in connection kwargs
5. Ensure connection pools have adequate health checking
6. Review error logs to identify systemic issues

Performance Considerations:
-------------------------
- Retry delays use exponential backoff to avoid overwhelming database
- Cleanup operations are non-blocking and continue on individual failures
- Connection pool recreation is only triggered when necessary
- Debug logging can be disabled in production for performance
- Maximum retry delays capped at 30 seconds to prevent excessive waiting

Security Notes:
--------------
- Cleanup connections use unique application names for traceability
- Connection strings include SSL/TLS requirement (sslmode=require)
- Prepared statement queries limited to psycopg naming patterns
- No sensitive data logged in error messages
- Connection credentials managed through secure config module

Troubleshooting:
---------------
1. Enable debug logging: Check print__checkpointers_debug output
2. Review error patterns: Identify specific error types and frequencies
3. Check connection pool: Monitor pool size and connection health
4. Verify PostgreSQL: Check pg_prepared_statements catalog
5. Network diagnostics: Test SSL connection stability
6. Resource limits: Ensure adequate database connection limits

Known Limitations:
----------------
- Retry logic cannot recover from permanent configuration errors
- Some edge cases may require manual intervention
- Excessive retries may indicate underlying system issues
- Debug logging overhead in high-frequency operations
- Prepared statement cleanup requires appropriate database permissions

Future Enhancements:
------------------
- Circuit breaker pattern for repeated failures
- Metrics collection for error rates and recovery times
- Adaptive retry delays based on error patterns
- Connection pool size auto-tuning
- Enhanced error classification and routing"""
